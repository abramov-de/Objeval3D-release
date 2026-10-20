"""SAM 3 masks for the view sets of a manifest, prompted with the object's name.

The comparison with the head is fair only if SAM 3 gets no more supervision at test time than the head does. The head sees an RGB
frame and nothing else, so SAM 3 is prompted with the object's NAME and nothing else - no ground-truth box, no point, no mask. The
prompts are the final ones of the paper (`objeval3d/data/sam3_prompts.json`, Table F.1); `--prompt OBJ_ID=TEXT` overrides one.

SAM 3 segments a *concept*, so in a scene holding both a large clamp and an extra large clamp, or two cans, the prompt does not say
which one the manifest means; `--select top1` (the default) takes the most confident instance. That is the same "which object is
this" failure a many-class head has, arriving by a different route, and it is a result rather than a bug.

Runs in the dedicated SAM 3 environment (Python 3.12+, CUDA 12.6+, PyTorch 2.7+, `pip install -e .` from a checkout of
github.com/facebookresearch/sam3), NOT in the reconstruction environment. The weights are gated on Hugging Face (facebook/sam3):
request access, `hf auth login`, or pass a local `--checkpoint-path sam3.pt`.

    python -m objeval3d.seg.sources.sam3 --manifest configs/manifests/ycbv_b20_v3.json --out-dir workspace/masks/ycbv_b20_sam3 \\
        --confidence-threshold 0.2

Masks are written at the native resolution of the image, one PNG per view, plus `sam3_run.json` with the prompts, the number of empty
masks (a view in which SAM 3 did not find the concept; downstream it contributes no points at all) and the checkpoint used.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from objeval3d.data.bop import git_revision
from objeval3d.data.manifest import load_manifest
from objeval3d.seg.prompts import prompt_for

GATED_HELP = """SAM 3's weights are gated on Hugging Face and this account is not on the authorised list, so facebook/sam3 returned 403.
Either request access at https://huggingface.co/facebook/sam3, wait for approval and authenticate on this machine (`hf auth login`),
or pass a local checkpoint with --checkpoint-path <sam3.pt>, in which case nothing is downloaded."""


def _is_gated(err: Exception) -> bool:
    """Recognise the gated-repo refusal without importing huggingface_hub here: match on what the message says."""
    text = f"{type(err).__name__}: {err}"
    return "GatedRepoError" in text or "gated repo" in text.lower() or ("403" in text and "huggingface" in text.lower())


def build_processor(checkpoint_path: str | None, device: str, confidence_threshold: float | None):
    import sam3.model_builder as _builder
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # build_sam3_image_model(bpe_path=None) resolves the BPE vocabulary via pkg_resources.resource_filename("sam3", ...), which
    # needs sam3.__file__ to be set; for some editable installs the top-level package has __file__ == None. The path is derived
    # from the model builder's own __file__ instead, which resolves in both cases.
    bpe_path = str(Path(_builder.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(
        bpe_path=bpe_path, device=device, checkpoint_path=checkpoint_path, load_from_HF=checkpoint_path is None
    )
    processor = Sam3Processor(model)
    if confidence_threshold is not None:
        processor.set_confidence_threshold(confidence_threshold)
    return processor


def segment_images(
    processor, image_paths: list[str], text_prompt: str, device: str, select: str = "top1"
) -> list[np.ndarray]:
    """One boolean mask (H, W) per image, from SAM 3's promptable concept segmentation with the same text prompt.

    SAM 3 returns every instance of the concept; this protocol assumes one target object per scene, so `select` collapses them:
    "top1" keeps the highest-scoring instance (safer when the prompt also matches background clutter), "union" ORs all of them
    (for a target that gets split into several detections, but for two objects of the same kind it merges them into one mask).
    """
    import torch
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    if select not in ("top1", "union"):
        raise ValueError(f"select must be 'top1' or 'union', got {select!r}")
    # SAM 3's ViT runs fc1 + activation through a fused kernel that casts to bfloat16 and feeds the result into a plain float32
    # Linear; without autocast that crashes with "mat1 and mat2 must have the same dtype". Its own predictor classes wrap their
    # forward passes in this same bf16 autocast; the basic image path does not, so it is done here.
    autocast_device_type = "cuda" if "cuda" in device else "cpu"
    masks = []
    for path in image_paths:
        # exif_transpose: a portrait photo stored with landscape pixel data and an EXIF orientation tag would otherwise be
        # segmented in the raw orientation, and the mask would look rotated relative to the reconstruction's view of it
        image = exif_transpose(Image.open(path)).convert("RGB")
        with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
            state = processor.set_image(image)
            output = processor.set_text_prompt(prompt=text_prompt, state=state)
        masks_t, scores_t = output["masks"], output["scores"]
        n = int(masks_t.shape[0])
        if n == 0:
            print(
                f"[WARN] SAM 3: prompt {text_prompt!r} matched 0 instances in {path} - using an all-False mask for this view."
            )
            w, h = image.size
            masks.append(np.zeros((h, w), dtype=bool))
            continue
        # tensors computed under the bf16 autocast can come back as bfloat16, which numpy does not have
        masks_np = masks_t.detach().float().cpu().numpy()
        if masks_np.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
            masks_np = masks_np[:, 0]
        if masks_np.dtype != np.bool_:
            masks_np = masks_np > 0.5
        if select == "top1":
            scores_np = scores_t.detach().float().cpu().numpy()
            masks.append(masks_np[int(np.argmax(scores_np))])
        else:
            masks.append(masks_np.any(axis=0))
    return masks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAM 3 masks for a view-set manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--checkpoint-path",
        default=None,
        help="local sam3.pt. Given, nothing is fetched from Hugging Face; omitted, the gated facebook/sam3 repo is "
        "downloaded and the run fails without approved access",
    )
    ap.add_argument("--confidence-threshold", type=float, default=None, help="the paper uses 0.2")
    ap.add_argument("--select", choices=["top1", "union"], default="top1")
    ap.add_argument("--obj-ids", default="", help="comma separated subset, for a smoke run")
    ap.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="OBJ_ID=TEXT",
        help="override the prompt for one object, repeatable",
    )
    ap.add_argument(
        "--prompts-file", default=None, help="a JSON file of prompts in the format of objeval3d/data/sam3_prompts.json"
    )
    ap.add_argument("--limit-sets", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    overrides = {}
    for spec in args.prompt:
        if "=" not in spec:
            raise SystemExit(f"--prompt wants OBJ_ID=TEXT, got {spec!r}")
        oid, text = spec.split("=", 1)
        if not text.strip():
            raise SystemExit(f"--prompt {spec!r} has an empty prompt")
        overrides[int(oid)] = text

    manifest = load_manifest(args.manifest, args.bop_root)
    sets = manifest["sets"]
    if args.obj_ids:
        wanted = {int(x) for x in args.obj_ids.split(",") if x.strip()}
        sets = [s for s in sets if s["obj_id"] in wanted]
    if args.limit_sets > 0:
        step = max(1, len(sets) // args.limit_sets)
        sets = sets[::step][: args.limit_sets]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One call per object rather than per view set: the prompt is the same for every set of an object.
    by_obj: dict[int, list[dict]] = defaultdict(list)
    for vs in sets:
        by_obj[vs["obj_id"]].append(vs)

    ckpt = Path(args.checkpoint_path).expanduser() if args.checkpoint_path else None
    if ckpt and not ckpt.exists():
        raise SystemExit(f"--checkpoint-path {ckpt} does not exist")
    print(f"{len(sets)} view sets over {len(by_obj)} object(s) -> {out_dir}")
    record = {
        "manifest": args.manifest,
        "select": args.select,
        "git_revision": git_revision(),
        "confidence_threshold": args.confidence_threshold,
        "checkpoint_path": str(ckpt) if ckpt else "facebook/sam3 (hugging face)",
        "checkpoint_fingerprint": (f"{ckpt.stat().st_size}:{int(ckpt.stat().st_mtime)}" if ckpt else None),
        "objects": {},
    }

    try:
        processor = build_processor(args.checkpoint_path, args.device, args.confidence_threshold)
    except Exception as e:
        if _is_gated(e):
            raise SystemExit(GATED_HELP) from e
        raise

    for oi, (obj_id, obj_sets) in enumerate(sorted(by_obj.items()), 1):
        obj_name = obj_sets[0]["obj_name"]
        prompt = overrides.get(obj_id, prompt_for(obj_name, args.prompts_file))

        # flat image list plus the (set, view) each entry came from, so the masks can be written back under the right name
        paths, where = [], []
        for vs in obj_sets:
            for i, img in enumerate(vs["images"]):
                paths.append(img)
                where.append((vs["set_id"], i))

        print(
            f"\n[{oi}/{len(by_obj)}] obj {obj_id} ({obj_name}) - prompt {prompt!r}, {len(obj_sets)} sets, {len(paths)} images"
        )
        t0 = time.time()
        masks = segment_images(processor, paths, prompt, args.device, args.select)
        dt = time.time() - t0

        empty = 0
        for (set_id, view), mask in zip(where, masks):
            m = np.asarray(mask).astype(bool)
            empty += int(not m.any())
            if not cv2.imwrite(str(out_dir / f"{set_id}_{view:02d}.png"), m.astype(np.uint8) * 255):
                raise SystemExit(f"failed to write mask for {set_id} view {view}")
        print(
            f"    {len(paths)} masks in {dt:.1f}s ({dt / len(paths):.2f}s/image)"
            + (f" - {empty} EMPTY (concept not found)" if empty else "")
        )
        record["objects"][str(obj_id)] = {
            "obj_name": obj_name,
            "prompt": prompt,
            "n_sets": len(obj_sets),
            "n_images": len(paths),
            "n_empty_masks": empty,
            "prompt_overridden": obj_id in overrides,
            "seconds": round(dt, 1),
        }

    # Re-running a subset must not erase the record of the objects it did not touch: their masks are still in this directory.
    # Merge when the settings match, replace and say so when they do not.
    rec_path = out_dir / "sam3_run.json"
    if rec_path.exists():
        try:
            prev = json.loads(rec_path.read_text())
        except (json.JSONDecodeError, OSError):
            prev = None
        if prev:
            same = all(
                prev.get(k) == record.get(k)
                for k in ("manifest", "select", "confidence_threshold", "checkpoint_fingerprint")
            )
            if same:
                merged = dict(prev.get("objects") or {})
                merged.update(record["objects"])
                record["objects"] = merged
            else:
                print(
                    "[replace] the existing record used different settings (manifest/select/threshold/checkpoint) - "
                    "it described other masks and is not merged"
                )
    rec_path.write_text(json.dumps(record, indent=2))
    print(f"\n-> {out_dir}\n-> {rec_path} (prompts and timings)")


if __name__ == "__main__":
    main()
