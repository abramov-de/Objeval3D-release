"""ZebraPose masks for the view sets of a manifest, given the ground-truth box of the object.

Not a comparison at equal annotation budget. ZebraPose (Su et al., CVPR 2022) is a 6D pose network, but its own segmentation branch is a
real, separately supervised output (`common_ops.from_output_to_class_mask`: sigmoid + threshold on one output channel of
`BinaryCodeNet_Deeplab`, trained with its own L1 mask loss). The released checkpoints are trained per object on BOP's full training
regime (hundreds of thousands of PBR/real images), which was not collected here. So this answers "how far is the head from a
fully-supervised network of the field, one per object, that is given the box", not "does reusing DA3's features beat the obvious
alternative at the same labelling cost".

**This is a driver, not a reimplementation.** The things that decide anything are the authors' own code, called by name:

    BinaryCodeNet_Deeplab(...)             the trained network (ResNet-34 backbone, DeepLabV3-style ASPP decoder, concat skips)
    common_ops.from_output_to_class_mask   sigmoid + threshold on the mask channel - their own function, unedited
    common_ops.compute_original_mask       pastes the 128x128 crop-space mask back into full-image coordinates

The crop, the resize and the normalisation are their `bop_dataset_pytorch.py` / `common_ops.py` logic for the non-training, no-detector
path, copied by reading it rather than imported, because importing that module drags in `GDR_Net_Augmentation` (training-time augmentation)
and a hard dependency on the correspondence-colour GT images that only exist if their whole GT-generation pipeline has been run, and
mask-only inference touches neither.

Provenance: which CNOS Table 1 row do these checkpoints match
-------------------------------------------------------------

CNOS's Table 1 lists "ZebraPose (Synth)" and "ZebraPose (Real)" at different mean AP. The released `dfki-av/ZebraPose` checkpoints carry no
tag saying which; the answer is in the training configs shipped in the same checkout (`config/config_BOP/{lmo,ycbv}/exp_*_BOP.txt`):

    lmo:   training_data_folder = train_pbr,  training_data_folder_2 = none
    ycbv:  training_data_folder = train_real, training_data_folder_2 = train_pbr      (second_dataset_ratio = 0.875)

LM-O's checkpoint is trained on synthetic (PBR) images only (CNOS's "Synth" regime). YCB-V's mixes real and PBR in every batch, closer to
"Real", though PBR is still in the mix. Recorded into every run's JSON as `provenance`.

Two workarounds, both harmless
------------------------------

1. `torch.load(checkpoint_file)` fails under PyTorch >= 2.6 with a `weights_only` UnpicklingError, because the checkpoint dict also holds
   `optimizer_state_dict`. Loading with `weights_only=False` is required and is safe here: the file comes from the authors' own
   `dfki-av/ZebraPose` repository. Only `model_state_dict` is read.

2. `ResNet_BasicBlock_OS8.__init__` (in `model/resnet.py`) unconditionally loads an ImageNet-pretrained ResNet-34 from
   `<zebrapose-root>/zebrapose/pretrained_backbone/resnet/resnet34-333f7ec4.pth` at network *construction* time, before the trained
   checkpoint is applied. Its values do not matter: `net.load_state_dict(checkpoint[...], strict=True)` overwrites every parameter
   immediately afterwards. `ensure_backbone_placeholder()` writes a same-shape `torchvision.models.resnet34(weights=None).state_dict()`
   there if the file is missing, so no download or manual setup step is needed.

Crop and preprocessing, replicated by reading `bop_dataset_pytorch.py`: the test-time, no-detector path (`is_train=False`,
`Detect_Bbox=None`).

    Bbox = gt_info['bbox_visib']                      # the ground-truth box of the visible part
    Bbox = padding_Bbox(Bbox, padding_ratio)           # pad by 1.5x, config default
    roi  = crop_square_resize(rgb, Bbox, 256)          # square crop, INTER_LINEAR
    final_Bbox = get_final_Bbox(Bbox)                  # the *square* region, for pasting back
    tensor = Normalize(ToTensor(PIL.Image.fromarray(roi).convert('RGB')))

One easy-to-miss detail, load-bearing for correctness: `cv2.imread` returns BGR, and their `transform_pre` does
`Image.fromarray(sample_x).convert('RGB')` on that array *without* a `cv2.cvtColor(BGR2RGB)` first - `Image.fromarray` on a (H,W,3)
uint8 array just labels the three channels R,G,B in array order - so the network was trained (and is tested, in their own `test.py`) on
**BGR pixel values wearing an RGB label**, consistently in both directions. This driver reproduces that exactly; do not "fix" it with a
channel swap, that would feed the network something it never saw.

The box is the ground-truth `bbox_visib`: using a detector's box here would conflate detection quality with ZebraPose's own mask quality,
which is not the question this baseline answers. (The cost of a detector for the box is measured separately, `objeval3d.bench.detector`.)

Environment (no separate one is needed beyond the clone; plain `nn.Module`, no custom ops, runs in the reconstruction environment):

    git clone https://github.com/suyz526/ZebraPose.git && git -C ZebraPose checkout 1cb0a1c717c61d3d23171c72fa3bfd22c17d9a76
    hf download dfki-av/ZebraPose zebrapose_ckpts/bop/lmo zebrapose_ckpts/bop/ycbv --local-dir zebrapose_ckpts_downloaded

    python -m objeval3d.seg.sources.zebrapose --manifest configs/manifests/lmo_v3.json --zebrapose-root ~/ZebraPose \\
        --ckpt-root zebrapose_ckpts_downloaded --out-dir workspace/masks/lmo_zebrapose

`--ckpt-root` is the directory the `hf download` above was pointed at; the driver looks for `<ckpt-root>/zebrapose_ckpts/bop/<dataset>/
<obj_name>` per object, with `<obj_name>` as in `objeval3d.data.bop.OBJECT_NAMES`, which is identical to ZebraPose's own per-object
checkpoint filenames for every LM-O and YCB-V object, both clamps included. Do NOT point at `bop_effnet/*.zip` (a different architecture)
or `paper_ablation/...` (ablations).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from objeval3d.data.bop import BopScene, git_revision, object_name
from objeval3d.data.manifest import load_manifest

# Sourced from config/config_BOP/{lmo,ycbv}/exp_*_BOP.txt in the ZebraPose checkout (see "Provenance" above).
PROVENANCE = {
    "lmo": {
        "training_data_folder": "train_pbr",
        "training_data_folder_2": "none",
        "matches_cnos_table1_row": "ZebraPose (Synth)",
    },
    "ycbv": {
        "training_data_folder": "train_real",
        "training_data_folder_2": "train_pbr",
        "second_dataset_ratio": 0.875,
        "matches_cnos_table1_row": "ZebraPose (Real) -- real+pbr mixed, not pure real",
    },
}


def import_zebrapose(zebrapose_root: Path):
    """Locate the inner `zebrapose/` package dir and import just enough of it.

    The checkout nests one level: `<zebrapose_root>/zebrapose/common_ops.py`, `.../zebrapose/model/BinaryCodeNet.py`. Pointing at the
    outer checkout fails exactly like a missing clone, so say which directory was checked.
    """
    inner = zebrapose_root / "zebrapose"
    if not (inner / "common_ops.py").exists():
        if (zebrapose_root / "common_ops.py").exists():
            inner = zebrapose_root
        else:
            raise SystemExit(
                f"--zebrapose-root {zebrapose_root} has no common_ops.py, checked there and at {inner}. Clone "
                f"https://github.com/suyz526/ZebraPose.git and pass its root (the directory that itself contains a `zebrapose/` "
                f"subdirectory holding common_ops.py and model/)."
            )
    sys.path.insert(0, str(inner))
    from common_ops import compute_original_mask, from_output_to_class_mask
    from model.BinaryCodeNet import BinaryCodeNet_Deeplab

    return inner, from_output_to_class_mask, compute_original_mask, BinaryCodeNet_Deeplab


def ensure_backbone_placeholder(inner_root: Path) -> None:
    """Write a same-shape ResNet-34 state dict so construction does not crash.

    `model/resnet.py` loads this file at construction time, before the trained ZebraPose checkpoint overwrites every parameter via
    `load_state_dict`. Its values are therefore never used - only its key/shape agreement with `torchvision.models.resnet34()`'s
    own state dict matters, which generating it from that exact class guarantees.
    """
    path = inner_root / "pretrained_backbone" / "resnet" / "resnet34-333f7ec4.pth"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import torchvision.models as models

    torch.save(models.resnet34(weights=None).state_dict(), path)
    print(
        f"  [setup] wrote placeholder backbone init -> {path} (overwritten in full by the real checkpoint right after)"
    )


# ---------------------------------------------------------------------------
# Crop/resize, copied from bop_dataset_pytorch.py (crop_square_resize / padding_Bbox / get_final_Bbox) rather than imported -
# importing that module pulls in GDR_Net_Augmentation and the correspondence-GT loading this driver deliberately does not need.
# The three functions below are from ZebraPose, https://github.com/suyz526/ZebraPose (zebrapose/bop_dataset_pytorch.py), MIT License:
#
#   Copyright (c) 2022 suyz526
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
#   (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
#   publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
#   subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------
def padding_Bbox(Bbox: np.ndarray, padding_ratio: float) -> np.ndarray:
    x1, y1 = Bbox[0], Bbox[1]
    x2, y2 = Bbox[0] + Bbox[2], Bbox[1] + Bbox[3]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw, bh = x2 - x1, y2 - y1
    pw, ph = int(bw * padding_ratio), int(bh * padding_ratio)
    return np.array([int(cx - pw / 2), int(cy - ph / 2), pw, ph])


def crop_square_resize(img: np.ndarray, Bbox: np.ndarray, crop_size: int, interpolation) -> np.ndarray:
    x1, y1 = Bbox[0], Bbox[1]
    x2, y2 = Bbox[0] + Bbox[2], Bbox[1] + Bbox[3]
    bw, bh = Bbox[2], Bbox[3]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    if bh > bw:
        x1, x2 = cx - bh / 2, cx + bh / 2
    else:
        y1, y2 = cy - bw / 2, cy + bw / 2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    side = max(bh, bw)
    roi = (
        np.zeros((side, side, img.shape[2]), dtype=img.dtype)
        if img.ndim > 2
        else np.zeros((side, side), dtype=img.dtype)
    )
    roi_x1 = max(-x1, 0)
    x1c = max(x1, 0)
    roi_x2 = roi_x1 + min(img.shape[1] - x1c, x2 - x1c)
    roi_y1 = max(-y1, 0)
    y1c = max(y1, 0)
    roi_y2 = roi_y1 + min(img.shape[0] - y1c, y2 - y1c)
    x2c, y2c = min(x2, img.shape[1]), min(y2, img.shape[0])
    roi[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1c:y2c, x1c:x2c]
    return cv2.resize(roi, (crop_size, crop_size), interpolation=interpolation)


def get_final_Bbox(Bbox: np.ndarray) -> np.ndarray:
    """crop_square_resize's square region, in ORIGINAL image coordinates - may extend past the image edges; `compute_original_mask`
    clips that itself."""
    x1, y1 = Bbox[0], Bbox[1]
    x2, y2 = Bbox[0] + Bbox[2], Bbox[1] + Bbox[3]
    bw, bh = Bbox[2], Bbox[3]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    if bh > bw:
        x1, x2 = cx - bh / 2, cx + bh / 2
    else:
        y1, y2 = cy - bw / 2, cy + bw / 2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return np.array([x1, y1, x2 - x1, y2 - y1])


def preprocess(rgb_bgr: np.ndarray, bbox_visib, padding_ratio: int, crop_size_img: int):
    """-> (network input tensor [3,H,W], final_Bbox for pasting the mask back)."""
    Bbox = padding_Bbox(np.asarray(bbox_visib, dtype=np.int64), padding_ratio)
    roi = crop_square_resize(rgb_bgr, Bbox, crop_size_img, cv2.INTER_LINEAR)
    final_bbox = get_final_Bbox(Bbox)
    # No BGR->RGB swap: see the module docstring, "one easy-to-miss detail".
    pil = Image.fromarray(np.uint8(roi)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
    return tensor, final_bbox


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ZebraPose masks for a view-set manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zebrapose-root", required=True, help="the ZebraPose checkout")
    ap.add_argument(
        "--ckpt-root",
        required=True,
        help="holds zebrapose_ckpts/bop/{lmo,ycbv}/<obj_name>, i.e. whatever --local-dir `hf download dfki-av/ZebraPose "
        "zebrapose_ckpts/bop/lmo zebrapose_ckpts/bop/ycbv` used",
    )
    ap.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    ap.add_argument("--padding-ratio", type=float, default=1.5, help="their config default")
    ap.add_argument("--crop-size-img", type=int, default=256)
    ap.add_argument("--crop-size-gt", type=int, default=128, help="sanity-checked against the network's actual output")
    ap.add_argument("--threshold", type=float, default=0.5, help="their default")
    ap.add_argument("--obj-ids", default="", help="comma separated subset, for a smoke run")
    ap.add_argument("--limit-sets", type=int, default=0)
    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed passes through each freshly loaded network before the timed loop. The first passes pay for cuDNN "
        "autotuning and allocator growth",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    zebrapose_root = Path(args.zebrapose_root).expanduser().resolve()
    ckpt_root = Path(args.ckpt_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inner, from_output_to_class_mask, compute_original_mask, BinaryCodeNet_Deeplab = import_zebrapose(zebrapose_root)
    ensure_backbone_placeholder(inner)

    manifest = load_manifest(args.manifest, args.bop_root)
    dataset = manifest["dataset"]
    bop_root = Path(manifest["bop_root"])
    sets = manifest["sets"]
    if args.obj_ids:
        wanted = {int(x) for x in args.obj_ids.split(",") if x.strip()}
        sets = [s for s in sets if s["obj_id"] in wanted]
    if args.limit_sets > 0:
        step = max(1, len(sets) // args.limit_sets)
        sets = sets[::step][: args.limit_sets]
    if not sets:
        raise SystemExit("no view sets selected")

    if dataset not in PROVENANCE:
        raise SystemExit(
            f"no training-data provenance recorded for dataset={dataset!r} "
            f"- read its config_BOP/<dataset>/exp_*_BOP.txt before adding one"
        )

    by_obj: dict[int, list[dict]] = defaultdict(list)
    for vs in sets:
        by_obj[vs["obj_id"]].append(vs)

    device = torch.device(args.device)
    print(f"{len(sets)} view sets over {len(by_obj)} object(s), dataset={dataset} -> {out_dir}")
    print(f"provenance: {PROVENANCE[dataset]}")

    scene_cache: dict[int, BopScene] = {}

    def scene_for(scene_id: int) -> BopScene:
        if scene_id not in scene_cache:
            scene_cache[scene_id] = BopScene(bop_root / manifest["split"] / f"{scene_id:06d}")
        return scene_cache[scene_id]

    record = {
        "manifest": args.manifest,
        "dataset": dataset,
        "git_revision": git_revision(),
        "zebrapose_root": str(zebrapose_root),
        "padding_ratio": args.padding_ratio,
        "resize_method": "crop_square_resize",
        "crop_size_img": args.crop_size_img,
        "threshold": args.threshold,
        "provenance": PROVENANCE[dataset],
        "objects": {},
    }

    t_all = []
    total_written, total_empty = 0, 0
    for oi, (obj_id, obj_sets) in enumerate(sorted(by_obj.items()), 1):
        obj_name = obj_sets[0]["obj_name"]
        expect = object_name(dataset, obj_id)
        if obj_name != expect:
            raise SystemExit(
                f"manifest names object {obj_id} {obj_name!r}, objeval3d.data.bop.object_name says {expect!r} - "
                f"checkpoint filenames follow that convention, this mismatch would silently load the wrong checkpoint"
            )

        ckpt_path = ckpt_root / "zebrapose_ckpts" / "bop" / dataset / obj_name
        if not ckpt_path.exists():
            raise SystemExit(
                f"no checkpoint at {ckpt_path}\nfetch it with:\n"
                f"  hf download dfki-av/ZebraPose zebrapose_ckpts/bop/{dataset}/{obj_name} --local-dir {ckpt_root}"
            )

        net = BinaryCodeNet_Deeplab(
            num_resnet_layers=34,
            concat=True,
            binary_code_length=16,
            divided_number_each_iteration=2,
            output_kernel_size=1,
            efficientnet_key=None,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(ckpt["model_state_dict"], strict=True)
        if missing or unexpected:
            raise SystemExit(
                f"checkpoint {ckpt_path} does not match the network architecture - missing={missing} unexpected={unexpected}"
            )
        net = net.to(device).eval()

        paths, where = [], []
        for vs in obj_sets:
            for i, im_id in enumerate(vs["im_ids"]):
                paths.append((vs["images"][i], vs["scene_id"], im_id, vs["gt_ids"][i]))
                where.append((vs["set_id"], i))

        print(
            f"\n[{oi}/{len(by_obj)}] obj {obj_id} ({obj_name}) -- "
            f"ckpt best_score={ckpt.get('best_score')} step={ckpt.get('iteration_step')}, "
            f"{len(obj_sets)} sets, {len(paths)} frames"
        )

        if args.warmup > 0:
            with torch.no_grad():
                for (img_path, scene_id, im_id, gt_id), _ in zip(paths, where):
                    anns = scene_for(scene_id).gt_info.get(im_id, [])
                    bb = anns[gt_id].get("bbox_visib") if gt_id < len(anns) else None
                    if not bb or bb[2] <= 0 or bb[3] <= 0:
                        continue
                    try:
                        warm_t, _ = preprocess(cv2.imread(img_path), bb, args.padding_ratio, args.crop_size_img)
                    except (ValueError, IndexError):
                        continue
                    for _ in range(args.warmup):
                        net(warm_t.unsqueeze(0).to(device))
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    break

        empty, degenerate = 0, 0
        t0 = time.time()
        with torch.no_grad():
            for (img_path, scene_id, im_id, gt_id), (set_id, view) in zip(paths, where):
                scene = scene_for(scene_id)
                anns = scene.gt_info.get(im_id, [])
                bbox_visib = anns[gt_id].get("bbox_visib") if gt_id < len(anns) else None
                rgb = cv2.imread(img_path)  # BGR - kept as-is, see module docstring
                H, W = rgb.shape[:2]

                if not bbox_visib or bbox_visib[2] <= 0 or bbox_visib[3] <= 0:
                    m = np.zeros((H, W), bool)
                    empty += 1
                else:
                    try:
                        t_pre0 = time.time()
                        tensor, final_bbox = preprocess(rgb, bbox_visib, args.padding_ratio, args.crop_size_img)
                        pred_mask_prob, _ = net(tensor.unsqueeze(0).to(device))
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        mask_crop = from_output_to_class_mask(pred_mask_prob, args.threshold)
                        mask_crop = mask_crop[0, 0].astype(np.uint8)  # (crop_gt, crop_gt)
                        if mask_crop.shape[0] != args.crop_size_gt:
                            raise SystemExit(
                                f"network output {mask_crop.shape} != expected crop_size_gt={args.crop_size_gt} - architecture "
                                f"assumption broken, do not paste this back silently"
                            )
                        m = compute_original_mask(final_bbox, H, W, mask_crop).astype(bool)
                        t_all.append(time.time() - t_pre0)
                        if not m.any():
                            empty += 1
                    except (ValueError, IndexError) as e:
                        # padding_Bbox can push the padded box almost entirely off the frame for an object right at the image edge;
                        # the authors' own crop_square_resize has the same unchecked negative-index arithmetic in that regime (a
                        # real edge case in their code too, not a transcription bug). Bail out to an empty mask for this one frame
                        # rather than losing every other frame of this object.
                        print(
                            f"  [warn] {set_id} view {view}: degenerate bbox {bbox_visib} on a {W}x{H} frame "
                            f"({type(e).__name__}: {e}) -- writing an empty mask"
                        )
                        m = np.zeros((H, W), bool)
                        degenerate += 1
                        empty += 1

                if not cv2.imwrite(str(out_dir / f"{set_id}_{view:02d}.png"), m.astype(np.uint8) * 255):
                    raise SystemExit(f"failed to write mask for {set_id} view {view}")
                total_written += 1
        dt = time.time() - t0
        total_empty += empty
        print(
            f"    {len(paths)} masks in {dt:.1f}s ({1000 * dt / len(paths):.0f} ms/frame)"
            + (f" -- {empty} EMPTY (no visible box, or predicted nothing)" if empty else "")
        )
        record["objects"][str(obj_id)] = {
            "obj_name": obj_name,
            "checkpoint": str(ckpt_path),
            "checkpoint_fingerprint": f"{ckpt_path.stat().st_size}:{int(ckpt_path.stat().st_mtime)}",
            "best_score": ckpt.get("best_score"),
            "iteration_step": ckpt.get("iteration_step"),
            "n_sets": len(obj_sets),
            "n_frames": len(paths),
            "n_empty_masks": empty,
            "n_degenerate_bbox": degenerate,
            "seconds": round(dt, 1),
        }
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def ms(xs):
        return (
            {"median": round(1000 * statistics.median(xs), 1), "mean": round(1000 * sum(xs) / len(xs), 1)}
            if xs
            else None
        )

    record["cost"] = {
        "forward_ms_per_frame": ms(t_all[1:] if len(t_all) > 1 else t_all),
        "n_timed_frames": len(t_all),
        "warmup_passes_per_network": args.warmup,
        "covers": "crop + resize + forward + threshold + paste-back, one object per frame, given a box; excludes "
        "image decoding, network loading and any detector",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }

    with open(out_dir / "zebrapose_run.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n{total_written} masks written, {total_empty} empty")
    print(f"-> {out_dir}\n-> {out_dir / 'zebrapose_run.json'} (provenance, checkpoints, cost)")


if __name__ == "__main__":
    main()
