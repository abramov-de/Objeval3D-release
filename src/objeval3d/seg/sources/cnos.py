"""CNOS masks for the view sets of a manifest: the CAD-based segmentation baseline.

CNOS (Nguyen et al., ICCV 2023 R6D workshop) takes the scan of an object, renders templates of it, runs SAM in "everything" mode on the
frame and matches every proposal to the templates by DINOv2 descriptors. It needs no supervision on the frames at all, and it is the
BOP-standard CAD-based segmenter. The head and CNOS are compared on the input the protocol already requires anyway: the scan.

**This is a driver, not a reimplementation.** Every step that decides anything is the authors' own code, called by its own name:

    model.segmentor_model.generate_masks   SAM "everything" proposals
    Detections / remove_very_small_detections
    model.descriptor_model(...)            DINOv2 CLS descriptors per proposal
    model.find_matched_proposals(...)      cosine similarity, avg_5 aggregation, confidence threshold, object assignment
    detections.apply_nms_per_object_id

A comparison computed by our own code would answer "you measured your own program". The only things that are ours are the loop over
the manifest and the choice of which mask to write out: from CNOS's detection list we take, for object o in a frame, the
highest-scoring detection CNOS itself assigned to o (`--select top1`), the only way to turn a detector's output into the single object
mask the reconstruction stage consumes.

The instance problem is not solved by it. CNOS crops every proposal and every template to 224x224, so absolute size does not survive
into the descriptor, and `large_clamp` / `extra_large_clamp`, which differ mostly in size, can be confused through geometry-blind
cropping where a text prompt confuses them through language.

Environment (its own)
---------------------

    git clone https://github.com/nv-nguyen/cnos.git && git -C cnos checkout 298d1f3366171464ca271659f0e2f7a6eb8e39b4
    conda create -n cnos python=3.10 -y && conda activate cnos
    pip install torch torchvision "numpy<2" opencv-python pillow scipy omegaconf hydra-core pytorch-lightning pycocotools \\
        ruamel.yaml pandas tqdm trimesh pyrender
    pip install git+https://github.com/thodan/bop_toolkit.git     # BOPTemplate imports it

Leaner than the authors' environment.yml, deliberately: it pins pytorch-lightning 1.8.6 for its `Trainer` path, while the import chain
used here takes only `seed_everything` and `LightningModule.__init__` from it. torchmetrics, wandb, distinctipy and scikit-image belong to
their training and visualisation code and are never imported. `segment_anything` is not a pip install: the repository vendors it, and
`PYTHONPATH=<cnos-root>` covers it. pandas is needed because `src/dataloader/base_bop.py` imports it at module scope and the template
loader comes through that file.

Two checkpoints are needed, both public on dl.fbaipublicfiles.com: SAM ViT-H (2.4 GB) and DINOv2 ViT-L/14 (1.1 GB). Their destinations:

    <weights-root>/pretrained/segment-anything/sam_vit_h_4b8939.pth
    $TORCH_HOME/hub/checkpoints/dinov2_vitl14_pretrain.pth           (~/.cache/torch by default)

Nothing verifies them on arrival, so check the sizes after a copy: a truncated checkpoint fails inside `torch.load` with a message
about the archive, several minutes into the run. DINOv2's CODE is separate from its weights: `torch.hub.load` fetches the repository
from GitHub to read `hubconf.py`. Pass `--dinov2-dir <checkout of facebookresearch/dinov2>` to load it with `source="local"`, which
contacts nothing at all.

Templates (once per dataset, a couple of minutes): `python -m objeval3d.seg.sources.cnos_templates`.

    PYTHONPATH=~/cnos python -m objeval3d.seg.sources.cnos --manifest configs/manifests/ycbv_b20_v3.json \\
        --cnos-root ~/cnos --template-dir workspace/masks/cnos_templates/ycbv --out-dir workspace/masks/ycbv_b20_cnos

The expensive half of CNOS (SAM proposals + a DINOv2 descriptor per proposal) is object-independent, so it runs ONCE per unique frame
and is then scored against every object that needs it (133 frames instead of 240 image slots on LM-O). The per-object scoring, which
is the part that could differ, still runs per object; this only refuses to pay for the same proposals twice, and it makes the cost
honest: what a frame costs is what one proposal pass costs.
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

from objeval3d.data.bop import git_revision
from objeval3d.data.manifest import load_manifest


def cnos_revision(root: Path) -> str | None:
    """Which CNOS commit produced these masks: the claim is about their method as released, so a reader who wants to repeat
    this needs the checkout, not the name."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def warn_missing_weights(cfg) -> None:
    """Say which file is missing, before a download that may not be possible.

    Both checkpoints live on dl.fbaipublicfiles.com, which some machines cannot reach or reach very slowly. When that happens the
    run does not fail - it hangs - so name the file and where it goes while the message is still cheap to act on.
    """
    name = str(cfg.model.descriptor_model.model_name)
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / f"{name}_pretrain.pth"
    if ckpt.exists():
        return
    print(
        f"[warn] {ckpt} is missing, so DINOv2 will try to download it from\n"
        f"       https://dl.fbaipublicfiles.com/dinov2/{name}/{name}_pretrain.pth\n"
        f"       If that host is unreachable here, fetch it on a machine that can and copy it to that path."
    )


def load_cnos(cnos_root: Path, args) -> tuple:
    """Instantiate the authors' CNOS exactly as their own `run_inference.py` does.

    Their entry point goes through hydra, so the same config is composed rather than hand-building the model: the thresholds, the
    aggregation function and the post-processing sizes then come from THEIR yaml, not from defaults retyped here, and
    `--segmentor fastsam` is their `model=cnos_fast` switch.
    """
    sys.path.insert(0, str(cnos_root))
    try:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
    except ImportError as e:
        raise SystemExit(
            f"hydra is missing from this env: {e}\nThis module runs in the `cnos` environment - see its docstring."
        )
    try:
        from src.model.utils import Detections  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"Could not import CNOS from {cnos_root}: {e}\n"
            f"--cnos-root must be the repository root (the directory holding `src/` and `run_inference.py`), and PYTHONPATH must "
            f"name it too because their modules import each other as `src.*`."
        )

    weights_root = (
        Path(args.weights_root).expanduser().resolve() if args.weights_root else cnos_root / "datasets/bop23_challenge"
    )
    overrides = [
        f"user.local_root_dir={weights_root}",
        f"save_dir={Path(args.out_dir).resolve() / 'cnos_logs'}",
        f"model.matching_config.confidence_thresh={args.confidence_thresh}",
    ]
    if args.segmentor == "fastsam":
        overrides.append("model=cnos_fast")
    else:
        overrides.append(f"model.segmentor_model.stability_score_thresh={args.stability_score_thresh}")
    if args.dinov2_dir:
        d = Path(args.dinov2_dir).expanduser().resolve()
        if not (d / "hubconf.py").exists():
            raise SystemExit(
                f"--dinov2-dir {d} has no hubconf.py - it must be a checkout of facebookresearch/dinov2, not its weights"
            )
        # source='local' skips torch.hub's repo resolution entirely: no call to github, no download, no trust prompt. The WEIGHTS
        # are a separate matter and still come through load_state_dict_from_url, i.e. out of <TORCH_HOME>/hub/checkpoints.
        overrides += [f"model.descriptor_model.model.repo_or_dir={d}", "+model.descriptor_model.model.source=local"]

    with initialize_config_dir(version_base=None, config_dir=str(cnos_root / "configs")):
        cfg = compose(config_name="run_inference", overrides=overrides)
    warn_missing_weights(cfg)
    model = instantiate(cfg.model)

    device = torch.device(args.device)
    # Their own move_to_device() reads `self.device`, which only a Lightning trainer sets; their `inference_custom.py` moves the two
    # submodels by hand for the same reason, and this is that code.
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    return model, cfg, device


def reference_descriptors(model, cnos_root: Path, template_dir: Path, level_templates: int, obj_ids, device) -> tuple:
    """Template descriptors, through the authors' BOPTemplate loader.

    Cached next to the templates: computing them is a minute of GPU, but the cache is also what makes a rerun on the same templates
    bit-exact, which matters when the run is split across sessions.
    """
    try:
        from omegaconf import OmegaConf
        from src.dataloader.bop import BOPTemplate
    except ImportError as e:
        # name the module that is actually missing
        missing = getattr(e, "name", None) or "?"
        hint = {
            "bop_toolkit_lib": "pip install git+https://github.com/thodan/bop_toolkit.git",
            "pandas": "pip install pandas",
            "tqdm": "pip install tqdm",
        }.get(missing, f"pip install {missing}")
        raise SystemExit(
            f"Could not import BOPTemplate: {e}\n"
            f"Its import chain (src.dataloader.bop -> base_bop -> src.utils, src.poses) needs: PIL, bop_toolkit_lib, numpy, "
            f"omegaconf, pandas, pytorch_lightning, ruamel.yaml, scipy, torch, torchvision, tqdm.\n"
            f"Missing here: {missing}\n    {hint}"
        )

    if not template_dir.is_dir():
        raise SystemExit(
            f"no template directory at {template_dir} - render them first with objeval3d.seg.sources.cnos_templates"
        )
    found = sorted(int(p.name[4:10]) for p in template_dir.glob("obj_*") if p.is_dir())
    if not found:
        raise SystemExit(f"{template_dir} holds no obj_XXXXXX/ directories")
    if obj_ids:
        missing = sorted(set(obj_ids) - set(found))
        if missing:
            raise SystemExit(f"no templates for object(s) {missing} in {template_dir}")
        found = sorted(obj_ids)

    cache = template_dir / f"descriptors_level{level_templates}.pth"
    ds = BOPTemplate(
        template_dir=str(template_dir) + "/",
        obj_ids=found,
        processing_config=OmegaConf.create({"image_size": model.descriptor_model.proposal_size}),
        level_templates=level_templates,
        pose_distribution="all",
    )
    if cache.exists():
        feats = torch.load(cache, map_location=device)
        if feats.shape[0] != len(found):
            raise SystemExit(
                f"{cache} holds {feats.shape[0]} objects but {len(found)} were asked for - it was computed for a different object "
                f"set. Delete it to recompute rather than matching against the wrong templates."
            )
        return feats.to(device), found, str(cache)

    feats = []
    for i in range(len(ds)):
        imgs = ds[i]["templates"].to(device)
        feats.append(model.descriptor_model.compute_features(imgs, token_name="x_norm_clstoken"))
    feats = torch.stack(feats)  # N_objects x N_templates x D
    torch.save(feats, cache)
    return feats, found, str(cache)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CNOS masks for a view-set manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cnos-root", required=True, help="the cnos checkout")
    ap.add_argument("--template-dir", required=True, help="rendered templates, one obj_XXXXXX/ per object")
    ap.add_argument(
        "--weights-root",
        default="",
        help="holds pretrained/segment-anything/sam_vit_h_4b8939.pth; defaults to <cnos-root>/datasets/bop23_challenge",
    )
    ap.add_argument(
        "--dinov2-dir",
        default="",
        help="a local checkout of facebookresearch/dinov2. Given, torch.hub loads it with source='local' and never "
        "contacts github; omitted, it fetches the repo as usual",
    )
    ap.add_argument(
        "--torch-home",
        default="",
        help="sets TORCH_HOME, i.e. where DINOv2's weights are looked up (<TORCH_HOME>/hub/checkpoints). Only needed "
        "if they live somewhere other than ~/.cache/torch",
    )
    ap.add_argument(
        "--segmentor",
        choices=["sam", "fastsam"],
        default="sam",
        help="sam is the published CNOS; fastsam is their cheap variant",
    )
    ap.add_argument("--level-templates", type=int, default=0, help="0 = the 42 templates CNOS uses by default")
    ap.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.15,
        help="their default. Below it a proposal is assigned to no object",
    )
    ap.add_argument("--stability-score-thresh", type=float, default=0.97, help="SAM's, their default for the BOP runs")
    ap.add_argument(
        "--select",
        choices=["top1", "union"],
        default="top1",
        help="top1 takes the highest-scoring detection CNOS assigned to the object; union merges every detection it "
        "assigned to it, which for a scene holding two of the same kind merges both",
    )
    ap.add_argument(
        "--ref-obj-ids",
        default="",
        help="comma separated object ids to put in the reference set. Empty means every object with templates, which "
        "is the real protocol: CNOS has to pick the right one out of the whole catalogue, including the sibling "
        "it might confuse",
    )
    ap.add_argument("--obj-ids", default="", help="comma separated subset of view sets")
    ap.add_argument("--limit-sets", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.torch_home:
        # torch.hub.get_dir() reads the environment on every call, so setting it here still takes effect for the load below
        import os

        os.environ["TORCH_HOME"] = str(Path(args.torch_home).expanduser().resolve())

    cnos_root = Path(args.cnos_root).expanduser().resolve()
    template_dir = Path(args.template_dir).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest, args.bop_root)
    sets = manifest["sets"]
    if args.obj_ids:
        wanted = {int(x) for x in args.obj_ids.split(",") if x.strip()}
        sets = [s for s in sets if s["obj_id"] in wanted]
    if args.limit_sets > 0:
        step = max(1, len(sets) // args.limit_sets)
        sets = sets[::step][: args.limit_sets]
    if not sets:
        raise SystemExit("no view sets selected")

    # frame -> every (set, view, object) that wants a mask out of it. This is the whole reason the proposal pass can run once per
    # frame.
    work: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for vs in sets:
        for i, img in enumerate(vs["images"]):
            work[img].append((vs["set_id"], i, int(vs["obj_id"])))
    images = sorted(work)

    ref_ids = [int(x) for x in args.ref_obj_ids.split(",") if x.strip()]
    print(
        f"{len(sets)} view sets, {len(images)} unique frames "
        f"({sum(len(v) for v in work.values())} masks to write) -> {out_dir}"
    )

    model, cfg, device = load_cnos(cnos_root, args)
    from src.model.utils import Detections  # noqa: E402  (on sys.path by now)

    ref_feats, ref_obj_ids, cache_path = reference_descriptors(
        model, cnos_root, template_dir, args.level_templates, ref_ids, device
    )
    model.ref_data = {"descriptors": ref_feats}
    obj_to_ref = {oid: i for i, oid in enumerate(ref_obj_ids)}
    print(f"reference set: {len(ref_obj_ids)} objects x {ref_feats.shape[1]} templates ({cache_path})")
    unknown = sorted({o for v in work.values() for (_, _, o) in v} - set(obj_to_ref))
    if unknown:
        raise SystemExit(
            f"view sets ask for object(s) {unknown}, which are not in the reference set - render their templates or "
            f"widen --ref-obj-ids"
        )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    t_proposal, t_matching, n_props = [], [], []
    rows, empty_per_obj = [], defaultdict(int)
    written = 0
    for ii, img_path in enumerate(images, 1):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        H, W = rgb.shape[:2]

        sync()
        t0 = time.time()
        proposals = model.segmentor_model.generate_masks(rgb)
        detections = Detections(proposals)
        detections.remove_very_small_detections(config=model.post_processing_config.mask_post_processing)
        # SAM finding nothing large enough is rare but not impossible, and every call below assumes a non-empty batch. Treat it as
        # the frame answering "no object here" rather than letting topk raise on an empty tensor.
        n_proposals = int(len(detections.masks))
        if n_proposals:
            descriptors = model.descriptor_model(rgb, detections)
        sync()
        t1 = time.time()

        if n_proposals:
            (idx_sel, pred_obj_idx, pred_scores, _dist) = model.find_matched_proposals(descriptors)
            detections.filter(idx_sel)
            detections.add_attribute("scores", pred_scores)
            detections.add_attribute("object_ids", pred_obj_idx)
            if len(pred_scores) > 0:
                detections.apply_nms_per_object_id(nms_thresh=model.post_processing_config.nms_thresh)
        sync()
        t2 = time.time()

        t_proposal.append(t1 - t0)
        t_matching.append(t2 - t1)
        n_props.append(n_proposals)

        keep = n_proposals and len(detections.object_ids) > 0
        masks = detections.masks.cpu().numpy().astype(bool) if keep else None
        obj_of = detections.object_ids.cpu().numpy() if keep else np.zeros(0, int)
        scores = detections.scores.cpu().numpy() if keep else np.zeros(0)

        for set_id, view, obj_id in work[img_path]:
            ref_i = obj_to_ref[obj_id]
            hit = np.flatnonzero(obj_of == ref_i) if masks is not None else np.zeros(0, int)
            if len(hit) == 0:
                # CNOS assigned nothing in this frame to this object. That is an answer, not a failure to record: downstream it
                # becomes a view contributing no points, exactly as an empty SAM 3 mask does.
                m = np.zeros((H, W), bool)
                score = float("nan")
            elif args.select == "union":
                m = masks[hit].any(axis=0)
                score = float(scores[hit].max())
            else:
                best = hit[int(np.argmax(scores[hit]))]
                m = masks[best]
                score = float(scores[best])
            if not m.any():
                empty_per_obj[obj_id] += 1
            if not cv2.imwrite(str(out_dir / f"{set_id}_{view:02d}.png"), m.astype(np.uint8) * 255):
                raise SystemExit(f"failed to write mask for {set_id} view {view}")
            written += 1
            rows.append(
                {
                    "set_id": set_id,
                    "view": view,
                    "obj_id": obj_id,
                    "image": img_path,
                    "score": None if score != score else round(score, 4),
                    "n_candidates": int(len(hit)),
                    "px": int(m.sum()),
                }
            )

        if ii % 10 == 0 or ii == len(images):
            print(
                f"  [{ii}/{len(images)}] {n_props[-1]} proposals, "
                f"{t_proposal[-1] * 1000:.0f} ms proposals + "
                f"{t_matching[-1] * 1000:.0f} ms matching"
            )

    # The first frame carries model warm-up and CUDA kernel compilation; it is part of nobody's per-frame cost. Report it
    # separately rather than folding it into a mean that then cannot be compared with anything.
    def ms(xs):
        return {
            "median": round(1000 * statistics.median(xs), 1),
            "mean": round(1000 * sum(xs) / len(xs), 1),
            "min": round(1000 * min(xs), 1),
            "max": round(1000 * max(xs), 1),
        }

    warm = slice(1, None) if len(images) > 1 else slice(None)
    cost = {
        "n_frames": len(images),
        "first_frame_s": round(t_proposal[0] + t_matching[0], 2),
        "proposal_stage_ms": ms(t_proposal[warm]),
        "matching_stage_ms": ms(t_matching[warm]),
        "total_ms": ms([a + b for a, b in zip(t_proposal[warm], t_matching[warm])]),
        "proposals_per_frame": {"median": statistics.median(n_props), "max": max(n_props)},
        "device": str(device),
        "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
    }

    record = {
        "manifest": args.manifest,
        "n_sets": len(sets),
        "n_frames": len(images),
        "n_masks_written": written,
        "git_revision": git_revision(),
        "cnos_revision": cnos_revision(cnos_root),
        "cnos_root": str(cnos_root),
        "template_dir": str(template_dir),
        "segmentor": args.segmentor,
        "select": args.select,
        "level_templates": args.level_templates,
        "confidence_thresh": args.confidence_thresh,
        "stability_score_thresh": (args.stability_score_thresh if args.segmentor == "sam" else None),
        "aggregation_function": str(model.matching_config.aggregation_function),
        "descriptor_model": str(model.descriptor_model.model_name),
        "reference_objects": ref_obj_ids,
        "n_templates_per_object": int(ref_feats.shape[1]),
        "cost": cost,
        "empty_masks_per_object": {str(k): v for k, v in sorted(empty_per_obj.items())},
    }
    with open(out_dir / "cnos_run.json", "w") as f:
        json.dump(record, f, indent=2)
    with open(out_dir / "cnos_detections.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{written} masks written, {sum(empty_per_obj.values())} empty (object not found)")
    print(
        f"per frame: {cost['proposal_stage_ms']['median']:.0f} ms proposals + "
        f"{cost['matching_stage_ms']['median']:.0f} ms matching "
        f"(median over {len(images) - 1} frames after warm-up)"
    )
    print(
        f"-> {out_dir}\n-> {out_dir / 'cnos_run.json'} (config, cost, empties)"
        f"\n-> {out_dir / 'cnos_detections.json'} (per-mask score and size)"
    )


if __name__ == "__main__":
    main()
