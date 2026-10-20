"""Stages 1-2 of the protocol: reconstruct one view set with a feed-forward model, segment the target object, and export the cleaned
object point cloud.

The source of the object mask is a switch, because the point of the paper is to measure how much it matters:

  --mask-source head   a trained segmentation head on the frozen backbone of Depth Anything 3 (--head CKPT)
  --mask-source gt     the BOP ground-truth visible masks (the upper bound)
  --mask-source dir    masks computed elsewhere, one PNG per view (--mask-dir): SAM 3, CNOS, ZebraPose, or the head's own masks
                       dumped once so that two reconstruction models are given identical ones
  --mask-source none   no cleaning at all: the whole scene is the "object" (the floor of the measurement; it shows what the mask buys)

`--crop-to-object` reconstructs from a crop framed on the object instead of the whole frame (`gt`: the BOP visible mask gives the box,
`head`: a first pass of the head does). It is a control, not a recommendation: with little scene context left the model's estimate of
the relative poses degrades (the paper's limitations). The head is always run on the FULL frame, as it was trained, and its mask is
carried into the crop with the same box.

Cleaning: the masked depth is back-projected into the reconstruction's own world frame and the points whose depth confidence is
below the `--conf-thresh-percentile`-th percentile are dropped (the 40th by default). The percentile is measured over the whole
frame (`--conf-percentile-scope frame`, the default) or among the object's own pixels (`mask`), which is the only scope that means
the same thing to two different models: use it to compare models.

Everything a later stage could need is written to `pred.npz`, so the expensive forward pass runs exactly once per view set. A run
is the folder `<out-root>/<set_id>__<tag>/` with `object_points.ply`, `pred.npz`, `meta.json` and `overlays/`.

    python -m objeval3d.recon.reconstruct --manifest configs/manifests/lmo_v3.json --out-root workspace/runs/lmo \\
        --mask-source head --head workspace/ckpt/lmo_seg_head.pt --tag head_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from objeval3d.data.bop import dir_masks_for_set, gt_masks_for_set, mask_iou, write_ply
from objeval3d.data.manifest import load_manifest
from objeval3d.models import DA3_REPO, DA3_REVISION, PROCESS_RES
from objeval3d.recon.backends import load_predictor
from objeval3d.recon.cloud import (
    backproject,
    ckpt_fingerprint,
    compute_crop_boxes,
    gt_masks_full_res,
    multiview_consistency,
    save_overlays,
)

CONF_PERCENTILE = 40.0  # drop the points below this percentile of the depth confidence


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct view sets, segment the object, export its point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", required=True, help="from objeval3d.data.view_sets, or configs/manifests/*.json")
    ap.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--set-ids", default="", help="comma separated subset of set_id to run")
    ap.add_argument("--limit", type=int, default=0, help="run only N view sets, spread over the manifest")

    ap.add_argument(
        "--backend",
        default="da3",
        choices=("da3", "vggt"),
        help="which feed-forward reconstruction model runs. They need separate conda environments, so masks are "
        "computed once and passed to both with --mask-source dir",
    )
    ap.add_argument(
        "--vggt-repo",
        default=None,
        help="checkout of facebookresearch/vggt; defaults to the pinned submodule third_party/vggt-repo",
    )
    ap.add_argument("--base-model", default=DA3_REPO, help="Hugging Face id or a local directory")
    ap.add_argument("--revision", default=DA3_REVISION, help="Hugging Face revision of --base-model")
    ap.add_argument("--head", default="", help="checkpoint of the segmentation head (--mask-source head)")
    ap.add_argument("--process-res", type=int, default=PROCESS_RES)

    ap.add_argument("--mask-source", choices=["head", "gt", "dir", "none"], default="head")
    ap.add_argument("--mask-dir", default="", help="--mask-source dir")
    ap.add_argument(
        "--crop-to-object",
        choices=["none", "gt", "head"],
        default="none",
        help="reconstruct from a crop framed on the object instead of the whole frame; the box comes from the BOP visible mask "
        "(gt, deterministic and identical across mask sources) or from a first pass of the head (head, no ground truth)",
    )
    ap.add_argument(
        "--crop-margin", type=float, default=0.35, help="padding around the object box, as a fraction of its size"
    )
    ap.add_argument("--crop-rect", action="store_true", help="keep the crop rectangular instead of square")
    ap.add_argument(
        "--conf-percentile-scope",
        choices=("frame", "mask"),
        default="frame",
        help="where the confidence percentile is measured. 'frame' (default) ranks over the whole image; 'mask' ranks "
        "among the object's own pixels, which is the only one comparable across models",
    )
    ap.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=CONF_PERCENTILE,
        help="drop points whose depth confidence falls below this percentile (of the whole frame, or of the object, "
        "see --conf-percentile-scope). Confidence is not the same quantity across backends, so a shared "
        "percentile is not a shared filter unless the scope is 'mask'",
    )
    ap.add_argument("--tag", default="", help="suffix for the run folder, e.g. 'gt_full'")
    ap.add_argument("--save-scene-ply", action="store_true", help="also write the whole back-projected scene")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave runs that already produced a cloud alone - makes a long run resumable after a failure part-way",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest, args.bop_root)
    bop_root = Path(manifest["bop_root"])
    sets = manifest["sets"]
    if args.set_ids:
        wanted = set(args.set_ids.split(","))
        sets = [s for s in sets if s["set_id"] in wanted]
    if args.limit > 0:
        # Spread the subset over the manifest rather than taking its head. The manifest is ordered scene-then-object, so
        # `sets[:6]` is six view sets of ONE object in ONE scene.
        step = max(1, len(sets) // args.limit)
        sets = sets[::step][: args.limit]
    print(f"{len(sets)} view sets to process")

    need_head = args.mask_source == "head" or args.crop_to_object == "head"
    if need_head and args.backend != "da3":
        # Refused here rather than after loading a 1B model: the head is trained on DA3's frozen backbone and has no
        # counterpart elsewhere.
        raise SystemExit(
            f"--backend {args.backend} has no segmentation head. Dump the head's masks once and pass "
            "--mask-source dir --mask-dir <path> to every backend."
        )
    if need_head and not args.head:
        raise SystemExit("--mask-source head (and --crop-to-object head) require --head")
    if args.mask_source == "dir" and not Path(args.mask_dir).is_dir():
        raise SystemExit(f"--mask-source dir requires --mask-dir to be a directory (got {args.mask_dir!r})")

    model, ckpt = load_predictor(
        args.backend,
        device=args.device,
        base_model=args.base_model,
        revision=args.revision,
        head_ckpt=args.head if need_head else "",
        vggt_repo=args.vggt_repo,
    )
    obj_to_class: dict[int, int] = {}
    if need_head:
        num_classes = int(ckpt.get("num_classes", 2))
        class_names = ckpt.get("class_names", [f"class_{i}" for i in range(num_classes)])
        obj_ids = ckpt.get("obj_ids") or manifest["obj_ids"]
        obj_to_class = {int(o): i + 1 for i, o in enumerate(obj_ids)}
        print(f"seg head: {num_classes} classes {class_names}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.mask_source

    head_fp = ckpt_fingerprint(args.head)

    for si, vs in enumerate(sets):
        run_dir = out_root / f"{vs['set_id']}__{tag}"
        if args.skip_existing and (run_dir / "meta.json").exists() and (run_dir / "object_points.ply").exists():
            # Only skip a run the *current* settings produced. A stale run is worse than a missing one: it survives into the
            # results file looking exactly like a fresh one. Everything that changes the output without changing the folder
            # name has to be compared here.
            want = {
                "head_fingerprint": head_fp,
                "backend": args.backend,
                "conf_thresh_percentile": args.conf_thresh_percentile,
                "conf_percentile_scope": args.conf_percentile_scope,
            }
            try:
                with open(run_dir / "meta.json") as f:
                    prev = json.load(f)
                have = {k: prev.get(k) for k in want}
            except Exception:
                have = {"head_fingerprint": "<unreadable>"}
            if have == want:
                print(f"[{si + 1}/{len(sets)}] {vs['set_id']}: already done, skipping")
                continue
            diff = ", ".join(f"{k}: {have.get(k)!r} != {v!r}" for k, v in want.items() if have.get(k) != v)
            print(f"[{si + 1}/{len(sets)}] {vs['set_id']}: existing run was made differently ({diff}), redoing")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{si + 1}/{len(sets)}] {vs['set_id']} -> {run_dir}")

        # The head is always run on the FULL frame, never on the crop. It was trained on full frames where an LM-O object spans ~30 px;
        # on a crop upscaled to the processing resolution the object fills the view, an appearance regime it has never seen, and it
        # collapses (mask IoU 0.76 -> 0.06). The full-frame mask is instead carried into the crop with the same box.
        boxes = None
        inputs = vs["images"]
        head_mask_full = None

        if args.crop_to_object != "none":
            originals = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in vs["images"]]
            H0, W0 = originals[0].shape[:2]

            if need_head:
                # One full-frame pass: gives the head's mask at native scale, used both to place the crop and as the object mask.
                probe = model.inference(
                    image=vs["images"], process_res=args.process_res, infer_seg=True, export_dir=None
                )
                cm = np.asarray(probe.seg_logits).argmax(axis=1)
                tc = obj_to_class.get(int(vs["obj_id"]), 1)
                head_mask_full = np.stack(
                    [
                        cv2.resize((cm[i] == tc).astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
                        for i in range(len(cm))
                    ]
                )
                del probe

            if args.crop_to_object == "gt":
                box_masks = list(gt_masks_full_res(vs, bop_root))
            else:
                box_masks = list(head_mask_full)

            box_masks = [m if m is not None else np.zeros((H0, W0), bool) for m in box_masks]
            boxes = compute_crop_boxes(box_masks, (H0, W0), args.crop_margin, square=not args.crop_rect)
            if boxes is None:
                print("  [warn] object not found in any view, reconstructing the full frame")
            else:
                inputs = [img[y0:y1, x0:x1] for img, (x0, y0, x1, y1) in zip(originals, boxes)]
                print(
                    f"  crop {boxes[0][2] - boxes[0][0]}x{boxes[0][3] - boxes[0][1]} px from {W0}x{H0} (source: {args.crop_to_object})"
                )

        pred = model.inference(
            image=inputs, process_res=args.process_res, infer_seg=need_head and head_mask_full is None, export_dir=None
        )

        depth = np.asarray(pred.depth)
        n, h, w = depth.shape
        conf = None if pred.conf is None else np.asarray(pred.conf)
        K = np.asarray(pred.intrinsics)
        E = np.asarray(pred.extrinsics)
        imgs = np.asarray(pred.processed_images)

        # ---- object mask ---------------------------------------------------
        class_map = np.zeros((n, h, w), dtype=np.int16)
        target_class = 1
        if args.mask_source == "head" and head_mask_full is not None:
            # full-frame head mask carried into the crop
            obj_mask = np.stack(
                [
                    cv2.resize(
                        head_mask_full[i][boxes[i][1] : boxes[i][3], boxes[i][0] : boxes[i][2]].astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                    for i in range(n)
                ]
            )
            class_map = obj_mask.astype(np.int16)
            target_class = 1
        elif args.mask_source == "head":
            logits = pred.seg_logits
            if logits is None:
                raise RuntimeError("seg_logits is None - the head did not run")
            class_map = np.asarray(logits).argmax(axis=1).astype(np.int16)
            target_class = obj_to_class.get(int(vs["obj_id"]), 1)
            obj_mask = class_map == target_class
        elif args.mask_source == "gt":
            obj_mask = gt_masks_for_set(vs, bop_root, (h, w), boxes)
            class_map = obj_mask.astype(np.int16)
        elif args.mask_source == "dir":
            obj_mask = dir_masks_for_set(vs, Path(args.mask_dir), (h, w))
            class_map = obj_mask.astype(np.int16)
        else:
            obj_mask = np.ones((n, h, w), dtype=bool)
            class_map = obj_mask.astype(np.int16)

        # ---- 2D mask quality vs GT (free, BOP always has it) ---------------
        gt_mask = gt_masks_for_set(vs, bop_root, (h, w), boxes)
        per_view_iou = [mask_iou(obj_mask[i], gt_mask[i]) for i in range(n)]
        agg = mask_iou(obj_mask, gt_mask)

        # ---- unprojection --------------------------------------------------
        # Where the percentile is measured decides whether the filter means the same thing to two different models. `frame`
        # thresholds a percentile of confidence over the whole frame: a small dark object can sit under a frame-wide
        # threshold on one model's scale and above it on another's (it cut a small object from 5,061 points to 68 on VGGT
        # while DA3 kept 1,186). `mask` ranks confidence among the object's own pixels, so each model discards the
        # least-confident fraction of the object it found.
        conf_thr = None
        if conf is not None and args.conf_thresh_percentile > 0:
            pool = conf
            if args.conf_percentile_scope == "mask":
                pool = conf[obj_mask & (depth > 0)]
                if pool.size < 20:
                    # Too few of the object's pixels to rank; filtering on them would be noise. Say so rather than
                    # thresholding on three values or silently falling back to the frame.
                    print(
                        f"    [warn] only {pool.size} masked pixels with depth - confidence filter skipped for this set"
                    )
                    pool = None
            if pool is not None:
                conf_thr = float(np.percentile(pool, args.conf_thresh_percentile))

        pts, cols, view_ids = backproject(depth, obj_mask, K, E, imgs, conf, conf_thr)
        write_ply(run_dir / "object_points.ply", pts, cols)
        consistency = multiview_consistency(pts, view_ids, n)
        cons_txt = f", worst view off by {100 * consistency['max_rel']:.0f}% of the cloud" if consistency else ""
        # How many of the object's pixels the confidence filter threw away: otherwise invisible, and model-specific.
        n_masked = int(((depth > 0) & obj_mask).sum())
        kept = f"{100 * len(pts) / n_masked:.0f}%" if n_masked else "n/a"
        print(f"  object cloud: {len(pts):,} points (mask IoU vs GT = {agg['iou']:.3f}{cons_txt})")
        print(
            f"    {n_masked:,} masked pixels with depth -> {kept} kept"
            + (
                f" (confidence below {conf_thr:.3g}, the {args.conf_thresh_percentile:g}th "
                f"percentile over the {args.conf_percentile_scope}, dropped)"
                if conf_thr
                else " (no confidence filter)"
            )
        )

        if args.save_scene_ply:
            sp, sc, _ = backproject(depth, np.ones_like(obj_mask), K, E, imgs, conf, conf_thr)
            write_ply(run_dir / "scene_points.ply", sp, sc)

        save_overlays(imgs, class_map, run_dir / "overlays", target_class)

        np.savez_compressed(
            run_dir / "pred.npz",
            depth=depth.astype(np.float32),
            conf=(conf.astype(np.float32) if conf is not None else np.zeros(0, np.float32)),
            intrinsics=K.astype(np.float64),
            extrinsics=E.astype(np.float64),
            obj_mask=obj_mask,
            gt_mask=gt_mask,
            class_map=class_map,
            view_ids=view_ids,
            conf_thresh=np.float64(conf_thr if conf_thr else 0.0),
            hw=np.asarray([h, w], np.int32),
        )
        with open(run_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "view_set": vs,
                    "tag": tag,
                    "mask_source": args.mask_source,
                    # WHICH external masks: without this a `dir` run cannot be traced to the directory it read
                    "mask_dir": (str(Path(args.mask_dir).resolve()) if args.mask_source == "dir" else None),
                    "mask_dir_n_files": (
                        len(list(Path(args.mask_dir).glob("*.png"))) if args.mask_source == "dir" else None
                    ),
                    "process_res": args.process_res,
                    "crop_to_object": args.crop_to_object,
                    "crop_margin": args.crop_margin,
                    "crop_boxes": ([[int(v) for v in b] for b in boxes] if boxes else None),
                    "processed_hw": [h, w],
                    "conf_thresh_percentile": args.conf_thresh_percentile,
                    "conf_percentile_scope": args.conf_percentile_scope,
                    "n_object_points": int(len(pts)),
                    "multiview_consistency": consistency,
                    "mask_metrics": {"aggregate": agg, "per_view": per_view_iou},
                    "backend": args.backend,
                    "is_metric": bool(pred.is_metric),
                    "scale_factor": pred.scale_factor,
                    "head": args.head,
                    "head_fingerprint": head_fp,
                    "base_model": args.base_model,
                    "revision": args.revision,
                },
                f,
                indent=2,
            )

    print(f"\nAll runs written under {out_root}")


if __name__ == "__main__":
    main()
