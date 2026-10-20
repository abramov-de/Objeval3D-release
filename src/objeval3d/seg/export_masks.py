"""Export the segmentation head's masks so that a second reconstruction model can reuse them.

    python -m objeval3d.seg.export_masks --runs-root workspace/runs/lmo --pattern "*__head_full" \\
        --manifest configs/manifests/lmo_v3.json --out-dir workspace/masks/lmo_head

Comparing two reconstruction models is only a comparison of the models if everything else is held fixed, and the mask is the
largest everything else. But the head runs on DA3's frozen backbone, so it cannot be evaluated inside VGGT's process - and the two
models cannot even share a conda environment. The way out is to compute the masks once and feed the same pixels to both through
`--mask-source dir`.

Nothing is recomputed here. The masks were written into every run's `pred.npz` by the pass that produced the cloud, so this is a
re-export and the head does not run again. That also means the exported mask is exactly the one the DA3 numbers were computed from:
any difference downstream is the reconstruction model, not a second inference of the segmenter.

Written at the resolution they were produced at, not upsampled to the original frame. `dir_masks_for_set` resizes to whatever the
consumer needs, so going through the full frame would add a second resampling for no gain. DA3's preprocessing preserves the
proportions (BOP's 640x480 becomes 504x378 at process-res 504, a uniform scale), so a plain resize on the far side lands correctly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-export head masks for another backend")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--pattern", default="*__head", help="glob over run folder names")
    ap.add_argument(
        "--manifest",
        required=True,
        help="run folders are reused between experiments, so the manifest is the authority on which sets belong here",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--allow-mask-source",
        default="head",
        help="refuse runs whose meta reports a different mask source. Dumping ground-truth masks under a name that "
        "says 'head' would quietly turn an ablation into an oracle",
    )
    args = ap.parse_args()

    with open(args.manifest) as f:
        wanted = {s["set_id"] for s in json.load(f)["sets"]}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(Path(args.runs_root).glob(args.pattern))
    written, skipped, provenance = 0, [], []
    for run in runs:
        meta_path, pred_path = run / "meta.json", run / "pred.npz"
        if not (meta_path.exists() and pred_path.exists()):
            skipped.append((run.name, "no meta.json or pred.npz"))
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        set_id = meta["view_set"]["set_id"]
        if set_id not in wanted:
            skipped.append((run.name, "not in the manifest"))
            continue
        src = meta.get("mask_source")
        if args.allow_mask_source and src != args.allow_mask_source:
            skipped.append((run.name, f"mask_source={src!r}"))
            continue

        masks = np.load(pred_path)["obj_mask"]
        for i, m in enumerate(masks):
            cv2.imwrite(str(out_dir / f"{set_id}_{i:02d}.png"), np.asarray(m).astype(np.uint8) * 255)
        written += len(masks)
        mm = meta.get("mask_metrics") or {}
        provenance.append(
            {
                "set_id": set_id,
                "run_dir": str(run),
                "n_views": int(len(masks)),
                "mask_source": src,
                # the identity of the head: a retrained head writes the same filename, so the path alone is a weak identity
                "head_fingerprint": meta.get("head_fingerprint") or meta.get("seg_ckpt_fingerprint"),
                "head": meta.get("head") or meta.get("seg_ckpt"),
                "backend": meta.get("backend", "da3"),
                # carried over rather than recomputed: it is what the DA3 numbers were scored against, and it is the check that the
                # right masks were exported
                "mask_iou_vs_gt": (mm.get("aggregate") or {}).get("iou"),
            }
        )

    ious = [p["mask_iou_vs_gt"] for p in provenance if p["mask_iou_vs_gt"] is not None]
    fps = {p["head_fingerprint"] or p["head"] for p in provenance}
    no_fp = sum(1 for p in provenance if not p["head_fingerprint"])
    with open(out_dir / "head_masks.json", "w") as f:
        json.dump(
            {
                "source_runs_root": args.runs_root,
                "pattern": args.pattern,
                "manifest": args.manifest,
                "n_sets": len(provenance),
                "n_masks": written,
                "mean_mask_iou_vs_gt": float(np.mean(ious)) if ious else None,
                "head_fingerprints": sorted(x for x in fps if x),
                "sets": provenance,
            },
            f,
            indent=2,
        )

    print(f"{len(provenance)} sets, {written} masks -> {out_dir}")
    if ious:
        print(f"  mask IoU vs GT, carried from the runs: mean {np.mean(ious):.3f} [{min(ious):.3f}, {max(ious):.3f}]")
    if len(fps) > 1:
        # Masks from two heads in one directory would be invisible downstream and would make the comparison meaningless.
        print(f"  [warn] masks come from {len(fps)} different checkpoints: {sorted(str(x) for x in fps)}")
    if no_fp:
        print(f"  [note] {no_fp} run(s) carry no checkpoint fingerprint and are identified by path only")
    if skipped:
        print(f"  {len(skipped)} run(s) skipped:")
        for name, why in skipped[:8]:
            print(f"    {name}: {why}")
        if len(skipped) > 8:
            print(f"    ... (+{len(skipped) - 8} more)")
    if not provenance:
        raise SystemExit("nothing exported - check --pattern and --allow-mask-source")


if __name__ == "__main__":
    main()
