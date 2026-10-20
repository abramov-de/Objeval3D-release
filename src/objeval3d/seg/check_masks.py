"""Score a directory of masks against the ground truth, without a GPU.

Masks from another model are cheap to produce and expensive to evaluate: the honest number only appears after a reconstruction and
alignment pass, which takes an hour. That is far too slow to choose a setting with, and choosing one from the final 3D metrics would
be fitting the baseline to the test set anyway. This reads the same BOP visible masks that `objeval3d.recon.reconstruct` compares
against, at the same processing resolution, and reports the same `mask_metrics` in seconds on the CPU. It is the check to run before
spending the hour, and the number it prints is the one the full run will report.

    python -m objeval3d.seg.check_masks --manifest configs/manifests/ycbv_v3.json --mask-dir workspace/masks/ycbv_sam3 --obj-ids 19

Precision and recall are reported next to IoU: on this protocol precision governs accuracy and recall governs completeness, and a
mask can gain IoU while losing the metric that matters. An empty mask counts as a view with zero recall, which is what it becomes
downstream - a view contributing no points at all.

Two statistics, and the difference between them matters. The paper's mask numbers are POOLED WITHIN A VIEW SET: `mask_iou` over the
set's three masks stacked, stored by `reconstruct` as `mask_metrics.aggregate`, then averaged over sets. The per-view mean is a
different number: a view landing entirely off the object adds its whole area to the pooled union, while in a per-view mean it is one
zero among three. On a source that is right in some views and wrong in others the gap is large, so comparing across the two looks
like a change in the masks. Both are printed, the pooled one first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from objeval3d.data.bop import dir_masks_for_set, gt_masks_for_set, mask_iou, target_hw
from objeval3d.data.manifest import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="IoU of a mask directory vs BOP ground truth")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--obj-ids", default="", help="comma separated subset")
    ap.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="must match the reconstruction run, or the comparison is between masks resized differently",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest, args.bop_root)
    bop_root = Path(manifest["bop_root"])
    sets = manifest["sets"]
    if args.obj_ids:
        wanted = {int(x) for x in args.obj_ids.split(",") if x.strip()}
        sets = [s for s in sets if s["obj_id"] in wanted]
    if not sets:
        raise SystemExit("no view sets selected")

    mask_dir = Path(args.mask_dir)
    if not mask_dir.is_dir():
        raise SystemExit(f"no such mask directory: {mask_dir}")

    per_obj_view = defaultdict(list)
    per_obj_set = defaultdict(list)
    per_view, per_set = [], []
    for vs in sets:
        hw = target_hw(Path(vs["images"][0]), args.process_res)
        pred = dir_masks_for_set(vs, mask_dir, hw)
        gt = gt_masks_for_set(vs, bop_root, hw)
        key = (vs["obj_id"], vs["obj_name"])
        pooled = mask_iou(pred, gt)
        per_obj_set[key].append(pooled)
        per_set.append(
            {
                "set_id": vs["set_id"],
                "obj_id": vs["obj_id"],
                "obj_name": vs["obj_name"],
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in pooled.items()},
            }
        )
        for i in range(len(vs["im_ids"])):
            m = mask_iou(pred[i], gt[i])
            per_obj_view[key].append(m)
            per_view.append(
                {
                    "set_id": vs["set_id"],
                    "view": i,
                    "obj_id": vs["obj_id"],
                    "obj_name": vs["obj_name"],
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                }
            )

    print(
        f"{'object':22s} {'sets':>4s} {'views':>5s} {'empty':>5s} "
        f"{'IoU':>6s} {'prec':>6s} {'recall':>6s}   {'IoU/v':>6s} {'prec/v':>6s}"
    )
    rows = []

    # nanmean: a view with no predicted and no ground-truth pixels has an undefined IoU, and averaging it in as 0 would punish the
    # wrong thing. When a prompt finds nothing anywhere, every precision is nan (precision over zero predicted pixels is
    # undefined), so it is reported as "-" rather than letting a warning interleave with every result line.
    def nanmean(ms, k):
        vals = [m[k] for m in ms if m[k] == m[k]]
        return float(np.mean(vals)) if vals else float("nan")

    def fmt(v):
        return "     -" if v != v else f"{v:6.3f}"

    for key in sorted(per_obj_set):
        oid, name = key
        ss, ms = per_obj_set[key], per_obj_view[key]
        empty = sum(1 for m in ms if m["pred_px"] == 0)
        print(
            f"{name:22s} {len(ss):4d} {len(ms):5d} {empty:5d} "
            f"{fmt(nanmean(ss, 'iou'))} {fmt(nanmean(ss, 'precision'))} "
            f"{fmt(nanmean(ss, 'recall'))}   "
            f"{fmt(nanmean(ms, 'iou'))} {fmt(nanmean(ms, 'precision'))}"
        )
        rows.append(
            {
                "obj_id": oid,
                "obj_name": name,
                "n_sets": len(ss),
                "n_views": len(ms),
                "n_empty": empty,
                "iou": nanmean(ss, "iou"),
                "precision": nanmean(ss, "precision"),
                "recall": nanmean(ss, "recall"),
                "iou_per_view": nanmean(ms, "iou"),
                "precision_per_view": nanmean(ms, "precision"),
                "recall_per_view": nanmean(ms, "recall"),
            }
        )

    if len(rows) > 1:
        alls = [m for ss in per_obj_set.values() for m in ss]
        allm = [m for ms in per_obj_view.values() for m in ms]
        print(
            f"{'ALL':22s} {len(alls):4d} {len(allm):5d} "
            f"{sum(r['n_empty'] for r in rows):5d} {fmt(nanmean(alls, 'iou'))} "
            f"{fmt(nanmean(alls, 'precision'))} {fmt(nanmean(alls, 'recall'))}   "
            f"{fmt(nanmean(allm, 'iou'))} {fmt(nanmean(allm, 'precision'))}"
        )
    print(
        "\nIoU/prec/recall are pooled within a view set then averaged over sets - the statistic `reconstruct` records as "
        "mask_metrics.aggregate, i.e. the one in the paper.\n"
        "IoU/v and prec/v are the per-view means, kept as a diagnostic: a gap between the two says the source is right in some "
        "views of a set and wrong in others."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            # Per view as well as per object. A segmenter that takes the right object in some views and the wrong one in others
            # averages to a middling IoU that describes neither.
            json.dump({"args": vars(args), "per_object": rows, "per_set": per_set, "per_view": per_view}, f, indent=2)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
