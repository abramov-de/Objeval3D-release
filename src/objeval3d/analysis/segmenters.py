"""Tables 2-4: the segmentation stage, measured by what the cloud built from each source of masks says about the reconstruction.

Sources of masks: the ground-truth annotation of the dataset, the trained head, and three segmenters of other kinds (a CAD-model
segmenter, a network trained on the whole dataset for each object, a language-driven segmenter). Every quantity is read from the
result files written by `objeval3d.align.evaluate` in the reference mode:

    accuracy      mean over view sets of the per-set median distance from the cloud to the scan
    completeness  mean over view sets of the per-set median distance from the visible part of the scan to the cloud
    F@2%          mean over view sets

A view set is *measurable* for a source if the object cloud built from its masks has at least 200 points; the others are not scored
(status `too_few_points`) and their number is part of the table.
"""

from __future__ import annotations

from pathlib import Path

from objeval3d.analysis.common import accuracy, completeness, fscore, load_sets, mean_over, object_key
from objeval3d.analysis.table import Table, fmt

# result files of the five sources of masks: LM-O (one capture, all objects) and YCB-V within one capture (the split the head is made for)
LMO = {
    "gt": "lmo_gt_full",
    "head": "lmo_head_full",
    "zebrapose": "lmo_dir_zebrapose",
    "cnos": "lmo_dir_cnos",
    "sam3": "lmo_dir_sam3_tuned",
}
YCBV = {
    "gt": "ycbv_b20_gt",
    "head": "ycbv_b20_head",
    "zebrapose": "ycbv_b20_dir_zebrapose",
    "cnos": "ycbv_b20_dir_cnos",
    "sam3": "ycbv_b20_dir_sam3",
}
SOURCES = ("gt", "head", "zebrapose", "cnos", "sam3")
LABEL = {"gt": "Ground-truth", "head": "Our method", "zebrapose": "ZebraPose", "cnos": "CNOS", "sam3": "SAM 3"}
CLAMPS = (
    "o19",
    "o20",
)  # "large clamp" and "extra large clamp": one object of two sizes, the name does not tell them apart


def head_vs_ground_truth(results: Path) -> Table:
    """Table 2: the trained head against the ground-truth masks on LM-O, on the view sets common to both."""
    gt, hd = load_sets(results, LMO["gt"]), load_sets(results, LMO["head"])
    ids = sorted(set(gt) & set(hd))

    def mask_stat(d, key):
        return mean_over(d, ids, lambda r: r["mask_metrics"]["aggregate"][key])

    f_gt, f_hd = (mean_over(d, ids, fscore) for d in (gt, hd))
    c_gt, c_hd = (mean_over(d, ids, completeness) for d in (gt, hd))
    a_gt, a_hd = (mean_over(d, ids, accuracy) for d in (gt, hd))

    def rel(new, old):
        return fmt(100 * (new - old) / old, 1, sign=True) + "%"

    def mask(d):
        return f"{mask_stat(d, 'precision'):.3f} / {mask_stat(d, 'recall'):.3f} ({mask_stat(d, 'iou'):.3f})"

    return Table(
        "table2",
        "3D reconstruction quality with predicted and ground-truth masks on LM-O.",
        ["Metric", "GT masks", "Head masks", "Relative change"],
        [
            ["Precision / Recall (IoU) ↑", mask(gt), mask(hd), "---"],
            ["F@2% ↑", fmt(f_gt, 3), fmt(f_hd, 3), rel(f_hd, f_gt)],
            ["Completeness (mm) ↓", fmt(c_gt), fmt(c_hd), rel(c_hd, c_gt)],
            ["Accuracy (mm) ↓", fmt(a_gt), fmt(a_hd), rel(a_hd, a_gt)],
        ],
        [f"{len(ids)} view sets common to both sources, reference mode, means over view sets."],
    )


def quality_by_source(results: Path) -> Table:
    """Table 3: F@2% and accuracy of the five sources of masks, LM-O and YCB-V, with the accuracy without the failed sets.

    LM-O is read on the view sets common to the ground truth, the head, ZebraPose and CNOS; the language-driven segmenter on those of
    them where its masks give a measurable cloud (marked with a dagger), and the last column of each group leaves out the sets that
    some source loses: on LM-O those where the language-driven segmenter has no cloud, on YCB-V the sets of the two clamp objects.
    """
    lmo = {k: load_sets(results, f) for k, f in LMO.items()}
    common = set.intersection(*(set(lmo[k]) for k in SOURCES if k != "sam3"))
    measurable = common & set(lmo["sam3"])
    ycbv = {k: load_sets(results, f) for k, f in YCBV.items()}
    base = set(ycbv["gt"])

    rows = []
    for k in SOURCES:
        ids_l = measurable if k == "sam3" else common
        dag = "†" if k == "sam3" else ""
        ids_y = set(ycbv[k]) & base
        rest_y = {s for s in ids_y if object_key(s) not in CLAMPS}
        rows.append(
            [
                LABEL[k],
                fmt(mean_over(lmo[k], ids_l, fscore), 3) + dag,
                fmt(mean_over(lmo[k], ids_l, accuracy)) + dag,
                fmt(mean_over(lmo[k], measurable, accuracy)),
                fmt(mean_over(ycbv[k], ids_y, fscore), 3),
                fmt(mean_over(ycbv[k], ids_y, accuracy)),
                fmt(mean_over(ycbv[k], rest_y, accuracy)),
                f"{len(ids_l)} / {len(ids_y)}",
            ]
        )
    return Table(
        "table3",
        "3D reconstruction quality using masks from different segmentation methods on LM-O and YCB-V.",
        [
            "Mask source",
            "LM-O F@2% ↑",
            "LM-O Accuracy (mm) ↓",
            "LM-O Accuracy without failed sets (mm) ↓",
            "YCB-V F@2% ↑",
            "YCB-V Accuracy (mm) ↓",
            "YCB-V Accuracy without failed sets (mm) ↓",
            "Valid sets LM-O / YCB-V",
        ],
        rows,
        [
            f"LM-O: {len(common)} view sets common to four sources; the language-driven segmenter has a measurable cloud on {len(measurable)} of them (†). "
            f"YCB-V within one capture: {len(base)} sets.",
            "Reference mode, means over view sets. A set is valid if the cloud built from the masks has at least 200 points.",
        ],
    )
