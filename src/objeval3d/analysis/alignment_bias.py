"""Tables 5 and 6: does a free fit understate the error, and is it the search or the criterion?

Table 5 puts the free fit (a search over the seven parameters of a similarity transform on the two clouds alone, the usual way of
aligning) next to the reference mode (the position from the independent anchor, only the depth scale searched). Table 6 repeats the
measurement through four third-party implementations of the search.

Statistic: the median over view sets (the paper's convention for Tables 5-9); counts are counts. A set *converged* if the search
found the reference placement, by the rule of `objeval3d.analysis.placement` (rotation under 15 degrees and the centre of the cloud
under 0.10 diameters, symmetries of the object respected).
"""

from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from objeval3d.analysis.common import accuracy, chamfer, fscore, load_sets
from objeval3d.analysis.placement import dataset_of, gaps, object_symmetries, recovered
from objeval3d.analysis.table import Table, fmt
from objeval3d.modes import FREE_FIT, REFERENCE

FILES = {"LM-O": "lmo_gt_full", "YCB-V": "ycbv_gt"}  # Table 5: three views, ground-truth masks
FILES_METHODS = {
    "LM-O": "lmo_gt_full_e9",
    "YCB-V": "ycbv_gt_full_e9",
}  # Table 6: the same sets with the third-party searches
METHODS = (
    ("fpfh_ransac", "FPFH + RANSAC"),
    ("icp_pca", "Principal axes + ICP"),
    ("teaserpp", "TEASER++"),
    ("fgr", "Fast Global Registration"),
)


def spearman(a, b) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def per_object(recs: list[dict], mode: str, f) -> np.ndarray:
    """Median over the view sets of each object, objects in alphabetical order."""
    objs = sorted({r["obj_name"] for r in recs})
    return np.array([np.median([f(r, mode) for r in recs if r["obj_name"] == o]) for o in objs])


def converged_flags(recs: list[dict], mode: str) -> list[bool]:
    flags = []
    for r in recs:
        syms = object_symmetries(dataset_of(r["set_id"]), int(r["obj_id"]))
        flags.append(recovered(gaps(r, mode, syms)))
    return flags


def free_fit_vs_reference(results: Path) -> Table:
    """Table 5: the free fit against the reference mode, three views, ground-truth masks."""
    cols = {}
    for ds, name in FILES.items():
        recs = list(load_sets(results, name, modes=(FREE_FIT, REFERENCE)).values())
        ch_f = np.array([chamfer(r, FREE_FIT) for r in recs])
        ch_r = np.array([chamfer(r, REFERENCE) for r in recs])
        ac_f = np.array([accuracy(r, FREE_FIT) for r in recs])
        ac_r = np.array([accuracy(r, REFERENCE) for r in recs])
        conv = np.array(converged_flags(recs, FREE_FIT))
        cols[ds] = {
            "n": len(recs),
            "lower_chamfer": float((ch_f < ch_r).mean()),
            "optimistic": float((ac_f < ac_r).mean()),
            "acc_free": float(np.median(ac_f)),
            "acc_ref": float(np.median(ac_r)),
            "converged": int(conv.sum()),
            "rot_median": float(np.median([r["auto_vs_oracle"]["rotation_deg"] for r in recs])),
            "conv_lower_chamfer": float((ch_f[conv] < ch_r[conv]).mean()) if conv.any() else float("nan"),
            "conv_optimistic": float((ac_f[conv] < ac_r[conv]).mean()) if conv.any() else float("nan"),
            "rho_f2": spearman(per_object(recs, FREE_FIT, fscore), per_object(recs, REFERENCE, fscore)),
            "rho_acc": spearman(per_object(recs, FREE_FIT, accuracy), per_object(recs, REFERENCE, accuracy)),
        }

    def both(f):
        return [f(cols["LM-O"]), f(cols["YCB-V"])]

    def pc(x):
        return f"{100 * x:.0f} %"

    rows = [
        ["View sets with lower Chamfer distance than at the ground-truth pose (%)"]
        + both(lambda c: pc(c["lower_chamfer"])),
        ["View sets with underestimated accuracy (%)"] + both(lambda c: pc(c["optimistic"])),
        ["Accuracy: free fit vs. reference mode (mm)"] + both(lambda c: f"{c['acc_free']:.2f} vs. {c['acc_ref']:.2f}"),
        ["View sets converged to the ground-truth pose"] + both(lambda c: f"{c['converged']} / {c['n']}"),
        ["Median rotation error from the ground-truth pose (°)"] + both(lambda c: f"{c['rot_median']:.1f}°"),
        ["Among converged sets: lower Chamfer distance than at the ground-truth pose (%)"]
        + both(lambda c: f"{pc(c['conv_lower_chamfer'])} (n={c['converged']})"),
        ["Among converged sets: underestimated accuracy (%)"]
        + both(lambda c: f"{pc(c['conv_optimistic'])} (n={c['converged']})"),
        ["Rank correlation with reference-mode accuracy"] + both(lambda c: fmt(c["rho_f2"], 2, sign=True)),
    ]
    note = (
        "The last row is Spearman's rho between the objects ordered by the per-object median F@2% of the free fit and of the reference "
        "mode; ordered by the median accuracy instead it is "
        + " and ".join(f"{fmt(cols[d]['rho_acc'], 2, sign=True)} ({d})" for d in cols)
        + "."
    )
    return Table(
        "table5",
        "Comparison of free-fit and reference-mode alignment on LM-O and YCB-V.",
        ["Metric", "LM-O", "YCB-V"],
        rows,
        [note],
    )


def alignment_methods(results: Path) -> Table:
    """Table 6: the same clouds under the reference mode, our free fit and four third-party searches."""
    cols = {}
    for ds, name in FILES_METHODS.items():
        recs = list(load_sets(results, name, modes=(REFERENCE, FREE_FIT) + tuple(m for m, _ in METHODS)).values())
        cols[ds] = {"n": len(recs), "recs": recs}
    rows = []
    for mode, label in ((REFERENCE, "Reference mode"), (FREE_FIT, "Free fit (own code)")) + METHODS:
        row = [label]
        for ds in FILES_METHODS:
            recs = cols[ds]["recs"]
            acc = st.median(accuracy(r, mode) for r in recs)
            f2 = st.median(fscore(r, mode) for r in recs)
            conv = "---" if mode == REFERENCE else f"{sum(converged_flags(recs, mode))} of {len(recs)}"
            row += [fmt(acc), fmt(f2, 3), conv]
        rows.append(row)
    return Table(
        "table6",
        "Comparison of point-cloud alignment methods on LM-O and YCB-V.",
        [
            "Alignment method",
            "LM-O Accuracy (mm) ↓",
            "LM-O F@2% ↑",
            "LM-O Converged sets ↑",
            "YCB-V Accuracy (mm) ↓",
            "YCB-V F@2% ↑",
            "YCB-V Converged sets ↑",
        ],
        rows,
        [
            "Medians over view sets; ground-truth masks, three views. The rows are not a ranking of the methods: accuracy and F@2% "
            "reward a placement for fitting the errors of the cloud, completeness (not shown) is measured against the part of the scan "
            "visible from the true camera poses."
        ],
    )
