"""Tables A.1-A.3: the number of views, three against six on the same sets, and completeness against the visible part of the scan.

Why the number of views is a fixed design choice of the paper (three): more views raise the *coverage* of the object, not the
quality of the reconstruction in a single view, so the bias of the free fit does not shrink with them (A.2), and the mean distance
is a poor summary because it grows with the size of the cloud (A.1).
"""

from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np

from objeval3d.analysis.common import accuracy, chamfer, fscore, load_sets
from objeval3d.analysis.table import Table, fmt
from objeval3d.modes import FREE_FIT, REFERENCE

A1_FILES = {2: "lmo_head_v2", 4: "lmo_head_v4", 6: "lmo_head_v6"}  # the head's masks, LM-O
A2_FILES = {
    "LM-O": ("lmo_gt_full", "lmo_gt_full_v6n"),
    "YCB-V": ("ycbv_gt", "ycbv_gt_v6n"),
}  # ground-truth masks, same sets
A3_FILES = {"LM-O": "lmo_gt_full", "YCB-V": "ycbv_gt"}


def views_per_set(results: Path) -> Table:
    """Table A.1: more views, same masks: F@2%, how inconsistent the views of a set are, and three summaries of the distance.

    F@2% and the three distances are means over the view sets of the per-set value (mean, median and 95th percentile of the distances
    of the points of a set). Inconsistency is the mean over view sets of the largest relative disagreement between the views of a set.
    """
    rows = []
    for n, name in A1_FILES.items():
        recs = list(load_sets(results, name).values())

        def acc(k):
            return st.mean(r["alignments"][REFERENCE]["accuracy"][k] for r in recs)

        incons = [
            r["multiview_consistency"]["max_rel"]
            for r in recs
            if (r.get("multiview_consistency") or {}).get("max_rel") is not None
        ]
        rows.append(
            [
                str(n),
                fmt(st.mean(fscore(r) for r in recs), 3),
                fmt(st.mean(incons), 3),
                fmt(acc("mean"), 1),
                fmt(acc("median"), 1),
                f"{acc('p95'):.0f}",
            ]
        )
    return Table(
        "tableA1",
        "Effect of the number of views per set on reconstruction quality on LM-O.",
        [
            "Views per set",
            "F@2% ↑",
            "View inconsistency ↓",
            "Accuracy mean (mm) ↓",
            "Accuracy median (mm) ↓",
            "Accuracy p95 (mm) ↓",
        ],
        rows,
    )


def three_vs_six(results: Path) -> Table:
    """Table A.2: the same view sets with three more frames; medians over the sets present in both arms."""
    cells = []
    for ds, (a, b) in A2_FILES.items():
        ra, rb = (load_sets(results, f, modes=(REFERENCE, FREE_FIT)) for f in (a, b))
        ids = sorted(set(ra) & set(rb))
        for recs in (ra, rb):
            sub = [recs[s] for s in ids]
            within20 = [x < 20 for r in sub for x in r["oracle_diagnostics"]["pose_disagreement_deg"]]
            cells.append(
                {
                    "points": f"{st.median(r['n_recon_points'] for r in sub):.0f}",
                    "views_ok": f"{100 * np.mean(within20):.0f}",
                    "acc": fmt(st.median(accuracy(r) for r in sub)),
                    "f2": fmt(st.median(fscore(r) for r in sub), 3),
                    "ratio": "×" + fmt(st.median(accuracy(r) / accuracy(r, FREE_FIT) for r in sub)),
                    "below": f"{100 * np.mean([chamfer(r, FREE_FIT) < chamfer(r) for r in sub]):.0f}",
                }
            )
    labels = [
        ("Number of points in the cloud", "points"),
        ("Views within 20° of the best view (%)", "views_ok"),
        ("Accuracy at the ground-truth pose (mm) ↓", "acc"),
        ("F@2% at the ground-truth pose ↑", "f2"),
        ("Accuracy ratio: ground-truth pose / free fit", "ratio"),
        ("Sets with lower Chamfer distance after free-fit alignment (%)", "below"),
    ]
    return Table(
        "tableA2",
        "Effect of increasing the number of views from three to six on the same view sets.",
        ["Quantity", "LM-O, 3 views", "LM-O, 6 views", "YCB-V, 3 views", "YCB-V, 6 views"],
        [[label] + [c[k] for c in cells] for label, k in labels],
    )


def completeness_visible_vs_full(results: Path) -> Table:
    """Table A.3: completeness against the visible part of the scan and against the whole scan (reference mode, ground-truth masks)."""
    rows = []
    for ds, name in A3_FILES.items():
        recs = list(load_sets(results, name).values())
        vis = st.mean(r["alignments"][REFERENCE]["completeness"]["median"] for r in recs)
        full = st.mean(r["alignments"][REFERENCE]["completeness_full"]["median"] for r in recs)
        rows.append(
            [
                ds,
                fmt(vis),
                fmt(full),
                f"{100 * (full - vis) / vis:.0f}",
                fmt(st.mean(r["gt_visible_fraction"] for r in recs)),
            ]
        )
    return Table(
        "tableA3",
        "Completeness evaluated against the visible part and the full scan.",
        [
            "Dataset",
            "Completeness against visible scan (mm) ↓",
            "Completeness against full scan (mm) ↓",
            "Increase in error (%) ↓",
            "Visible scan fraction",
        ],
        rows,
        ["Means over view sets of the per-set median distance from the scan to the cloud."],
    )
