"""Tables 8-9 and Figure 6: do the conclusions survive a change of the reconstruction model?

The claim is ordinal. Two models must disagree on the absolute value (that is what the protocol reports), but the order in which the
objects come out has to survive the change of model, because every operational statement of the protocol is ordinal: this object is
reconstructed worse than that one.

Four cells: {LM-O, YCB-V} x {ground-truth masks, masks of the head}. Both legs of a cell see identical pixels. The head runs on the
frozen backbone of DA3 and cannot run inside VGGT, so its masks are exported once and replayed to both models
(`--mask-source dir` of `objeval3d.recon.reconstruct`). The confidence percentile is taken inside the mask, because confidence is not
a shared quantity between two models (a percentile over the whole frame cut VGGT's `ape` to 68 points where DA3 kept 1186).

Aggregation: the median over view sets (Tables 5-9), per-object medians for the rank correlation. With 8 and 13 objects the
significance comes from a permutation test of Spearman's rho, exact where 8! orderings are affordable and Monte Carlo (2 000 000
draws, seed 0, the floor of the p-value is 1/2 000 000) otherwise, and from Holm's correction over the four cells.
"""

from __future__ import annotations

import itertools
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from objeval3d.analysis.common import read_json
from objeval3d.analysis.table import Table, fmt, sci
from objeval3d.modes import REFERENCE

# (dataset, mask source) -> (DA3 result file, VGGT result file)
CELLS = {
    ("LM-O", "gt"): ("lmo_gt_da3_gt_maskconf", "lmo_vggt_gt"),
    ("LM-O", "head"): ("lmo_head_maskconf", "lmo_vggt_dir"),
    ("YCB-V", "gt"): ("ycbv_gt_maskconf", "ycbv_vggt_gt"),
    ("YCB-V", "head"): ("ycbv_head_maskconf", "ycbv_vggt_dir"),
}
CELL_ORDER = [("LM-O", "gt"), ("LM-O", "head"), ("YCB-V", "gt"), ("YCB-V", "head")]
MASK_LABEL = {"gt": "Ground-truth masks", "head": "Predicted masks (ours)"}
N_MC, MC_CHUNK = 2_000_000, 250_000


def _load(results: Path, name: str) -> dict[str, dict]:
    """One compact record per scored view set: the quantities the cells are made of."""
    out = {}
    for r in read_json(results / f"{name}.json")["results"]:
        if r.get("status") != "ok" or REFERENCE not in r.get("alignments", {}):
            continue
        a = r["alignments"][REFERENCE]
        f2 = a["fscore"]["tau_2pct"]
        out[r["set_id"]] = {
            "obj": r["obj_name"],
            "F2": f2["fscore"],
            "prec2": f2["precision"],
            "rec2": f2["recall"],
            "acc": a["accuracy"]["median"],
            "comp": a["completeness"]["mean"],
        }
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def permutation_p(a: np.ndarray, b: np.ndarray, n_mc: int = N_MC, chunk: int = MC_CHUNK, seed: int = 0) -> float:
    """Two-sided permutation p-value of Spearman's rho between `a` and `b`; exact for up to 9 objects, Monte Carlo above."""
    n = len(a)
    rho = abs(spearman(a, b))
    ra = rankdata(a)
    x = (ra - ra.mean()) / ra.std()

    def hits(perms: np.ndarray) -> int:
        y = (perms - perms.mean(1, keepdims=True)) / perms.std(1, keepdims=True)
        return int((np.abs(y @ x) / n >= rho - 1e-12).sum())

    if n <= 9:
        perms = np.array(list(itertools.permutations(range(1, n + 1))), dtype=float)
        return hits(perms) / len(perms)
    rng = np.random.default_rng(seed)
    rb = rankdata(b)
    total = 0
    for start in range(0, n_mc, chunk):  # the draws are taken in the same order as one block of n_mc rows would be
        m = min(chunk, n_mc - start)
        total += hits(rb[np.argsort(rng.random((m, n)), axis=1)])
    return max(total / n_mc, 1.0 / n_mc)


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for i, idx in enumerate(order):
        running = max(running, ps[idx] * (len(ps) - i))
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def per_object_median(recs: dict, ids: list[str], key: str) -> tuple[list[str], np.ndarray]:
    objs = sorted({recs[s]["obj"] for s in ids})
    return objs, np.array([np.median([recs[s][key] for s in ids if recs[s]["obj"] == o]) for o in objs])


def cells(results: Path) -> dict:
    """Everything the two tables and the figure need, per cell."""
    out = {}
    for cell in CELL_ORDER:
        fa, fb = CELLS[cell]
        a, b = _load(results, fa), _load(results, fb)
        ids = sorted(set(a) & set(b))
        objs, va = per_object_median(a, ids, "F2")
        _, vb = per_object_median(b, ids, "F2")

        def med(recs, key):
            return st.median(recs[s][key] for s in ids)

        out[cell] = {
            "n_sets": len(ids),
            "n_objects": len(objs),
            "objects": objs,
            "per_object": {"da3": va, "vggt": vb},
            "rho": spearman(va, vb),
            "p": permutation_p(va, vb),
            "da3": {k: med(a, k) for k in ("F2", "prec2", "rec2", "acc", "comp")},
            "vggt": {k: med(b, k) for k in ("F2", "prec2", "rec2", "acc", "comp")},
        }
    for cell, adj in zip(CELL_ORDER, holm([out[c]["p"] for c in CELL_ORDER])):
        out[cell]["p_holm"] = adj
    return out


def rank_agreement(results: Path) -> Table:
    """Table 8: agreement in object rankings between the two reconstruction models."""
    c = cells(results)
    rows = [
        [
            ds,
            MASK_LABEL[src],
            str(c[(ds, src)]["n_sets"]),
            str(c[(ds, src)]["n_objects"]),
            fmt(c[(ds, src)]["rho"], 2, sign=True),
            sci(c[(ds, src)]["p"]),
            sci(c[(ds, src)]["p_holm"]),
        ]
        for ds, src in CELL_ORDER
    ]
    return Table(
        "table8",
        "Agreement in object rankings between the two reconstruction models on LM-O and YCB-V.",
        [
            "Dataset",
            "Mask source",
            "View sets",
            "Objects",
            "Spearman's ρ ↑",
            "Permutation-test p ↓",
            "Holm-adjusted p_H ↓",
        ],
        rows,
        [
            "Per-object median F@2% of DA3-LARGE against VGGT-1B, reference mode, confidence percentile inside the mask, "
            "median over view sets."
        ],
    )


def model_comparison(results: Path) -> Table:
    """Table 9: DA3 and VGGT with the ground-truth masks, in the decomposition into precision and recall (medians over view sets)."""
    c = cells(results)
    rows = []
    for ds in ("LM-O", "YCB-V"):
        for model, name in (("da3", "DA3"), ("vggt", "VGGT")):
            v = c[(ds, "gt")][model]
            rows.append(
                [ds, name, fmt(v["F2"], 3), fmt(v["prec2"], 3), fmt(v["rec2"], 3), fmt(v["acc"]), fmt(v["comp"])]
            )
    return Table(
        "table9",
        "Comparison of DA3 and VGGT reconstruction quality on LM-O and YCB-V using ground-truth masks.",
        [
            "Dataset",
            "Reconstruction model",
            "F@2% ↑",
            "Precision@2% ↑",
            "Recall@2% ↑",
            "Accuracy (mm) ↓",
            "Completeness (mm) ↓",
        ],
        rows,
        ["Reference mode, ground-truth masks, confidence percentile inside the mask, median over view sets."],
    )
