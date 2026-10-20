"""Table G.2: how much a free fit understates the accuracy, as the error of the cloud grows.

Accuracy at the reference placement divided by accuracy after the free fit: 1 means the fit changes nothing, 2 means the fit reports
half of the true distance. Three kinds of clouds, in increasing order of their own error:

  * controls built from the BOP model itself, which sit at the true pose by construction: a copy with independent noise (0.5-2 mm),
    a copy with a smooth bend (1-3 mm); the fit is started at the true pose and should not move (ratio near 1);
  * complete meshes of other reconstruction methods (Burde et al., all images), the fit from an arbitrary start;
  * our own clouds, cut in thirds by the error of their own depth (the best single view's residual in diameters, which depends on
    neither the camera poses nor the search), within each dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from objeval3d.analysis.alignment_bias import FILES, converged_flags
from objeval3d.analysis.benchmark_meshes import column, load_meshes
from objeval3d.analysis.common import accuracy, load_sets, read_json
from objeval3d.analysis.table import Table
from objeval3d.modes import FREE_FIT, REFERENCE


def _span(vals: list[float], fmt: str = "{:.2f}") -> str:
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def _controls(results: Path) -> dict:
    ctl = read_json(results / "burde_controls.json")

    def per_level(kinds, fn):
        out = []
        for lvl in sorted({(r["control"], r["level_mm"]) for r in ctl if r["control"] in kinds}):
            out.append(float(np.median([fn(r) for r in ctl if (r["control"], r["level_mm"]) == lvl])))
        return out

    def ratio(r):
        return r["alignments"]["as_lies"]["accuracy"]["median"] / r["alignments"]["local"]["accuracy"]["median"]

    def shift(r):
        return r["vs_as_lies"]["local"]["placement_centroid_rel_diam"]

    def rot(r):
        return r["vs_as_lies"]["local"]["rotation_deg"]

    return {
        "noise_ratio": per_level(("copy", "noise"), ratio),
        "noise_shift": per_level(("copy", "noise"), shift),
        "bend_ratio": per_level(("bend",), ratio),
        "bend_rot": per_level(("bend",), rot),
    }


def _thirds(results: Path, dataset_file: str) -> list[tuple[float, int, int]]:
    """(median accuracy ratio, converged sets, sets) in the thirds of the sets with the smallest, middle and largest depth error."""
    recs = [r for r in load_sets(results, dataset_file, modes=(FREE_FIT, REFERENCE)).values()]

    def depth(r):
        return min([v for v in r["oracle_diagnostics"]["score_own_per_view_mm"] if v == v] or [np.nan]) / float(
            r["diameter_mm"]
        )

    recs = [r for r in recs if depth(r) == depth(r)]
    order = np.argsort([depth(r) for r in recs])
    out = []
    for idx in np.array_split(order, 3):
        sub = [recs[i] for i in idx]
        out.append(
            (
                float(np.median([accuracy(r) / accuracy(r, FREE_FIT) for r in sub])),
                sum(converged_flags(sub, FREE_FIT)),
                len(sub),
            )
        )
    return out


def understatement_ladder(results: Path) -> Table:
    c = _controls(results)
    mesh = column(load_meshes(results), "all")
    rows = [
        [
            "BOP model copy + independent noise (0.5–2 mm)",
            f"Ground-truth pose recovered; shift <= {100 * max(c['noise_shift']):.1f}% of diameter",
            _span(c["noise_ratio"]),
        ],
        [
            "Smoothly bent model (1–3 mm)",
            f"Rotation: {_span(c['bend_rot'], '{:.1f}')}°; negligible shift",
            _span(c["bend_ratio"]),
        ],
        [
            f"Burde et al., all-image meshes (n={mesh['n']})",
            f"Shift: {mesh['auto']['shift']:.1f}% of diameter; arbitrary initialization",
            f"{mesh['ratio_of_medians']:.2f}",
        ],
    ]
    for ds, name in zip(FILES, FILES.values()):
        for third, label in zip(_thirds(results, name), ("lowest", "middle", "highest")):
            rows.append(
                [
                    f"{ds}, {label} depth-error third" if label == "middle" else f"{ds}, {label} depth-error third",
                    f"Converged: {third[1]}/{third[2]}",
                    f"{third[0]:.2f}",
                ]
            )
    return Table(
        "tableG2",
        "Accuracy understatement after free-fit alignment as a function of cloud quality.",
        ["Cloud type", "Free-fit result", "Accuracy ratio ↑"],
        rows,
        ["Accuracy ratio: accuracy at the reference placement over accuracy after the free fit."],
    )
