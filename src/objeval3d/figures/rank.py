"""Figure 6: the ordering of the objects, DA3-LARGE against VGGT-1B.

The claim of the model-transfer section is ordinal, so the scatter panels carry the identity line in light grey as a *reference* and
never as a fit: the models are not asked to agree on values, only on order, and drawing a regression through those points would assert
something the section does not claim.

Two of the four cells are drawn, the ones with the dataset's own masks, where the two models are fed identical pixels and nothing but
the model differs; the other two keep their row in Table 8. Everything is read from the result files by
`objeval3d.analysis.model_transfer.cells`.

    python -m objeval3d.figures.rank --results results --out-dir workspace/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from objeval3d.analysis.common import results_dir
from objeval3d.analysis.model_transfer import CELL_ORDER, cells
from objeval3d.figures.style import FULL, OKABE, place_labels, save, style

CELL_TITLE = {
    ("LM-O", "gt"): "LM-O, ground-truth masks",
    ("LM-O", "head"): "LM-O, learned-head masks",
    ("YCB-V", "gt"): "YCB-V, ground-truth masks",
    ("YCB-V", "head"): "YCB-V, learned-head masks",
}


def fig_rank(res, out_dir) -> None:
    """Fig. 6 -- the object ordering, model against model.

    Two of the four cells are drawn, the ones with the dataset's own masks, where
    the two models are fed identical pixels and nothing but the model differs.
    Four square panels with the identity line in each ran to six inches of page,
    over half a column, to show the same thing four times; the remaining two
    cells keep their row in Table 8.
    """
    cells = [c for c in CELL_ORDER if c[1] == "gt"]
    fig, axes = plt.subplots(1, len(cells), figsize=(FULL, 3.15))
    for ax, cell in zip(np.atleast_1d(axes).ravel(), cells):
        v = res["cells"][cell]
        po = v["per_object"]
        x, y = np.array(po["da3"]), np.array(po["vggt"])
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = 0.10 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.7, color=OKABE["light"], zorder=1)
        colour = OKABE["blue"] if cell[1] == "gt" else OKABE["vermillion"]
        ax.scatter(x, y, s=26, facecolor=colour, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        place_labels(ax, x, y, [n.replace("_", " ") for n in v["objects"]])
        ax.text(
            0.045,
            0.955,
            rf"$\rho = {v['rho']:+.2f}$" + "\n"
            rf"$p_{{\mathrm{{Holm}}}} = {v['p_holm']:.1e}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=OKABE["light"], lw=0.5),
        )
        ax.set_title(CELL_TITLE[cell] + f"   (n = {v['n_sets']} view sets)", pad=5)
        ax.grid(True, color="#EEEEEE", lw=0.4, zorder=0)
        ax.set_axisbelow(True)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_xlabel("DA3-LARGE,  per-object median F@2%")
    np.atleast_1d(axes).ravel()[0].set_ylabel("VGGT-1B,  per-object median F@2%")
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "fig6_rank_agreement")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("workspace/figures"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    fig_rank({"cells": cells(results_dir(a.results))}, a.out_dir)
    print(f"figure -> {a.out_dir}/fig6_rank_agreement.pdf")


if __name__ == "__main__":
    main()
