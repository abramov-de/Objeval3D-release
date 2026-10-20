"""One style for every figure of the paper: single column 3.5 in, double column 7.16 in, serif face, hairline rules, no chartjunk, and a
palette (Okabe-Ito) that survives greyscale printing and the common colour deficiencies - the figures are read by reviewers on paper as
often as on screen."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#767676",
    "light": "#D9D9D9",
}
COL = 3.5
FULL = 7.16


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, out_dir, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def place_labels(ax, xs, ys, names, fontsize=5.4):
    """Greedy label placement: eight candidate offsets, keep the first that clears the other points and the labels already placed.

    Thirteen objects in one panel collide often enough that hand-tuning would not survive the next run, and a label sitting on top of
    another is worse than a leader line: the reader cannot tell which point is being named.
    """
    ax.figure.canvas.draw()
    pts = np.column_stack([xs, ys])
    disp = ax.transData.transform(pts)
    cands = [(6, -1.5), (6, 5), (6, -8), (-6, -1.5), (-6, 5), (-6, -8), (0, 8), (0, -11)]
    placed = []
    for i, (name, d) in enumerate(zip(names, disp)):
        best, best_cost = cands[0], np.inf
        for dx, dy in cands:
            w = 3.05 * fontsize * len(name) / 5.4 * 0.62
            h = 1.45 * fontsize
            x0 = d[0] + dx - (w if dx < 0 else 0)
            y0 = d[1] + dy - h / 2
            box = (x0, y0, x0 + w, y0 + h)
            cost = 0.0
            for j, o in enumerate(disp):
                if j == i:
                    continue
                if box[0] - 3 < o[0] < box[2] + 3 and box[1] - 3 < o[1] < box[3] + 3:
                    cost += 12.0
            for b in placed:
                ox = min(box[2], b[2]) - max(box[0], b[0])
                oy = min(box[3], b[3]) - max(box[1], b[1])
                if ox > 0 and oy > 0:
                    cost += 6.0 + ox * oy / 400.0
            cost += 0.55 * (abs(dx) + abs(dy)) / 10.0
            if dx < 0:
                cost += 0.4
            if cost < best_cost:
                best, best_cost, best_box = (dx, dy), cost, box
        placed.append(best_box)
        ax.annotate(
            name,
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=best,
            fontsize=fontsize,
            color="#333333",
            zorder=4,
            ha="right" if best[0] < 0 else "left",
            va="center",
        )
