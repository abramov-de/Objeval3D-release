"""Figure 5: the free criterion is displaced from the true placement, and where it puts the cloud instead.

(a) For every view set, the Chamfer distance at the reference placement (the true pose) against the one at the placement the free search
found. Below the line the search fits the scan better than the true placement does; the filled points are the sets where the search did
reach the true placement (rotation under 15 degrees and the centre of the cloud within 0.10 diameters, symmetries of the object
respected) and are below the line even so.
(b) How far from the true placement the minimum sits: rotation gap against the displacement of the cloud, in object diameters.

Everything is read from the result files (`lmo_gt_full`, `ycbv_gt`): the displacement of the cloud needs only its centroid, which the
files carry (`recon_centroid`).

    python -m objeval3d.figures.alignment --results results --out-dir workspace/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from objeval3d.analysis.alignment_bias import FILES
from objeval3d.analysis.common import chamfer, load_sets, results_dir
from objeval3d.analysis.placement import dataset_of, gaps, object_symmetries
from objeval3d.figures.style import FULL, OKABE, save, style
from objeval3d.modes import FREE_FIT, REFERENCE

DS_COLOUR = {"LM-O": OKABE["blue"], "YCB-V": OKABE["vermillion"]}


def placement_data(results: Path) -> dict:
    """Per dataset: the arrays the two panels draw. Each set is drawn at the symmetry element that satisfies both halves of the rule
    when one exists, and otherwise at the element of smallest rotation gap."""
    out = {}
    for ds, name in FILES.items():
        recs = [r for r in load_sets(results, name, modes=(FREE_FIT, REFERENCE)).values() if "recon_centroid" in r]
        rot, cen, conv, ch_a, ch_o = [], [], [], [], []
        for r in recs:
            g = gaps(r, FREE_FIT, object_symmetries(dataset_of(r["set_id"]), int(r["obj_id"])))
            ok = (g[:, 0] < 15.0) & (g[:, 1] < 0.10)
            j = int(np.argmin(np.maximum(g[:, 0] / 15.0, g[:, 1] / 0.10)))
            if not ok.any():
                j = int(np.argmin(g[:, 0]))
            rot.append(float(g[j, 0]))
            cen.append(float(g[j, 1]))
            conv.append(bool(ok.any()))
            ch_a.append(chamfer(r, FREE_FIT))
            ch_o.append(chamfer(r, REFERENCE))
        conv, ch_a, ch_o = np.array(conv), np.array(ch_a), np.array(ch_o)
        out[ds] = {
            "n_placement": len(recs),
            "converged": int(conv.sum()),
            "frac_beats_truth": float((ch_a < ch_o).mean()),
            "beats_truth_converged": float((ch_a[conv] < ch_o[conv]).mean()) if conv.any() else float("nan"),
            "placement": {
                "rot_deg": rot,
                "centroid": cen,
                "converged": conv.tolist(),
                "chamfer_auto": ch_a.tolist(),
                "chamfer_oracle": ch_o.tolist(),
            },
        }
    return out


def fig_alignment(res, out_dir) -> None:
    """Fig. 5 -- the free criterion is displaced, and where it puts the cloud.

    The version with the sets on which the search recovered the reference placement marked carries the argument: if those sets are below
    the line too, the bias is not a by-product of a failed search, and no better optimiser removes it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.30))

    # (a) the criterion prefers a placement that is not the true one, and goes on
    # preferring it on the sets where the search did reach the true one.
    ax = axes[0]
    lo, hi = 1e9, 0.0
    for v in res["auto"].values():
        pl = v.get("placement")
        if not pl:
            continue
        lo = min(lo, min(pl["chamfer_oracle"]), min(pl["chamfer_auto"]))
        hi = max(hi, max(pl["chamfer_oracle"]), max(pl["chamfer_auto"]))
    lo *= 0.7
    hi *= 1.45
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="#FBEEE6", zorder=0)
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.8, color=OKABE["grey"], zorder=2)
    for ds, v in res["auto"].items():
        pl = v.get("placement")
        if not pl:
            continue
        x = np.array(pl["chamfer_oracle"])
        y = np.array(pl["chamfer_auto"])
        c = np.array(pl["converged"], dtype=bool)
        ax.scatter(
            x[~c],
            y[~c],
            s=12,
            facecolor="none",
            edgecolor=DS_COLOUR[ds],
            linewidth=0.6,
            alpha=0.6,
            zorder=3,
            label=f"{ds},  {100 * v['frac_beats_truth']:.0f}% below the line",
        )
        ax.scatter(x[c], y[c], s=22, facecolor=DS_COLOUR[ds], edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Chamfer at the true pose, mm", fontsize=7.2)
    ax.set_ylabel("Chamfer at the pose the free search found, mm", fontsize=7.2)
    filled = ", ".join(
        f"{ds} {100 * v['beats_truth_converged']:.0f}%" for ds, v in res["auto"].items() if "beats_truth_converged" in v
    )
    ax.text(
        0.035,
        0.965,
        "below the line the search fits the scan\n"
        "better than the true placement does\n\n"
        "filled: the search did reach the true pose,\n"
        f"and is below the line even so \u2014 {filled}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        linespacing=1.45,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=6.2, handletextpad=0.3, borderpad=0.2, labelspacing=0.35)
    ax.set_title("(a) the minimum does not sit at the true placement", fontsize=7.8, pad=6)
    ax.grid(True, which="major", color="#EEEEEE", lw=0.4)
    ax.set_axisbelow(True)

    # (b) how far from the true placement the minimum sits.
    ax = axes[1]
    for ds, v in res["auto"].items():
        pl = v.get("placement")
        if not pl:
            continue
        ax.scatter(
            pl["rot_deg"],
            pl["centroid"],
            s=13,
            alpha=0.85,
            facecolor=DS_COLOUR[ds],
            edgecolor="white",
            linewidth=0.4,
            label=f"{ds},  {v['converged']} of {v['n_placement']} reached",
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_ylim(6e-3, 3.0)
    ax.add_patch(
        plt.Rectangle((0, 6e-3), 15, 0.10 - 6e-3, facecolor="#E8F4EC", edgecolor=OKABE["green"], lw=0.7, zorder=1)
    )
    ax.text(
        19,
        0.019,
        "the true placement reached\n(under 15\u00b0 and 0.10 diameters)",
        fontsize=6.1,
        color=OKABE["green"],
        va="center",
        linespacing=1.4,
    )
    ax.set_xlim(-4, 184)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("rotation gap to the true pose, deg", fontsize=7.2)
    ax.set_ylabel("cloud displacement, object diameters", fontsize=7.2)
    ax.set_title("(b) where the free search puts the cloud instead", fontsize=7.8, pad=6)
    ax.legend(loc="lower right", frameon=False, fontsize=6.2, handletextpad=0.3, borderpad=0.2, labelspacing=0.35)
    ax.grid(True, color="#EEEEEE", lw=0.4)
    ax.set_axisbelow(True)

    save(fig, out_dir, "fig5_alignment")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("workspace/figures"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    fig_alignment({"auto": placement_data(results_dir(a.results))}, a.out_dir)
    print(f"figure -> {a.out_dir}/fig5_alignment.pdf")


if __name__ == "__main__":
    main()
