"""Figures 2 and 3: what the protocol actually sees at each stage.

Numbers carry the argument in the paper, but a reader who has never run a feed-forward reconstruction on a cluttered tabletop cannot
picture what a "view set", an "object cloud" or a "mask of the wrong object" looks like, and a table cannot supply that. These figures are
the picture, and nothing more is claimed by them than is claimed in the text.

The object cloud is never drawn on its own. A bare scatter of a few thousand points is unreadable at figure size: it shows neither which
object it is nor whether it is any good. Every cloud is therefore drawn *on* the scanned reference of the same object, uniformly sampled
and greyed, in the frame the protocol aligns to. The grey body supplies the shape, the coloured points supply the reconstruction, and the
reader can see for himself how the two relate. Two viewpoints are shown because one orthographic projection of a partial surface is
ambiguous.

Needs the run folders of one view set under two mask sources (`<set>__gt_full`, `<set>__head_full`, with `--save-aligned` so that
`recon_aligned_oracle_raw.ply` exists) and the result files of those runs.

    python -m objeval3d.figures.qualitative --set-id lmo_s000002_o05_v03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from objeval3d.data.bop import load_mesh, model_path, read_ply, sample_mesh_surface
from objeval3d.figures.style import FULL, OKABE
from objeval3d.figures.style import style as _style
from objeval3d.paths import bop_root


def _save_exact(fig, out_dir: Path, stem: str) -> None:
    r"""Save at the declared canvas size rather than trimmed to the ink.

    These two figures are laid out in inches against the text width, so the
    panels and the type are already the size they should print at. A tight
    bounding box would crop to whatever the content happens to span and
    \includegraphics would then scale the whole thing back up, changing every
    type size on the page.
    """
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


SCAN_GREY = "#DBDBDB"
N_SCAN_DRAWN = 30000  # sampled once, drawn identically in every panel
VIEW_AR = 640 / 480  # BOP LM-O frames

# Azimuth 345 looks along the mean direction of the three cameras, so the front
# panel shows the surface the views could actually see; 255 is a quarter turn
# from it, where the spout and the handle separate. Elevation is a little below
# the cameras', which sit high over the table.
CLOUD_VIEWS = ((345.0, 22.0), (255.0, 22.0))


def _rot(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = np.radians(az_deg), np.radians(el_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def _project(pts: np.ndarray, centre: np.ndarray, az: float, el: float):
    """Orthographic projection -> (screen x, screen y, depth into the page)."""
    p = (pts - centre) @ _rot(az, el).T
    return p[:, 0], p[:, 2], p[:, 1]


def draw_cloud_on_scan(ax, recon, scan_drawn, centre, az, el, margin=1.28) -> float:
    """One orthographic panel: the scanned reference in grey, the cloud over it.

    The window is set by the reference and by nothing else, so that the same
    object is framed identically in every panel of the figure and the two rows
    can be compared point for point. The share of reconstructed points that
    falls outside that window is returned and reported, since a window chosen
    this way is free to cut off a distant flyer.
    """
    gx, gy, _ = _project(scan_drawn, centre, az, el)
    ax_w, ax_h = ax.get_position().size * ax.figure.get_size_inches()

    half_y = margin * np.ptp(gy) / 2
    half_x = half_y * (ax_w / ax_h)
    cx, cy = (gx.max() + gx.min()) / 2, (gy.max() + gy.min()) / 2
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)

    ax.scatter(gx, gy, s=0.32, c=SCAN_GREY, marker=".", linewidths=0, zorder=1, rasterized=True)

    qx, qy, qz = _project(recon, centre, az, el)
    order = np.argsort(-qz)
    ax.scatter(qx[order], qy[order], s=0.95, c=OKABE["blue"], marker=".", linewidths=0, zorder=3, rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.6)

    outside = (qx < cx - half_x) | (qx > cx + half_x) | (qy < cy - half_y) | (qy > cy + half_y)
    return float(outside.mean())


def _scale_bar(ax, length_mm=50.0, label="50 mm"):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xa = x1 - 0.06 * (x1 - x0) - length_mm
    ya = y0 + 0.055 * (y1 - y0)
    ax.plot([xa, xa + length_mm], [ya, ya], color="#555555", lw=1.0, solid_capstyle="butt", zorder=5)
    ax.text(
        xa + length_mm / 2,
        ya + 0.022 * (y1 - y0),
        label,
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#555555",
        zorder=5,
        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85),
    )


def _load_metrics(results_dir: Path, tag: str, set_id: str) -> dict:
    """The numbers printed in the figure come from the result files, not from a
    recomputation here, so that the figure cannot drift away from the tables."""
    recs = json.loads((results_dir / f"lmo_{tag}.json").read_text())["results"]
    r = next(x for x in recs if x["set_id"] == set_id)
    a = r["alignments"]["oracle_raw"]
    return {
        "f2": a["fscore"]["tau_2pct"]["fscore"],
        "median_mm": a["accuracy"]["median"],
        "iou": r["mask_metrics"]["aggregate"]["iou"],
        "n": r["n_recon_points"],
    }


def fig_segmentation(run_gt: Path, run_head: Path, out_dir: Path, obj_name: str, metrics: dict, stem: str) -> None:
    """The same three views under the two mask sources."""
    # Six frames stretched across the whole text width would make the figure
    # four inches high; they are drawn at a size that still reads and centred.
    label_w, row_span, gap_x, gap_row = 0.38, 5.75, 0.05, 0.05
    top_pad, bot_pad = 0.38, 0.30
    view_w = (row_span - 2 * gap_x) / 3
    view_h = view_w / VIEW_AR
    fig_h = top_pad + 2 * view_h + gap_row + bot_pad
    fig = plt.figure(figsize=(FULL, fig_h))
    x0 = (FULL - label_w - row_span) / 2

    rows = ((run_gt, "ground-truth masks", "gt_full"), (run_head, "learned-head masks", "head_full"))
    for row, (run, label, tag) in enumerate(rows):
        y = fig_h - top_pad - (row + 1) * view_h - row * gap_row
        for col, ov in enumerate(sorted((run / "overlays").glob("*.jpg"))[:3]):
            ax = fig.add_axes(
                [(x0 + label_w + col * (view_w + gap_x)) / FULL, y / fig_h, view_w / FULL, view_h / fig_h]
            )
            ax.imshow(mpimg.imread(ov))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("#CCCCCC")
                sp.set_linewidth(0.6)
            if row == 0:
                ax.set_title(f"view {col + 1}", fontsize=7.4, pad=3)
        fig.text(
            (x0 + label_w - 0.11) / FULL,
            (y + view_h / 2) / fig_h,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7.6,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.045,
        "the target is outlined in yellow in both rows",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    fig.text(
        0.5,
        1 - 0.11 / fig_h,
        f"LM-O, object \u201c{obj_name}\u201d: one view set, segmented two ways",
        ha="center",
        va="top",
        fontsize=8.2,
    )
    _save_exact(fig, out_dir, stem)


def fig_cloud(
    run_gt: Path, run_head: Path, out_dir: Path, obj_name: str, scan: np.ndarray, metrics: dict, stem: str
) -> None:
    """The cleaned cloud of that same view set, laid on the scan it is measured
    against, once per mask source and from two directions each."""
    # The panels do not fill the text width: four tall silhouettes stretched
    # across it would make the figure half a column high for no gain in
    # legibility, so they are drawn at a readable size and centred.
    panel_span, gap_in, gap_group = 5.55, 0.05, 0.40
    top_pad, bot_pad = 0.44, 0.48

    centre = scan.mean(axis=0)
    rng = np.random.default_rng(0)
    drawn = scan[rng.choice(len(scan), N_SCAN_DRAWN, replace=False)]
    aspects = []
    for az, el in CLOUD_VIEWS:
        gx, gy, _ = _project(drawn, centre, az, el)
        aspects.append(np.ptp(gx) / np.ptp(gy))

    panel_h = (panel_span - gap_group - 2 * gap_in) / (2 * sum(aspects))
    fig_h = top_pad + panel_h + bot_pad
    fig = plt.figure(figsize=(FULL, fig_h))

    y = bot_pad
    x = (FULL - panel_span) / 2
    outside_all = []
    groups = ((run_gt, "ground-truth masks", "gt_full"), (run_head, "learned-head masks", "head_full"))
    for run, label, tag in groups:
        recon, _ = read_ply(run / "recon_aligned_oracle_raw.ply")
        x0 = x
        for k, ((az, el), a) in enumerate(zip(CLOUD_VIEWS, aspects)):
            w = panel_h * a
            ax = fig.add_axes([x / FULL, y / fig_h, w / FULL, panel_h / fig_h])
            outside_all.append(draw_cloud_on_scan(ax, recon, drawn, centre, az, el))
            if k == len(CLOUD_VIEWS) - 1:
                _scale_bar(ax)
            x += w + gap_in
        x1 = x - gap_in
        m = metrics[tag]
        fig.text(
            (x0 + x1) / 2 / FULL,
            (y + panel_h + 0.06) / fig_h,
            label,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )
        fig.text(
            (x0 + x1) / 2 / FULL,
            (y - 0.10) / fig_h,
            f"median distance to the scan {m['median_mm']:.1f} mm        $F@2\\%$ = {m['f2']:.2f}",
            ha="center",
            va="top",
            fontsize=7.2,
        )
        x = x1 + gap_group

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.6,
            markerfacecolor=SCAN_GREY,
            markeredgecolor="none",
            label="the scanned reference, sampled over its surface",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.6,
            markerfacecolor=OKABE["blue"],
            markeredgecolor="none",
            label="the reconstructed object cloud after cleaning",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        frameon=False,
        fontsize=7.0,
        handletextpad=0.35,
        columnspacing=1.8,
        ncol=2,
    )
    fig.text(
        0.5,
        1 - 0.11 / fig_h,
        f"the object cloud of that view set, laid on the scan of \u201c{obj_name}\u201d and seen from two directions",
        ha="center",
        va="top",
        fontsize=8.2,
    )
    _save_exact(fig, out_dir, stem)
    print(f"points outside the window, per panel: {['%.1f%%' % (100 * o) for o in outside_all]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("workspace/runs/lmo"))
    ap.add_argument("--results-dir", type=Path, default=Path("workspace/results"))
    ap.add_argument("--bop-root", type=Path, default=bop_root("lmo"))
    ap.add_argument(
        "--gt-mesh",
        type=Path,
        default=None,
        help="scanned model; by default it is looked up in the dataset by the object id of the view set",
    )
    ap.add_argument("--set-id", default="lmo_s000002_o05_v03")
    ap.add_argument("--n-scan", type=int, default=60000, help="surface samples of the scan, as in the metric itself")
    ap.add_argument("--out-dir", type=Path, default=Path("workspace/figures"))
    ap.add_argument("--stem-segmentation", default="fig2_segmentation")
    ap.add_argument("--stem-cloud", default="fig3_cloud")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_gt = args.runs_root / f"{args.set_id}__gt_full"
    run_head = args.runs_root / f"{args.set_id}__head_full"
    vs = json.loads((run_gt / "meta.json").read_text())["view_set"]
    mesh = args.gt_mesh or model_path(args.bop_root, int(vs["obj_id"]))
    verts, faces = load_mesh(Path(mesh))
    scan = sample_mesh_surface(verts, faces, args.n_scan, seed=0)

    metrics = {t: _load_metrics(args.results_dir, t, args.set_id) for t in ("gt_full", "head_full")}

    _style()
    name = vs["obj_name"].replace("_", " ")
    fig_segmentation(run_gt, run_head, args.out_dir, name, metrics, args.stem_segmentation)
    fig_cloud(run_gt, run_head, args.out_dir, name, scan, metrics, args.stem_cloud)
    print(f"figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
