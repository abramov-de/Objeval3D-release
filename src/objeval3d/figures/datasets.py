"""Figure 1: what "a few per cent of the frame" looks like.

The claim that the object occupies about one per cent of an LM-O frame and a few per cent of a YCB-V one carries the whole segmentation
stage, and a number alone does not convey it. Each panel is one input view with the target outlined and magnified, so the reader sees
the clutter the mask has to cut the object out of.

Frames come from the runs themselves (`overlays/overlay_00.jpg` of a run folder written by `objeval3d.recon.reconstruct`), not from a
separate rendering, so what is shown is the data the protocol actually consumed. The results files name the run folders
(`run_dir`), so run this where the runs are, with `--results-dir` pointing at the result files of those runs.

    python -m objeval3d.figures.datasets --results-dir workspace/results --out-dir workspace/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from objeval3d.figures.style import FULL, OKABE
from objeval3d.figures.style import save as _save
from objeval3d.figures.style import style as _style

# Two objects per dataset, bracketing the mean rather than flattering it: LM-O
# averages 0.97 % of the frame and YCB-V 3.60 %, so each pair sits either side.
PICKS = [
    ("LM-O", "lmo_gt_full.json", "cat"),
    ("LM-O", "lmo_gt_full.json", "driller"),
    ("YCB-V", "ycbv_gt.json", "tuna_fish_can"),
    ("YCB-V", "ycbv_gt.json", "mustard_bottle"),
]
ZOOM = 0.34  # inset side, as a fraction of the frame's short side
BOP_FRAME = (640, 480)  # resolution the stored intrinsics refer to
PAD = 1.6  # object bounding box is padded by this factor before cropping


def pick_run(results_dir: Path, results_file: str, obj: str) -> tuple[dict, Path]:
    """The view set of `obj` whose first view has the median object area.

    The median rather than the largest, because the figure is making a claim
    about the typical frame and a hand-picked extreme would undercut it.
    """
    rows = []
    for r in json.loads((results_dir / results_file).read_text())["results"]:
        if r.get("status") != "ok" or r["obj_name"] != obj:
            continue
        overlay = Path(r["run_dir"]) / "overlays" / "overlay_00.jpg"
        if overlay.exists():
            rows.append((r["mask_metrics"]["per_view"][0]["gt_px"], r, overlay))
    if not rows:
        raise SystemExit(f"no local overlay for {obj} in {results_file}")
    rows.sort(key=lambda t: t[0])
    return rows[len(rows) // 2][1], rows[len(rows) // 2][2]


def object_box(
    run_dir: Path, overlay_size: tuple[int, int], diameter_mm: float, pad: float
) -> tuple[float, float, float]:
    """Centre and side of the crop box, projected from the stored object pose.

    Finding the target in the overlay by the colour of its outline is not
    reliable: the LM-O scenes contain saturated yellow objects of their own, and
    the largest yellow component is sometimes one of those. The pose is exact and
    already in `meta.json`, so the object's image centre is `K t_m2c` in
    homogeneous coordinates, and its apparent radius is `f D / (2 z)`. Poses are
    stored for the full-resolution BOP frame, hence the rescaling to the saved
    overlay.
    """
    vs = json.loads((run_dir / "meta.json").read_text())["view_set"]
    K = np.asarray(vs["K"][0], dtype=float)
    t = np.asarray(vs["t_m2c"][0], dtype=float).reshape(3)
    p = K @ t
    s = overlay_size[0] / BOP_FRAME[0]
    cx, cy = p[0] / p[2] * s, p[1] / p[2] * s
    f = 0.5 * (K[0, 0] + K[1, 1])
    side = max(pad * f * diameter_mm / t[2] * s, 26.0)
    return cx, cy, side


def draw(ax, overlay: Path, run_dir: Path, title: str, frac: float, diameter_mm: float) -> None:
    img = np.asarray(Image.open(overlay).convert("RGB"))
    h, w = img.shape[:2]
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#BBBBBB")

    cx, cy, side = object_box(run_dir, (w, h), diameter_mm, PAD)
    x0 = float(np.clip(cx - side / 2, 0, w - side))
    y0 = float(np.clip(cy - side / 2, 0, h - side))
    ax.add_patch(Rectangle((x0, y0), side, side, fill=False, lw=0.9, edgecolor=OKABE["vermillion"], zorder=3))

    k = int(min(h, w) * ZOOM)
    crop = Image.fromarray(img[int(y0) : int(y0 + side), int(x0) : int(x0 + side)])
    crop = crop.resize((k, k), Image.LANCZOS)
    ox = 2 if cx > w / 2 else w - k - 2  # corner furthest from the object
    oy = h - k - 2
    ax.imshow(np.asarray(crop), extent=(ox, ox + k, oy + k, oy), zorder=4)
    ax.add_patch(Rectangle((ox, oy), k, k, fill=False, lw=0.9, edgecolor=OKABE["vermillion"], zorder=5))

    ax.set_title(f"{title}\n{frac:.2f} % of the frame", fontsize=7.0, pad=4, linespacing=1.35)


def render(results_dir: Path, out_dir: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.95), gridspec_kw=dict(wspace=0.06))
    for ax, (ds, f, obj) in zip(axes, PICKS):
        rec, overlay = pick_run(results_dir, f, obj)
        w, h = Image.open(overlay).size
        frac = 100 * rec["mask_metrics"]["per_view"][0]["gt_px"] / (w * h)
        draw(ax, overlay, Path(rec["run_dir"]), f"{ds}, {obj.replace('_', ' ')}", frac, rec["diameter_mm"])
    _save(fig, out_dir, "fig1_datasets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("workspace/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("workspace/figures"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    render(a.results_dir, a.out_dir)
    print(f"figure -> {a.out_dir}/fig1_datasets.pdf")


if __name__ == "__main__":
    main()
