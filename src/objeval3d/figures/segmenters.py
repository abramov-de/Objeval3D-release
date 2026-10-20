"""Figure 4: what each segmenter returns on the same frame of LM-O.

A table of IoU says how far a mask is from the ground truth, but not what kind of miss it is, and here the kinds differ in what they do
to the measurement. A mask that is empty gives a view with no points; a mask on *another object* gives a cloud of the wrong thing, which
no alignment can repair; a mask that is merely a little too large or too small shifts accuracy and completeness slightly. The comparison
of the learned head with the language segmenter (SAM 3), the CAD-model segmenter (CNOS) and ZebraPose rests on exactly this difference,
so the reader is shown it.

Every row is one frame and one target object. Every segmenter is run on the same frame, the ground-truth outline is yellow in every
panel, and the number under a panel is the IoU with that ground truth at the processing resolution, i.e. the number the tables are
built from. Nothing is recomputed for the picture except that IoU, from the same mask files the tables read.

The examples are given explicitly (--examples) and are illustrations of kinds of miss, not a sample: how often each segmenter misses is
in the tables of the paper, not in this figure. When a frame occurs in several view sets the first one in the manifest is used: the head
sees the whole set of three views, so its mask for the same frame differs a little from set to set, and picking the best-looking set
would flatter it.

    python -m objeval3d.figures.segmenters --examples cat:257,glue:257,duck:257,holepuncher:865
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from objeval3d.figures.style import FULL, OKABE
from objeval3d.figures.style import style as _style
from objeval3d.paths import bop_root

SOURCES = (
    ("learned head (ours)", "lmo_head"),
    ("SAM 3 (language)", "lmo_sam3_tuned"),
    ("CNOS (CAD model)", "lmo_cnos"),
    ("ZebraPose (trained on BOP)", "lmo_zebrapose"),
)

MASK_COL = OKABE["vermillion"]
GT_COL = "#FFE600"
FRAME_HW = (480, 640)  # BOP LM-O frames


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = float((pred | gt).sum())
    return float((pred & gt).sum()) / union if union > 0 else float("nan")


def _resize_mask(m: np.ndarray, hw) -> np.ndarray:
    return cv2.resize(m.astype(np.uint8) * 255, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 127


def _find_set(sets, obj_name: str, im_id: int):
    for s in sets:
        if s["obj_name"] == obj_name and im_id in s["im_ids"]:
            return s, s["im_ids"].index(im_id)
    raise SystemExit(f"no view set of '{obj_name}' contains frame {im_id}")


def _window(gt: np.ndarray, hw, factor=2.1, min_side=130) -> tuple[int, int, int]:
    """Square crop around the target, kept inside the frame: (x0, y0, side)."""
    ys, xs = np.nonzero(gt)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = int(round(max(factor * max(np.ptp(xs), np.ptp(ys)), min_side)))
    side = min(side, hw[0])
    x0 = int(np.clip(cx - side / 2, 0, hw[1] - side))
    y0 = int(np.clip(cy - side / 2, 0, hw[0] - side))
    return x0, y0, side


def _outline(ax, mask: np.ndarray, color, lw, x0=0, y0=0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        c = c[:, 0, :].astype(float)
        ax.plot(c[:, 0] - x0, c[:, 1] - y0, color=color, lw=lw, solid_joinstyle="round", zorder=4)


def _tint(rgb: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    out = rgb.astype(float).copy()
    col = np.array(matplotlib.colors.to_rgb(MASK_COL)) * 255
    out[mask] = (1 - alpha) * out[mask] + alpha * col
    return out.astype(np.uint8)


def _box(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.6)


def _tag(ax, text, loc="tl", size=7.0):
    """A small label on a dark ground. Top left carries the IoU, bottom centre the
    note on what kind of miss it is; neither covers the middle of the panel,
    where the target is."""
    x, y, ha, va = {"tl": (0.04, 0.96, "left", "top"), "bc": (0.5, 0.04, "center", "bottom")}[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        color="white",
        zorder=6,
        linespacing=1.1,
        bbox=dict(boxstyle="square,pad=0.15", fc="#222222", ec="none", alpha=0.78),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("configs/manifests/lmo_v3.json"))
    ap.add_argument("--runs-root", type=Path, default=Path("workspace/runs/lmo"))
    ap.add_argument("--masks-root", type=Path, default=Path("workspace/masks"))
    ap.add_argument("--bop-root", type=Path, default=bop_root("lmo"))
    ap.add_argument(
        "--examples",
        default="cat:257,glue:257,duck:257,holepuncher:865",
        help="comma-separated object:frame pairs, in row order",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("workspace/figures"))
    ap.add_argument("--stem", default="fig4_segmenters")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets = json.loads(args.manifest.read_text())["sets"]
    examples = [(e.split(":")[0], int(e.split(":")[1])) for e in args.examples.split(",") if e.strip()]

    rows = []
    for obj, im_id in examples:
        vs, i = _find_set(sets, obj, im_id)
        rgb_path = args.bop_root / vs["split"] / f"{vs['scene_id']:06d}" / "rgb" / f"{im_id:06d}.png"
        if not rgb_path.exists():
            raise SystemExit(
                f"frame not found: {rgb_path} (the BOP test images of "
                f"scene {vs['scene_id']} are needed for this figure)"
            )
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        # ground truth at the processing resolution, as stored by the GT run
        gt_proc = np.load(args.runs_root / f"{vs['set_id']}__gt_full" / "pred.npz")["gt_mask"][i]
        proc_hw = gt_proc.shape
        gt = _resize_mask(gt_proc, rgb.shape[:2])
        preds = []
        for label, d in SOURCES:
            raw = cv2.imread(str(args.masks_root / d / f"{vs['set_id']}_{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise SystemExit(f"missing mask: {d}/{vs['set_id']}_{i:02d}.png")
            m_proc = _resize_mask(raw > 127, proc_hw)
            m_draw = _resize_mask(raw > 127, rgb.shape[:2])
            preds.append(dict(label=label, mask=m_draw, n_px=int(m_proc.sum()), iou=_iou(m_proc, gt_proc)))
        rows.append(
            dict(obj=obj, im_id=im_id, set_id=vs["set_id"], rgb=rgb, gt=gt, win=_window(gt, rgb.shape[:2]), preds=preds)
        )
        print(
            f"{obj:9s} frame {im_id}  {vs['set_id']}  "
            + "  ".join(f"{p['label'].split()[0]}: IoU {p['iou']:.3f} ({p['n_px']} px)" for p in preds)
        )

    # ------------------------------------------------------------------ layout
    gap, lab_w = 0.045, 0.30
    # one row is as tall as a panel is wide; the context column keeps the 4:3 shape of the frame, so
    # lab_w + 4/3 h + 4 h + 5 gaps = FULL and the figure fills the text width
    row_h = (FULL - lab_w - 5 * gap) / (4 / 3 + len(SOURCES))
    ctx_w = row_h * 4 / 3
    head_h = 0.34
    n = len(rows)
    foot_h = 0.30
    fig_h = head_h + n * row_h + (n - 1) * gap + foot_h
    span = lab_w + ctx_w + len(SOURCES) * row_h + (len(SOURCES) + 0) * gap
    x_left = (FULL - span) / 2  # equals zero up to the gaps
    _style()
    fig = plt.figure(figsize=(FULL, fig_h))

    def add(x, y, w, h):
        return fig.add_axes([x / FULL, y / fig_h, w / FULL, h / fig_h])

    # column headers
    hx = x_left + lab_w + ctx_w + gap
    fig.text(
        (x_left + lab_w + ctx_w / 2) / FULL,
        1 - 0.05 / fig_h,
        "the frame",
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="bold",
    )
    for k, (label, _) in enumerate(SOURCES):
        fig.text(
            (hx + k * (row_h + gap) + row_h / 2) / FULL,
            1 - 0.05 / fig_h,
            label.replace(" (", "\n("),
            ha="center",
            va="top",
            fontsize=7.4,
            fontweight="bold",
            linespacing=1.05,
        )

    for r, row in enumerate(rows):
        y = fig_h - head_h - (r + 1) * row_h - r * gap
        x0, y0, side = row["win"]
        h_img, w_img = row["rgb"].shape[:2]

        # row label
        fig.text(
            (x_left + lab_w * 0.45) / FULL,
            (y + row_h / 2) / fig_h,
            f"“{row['obj']}”",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )

        # context: the whole frame, the target outlined, the crop window boxed
        ax = add(x_left + lab_w, y, ctx_w, row_h)
        ax.imshow(row["rgb"], aspect="auto")
        _outline(ax, row["gt"], GT_COL, 0.9)
        ax.add_patch(plt.Rectangle((x0, y0), side, side, fill=False, ec="white", lw=0.8, zorder=5))
        ax.set_xlim(0, w_img)
        ax.set_ylim(h_img, 0)
        _box(ax)

        for k, p in enumerate(row["preds"]):
            ax = add(hx + k * (row_h + gap), y, row_h, row_h)
            crop = row["rgb"][y0 : y0 + side, x0 : x0 + side]
            m = p["mask"][y0 : y0 + side, x0 : x0 + side]
            ax.imshow(_tint(crop, m), interpolation="lanczos")
            if m.any():
                _outline(ax, m, MASK_COL, 0.7)
            _outline(ax, row["gt"][y0 : y0 + side, x0 : x0 + side], GT_COL, 0.9)
            ax.set_xlim(-0.5, side - 0.5)
            ax.set_ylim(side - 0.5, -0.5)
            _box(ax)

            if p["n_px"] == 0:
                _tag(ax, "IoU 0.00")
                _tag(ax, "no mask returned", loc="bc")
            else:
                _tag(ax, f"IoU {p['iou']:.2f}")
                if p["iou"] < 0.05:
                    # the mask lies on something else: say so, and where
                    ys, xs = np.nonzero(p["mask"])
                    dx, dy = xs.mean() - (x0 + side / 2), ys.mean() - (y0 + side / 2)
                    d = np.hypot(dx, dy)
                    ux, uy = dx / d, dy / d
                    c = side / 2
                    ax.annotate(
                        "",
                        xy=(c + ux * 0.46 * side, c + uy * 0.46 * side),
                        xytext=(c + ux * 0.18 * side, c + uy * 0.18 * side),
                        arrowprops=dict(arrowstyle="-|>", color=MASK_COL, lw=1.6, mutation_scale=9),
                        zorder=7,
                    )
                    _tag(ax, "mask on another object\n(outside this window)", loc="bc", size=6.2)

    fig.text(
        0.5,
        0.03 / fig_h,
        "yellow outline: the ground-truth mask of the target;  red tint: the mask "
        "the segmenter returns;  IoU: with the ground truth on this frame\n"
        "white frame in the left column: the window shown in the other columns",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
        linespacing=1.25,
    )
    out_dir = args.out_dir
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0, "savefig.dpi": 300}):
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{args.stem}.{ext}")
    plt.close(fig)
    print(f"figure -> {out_dir / args.stem}.pdf")


if __name__ == "__main__":
    main()
