"""The four stages of the protocol, drawn for the README.

A schematic, not a result: the clouds are synthetic (a mug seen from above) and nothing here is read from the result files.

    python -m objeval3d.figures.pipeline --out docs/pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from objeval3d.figures.style import OKABE, style

INK = "#1F2A44"
TEXT = "#26292E"
MUTED = "#6B7280"
CARD = "#F6F7F9"
EDGE = "#C7CCD4"
BLUE, ORANGE, GREEN, RED, GREY = OKABE["blue"], OKABE["orange"], OKABE["green"], OKABE["vermillion"], "#80868E"
W, H = 110.0, 70.0  # drawing units; one unit is 0.08 inch


# --- the synthetic object -------------------------------------------------------------------------------------------------------------


def mug(rng: np.random.Generator):
    """Points and outward normals on a mug: side wall, two discs, a handle."""
    n_side, n_disc, n_handle = 2600, 520, 560
    th, z = rng.uniform(0, 2 * np.pi, n_side), rng.uniform(0, 1.5, n_side)
    parts = [np.c_[np.cos(th), np.sin(th), z]]
    normals = [np.c_[np.cos(th), np.sin(th), np.zeros(n_side)]]
    for zz, nz in ((1.5, 1.0), (0.0, -1.0)):
        r, a = np.sqrt(rng.uniform(0, 1, n_disc)), rng.uniform(0, 2 * np.pi, n_disc)
        parts.append(np.c_[r * np.cos(a), r * np.sin(a), np.full(n_disc, zz)])
        normals.append(np.tile([0.0, 0.0, nz], (n_disc, 1)))
    t, phi = rng.uniform(-np.pi / 2, np.pi / 2, n_handle), rng.uniform(0, 2 * np.pi, n_handle)
    radial = np.c_[np.cos(t), np.zeros(n_handle), np.sin(t)]
    off = 0.14 * (np.cos(phi)[:, None] * radial + np.sin(phi)[:, None] * np.array([0.0, 1.0, 0.0]))
    parts.append(np.c_[1.0 + 0.5 * np.cos(t), np.zeros(n_handle), 0.75 + 0.5 * np.sin(t)] + off)
    normals.append(off / 0.14)
    p, n = np.vstack(parts), np.vstack(normals)
    yaw = np.deg2rad(-32)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return p @ rz.T, n @ rz.T


class Camera:
    """Orthographic view from above: image coordinates, depth towards the camera, and the cosine between normal and view direction."""

    def __init__(self, p, n, elev=34.0):
        e = np.deg2rad(elev)
        cam, up = np.array([0.0, -np.cos(e), np.sin(e)]), np.array([0.0, np.sin(e), np.cos(e)])
        self.uv = np.c_[p[:, 0], p @ up]
        self.depth = p @ cam
        self.cos = n @ cam
        self.lo, self.hi = self.uv.min(0), self.uv.max(0)

    def place(self, uv, cx, cy, width):
        """Image coordinates into drawing units, with one scale for every copy of the object."""
        s = width / (self.hi[0] - self.lo[0])
        return (uv - (self.lo + self.hi) / 2) * s + np.array([cx, cy])


# --- small drawing helpers ------------------------------------------------------------------------------------------------------------


def card(ax, x, y, w, h, fc=CARD, ec=EDGE, r=1.4, lw=0.8, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.28, y - 0.36),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc="#000000",
            ec="none",
            alpha=0.07,
            zorder=z - 0.1,
        )
    )
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def label(ax, x, y, s, size=7.6, color=TEXT, weight="normal", style_="normal", ha="left", va="center", z=6, **kw):
    return ax.text(
        x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style_, ha=ha, va=va, zorder=z, **kw
    )


def chip(ax, x, y, w, h, s, fc="white", ec=EDGE, tc=TEXT, size=6.8, weight="normal", ls="-", z=6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=fc, ec=ec, lw=0.8, ls=ls, zorder=z
        )
    )
    label(ax, x + w / 2, y + h / 2 - 0.05, s, size=size, color=tc, weight=weight, ha="center", z=z + 1)


def wire(ax, pts, color="#4B5563", lw=1.15, head=True, z=4):
    pts = np.asarray(pts, float)
    ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, solid_capstyle="butt", zorder=z)
    if head:
        ax.add_patch(
            FancyArrowPatch(
                pts[-2],
                pts[-1],
                arrowstyle="-|>",
                mutation_scale=8.5,
                color=color,
                lw=lw,
                shrinkA=0,
                shrinkB=0,
                zorder=z,
            )
        )
    else:
        ax.plot(pts[-2:, 0], pts[-2:, 1], color=color, lw=lw, solid_capstyle="butt", zorder=z)


def badge(ax, x, y, n):
    ax.add_patch(Circle((x, y), 1.85, fc=INK, ec="white", lw=1.2, zorder=7))
    label(ax, x, y - 0.05, str(n), size=9.5, color="white", weight="bold", ha="center", z=8)


def triad(ax, x, y, s=3.2, z=7):
    """The three axes of a coordinate frame: the pose that anchors the scan."""
    for dx, dy, c in ((s, -0.15 * s, RED), (0.0, s, BLUE), (-0.62 * s, -0.55 * s, GREEN)):
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle="-|>",
                mutation_scale=6.5,
                color=c,
                lw=1.25,
                shrinkA=0,
                shrinkB=0,
                zorder=z,
            )
        )
    ax.add_patch(Circle((x, y), 0.32, fc=INK, ec="none", zorder=z + 1))


def shade(light, dark, t):
    """Colours from `light` (grazing) to `dark` (facing the camera)."""
    lo, hi = np.array(to_rgb(light)), np.array(to_rgb(dark))
    t = np.clip(t, 0, 1)[:, None]
    return lo * (1 - t) + hi * t


def points(ax, xy, depth, t, light, dark, s=1.0, z=5, alpha=1.0):
    """A cloud drawn far to near, so the near side hides the far side."""
    o = np.argsort(depth)
    ax.scatter(xy[o, 0], xy[o, 1], s=s, c=shade(light, dark, t[o]), alpha=alpha, linewidths=0, zorder=z)


def photo_mug(ax, cx, cy, sc, fc, ec, alpha=1.0, lw=0.6, z=5, handle=None):
    """The mug in a photograph: a body, an elliptical rim and a handle (the same shapes serve for the object and for its mask)."""
    bw, bh = 3.4 * sc, 4.6 * sc
    ax.add_patch(
        Arc(
            (cx + bw / 2, cy + bh * 0.02),
            2.6 * sc,
            2.9 * sc,
            theta1=-80,
            theta2=80,
            color=handle or fc,
            lw=1.6 * sc,
            alpha=alpha,
            zorder=z - 0.1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2),
            bw,
            bh,
            boxstyle=f"round,pad=0,rounding_size={0.35 * sc}",
            fc=fc,
            ec=ec,
            lw=lw,
            alpha=alpha,
            zorder=z,
        )
    )
    ax.add_patch(Ellipse((cx, cy + bh / 2), bw, 1.15 * sc, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z + 0.1))


def photo(ax, x, y, w, h, ox, oy, mask=False, z=3, sc=1.0):
    """A stand-in for a camera frame: a table, some clutter and the target mug at (ox, oy) in frame coordinates."""
    t = np.linspace(0, 1, 40)[:, None, None]
    img = np.repeat(np.array([0.90, 0.92, 0.95]) * (1 - t) + np.array([0.72, 0.77, 0.83]) * t, 40, axis=1)
    clip = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec="none", zorder=z)
    ax.add_patch(clip)
    im = ax.imshow(img, extent=(x, x + w, y, y + h), origin="upper", zorder=z, interpolation="bilinear", aspect="auto")
    im.set_clip_path(clip)
    ax.add_patch(Rectangle((x, y), w, h * 0.42, fc="#FFFFFF", alpha=0.35, ec="none", zorder=z + 0.1))
    for fx, fy, fw, fh, fc in (
        (0.18, 0.30, 0.20, 0.17, "#6F86A0"),
        (0.74, 0.62, 0.17, 0.15, "#5FA79B"),
        (0.80, 0.26, 0.15, 0.13, "#D6C29A"),
        (0.30, 0.70, 0.13, 0.12, "#8C7FA6"),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x + fx * w - fw * w / 2, y + fy * h - fh * h / 2),
                fw * w,
                fh * h,
                boxstyle="round,pad=0,rounding_size=0.5",
                fc=fc,
                ec="none",
                alpha=0.9,
                zorder=z + 0.2,
            )
        )
    cx, cy = x + ox * w, y + oy * h
    if mask:
        photo_mug(ax, cx, cy, sc, ORANGE, "#B87700", alpha=0.6, lw=1.2, z=z + 0.4, handle=ORANGE)
    else:
        photo_mug(ax, cx, cy, sc, "#B4573B", "#8E3F29", z=z + 0.4)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec="#98A2AE", lw=0.8, zorder=z + 2
        )
    )


def camera_icon(ax, pos, target, z=7):
    """A camera as a frustum: the centre of projection, and the image plane in front of it."""
    d = (target - pos) / np.linalg.norm(target - pos)
    nrm = np.array([-d[1], d[0]])
    base = pos + d * 1.9
    ax.plot(*np.c_[base, target - d * 0.7], color=INK, lw=0.6, ls=(0, (1.6, 1.4)), alpha=0.6, zorder=z - 1)
    ax.add_patch(
        Polygon([pos, base + nrm * 1.25, base - nrm * 1.25], fc="#DDE6EF", ec=INK, lw=0.8, joinstyle="round", zorder=z)
    )
    ax.add_patch(Circle(pos, 0.38, fc=INK, ec="none", zorder=z + 1))


def tile(ax, x, y, w, h, arr, cmap, z=5):
    clip = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", fc="none", ec="none", zorder=z)
    ax.add_patch(clip)
    im = ax.imshow(
        arr,
        extent=(x, x + w, y, y + h),
        origin="upper",
        cmap=cmap,
        zorder=z,
        interpolation="bilinear",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    im.set_clip_path(clip)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", fc="none", ec="#98A2AE", lw=0.8, zorder=z + 1
        )
    )


# --- the figure -----------------------------------------------------------------------------------------------------------------------


def draw(out: Path) -> None:
    style()
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    fig = plt.figure(figsize=(W * 0.08, H * 0.08))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=0))

    rng = np.random.default_rng(3)
    p, n = mug(rng)
    cam = Camera(p, n)
    vis = cam.cos > 0.04
    noise = rng.normal(0, 0.022, (len(p), 2))

    # ---- the four stage cards -------------------------------------------------------------------------------------------------------
    cw, ch, y0, gap = 24.0, 32.0, 11.5, 3.5
    top = y0 + ch
    xs = [1.75 + i * (cw + gap) for i in range(4)]
    mid = [x + cw / 2 for x in xs]
    for x in xs:
        card(ax, x, y0, cw, ch)
    titles = ["Segment", "Clean", "Align", "Score"]
    descs = [
        "mask of the object\nin every view",
        "masked depth to a\npoint cloud",
        "cloud into the frame\nof the scan",
        "distances to the visible\npart of the scan",
    ]
    for i, x in enumerate(xs):
        badge(ax, x + 2.7, top - 2.9, i + 1)
        label(ax, x + 5.5, top - 2.9, titles[i], size=11.5, weight="bold", color=INK)
        label(ax, x + 1.8, top - 7.5, descs[i], size=7.1, color=MUTED, style_="italic", linespacing=1.3)
    for i in range(3):
        wire(ax, [(xs[i] + cw + 0.25, y0 + ch / 2), (xs[i + 1] - 0.25, y0 + ch / 2)], lw=1.4)

    # 1 segment: a frame with the mask, and the sources of masks
    photo(ax, mid[0] - 7.1, y0 + 14.4, 14.2, 7.6, 0.5, 0.50, mask=True)
    rows = [
        [("Our method (head)", ORANGE, INK, "-")],
        [("ground truth", "white", TEXT, "-"), ("SAM 3", "white", TEXT, "-")],
        [("CNOS", "white", TEXT, "-"), ("ZebraPose", "white", TEXT, "-")],
        [("your own", "white", MUTED, (0, (2, 1.6)))],
    ]
    for r, row in enumerate(rows):
        wd = (cw - 3.2 - 0.9 * (len(row) - 1)) / len(row)
        for c, (s, fc, tc, ls) in enumerate(row):
            chip(
                ax,
                xs[0] + 1.6 + c * (wd + 0.9),
                y0 + 10.1 - r * 2.9,
                wd,
                2.3,
                s,
                fc=fc,
                ec=ORANGE if fc == ORANGE else EDGE,
                tc=tc,
                size=6.8,
                weight="bold" if fc == ORANGE else "normal",
                ls=ls,
            )

    # 2 clean: the depth inside the mask, then the cloud it makes
    uv_c, tv, dv = cam.uv[vis] + noise[vis], cam.cos[vis], cam.depth[vis]
    cy2 = y0 + 14.0
    pa, pb = cam.place(uv_c, xs[1] + 6.4, cy2, 8.8), cam.place(uv_c, xs[1] + 17.6, cy2, 8.8)
    o = np.argsort(dv)
    ax.scatter(pa[o, 0], pa[o, 1], s=1.6, c=plt.cm.viridis((dv[o] - dv.min()) / np.ptp(dv)), linewidths=0, zorder=5)
    points(ax, pb, dv, tv, "#9CCBE8", BLUE)
    wire(ax, [(xs[1] + 10.9, cy2), (xs[1] + 13.1, cy2)], lw=1.0)
    label(ax, xs[1] + 6.4, y0 + 7.4, "masked depth", size=6.5, color=MUTED, style_="italic", ha="center")
    label(ax, xs[1] + 17.6, y0 + 7.4, "point cloud", size=6.5, color=MUTED, style_="italic", ha="center")

    # 3 align: the cloud and the scan before the alignment, and after it
    cy3 = y0 + 14.0
    cl_uv = cam.uv[vis] + 1.6 * noise[vis]
    half = rng.random(len(cl_uv)) < 0.55  # a sparser cloud, so that the scan under it shows
    ang = np.deg2rad(22)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    centre = (cam.lo + cam.hi) / 2
    before = (cl_uv - centre) @ rot.T * 0.8 + centre + np.array([-0.1, -0.22])
    for cx, cl in ((xs[2] + 6.2, before), (xs[2] + 17.8, cl_uv)):
        scan_xy = cam.place(cam.uv, cx, cy3, 8.4)
        points(ax, scan_xy, cam.depth, np.where(cam.cos > 0.04, 0.55 + 0.4 * cam.cos, 0.0), "#D3D6DB", GREY, s=1.0, z=4)
        cl_xy = cam.place(cl[half], cx, cy3, 8.4)
        points(ax, cl_xy, dv[half], tv[half], "#9CCBE8", BLUE, s=1.4, z=5, alpha=0.9)
    wire(ax, [(xs[2] + 10.9, cy3), (xs[2] + 13.1, cy3)], lw=1.0)
    label(ax, xs[2] + 6.4, y0 + 7.4, "before", size=6.5, color=MUTED, style_="italic", ha="center")
    label(ax, xs[2] + 17.6, y0 + 7.4, "after", size=6.5, color=MUTED, style_="italic", ha="center")

    # 4 score: the distances to the scan, and a threshold on them
    hx0, hy0, hw, hh = xs[3] + 3.0, y0 + 12.6, 18.0, 6.8
    bars = np.array([1.0, 2.7, 4.4, 5.6, 5.0, 3.6, 2.4, 1.5, 0.9, 0.55, 0.35, 0.2])
    bw, thr = hw / len(bars), 5
    for k, b in enumerate(bars):
        ax.add_patch(
            Rectangle(
                (hx0 + k * bw + 0.12, hy0),
                bw - 0.24,
                b / bars.max() * hh,
                fc=BLUE if k < thr else "#B8C0CA",
                ec="none",
                zorder=5,
            )
        )
    ax.plot([hx0 - 0.3, hx0 + hw + 0.3], [hy0, hy0], color="#4B5563", lw=0.9, zorder=6)
    ax.plot([hx0 + thr * bw] * 2, [hy0 - 0.5, hy0 + hh + 1.0], color=INK, lw=0.9, ls=(0, (2, 1.4)), zorder=6)
    label(ax, hx0 + thr * bw + 0.6, hy0 + hh + 0.9, "threshold", size=6.2, color=INK, style_="italic")
    label(ax, hx0 + hw / 2, hy0 - 1.8, "distance to the scan", size=6.3, color=MUTED, style_="italic", ha="center")
    xc = xs[3] + 0.9
    for s, wd in (("accuracy", 6.4), ("completeness", 8.9), ("F-score", 5.8)):
        chip(ax, xc, y0 + 4.4, wd, 2.7, s, fc="white", tc=TEXT, size=5.9)
        xc += wd + 0.55

    # ---- the images, the model, its output --------------------------------------------------------------------------------------
    for k, (dx, dy, ox) in enumerate(((2.4, -2.0, 0.60), (1.2, -1.0, 0.50), (0.0, 0.0, 0.42))):
        photo(ax, 3.6 + dx, 55.0 + dy, 13.6, 9.3, ox, 0.48, z=3 + k * 0.5, sc=0.95)
    label(ax, 3.6, 67.4, "images of one static scene", size=7.2, color=INK, weight="bold")
    label(ax, 3.6, 49.6, "2–6 views", size=6.6, color=MUTED, style_="italic")
    wire(ax, [(13.4, 51.6), (13.4, top + 0.35)])

    mx, my, mw, mh = 30.0, 49.0, 36.0, 18.6
    card(ax, mx, my, mw, mh, fc="#EEF4FA", ec="#B5CBE0")
    label(
        ax,
        mx + mw / 2,
        my + mh - 2.6,
        "feed-forward reconstruction model",
        size=8.4,
        weight="bold",
        color=INK,
        ha="center",
    )
    gx, gy = mx + mw / 2, my + 8.4
    for sgn in (-1, 1):
        ax.fill(
            [gx + sgn * 1.2, gx + sgn * 1.2, gx + sgn * 9.4, gx + sgn * 9.4],
            [gy - 1.8, gy + 1.8, gy + 4.6, gy - 4.6],
            fc="#CFE0F0",
            ec=BLUE,
            lw=0.9,
            zorder=5,
        )
    for k in range(3):
        ax.add_patch(Rectangle((gx - 0.7, gy - 1.5 + k * 1.1), 1.4, 0.7, fc=BLUE, ec="none", zorder=6))
    label(ax, mx + mw / 2, my + 2.2, "DA3  ·  VGGT  ·  your own", size=7.0, color=MUTED, ha="center")
    wire(ax, [(20.6, 58.3), (mx - 0.3, 58.3)])

    ox0, oy0, ow, oh = 71.0, 49.0, 38.0, 18.6
    card(ax, ox0, oy0, ow, oh, fc="white", ec=EDGE)
    tw, th_ = 15.0, 9.4
    tx = [ox0 + 2.4, ox0 + 2.4 + tw + 3.2]
    ty = oy0 + 5.0
    yy, xx = np.mgrid[0:1:48j, 0:1:64j]
    blob = np.exp(-(((xx - 0.5) / 0.20) ** 2 + ((yy - 0.45) / 0.30) ** 2))
    tile(ax, tx[0], ty, tw, th_, 0.30 + 0.55 * yy - 0.30 * blob, "viridis")
    ax.add_patch(
        FancyBboxPatch(
            (tx[1], ty), tw, th_, boxstyle="round,pad=0,rounding_size=0.7", fc="#F6F7F9", ec="#98A2AE", lw=0.8, zorder=5
        )
    )
    target = np.array([tx[1] + tw / 2, ty + th_ / 2 + 0.3])
    for ang in (-155, -90, -25):
        a = np.deg2rad(ang)
        camera_icon(ax, target + np.array([5.4 * np.cos(a), 3.4 * np.sin(a)]), target)
    ax.add_patch(Circle(target, 0.62, fc=ORANGE, ec="white", lw=0.6, zorder=8))
    for k, s in enumerate(("depth", "cameras")):
        label(ax, tx[k] + tw / 2, oy0 + 2.7, s, size=7.0, color=TEXT, ha="center")
    label(ax, ox0 + ow / 2, oy0 + oh - 1.7, "output", size=6.4, color=MUTED, style_="italic", ha="center")
    wire(ax, [(mx + mw + 0.3, 58.3), (ox0 - 0.3, 58.3)])

    # the depth goes to the cleaning, the cameras to the alignment
    y_bus, y_cam = 46.6, 45.1
    dx_, camx = tx[0] + tw / 2, tx[1] + tw / 2
    wire(ax, [(dx_, oy0 - 0.2), (dx_, y_bus), (mid[1], y_bus), (mid[1], top + 0.35)])
    wire(ax, [(camx, oy0 - 0.2), (camx, y_cam), (mid[2], y_cam), (mid[2], top + 0.35)])

    # ---- what the dataset gives for the object: its pose and its scan ----------------------------------------------------------------
    gy0, gh = 1.2, 6.4
    card(ax, xs[2], gy0, cw, gh, fc="#F1F9F6", ec="#9AD3C0")
    triad(ax, xs[2] + 4.2, gy0 + 3.0, s=2.5)
    label(ax, xs[2] + 8.4, gy0 + gh * 0.68, "pose in the scene", size=7.2, weight="bold")
    label(ax, xs[2] + 8.4, gy0 + gh * 0.28, "the independent anchor", size=6.6, color="#0B6B50", style_="italic")
    wire(ax, [(mid[2], gy0 + gh + 0.2), (mid[2], y0 - 0.3)], color=GREEN, lw=1.5)

    card(ax, xs[3], gy0, cw, gh, fc="#F3F4F6", ec=EDGE)
    sc_xy = cam.place(cam.uv, xs[3] + 4.5, gy0 + gh / 2 + 0.1, 4.8)
    points(ax, sc_xy, cam.depth, np.where(cam.cos > 0.04, 0.55 + 0.4 * cam.cos, 0.0), "#D3D6DB", GREY, s=0.7, z=6)
    label(ax, xs[3] + 8.4, gy0 + gh * 0.68, "scan of the object", size=7.2, weight="bold")
    label(ax, xs[3] + 8.4, gy0 + gh * 0.28, "the reference surface", size=6.6, color=MUTED, style_="italic")
    wire(ax, [(mid[3], gy0 + gh + 0.2), (mid[3], y0 - 0.3)], color=GREY, lw=1.5)
    # the scan is also what the alignment searches the depth scale against
    wire(
        ax,
        [(mid[3], gy0 + gh + 1.9), (xs[2] + 18.6, gy0 + gh + 1.9), (xs[2] + 18.6, y0 - 0.3)],
        color=GREY,
        lw=1.5,
        head=True,
    )
    ax.add_patch(Circle((mid[3], gy0 + gh + 1.9), 0.5, fc=GREY, ec="none", zorder=6))

    # the colours, in the free corner
    names = [(BLUE, "reconstruction"), (GREY, "scan")]
    texts = [label(ax, 0, gy0 + gh / 2, s, size=6.8) for _, s in names]
    fig.canvas.draw()
    inv = ax.transData.inverted()
    widths = [
        inv.transform(t.get_window_extent())[1, 0] - inv.transform(t.get_window_extent())[0, 0] + 2.0 for t in texts
    ]
    gap = 5.0
    ax.add_patch(
        FancyBboxPatch(
            (1.75, gy0),
            sum(widths) + gap * (len(widths) - 1) + 4.4,
            gh,
            boxstyle="round,pad=0,rounding_size=1.4",
            fc="white",
            ec=EDGE,
            lw=0.8,
            zorder=1,
        )
    )
    xk = 3.6
    for (c, _), t, wd in zip(names, texts, widths):
        ax.add_patch(Circle((xk, gy0 + gh / 2), 0.8, fc=c, ec="none", zorder=6))
        t.set_x(xk + 1.7)
        xk += wd + gap

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/pipeline.png"))
    a = ap.parse_args()
    draw(a.out)
    print(f"figure -> {a.out}")


if __name__ == "__main__":
    main()
