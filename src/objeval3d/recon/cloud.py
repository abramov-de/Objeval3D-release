"""Stage 2 helpers that need neither a model nor a GPU: back-projection of masked depth into a point cloud, the agreement of the
per-view clouds with each other, the crop around the object (a control that is not recommended, see the paper's limitations), and
the overlays used to look at a mask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from objeval3d.data.bop import BopScene
from objeval3d.threads import workers

PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 120, 255],
        [255, 140, 0],
        [220, 30, 30],
        [40, 200, 90],
        [220, 0, 220],
        [0, 200, 200],
        [255, 220, 0],
        [150, 80, 255],
        [120, 200, 40],
        [255, 90, 140],
        [90, 140, 255],
        [200, 200, 200],
        [255, 170, 90],
        [30, 160, 120],
        [180, 60, 60],
        [60, 60, 180],
        [200, 120, 200],
        [110, 110, 40],
        [40, 110, 110],
        [230, 230, 120],
        [160, 20, 90],
    ],
    dtype=np.uint8,
)


def multiview_consistency(pts, view_ids, n_views: int) -> dict:
    """How well the per-view point clouds agree with each other.

    Median distance from each view's points to the nearest point contributed by
    any *other* view, as a fraction of the cloud's bounding-box diagonal. A view
    that failed to register shows up here as a large value long before it shows
    up as a bad metric, and unlike the metrics this needs no ground truth.
    """
    if len(pts) < 50 or n_views < 2:
        return {}
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    per_view = {}
    for i in range(n_views):
        a, b = pts[view_ids == i], pts[view_ids != i]
        if len(a) < 10 or len(b) < 10:
            continue
        d, _ = cKDTree(b).query(a, k=1, workers=workers())
        per_view[str(i)] = float(np.median(d) / max(diag, 1e-9))
    if not per_view:
        return {}
    return {"per_view_rel": per_view, "max_rel": max(per_view.values())}


def backproject(depth, mask, K, E, images_u8, conf=None, conf_thresh=None):
    """Per-view unprojection into the reconstruction's own world frame.

    `E` is world->camera; a point at pixel (u,v) with depth d is
    X_world = (E^-1) @ [d * K^-1 @ [u,v,1], 1].
    """
    n, h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3).astype(np.float64)

    pts, cols, view_ids = [], [], []
    for i in range(n):
        valid = (depth[i] > 0) & mask[i]
        if conf is not None and conf_thresh is not None:
            valid &= conf[i] >= conf_thresh
        if not valid.any():
            continue
        idx = np.flatnonzero(valid.reshape(-1))
        d = depth[i].reshape(-1)[idx]
        Kinv = np.linalg.inv(K[i])
        Eh = np.eye(4)
        Eh[:3, :4] = E[i][:3, :4]
        c2w = np.linalg.inv(Eh)
        rays = Kinv @ pix[idx].T
        xc = rays * d[None, :]
        xw = (c2w @ np.vstack([xc, np.ones((1, xc.shape[1]))]))[:3].T
        pts.append(xw.astype(np.float64))
        cols.append(images_u8[i].reshape(-1, 3)[idx].astype(np.uint8))
        view_ids.append(np.full(len(idx), i, dtype=np.int16))
    if not pts:
        return (np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0,), np.int16))
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(view_ids)


def save_overlays(images_u8, class_map, out_dir: Path, target_class: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, cm) in enumerate(zip(images_u8, class_map)):
        ov = img.copy()
        for c in np.unique(cm):
            if c == 0:
                continue
            m = cm == c
            col = PALETTE[int(c) % len(PALETTE)]
            alpha = 0.55 if c == target_class else 0.25
            ov[m] = (ov[m] * (1 - alpha) + col * alpha).astype(np.uint8)
            border = m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            ov[border] = [255, 230, 0] if c == target_class else col
        cv2.imwrite(str(out_dir / f"overlay_{i:02d}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))


def ckpt_fingerprint(path: str) -> str | None:
    """Identity of the segmentation checkpoint, not just its path.

    Retraining writes the same filename, so `--skip-existing` comparing paths
    would happily reuse masks produced by a previous head. Size and mtime are
    enough to notice that and cheap enough to take on every run.
    """
    if not path:
        return None
    f = Path(path)
    if not f.exists():
        return None
    st = f.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compute_crop_boxes(masks, img_hw, margin: float, square: bool = True):
    """One crop box per view, centred on that view's object.

    Every box gets the **same pixel size**, deliberately: DepthAnything3's input
    processor centre-crops a batch to its smallest member when the images differ
    in size, which would silently shift some views relative to the masks and
    ground truth derived from the same boxes.

    Cropping is what makes a small distant object measurable at all. Depth error
    is relative to distance, so on LM-O — a 100 mm object at 850 mm covering
    ~2000 px — a 10% depth error displaces a view's points by ~90 mm, i.e. by the
    whole object, and the per-view clouds stop overlapping. Framing the object
    spends the network's resolution on it instead of on the table around it.
    """
    H, W = img_hw
    bbs = [mask_bbox(m) for m in masks]
    if all(b is None for b in bbs):
        return None
    bw = max((b[2] - b[0]) for b in bbs if b) * (1 + 2 * margin)
    bh = max((b[3] - b[1]) for b in bbs if b) * (1 + 2 * margin)
    if square:
        bw = bh = max(bw, bh)
    bw, bh = int(min(W, round(bw))), int(min(H, round(bh)))

    boxes = []
    for b in bbs:
        cx, cy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else (W / 2, H / 2)
        x0 = max(0, min(int(round(cx - bw / 2)), W - bw))
        y0 = max(0, min(int(round(cy - bh / 2)), H - bh))
        boxes.append((x0, y0, x0 + bw, y0 + bh))
    return boxes


def gt_masks_full_res(vs: dict, bop_root: Path) -> list[np.ndarray]:
    scene = BopScene(bop_root / vs["split"] / f"{vs['scene_id']:06d}")
    out = []
    for im_id, gt_id in zip(vs["im_ids"], vs["gt_ids"]):
        m = cv2.imread(str(scene.mask_visib_path(im_id, gt_id)), cv2.IMREAD_GRAYSCALE)
        out.append(None if m is None else (m > 127))
    return out
