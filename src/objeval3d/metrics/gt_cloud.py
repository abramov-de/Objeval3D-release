"""Ground-truth object geometry for one view set.

Two clouds are produced, and the difference between them is the whole reason
this module exists:

  gt_full     - the complete scanned model, uniformly sampled on its surface.
                Used as the reference for *accuracy*: a reconstructed point
                that lands on the far side of the object is still on the
                object, and must not be counted as an error.

  gt_visible  - the subset of the model that at least one input view could
                actually see.  Used as the reference for *completeness*: a
                3-view feed-forward model cannot reconstruct the bottom of a
                mug, and scoring it against the full model measures the
                capture setup, not the model.

Visibility is decided per view by three tests, all of which must pass:
  1. the point projects inside the image;
  2. it survives a z-buffer test against the object's own sampled surface
     (self-occlusion - the far side of the object);
  3. its pixel is inside BOP's `mask_visib` for that object in that frame
     (occlusion by *other* objects, and truncation at the image border).

Test 3 is what makes this cheap and exact: BOP already ships per-annotation
visible masks, so no renderer and no other-object geometry is needed.

Debug export of the two clouds of one view set:

    python -m objeval3d.metrics.gt_cloud --manifest M.json --set-id SET --out-dir DIR
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from objeval3d.data.bop import (
    BopScene,
    cam_center_in_model_frame,
    cam_to_model,
    load_mesh,
    load_models_info,
    model_path,
    sample_mesh_surface,
)
from objeval3d.symmetry import symmetry_transforms  # noqa: F401  (re-exported for callers of the old module)


def project(
    points_model: np.ndarray, R_m2c: np.ndarray, t_m2c: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Model-frame points -> (pixel coords [N,2] float, depth [N] mm)."""
    Xc = (R_m2c @ points_model.T + t_m2c.reshape(3, 1)).T
    z = Xc[:, 2]
    safe = np.maximum(z, 1e-6)
    uv = (K @ (Xc / safe[:, None]).T).T[:, :2]
    return uv, z


def visible_mask_one_view(
    points_model: np.ndarray,
    normals_model: np.ndarray | None,
    R_m2c: np.ndarray,
    t_m2c: np.ndarray,
    K: np.ndarray,
    mask_visib: np.ndarray | None,
    image_hw: tuple[int, int],
    z_eps: float,
    zbuf_scale: int = 4,
) -> np.ndarray:
    """Which sampled model points this one view can see.

    Three independent tests, all of which must pass:

    1. **Back-face culling** - the sample's outward normal must point towards
       the camera.  This is the primary self-occlusion test and it is exact for
       a convex object.  A per-pixel z-buffer over point samples cannot do this
       job on its own: point samples do not tile the image, so many pixels
       receive a sample from the far surface and none from the near one, and
       the far surface is then wrongly declared visible.  (Widening the
       z-buffer with a min-filter fixes those holes but starts culling
       legitimate points on slanted surfaces, where depth varies by far more
       than any sane tolerance across the filter window.)
    2. **Coarse z-buffer** - catches genuine self-occlusion that is not
       back-facing (a drill body behind its own handle).  Run at 1/`zbuf_scale`
       resolution so every buffer cell collects enough samples to be
       hole-free, with a deliberately generous `z_eps`: it should only reject
       points that are clearly behind another surface.
    3. **BOP `mask_visib`** - occlusion by *other* objects and truncation at
       the image border, straight from the dataset, at full resolution.
    """
    h, w = image_hw
    uv, z = project(points_model, R_m2c, t_m2c, K)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    if normals_model is not None:
        cam_c = cam_center_in_model_frame(R_m2c, t_m2c)
        to_cam = cam_c[None, :] - points_model
        inside &= np.einsum("ij,ij->i", normals_model, to_cam) > 0

    vis = np.zeros(len(points_model), dtype=bool)
    if not inside.any():
        return vis

    hs, ws = max(1, h // zbuf_scale), max(1, w // zbuf_scale)
    us = np.clip(u[inside] // zbuf_scale, 0, ws - 1)
    vs_ = np.clip(v[inside] // zbuf_scale, 0, hs - 1)
    flat_s = vs_ * ws + us
    zbuf = np.full(hs * ws, np.inf)
    np.minimum.at(zbuf, flat_s, z[inside])
    front = z[inside] <= (zbuf[flat_s] + z_eps)

    ok = front
    if mask_visib is not None:
        ok = ok & (mask_visib.reshape(-1)[v[inside] * w + u[inside]] > 0)

    idx = np.flatnonzero(inside)
    vis[idx[ok]] = True
    return vis


def build_gt_clouds(
    view_set: dict,
    bop_root: Path,
    *,
    n_sample: int = 200_000,
    n_keep: int = 60_000,
    z_eps_frac: float = 0.05,
    zbuf_scale: int = 4,
    seed: int = 0,
) -> dict:
    """Returns gt_full / gt_visible clouds (mm, model frame), per-view
    visibility, the object diameter and the ground-truth camera poses."""
    bop_root = Path(bop_root)
    obj_id = int(view_set["obj_id"])
    info = load_models_info(bop_root)[obj_id]
    diameter = float(info["diameter"])

    verts, faces = load_mesh(model_path(bop_root, obj_id))
    pts, nrm = sample_mesh_surface(verts, faces, n_sample, seed=seed, return_normals=True)

    scene = BopScene(bop_root / view_set["split"] / f"{view_set['scene_id']:06d}")
    z_eps = max(0.5, z_eps_frac * diameter)

    vis_any = np.zeros(len(pts), dtype=bool)
    per_view = []
    cam_to_model_poses = []
    for i, (im_id, gt_id) in enumerate(zip(view_set["im_ids"], view_set["gt_ids"])):
        K = np.asarray(view_set["K"][i], dtype=np.float64).reshape(3, 3)
        R = np.asarray(view_set["R_m2c"][i], dtype=np.float64).reshape(3, 3)
        t = np.asarray(view_set["t_m2c"][i], dtype=np.float64).reshape(3, 1)
        mv = cv2.imread(str(scene.mask_visib_path(im_id, gt_id)), cv2.IMREAD_GRAYSCALE)
        if mv is None:
            raise FileNotFoundError(scene.mask_visib_path(im_id, gt_id))
        h, w = mv.shape[:2]
        vis = visible_mask_one_view(pts, nrm, R, t, K, mv, (h, w), z_eps, zbuf_scale=zbuf_scale)
        vis_any |= vis
        per_view.append(int(vis.sum()))
        cam_to_model_poses.append(cam_to_model(R, t))

    rng = np.random.default_rng(seed)

    def cap(a: np.ndarray) -> np.ndarray:
        return a if len(a) <= n_keep else a[rng.choice(len(a), n_keep, replace=False)]

    gt_full = cap(pts)
    gt_visible = cap(pts[vis_any])
    return {
        "gt_full": gt_full,
        "gt_visible": gt_visible,
        "visible_fraction": float(vis_any.mean()),
        "visible_per_view": per_view,
        "diameter": diameter,
        "obj_id": obj_id,
        "cam_to_model": np.stack(cam_to_model_poses) if cam_to_model_poses else None,
        "symmetries": {
            "discrete": info.get("symmetries_discrete", []),
            "continuous": info.get("symmetries_continuous", []),
        },
    }


def main() -> None:
    import argparse
    import json

    from objeval3d.data.bop import write_ply

    ap = argparse.ArgumentParser(description="Export the ground-truth clouds of one view set (debugging aid)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    with open(a.manifest) as f:
        man = json.load(f)
    vs = next(s for s in man["sets"] if s["set_id"] == a.set_id)
    g = build_gt_clouds(vs, Path(man["bop_root"]))
    out = Path(a.out_dir)
    write_ply(out / "gt_full.ply", g["gt_full"])
    write_ply(out / "gt_visible.ply", g["gt_visible"])
    print(
        f"diameter={g['diameter']:.1f} mm  visible={100 * g['visible_fraction']:.1f}%  per view {g['visible_per_view']}"
    )


if __name__ == "__main__":
    main()
