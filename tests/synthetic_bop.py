"""A tiny BOP-format dataset and a matching run folder, built from a box, for tests that must run without the real data or a GPU.

The box (80 x 50 x 120 mm) is seen by two cameras. The "reconstruction" is the visible part of its surface, expressed in the world
frame of a scale-free predictor (the frame of camera 0, divided by an unknown depth scale `K_TRUE`), with Gaussian noise added; the
predicted extrinsics are consistent with that frame. This is exactly what `objeval3d.align.evaluate` expects of a run folder, so it
exercises the ground-truth clouds, the visibility test, the reference mode, the free fit and the metrics end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from objeval3d.data.bop import cam_center_in_model_frame, sample_mesh_surface, write_ply  # noqa: F401
from objeval3d.metrics.gt_cloud import project, visible_mask_one_view

DIMS = (80.0, 50.0, 120.0)
K_TRUE = 3.7  # mm per unit of the predictor's world
IMAGE_HW = (480, 640)
CAM_K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
POSES = [  # (rotation vector in degrees about x, y, z; translation in mm) of model -> camera
    ((30.0, 10.0, 0.0), (10.0, -20.0, 700.0)),
    ((45.0, -25.0, 25.0), (-30.0, 10.0, 750.0)),
]


def box_mesh(dims=DIMS):
    """Closed, consistently wound (outward) triangle mesh of a box."""
    hx, hy, hz = (d / 2 for d in dims)
    v = np.array([[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)], dtype=np.float64)
    # vertex index = 4*(z>0) + 2*(y>0) + (x>0)
    f = [
        (0, 2, 3),
        (0, 3, 1),  # -z
        (4, 5, 7),
        (4, 7, 6),  # +z
        (0, 1, 5),
        (0, 5, 4),  # -y
        (2, 6, 7),
        (2, 7, 3),  # +y
        (0, 4, 6),
        (0, 6, 2),  # -x
        (1, 3, 7),
        (1, 7, 5),
    ]  # +x
    return v, np.array(f, dtype=np.int64)


def write_mesh_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """The model as the BOP archives ship it: an ASCII PLY with float vertices and `list uchar int vertex_indices` faces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        "comment synthetic",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines += [" ".join(f"{float(np.float32(c)):.9g}" for c in v) for v in verts]
    lines += [f"3 {f[0]} {f[1]} {f[2]}" for f in faces]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def pose(i: int):
    (rx, ry, rz), t = POSES[i]
    R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    return R, np.asarray(t, dtype=np.float64).reshape(3, 1)


def silhouette(verts, R, t) -> np.ndarray:
    """Visible mask of a convex object: the filled convex hull of its projected vertices."""
    uv, _ = project(verts, R, t, CAM_K)
    mask = np.zeros(IMAGE_HW, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    return mask


def build_dataset(root: Path, *, noise_mm: float = 0.5, seed: int = 0) -> dict:
    """Write the BOP dataset under `root` and return its manifest (one view set)."""
    verts, faces = box_mesh()
    diameter = float(np.linalg.norm(np.asarray(DIMS)))
    write_mesh_ply(root / "models" / "obj_000001.ply", verts, faces)
    # a box with three different edges maps onto itself under a half turn about each axis (BOP lists such symmetries as flat 4x4 matrices)
    half_turns = [np.diag([*d, 1.0]).reshape(-1).tolist() for d in ((1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    (root / "models" / "models_info.json").write_text(
        json.dumps(
            {
                "1": {
                    "diameter": diameter,
                    "min_x": -DIMS[0] / 2,
                    "min_y": -DIMS[1] / 2,
                    "min_z": -DIMS[2] / 2,
                    "size_x": DIMS[0],
                    "size_y": DIMS[1],
                    "size_z": DIMS[2],
                    "symmetries_discrete": half_turns,
                }
            }
        )
    )

    scene = root / "test" / "000001"
    cam, gt = {}, {}
    for i in range(len(POSES)):
        R, t = pose(i)
        cam[str(i)] = {"cam_K": CAM_K.reshape(-1).tolist(), "depth_scale": 1.0}
        gt[str(i)] = [{"cam_R_m2c": R.reshape(-1).tolist(), "cam_t_m2c": t.reshape(-1).tolist(), "obj_id": 1}]
        mdir = scene / "mask_visib"
        mdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mdir / f"{i:06d}_000000.png"), silhouette(verts, R, t))
    (scene / "scene_camera.json").write_text(json.dumps(cam))
    (scene / "scene_gt.json").write_text(json.dumps(gt))

    view_set = {
        "set_id": "synthetic_s000001_o01_v00",
        "obj_id": 1,
        "obj_name": "box",
        "scene_id": 1,
        "split": "test",
        "im_ids": list(range(len(POSES))),
        "gt_ids": [0] * len(POSES),
        "K": [CAM_K.reshape(-1).tolist()] * len(POSES),
        "R_m2c": [pose(i)[0].reshape(-1).tolist() for i in range(len(POSES))],
        "t_m2c": [pose(i)[1].reshape(-1).tolist() for i in range(len(POSES))],
        "pair_angles_deg": [30.0],
    }
    return {"bop_root": str(root), "sets": [view_set]}


def build_run(run_dir: Path, view_set: dict, bop_root: Path, *, noise_mm: float = 0.5, seed: int = 0) -> dict:
    """Write the run folder (`meta.json`, `pred.npz`, `object_points.ply`) a reconstruction would leave."""
    rng = np.random.default_rng(seed)
    verts, faces = box_mesh()
    pts, nrm = sample_mesh_surface(verts, faces, 40000, seed=seed, return_normals=True)
    diameter = float(np.linalg.norm(np.asarray(DIMS)))

    n_views = len(POSES)
    seen = np.full(len(pts), -1, dtype=np.int64)  # first view that sees the point
    for i in range(n_views):
        R, t = pose(i)
        mv = cv2.imread(str(bop_root / "test" / "000001" / "mask_visib" / f"{i:06d}_000000.png"), cv2.IMREAD_GRAYSCALE)
        vis = visible_mask_one_view(pts, nrm, R, t, CAM_K, mv, IMAGE_HW, z_eps=0.05 * diameter)
        seen[(seen < 0) & vis] = i
    keep = seen >= 0
    cloud_model, view_ids = pts[keep], seen[keep]

    R0, t0 = pose(0)
    cloud_c0 = (R0 @ cloud_model.T + t0).T  # camera-0 frame, mm
    cloud_world = cloud_c0 / K_TRUE + rng.normal(scale=noise_mm / K_TRUE, size=cloud_c0.shape)

    extr, depth, obj_mask = [], [], []
    for i in range(n_views):
        R, t = pose(i)
        R_rel = R @ R0.T
        t_rel = t - R_rel @ t0
        extr.append(np.hstack([R_rel, t_rel / K_TRUE]))
        mv = (
            cv2.imread(str(bop_root / "test" / "000001" / "mask_visib" / f"{i:06d}_000000.png"), cv2.IMREAD_GRAYSCALE)
            > 0
        )
        z = (
            np.median((extr[-1][:, :3] @ cloud_world[view_ids == i].T + extr[-1][:, 3:4])[2])
            if (view_ids == i).any()
            else float(t[2, 0]) / K_TRUE
        )
        d = np.zeros(IMAGE_HW, dtype=np.float32)
        d[mv] = z
        depth.append(d)
        obj_mask.append(mv)

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "pred.npz",
        extrinsics=np.stack(extr),
        intrinsics=np.stack([CAM_K] * n_views),
        depth=np.stack(depth),
        obj_mask=np.stack(obj_mask),
        view_ids=view_ids,
    )
    write_ply(run_dir / "object_points.ply", cloud_world)
    meta = {
        "view_set": view_set,
        "tag": "synthetic",
        "mask_source": "gt",
        "mask_morph": 0,
        "mask_metrics": None,
        "multiview_consistency": None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    return {"n_points": int(len(cloud_world)), "k_true": K_TRUE, "centroid": cloud_world.mean(0)}


def look_at(center: np.ndarray):
    """model -> camera pose of a camera at `center` (model frame, mm) looking at the origin, OpenCV axes (x right, y down, z forward)."""
    z = -center / np.linalg.norm(center)
    up = np.array([0.0, 1.0, 0.0])
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z])
    return R, (-R @ center).reshape(3, 1)


def build_orbit_dataset(
    root: Path, *, n_frames: int = 60, step_deg: float = 6.0, distance: float = 700.0, elevation_deg: float = 25.0
) -> dict:
    """A BOP scene of one box seen from a camera that orbits it: the kind of video sequence BOP scenes are.

    Writes the images (black), the visible masks (convex silhouettes), `scene_gt.json`, `scene_camera.json` and `scene_gt_info.json`.
    """
    verts, faces = box_mesh()
    diameter = float(np.linalg.norm(np.asarray(DIMS)))
    write_mesh_ply(root / "models" / "obj_000001.ply", verts, faces)
    (root / "models" / "models_info.json").write_text(json.dumps({"1": {"diameter": diameter}}))
    scene = root / "test" / "000001"
    (scene / "rgb").mkdir(parents=True, exist_ok=True)
    (scene / "mask_visib").mkdir(parents=True, exist_ok=True)
    black = np.zeros((*IMAGE_HW, 3), dtype=np.uint8)
    cam, gt, info = {}, {}, {}
    el = np.radians(elevation_deg)
    for i in range(n_frames):
        th = np.radians(i * step_deg)
        center = distance * np.array([np.sin(th) * np.cos(el), np.sin(el), np.cos(th) * np.cos(el)])
        R, t = look_at(center)
        mask = silhouette(verts, R, t)
        cv2.imwrite(str(scene / "rgb" / f"{i:06d}.png"), black)
        cv2.imwrite(str(scene / "mask_visib" / f"{i:06d}_000000.png"), mask)
        cam[str(i)] = {"cam_K": CAM_K.reshape(-1).tolist(), "depth_scale": 1.0}
        gt[str(i)] = [{"cam_R_m2c": R.reshape(-1).tolist(), "cam_t_m2c": t.reshape(-1).tolist(), "obj_id": 1}]
        info[str(i)] = [{"px_count_visib": int((mask > 0).sum()), "visib_fract": 1.0}]
    (scene / "scene_camera.json").write_text(json.dumps(cam))
    (scene / "scene_gt.json").write_text(json.dumps(gt))
    (scene / "scene_gt_info.json").write_text(json.dumps(info))
    return {"n_frames": n_frames}
