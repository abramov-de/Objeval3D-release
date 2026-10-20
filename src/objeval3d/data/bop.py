"""Readers and geometry helpers for BOP-format datasets (LM-O, YCB-V, ...).

BOP layout assumed (as produced by unzipping the official archives):

    <bop_root>/
        camera.json                 (default intrinsics, unused here)
        models/
            models_info.json        per-object diameter + symmetries (mm)
            obj_000001.ply          triangle mesh, millimetres
        test/                       (or train_real/, val/, ...)
            000002/
                scene_camera.json   per-image cam_K, depth_scale
                scene_gt.json       per-image list of {obj_id, cam_R_m2c, cam_t_m2c}
                scene_gt_info.json  per-annotation bbox_visib, visib_fract, px_count_visib
                rgb/000000.png
                mask_visib/000000_000000.png     <im_id>_<gt_id>.png

Conventions used across the whole pipeline:
  * "model frame"  — the object's own coordinate frame, the frame the BOP mesh
    lives in. Millimetres. This is the frame all ground truth is expressed in.
  * cam_R_m2c / cam_t_m2c map model -> camera:  X_cam = R @ X_model + t
  * therefore the camera centre in the model frame is  C = -R^T @ t,
    and camera-to-model (c2w with world := model frame) is [R^T | -R^T t].
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from objeval3d.data.ply import read_cloud, read_mesh, write_cloud

# ----------------------------------------------------------------------------
# Object id -> human readable name.  Only used for file naming / reports; if a
# dataset is not listed the code falls back to "obj_000005".
# ----------------------------------------------------------------------------
OBJECT_NAMES: dict[str, dict[int, str]] = {
    "lm": {
        1: "ape",
        2: "benchvise",
        3: "bowl",
        4: "camera",
        5: "can",
        6: "cat",
        7: "cup",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    },
    "ycbv": {
        1: "master_chef_can",
        2: "cracker_box",
        3: "sugar_box",
        4: "tomato_soup_can",
        5: "mustard_bottle",
        6: "tuna_fish_can",
        7: "pudding_box",
        8: "gelatin_box",
        9: "potted_meat_can",
        10: "banana",
        11: "pitcher_base",
        12: "bleach_cleanser",
        13: "bowl",
        14: "mug",
        15: "power_drill",
        16: "wood_block",
        17: "scissors",
        18: "large_marker",
        19: "large_clamp",
        20: "extra_large_clamp",
        21: "foam_brick",
    },
}
OBJECT_NAMES["lmo"] = OBJECT_NAMES["lm"]

# LM-O annotates only these 8 of the 15 LM objects in its single test scene.
LMO_OBJ_IDS = [1, 5, 6, 8, 9, 10, 11, 12]

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def git_revision() -> str | None:
    """The commit this code is running from, for stamping into run records.

    Version skew between a laptop and the machine that runs the jobs is silent
    and expensive: a run completes, the numbers look ordinary, and only a missing
    field or a stale string in the output gives it away afterwards. Recording the
    revision makes the question answerable from the result itself. Returns None
    when that cannot be determined rather than guessing.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return rev + ("-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else "")
    except Exception:
        return None


DA3_PATCH = 14


def target_hw(image_path: Path, long_side: int, patch: int = DA3_PATCH) -> tuple[int, int]:
    """Replicates InputProcessor.upper_bound_resize + make_divisible_by_resize.

    Lives here rather than beside the trainer that first needed it because
    anything comparing masks to what a reconstruction saw has to resize them the
    same way, and a CPU-only checker should not import torch to find out how.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    scale = long_side / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    def nearest_mult(x: int) -> int:
        down = (x // patch) * patch
        up = down + patch
        return up if abs(up - x) <= abs(x - down) else down

    return max(patch, nearest_mult(nh)), max(patch, nearest_mult(nw))


# ---------------------------------------------------------------------------
# Mask loading and scoring. These live here rather than in `objeval3d.recon.reconstruct`
# because comparing masks needs neither torch nor the reconstruction model, and a CPU-only
# checker should not import a 340M parameter network to resize a PNG.
# ---------------------------------------------------------------------------


def gt_masks_for_set(vs: dict, bop_root: Path, hw: tuple[int, int], boxes=None) -> np.ndarray:
    """Ground-truth visible mask of the target object, resized to DA3's
    processing resolution (the same plain aspect-preserving resize DA3's
    InputProcessor applies to the RGB, so pixels stay in correspondence)."""
    scene = BopScene(bop_root / vs["split"] / f"{vs['scene_id']:06d}")
    h, w = hw
    out = np.zeros((len(vs["im_ids"]), h, w), dtype=bool)
    for i, (im_id, gt_id) in enumerate(zip(vs["im_ids"], vs["gt_ids"])):
        m = cv2.imread(str(scene.mask_visib_path(im_id, gt_id)), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if boxes is not None:
            x0, y0, x1, y1 = boxes[i]
            m = m[y0:y1, x0:x1]
        out[i] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    return out


def dir_masks_for_set(vs: dict, mask_dir: Path, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    out = np.zeros((len(vs["im_ids"]), h, w), dtype=bool)
    for i, im_id in enumerate(vs["im_ids"]):
        cands = [
            mask_dir / f"{vs['set_id']}_{i:02d}.png",
            mask_dir / f"{vs['scene_id']:06d}_{im_id:06d}.png",
            mask_dir / f"{im_id:06d}.png",
        ]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            print(f"  [warn] no external mask for view {i} (im_id={im_id})")
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out[i] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    return out


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    p_sum, g_sum = float(pred.sum()), float(gt.sum())
    return {
        "iou": inter / union if union > 0 else float("nan"),
        "dice": 2 * inter / (p_sum + g_sum) if (p_sum + g_sum) > 0 else float("nan"),
        "precision": inter / p_sum if p_sum > 0 else float("nan"),
        "recall": inter / g_sum if g_sum > 0 else float("nan"),
        "pred_px": int(p_sum),
        "gt_px": int(g_sum),
    }


def object_name(dataset: str, obj_id: int) -> str:
    return OBJECT_NAMES.get(dataset, {}).get(obj_id, f"obj_{obj_id:06d}")


# ----------------------------------------------------------------------------
# JSON readers.  BOP stores per-image dicts keyed by the image id as a *string*,
# which is a constant source of off-by-one/str-vs-int bugs — normalise to int.
# ----------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_scene_json(path: Path) -> dict[int, object]:
    return {int(k): v for k, v in _load_json(path).items()}


class BopScene:
    """One BOP scene directory (e.g. `<root>/test/000002`)."""

    def __init__(self, scene_dir: Path):
        self.dir = Path(scene_dir)
        self.scene_id = int(self.dir.name)
        self.camera = load_scene_json(self.dir / "scene_camera.json")
        self.gt = load_scene_json(self.dir / "scene_gt.json")
        gt_info_path = self.dir / "scene_gt_info.json"
        self.gt_info = load_scene_json(gt_info_path) if gt_info_path.exists() else {}
        self.rgb_dir = self.dir / "rgb" if (self.dir / "rgb").exists() else self.dir / "gray"
        self.mask_visib_dir = self.dir / "mask_visib"
        self.mask_dir = self.dir / "mask"

    # -- images -------------------------------------------------------------
    @property
    def im_ids(self) -> list[int]:
        return sorted(self.gt.keys())

    def rgb_path(self, im_id: int) -> Path:
        for ext in (".png", ".jpg"):
            p = self.rgb_dir / f"{im_id:06d}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No RGB image for im_id={im_id} in {self.rgb_dir}")

    def mask_visib_path(self, im_id: int, gt_id: int) -> Path:
        return self.mask_visib_dir / f"{im_id:06d}_{gt_id:06d}.png"

    def has_visib_masks(self) -> bool:
        return self.mask_visib_dir.exists() and any(self.mask_visib_dir.iterdir())

    # -- camera / poses -----------------------------------------------------
    def K(self, im_id: int) -> np.ndarray:
        return np.asarray(self.camera[im_id]["cam_K"], dtype=np.float64).reshape(3, 3)

    def annotations(self, im_id: int) -> list[dict]:
        """GT annotations of one image, each enriched with its gt_id and the
        matching entry from scene_gt_info.json (visib_fract etc.)."""
        out = []
        infos = self.gt_info.get(im_id, [])
        for gt_id, ann in enumerate(self.gt[im_id]):
            rec = {
                "gt_id": gt_id,
                "obj_id": int(ann["obj_id"]),
                "R_m2c": np.asarray(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                "t_m2c": np.asarray(ann["cam_t_m2c"], dtype=np.float64).reshape(3, 1),
            }
            if gt_id < len(infos):
                rec.update(
                    visib_fract=float(infos[gt_id].get("visib_fract", 1.0)),
                    px_count_visib=int(infos[gt_id].get("px_count_visib", -1)),
                    bbox_visib=infos[gt_id].get("bbox_visib"),
                )
            else:
                rec.update(visib_fract=1.0, px_count_visib=-1, bbox_visib=None)
            out.append(rec)
        return out

    def annotation_for(self, im_id: int, obj_id: int) -> dict | None:
        """First annotation of `obj_id` in `im_id` (BOP scenes may contain the
        same object twice; LM-O/YCB-V do not, so first is unambiguous there)."""
        for rec in self.annotations(im_id):
            if rec["obj_id"] == obj_id:
                return rec
        return None


def cam_center_in_model_frame(R_m2c: np.ndarray, t_m2c: np.ndarray) -> np.ndarray:
    """Camera centre expressed in the object's model frame (mm)."""
    return (-R_m2c.T @ t_m2c).reshape(3)


def cam_to_model(R_m2c: np.ndarray, t_m2c: np.ndarray) -> np.ndarray:
    """4x4 camera->model transform (i.e. camera pose with world := model frame)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_m2c.T
    T[:3, 3] = (-R_m2c.T @ t_m2c).reshape(3)
    return T


def viewing_direction(R_m2c: np.ndarray, t_m2c: np.ndarray) -> np.ndarray:
    """Unit vector from the object's origin towards the camera, in model frame.
    Used to measure the angular baseline between two views of the same object."""
    c = cam_center_in_model_frame(R_m2c, t_m2c)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def load_models_info(bop_root: Path) -> dict[int, dict]:
    info = _load_json(Path(bop_root) / "models" / "models_info.json")
    return {int(k): v for k, v in info.items()}


def model_path(bop_root: Path, obj_id: int, models_dir: str = "models") -> Path:
    return Path(bop_root) / models_dir / f"obj_{obj_id:06d}.ply"


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a BOP .ply mesh -> (vertices [V,3] float64 mm, faces [F,3] int64)."""
    return read_mesh(path)


def sample_mesh_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    seed: int = 0,
    return_normals: bool = False,
):
    """Area-weighted uniform sampling of `n_points` on the mesh surface.

    Raw mesh vertices are *not* uniformly distributed (scanners put many more
    vertices on high-curvature regions), which biases Chamfer-style metrics
    towards detailed areas.  Sampling the surface removes that bias.

    With `return_normals`, also returns the outward unit normal of the face
    each sample came from.  Winding is checked globally against the mesh
    centroid and flipped once if the file stores it inverted; individual faces
    are never flipped, since that would silently corrupt a correct mesh.
    """
    rng = np.random.default_rng(seed)
    if faces.size == 0:
        idx = rng.choice(len(verts), size=min(n_points, len(verts)), replace=False)
        return (verts[idx], None) if return_normals else verts[idx]

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = areas.sum()
    if total <= 0:
        pts = verts[: min(n_points, len(verts))]
        return (pts, None) if return_normals else pts

    fi = rng.choice(len(faces), size=n_points, p=areas / total)
    u = rng.random((n_points, 1))
    w = rng.random((n_points, 1))
    flip = (u + w) > 1.0
    u[flip] = 1.0 - u[flip]
    w[flip] = 1.0 - w[flip]
    pts = (v0[fi] + u * (v1[fi] - v0[fi]) + w * (v2[fi] - v0[fi])).astype(np.float64)
    if not return_normals:
        return pts

    nrm = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    nrm = orient_normals_outward(verts, faces, nrm)
    return pts, nrm[fi]


def orient_normals_outward(verts: np.ndarray, faces: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Make per-face normals point out of the solid.

    Preferred path: trust the file's winding.  A watertight, consistently wound
    mesh has every undirected edge traversed once in each direction, which is
    cheap to verify; the remaining sign ambiguity (inside-out winding) is fixed
    by the sign of the enclosed signed volume.  This is the only path that is
    correct for concave geometry — the inside wall of a mug or a bowl, both of
    which are in YCB-V.

    Fallback, when the winding is not consistent: orient each face away from
    the mesh centroid.  Exact for convex shapes, wrong on concave interiors, so
    it warns.
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    key = e[:, 0].astype(np.int64) * (verts.shape[0] + 1) + e[:, 1]
    rev = e[:, 1].astype(np.int64) * (verts.shape[0] + 1) + e[:, 0]
    consistent = float(np.isin(rev, key).mean())

    if consistent >= 0.95:
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum())
        return -nrm if vol < 0 else nrm

    print(
        f"  [warn] mesh winding is inconsistent ({100 * consistent:.0f}% of edges "
        f"paired); orienting normals from the centroid instead — visibility on "
        f"concave surfaces will be approximate"
    )
    face_c = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    sign = np.sign(np.einsum("ij,ij->i", nrm, face_c - verts.mean(0)))
    sign[sign == 0] = 1.0
    return nrm * sign[:, None]


# ----------------------------------------------------------------------------
# PLY point-cloud IO (colours optional), so that clouds open in MeshLab / CloudCompare.
# ----------------------------------------------------------------------------
def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    write_cloud(path, points, colors)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    return read_cloud(path)
