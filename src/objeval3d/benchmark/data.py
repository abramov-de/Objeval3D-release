"""Benchmark data of Burde et al. (WACV 2025), for the external check of the alignment result.

    Burde, Benbihi, Burget, Sattler. Comparative Evaluation of 3D Reconstruction Methods for Object Pose Estimation. arXiv:2408.08234.
    Files: data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/
    Code:  github.com/VarunBurde/reconstruction_pose_benchmark

What is taken from the release
------------------------------
* `reconstructed_meshes/<method>_<subset>.zip` - one zip per method and image count, holding the 21 YCB-V objects as
  `<NN>_<name>/mesh.obj`. Millimetres, in the object frame of the BOP YCB-V models (`models/obj_0000NN.ply`): checked per zip by the
  `check` command below, not assumed.
* `Image_dataset/<NN>_<name>.zip` - 2.3 GB each, of which only the camera files are read, by HTTP range requests
  (`nerf/transforms_<N>.json`, a few hundred KB). The images are never downloaded: this check has no masks, no depth model and no
  segmentation, only a mesh, a scan and a pose.

What the release does NOT contain: no `colmap_*` and no `capturereality_*` meshes (both 404; the index lists nine methods), and the
subsets 3...300 exist for six methods only - BakedSDF, MonoSDF, Nerfacto, NeuS, UniSURF, VolSDF. Neuralangelo, Instant-NGP and Plenoxels
are `_all` only. `index` prints the current listing.

How the "true pose" of these meshes came about (paper, Sec. 4 'Mesh Registration' and the appendix on image registration), because the
interpretation of the check hangs on it: camera extrinsics come from the robot's kinematic chain, are then REFINED (iNGP pose refinement,
then COLMAP bundle adjustment) and frozen. The transform between that world frame and the object frame of the BOP model is found ONCE per
object by ICP (rigid, no scale, user-defined coarse start) between a COLMAP reconstruction of the object and the BOP model, and then
applied to every method's mesh. So the pose is not fitted per mesh, which is what makes it an independent anchor, but it is not
independent of the scan either: it is the ICP minimum of one reconstruction against the very scan the distances are measured to.

Visible part of the scan
------------------------
Completeness is scored against the part of the scan that at least one camera of the subset can see, as everywhere else in the protocol.
Burde's cameras are known for every subset, so the same test as `objeval3d.metrics.gt_cloud` is applied: back-face culling plus a coarse
z-buffer. Their masks play no role; there is no other object in front of the target, only the table, which faces away from every camera.

    python -m objeval3d.benchmark.data index
    python -m objeval3d.benchmark.data fetch --methods nerfacto,volsdf --subsets 3,all
    python -m objeval3d.benchmark.data cameras
    python -m objeval3d.benchmark.data check
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from objeval3d.data.bop import load_mesh, load_models_info, model_path, sample_mesh_surface
from objeval3d.data.ply import read_mesh
from objeval3d.metrics.gt_cloud import visible_mask_one_view
from objeval3d.paths import bop_root as default_bop_root
from objeval3d.paths import burde_root as default_burde_root

BASE_URL = "https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh"
SUBSETS = ("3", "5", "10", "15", "20", "25", "50", "75", "100", "150", "300", "all")

# Burde's folder names, in BOP's order: the two digits are the BOP object id.
OBJECT_NAMES = {
    1: "01_master_chef_can",
    2: "02_cracker_box",
    3: "03_sugar_box",
    4: "04_tomatoe_soup_can",
    5: "05_mustard_bottle",
    6: "06_tuna_fish_can",
    7: "07_pudding_box",
    8: "08_gelatin_box",
    9: "09_potted_meat_can",
    10: "10_banana",
    11: "11_pitcher_base",
    12: "12_bleach_cleanser",
    13: "13_bowl",
    14: "14_mug",
    15: "15_power_drill",
    16: "16_wood_block",
    17: "17_scissors",
    18: "18_large_marker",
    19: "19_large_clamp",
    20: "20_extra_large_clamp",
    21: "21_foam_brick",
}

# Table 1 of their paper: objects whose manufacturer changed them since YCB-V was recorded, so that the physical object scanned here
# differs from the BOP model in texture and, by their own account, in scale by 4 % of the dimensions on average. The rest are 'legacy'.
# Used only as a cross-check of the scale screening, which measures the size mismatch instead of trusting this list.
UPDATED_OBJECTS = (1, 2, 5, 6, 7, 8, 9, 11, 15)


# ---------------------------------------------------------------------------
# What the server offers
# ---------------------------------------------------------------------------
def _head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def remote_index() -> dict[str, tuple[str, str]]:
    """Mesh zips on the server -> {stem: (method, subset)}, from the directory listing, so the table of what exists is read and not
    remembered."""
    with urllib.request.urlopen(f"{BASE_URL}/reconstructed_meshes/", timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    out = {}
    for name in re.findall(r'href="([^"?/]+\.zip)"', html):
        stem = name[:-4]
        method, _, subset = stem.rpartition("_")
        out[stem] = (method, subset)
    return out


def mesh_zip_path(root: Path, method: str, subset: str) -> Path:
    return Path(root) / "zips" / f"{method}_{subset}.zip"


def fetch_meshes(root: Path, methods, subsets, dry_run: bool = False, keep_free_gb: float = 50.0) -> list[str]:
    """Download `<method>_<subset>.zip` for each pair the server has. Sizes and free space are checked first; an interrupted download
    resumes."""
    idx = remote_index()
    want, missing = [], []
    for m in methods:
        for s in subsets:
            (want if f"{m}_{s}" in idx else missing).append(f"{m}_{s}")
    if missing:
        print(f"not on the server, skipped: {', '.join(missing)}")
    sizes = {n: _head_size(f"{BASE_URL}/reconstructed_meshes/{n}.zip") for n in want}
    todo = [
        n
        for n in want
        if not (mesh_zip_path(root, *idx[n]).exists() and mesh_zip_path(root, *idx[n]).stat().st_size == sizes[n])
    ]
    need = sum(sizes[n] for n in todo)
    root = Path(root)
    (root / "zips").mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    print(f"{len(want)} requested, {len(todo)} to download, {need / 1e6:.0f} MB; free {free / 1e9:.0f} GB on {root}")
    if need + keep_free_gb * 1e9 > free:
        raise SystemExit("not enough free space, refusing to download")
    if dry_run:
        return todo
    for n in todo:
        dst = mesh_zip_path(root, *idx[n])
        part = dst.with_suffix(".zip.part")
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "-C",
                "-",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "-o",
                str(part),
                f"{BASE_URL}/reconstructed_meshes/{n}.zip",
            ],
            check=True,
        )
        if part.stat().st_size != sizes[n]:
            raise SystemExit(f"{n}: got {part.stat().st_size} bytes, expected {sizes[n]}")
        part.rename(dst)
        print(f"  {n}: {sizes[n] / 1e6:.0f} MB ok")
    return todo


class RangeFile(io.RawIOBase):
    """A zip on an HTTP server, read by byte ranges: lets `zipfile` pull one small member out of a 2 GB archive without downloading the
    rest."""

    def __init__(self, url: str):
        self.url, self.size, self.pos = url, _head_size(url), 0
        self.n_bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        self.n_bytes += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def fetch_cameras(root: Path, obj_ids=None) -> None:
    """`nerf/transforms_<N>.json` of every subset, per object, into `<root>/cameras/<object>/`."""
    for i in obj_ids or sorted(OBJECT_NAMES):
        name = OBJECT_NAMES[i]
        out = Path(root) / "cameras" / name
        if out.exists() and len(list(out.glob("transforms_*.json"))) == len(SUBSETS):
            continue
        out.mkdir(parents=True, exist_ok=True)
        f = RangeFile(f"{BASE_URL}/Image_dataset/{name}.zip")
        z = zipfile.ZipFile(f)
        for m in z.namelist():
            if re.fullmatch(r"nerf/transforms_(\d+|all)\.json", m):
                (out / Path(m).name).write_bytes(z.read(m))
        print(
            f"  {name}: {len(list(out.glob('transforms_*.json')))} files, "
            f"{f.n_bytes / 1e3:.0f} KB read of {f.size / 1e9:.1f} GB"
        )


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
_MEMBER = re.compile(r"^[^/]+/(\d{2})_[^/]+/[^/]+\.(obj|ply)$")


def mesh_members(zpath: Path) -> dict[int, str]:
    """BOP object id -> member name of its mesh inside the zip. Where a folder holds several mesh files the one called `mesh.*` wins."""
    out: dict[int, str] = {}
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            m = _MEMBER.match(n)
            if not m:
                continue
            i = int(m.group(1))
            if i not in out or Path(n).stem == "mesh":
                out[i] = n
    return out


def parse_obj(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a Wavefront OBJ. Texture and normal indices are dropped, polygons are fan-triangulated, negative
    indices are honoured."""
    v, tri = [], []
    for line in data.split(b"\n"):
        if line.startswith(b"v "):
            p = line.split()
            v.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith(b"f "):
            idx = [int(t.split(b"/")[0]) for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                tri.append((idx[0], idx[k], idx[k + 1]))
    verts = np.asarray(v, dtype=np.float64)
    faces = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
    faces = np.where(faces > 0, faces - 1, faces + len(verts))
    return verts, faces


def load_burde_mesh(zpath: Path, obj_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    members = mesh_members(zpath)
    if obj_id not in members:
        raise KeyError(f"{Path(zpath).name} has no mesh for object {obj_id}")
    name = members[obj_id]
    with zipfile.ZipFile(zpath) as z:
        data = z.read(name)
    if name.endswith(".obj"):
        verts, faces = parse_obj(data)
    else:
        verts, faces = read_mesh(data)
    if faces.size:
        # Many of the released meshes keep every vertex the reconstruction produced and delete the triangles that fall outside the
        # authors' crop: the file of the three-image banana says 38983 vertices and 1469 triangles, of which 1072 vertices are
        # used. Surfaces are sampled from triangles, so the numbers never depended on the orphans, but the bounding box and the
        # vertex count did.
        used, inv = np.unique(faces, return_inverse=True)
        verts, faces = verts[used], inv.reshape(faces.shape).astype(np.int64)
    return verts, faces, name


def mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    if faces.size == 0:
        return 0.0
    c = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return float(0.5 * np.linalg.norm(c, axis=1).sum())


# ---------------------------------------------------------------------------
# Cameras and the scan
# ---------------------------------------------------------------------------
FLIP_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_cameras(root: Path, obj_id: int, subset: str) -> dict:
    """Cameras of one subset in BOP convention (model -> camera, millimetres).

    The conversion is `nerf_to_bop.py` of the authors: the NerfStudio pose is an OpenGL camera-to-world, flipped to OpenCV, its
    translation scaled by `real_world_scale` (to metres) and by 1000 (to millimetres), then inverted. World here IS the BOP object
    frame, as the release states.
    """
    p = Path(root) / "cameras" / OBJECT_NAMES[obj_id] / f"transforms_{subset}.json"
    d = json.loads(p.read_text())
    K = np.array([[d["fl_x"], 0, d["cx"]], [0, d["fl_y"], d["cy"]], [0, 0, 1.0]])
    Rs, ts = [], []
    for fr in d["frames"]:
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64) @ FLIP_GL_TO_CV
        c2w[:3, 3] *= d["real_world_scale"] * 1000.0
        w2c = np.linalg.inv(c2w)
        Rs.append(w2c[:3, :3])
        ts.append(w2c[:3, 3])
    return {
        "K": K,
        "hw": (int(d["h"]), int(d["w"])),
        "R_m2c": np.stack(Rs),
        "t_m2c": np.stack(ts),
        "n": len(Rs),
        "files": [fr["file_path"] for fr in d["frames"]],
    }


def scan_samples(bop_root: Path, obj_id: int, *, n_sample: int = 200_000, n_keep: int = 60_000, seed: int = 0) -> dict:
    """The scan as the protocol samples it: `build_gt_clouds` draws `n_sample` surface points with the seed, then `n_keep` of them for
    `gt_full`. Same calls in the same order, so `gt_full` is the array the published numbers were computed against."""
    info = load_models_info(bop_root)[obj_id]
    verts, faces = load_mesh(model_path(bop_root, obj_id))
    pts, nrm = sample_mesh_surface(verts, faces, n_sample, seed=seed, return_normals=True)
    rng = np.random.default_rng(seed)
    full_idx = np.arange(len(pts)) if len(pts) <= n_keep else rng.choice(len(pts), n_keep, replace=False)
    return {
        "pts": pts,
        "nrm": nrm,
        "gt_full": pts[full_idx],
        "rng": rng,
        "n_keep": n_keep,
        "diameter": float(info["diameter"]),
        "symmetries": {
            "discrete": info.get("symmetries_discrete", []),
            "continuous": info.get("symmetries_continuous", []),
        },
    }


def visible_part(scan: dict, cams: dict, *, z_eps_frac: float = 0.05, zbuf_scale: int = 4) -> tuple[np.ndarray, float]:
    """`gt_visible` for one camera set, and the fraction of the scan it covers.

    The cap on its size is drawn from a copy of the generator as `gt_full` left it, so every subset sees the draw a separate
    `build_gt_clouds` call would have made."""
    z_eps = max(0.5, z_eps_frac * scan["diameter"])
    vis = np.zeros(len(scan["pts"]), dtype=bool)
    for R, t in zip(cams["R_m2c"], cams["t_m2c"]):
        vis |= visible_mask_one_view(
            scan["pts"], scan["nrm"], R, t.reshape(3, 1), cams["K"], None, cams["hw"], z_eps, zbuf_scale=zbuf_scale
        )
    sel = scan["pts"][vis]
    rng = copy.deepcopy(scan["rng"])
    if len(sel) > scan["n_keep"]:
        sel = sel[rng.choice(len(sel), scan["n_keep"], replace=False)]
    return sel, float(vis.mean())


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _cmd_index(a):
    idx = remote_index()
    by: dict[str, list[str]] = {}
    for stem, (m, s) in idx.items():
        by.setdefault(m, []).append(s)
    for m in sorted(by):
        order = sorted(by[m], key=lambda s: (s == "all", int(s) if s != "all" else 0))
        print(f"{m:14s} {', '.join(order)}")
    print(f"\n{len(idx)} zips, {len(by)} methods")


def _cmd_fetch(a):
    methods = a.methods.split(",")
    subsets = a.subsets.split(",")
    fetch_meshes(Path(a.root), methods, subsets, dry_run=a.dry_run)


def _cmd_cameras(a):
    fetch_cameras(Path(a.root))


def _cmd_check(a):
    """Units and frame, per zip: if a method's meshes were in metres or in the unit cube, or in another frame, it shows here and
    nowhere later."""
    bop_root = Path(a.bop_root)
    from scipy.spatial import cKDTree

    zips = sorted((Path(a.root) / "zips").glob("*.zip"))
    print(f"{'zip':20s} {'objs':>4s} {'extent ratio':>13s} {'centre shift mm':>16s} {'median dist to scan, mm':>24s}")
    for zp in zips:
        members = mesh_members(zp)
        ratios, shifts, dists = [], [], []
        for i in sorted(members):
            verts, faces, _ = load_burde_mesh(zp, i)
            bv, bf = load_mesh(model_path(bop_root, i))
            ratios.append(float(np.median((verts.max(0) - verts.min(0)) / (bv.max(0) - bv.min(0)))))
            shifts.append(float(np.linalg.norm((verts.max(0) + verts.min(0)) / 2 - (bv.max(0) + bv.min(0)) / 2)))
            pts = sample_mesh_surface(verts, faces, 20_000, seed=0)
            scan = sample_mesh_surface(bv, bf, 40_000, seed=0)
            dists.append(float(np.median(cKDTree(scan).query(pts)[0])))
        print(
            f"{zp.stem:20s} {len(members):4d} {np.median(ratios):13.3f} "
            f"{np.median(shifts):16.1f} {np.median(dists):24.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Burde et al. benchmark data for the external check of the alignment result",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default=str(default_burde_root()))
    ap.add_argument("--bop-root", default=str(default_bop_root("ycbv")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").set_defaults(fn=_cmd_index)
    f = sub.add_parser("fetch")
    f.add_argument("--methods", required=True)
    f.add_argument("--subsets", default="3,all")
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(fn=_cmd_fetch)
    sub.add_parser("cameras").set_defaults(fn=_cmd_cameras)
    sub.add_parser("check").set_defaults(fn=_cmd_check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
