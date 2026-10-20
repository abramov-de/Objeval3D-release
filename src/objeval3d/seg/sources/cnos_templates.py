"""Render CNOS's onboarding templates for a BOP dataset.

CNOS ships `src/poses/pyrender.py` for exactly this, and its `render()` - camera, lights, poses, EGL, file naming - does the work here,
imported and called unmodified. What this module replaces is that script's `__main__` block, for one reason:

    src/poses/pyrender.py:104   diameter = get_obj_diameter(mesh)
    src/utils/trimesh_utils.py  def get_obj_diameter(mesh_path):
                                    mesh = load_mesh(mesh_path)   # trimesh.load(path)

The function takes a PATH and is handed a loaded mesh. Older trimesh returned a Geometry passed to `load()` straight back, so the mismatch
was invisible; trimesh 4.x raises `NotImplementedError: file_type 'None' not supported`, and every object fails. Their BOP runs use
BlenderProc-rendered templates, so this path sees little traffic upstream. Pinning an old trimesh was the alternative; their
environment.yml does not pin trimesh at all (it arrives as a dependency of pyrender), so there is no "authors' version" to be faithful
to, and an old pin would drag numpy and pyrender back with it for the sake of one argument.

**This does not make the comparison ours.** Nothing here touches what CNOS *is*: SAM proposals, DINOv2 descriptors, cosine matching
with avg_5 aggregation and the confidence threshold all run from the authors' code in `objeval3d.seg.sources.cnos`. This module loads
a mesh, decides millimetres from metres, and hands the result to their renderer, making the same decisions their `__main__` does,
from the same values:

    mesh          trimesh.load_mesh(cad_path)
    re-centre     -mesh.bounding_box.centroid            (exactly [0,0,0] for BOP models, which are bounding-box-centred)
    scale         diameter > 100 -> apply_scale(0.001)   diameter now from the path
    intrinsics    the LM-O matrix at 480x640, which their script uses for every dataset except T-LESS

T-LESS is the one case their `__main__` treats differently (it paints the mesh a uniform grey, since T-LESS models carry no useful
colour). That branch is not reproduced here and this refuses to run on it rather than rendering something subtly different from what the
authors would have.

Template NAMES follow the level-2 numbering, which is not optional: `BOPTemplate` selects level 0 as 42 indices into the level-2 list
(0, 46, 47, ...), so files named 000000..000041 make it ask for 000046.png and die. Rendering all 642 level-2 poses would name them
correctly and then read 42 of them. Instead the 42 selected level-2 poses are rendered in order and renamed to their level-2 indices,
which is what the full render would have written at those names.

Offscreen rendering needs EGL on a headless machine (`pip install 'pyopengl==3.1.0'`: 3.1.7 breaks EGL device enumeration; the library
must come from the NVIDIA driver). OSMesa is the software fallback: `conda install -c conda-forge osmesa`,
`export PYOPENGL_PLATFORM=osmesa`; it is slower, but 882 renders is a small job.

    PYTHONPATH=~/cnos python -m objeval3d.seg.sources.cnos_templates --cnos-root ~/cnos \\
        --models workspace/data/bop/ycbv/models --out-dir workspace/masks/cnos_templates/ycbv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Their pyrender module sets PYOPENGL_PLATFORM and DISPLAY at import, so the offscreen setup is theirs too.
INTRINSIC = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
IMG_SIZE = (480, 640)


def check_templates(obj_dir: Path, idx) -> tuple[int, float]:
    """Refuse a directory of blank renders, whatever wrote it.

    BOPTemplate takes each template's crop box from `Image.getbbox()`, which is None when the image is empty; a directory of blank PNGs
    therefore fails later, inside torch, as "can't convert np.ndarray of type numpy.object_" - a message that says nothing about
    rendering. Checking here also catches a STALE directory that an earlier, wrong run marked done.
    """
    from PIL import Image

    empty, coverage = 0, []
    for j in idx:
        with Image.open(obj_dir / f"{j:06d}.png") as im:
            box = im.getbbox()
        if box is None:
            empty += 1
        else:
            x0, y0, x1, y1 = box
            coverage.append((x1 - x0) * (y1 - y0) / (IMG_SIZE[0] * IMG_SIZE[1]))
    if empty:
        raise SystemExit(
            f"{obj_dir}: {empty} of {len(idx)} templates are blank.\n"
            f"The object is being rendered outside the frame or too far away - check the pose units (render() wants metres) - or "
            f"these templates are left over from an earlier run: delete {obj_dir.parent} and render again."
        )
    return empty, float(np.mean(coverage))


def main() -> None:
    ap = argparse.ArgumentParser(description="CNOS onboarding templates for a BOP dataset")
    ap.add_argument("--cnos-root", required=True)
    ap.add_argument("--models", required=True, help="BOP models dir holding obj_*.ply")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--level", type=int, default=0, help="0 = the 42 templates CNOS uses by default")
    ap.add_argument("--obj-ids", default="", help="comma separated subset")
    ap.add_argument("--force", action="store_true", help="re-render objects already marked done")
    args = ap.parse_args()

    cnos_root = Path(args.cnos_root).expanduser().resolve()
    models = Path(args.models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "tless" in str(models).lower():
        raise SystemExit(
            "T-LESS needs the uniform-grey branch of the authors' own __main__, which this module does not reproduce - render it "
            "with their script."
        )

    sys.path.insert(0, str(cnos_root))
    try:
        import pyrender  # noqa: F401  (imported for from_trimesh below)
        import trimesh
        from src.poses.pyrender import render
        from src.utils.trimesh_utils import as_mesh, get_obj_diameter
    except ImportError as e:
        raise SystemExit(f"Could not import CNOS or its renderer from {cnos_root}: {e}")

    pre = cnos_root / "src/poses/predefined_poses"
    idx = np.load(pre / f"idx_all_level{args.level}_in_level2.npy")
    poses = np.load(pre / "obj_poses_level2.npy")[idx].copy()
    # 0.4 is the zoom their own template script applies before handing poses to pyrender, which then divides the translation by 1000
    # (mm -> m).
    poses[:, :3, 3] *= 0.4
    # Saved in millimetres, byte for byte what their own template script writes, so the file stays comparable with a render done
    # their way.
    np.save(out_dir / f"template_poses_level{args.level}.npy", poses)
    np.save(out_dir / f"template_index_level{args.level}.npy", idx)
    # The millimetres-to-metres conversion lives in their `__main__`, NOT in `render()`. Replacing `__main__` drops it, and the object
    # is then placed 400 m from the camera instead of 0.4: every template comes out empty, and the failure surfaces two steps later as
    # a dtype error inside BOPTemplate. Hence check_templates() below - a blank render must be caught where it happens.
    render_poses = poses.copy()
    render_poses[:, :3, 3] /= 1000.0
    print(f"{len(idx)} template poses at level {args.level}, object at {np.linalg.norm(render_poses[0, :3, 3]):.3f} m")

    plys = sorted(models.glob("obj_*.ply"))
    if not plys:
        raise SystemExit(f"no obj_*.ply under {models}")
    if args.obj_ids:
        wanted = {int(x) for x in args.obj_ids.split(",") if x.strip()}
        plys = [p for p in plys if int(p.stem[4:10]) in wanted]

    t_start = time.time()
    for n, ply in enumerate(plys, 1):
        obj_dir = out_dir / ply.stem
        if (obj_dir / ".done").exists() and not args.force:
            _, cov = check_templates(obj_dir, idx)
            print(f"[{n}/{len(plys)}] {ply.stem}: already rendered, object fills {100 * cov:.1f}% of the frame")
            continue
        if obj_dir.exists():
            for f in obj_dir.iterdir():
                f.unlink()
        obj_dir.mkdir(parents=True, exist_ok=True)

        mesh = trimesh.load_mesh(ply)
        re_center = np.eye(4)
        re_center[:3, 3] = -mesh.bounding_box.centroid
        # Their line, with the argument the function actually wants. Note that the re-centre offset above is read in millimetres
        # while the mesh is rescaled to metres below - an inconsistency in their code that costs nothing on BOP, where the centroid
        # is exactly zero, and that is left alone rather than "fixed" into a different render.
        diameter = get_obj_diameter(str(ply))
        if diameter > 100:  # the object is in millimetres
            mesh.apply_scale(0.001)

        t0 = time.time()
        render(
            mesh=pyrender.Mesh.from_trimesh(as_mesh(mesh)),
            output_dir=str(obj_dir),
            obj_poses=render_poses,
            img_size=IMG_SIZE,
            intrinsic=INTRINSIC,
            re_center_transform=re_center,
        )

        # Two passes through a temporary name: the sequential names and the level-2 names overlap (000000 is both), so renaming in
        # place would overwrite a file that has not been moved yet.
        for i, j in enumerate(idx):
            os.rename(obj_dir / f"{i:06d}.png", obj_dir / f"tmp_{j:06d}.png")
        for j in idx:
            os.rename(obj_dir / f"tmp_{j:06d}.png", obj_dir / f"{j:06d}.png")
        written = len(list(obj_dir.glob("*.png")))
        if written != len(idx):
            raise SystemExit(f"{obj_dir} holds {written} templates, expected {len(idx)}")
        _, cov = check_templates(obj_dir, idx)
        (obj_dir / ".done").touch()
        print(
            f"[{n}/{len(plys)}] {ply.stem}: {written} templates, "
            f"diameter {diameter:.0f} -> {'m' if diameter > 100 else 'unchanged'}, "
            f"fills {100 * cov:.1f}% of the frame, {time.time() - t0:.1f}s"
        )

    print(f"\n{len(plys)} object(s) in {time.time() - t_start:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
