"""Reading a view-set manifest on a machine other than the one that wrote it.

A manifest (`configs/manifests/*.json`) lists the view sets of an experiment: for each, the dataset, scene, object, frames, camera
intrinsics and ground-truth poses. It also stores the location of the BOP dataset and of every image as they were on the machine
that built it. `load_manifest` replaces those two with the reader's own, so the manifests that come with the paper work unchanged:

    1. `bop_root=` argument, or `--bop-root` on the command line (the root of ONE dataset, e.g. .../bop/lmo);
    2. `$OBJEVAL_BOP_ROOT/<dataset>`, where `$OBJEVAL_BOP_ROOT` holds `lmo/` and `ycbv/`;
    3. the path in the manifest.

The image paths are rebuilt from the root, the split, the scene and the frame number. Nothing else in the manifest is touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_bop_root(manifest_root: str, dataset: str, override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("OBJEVAL_BOP_ROOT")
    if env:
        return Path(env) / dataset
    return Path(manifest_root)


def rgb_path(bop_root: Path, split: str, scene_id: int, im_id: int) -> Path:
    scene = Path(bop_root) / split / f"{scene_id:06d}"
    for sub in ("rgb", "gray"):
        for ext in (".png", ".jpg"):
            p = scene / sub / f"{im_id:06d}{ext}"
            if p.exists():
                return p
    return scene / "rgb" / f"{im_id:06d}.png"  # the usual place, for the error message of whoever opens it


def load_manifest(path: str | os.PathLike, bop_root: str | os.PathLike | None = None) -> dict:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    root = resolve_bop_root(man.get("bop_root", ""), man.get("dataset", ""), bop_root)
    man["bop_root"] = str(root)
    for s in man["sets"]:
        s["images"] = [str(rgb_path(root, s["split"], s["scene_id"], i)) for i in s["im_ids"]]
    return man
