"""Symmetries of the BOP objects and the object metadata (diameter, symmetry group) the analysis needs without the dataset."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import numpy as np

DATASETS = {"lm": "lmo", "lmo": "lmo", "ycbv": "ycbv"}


@lru_cache(maxsize=None)
def models_info(dataset: str) -> dict[int, dict]:
    """`models_info.json` of the BOP dataset (diameter in mm, bounding box, symmetries), as shipped with the package.

    Only these small metadata files are copied out of the BOP archives, so that the tables can be recomputed without the dataset.
    """
    name = DATASETS[dataset]
    text = resources.files("objeval3d.data").joinpath(f"models_info_{name}.json").read_text(encoding="utf-8")
    return {int(k): v for k, v in json.loads(text).items()}


def symmetry_spec(dataset: str, obj_id: int) -> dict:
    info = models_info(dataset)[obj_id]
    return {"discrete": info.get("symmetries_discrete", []), "continuous": info.get("symmetries_continuous", [])}


def symmetry_transforms(sym: dict, n_continuous: int = 36) -> list[np.ndarray]:
    """All 4x4 model-frame transforms that map the object onto itself.

    BOP gives discrete symmetries as explicit 4x4 matrices and continuous ones as (axis, offset); the latter are discretised, exactly
    as `bop_toolkit` does when it evaluates the pose error of rotationally symmetric objects.
    """
    out = [np.eye(4)]
    for m in sym.get("discrete", []) or []:
        out.append(np.asarray(m, dtype=np.float64).reshape(4, 4))

    for cont in sym.get("continuous", []) or []:
        axis = np.asarray(cont["axis"], dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        offset = np.asarray(cont.get("offset", [0, 0, 0]), dtype=np.float64)
        base = list(out)
        for k in range(1, n_continuous):
            ang = 2 * np.pi * k / n_continuous
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = offset - R @ offset
            for b in base:
                out.append(T @ b)
    return out
