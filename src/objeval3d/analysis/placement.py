"""Where a free fit puts the cloud, compared with the reference placement.

A solution of the alignment is a similarity transform (scale, rotation, translation) from the world frame of the reconstruction to the
frame of the scan. Two solutions are compared by the rotation between them and by how far they move the cloud. The cloud enters
only through its centroid (both transforms are affine, so the centroid of the moved cloud is the transform of the centroid), which
is why the result files carry `recon_centroid` and the analysis needs neither the cloud nor the dataset.

The object's symmetries are respected: a placement that differs from the reference by a symmetry of the object is physically the
same placement. The rule of the paper uses one element of the symmetry group for both halves, the rotation (under 15 degrees) and
the centre of the cloud (under 0.10 of the object's diameter).
"""

from __future__ import annotations

import numpy as np

from objeval3d.modes import REFERENCE
from objeval3d.symmetry import symmetry_spec, symmetry_transforms

ROT_DEG, CENTRE_REL = 15.0, 0.10  # the paper's thresholds for "the search found the true placement"
N_CONTINUOUS = 24  # a continuous symmetry is discretised to this many elements


def object_symmetries(dataset: str, obj_id: int) -> list[np.ndarray]:
    return symmetry_transforms(symmetry_spec(dataset, obj_id), n_continuous=N_CONTINUOUS)


def gaps(record: dict, mode: str, syms: list[np.ndarray]) -> np.ndarray:
    """(rotation in degrees, centre shift in diameters) of `mode` against the reference, one row per symmetry element."""
    centroid = np.asarray(record["recon_centroid"], dtype=float)
    diameter = float(record["diameter_mm"])

    def placed(sim3: dict) -> np.ndarray:
        return sim3["scale"] * (np.asarray(sim3["R"]) @ centroid) + np.asarray(sim3["t"])

    a, ref = record["alignments"][mode]["sim3"], record["alignments"][REFERENCE]["sim3"]
    pa, pr = placed(a), placed(ref)
    Ra, Rr = np.asarray(a["R"]), np.asarray(ref["R"])
    out = []
    for S in syms:
        Rs, ts = S[:3, :3], S[:3, 3]
        ang = np.degrees(np.arccos(np.clip((np.trace(Ra @ (Rs @ Rr).T) - 1) / 2, -1, 1)))
        out.append((ang, np.linalg.norm(pa - (Rs @ pr + ts)) / diameter))
    return np.array(out)


def recovered(g: np.ndarray, rot_deg: float = ROT_DEG, centre_rel: float = CENTRE_REL) -> bool:
    """The search found the reference placement: one element of the symmetry group satisfies both halves of the rule."""
    return bool(((g[:, 0] < rot_deg) & (g[:, 1] < centre_rel)).any())


def dataset_of(set_id: str) -> str:
    return set_id.split("_")[0]
