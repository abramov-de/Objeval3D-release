"""Controls for the benchmark check: does the pipeline invent the effect?

Each control is a cloud that sits at the true pose by construction, built from the BOP model itself and scored by the very function that
scores the benchmark meshes (`objeval3d.benchmark.bias.evaluate_cloud`: the same placements, the same metrics). Nothing here touches
Burde's data.

  copy        the model sampled again with another seed. Zero shape error: the honest distance is the sampling floor, and no placement
              can do better than the truth, so every free mode must stay put and gain nothing. A gain here is an artefact of the
              implementation.
  noise S     the same plus independent Gaussian noise of S mm per point. Noise with no preferred direction: in expectation the truth is
              the minimiser, so the displacement and the gain should be small and should not grow into the millimetres the real meshes
              show.
  swell D     the surface pushed outward by D mm along its normals, the shell error of a volumetric reconstruction. A uniform offset is
              a scale error in disguise: the 7-parameter fit should take it, the rigid one should not move.
  bend A      a smooth non-rigid deformation of amplitude A mm (a sinusoidal bend along the object's long axis). Errors that no rigid
              pose can undo and that are not symmetric about the truth: this is what the bias of the criterion is made of.

    python -m objeval3d.benchmark.controls --objects 10,5,15 --out workspace/results/burde/controls.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from objeval3d.benchmark import bias as bb
from objeval3d.benchmark import data as bd
from objeval3d.data.bop import load_mesh, model_path, sample_mesh_surface
from objeval3d.paths import bop_root as default_bop_root
from objeval3d.symmetry import symmetry_transforms

CONTROLS = (
    ("copy", 0.0),
    ("noise", 0.5),
    ("noise", 1.0),
    ("noise", 2.0),
    ("swell", 1.0),
    ("swell", 2.0),
    ("bend", 1.0),
    ("bend", 3.0),
)


def make_cloud(kind: str, level: float, verts, faces, diameter: float, n: int, seed: int):
    pts, nrm = sample_mesh_surface(verts, faces, n, seed=seed, return_normals=True)
    rng = np.random.default_rng(seed + 100)
    if kind == "noise":
        pts = pts + rng.normal(scale=level, size=pts.shape)
    elif kind == "swell":
        pts = pts + level * nrm
    elif kind == "bend":
        u, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        axes = vt  # principal axes, long first
        s = (pts - pts.mean(0)) @ axes[0]
        span = float(np.ptp(s))
        pts = pts + level * np.sin(2 * np.pi * s / span)[:, None] * axes[1][None, :]
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description="Null and bias controls for the Burde check")
    ap.add_argument("--bop-root", default=str(default_bop_root("ycbv")))
    ap.add_argument("--objects", default="10,5,15")
    ap.add_argument("--modes", default="as_lies,pose_scale,auto_rigid,local,local_rigid")
    ap.add_argument("--n-points", type=int, default=60_000)
    ap.add_argument("--n-rot", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--gt-overlap", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=1, help="of the control clouds; the scan keeps seed 0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.cpu_count = lambda: a.threads
    args = SimpleNamespace(
        modes=a.modes.split(","), n_rot=a.n_rot, overlap=a.overlap, gt_overlap=a.gt_overlap, seed=0, n_points=a.n_points
    )

    results = []
    print(
        f"{'control':10s} {'obj':>3s} | {'acc rel':>7s} {'F@2% rel':>8s} | local: {'acc':>5s} {'rot':>5s} "
        f"{'cen/D':>6s} {'scl':>5s} {'<truth':>6s} | local_rigid: {'acc':>5s} {'rot':>5s} {'cen/D':>6s} {'<truth':>6s} | "
        f"auto_rigid: {'acc':>5s} {'rot':>6s} {'conv':>5s}"
    )
    for obj in (int(x) for x in a.objects.split(",")):
        scan = bd.scan_samples(Path(a.bop_root), obj)
        scan_tree = cKDTree(scan["gt_full"])
        syms = symmetry_transforms(scan["symmetries"], n_continuous=24)
        verts, faces = load_mesh(model_path(Path(a.bop_root), obj))
        for kind, level in CONTROLS:
            cloud = make_cloud(kind, level, verts, faces, scan["diameter"], a.n_points, a.seed)
            rec = {
                "control": kind,
                "level_mm": level,
                "obj_id": obj,
                "n_symmetries": len(syms),
                "diameter_mm": scan["diameter"],
            }
            bb.evaluate_cloud(rec, cloud, scan, scan_tree, syms, scan["gt_full"], args)
            results.append(rec)
            al, vs = rec["alignments"], rec["vs_as_lies"]
            lo, lr, ar = al["local"], al["local_rigid"], al.get("auto_rigid")

            def ch(m):
                return al[m]["chamfer_l1_mm"] < al["as_lies"]["chamfer_l1_mm"]

            print(
                f"{kind + ' ' + format(level, 'g'):10s} {obj:3d} | "
                f"{al['as_lies']['accuracy']['median']:7.2f} {al['as_lies']['fscore']['tau_2pct']['fscore']:8.3f} | "
                f"       {lo['accuracy']['median']:5.2f} {vs['local']['rotation_deg']:5.1f} "
                f"{vs['local']['placement_centroid_rel_diam']:6.3f} {vs['local']['scale_ratio']:5.3f} {str(ch('local')):>6s} | "
                f"             {lr['accuracy']['median']:5.2f} {vs['local_rigid']['rotation_deg']:5.1f} "
                f"{vs['local_rigid']['placement_centroid_rel_diam']:6.3f} {str(ch('local_rigid')):>6s} | "
                f"            "
                + (
                    f"{ar['accuracy']['median']:5.2f} {vs['auto_rigid']['rotation_deg']:6.1f} "
                    f"{str(vs['auto_rigid']['converged']):>5s}"
                    if ar
                    else ""
                ),
                flush=True,
            )
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
