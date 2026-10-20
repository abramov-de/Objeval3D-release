"""Does free alignment reach a lower distance than the true pose on meshes that are not ours? The external check of the alignment result.

Burde et al. publish meshes of the 21 YCB-V objects, built by other methods from 3...480 images, that already sit in the object frame of
the BOP models, in millimetres (see `objeval3d.benchmark.data` for what that frame is and how it was obtained). So for each triple
(object, method, image count) the true placement is known without any depth model, mask or segmentation, and the scan the protocol
measures against is the BOP model, the same one as everywhere else.

For each triple the mesh is sampled to a cloud and placed six ways, each scored with the same functions that scored the BOP experiments
(`reconstruction_metrics`, symmetry resolution as in `objeval3d.align.evaluate`), so a row here is comparable with a row of the
alignment table sign for sign:

  as_lies     0 parameters. The mesh where the release put it. The honest shape error of a metric mesh: nothing was fitted to the scan
              for this triple.
  pose_scale  1 parameter. Pose as released, scale found by a one-dimensional sweep about the scan's centroid. The analogue of the
              reference mode, and the screen for substitute objects: a physical object whose size differs from the BOP model shows as
              a scale off 1.
  auto        7 parameters, global search, the engine of `objeval3d.align.registration`.
  auto_rigid  6 parameters (scale locked at 1, the meshes are metric). The field's practice for metric reconstructions and Burde's own
              kind of alignment.
  local       7 parameters, but started AT the true pose: refined by the same trimmed ICP. Separates the criterion from the search:
              where `auto` lands elsewhere, does descent from the truth still leave it?
  local_rigid the same with the scale locked.

Questions it answers, per mode against `as_lies`: is the Chamfer lower than at the true placement; how far did the placement move
(rotation modulo the object's symmetries, and the displacement of the cloud itself by `placement_gap_points`, the rule of the
paper); by what factor is the accuracy more optimistic; and the same under Burde's own metric (the fraction of points within
1 mm...2 cm of the other surface, no trimming).

The mesh is not cleaned, cropped or masked in any way beyond what the authors released. Whatever junk their cropping left is part of the
shape error. That is what makes the few-image meshes hard to read: a quarter to a half of a three-image mesh lies far from the object,
which our own pipeline would have removed with a mask. `--crop-from-all` is the control for it: the method's many-image mesh cut down
to what the few cameras see, a clean partial cloud of an accurate shape.

Cores: scipy's `workers=-1` takes `os.cpu_count()` and ignores affinity, so on a shared machine `--threads` replaces that count for this
process. Pin with `taskset` as well.

    python -m objeval3d.benchmark.bias --methods nerfacto --subsets 3,all --objects 10 --out workspace/results/burde/main
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from objeval3d.align.evaluate import placement_gap_points, resolve_symmetry, transform_discrepancy
from objeval3d.align.registration import (
    align_sim3,
    apply_sim3,
    reconstruction_metrics,
    subsample,
    trimmed_mean_nn,
)
from objeval3d.benchmark import data as bd
from objeval3d.data.bop import git_revision, sample_mesh_surface
from objeval3d.metrics.gt_cloud import visible_mask_one_view
from objeval3d.paths import bop_root as default_bop_root
from objeval3d.paths import burde_root as default_burde_root
from objeval3d.symmetry import symmetry_transforms

ALL_MODES = ("as_lies", "pose_scale", "auto", "auto_rigid", "local", "local_rigid")
THRESH_MM = (1.0, 2.0, 5.0, 10.0, 20.0)  # Burde's 1 mm ... 2 cm
CONVERGED_ROT_DEG, CONVERGED_CENTROID = 15.0, 0.10  # the rule of the paper
IDENT = (1.0, np.eye(3), np.zeros(3))


def criterion_mm(
    placed: np.ndarray, scan_tree: cKDTree, scan_pts: np.ndarray, overlap: float, gt_overlap: float
) -> dict:
    """The free search's own criterion, in millimetres, on full-size clouds:
    trimmed forward distance plus trimmed backward distance. Evaluated at the
    placement the search returned, before any symmetry element is applied."""
    fwd = trimmed_mean_nn(scan_tree, placed, overlap)
    bwd = trimmed_mean_nn(cKDTree(placed), scan_pts, gt_overlap)
    return {"fwd": fwd, "bwd": bwd, "total": fwd + bwd}


def burde_fractions(placed: np.ndarray, scan_full: np.ndarray, scan_vis: np.ndarray, scan_tree: cKDTree) -> dict:
    """The authors' own metric: the share of points within a threshold of the other
    surface, no trimming. `acc` is mesh -> scan; `comp_full` is scan -> mesh over
    the whole scan, as they define completeness; `comp_vis` restricts the scan to
    what the subset's cameras can see, as the protocol does everywhere."""
    d_acc = scan_tree.query(placed, k=1, workers=-1)[0]
    rt = cKDTree(placed)
    d_cf = rt.query(scan_full, k=1, workers=-1)[0]
    d_cv = rt.query(scan_vis, k=1, workers=-1)[0]
    return {
        f"tau_{t:g}mm": {
            "acc": float((d_acc < t).mean()),
            "comp_full": float((d_cf < t).mean()),
            "comp_vis": float((d_cv < t).mean()),
        }
        for t in THRESH_MM
    }


def fit_scale_about(
    recon: np.ndarray,
    scan_pts: np.ndarray,
    scan_tree: cKDTree,
    centre: np.ndarray,
    *,
    overlap: float,
    gt_overlap: float,
    n_probe: int = 8192,
    seed: int = 0,
) -> dict:
    """Best uniform scale of the mesh about `centre`, pose untouched.

    Two criteria, because they answer slightly different questions. `trimmed` is
    the search's own (the reference mode of the paper uses it); `median` is the median
    distance from the mesh to the scan, which junk in the mesh cannot move. A
    coarse geometric sweep over 0.8...1.25 is followed by a fine one around the
    minimum, which resolves the scale to ~0.1 %."""
    probe = subsample(recon, n_probe, seed + 2)
    gprobe = subsample(scan_pts, n_probe, seed)

    def moved(k):
        return centre + k * (probe - centre)

    def crit_trim(k):
        m = moved(k)
        return trimmed_mean_nn(scan_tree, m, overlap) + trimmed_mean_nn(cKDTree(m), gprobe, gt_overlap)

    def crit_med(k):
        return float(np.median(scan_tree.query(moved(k), k=1, workers=-1)[0]))

    out = {}
    for name, fn in (("trimmed", crit_trim), ("median", crit_med)):
        ks = np.geomspace(0.8, 1.25, 91)
        k1 = ks[int(np.argmin([fn(k) for k in ks]))]
        kf = np.geomspace(k1 / 1.015, k1 * 1.015, 31)
        vals = [fn(k) for k in kf]
        out[f"k_{name}"] = float(kf[int(np.argmin(vals))])
        out[f"at_k1_{name}"] = fn(1.0)
        out[f"at_best_{name}"] = float(min(vals))
    return out


def crop_to_cameras(
    verts: np.ndarray, faces: np.ndarray, cams: dict, diameter: float, n_points: int, seed: int
) -> tuple[np.ndarray, float]:
    """The part of a mesh that the cameras of one subset can see, as a cloud.

    Sampled densely with outward normals, then each point is kept if at least one
    camera sees it by the test that decides the scan's visible part (the sample's
    normal faces the camera, a coarse z-buffer over the mesh's own samples). What
    remains is a partial, clean cloud with exactly the shape of the donor mesh, so
    coverage is separated from the junk that raw few-image meshes carry."""
    pts, nrm = sample_mesh_surface(verts, faces, 4 * n_points, seed=seed, return_normals=True)
    z_eps = max(0.5, 0.05 * diameter)
    vis = np.zeros(len(pts), dtype=bool)
    for R, t in zip(cams["R_m2c"], cams["t_m2c"]):
        vis |= visible_mask_one_view(pts, nrm, R, t.reshape(3, 1), cams["K"], None, cams["hw"], z_eps)
    sel = pts[vis]
    if len(sel) > n_points:
        sel = sel[np.random.default_rng(seed).choice(len(sel), n_points, replace=False)]
    return sel, float(vis.mean())


def run_triple(
    method: str,
    subset: str,
    obj: int,
    zpath: Path,
    scan: dict,
    scan_tree: cKDTree,
    syms: list,
    scan_vis: np.ndarray,
    vis_frac: float,
    cams: dict,
    args,
    crop: bool = False,
) -> dict:
    t_all = time.time()
    verts, faces, member = bd.load_burde_mesh(zpath, obj)
    kept = None
    if crop:
        recon, kept = crop_to_cameras(verts, faces, cams, scan["diameter"], args.n_points, args.seed)
    else:
        recon = sample_mesh_surface(verts, faces, args.n_points, seed=args.seed)
    rec = {
        "key": f"{method}_{subset}_{obj:02d}",
        "status": "ok",
        "method": method,
        "subset": subset,
        "obj_id": obj,
        "obj_name": bd.OBJECT_NAMES[obj].split("_", 1)[1],
        "updated_object": obj in bd.UPDATED_OBJECTS,
        "diameter_mm": scan["diameter"],
        "n_symmetries": len(syms),
        "mesh": {
            "member": member,
            "n_verts": int(len(verts)),
            "n_faces": int(len(faces)),
            "area_mm2": bd.mesh_area(verts, faces),
            "extent_mm": (verts.max(0) - verts.min(0)).tolist(),
        },
        "views": {
            "n_cameras": int(cams["n"]),
            "visible_fraction": vis_frac,
            "mean_camera_distance_mm": float(np.linalg.norm(cams["t_m2c"], axis=1).mean()),
        },
    }
    if crop:
        rec["crop"] = {"donor_zip": Path(zpath).name, "mesh_fraction_seen": kept}
    evaluate_cloud(rec, recon, scan, scan_tree, syms, scan_vis, args)
    rec["seconds"]["total"] = time.time() - t_all
    return rec


def evaluate_cloud(
    rec: dict, recon: np.ndarray, scan: dict, scan_tree: cKDTree, syms: list, scan_vis: np.ndarray, args
) -> dict:
    """Place `recon` (a cloud already in the object frame) every requested way and
    score each placement; fills `rec` in place. Split from `run_triple` so that
    `objeval3d.benchmark.controls` can feed it clouds that are not Burde's meshes."""
    diameter = scan["diameter"]
    scan_full = scan["gt_full"]
    centre = scan_full.mean(0)
    rec.update(
        {
            "n_recon": int(len(recon)),
            "n_scan_full": int(len(scan_full)),
            "n_scan_visible": int(len(scan_vis)),
            "alignments": {},
            "vs_as_lies": {},
            "seconds": {},
        }
    )

    def place(mode: str):
        t0 = time.time()
        if mode == "as_lies":
            res = {"scale": 1.0, "R": np.eye(3), "t": np.zeros(3)}
        elif mode == "pose_scale":
            fit = fit_scale_about(
                recon, scan_full, scan_tree, centre, overlap=args.overlap, gt_overlap=args.gt_overlap, seed=args.seed
            )
            rec["pose_scale"] = fit
            k = fit["k_trimmed"]
            res = {"scale": k, "R": np.eye(3), "t": centre * (1.0 - k)}
        else:
            kw = dict(n_rot=args.n_rot, overlap=args.overlap, gt_overlap=args.gt_overlap, seed=args.seed)
            if mode.endswith("_rigid"):
                kw.update(fixed_world_scale=1.0, refit_after_clip=True)
            if mode.startswith("local"):
                kw["init"] = IDENT
            res = align_sim3(recon, scan_full, **kw)
        rec["seconds"][mode] = time.time() - t0
        return res

    placements = {m: place(m) for m in ("as_lies",) + tuple(m for m in args.modes if m != "as_lies")}

    for mode, res in placements.items():
        placed = apply_sim3(recon, res["scale"], res["R"], res["t"])
        S, s_idx = resolve_symmetry(placed, scan_vis, syms, seed=args.seed)
        placed_s = (S[:3, :3] @ placed.T).T + S[:3, 3]
        m = reconstruction_metrics(placed_s, scan_full, scan_vis, diameter)
        m["symmetry_index"] = int(s_idx)
        m["sim3"] = {
            "scale": float(res["scale"]),
            "R": np.asarray(res["R"]).tolist(),
            "t": np.asarray(res["t"]).tolist(),
        }
        m["burde_metric"] = burde_fractions(placed_s, scan_full, scan_vis, scan_tree)
        m["criterion_mm"] = criterion_mm(placed, scan_tree, scan_full, args.overlap, args.gt_overlap)
        if "score_normalised" in res:
            m["search"] = {k: res[k] for k in ("score_normalised", "n_hypotheses", "used_init") if k in res}
        rec["alignments"][mode] = m
        if mode == "as_lies":
            d_acc = scan_tree.query(placed, k=1, workers=-1)[0]
            rec["junk"] = {
                "frac_beyond_5pct_diam": float((d_acc > 0.05 * diameter).mean()),
                "frac_beyond_10pct_diam": float((d_acc > 0.10 * diameter).mean()),
                "max_mm": float(d_acc.max()),
            }

    ref = rec["alignments"]["as_lies"]
    for mode, res in placements.items():
        if mode == "as_lies":
            continue
        a = rec["alignments"][mode]
        d = transform_discrepancy((res["scale"], res["R"], res["t"]), IDENT, syms, diameter)
        cen, rms = placement_gap_points(recon, a["sim3"], ref["sim3"], diameter)
        rec["vs_as_lies"][mode] = {
            "rotation_deg": d["rotation_deg"],
            "scale_ratio": d["scale_ratio"],
            "placement_centroid_rel_diam": cen,
            "placement_rms_rel_diam": rms,
            "converged": bool(d["rotation_deg"] < CONVERGED_ROT_DEG and cen < CONVERGED_CENTROID),
            "delta_chamfer_mm": a["chamfer_l1_mm"] - ref["chamfer_l1_mm"],
            "delta_criterion_mm": a["criterion_mm"]["total"] - ref["criterion_mm"]["total"],
        }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Free alignment against the true pose on Burde meshes")
    ap.add_argument("--root", default=str(default_burde_root()))
    ap.add_argument("--bop-root", default=str(default_bop_root("ycbv")))
    ap.add_argument("--methods", required=True, help="comma-separated, e.g. nerfacto,volsdf")
    ap.add_argument("--subsets", default="3,all")
    ap.add_argument("--objects", default="all", help="BOP ids, comma-separated, or `all`")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--out", required=True, help="path stem; shards add .shardIofN.jsonl")
    ap.add_argument("--shard", default="0/1", help="I/N: take objects I, I+N, ...")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-points", type=int, default=60_000, help="surface samples per mesh")
    ap.add_argument("--n-rot", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--gt-overlap", type=float, default=0.45)
    ap.add_argument("--n-continuous-sym", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N triples (smoke test)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--crop-from-all",
        action="store_true",
        help="for every subset but `all`, do not read that subset's mesh: take the "
        "method's `all` mesh and keep the part its cameras see (see "
        "crop_to_cameras). A partial, clean cloud with an accurate shape",
    )
    args = ap.parse_args()
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in args.modes:
        if m not in ALL_MODES:
            raise SystemExit(f"unknown mode {m}; choose from {', '.join(ALL_MODES)}")

    os.cpu_count = lambda: args.threads  # what scipy's workers=-1 reads, see the module docstring

    methods = args.methods.split(",")
    subsets = args.subsets.split(",")
    objects = sorted(bd.OBJECT_NAMES) if args.objects == "all" else [int(x) for x in args.objects.split(",")]
    si, sn = (int(x) for x in args.shard.split("/"))
    objects = objects[si::sn]
    out = Path(f"{args.out}.shard{si}of{sn}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists() and not args.no_resume:
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and set(args.modes) <= set(r.get("alignments", {})):
                done.add(r["key"])
    print(
        f"{len(objects)} objects x {len(methods)} methods x {len(subsets)} subsets, "
        f"modes {','.join(args.modes)}; {len(done)} already done -> {out}",
        flush=True,
    )

    root, bop_root = Path(args.root), Path(args.bop_root)
    rev = git_revision()
    n_done = 0
    with open(out, "a") as fh:
        for obj in objects:
            scan = bd.scan_samples(bop_root, obj, seed=args.seed)
            scan_tree = cKDTree(scan["gt_full"])
            syms = symmetry_transforms(scan["symmetries"], n_continuous=args.n_continuous_sym)
            for subset in subsets:
                crop = args.crop_from_all and subset != "all"

                def src(m):
                    return bd.mesh_zip_path(root, m, "all" if crop else subset)

                todo = [m for m in methods if f"{m}_{subset}_{obj:02d}" not in done and src(m).exists()]
                if not todo:
                    continue
                t0 = time.time()
                cams = bd.load_cameras(root, obj, subset)
                scan_vis, vis_frac = bd.visible_part(scan, cams)
                print(
                    f"[obj {obj:2d} {subset:>3s}] {cams['n']} cameras see "
                    f"{100 * vis_frac:.0f}% of the scan ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                for method in todo:
                    try:
                        rec = run_triple(
                            method,
                            subset,
                            obj,
                            src(method),
                            scan,
                            scan_tree,
                            syms,
                            scan_vis,
                            vis_frac,
                            cams,
                            args,
                            crop=crop,
                        )
                    except Exception as e:  # one bad mesh must not lose the batch
                        rec = {
                            "key": f"{method}_{subset}_{obj:02d}",
                            "method": method,
                            "subset": subset,
                            "obj_id": obj,
                            "status": f"error: {type(e).__name__}: {e}",
                        }
                    rec["git_revision"] = rev
                    rec["params"] = {
                        k: getattr(args, k) for k in ("n_points", "n_rot", "overlap", "gt_overlap", "seed")
                    }
                    fh.write(json.dumps(rec, default=float) + "\n")
                    fh.flush()
                    n_done += 1
                    if rec["status"] == "ok":
                        a, f = rec["alignments"]["as_lies"], rec["alignments"].get("auto")
                        print(
                            f"  {method:10s} acc {a['accuracy']['median']:6.2f} mm  "
                            f"F@2% {a['fscore']['tau_2pct']['fscore']:.3f}"
                            + (
                                f"  | auto acc {f['accuracy']['median']:6.2f}  "
                                f"rot {rec['vs_as_lies']['auto']['rotation_deg']:5.1f} deg  "
                                f"conv {rec['vs_as_lies']['auto']['converged']}"
                                if f
                                else ""
                            )
                            + f"  [{rec['seconds']['total']:.0f}s]",
                            flush=True,
                        )
                    else:
                        print(f"  {method:10s} {rec['status']}", flush=True)
                    if args.limit and n_done >= args.limit:
                        print("limit reached")
                        return
    print(f"done, {n_done} triples written to {out}")


if __name__ == "__main__":
    main()
