"""Stages 3-4 of the protocol: align the cleaned object cloud to the scanned ground truth and compute the reconstruction-quality
metrics (accuracy, completeness, F-score).

Alignment modes, by what each is allowed to look at (paper term, then the identifier the result files keep):

  reference mode, ``oracle_raw`` - the ground-truth object pose, leaving only the depth scale free. One degree of freedom
           against the seven of a free search. This is the measuring instrument of the paper: it is the only mode whose error
           is the reconstruction's rather than a sum of two errors that cannot be told apart. ``oracle`` is the same solution
           with a local trimmed ICP on top (the free fit started at the ground-truth pose).

  free fit, ``auto`` - the deterministic Sim(3) search on the two point clouds alone, the usual way of aligning a reconstruction
           to a scan. It is reported as a result about the practice, not proposed: it reaches a *lower* criterion value than the
           true pose, so its numbers understate the error of the cloud.

  ``auto_fixedscale`` - the same search with the scale locked to the one the reference mode found: a metric reconstruction would
           have it. It separates what the scale contributes to the bias from what rotation and translation do.

  ``fpfh_ransac``, ``fgr``, ``icp_pca``, ``teaserpp`` - third-party implementations of the same free search
           (`objeval3d.align.baselines`), each also with a ``_rigid`` suffix for the six-parameter variant with the scale
           locked. Everything downstream (criterion, symmetry resolution, metrics) is shared with ``auto``, so the rows are
           comparable sign for sign.

Object symmetries are resolved the way bop_toolkit resolves them for pose error: a pose that differs by a symmetry of the model
is indistinguishable from the truth, so the best-matching element of the symmetry group is used.

The result file is one JSON per call, with one record per view set (`alignments` per mode, `oracle_diagnostics`,
`vs_oracle`, `recon_centroid`, ...); `objeval3d.analysis` reads it. A run folder is what `objeval3d.recon.reconstruct` writes.

    python -m objeval3d.align.evaluate --runs-root RUNS --manifest M.json --out results.json --modes oracle_raw,auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from objeval3d.align import baselines as reg_baselines
from objeval3d.align.registration import align_sim3, apply_sim3, reconstruction_metrics
from objeval3d.data.bop import git_revision, read_ply, write_ply
from objeval3d.data.manifest import load_manifest
from objeval3d.metrics.gt_cloud import build_gt_clouds
from objeval3d.symmetry import symmetry_transforms
from objeval3d.threads import workers


def oracle_object_sim3(
    E_da3,
    R_m2c,
    t_m2c,
    gt_full,
    recon,
    k_init,
    view_ids=None,
    n_k: int = 61,
    k_lo: float = 0.4,
    k_hi: float = 2.5,
    overlap: float = 0.8,
    n_probe: int = 4096,
    seed: int = 0,
):
    """Sim(3) from the ground-truth *object* pose, leaving only the scale free (the reference mode).

    Anchoring on the camera centres (matching the predicted camera positions to the true ones) is the wrong reference for an
    object-level protocol: it is sensitive to the pose error over the whole scene, and the ICP refinement that follows drags the
    result away from the geometry it was supposed to pin down.

    The object's own ground-truth pose fixes everything except the one quantity a scale-free reconstruction genuinely cannot know.
    With `x_cam = k * x_cam_pred` for an unknown depth scale k, and the predicted extrinsic (R_E, t_E) mapping world to camera,

        x_model = k * (R_m2c^T R_E) x_world + (k * R_m2c^T t_E - R_m2c^T t_m2c)

    - a one-parameter family. k is found by a deterministic 1-D sweep scored by trimmed one-way Chamfer distance, so the reference
    concedes exactly one degree of freedom against the seven the free fit gets.

    Each view yields its own transform; how far they disagree is a direct measure of the relative pose error of the reconstruction,
    and is returned alongside the best.

    Two scores are reported per view. `score_own` uses only the points that view contributed: unprojecting a view's depth through
    its own extrinsic cancels that extrinsic exactly, so this number depends on the view's *depth* alone. `score_all` uses every
    point, and the gap between the two is what the multi-view registration costs. Keeping them apart matters because at LM-O's
    working distance an 8 deg pose error moves the object by 120 mm - more than the object is wide - and would otherwise be
    indistinguishable from bad depth.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(recon)) if len(recon) <= n_probe else rng.choice(len(recon), n_probe, False)
    probe = recon[idx]
    probe_ids = None if view_ids is None else np.asarray(view_ids)[idx]
    tree = cKDTree(gt_full)
    keep = max(3, int(round(overlap * len(probe))))

    per_view = []
    for i in range(len(E_da3)):
        Eh = np.eye(4)
        Eh[:3, :4] = E_da3[i][:3, :4]
        Rm = np.asarray(R_m2c[i], dtype=np.float64).reshape(3, 3)
        tm = np.asarray(t_m2c[i], dtype=np.float64).reshape(3)
        R_tot = Rm.T @ Eh[:3, :3]
        t_var = Rm.T @ Eh[:3, 3]
        t_fix = -Rm.T @ tm

        own = None
        if view_ids is not None:
            sel = probe_ids == i
            if sel.sum() >= 20:
                own = probe[sel]

        best_k, best_score, best_own = None, np.inf, float("nan")
        # one thread per query: a pool of threads costs more than a query of 4,096 points (see objeval3d.threads)
        for k in np.geomspace(k_init * k_lo, k_init * k_hi, n_k):
            t_k = k * t_var + t_fix
            d, _ = tree.query(apply_sim3(probe, float(k), R_tot, t_k), k=1, workers=1)
            sc = float(np.partition(d, keep - 1)[:keep].mean())
            if sc < best_score:
                best_score, best_k = sc, float(k)
                if own is not None:
                    do, _ = tree.query(apply_sim3(own, float(k), R_tot, t_k), k=1, workers=1)
                    ko = max(3, int(round(overlap * len(do))))
                    best_own = float(np.partition(do, ko - 1)[:ko].mean())
        per_view.append(
            {
                "view": i,
                "scale": best_k,
                "score": best_score,
                "score_own": best_own,
                "R": R_tot,
                "t": best_k * t_var + t_fix,
            }
        )

    best = min(per_view, key=lambda v: v["score"])
    # spread between the per-view solutions, modulo nothing - pure pose disagreement
    angles = []
    for v in per_view:
        if v is best:
            continue
        c = (np.trace(best["R"] @ v["R"].T) - 1.0) / 2.0
        angles.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    # A scale sitting on the first or last sample of the sweep was clipped, not
    # found: the true value is outside [k_init*k_lo, k_init*k_hi] and the metrics
    # downstream report the edge of the search. This catches only the clean case
    # where the score falls monotonically toward the edge. It does NOT prove the
    # sweep was wide enough -- a k_init five times too low produces a spurious
    # interior minimum well inside the range, which looks converged by every
    # number here. The sweep bounds are the real defence; this flag only refuses
    # to let the obvious failure pass silently.
    edge = 1.0 + 1e-9
    clipped = sum(1 for v in per_view if not (k_init * k_lo * edge < v["scale"] < k_init * k_hi / edge))
    return (best["scale"], best["R"], best["t"]), {
        "best_view": best["view"],
        "scale_per_view": [round(v["scale"], 1) for v in per_view],
        "score_all_per_view_mm": [round(v["score"], 2) for v in per_view],
        "score_own_per_view_mm": [round(v["score_own"], 2) for v in per_view],
        "pose_disagreement_deg": [round(a, 2) for a in angles],
        "scale_sweep": [round(k_init * k_lo, 1), round(k_init * k_hi, 1)],
        "scale_clipped_views": clipped,
    }


def depth_scale_init(depth, obj_mask, t_m2c) -> float:
    """Rough mm-per-prediction-unit, used only to centre the 1-D sweep above."""
    ks = []
    for i in range(len(depth)):
        m = obj_mask[i] & (depth[i] > 0)
        if m.sum() < 20:
            continue
        dm = float(np.median(depth[i][m]))
        if dm > 1e-6:
            ks.append(float(np.linalg.norm(np.asarray(t_m2c[i]))) / dm)
    return float(np.median(ks)) if ks else 1.0


# ---------------------------------------------------------------------------
# Symmetry resolution
# ---------------------------------------------------------------------------
def resolve_symmetry(
    recon_aligned: np.ndarray, gt_visible: np.ndarray, syms: list[np.ndarray], max_probe: int = 8000, seed: int = 0
):
    """Pick the symmetry element under which the aligned reconstruction best
    explains the observable part of the model.

    dist(gt_visible -> S @ recon) == dist(S^-1 @ gt_visible -> recon), so one
    KD-tree over the reconstruction serves the whole group.
    """
    if len(syms) <= 1:
        return np.eye(4), 0
    rng = np.random.default_rng(seed)
    probe = (
        gt_visible
        if len(gt_visible) <= max_probe
        else gt_visible[rng.choice(len(gt_visible), max_probe, replace=False)]
    )
    tree = cKDTree(recon_aligned)
    best_i, best_d = 0, np.inf
    for i, S in enumerate(syms):
        Si = np.linalg.inv(S)
        q = (Si[:3, :3] @ probe.T).T + Si[:3, 3]
        d, _ = tree.query(q, k=1, workers=workers())
        m = float(d.mean())
        if m < best_d:
            best_d, best_i = m, i
    return syms[best_i], best_i


def transform_discrepancy(a, b, syms: list[np.ndarray], diameter: float) -> dict:
    """How far apart two Sim(3) solutions are, modulo the object's symmetries."""
    sa, Ra, ta = a
    sb, Rb, tb = b
    best = None
    for S in syms:
        Rs, ts = S[:3, :3], S[:3, 3]
        # b composed with the symmetry, then compared to a
        Rb_s = Rs @ Rb
        tb_s = Rs @ tb + ts
        cos = (np.trace(Ra @ Rb_s.T) - 1.0) / 2.0
        ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        tr = float(np.linalg.norm(ta - tb_s))
        if best is None or ang < best[0]:
            best = (ang, tr)
    return {
        "rotation_deg": best[0],
        "translation_mm": best[1],
        "translation_rel_diam": best[1] / diameter,
        "scale_ratio": float(sa / sb) if sb else float("nan"),
    }


def placement_gap_points(x: np.ndarray, sim3_a: dict, sim3_b: dict, diameter: float) -> tuple[float, float]:
    """How far two Sim(3) solutions move one cloud apart: (centroid shift, RMS displacement), both in object diameters.

    For a reference placement that is not a run folder (the benchmark meshes are already in the object frame and have no
    `object_points.ply`), so that it is judged by the same rule as every other row.
    """
    p = [s["scale"] * (np.asarray(s["R"]) @ x.T).T + np.asarray(s["t"]) for s in (sim3_a, sim3_b)]
    centroid = float(np.linalg.norm(p[0].mean(0) - p[1].mean(0))) / diameter
    rms = float(np.sqrt(((p[0] - p[1]) ** 2).sum(1).mean())) / diameter
    return centroid, rms


# ---------------------------------------------------------------------------
def run_set_id(run_dir: Path) -> str:
    """The view set a run folder belongs to, taken from its own meta.json.

    The folder is named `{set_id}__{tag}`, but tags may contain underscores, so
    stripping the suffix is guesswork; meta.json carries the set verbatim.
    """
    try:
        with open(run_dir / "meta.json") as f:
            return json.load(f)["view_set"]["set_id"]
    except Exception:
        return run_dir.name


def evaluate_run(run_dir: Path, bop_root: Path, args) -> dict:
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    vs = meta["view_set"]

    pred = np.load(run_dir / "pred.npz")
    recon, _ = read_ply(run_dir / "object_points.ply")
    # The centre of the whole cleaned cloud. Two Sim(3) solutions differ in where they put the cloud by an affine map, so the
    # analysis of placement (`objeval3d.analysis.placement`) needs only this point, not the 10^5 points themselves.
    recon_centroid = [round(float(v), 4) for v in recon.mean(0)] if len(recon) else None
    view_ids = pred["view_ids"] if "view_ids" in pred.files else None
    if view_ids is not None and len(view_ids) != len(recon):
        view_ids = None  # written by an older run - drop rather than misalign
    if len(recon) < args.min_points:
        return {
            "set_id": vs["set_id"],
            "run_dir": str(run_dir),
            "status": "too_few_points",
            "n_recon": int(len(recon)),
            "mask_metrics": meta.get("mask_metrics"),
            "multiview_consistency": meta.get("multiview_consistency"),
        }

    if args.max_recon_points and len(recon) > args.max_recon_points:
        rng = np.random.default_rng(args.seed)
        keep_idx = rng.choice(len(recon), args.max_recon_points, replace=False)
        recon = recon[keep_idx]
        if view_ids is not None:
            view_ids = view_ids[keep_idx]

    g = build_gt_clouds(vs, bop_root, n_sample=args.gt_sample, n_keep=args.gt_keep, seed=args.seed)
    syms = symmetry_transforms(g["symmetries"], n_continuous=args.n_continuous_sym)

    E_da3 = pred["extrinsics"]

    out = {
        "set_id": vs["set_id"],
        "run_dir": str(run_dir),
        "status": "ok",
        "obj_id": g["obj_id"],
        "obj_name": vs.get("obj_name"),
        "scene_id": vs["scene_id"],
        "im_ids": vs["im_ids"],
        "tag": meta.get("tag"),
        "mask_source": meta.get("mask_source"),
        "mask_morph": meta.get("mask_morph", 0),
        "pair_angles_deg": vs.get("pair_angles_deg"),
        "diameter_mm": g["diameter"],
        "gt_visible_fraction": g["visible_fraction"],
        "n_recon_points": int(len(recon)),
        "recon_centroid": recon_centroid,
        "n_symmetries": len(syms),
        "mask_metrics": meta.get("mask_metrics"),
        "crop_to_object": meta.get("crop_to_object", "none"),
        "multiview_consistency": meta.get("multiview_consistency"),
        "alignments": {},
    }

    transforms = {}
    if "auto" in args.modes:
        transforms["auto"] = align_sim3(
            recon, g["gt_full"], n_rot=args.n_rot, overlap=args.overlap, gt_overlap=args.gt_overlap, seed=args.seed
        )

    # Third-party implementations of the same free search, for the generality check: the bias would otherwise have been measured
    # only through the engine in `registration.py`, which is ours. Everything downstream - criterion, symmetry resolution,
    # metrics - stays the same, so the rows are comparable with `auto` sign for sign. A `_rigid` suffix locks the scale and gives
    # the six-parameter variant.
    for mode in args.modes:
        base = mode[:-6] if mode.endswith("_rigid") else mode
        if base not in reg_baselines.BACKENDS:
            continue
        # Not RANSAC alone: FGR is just as noisy per call (a 38 % relative spread on one LM-O view set, found by measuring, not
        # from its documentation). icp_pca and teaserpp vary only at floating-point noise level and are called once.
        kw = {"n_repeats": args.ransac_repeats} if base in ("fpfh_ransac", "fgr") else {}
        try:
            transforms[mode] = reg_baselines.BACKENDS[base](
                recon,
                g["gt_full"],
                overlap=args.overlap,
                gt_overlap=args.gt_overlap,
                seed=args.seed,
                voxel=args.baseline_voxel,
                with_scaling=not mode.endswith("_rigid"),
                **kw,
            )
        except Exception as e:
            # A backend that cannot run on this cloud is itself a result (descriptor methods fail on smooth objects), so it is
            # recorded rather than allowed to abort the view set.
            out.setdefault("backend_failures", {})[mode] = f"{type(e).__name__}: {e}"

    if "oracle" in args.modes or "oracle_raw" in args.modes:
        k0 = depth_scale_init(pred["depth"], pred["obj_mask"], vs["t_m2c"])
        raw, diag = oracle_object_sim3(
            E_da3,
            vs["R_m2c"],
            vs["t_m2c"],
            g["gt_full"],
            recon,
            k0,
            view_ids=view_ids,
            overlap=args.overlap,
            seed=args.seed,
            n_k=args.n_k,
            k_lo=args.k_lo,
            k_hi=args.k_hi,
        )
        out["oracle_diagnostics"] = {**diag, "k_init": k0}
        if "oracle_raw" in args.modes:
            transforms["oracle_raw"] = {
                "scale": raw[0],
                "R": raw[1],
                "t": raw[2],
                "score_normalised": float("nan"),
                "used_init": True,
                "trace": [],
                "n_hypotheses": 1,
            }
        if "oracle" in args.modes:
            transforms["oracle"] = align_sim3(
                recon, g["gt_full"], init=raw, overlap=args.overlap, gt_overlap=args.gt_overlap, seed=args.seed
            )
            transforms["oracle"]["init_scale"] = float(raw[0])
        if "auto_fixedscale" in args.modes:
            # Same global search as `auto`, but with the scale locked to the one the reference establishes. Isolates one question:
            # is the free scale the whole reason the free fit disagrees with the reference, or is the search over rotation and
            # translation wrong too? If this mode matched the reference, a metric reconstruction would fix the protocol; the paper
            # shows it does not (the bias of F@2% is unchanged).
            transforms["auto_fixedscale"] = align_sim3(
                recon,
                g["gt_full"],
                n_rot=args.n_rot,
                overlap=args.overlap,
                gt_overlap=args.gt_overlap,
                fixed_world_scale=float(raw[0]),
                seed=args.seed,
            )

    for mode, res in transforms.items():
        aligned = apply_sim3(recon, res["scale"], res["R"], res["t"])
        S, s_idx = resolve_symmetry(aligned, g["gt_visible"], syms, seed=args.seed)
        aligned_s = (S[:3, :3] @ aligned.T).T + S[:3, 3]
        m = reconstruction_metrics(aligned_s, g["gt_full"], g["gt_visible"], g["diameter"])
        m["symmetry_index"] = int(s_idx)
        m["sim3"] = {
            "scale": float(res["scale"]),
            "R": np.asarray(res["R"]).tolist(),
            "t": np.asarray(res["t"]).tolist(),
        }
        m["search"] = {
            k: res[k]
            for k in ("score_normalised", "fwd_normalised", "bwd_normalised", "n_hypotheses", "used_init", "init_scale")
            if k in res
        }
        # the spread of the repeated calls of the non-reproducible backends (fpfh_ransac, fgr)
        if "repeats" in res:
            m["search"]["repeats"] = res["repeats"]
        out["alignments"][mode] = m

        if args.save_aligned:
            write_ply(run_dir / f"recon_aligned_{mode}.ply", aligned_s)

    # Discrepancy against the pose-anchored reference, for EVERY free mode and not just `auto`: what the criterion does on the
    # sets where the search actually recovered the pose can only be computed where this field exists.
    ref = "oracle_raw" if "oracle_raw" in transforms else "oracle"
    if ref in transforms:
        b = transforms[ref]
        fo = out["alignments"][ref]["fscore"]["tau_2pct"]["fscore"]
        out["vs_oracle"] = {}
        for mode, a in transforms.items():
            if mode == ref:
                continue
            d = transform_discrepancy((a["scale"], a["R"], a["t"]), (b["scale"], b["R"], b["t"]), syms, g["diameter"])
            fa = out["alignments"][mode]["fscore"]["tau_2pct"]["fscore"]
            d["delta_fscore_2pct"] = float(fa - fo)
            # NOT the recovery test the paper reports, and it must not be read as one. `translation_rel_diam` differences the raw
            # translation vectors of two Sim(3) solutions, and a free fit moves the scale too, so `t` absorbs a rescaling about
            # a scene origin far from the object: on LM-O it reads 5.9 diameters median while the cloud demonstrably lies on the
            # scan. The test that counts applies both transforms to the same cloud and measures how far its points move
            # (`objeval3d.analysis.placement`). Kept only so that the two can be reported side by side.
            d["converged_tvec_rule"] = bool(
                d["rotation_deg"] < args.converged_rot_deg and d["translation_rel_diam"] < args.converged_trans_rel
            )
            out["vs_oracle"][mode] = d
        # the same dict under the name the early result files used
        if "auto" in out["vs_oracle"]:
            out["auto_vs_oracle"] = dict(out["vs_oracle"]["auto"])
            out["auto_vs_oracle"]["converged"] = out["vs_oracle"]["auto"]["converged_tvec_rule"]

    if args.save_gt:
        write_ply(run_dir / "gt_full.ply", g["gt_full"])
        write_ply(run_dir / "gt_visible.ply", g["gt_visible"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Align object clouds to GT and score them")
    ap.add_argument("--runs-root", required=True, help="output root of objeval3d.recon.reconstruct")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    ap.add_argument("--out", required=True, help="results JSON")
    ap.add_argument("--pattern", default="*", help="glob over run folder names")
    ap.add_argument(
        "--modes",
        default="auto,oracle_raw,oracle",
        help="auto | oracle_raw | oracle | auto_fixedscale | fpfh_ransac | fgr | icp_pca | teaserpp, "
        "each third-party search also with a _rigid suffix for the six-parameter variant; "
        "oracle and auto_fixedscale need oracle_raw's solution, which is computed whenever either is requested",
    )
    ap.add_argument("--n-rot", type=int, default=1536)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--gt-overlap", type=float, default=0.45)
    ap.add_argument(
        "--baseline-voxel",
        type=float,
        default=0.05,
        help="voxel size for the third-party backends, in the "
        "normalised frame where the RMS radius is 1. Open3D's "
        "own recipe; sweeping it shows the conclusion does not "
        "rest on our tuning of someone else's method",
    )
    ap.add_argument(
        "--ransac-repeats",
        type=int,
        default=1,
        help="repeats for fpfh_ransac AND fgr, both measured to "
        "answer differently every call on identical input "
        "(RANSAC has no seed in Open3D 0.19; FGR's source of "
        "variation is unconfirmed but was measured, not "
        "assumed away). >1 reports the median call and the "
        "spread across calls per view set; icp_pca and "
        "teaserpp are unaffected, called once",
    )
    ap.add_argument("--gt-sample", type=int, default=200_000)
    ap.add_argument("--gt-keep", type=int, default=60_000)
    ap.add_argument("--max-recon-points", type=int, default=120_000)
    ap.add_argument("--min-points", type=int, default=200)
    ap.add_argument("--n-continuous-sym", type=int, default=24)
    ap.add_argument("--converged-rot-deg", type=float, default=15.0)
    ap.add_argument("--converged-trans-rel", type=float, default=0.10)
    ap.add_argument("--save-aligned", action="store_true")
    ap.add_argument("--save-gt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--k-lo",
        type=float,
        default=0.4,
        help="the reference scale sweep runs over k_init * [k-lo, k-hi]. Widen it "
        "when k_init cannot be trusted - with --mask-source none it is "
        "the median depth of the whole frame, not of the object",
    )
    ap.add_argument("--k-hi", type=float, default=2.5)
    ap.add_argument(
        "--n-k",
        type=int,
        default=61,
        help="number of scales in the reference sweep: a geometric grid from k-lo to k-hi, 61 values 3.1 %% apart by default (the paper's setting)",
    )
    ap.add_argument(
        "--keep-stale",
        action="store_true",
        help="score run folders that the manifest does not list (off by default: they are usually leftovers)",
    )
    args = ap.parse_args()
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    manifest = load_manifest(args.manifest, args.bop_root)
    bop_root = Path(manifest["bop_root"])
    wanted = {s["set_id"] for s in manifest["sets"]}

    # The run folder is reused across experiments, so a glob over it also picks up runs left by an earlier manifest - a different
    # split, a different set of objects. Those would score silently and land in the same results file. The manifest is the
    # authority on what is being measured.
    runs, stale = [], []
    for p in sorted(Path(args.runs_root).glob(args.pattern)):
        if not (p / "meta.json").exists():
            continue
        (runs if run_set_id(p) in wanted else stale).append(p)
    if stale:
        print(
            f"[stale] {len(stale)} run(s) under {args.runs_root} are not in "
            f"{args.manifest} and were skipped: "
            + ", ".join(p.name for p in stale[:6])
            + (f", ... (+{len(stale) - 6} more)" if len(stale) > 6 else "")
        )
        if args.keep_stale:
            print("[stale] --keep-stale given: scoring them anyway")
            runs = sorted(runs + stale)
    print(f"{len(runs)} runs to evaluate")

    results = []
    for i, run in enumerate(runs):
        try:
            r = evaluate_run(run, bop_root, args)
        except Exception as e:  # keep going: one bad run must not lose the batch
            r = {"set_id": run.name, "run_dir": str(run), "status": f"error: {e}"}
        results.append(r)
        alns = r.get("alignments") or {}
        # Report whichever mode was actually asked for, preferring the pose-anchored reference.
        a = next((alns[m] for m in ("oracle_raw", "auto", "oracle") if m in alns), None)
        if r["status"] == "ok" and a is not None:
            print(
                f"[{i + 1}/{len(runs)}] {run.name}: "
                f"acc={a['accuracy']['mean']:.2f}mm "
                f"({100 * a['accuracy']['mean_rel_diam']:.2f}% diam) "
                f"comp={a['completeness']['mean']:.2f}mm "
                f"F@2%={a['fscore']['tau_2pct']['fscore']:.3f}"
                + (
                    f"  auto-vs-oracle rot={r['auto_vs_oracle']['rotation_deg']:.1f}deg"
                    if "auto_vs_oracle" in r
                    else ""
                )
            )
        elif r["status"] == "ok":
            print(
                f"[{i + 1}/{len(runs)}] {run.name}: ok, but no alignment mode "
                f"produced a result (modes requested: {','.join(args.modes)})"
            )
        else:
            print(f"[{i + 1}/{len(runs)}] {run.name}: {r['status']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "git_revision": git_revision(), "results": results}, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
