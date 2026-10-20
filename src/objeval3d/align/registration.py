"""Deterministic Sim(3) registration of a partial reconstruction to a complete
ground-truth model, plus the reconstruction-quality metrics computed after it.

Why not FPFH + RANSAC
---------------------
Feature-based global registration is unreliable on exactly the objects this
protocol targets: it is stochastic (a different answer every run), it needs
distinctive local geometry, and it collapses on rotationally symmetric or smooth
objects - cans, bowls, boxes, i.e. most of YCB-V and LM-O.  (FPFH + RANSAC is
still implemented in `objeval3d.align.baselines`, as one of the third-party
searches the protocol is compared with.)

Here the problem is much better posed than generic registration: we are
matching an object to *its own* scanned model, so the only unknowns are a
rotation, a translation and one global scale, and the object is compact and
centred. That admits an exhaustive, deterministic solution:

  1. normalise both clouds (centroid + RMS radius) -> scale roughly removed;
  2. score a near-uniform grid of rotations on SO(3) (super-Fibonacci spirals,
     Alexa 2022) crossed with a small band of scale multipliers, using a
     trimmed one-way Chamfer distance on subsampled clouds;
  3. refine the best K hypotheses with trimmed scaled ICP;
  4. keep the hypothesis with the lowest symmetric trimmed Chamfer.

Trimming matters throughout: the reconstruction only ever covers the visible
side of the object (no bottom, no far side), so a plain least-squares fit to
the full model is biased by construction.  Trimmed ICP (Chetverikov et al.,
2002) is the standard fix and it is what makes partial-to-complete alignment
behave.

The result is reproducible bit-for-bit given the same inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from objeval3d.threads import workers


# ---------------------------------------------------------------------------
# Sim(3) helpers
# ---------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping `src` onto `dst`.

    Returns (s, R, t) with  dst ~= s * R @ src.T + t.
    Umeyama (1991), including the reflection guard via det(U)det(V).
    """
    assert src.shape == dst.shape and src.shape[0] >= 3
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd.T @ xs) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs**2).sum() / len(src)
        s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_sim3(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (R @ pts.T)).T + t


def compose_sim3(a, b):
    """Return the Sim(3) equal to applying `b` after `a`."""
    sa, Ra, ta = a
    sb, Rb, tb = b
    return sb * sa, Rb @ Ra, sb * (Rb @ ta) + tb


# ---------------------------------------------------------------------------
# SO(3) sampling
# ---------------------------------------------------------------------------
def super_fibonacci(n: int) -> np.ndarray:
    """`n` near-uniform unit quaternions (x, y, z, w) - Alexa, "Super-Fibonacci
    Spirals", CVPR 2022.  Deterministic and O(n), unlike rejection sampling."""
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    t = i / n
    d = 2.0 * np.pi * i
    r, R = np.sqrt(t), np.sqrt(1.0 - t)
    a, b = d / phi, d / psi
    return np.stack([r * np.sin(a), r * np.cos(a), R * np.sin(b), R * np.cos(b)], axis=1)


def quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """(N,4) quaternions in (x,y,z,w) order -> (N,3,3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(q)
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------------------
# Sampling / normalisation
# ---------------------------------------------------------------------------
def subsample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), size=n, replace=False)]


def normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre at the centroid and divide by the RMS radius."""
    c = pts.mean(0)
    q = pts - c
    s = float(np.sqrt((q**2).sum(1).mean()))
    s = s if s > 1e-12 else 1.0
    return q / s, c, s


def trimmed_mean_nn(tree: cKDTree, pts: np.ndarray, keep: float) -> float:
    d, _ = tree.query(pts, k=1, workers=workers())
    if keep >= 1.0:
        return float(d.mean())
    k = max(1, int(round(keep * len(d))))
    return float(np.partition(d, k - 1)[:k].mean())


# ---------------------------------------------------------------------------
# Trimmed scaled ICP
# ---------------------------------------------------------------------------
def trimmed_icp(
    src: np.ndarray,
    dst: np.ndarray,
    dst_tree: cKDTree,
    init: tuple[float, np.ndarray, np.ndarray],
    overlap: float = 0.8,
    gt_overlap: float = 0.45,
    max_iter: int = 60,
    tol: float = 1e-6,
    scale_band: float = 0.30,
    bidirectional: bool = True,
    refit_after_clip: bool = False,
):
    """Bidirectional trimmed ICP with scale, `src` -> `dst`.

    Forward pairs (each source point to its nearest model point) are trimmed to
    the best `overlap` fraction, which discards segmentation leakage and points
    with no counterpart.  Backward pairs (each model point to its nearest
    source point) are trimmed to the best `gt_overlap` fraction - roughly the
    share of the model one viewpoint band can see.

    The backward term is not cosmetic: with forward pairs alone and a free
    scale, shrinking the reconstruction towards the interior of the model
    monotonically reduces the objective and the fit collapses.  Pulling the
    covered part of the model back onto the reconstruction removes that
    degenerate direction; `scale_band` is a second, hard guard.

    `refit_after_clip` - when the guard clips the scale (always, if
    `scale_band` is 0, i.e. the rigid six-parameter variant), the rotation and
    translation Umeyama returned were solved for the unclipped scale, so the
    pair (clipped scale, translation) is not a least-squares solution of
    anything. With the flag on they are solved again at the clipped scale. Off
    by default because every number in the paper was computed without it; where
    the scale is not clipped the two behaviours are identical. Measured once,
    on one BOP model with three random poses per case and the scale held
    fixed: with no size mismatch the two agree to the digit; with a 4 %
    mismatch the criterion is 3.5 % lower with the flag on for a complete cloud
    (median distance to the scan 1.20 -> 1.14 mm) and under 0.5 % lower for a
    half-covered one.
    """
    s, R, t = init
    s0 = s
    kf = max(3, int(round(overlap * len(src))))
    kb = max(3, int(round(gt_overlap * len(dst))))
    prev = np.inf
    hist = []
    for _ in range(max_iter):
        p = apply_sim3(src, s, R, t)
        d, idx = dst_tree.query(p, k=1, workers=workers())
        order = np.argpartition(d, kf - 1)[:kf]
        err = float(d[order].mean())
        A = [src[order]]
        B = [dst[idx[order]]]

        if bidirectional:
            src_tree = cKDTree(p)
            db, ib = src_tree.query(dst, k=1, workers=workers())
            ob = np.argpartition(db, kb - 1)[:kb]
            A.append(src[ib[ob]])
            B.append(dst[ob])
            err = 0.5 * (err + float(db[ob].mean()))

        s_new, R_new, t_new = umeyama(np.vstack(A), np.vstack(B), with_scale=True)
        s = float(np.clip(s_new, s0 * (1 - scale_band), s0 * (1 + scale_band)))
        if refit_after_clip and s != s_new:
            _, R_new, t_new = umeyama(s * np.vstack(A), np.vstack(B), with_scale=False)
        R, t = R_new, t_new
        hist.append(err)
        if abs(prev - err) < tol * max(1.0, abs(prev)):
            break
        prev = err
    return (s, R, t), hist


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def align_sim3(
    recon: np.ndarray,
    gt: np.ndarray,
    *,
    n_rot: int = 1536,
    scale_mults=(0.7, 0.85, 1.0, 1.2),
    n_coarse: int = 384,
    n_mid: int = 1024,
    n_fine: int = 4096,
    top_k_coarse: int = 96,
    top_k_fine: int = 8,
    mid_iter: int = 20,
    overlap: float = 0.8,
    gt_overlap: float = 0.45,
    scale_band: float = 0.30,
    init: tuple[float, np.ndarray, np.ndarray] | None = None,
    fixed_world_scale: float | None = None,
    refit_after_clip: bool = False,
    seed: int = 0,
) -> dict:
    """Align `recon` onto `gt` (both (N,3), any units) with a Sim(3).

    Three-stage cascade - a wide, cheap rotation sweep followed by
    progressively fewer, progressively more expensive refinements.  A single
    coarse ranking is not trustworthy on its own: both clouds are normalised
    about their own centroids, and a partial reconstruction's centroid is
    offset from the full model's, so the coarse stage scores every rotation
    under a translation that is systematically wrong.  Carrying 32 hypotheses
    into a short ICP and only then narrowing to 6 lets the correct rotation
    recover from that handicap.

    `init` skips the global search and refines the supplied world-space
    transform instead - how the ground-truth-pose oracle is turned into the
    best locally achievable alignment.

    `fixed_world_scale` locks the similarity's scale, reducing the search to a
    rigid one. This is the mode to use when the reconstruction is already
    metric: a free scale lets the fit shrink a poor reconstruction until it sits
    inside the model, which on the smallest LM-O objects halved the recovered
    scale and made the resulting metrics look twice as good as they were. Pass
    `refit_after_clip=True` with it: the scale is then clipped on every
    iteration, and without the flag the translation is solved for a different
    scale from the one applied (see `trimmed_icp`).

    Returns the transform, its diagnostics and the full search trace.
    """
    if len(recon) < 10 or len(gt) < 10:
        raise ValueError(f"too few points: recon={len(recon)}, gt={len(gt)}")

    rn, c_r, s_r = normalise(recon)
    gn, c_g, s_g = normalise(gt)

    gn_fine = subsample(gn, n_fine, seed)
    tree_fine = cKDTree(gn_fine)
    rn_fine = subsample(rn, n_fine, seed + 2)
    rn_mid = subsample(rn, n_mid, seed + 3)

    def to_world(sn, Rn, tn):
        s = s_g * sn / s_r
        return float(s), Rn, -s * (Rn @ c_r) + s_g * tn + c_g

    if fixed_world_scale is not None:
        # world scale s relates to the normalised one by s = s_g * sn / s_r
        sn_fixed = fixed_world_scale * s_r / s_g
        scale_mults = (sn_fixed,)
        scale_band = 0.0
    icp_kw = dict(overlap=overlap, gt_overlap=gt_overlap, scale_band=scale_band, refit_after_clip=refit_after_clip)

    def score_of(sn, Rn, tn, src_pts):
        moved = apply_sim3(src_pts, sn, Rn, tn)
        fwd = trimmed_mean_nn(tree_fine, moved, overlap)
        bwd = trimmed_mean_nn(cKDTree(moved), gn_fine, gt_overlap)
        return fwd + bwd, fwd, bwd

    trace = []
    if init is not None:
        # World-space transform -> normalised space:  gn = sn Rn rn + tn
        s_w, R_w, t_w = init
        sn = s_w * s_r / s_g
        tn = (s_w * (R_w @ c_r) + t_w - c_g) / s_g
        finalists = [(sn, R_w, tn)]
    else:
        rn_coarse = subsample(rn, n_coarse, seed)
        gn_coarse = subsample(gn, min(n_fine, 8192), seed + 1)
        tree_coarse = cKDTree(gn_coarse)
        Rs = quats_to_matrices(super_fibonacci(n_rot))
        kk = max(1, int(round(overlap * len(rn_coarse))))

        # Two translation hypotheses per rotation.  Both clouds are centred on
        # their own centroids, but a reconstruction covering only part of the
        # object has its centroid pulled towards the observed side, so the
        # centroid-aligned hypothesis can be off by a large fraction of the
        # object.  The bounding-box centre is far less sensitive to which
        # surfaces are missing, as long as the visible part reaches the
        # extremes - which it usually does.  Keeping both costs one extra pass
        # and rescues exactly the partial-coverage case that matters here.
        gn_bbox_c = (gn_coarse.min(0) + gn_coarse.max(0)) / 2.0

        cands = []
        for m in scale_mults:
            rot = np.einsum("kij,nj->kni", Rs, rn_coarse) * m
            for shift_mode in ("centroid", "bbox"):
                if shift_mode == "centroid":
                    cand_pts = rot
                    shifts = np.zeros((len(Rs), 3))
                else:
                    bc = (rot.min(1) + rot.max(1)) / 2.0  # [n_rot, 3]
                    shifts = gn_bbox_c[None, :] - bc
                    cand_pts = rot + shifts[:, None, :]
                d, _ = tree_coarse.query(cand_pts.reshape(-1, 3), k=1, workers=workers())
                d = d.reshape(len(Rs), -1)
                sc = np.partition(d, kk - 1, axis=1)[:, :kk].mean(axis=1)
                for i in np.argsort(sc)[:top_k_coarse]:
                    cands.append((float(sc[i]), float(m), Rs[i], shifts[i].copy()))
        cands.sort(key=lambda x: x[0])
        cands = cands[:top_k_coarse]

        # short ICP on a mid-resolution cloud, purely to re-rank
        mid = []
        for coarse_score, m, R0, t0 in cands:
            (sn, Rn, tn), _ = trimmed_icp(rn_mid, gn_fine, tree_fine, (m, R0, t0), max_iter=mid_iter, **icp_kw)
            sc, fwd, bwd = score_of(sn, Rn, tn, rn_mid)
            mid.append((sc, sn, Rn, tn, coarse_score))
        mid.sort(key=lambda x: x[0])
        finalists = [(sn, Rn, tn) for _, sn, Rn, tn, _ in mid[:top_k_fine]]
        trace.append({"stage": "coarse->mid", "mid_scores": [round(x[0], 5) for x in mid]})

    best = None
    for hi, hyp in enumerate(finalists):
        (sn, Rn, tn), hist = trimmed_icp(rn_fine, gn_fine, tree_fine, hyp, max_iter=80, **icp_kw)
        sc, fwd, bwd = score_of(sn, Rn, tn, rn_fine)
        trace.append({"stage": "fine", "hypothesis": hi, "icp_iters": len(hist), "fwd": fwd, "bwd": bwd, "score": sc})
        if best is None or sc < best[0]:
            best = (sc, sn, Rn, tn, fwd, bwd)

    sc, sn, Rn, tn, fwd, bwd = best
    s, R, t = to_world(sn, Rn, tn)
    return {
        "scale": s,
        "R": R,
        "t": t,
        "score_normalised": sc,
        "fwd_normalised": fwd,
        "bwd_normalised": bwd,
        "n_hypotheses": len(finalists),
        "trace": trace,
        "used_init": init is not None,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def reconstruction_metrics(
    recon_aligned: np.ndarray,
    gt_full: np.ndarray,
    gt_visible: np.ndarray,
    diameter: float,
    taus=(0.01, 0.02, 0.05),
) -> dict:
    """Accuracy / completeness / Chamfer / F-score, all in GT units (mm) and
    also as a fraction of the object's diameter so objects of different sizes
    are comparable.

    * accuracy uses the FULL model as reference: a reconstructed point is
      correct if it lies on the object anywhere, even on a surface no input
      view could see (it cannot be penalised for being right).
    * completeness uses the VISIBLE subset: the model cannot be blamed for not
      reconstructing the bottom of an object it never saw.  Completeness
      against the full model is reported too, as `completeness_full`, since
      that is what a naive evaluation would report.
    """
    tree_full = cKDTree(gt_full)
    tree_recon = cKDTree(recon_aligned)

    d_acc, _ = tree_full.query(recon_aligned, k=1, workers=workers())
    d_comp, _ = tree_recon.query(gt_visible, k=1, workers=workers())
    d_comp_full, _ = tree_recon.query(gt_full, k=1, workers=workers())

    def stats(d):
        return {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "rmse": float(np.sqrt((d**2).mean())),
            "p95": float(np.percentile(d, 95)),
            "mean_rel_diam": float(d.mean() / diameter),
            "median_rel_diam": float(np.median(d) / diameter),
        }

    # Median nearest-neighbour spacing inside the reconstruction. Completeness
    # cannot go below roughly half of it: a dense ground-truth surface can be no
    # closer to the reconstruction than its own sampling allows. Small objects
    # produce sparse clouds (a 30x30 px object over 3 views is ~1k points), and
    # without this number a completeness figure cannot be told apart from a
    # sampling artefact.
    d_self, _ = tree_recon.query(recon_aligned, k=2, workers=workers())
    spacing = float(np.median(d_self[:, 1])) if len(recon_aligned) > 1 else float("nan")

    out = {
        "diameter_mm": float(diameter),
        "n_recon": int(len(recon_aligned)),
        "recon_spacing_mm": spacing,
        "recon_spacing_rel_diam": spacing / diameter,
        "n_gt_full": int(len(gt_full)),
        "n_gt_visible": int(len(gt_visible)),
        "accuracy": stats(d_acc),
        "completeness": stats(d_comp),
        "completeness_full": stats(d_comp_full),
        "chamfer_l1_mm": float(d_acc.mean() + d_comp.mean()),
        "chamfer_l1_rel_diam": float((d_acc.mean() + d_comp.mean()) / diameter),
        # >> 1 means completeness reflects reconstruction quality; ~1 means it is
        # measuring how sparse the cloud is.
        "completeness_over_spacing": float(d_comp.mean() / spacing) if spacing > 0 else float("nan"),
        "fscore": {},
    }
    for tau in taus:
        thr = tau * diameter
        prec = float((d_acc < thr).mean())
        rec = float((d_comp < thr).mean())
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out["fscore"][f"tau_{int(tau * 100)}pct"] = {
            "threshold_mm": float(thr),
            "precision": prec,
            "recall": rec,
            "fscore": f,
        }
    return out
