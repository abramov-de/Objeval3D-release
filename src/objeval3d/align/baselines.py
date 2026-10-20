"""Third-party free-alignment searches, for the generality check of the alignment-bias result.

The finding about free alignment (the aligned number is optimistic in proportion to the error of the cloud) would be a statement
about one implementation if it rested on `objeval3d.align.registration` alone. These backends repeat the measurement with the
descriptor pipelines and the iterative refinement that the literature uses for the same purpose, taken from Open3D (and TEASER++)
rather than reimplemented:

  fpfh_ransac - FPFH descriptors + RANSAC (Rusu et al., ICRA 2009)
  fgr         - Fast Global Registration (Zhou et al., ECCV 2016)
  icp_pca     - principal-axis initialisation + ICP, the cheap default
  teaserpp    - TEASER++ (Yang et al., IEEE T-RO 2021), outlier-robust by construction (truncated least squares and a
                maximum-clique inlier selection)

Each returns the same dict as `align_sim3`, so `objeval3d.align.evaluate` treats them as further alignment modes and the results
land in the same JSON. The comparison is only meaningful if the criterion, the symmetry resolution and the metrics are computed
by the same code for every backend, and the only way to guarantee that is to reuse it.

This is NOT a contest between registration engines. The protocol does not propose free alignment for the third stage, so there is
no sense in which one of these backends "wins": they are four independent instances of the same practice, and what is measured is
whether the practice carries the bias, not which instance carries it best. Ranking them by the achieved criterion value would
invert the argument, since a lower value is precisely what the bias consists of.

Scale
-----
Open3D's global registration is rigid, and the free mode of the protocol has seven parameters, so leaving it that way would
compare unlike things. Both clouds are first centred and divided by their RMS radius, which removes the scale only approximately
(a reconstruction covering one side of the object has a systematically different radius from the full model), and the ICP that
finishes each backend then re-estimates it (`with_scaling`, i.e. Umeyama with scale). Every backend therefore ends with the same
seven free parameters as `align_sim3`. `with_scaling=False` gives the six-parameter variant, which is what a metric
reconstruction admits.

Determinism
-----------
Measured, not assumed (three calls each on the same LM-O view set, identical inputs): `fpfh_ransac` (Open3D exposes no seed for
RANSAC) and `fgr` (Open3D documents no randomness in it; most likely thread-scheduling-dependent tie-breaking in a parallel
nearest-neighbour search over near-duplicate FPFH descriptors) both vary by tens of percent between calls, comparable to the
margin by which free alignment beats the true pose. `icp_pca` and `teaserpp` vary too, but only at the 1e-15 relative level
(floating-point summation order in a multi-threaded KD-tree reduction), never enough to flip a comparison. `n_repeats` with
`select="median"` covers the first two; the other two are called once. Do not extend this to a new backend without measuring it
the same way: "the documentation describes no randomness" is not evidence.

Installing TEASER++
-------------------
There is no prebuilt wheel on PyPI; `teaserpp_python` is built from source and is optional (a failure to import it loses that one
backend only: `objeval3d.align.evaluate` records the error per view set and carries on with the others). The repository root of
TEASER++ is a scikit-build-core package, so one `pip install .` does the configure, build and install. It needs Eigen 3 (the
CMake file asks for 3.2 via `NO_MODULE`, so pin a 3.x release; a conda-forge 5.x build does not satisfy it), Boost, a C++ compiler
with OpenMP, and pybind11. PCL, which the TEASER++ docs list, is only needed by C++ demos that read .pcd files:

    conda install -y -c conda-forge "eigen=3.4.0" boost pybind11
    git clone https://github.com/MIT-SPARK/TEASER-plusplus.git /tmp/teaser
    # the build fetches a maximum-clique library at configure time; clone it first so the build does not touch the network for it
    git clone https://github.com/jingnanshi/pmc.git /tmp/pmc-src
    cd /tmp/teaser && pip install . -C cmake.define.FETCHCONTENT_SOURCE_DIR_PMC=/tmp/pmc-src
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from objeval3d.align.registration import apply_sim3, normalise, subsample, trimmed_mean_nn
from objeval3d.threads import workers


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "the free-alignment baselines need Open3D, which is deliberately not "
            "a dependency of the protocol itself. `pip install open3d` into the "
            "environment running objeval3d.align.evaluate (0.19 is what these defaults "
            "were tuned on)."
        ) from e
    return o3d


# ---------------------------------------------------------------------------
# Normalised frame, shared by every backend and by `align_sim3`
# ---------------------------------------------------------------------------
class _Frame:
    """Both clouds centred on their centroid and scaled to unit RMS radius.

    `to_world` is the inverse map, copied from `align_sim3` so that a transform
    solved in the normalised frame becomes the same world-space Sim(3) it would
    have become there.
    """

    def __init__(
        self,
        recon: np.ndarray,
        gt: np.ndarray,
        *,
        n_fine: int = 4096,
        overlap: float = 0.8,
        gt_overlap: float = 0.45,
        seed: int = 0,
    ):
        self.rn, self.c_r, self.s_r = normalise(recon)
        self.gn, self.c_g, self.s_g = normalise(gt)
        self.overlap = overlap
        self.gt_overlap = gt_overlap
        self.gn_fine = subsample(self.gn, n_fine, seed)
        self.rn_fine = subsample(self.rn, n_fine, seed + 2)
        self.tree_fine = cKDTree(self.gn_fine)

    def to_world(self, sn: float, Rn: np.ndarray, tn: np.ndarray):
        s = self.s_g * sn / self.s_r
        return float(s), Rn, -s * (Rn @ self.c_r) + self.s_g * tn + self.c_g

    def score(self, sn: float, Rn: np.ndarray, tn: np.ndarray):
        """Symmetric trimmed Chamfer in the normalised frame.

        Identical to the `score_of` closure inside `align_sim3`; every backend
        is scored by it so that `score_normalised` means one thing across the
        whole table.
        """
        moved = apply_sim3(self.rn_fine, sn, Rn, tn)
        fwd = trimmed_mean_nn(self.tree_fine, moved, self.overlap)
        bwd = trimmed_mean_nn(cKDTree(moved), self.gn_fine, self.gt_overlap)
        return fwd + bwd, fwd, bwd


def _result(frame: _Frame, sn, Rn, tn, *, n_hypotheses: int, trace: list) -> dict:
    sc, fwd, bwd = frame.score(sn, Rn, tn)
    s, R, t = frame.to_world(sn, Rn, tn)
    return {
        "scale": s,
        "R": R,
        "t": t,
        "score_normalised": sc,
        "fwd_normalised": fwd,
        "bwd_normalised": bwd,
        "n_hypotheses": n_hypotheses,
        "trace": trace,
        "used_init": False,
    }


def _split_scaled_rotation(M: np.ndarray):
    """Split a 3x3 `s*R` into (s, R), guarding against a reflected solution."""
    s = float(np.cbrt(max(np.linalg.det(M), 1e-24)))
    R = M / s
    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    return s, U @ S @ Vt


# ---------------------------------------------------------------------------
# Open3D plumbing
# ---------------------------------------------------------------------------
def _o3d_pair(frame: _Frame, voxel: float, *, feature: bool = True):
    """Downsampled clouds (and FPFH features) in the normalised frame.

    Radii follow Open3D's own global-registration recipe: normals at 2 voxels,
    features at 5. The RMS radius is 1 by construction here, so a single voxel
    size is meaningful across objects of every physical size - which is the
    other reason everything happens in this frame.
    """
    o3d = _require_open3d()
    out = []
    for pts in (frame.rn, frame.gn):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))
        down = pcd.voxel_down_sample(voxel)
        if len(down.points) < 10:  # tiny cloud: keep every point rather than fail
            down = pcd
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30))
        if feature:
            f = o3d.pipelines.registration.compute_fpfh_feature(
                down, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0 * voxel, max_nn=100)
            )
            out.append((down, f))
        else:
            out.append((down, None))
    return out[0], out[1]


def _icp_refine(frame: _Frame, init4: np.ndarray, voxel: float, *, with_scaling: bool, max_iter: int = 60):
    """Open3D point-to-point ICP in the normalised frame, from `init4`."""
    o3d = _require_open3d()
    (src, _), (dst, _) = _o3d_pair(frame, voxel, feature=False)
    res = o3d.pipelines.registration.registration_icp(
        src,
        dst,
        1.5 * voxel,
        init4,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return np.asarray(res.transformation, float)


def _unpack(T: np.ndarray):
    s, R = _split_scaled_rotation(T[:3, :3])
    return s, R, T[:3, 3]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _repeat_and_select(
    frame: _Frame, voxel: float, refine: bool, with_scaling: bool, n_repeats: int, select: str, draw
) -> dict:
    """Call `draw() -> (T_4x4, fitness, inlier_rmse)` up to `n_repeats` times,
    finish each with the shared ICP polish, score each by the shared criterion,
    and return one of them plus a summary of the spread across calls.

    Shared by every backend whose single call is not reproducible (see the
    module docstring). `select` picks which repeat is kept: "median" (default)
    because a practitioner runs the method once, so the typical draw is what the
    accepted practice actually delivers, and returning the best of N would let
    our own claim be inflated by resampling; "best" models the other real
    behaviour - rerunning until the number looks good - and is reported
    alongside rather than instead.
    """
    draws, repeats = [], []
    for i in range(max(1, n_repeats)):
        T, fitness, inlier_rmse = draw()
        if refine:
            T = _icp_refine(frame, T, voxel, with_scaling=with_scaling)
        sn, Rn, tn = _unpack(T)
        sc, _, _ = frame.score(sn, Rn, tn)
        repeats.append({"repeat": i, "score": sc, "fitness": fitness, "inlier_rmse": inlier_rmse})
        draws.append((sc, sn, Rn, tn))

    order = sorted(range(len(draws)), key=lambda i: draws[i][0])
    pick = order[0] if select == "best" else order[(len(order) - 1) // 2]
    _, sn, Rn, tn = draws[pick]
    out = _result(frame, sn, Rn, tn, n_hypotheses=len(repeats), trace=repeats)
    scores = [r["score"] for r in repeats]
    out["repeats"] = {
        "n": len(scores),
        "select": select,
        "selected": pick,
        "score_min": min(scores),
        "score_max": max(scores),
        "score_spread": max(scores) - min(scores),
        "score_median": float(np.median(scores)),
    }
    return out


def align_fpfh_ransac(
    recon: np.ndarray,
    gt: np.ndarray,
    *,
    voxel: float = 0.05,
    overlap: float = 0.8,
    gt_overlap: float = 0.45,
    refine: bool = True,
    with_scaling: bool = True,
    n_repeats: int = 1,
    select: str = "median",
    max_iteration: int = 1_000_000,
    confidence: float = 0.999,
    seed: int = 0,
) -> dict:
    """FPFH + RANSAC, optionally finished with ICP, repeated `n_repeats` times.

    The repeats exist because this backend has no seed in Open3D 0.19 and
    returns a different answer every call; on LM-O the spread between calls is
    of the same order as the margin by which free alignment beats the true pose.
    """
    o3d = _require_open3d()
    frame = _Frame(recon, gt, overlap=overlap, gt_overlap=gt_overlap, seed=seed)
    (src, fsrc), (dst, fdst) = _o3d_pair(frame, voxel)

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(1.5 * voxel),
    ]

    def draw():
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src,
            dst,
            fsrc,
            fdst,
            True,
            1.5 * voxel,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            checkers,
            o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence),
        )
        return (np.asarray(res.transformation, float), float(res.fitness), float(res.inlier_rmse))

    return _repeat_and_select(frame, voxel, refine, with_scaling, n_repeats, select, draw)


def align_fgr(
    recon: np.ndarray,
    gt: np.ndarray,
    *,
    voxel: float = 0.05,
    overlap: float = 0.8,
    gt_overlap: float = 0.45,
    refine: bool = True,
    with_scaling: bool = True,
    n_repeats: int = 1,
    select: str = "median",
    seed: int = 0,
) -> dict:
    """Fast Global Registration on the same FPFH features, repeated `n_repeats`
    times.

    Not reproducible between calls although Open3D's API describes no explicit
    randomness: three calls on the same LM-O view set with identical inputs gave
    criterion values 0.1509 / 0.1738 / 0.2086, a 38 % relative spread, three
    orders of magnitude past the floating-point noise `icp_pca` and `teaserpp`
    show under the same test. The likely source is thread-scheduling-dependent
    tie-breaking in a parallel nearest-neighbour search over near-duplicate FPFH
    descriptors on a mostly smooth BOP object, but the mechanism does not need
    to be pinned down for the remedy, the same one `fpfh_ransac` has: the median
    of `n_repeats` calls.
    """
    o3d = _require_open3d()
    frame = _Frame(recon, gt, overlap=overlap, gt_overlap=gt_overlap, seed=seed)
    (src, fsrc), (dst, fdst) = _o3d_pair(frame, voxel)

    def draw():
        opt = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=1.5 * voxel)
        res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(src, dst, fsrc, fdst, opt)
        return (np.asarray(res.transformation, float), float(res.fitness), float(res.inlier_rmse))

    return _repeat_and_select(frame, voxel, refine, with_scaling, n_repeats, select, draw)


def _pca_rotations(pts: np.ndarray) -> np.ndarray:
    """Principal axes as a rotation, sign-disambiguated to a proper rotation."""
    q = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(q, full_matrices=False)
    R = Vt
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    return R


def align_icp_pca(
    recon: np.ndarray,
    gt: np.ndarray,
    *,
    voxel: float = 0.05,
    overlap: float = 0.8,
    gt_overlap: float = 0.45,
    with_scaling: bool = True,
    seed: int = 0,
) -> dict:
    """Principal-axis initialisation plus ICP - the cheap default.

    A PCA frame is defined only up to the sign of each axis, so all four proper
    sign combinations are tried and each is refined; the best by the shared
    criterion is returned. Without this the backend fails for a trivial reason
    (an axis pointing the wrong way) rather than for the reason under study.
    """
    frame = _Frame(recon, gt, overlap=overlap, gt_overlap=gt_overlap, seed=seed)
    Rr, Rg = _pca_rotations(frame.rn), _pca_rotations(frame.gn)

    best, trace = None, []
    for flip in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        R0 = Rg.T @ np.diag(flip) @ Rr
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T = _icp_refine(frame, T0, voxel, with_scaling=with_scaling)
        sn, Rn, tn = _unpack(T)
        sc, _, _ = frame.score(sn, Rn, tn)
        trace.append({"flip": list(flip), "score": sc})
        if best is None or sc < best[0]:
            best = (sc, sn, Rn, tn)

    _, sn, Rn, tn = best
    return _result(frame, sn, Rn, tn, n_hypotheses=len(trace), trace=trace)


def _require_teaserpp():
    try:
        import teaserpp_python
    except ImportError as e:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "TEASER++ needs a source build; see the module docstring "
            "('Installing TEASER++') for the conda-forge path that needs no "
            "sudo, or the apt path if root is available."
        ) from e
    return teaserpp_python


def _mutual_fpfh_correspondences(fsrc, fdst, *, max_corr: int = 2000, seed: int = 0):
    """Mutual-nearest-neighbour index pairs between two FPFH feature sets.

    TEASER++ solves from tentative correspondences, expecting many to be
    wrong - that is the problem it is built to be robust to - rather than
    from two raw point sets the way Open3D's RANSAC does internally. Mutual
    nearest neighbour in descriptor space is the standard way to produce that
    tentative set, and it is the same rule `fpfh_ransac` applies to its own
    matches via Open3D's `mutual_filter=True`; it is written out by hand here
    because TEASER++ needs the index pairs directly.

    `fsrc.data` / `fdst.data` are (33, N) - FPFH dimension by point count -
    so each is transposed to (N, 33) before the nearest-neighbour query.
    """
    a = np.asarray(fsrc.data).T
    b = np.asarray(fdst.data).T
    _, nn_ab = cKDTree(b).query(a, k=1, workers=workers())  # a[i]'s nearest in b
    _, nn_ba = cKDTree(a).query(b, k=1, workers=workers())  # b[j]'s nearest in a
    mutual = np.nonzero(nn_ba[nn_ab] == np.arange(len(a)))[0]
    if len(mutual) > max_corr:
        rng = np.random.default_rng(seed)
        mutual = np.sort(rng.choice(mutual, max_corr, replace=False))
    return mutual, nn_ab[mutual]


def align_teaserpp(
    recon: np.ndarray,
    gt: np.ndarray,
    *,
    voxel: float = 0.05,
    overlap: float = 0.8,
    gt_overlap: float = 0.45,
    refine: bool = True,
    with_scaling: bool = True,
    max_correspondences: int = 2000,
    seed: int = 0,
) -> dict:
    """FPFH correspondences + TEASER++'s outlier-robust solver (GNC-TLS with
    maximum-clique inlier selection), finished with the same ICP polish as the
    other backends.

    Parameters mirror the reference example distributed with TEASER++
    (`examples/teaser_python_fpfh_icp`): `cbar2=1.0`, `rotation_gnc_factor=1.4`,
    `rotation_max_iterations=10000`, `rotation_cost_threshold=1e-16`. Two
    deviations, both deliberate. `estimate_scaling` follows this function's own
    `with_scaling` rather than the example's hardcoded `False`, for the same
    reason every other backend re-estimates scale (see 'Scale' in the module
    docstring): a rigid-only comparison would understate what a 7-DoF free
    search can reach. `inlier_selection_mode` defaults to the heuristic
    max-clique (`PMC_HEU`) rather than the example's exact solver
    (`PMC_EXACT`), because exact max-clique has no runtime guarantee as the
    correspondence count grows and this runs over many view sets unattended;
    `getattr` falls back to `PMC_EXACT` if a binding is missing it, so this
    does not hard-fail across TEASER++ versions.

    `noise_bound` is set from `voxel`, the one parameter this module already
    exposes and sweeps for the other backends, rather than introduced as a
    second free knob: 1.5 * voxel matches the correspondence-distance
    threshold `fpfh_ransac` and `fgr` use in the same normalised frame.
    """
    teaserpp_python = _require_teaserpp()
    frame = _Frame(recon, gt, overlap=overlap, gt_overlap=gt_overlap, seed=seed)
    (src, fsrc), (dst, fdst) = _o3d_pair(frame, voxel)

    idx_src, idx_dst = _mutual_fpfh_correspondences(fsrc, fdst, max_corr=max_correspondences, seed=seed)
    if len(idx_src) < 3:
        raise RuntimeError(
            f"only {len(idx_src)} mutual FPFH correspondences - too few for "
            "TEASER++ (needs >=3). The object is likely too sparse at this "
            "voxel size."
        )
    A = np.asarray(src.points)[idx_src].T  # (3, M), TEASER++'s expected shape
    B = np.asarray(dst.points)[idx_dst].T

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.cbar2 = 1.0
    params.noise_bound = 1.5 * voxel
    params.estimate_scaling = with_scaling
    mode_enum = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE
    params.inlier_selection_mode = getattr(mode_enum, "PMC_HEU", mode_enum.PMC_EXACT)
    params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 10000
    params.rotation_cost_threshold = 1e-16

    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(A, B)
    sol = solver.getSolution()

    T0 = np.eye(4)
    T0[:3, :3] = (float(sol.scale) if with_scaling else 1.0) * np.asarray(sol.rotation)
    T0[:3, 3] = np.asarray(sol.translation)
    T = _icp_refine(frame, T0, voxel, with_scaling=with_scaling) if refine else T0
    sn, Rn, tn = _unpack(T)
    return _result(frame, sn, Rn, tn, n_hypotheses=1, trace=[{"n_correspondences": int(len(idx_src))}])


BACKENDS = {
    "fpfh_ransac": align_fpfh_ransac,
    "fgr": align_fgr,
    "icp_pca": align_icp_pca,
    "teaserpp": align_teaserpp,
}
