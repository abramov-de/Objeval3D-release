"""The Sim(3) aligner on synthetic surface clouds.

Builds surface clouds (sphere, box, cylinder, drill-like), keeps only a visible cap of each (as a real reconstruction would), applies
a known similarity transform plus noise plus a controlled amount of segmentation leakage, and checks that `align_sim3` recovers the
scale and produces sane metrics.

Symmetric shapes (sphere, box, cylinder) are aligned only up to their symmetry group, so their *completeness* against a fixed
visible cap is meaningless here and is not asserted - which is why the real pipeline resolves symmetries explicitly
(`objeval3d.align.evaluate.resolve_symmetry`). Scale and accuracy are asserted for every shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from objeval3d.align.registration import align_sim3, apply_sim3, reconstruction_metrics


def surf_sphere(rng, n=20000, r=50.0):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * r


def surf_box(rng, n=20000, dims=(80, 50, 120)):
    d = np.array(dims) / 2.0
    p = (rng.random((n, 3)) * 2 - 1) * d
    f = rng.integers(0, 6, n)
    for i in range(6):
        ax, sgn = i // 2, 1 if i % 2 == 0 else -1
        p[f == i, ax] = sgn * d[ax]
    return p


def surf_cyl(rng, n=20000, r=25.0, h=100.0):
    p = np.empty((n, 3))
    a = rng.random(n) * 2 * np.pi
    side = rng.random(n) < 0.75
    p[side, 0] = r * np.cos(a[side])
    p[side, 1] = r * np.sin(a[side])
    p[side, 2] = (rng.random(side.sum()) - 0.5) * h
    ns = int((~side).sum())
    rr = r * np.sqrt(rng.random(ns))
    p[~side, 0] = rr * np.cos(a[~side])
    p[~side, 1] = rr * np.sin(a[~side])
    p[~side, 2] = np.where(rng.random(ns) < 0.5, -h / 2, h / 2)
    return p


def surf_drill(rng, n=20000):
    body = surf_cyl(rng, int(n * 0.6), 18.0, 90.0)
    handle = surf_box(rng, int(n * 0.4), (24, 20, 70))
    handle[:, 2] -= 60
    handle[:, 0] += 10
    return np.vstack([body, handle])


SHAPES = {"sphere": surf_sphere, "box": surf_box, "cyl": surf_cyl, "drill": surf_drill}
CASES = [(0.00, 0.0), (0.00, 0.8), (0.06, 0.8)]  # (leakage fraction, noise mm)


# Each case takes about 20 s (the search is exhaustive), so the default run keeps one and `pytest -m slow` runs the whole matrix.
MATRIX = [
    pytest.param(
        name,
        leak,
        noise,
        marks=() if (name, leak, noise) == ("box", 0.00, 0.8) else pytest.mark.slow,
        id=f"{name}-leak{leak:.2f}-noise{noise}",
    )
    for name in SHAPES
    for leak, noise in CASES
]


@pytest.mark.parametrize("name,leak,noise", MATRIX)
def test_scale_and_accuracy_are_recovered(name, leak, noise):
    rng = np.random.default_rng(0)
    gt = SHAPES[name](rng)
    diam = float(np.linalg.norm(gt.max(0) - gt.min(0)))
    vis = gt[:, 2] > np.percentile(gt[:, 2], 45)
    src = gt[vis] + (rng.normal(scale=noise, size=gt[vis].shape) if noise else 0.0)
    if leak:
        extra = rng.normal(scale=50, size=(int(leak * vis.sum()), 3)) + np.array([0, 0, 90])
        src = np.vstack([src, extra])

    s_true = 0.0137
    R_true = Rotation.random(random_state=7).as_matrix()
    recon = (s_true * (R_true @ src.T)).T + np.array([3.0, -2.0, 7.0])

    res = align_sim3(recon, gt)
    aligned = apply_sim3(recon, res["scale"], res["R"], res["t"])
    m = reconstruction_metrics(aligned, gt, gt[vis], diameter=diam)

    assert abs(res["scale"] * s_true - 1.0) < 0.08, "scale error above 8 %"
    # the median accuracy stays near the injected noise; leakage inflates the mean by construction, which is the point of the metric
    assert m["accuracy"]["median"] < max(3.0, 3.0 * max(noise, 0.5))


def test_fixed_scale_keeps_the_scale():
    rng = np.random.default_rng(1)
    gt = surf_drill(rng)
    vis = gt[:, 2] > np.percentile(gt[:, 2], 45)
    s_true = 0.02
    R_true = Rotation.random(random_state=3).as_matrix()
    recon = (s_true * (R_true @ gt[vis].T)).T + np.array([1.0, 2.0, 3.0])
    res = align_sim3(recon, gt, fixed_world_scale=1.0 / s_true)
    assert res["scale"] == pytest.approx(1.0 / s_true, rel=1e-9)


def test_thread_count_follows_the_cpu_affinity(monkeypatch):
    """scipy's workers=-1 would start os.cpu_count() threads however few CPUs the process may use."""
    import os

    from objeval3d.threads import workers

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2})
    assert workers() == 3
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set())
    assert workers() == 1
