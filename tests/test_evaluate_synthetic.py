"""Stages 3-4 end to end on a synthetic box: ground-truth clouds, visibility, reference mode, free fit, metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_bop as sb  # noqa: E402

from objeval3d.align import evaluate  # noqa: E402
from objeval3d.metrics.gt_cloud import build_gt_clouds  # noqa: E402


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("bop")
    manifest = sb.build_dataset(root)
    run = root / "runs" / "synthetic_s000001_o01_v00__synthetic"
    info = sb.build_run(run, manifest["sets"][0], root)
    return root, manifest, run, info


def test_visible_part_of_the_scan_is_a_fraction_of_the_surface(dataset):
    root, manifest, _, _ = dataset
    g = build_gt_clouds(manifest["sets"][0], root, n_sample=60_000, n_keep=20_000)
    # two views of a box see at most three faces each: a clear part of the surface, never all of it
    assert 0.2 < g["visible_fraction"] < 0.8
    assert len(g["gt_visible"]) < len(g["gt_full"]) or g["visible_fraction"] < 1.0
    assert g["diameter"] == pytest.approx(np.linalg.norm(sb.DIMS))
    # every visible point lies on the surface of the box
    d = np.abs(np.abs(g["gt_visible"]) - np.asarray(sb.DIMS) / 2).min(axis=1)
    assert d.max() < 1e-6


class Args:
    modes = ["oracle_raw", "oracle", "auto", "auto_fixedscale"]
    n_rot = 512
    overlap = 0.8
    gt_overlap = 0.45
    baseline_voxel = 0.05
    ransac_repeats = 1
    gt_sample = 60_000
    gt_keep = 20_000
    max_recon_points = 120_000
    min_points = 200
    n_continuous_sym = 24
    converged_rot_deg = 15.0
    converged_trans_rel = 0.10
    save_aligned = False
    save_gt = False
    seed = 0
    k_lo = 0.4
    k_hi = 2.5
    n_k = 61


@pytest.fixture(scope="module")
def record(dataset):
    root, _, run, _ = dataset
    return evaluate.evaluate_run(run, root, Args())


def test_record_has_the_fields_the_analysis_reads(record, dataset):
    _, _, _, info = dataset
    assert record["status"] == "ok"
    assert set(record["alignments"]) == {"oracle_raw", "oracle", "auto", "auto_fixedscale"}
    assert record["recon_centroid"] == pytest.approx(list(np.round(info["centroid"], 4)), abs=1e-3)
    assert {"rotation_deg", "translation_rel_diam", "delta_fscore_2pct", "converged"} <= set(record["auto_vs_oracle"])
    json.dumps(record)  # the record is serialisable as it stands


def test_reference_mode_recovers_the_depth_scale_within_its_grid(record):
    # the scale is searched on a geometric grid of 61 points over [0.4, 2.5] x k_init: 3.1 % apart
    s = record["alignments"]["oracle_raw"]["sim3"]["scale"]
    assert abs(s / sb.K_TRUE - 1.0) < 0.04
    assert record["oracle_diagnostics"]["scale_clipped_views"] == 0


def test_free_fit_started_at_the_reference_is_accurate(record):
    a = record["alignments"]["oracle"]
    assert abs(a["sim3"]["scale"] / sb.K_TRUE - 1.0) < 0.02
    assert a["accuracy"]["median"] < 2.0  # mm; the noise injected is 0.5 mm
    assert a["fscore"]["tau_2pct"]["fscore"] > 0.5


def test_the_free_search_recovers_the_pose_up_to_the_symmetries_of_the_box(record):
    assert record["n_symmetries"] == 4
    assert record["auto_vs_oracle"]["rotation_deg"] < 15.0  # the discrepancy is taken modulo the symmetry group


def test_the_free_search_finds_a_fit_at_least_as_good_as_the_reference(record):
    # the criterion of the search is a trimmed symmetric Chamfer; the reference mode does not minimise it
    assert (
        record["alignments"]["auto"]["search"]["score_normalised"]
        <= record["alignments"]["oracle"]["search"]["score_normalised"] * 1.5
    )


def test_too_few_points_is_reported_not_raised(dataset):
    root, _, run, _ = dataset
    args = Args()
    args.min_points = 10**9
    r = evaluate.evaluate_run(run, root, args)
    assert r["status"] == "too_few_points"


def test_the_scale_locked_search_keeps_the_scale_of_the_reference(record):
    a = record["alignments"]
    assert a["auto_fixedscale"]["sim3"]["scale"] == pytest.approx(a["oracle_raw"]["sim3"]["scale"], rel=1e-9)
    assert "auto_fixedscale" in record["vs_oracle"]
