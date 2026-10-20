"""Segmentation dataset and view sets from a synthetic BOP scene (a box seen from an orbiting camera)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_bop as sb  # noqa: E402

from objeval3d.data.bop import BopScene, angle_between, viewing_direction  # noqa: E402
from objeval3d.data.manifest import load_manifest  # noqa: E402
from objeval3d.seg.prepare_data import IGNORE_INDEX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(module: str, *args) -> str:
    out = subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


@pytest.fixture(scope="module")
def orbit(tmp_path_factory):
    root = tmp_path_factory.mktemp("orbit")
    sb.build_orbit_dataset(root / "bop")
    seg = root / "seg"
    run(
        "objeval3d.seg.prepare_data",
        "--bop-root",
        root / "bop",
        "--dataset",
        "synthetic",
        "--split",
        "test",
        "--scenes",
        "1",
        "--obj-ids",
        "1",
        "--split-by",
        "block",
        "--n-blocks",
        "5",
        "--min-px",
        "400",
        "--out-dir",
        seg,
    )
    return root, seg


def test_block_split_cycles_train_train_train_val_test(orbit):
    _, seg = orbit
    idx = json.loads((seg / "index.json").read_text())
    n = {k: len(v) for k, v in idx["splits"].items()}
    assert sum(n.values()) == 60
    assert n == {"train": 36, "val": 12, "test": 12}  # five blocks of 12 frames: train, train, train, val, test
    im_ids = [r["im_id"] for r in idx["splits"]["test"]]
    assert im_ids == list(range(48, 60))


def test_labels_are_the_visible_masks(orbit):
    root, seg = orbit
    stem = json.loads((seg / "index.json").read_text())["splits"]["test"][0]["stem"]
    label = cv2.imread(str(seg / "test" / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    im_id = int(stem.rsplit("_i", 1)[1])
    visib = (
        cv2.imread(
            str(root / "bop" / "test" / "000001" / "mask_visib" / f"{im_id:06d}_000000.png"), cv2.IMREAD_GRAYSCALE
        )
        > 127
    )
    assert set(np.unique(label)) == {0, 1}
    assert ((label == 1) == visib).all()
    classes = json.loads((seg / "classes.json").read_text())
    assert classes["obj_to_class"] == {"1": 1} and classes["ignore_index"] == IGNORE_INDEX


def test_view_sets_keep_to_the_test_frames_and_the_angle_band(orbit):
    root, seg = orbit
    out = root / "man.json"
    run(
        "objeval3d.data.view_sets",
        "build",
        "--bop-root",
        root / "bop",
        "--dataset",
        "synthetic",
        "--split",
        "test",
        "--scenes",
        "1",
        "--obj-ids",
        "1",
        "--seg-index",
        seg / "index.json",
        "--use-split",
        "test",
        "--n-views",
        "3",
        "--sets-per-object",
        "4",
        "--min-angle",
        "12",
        "--max-angle",
        "50",
        "--min-px",
        "400",
        "--out",
        out,
    )
    man = load_manifest(out)
    test_frames = {r["im_id"] for r in json.loads((seg / "index.json").read_text())["splits"]["test"]}
    assert 1 <= len(man["sets"]) <= 4
    scene = BopScene(root / "bop" / "test" / "000001")
    for s in man["sets"]:
        assert set(s["im_ids"]) <= test_frames
        d = [
            viewing_direction(np.asarray(R).reshape(3, 3), np.asarray(t).reshape(3, 1))
            for R, t in zip(s["R_m2c"], s["t_m2c"])
        ]
        for a in range(3):
            for b in range(a + 1, 3):
                assert 12 <= angle_between(d[a], d[b]) <= 50
        assert Path(s["images"][0]).exists()
    assert scene.im_ids[0] == 0


def test_extension_keeps_the_original_frames(orbit):
    root, seg = orbit
    man3, man6 = root / "man.json", root / "man6.json"
    if not man3.exists():
        pytest.skip("depends on the previous test")
    run(
        "objeval3d.data.view_sets",
        "extend",
        "--manifest",
        man3,
        "--bop-root",
        root / "bop",
        "--seg-index",
        seg / "index.json",
        "--n-views",
        "5",
        "--min-angle",
        "6",
        "--max-angle",
        "60",
        "--out",
        man6,
    )
    a, b = load_manifest(man3, root / "bop"), load_manifest(man6, root / "bop")
    base = {s["set_id"]: s for s in a["sets"]}
    assert b["sets"], "no set could be extended"
    for s in b["sets"]:
        assert set(base[s["set_id"]]["im_ids"]) <= set(s["im_ids"]) and len(s["im_ids"]) == 5
    assert b["extension_angle_band"] == [6, 60]
