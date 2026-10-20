"""tools/stage_heads.py refuses a wrong or damaged checkpoint and stages the right ones under the release names."""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("stage_heads", ROOT / "tools" / "stage_heads.py")
stage_heads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_heads)


def make(tmp_path, content=b"weights"):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "a_seg_head.pt").write_bytes(content)
    (ck / "b_seg_head.pt").write_bytes(b"other")
    manifest = tmp_path / "heads.json"
    entry = lambda f, role, data: {  # noqa: E731
        "role": role,
        "trained_as": f,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest.write_text(
        json.dumps(
            {
                "heads": {
                    "a.pt": entry("a_seg_head.pt", "main", b"weights"),
                    "variants/b.pt": entry("b_seg_head.pt", "split check", b"other"),
                }
            }
        )
    )
    card = tmp_path / "card.md"
    card.write_text("# card")
    return ck, manifest, card


def test_stages_every_head_under_its_release_name(tmp_path):
    ck, manifest, card = make(tmp_path)
    assert stage_heads.stage(ck, tmp_path / "out", manifest, card) == []
    assert (tmp_path / "out" / "a.pt").read_bytes() == b"weights"
    assert (tmp_path / "out" / "variants" / "b.pt").read_bytes() == b"other"
    assert (tmp_path / "out" / "README.md").read_text() == "# card"
    sums = (tmp_path / "out" / "SHA256SUMS").read_text().splitlines()
    assert sums[0].endswith("  a.pt") and sums[1].endswith("  variants/b.pt")


def test_only_main_leaves_out_the_variants(tmp_path):
    ck, manifest, card = make(tmp_path)
    assert stage_heads.stage(ck, tmp_path / "out", manifest, card, only="main") == []
    assert not (tmp_path / "out" / "variants").exists()


def test_a_damaged_file_stops_everything(tmp_path):
    ck, manifest, card = make(tmp_path, content=b"wrong")
    problems = stage_heads.stage(ck, tmp_path / "out", manifest, card)
    assert problems and "a_seg_head.pt" in problems[0]
    assert not (tmp_path / "out").exists()


def test_a_missing_file_is_named(tmp_path):
    ck, manifest, card = make(tmp_path)
    (ck / "b_seg_head.pt").unlink()
    assert "b_seg_head.pt: not in" in stage_heads.stage(ck, tmp_path / "out", manifest, card)[0]


def test_the_released_manifest_is_consistent():
    heads = json.loads((ROOT / "configs" / "heads.json").read_text())["heads"]
    assert len(heads) == 9
    assert all(len(h["sha256"]) == 64 and h["bytes"] > 1e8 for h in heads.values())
    assert {h["role"] for h in heads.values()} == {"main", "unfamiliar scenes", "split check", "training-size study"}
    assert all(h["num_classes"] == len(h["class_names"]) for h in heads.values())
