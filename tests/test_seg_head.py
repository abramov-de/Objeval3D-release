"""The segmentation head and the way it is attached to the backbone of Depth Anything 3, without the weights of any model.

Needs the upstream package of Depth Anything 3 (`pip install` from the commit in `objeval3d.models`) for its `DPT` module; skipped when
it is not installed.
"""

import pytest

pytest.importorskip("depth_anything_3")
torch = pytest.importorskip("torch")

from objeval3d.seg.head import HEAD_CONFIG, FeatureTap, build_head, head_logits, load_head  # noqa: E402

PATCH = 14


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def fake_features(views=2, grid=3, dim=HEAD_CONFIG["dim_in"], seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, views, grid * grid, dim, generator=g),) for _ in range(4)]


class Backbone(torch.nn.Module):
    """Returns what the backbone of DA3 returns: (features for the heads, auxiliary features)."""

    def __init__(self, feats):
        super().__init__()
        self.feats = feats

    def forward(self, x):
        return self.feats, None


def test_the_head_has_the_size_of_the_trained_ones():
    # 32,650,697 parameters for the nine classes of LM-O and 32,650,862 for the fourteen of YCB-V (training logs, weight files)
    assert n_params(build_head(9)) == 32_650_697
    assert n_params(build_head(14)) == 32_650_862


def test_the_output_has_one_logit_map_per_class():
    head = build_head(5).eval()
    with torch.no_grad():
        logits = head_logits(head, fake_features(views=2, grid=3), 3 * PATCH, 3 * PATCH)
    assert logits.shape == (1, 2, 5, 3 * PATCH, 3 * PATCH)


def test_the_tap_keeps_the_features_the_backbone_gives_to_the_heads():
    feats = fake_features()
    backbone = Backbone(feats)
    with FeatureTap(backbone) as tap:
        backbone(torch.zeros(1))
    assert tap.feats is feats
    # closed: a later call does not overwrite what was taken
    backbone.feats = fake_features(seed=1)
    backbone(torch.zeros(1))
    assert tap.feats is feats


def test_a_saved_head_loads_and_gives_the_same_logits(tmp_path):
    head = build_head(5).eval()
    for p in head.parameters():  # make the head non-trivial: the final layer starts at zero bias
        torch.nn.init.normal_(p, std=0.02)
    path = tmp_path / "head.pt"
    torch.save(
        {
            "seg_head": head.state_dict(),
            "num_classes": 5,
            "class_names": ["background", "a", "b", "c", "d"],
            "obj_ids": [1, 2, 3, 4],
        },
        path,
    )
    loaded, ckpt = load_head(path)
    feats = fake_features()
    with torch.no_grad():
        a = head_logits(head, feats, 3 * PATCH, 3 * PATCH)
        b = head_logits(loaded, feats, 3 * PATCH, 3 * PATCH)
    assert ckpt["obj_ids"] == [1, 2, 3, 4]
    assert torch.equal(a, b)
    assert not loaded.training


def test_head_logits_runs_in_full_precision_even_under_autocast():
    head = build_head(3).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = head_logits(head, fake_features(views=1), 3 * PATCH, 3 * PATCH)
    assert logits.dtype == torch.float32
