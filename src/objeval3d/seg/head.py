"""The segmentation head: a DPT decoder trained on the frozen backbone of Depth Anything 3.

The head is the decoder of Depth Anything 3's own depth head (`depth_anything_3.model.dpt.DPT`) with a different final layer: it
reads the same four intermediate feature maps of the backbone as the depth head does and outputs one logit per class (background
plus the objects of the capture). It has 32.65 M parameters, the backbone, the depth head and the camera head are not touched, and
Table 4 of the paper gives what it adds to the model's size and to the time of a forward pass.

Nothing of Depth Anything 3 is copied or patched. The head is an upstream `DPT` module, and it is attached from outside: a forward
hook on the backbone hands over the features the backbone gives to the other heads, and `head_logits` runs the decoder on them in
full precision, which is what the model's own forward does for each of its heads.

    head = load_head(ckpt_path)                      # a trained head
    with FeatureTap(da3.model.backbone) as tap:      # da3: depth_anything_3.api.DepthAnything3
        prediction = da3.inference(image=paths, process_res=504)
    logits = head_logits(head, tap.feats, h, w)      # [1, views, classes, h, w]

The checkpoints written by `objeval3d.seg.train` hold the head's weights and the class list of the capture (`num_classes`,
`class_names`, `obj_ids`, `dataset`) together with how it was trained.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from objeval3d.models import DA3_REPO, DA3_REVISION

# The configuration of the segmentation head. `output_dim=1` is deliberate: the DPT module reads a width above one as "value plus
# confidence" and splits the last channel off, so the head is built with one output and its last convolution is then replaced by
# one with `num_classes` outputs (`build_head`).
HEAD_CONFIG = dict(
    dim_in=2048,
    output_dim=1,
    activation="linear",
    features=256,
    out_channels=[256, 512, 1024, 1024],
    head_name="seg_logits",
    use_sky_head=False,
    pos_embed=False,
)


def build_head(num_classes: int, *, init_output: bool = True) -> nn.Module:
    """A new head for `num_classes` classes (background included).

    The final convolution gets Xavier-uniform weights and a zero bias when `init_output` is set, which is how training starts; a
    head that is about to be filled from a checkpoint does not need it.
    """
    from depth_anything_3.model.dpt import DPT

    head = DPT(**HEAD_CONFIG)
    layers = list(head.scratch.output_conv2)
    old: nn.Conv2d = layers[-1]
    new_final = nn.Conv2d(old.in_channels, num_classes, kernel_size=1)
    if init_output:
        nn.init.xavier_uniform_(new_final.weight)
        nn.init.zeros_(new_final.bias)
    head.scratch.output_conv2 = nn.Sequential(*layers[:-1], new_final)
    return head


def load_backbone(device: str = "cuda", base_model: str = DA3_REPO, revision: str | None = DA3_REVISION) -> nn.Module:
    """The frozen backbone of Depth Anything 3 on `device`, without the depth and camera heads (they are not needed to train or
    to score the segmentation head and are not moved to the GPU)."""
    from depth_anything_3.api import DepthAnything3

    kw = {} if os.path.exists(base_model) or not revision else {"revision": revision}
    backbone = DepthAnything3.from_pretrained(base_model, **kw).model.backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def load_head(ckpt_path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """The trained head and its checkpoint dict (class list, training record), in evaluation mode on `device`."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    head = build_head(int(ckpt.get("num_classes", 2)), init_output=False)
    head.load_state_dict(ckpt["seg_head"])
    return head.to(device).eval(), ckpt


class FeatureTap:
    """Keeps the features that the backbone of Depth Anything 3 returns, for the heads that are attached from outside.

    The backbone returns `(features, auxiliary features)`; the first element is what every head of the model reads. Use it as a
    context manager around the call that runs the model, or call `close()`.
    """

    def __init__(self, backbone: nn.Module):
        self.feats = None
        self._handle = backbone.register_forward_hook(self._store)

    def _store(self, module, args, output):
        self.feats = output[0]

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "FeatureTap":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def head_logits(head: nn.Module, feats, h: int, w: int) -> torch.Tensor:
    """Class logits [batch, views, classes, h, w] for the features of one forward pass at the processing resolution (h, w).

    Run in full precision with autocast off, exactly as the model's own forward runs its heads. Gradients flow if the caller
    allows them (training) and not otherwise.
    """
    device_type = feats[0][0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return head(feats, h, w, patch_start_idx=0)["seg_logits"]


def forward_logits(backbone: nn.Module, head: nn.Module, images: torch.Tensor, *, amp: bool = False) -> torch.Tensor:
    """[B,3,H,W] -> [B,C,H,W], the way the head is trained and scored on single frames.

    Each image is fed as its own one-view "scene": the segmentation is per view, and mixing unrelated images into one multi-view
    batch would let the cross-view attention of the backbone hallucinate a scene. The backbone is frozen (no gradient), `amp` runs
    it in bfloat16; the head itself always runs in full precision.
    """
    x = images[:, None]
    with torch.no_grad():
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
            feats, _ = backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
    return head_logits(head, feats, x.shape[-2], x.shape[-1])[:, 0]
