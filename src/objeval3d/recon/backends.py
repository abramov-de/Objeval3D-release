"""The feed-forward reconstruction models behind one interface, so the protocol compares models rather than pipelines.

The protocol's stages 2-4 never care which model produced the cloud: they take depth, cameras and a mask. Both models are wrapped
to present the same `inference()` and the same `Prediction`, and the two already agree on the conventions that matter: both return
extrinsics as (S,3,4) world-to-camera in the OpenCV convention, both return per-view intrinsics, and both are scale-free, which the
protocol handles anyway because the reference mode leaves exactly one scale parameter open.

Environments are separate on purpose
------------------------------------
Depth Anything 3 and VGGT cannot share a conda environment (VGGT needs its `training` package, which DA3's environment has no copy
of). So the import of each backend is deferred into its own constructor, and nothing at module scope touches either model: this
file imports in both environments, and so does everything that uses it. Anything shared between the two models is therefore
computed once and passed on disk: the masks of the segmentation head are dumped once (`objeval3d.seg.export_masks`) and given to
both backends with `--mask-source dir`.

Pinned versions are in `objeval3d.models`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from objeval3d.models import DA3_REPO, DA3_REVISION, PROCESS_RES, VGGT_REPO, VGGT_REVISION


@dataclass
class Prediction:
    """What stage 2 reads of a prediction: depth (N,H,W), conf (N,H,W) or None, intrinsics (N,3,3), extrinsics (N,3,4)
    world-to-camera, processed_images (N,H,W,3) uint8, seg_logits (N,C,H,W) or None, is_metric."""

    depth: np.ndarray
    conf: np.ndarray | None
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    processed_images: np.ndarray
    seg_logits: np.ndarray | None = None
    is_metric: bool = False
    # DA3's metric variant reports the factor it applied; a scale-free model has none and records null.
    scale_factor: float | None = None


class DA3Predictor:
    """Depth Anything 3 (DA3-LARGE), optionally with a trained segmentation head attached to its frozen backbone.

    The head is not part of the model: it is attached from outside (`objeval3d.seg.head`), so the upstream package of Depth
    Anything 3 is used as it is.
    """

    def __init__(
        self, device: str = "cuda", base_model: str = DA3_REPO, revision: str | None = DA3_REVISION, head_ckpt: str = ""
    ):
        import torch
        from depth_anything_3.api import DepthAnything3

        from objeval3d.seg.head import load_head

        self.torch = torch
        self.device = device
        kw = {} if os.path.exists(base_model) or not revision else {"revision": revision}
        self.model = DepthAnything3.from_pretrained(base_model, **kw).to(device).eval()
        self.head, self.ckpt = (None, {})
        if head_ckpt:
            self.head, self.ckpt = load_head(head_ckpt, device)

    def inference(self, image, process_res: int = PROCESS_RES, infer_seg: bool = False, export_dir=None) -> Prediction:
        from objeval3d.seg.head import FeatureTap, head_logits

        if infer_seg and self.head is None:
            raise ValueError("infer_seg=True needs a segmentation head: pass head_ckpt")
        seg = None
        with self.torch.no_grad():
            if infer_seg:
                with FeatureTap(self.model.model.backbone) as tap:
                    pred = self.model.inference(image=image, process_res=process_res, export_dir=export_dir)
                h, w = pred.processed_images.shape[1:3]
                seg = head_logits(self.head, tap.feats, h, w).squeeze(0).float().cpu().numpy()
            else:
                pred = self.model.inference(image=image, process_res=process_res, export_dir=export_dir)
        sf = getattr(pred, "scale_factor", None)
        return Prediction(
            depth=np.asarray(pred.depth),
            conf=None if pred.conf is None else np.asarray(pred.conf),
            intrinsics=np.asarray(pred.intrinsics),
            extrinsics=np.asarray(pred.extrinsics),
            processed_images=np.asarray(pred.processed_images),
            seg_logits=seg,
            # upstream builds these with getattr() on an addict.Dict, which returns an empty Dict instead of the default for a
            # missing key, so a non-metric model yields Dict(), not 0: only the truthiness can be trusted
            is_metric=bool(getattr(pred, "is_metric", 0)),
            scale_factor=float(sf) if isinstance(sf, (int, float)) else None,
        )


class VGGTPredictor:
    """VGGT-1B behind the same `inference()` signature.

    VGGT is a git submodule (`third_party/vggt-repo`, pinned in `objeval3d.models`), not a copy, because it ships under Meta's
    research licence and vendoring its source would carry that licence's obligations. Weights are not a path: `from_pretrained`
    pulls `facebook/VGGT-1B` from the Hugging Face cache. The checkout is named `vggt-repo` and not `vggt` on purpose: a directory
    named after the package would shadow it on `sys.path`.
    """

    def __init__(self, device: str = "cuda", repo: str | None = None, revision: str | None = VGGT_REVISION):
        import sys

        import torch

        repo = (
            repo
            or os.environ.get("OBJEVAL_VGGT_REPO")
            or str(Path(__file__).resolve().parents[3] / "third_party" / "vggt-repo")
        )
        if not (Path(repo) / "vggt").is_dir():
            raise FileNotFoundError(
                f"VGGT source not found under {repo}. It is a git submodule: run "
                "`git submodule update --init third_party/vggt-repo`, or point --vggt-repo (or $OBJEVAL_VGGT_REPO) at an existing "
                "checkout."
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from vggt.models.vggt import VGGT

        self.torch = torch
        self.device = device
        self.model = VGGT.from_pretrained(VGGT_REPO, revision=revision).to(device).eval()

    @staticmethod
    def _assert_no_crop(paths, target: int = 518) -> None:
        """Refuse any frame VGGT's preprocessing would centre-crop.

        This is the one place the two models could disagree silently. Masks are produced at full frame and brought to the
        prediction's resolution with a plain resize, which is exact as long as the model's preprocessing is a plain resize too.
        VGGT's default `mode="crop"` sets the width to 518 and the height to the nearest multiple of 14 - and then centre-crops
        the height if that exceeds 518. For a landscape frame it never does (BOP's 640x480 becomes 518x392), so the resize is
        exact and the masks land where they should. For a portrait frame it does (480x640 becomes 518x686, then crops to 518),
        and every mask would be silently offset by 84 rows.

        Nothing downstream could detect that: the cloud would simply be built from the wrong pixels and the metrics would be
        quietly wrong. So it is checked here, per frame, rather than assumed from the dataset.
        """
        from PIL import Image

        for pth in paths:
            with Image.open(pth) as im:
                w, h = im.size
            if round(h * (target / w) / 14) * 14 > target:
                raise ValueError(
                    f"{pth} is {w}x{h}: VGGT's preprocessing would centre-crop it, and the full-frame masks would no longer line "
                    "up with the prediction. Landscape frames are safe (640x480 -> 518x392). Handling this properly means "
                    "carrying the crop box through to the masks, which is not implemented because BOP has no portrait frames."
                )

    def inference(self, image, process_res: int | None = None, infer_seg: bool = False, export_dir=None) -> Prediction:
        """Signature matches DA3Predictor; two of its arguments have no meaning here.

        `infer_seg` is refused rather than ignored: the segmentation head is trained on DA3's frozen backbone and does not exist
        for VGGT, so a caller asking for it has a bug, not a preference. Use `--mask-source dir` with masks dumped from the DA3
        head, which is what a comparison of the two models needs anyway: the masks must be identical.

        `process_res` is ignored because VGGT's preprocessing fixes the width at 518 px. The resolution actually used is recorded
        in the run's `hw`, so nothing downstream has to assume it.
        """
        if infer_seg:
            raise ValueError(
                "VGGT has no segmentation head - it is trained on DA3's backbone. Use --mask-source dir|gt."
            )
        if not all(isinstance(p, (str, Path)) for p in image):
            raise ValueError(
                "the VGGT backend takes image paths, not arrays: reimplementing VGGT's own resize-and-crop on arrays would risk a "
                "silent mismatch with its training distribution."
            )

        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        paths = [str(p) for p in image]
        self._assert_no_crop(paths)
        imgs = load_and_preprocess_images(paths).to(self.device)
        H, W = imgs.shape[-2:]
        # bf16 autocast over the trunk, as in VGGT's own demo; the heads disable it themselves.
        with (
            self.torch.no_grad(),
            self.torch.autocast(
                self.device.split(":")[0], dtype=self.torch.bfloat16, enabled=self.device.startswith("cuda")
            ),
        ):
            pred = self.model(imgs)

        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (H, W))
        conf = pred.get("depth_conf")
        return Prediction(
            depth=pred["depth"][0, ..., 0].float().cpu().numpy(),
            conf=None if conf is None else conf[0].float().cpu().numpy(),
            intrinsics=intr[0].float().cpu().numpy(),
            extrinsics=extr[0].float().cpu().numpy(),
            processed_images=(imgs.permute(0, 2, 3, 1).float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8),
            seg_logits=None,
            # VGGT is scale-free, like DA3 without the metric head: the reference mode leaves exactly one scale parameter free
            is_metric=False,
        )


def load_predictor(
    backend: str,
    *,
    device: str = "cuda",
    base_model: str = DA3_REPO,
    revision: str | None = DA3_REVISION,
    head_ckpt: str = "",
    vggt_repo: str | None = None,
):
    """The one place that knows which model is being run.

    Returns `(predictor, ckpt)` from **every** branch. `ckpt` is the segmentation head's checkpoint dict, empty for any backend
    or configuration without a head; the caller reads `num_classes`, `class_names` and `obj_ids` out of it.
    """
    if backend == "vggt":
        return VGGTPredictor(device=device, repo=vggt_repo), {}
    if backend != "da3":
        raise ValueError(f"unknown backend {backend!r} (da3 | vggt)")
    predictor = DA3Predictor(device=device, base_model=base_model, revision=revision, head_ckpt=head_ckpt)
    return predictor, predictor.ckpt
