"""Train the segmentation head on the frozen backbone of Depth Anything 3.

The data is the output of `objeval3d.seg.prepare_data` (explicit train/ and val/ folders, `classes.json`), the class names and their
number come from it, so the same command trains the eight-object LM-O head and the thirteen-object YCB-V head. Only the head is
trained (32.65 M parameters of the DPT decoder); the backbone, the depth head and the camera head are not even loaded to the GPU.

* **Aspect-preserving resize matching DA3's own inference preprocessing.** `DepthAnything3.inference()` uses `upper_bound_resize`
  (longest side -> process_res, aspect preserved, each side rounded to a multiple of 14); training on squashed square images and
  running inference on un-squashed ones is a train/test mismatch that costs mask quality for free, so both sides use the same
  geometry (`objeval3d.data.bop.target_hw`).
* **Loss** = cross-entropy (ignored pixels skipped) + soft Dice over the foreground classes present in the target.
* **Validation mIoU** is computed every epoch, over the classes the validation split shows, and the checkpoint is selected on it
  rather than on the loss: the CE+Dice sum keeps improving on the background term long after the object masks stop getting better.

    python -m objeval3d.seg.train --data-dir workspace/seg_data/lmo --out-ckpt workspace/ckpt/lmo_seg_head.pt \\
        --epochs 20 --batch-size 2 --lr 1e-4 --image-size 504 --augment

The checkpoint holds the head's weights and the class list of the capture, which is everything `objeval3d.seg.head.load_head` needs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from objeval3d.data.bop import target_hw
from objeval3d.models import DA3_REPO, DA3_REVISION
from objeval3d.seg.dataset import IMAGE_EXTS, BopSegDataset, check_mask_labels
from objeval3d.seg.head import build_head, forward_logits, load_backbone
from objeval3d.seg.prepare_data import IGNORE_INDEX

log = logging.getLogger("objeval3d.seg.train")


def check_gpu(device: torch.device, need_gb: float = 6.0) -> None:
    """Fail early and legibly when the GPU is already occupied.

    Without this the run dies inside `.to(device)` with a PyTorch traceback whose real message - that another process holds the
    card - is buried in the middle of an allocator dump.
    """
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(idx)
    free_gb, total_gb = free / 2**30, total / 2**30
    log.info(f"GPU {idx} ({torch.cuda.get_device_name(idx)}): {free_gb:.1f} GiB free of {total_gb:.1f} GiB")
    if free_gb < need_gb:
        raise SystemExit(
            f"\nOnly {free_gb:.2f} GiB free on GPU {idx}, need roughly {need_gb:.0f} GiB.\n"
            f"Another process is holding the card. Check with `nvidia-smi`, then either\n"
            f"  - point at a free GPU:   --device cuda:1   (or CUDA_VISIBLE_DEVICES=1)\n"
            f"  - or free this one:      kill <PID from nvidia-smi>\n"
            f"Lowering --batch-size will not help: this fails before any batch is loaded.\n"
        )


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice over the foreground classes **present in the target**.

    Averaging over all classes breaks down as soon as the class list is larger than what a single image contains. YCB-V has 21
    objects and shows about five per frame, so sixteen absent classes each contribute a term of roughly 1.0 that can only be
    reduced by driving their probability to zero. Those terms swamp the handful that carry real signal, and the head collapses to
    predicting almost nothing. On LM-O the problem is invisible because all eight objects appear in every frame.

    Background is excluded: it covers 70-90% of every frame, so its term is nearly constant and would drown the objects.
    """
    probs = torch.softmax(logits, dim=1)
    valid = (targets != IGNORE_INDEX).unsqueeze(1).float()
    onehot = (
        torch.nn.functional.one_hot(targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float() * valid
    )
    p = probs * valid

    inter = (p * onehot).sum(dim=(2, 3))
    den = p.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (den + eps)

    present = onehot.sum(dim=(2, 3))[:, 1:] > 0
    loss = (1.0 - dice[:, 1:]) * present
    n = present.sum()
    return loss.sum() / n if n > 0 else logits.sum() * 0.0


def seg_loss(logits, targets, num_classes, ce_weight=None):
    ce = torch.nn.functional.cross_entropy(logits, targets, weight=ce_weight, ignore_index=IGNORE_INDEX)
    return ce + dice_loss(logits, targets, num_classes)


@torch.no_grad()
def evaluate(backbone, head, loader, device, num_classes, ce_weight=None):
    head.eval()
    losses = []
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    gt_px = np.zeros(num_classes, dtype=np.float64)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = forward_logits(backbone, head, images)
        losses.append(seg_loss(logits, masks, num_classes, ce_weight).item())
        pred = logits.argmax(dim=1)
        valid = masks != IGNORE_INDEX
        for c in range(num_classes):
            p, t = (pred == c) & valid, (masks == c) & valid
            inter[c] += float((p & t).sum())
            union[c] += float((p | t).sum())
            gt_px[c] += float(t.sum())
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    # Two mIoUs, because a split does not put every object in validation. Where a class has no positive pixel in validation its
    # IoU is either NaN (the head predicts it nowhere) or a flat 0 (it predicts it somewhere and every pixel is a false
    # positive); either way validation cannot say whether the head knows it. `miou` is over the classes validation can see and is
    # what the checkpoint is selected on; `miou_all` averages over all of them.
    fg = iou[1:]
    seen = gt_px[1:] > 0
    miou_all = float(np.nanmean(fg)) if np.isfinite(fg).any() else 0.0
    miou = float(np.nanmean(fg[seen])) if seen.any() and np.isfinite(fg[seen]).any() else miou_all
    return (
        float(np.mean(losses)) if losses else 0.0,
        miou,
        {"iou": iou, "present": np.concatenate([[True], seen]), "miou_all": miou_all},
    )


def save_curves(train_losses, val_losses, mious, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ep = list(range(1, len(train_losses) + 1))
    ax1.plot(ep, train_losses, label="train loss", color="#2196F3", lw=2)
    ax1.plot(ep, val_losses, label="val loss", color="#E91E63", lw=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ep, mious, label="val mIoU (fg)", color="#4CAF50", lw=2, ls="--")
    ax2.set_ylabel("mIoU")
    ax2.set_ylim(0, 1)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the segmentation head on a BOP-derived dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--data-dir", required=True, help="output of objeval3d.seg.prepare_data (contains train/, val/, classes.json)"
    )
    ap.add_argument("--base-model", default=DA3_REPO, help="Hugging Face id or a local directory")
    ap.add_argument("--revision", default=DA3_REVISION)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--out-loss-plot", default="")
    ap.add_argument("--log-file", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--image-size", type=int, default=504, help="long side, as in DA3 inference")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the (frozen) backbone")
    ap.add_argument("--max-train", type=int, default=0, help="cap training images (ablation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else out_ckpt.with_name(out_ckpt.stem + "_train_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
    )
    log.info("=" * 70)
    log.info(f"args: {vars(args)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    with open(data_dir / "classes.json") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    num_classes = len(class_names)
    log.info(f"{num_classes} classes: {class_names}")

    probe = next(p for p in sorted((data_dir / "train" / "images").iterdir()) if p.suffix.lower() in IMAGE_EXTS)
    hw = target_hw(probe, args.image_size)
    log.info(f"processing resolution (matches DA3 inference): {hw[0]}x{hw[1]}")

    n_checked = check_mask_labels(data_dir, num_classes)
    log.info(f"mask labels checked on {n_checked} files: all within [0,{num_classes - 1}] or {IGNORE_INDEX}")

    train_ds = BopSegDataset(data_dir / "train", hw, augment=args.augment)
    if args.max_train > 0 and args.max_train < len(train_ds):
        rng = np.random.default_rng(args.seed)
        keep = sorted(rng.choice(len(train_ds), size=args.max_train, replace=False).tolist())
        train_ds.samples = [train_ds.samples[i] for i in keep]
    val_ds = BopSegDataset(data_dir / "val", hw)
    log.info(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    check_gpu(device)

    # Only the backbone goes to the device: the depth and camera heads of the model are never used here.
    backbone = load_backbone(args.device, args.base_model, args.revision)
    head = build_head(num_classes).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n_head = sum(p.numel() for p in head.parameters())
    n_backbone = sum(p.numel() for p in backbone.parameters())
    log.info(f"trainable {n_head:,} (the head); frozen backbone {n_backbone:,}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    train_losses, val_losses, mious = [], [], []
    best_miou, best_epoch = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        batch_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            logits = forward_logits(backbone, head, images, amp=args.amp)
            loss = seg_loss(logits.float(), masks, num_classes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip_norm)
            opt.step()
            batch_losses.append(loss.item())

        tr = float(np.mean(batch_losses)) if batch_losses else 0.0
        vl, miou, ev = evaluate(backbone, head, val_loader, device, num_classes)
        iou, present = ev["iou"], ev["present"]
        train_losses.append(tr)
        val_losses.append(vl)
        mious.append(miou)
        n_seen = int(present[1:].sum())
        log.info(
            f"epoch {epoch:02d}/{args.epochs} | train={tr:.4f} | val={vl:.4f} | "
            f"val_mIoU={miou:.4f} ({n_seen}/{num_classes - 1} classes in val) | "
            f"val_mIoU_all={ev['miou_all']:.4f} | lr={sched.get_last_lr()[0]:.2e}"
        )
        log.info(
            "    per-class IoU: "
            + "  ".join(
                # a class validation never shows is marked, not silently scored 0
                f"{class_names[c]}={iou[c]:.3f}" + ("" if present[c] else "*")
                for c in range(num_classes)
            )
            + ("   (* absent from val, excluded from val_mIoU)" if not present[1:].all() else "")
        )
        sched.step()

        # Selected on mIoU, not loss.
        if miou > best_miou:
            best_miou, best_epoch = miou, epoch
            torch.save(
                {
                    "seg_head": head.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "obj_ids": meta.get("obj_ids"),
                    "dataset": meta.get("dataset"),
                    "epoch": epoch,
                    "val_loss": vl,
                    "val_miou": miou,
                    "val_miou_all": ev["miou_all"],
                    "val_classes_present": [class_names[c] for c in range(num_classes) if present[c]],
                    "image_size": args.image_size,
                    "hw": list(hw),
                    "n_train_images": len(train_ds),
                },
                out_ckpt,
            )
            log.info(f"    saved best checkpoint (mIoU {miou:.4f}) -> {out_ckpt}")

    plot = Path(args.out_loss_plot) if args.out_loss_plot else out_ckpt.with_name(out_ckpt.stem + "_curves.png")
    save_curves(train_losses, val_losses, mious, plot)
    log.info(f"done. best epoch {best_epoch} val_mIoU={best_miou:.4f}; curves -> {plot}")


if __name__ == "__main__":
    main()
