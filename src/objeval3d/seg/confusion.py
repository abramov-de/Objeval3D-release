"""Where do a class's pixels go when the head gets them wrong? Per-class mask quality of a trained head over a split of its dataset.

IoU counts every miss the same way. This splits the pixels a class loses into those that go to BACKGROUND (the head does not see
the object) and those that go to ANOTHER OBJECT class (it sees something and picks the wrong label, e.g. between near-duplicates
such as large_clamp / extra_large_clamp or four cylindrical cans). It also gives precision, recall and IoU per class and their mean
over the foreground classes, over ALL frames of the split rather than the high-visibility subset the view sets are drawn from, so
these numbers read lower than the ones of the 3D tables.

    python -m objeval3d.seg.confusion --data-dir workspace/seg_data/ycbv_b20 --head workspace/ckpt/ycbv_b20_seg_head.pt \\
        --split test --out workspace/results/ycbv_b20_seg_confusion.json

Uses the trainer's dataset and resize, so the numbers line up with the training log rather than measuring a slightly different thing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from objeval3d.data.bop import target_hw
from objeval3d.models import DA3_REPO, DA3_REVISION
from objeval3d.seg.dataset import IMAGE_EXTS, BopSegDataset
from objeval3d.seg.head import forward_logits, load_backbone, load_head
from objeval3d.seg.prepare_data import IGNORE_INDEX


@torch.no_grad()
def confusion(backbone, head, loader, device, num_classes: int) -> np.ndarray:
    """cm[t, p] - pixels labelled t that the head called p. Ignored pixels dropped."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        pred = forward_logits(backbone, head, images).argmax(dim=1)
        valid = masks != IGNORE_INDEX
        t, p = masks[valid].flatten(), pred[valid].flatten()
        idx = t * num_classes + p
        cm += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(loader)} batches", flush=True)
    return cm


def main() -> None:
    ap = argparse.ArgumentParser(description="Confusion matrix of the trained seg head")
    ap.add_argument("--data-dir", required=True, help="output of objeval3d.seg.prepare_data")
    ap.add_argument("--head", required=True, help="checkpoint of the head")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--base-model", default=DA3_REPO)
    ap.add_argument("--revision", default=DA3_REVISION)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "classes.json") as f:
        class_names = json.load(f)["class_names"]
    num_classes = len(class_names)

    split_dir = data_dir / args.split
    probe = next(p for p in sorted((split_dir / "images").iterdir()) if p.suffix.lower() in IMAGE_EXTS)
    hw = target_hw(probe, args.image_size)
    ds = BopSegDataset(split_dir, hw)
    print(f"{args.split}: {len(ds)} frames at {hw[0]}x{hw[1]}, {num_classes} classes")

    device = torch.device(args.device)
    backbone = load_backbone(args.device, args.base_model, args.revision)
    head, ckpt = load_head(args.head, args.device)
    ckpt_classes = int(ckpt.get("num_classes", 0))
    if ckpt_classes != num_classes:
        raise SystemExit(
            f"checkpoint has {ckpt_classes} classes, dataset has "
            f"{num_classes} - they were built from different --obj-ids"
        )
    print(
        f"checkpoint: epoch {ckpt.get('epoch')} val_mIoU {ckpt.get('val_miou')} "
        f"(over {len(ckpt.get('val_classes_present') or []) or '?'} classes present in val)"
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cm = confusion(backbone, head, loader, device, num_classes)

    gt = cm.sum(axis=1)  # pixels truly of each class
    pr = cm.sum(axis=0)  # pixels called each class
    hit = np.diag(cm)
    recall = np.divide(hit, gt, out=np.zeros(num_classes), where=gt > 0)
    prec = np.divide(hit, pr, out=np.zeros(num_classes), where=pr > 0)
    iou = np.divide(hit, gt + pr - hit, out=np.zeros(num_classes), where=(gt + pr - hit) > 0)

    # The question the module exists for: of the pixels a class LOSES, what share goes to background and what share to some
    # other object?
    lost = gt - hit
    to_bg = cm[:, 0].copy()
    to_bg[0] = 0
    to_obj = lost - to_bg

    print(
        f"\n{'class':22s} {'gt px':>10s} {'recall':>6s} {'prec':>6s} {'IoU':>5s} "
        f"| {'lost':>10s} {'->bg':>6s} {'->obj':>6s} | worst confusion"
    )
    rows = []
    for c in range(1, num_classes):
        if gt[c] == 0:
            print(f"{class_names[c]:22s} {'-':>10s}   absent from this split")
            continue
        others = [(cm[c, j], j) for j in range(1, num_classes) if j != c]
        n, j = max(others) if others else (0, 0)
        worst = f"{class_names[j]} {100 * n / gt[c]:.1f}%" if n else "-"
        print(
            f"{class_names[c]:22s} {gt[c]:10d} {recall[c]:6.3f} {prec[c]:6.3f} "
            f"{iou[c]:5.3f} | {lost[c]:10d} {100 * to_bg[c] / max(lost[c], 1):5.1f}% "
            f"{100 * to_obj[c] / max(lost[c], 1):5.1f}% | {worst}"
        )
        rows.append(
            {
                "class": class_names[c],
                "gt_px": int(gt[c]),
                "recall": float(recall[c]),
                "precision": float(prec[c]),
                "iou": float(iou[c]),
                "lost_px": int(lost[c]),
                "lost_to_background_frac": float(to_bg[c] / max(lost[c], 1)),
                "lost_to_other_object_frac": float(to_obj[c] / max(lost[c], 1)),
                "worst_confusion": {"class": class_names[j], "px": int(n), "frac_of_gt": float(n / gt[c])}
                if n
                else None,
            }
        )

    fg = [r for r in rows if r["gt_px"] > 0]
    tot_lost = sum(r["lost_px"] for r in fg)
    bg_share = sum(r["lost_px"] * r["lost_to_background_frac"] for r in fg) / max(tot_lost, 1)
    print(
        f"\nmean over the foreground classes: IoU {np.mean([r['iou'] for r in fg]):.3f}, "
        f"precision {np.mean([r['precision'] for r in fg]):.3f}, recall {np.mean([r['recall'] for r in fg]):.3f}"
    )
    print(f"{100 * bg_share:.1f}% of the lost pixels go to background, {100 * (1 - bg_share):.1f}% to another object")
    print(
        "a background-dominated split points at detection (train longer/balanced);\n"
        "an object-dominated one points at class confusion"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "class_names": class_names,
                    "confusion": cm.tolist(),
                    "per_class": rows,
                    "lost_to_background_frac_overall": float(bg_share),
                },
                f,
                indent=2,
            )
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
