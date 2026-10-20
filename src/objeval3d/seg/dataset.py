"""The segmentation dataset written by `objeval3d.seg.prepare_data`, as a torch dataset, and a check of its labels."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from objeval3d.seg.prepare_data import IGNORE_INDEX

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# OpenCV's own threading fights the DataLoader workers for cores and makes loading slower, not faster, once num_workers > 1.
cv2.setNumThreads(0)


class BopSegDataset(Dataset):
    def __init__(self, root: Path, hw: tuple[int, int], augment: bool = False):
        self.hw = hw
        self.augment = augment
        images_dir, masks_dir = root / "images", root / "masks"
        self.samples: list[tuple[Path, Path]] = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            m = masks_dir / f"{p.stem}.png"
            if m.exists():
                self.samples.append((p, m))
        if not self.samples:
            raise ValueError(f"No image/mask pairs in {root}")
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        h, w = self.hw

        img = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Photometric only. No flips/crops: the head sits on a frozen geometry backbone whose features are viewpoint-coupled,
            # and a mirrored image is not a pose the depth branch ever sees.
            if random.random() < 0.5:
                a = random.uniform(0.8, 1.2)
                b = random.uniform(-20, 20)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return self.norm(img_t), torch.from_numpy(mask.astype(np.int64))


def check_mask_labels(data_dir: Path, num_classes: int) -> int:
    """Verify every mask only contains labels this head can represent; returns the number of files checked.

    Out-of-range labels are otherwise reported by CUDA as an assertion inside cross-entropy's kernel, asynchronously, with a
    traceback pointing at whichever line happened to synchronise - nothing that leads back to the mask that caused it. The usual
    source is a dataset folder rebuilt with a different --obj-ids, since class indices are positions in that list.
    """
    bad = []
    n_seen = 0
    for split in ("train", "val"):
        d = data_dir / split / "masks"
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() != ".png":
                continue
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            n_seen += 1
            vals = np.unique(m)
            out_of_range = vals[(vals >= num_classes) & (vals != IGNORE_INDEX)]
            if len(out_of_range):
                bad.append((f, out_of_range.tolist()))
                if len(bad) >= 5:
                    break
        if len(bad) >= 5:
            break
    if bad:
        lines = "\n".join(f"    {f} contains labels {v}" for f, v in bad)
        raise SystemExit(
            f"\n{len(bad)}+ masks carry labels outside [0,{num_classes - 1}] "
            f"(ignore={IGNORE_INDEX}):\n{lines}\n\n"
            f"This dataset folder was almost certainly built with a different "
            f"--obj-ids. Rebuild it with objeval3d.seg.prepare_data; it wipes the split "
            f"folders before writing.\n"
        )
    return n_seen
