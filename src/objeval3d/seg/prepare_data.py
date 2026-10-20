"""Build the training/evaluation set of the segmentation head out of a BOP dataset.

BOP ships per-annotation *visible* masks (`mask_visib/`), so the labels the head needs come for free - no manual annotation. This
module turns them into an image folder plus a folder of class-index PNGs (pixel value == class index, 0 == background, 255 ==
ignored).

Class indices are assigned by the order of `--obj-ids`, so `--obj-ids 8,5,1` gives driller=1, can=2, ape=3. The mapping is written
to `classes.json` next to the splits and is read back by every downstream command.

Splits:

  --split-by block   (default) contiguous chunks of one scene's trajectory are assigned to different splits (train, train, train,
                     val, test, cyclically). Needed for single-scene datasets such as LM-O: adjacent video frames are near
                     duplicates, so a per-frame split would put frame i in train and frame i+1 in test and report a number that
                     means nothing. This is the split the paper's heads are trained on, for LM-O and for YCB-V.
  --split-by scene   whole scenes go to train / val / test. The strictest protocol, where the head never sees the evaluation
                     arrangement; it is the cross-scene transfer test of the paper's discussion.

The paper's heads (see docs/REPRODUCING.md):

    # LM-O: one scene, eight objects
    python -m objeval3d.seg.prepare_data --bop-root BOP/lmo --dataset lmo --split test --scenes 2 \\
        --obj-ids 1,5,6,8,9,10,11,12 --split-by block --n-blocks 20 --min-visib-fract 0.3 --min-px 400 \\
        --out-dir workspace/seg_data/lmo

    # YCB-V, within one capture: the same block split inside the four test scenes
    python -m objeval3d.seg.prepare_data --bop-root BOP/ycbv --dataset ycbv --split test --scenes 48,50,53,58 \\
        --obj-ids 1,2,4,5,6,9,10,11,13,14,15,19,20 --split-by block --n-blocks 20 --frame-stride 6 \\
        --min-visib-fract 0.3 --min-px 800 --out-dir workspace/seg_data/ycbv_b20
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from objeval3d.data.bop import BopScene, object_name

IGNORE_INDEX = 255


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def build_class_mask(
    scene: BopScene,
    im_id: int,
    obj_to_class: dict[int, int],
    min_visib_fract: float,
    min_px: int,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[int, int]]:
    """Class-index mask for one image + per-object visible pixel counts.

    Annotations are painted in order of *decreasing* distance to the camera so
    that a nearer object overwrites a farther one wherever the visible masks
    happen to overlap (they should not, but BOP masks are rendered per object
    and a 1px seam is common).

    A target object that is present but too occluded or too small to be a
    reliable label is painted `IGNORE_INDEX` rather than left as background.
    Leaving it as background would actively teach the head that this object is
    *not* the object - the opposite of what we want - and in a cluttered BOP
    scene that happens on a large share of frames. The training loss skips
    those pixels entirely.
    """
    h, w = image_hw
    out = np.zeros((h, w), dtype=np.uint8)
    px_counts: dict[int, int] = {}

    anns = [a for a in scene.annotations(im_id) if a["obj_id"] in obj_to_class]
    anns.sort(key=lambda a: -float(np.linalg.norm(a["t_m2c"])))  # far -> near

    for ann in anns:
        mpath = scene.mask_visib_path(im_id, ann["gt_id"])
        if not mpath.exists():
            continue
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = m > 127
        n_px = int(m.sum())
        if n_px == 0:
            continue
        if ann["visib_fract"] < min_visib_fract or n_px < min_px:
            out[m] = IGNORE_INDEX
            continue
        out[m] = obj_to_class[ann["obj_id"]]
        px_counts[ann["obj_id"]] = n_px
    return out, px_counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BOP -> class-index segmentation dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--bop-root", required=True, help="root of one BOP dataset, e.g. .../bop/lmo")
    ap.add_argument("--dataset", required=True, help="lmo | ycbv (naming only)")
    ap.add_argument("--split", default="test", help="BOP split folder: test / train_real / val")
    ap.add_argument("--scenes", default="", help="comma separated scene ids; empty = all")
    ap.add_argument("--obj-ids", required=True, help="comma separated BOP object ids to label")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument(
        "--split-by",
        choices=["block", "scene"],
        default="block",
        help="block: contiguous chunks of the trajectory go to different splits (default - neighbouring video "
        "frames are near duplicates, so a frame-wise split leaks the test set into training); "
        "scene: whole scenes",
    )
    ap.add_argument(
        "--n-blocks",
        type=int,
        default=20,
        help="--split-by block: number of contiguous chunks per scene; chunks are assigned "
        "train/train/train/val/test cyclically. Keep this well above 5 so each split lands on several "
        "chunks spread along the trajectory - otherwise the test frames form one short arc and no "
        "wide-baseline view set can be built from them.",
    )
    ap.add_argument("--val-scenes", default="", help="--split-by scene: scene ids for val")
    ap.add_argument("--test-scenes", default="", help="--split-by scene: scene ids for test")

    ap.add_argument("--frame-stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-frames-per-scene", type=int, default=0, help="0 = no cap")
    ap.add_argument(
        "--min-visib-fract",
        type=float,
        default=0.3,
        help="annotations more occluded than this are ignored, not labelled",
    )
    ap.add_argument(
        "--min-px",
        type=int,
        default=400,
        help="annotations whose visible mask is smaller than this are ignored, not labelled",
    )
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="keep whatever is already in --out-dir (default is to wipe the split folders, since class indices "
        "depend on --obj-ids)",
    )
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="copy RGB files instead of symlinking (use if the training machine cannot follow symlinks into the "
        "BOP tree)",
    )
    args = ap.parse_args()

    bop_root = Path(args.bop_root)
    split_dir = bop_root / args.split
    if not split_dir.exists():
        raise SystemExit(f"Split folder not found: {split_dir}")

    obj_ids = parse_int_list(args.obj_ids)
    obj_to_class = {oid: i + 1 for i, oid in enumerate(obj_ids)}

    scene_ids = (
        parse_int_list(args.scenes)
        if args.scenes
        else sorted(int(p.name) for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())
    )

    val_scenes = set(parse_int_list(args.val_scenes))
    test_scenes = set(parse_int_list(args.test_scenes))

    out_dir = Path(args.out_dir)
    for sp in ("train", "val", "test"):
        for sub in ("images", "masks"):
            d = out_dir / sp / sub
            # Wipe rather than overwrite. Class indices are positions in --obj-ids, so a run with a different object list
            # writes a different numbering, and a run with a different scene split moves frames between splits. Leftovers
            # from a previous run then sit alongside the new ones with labels that no longer mean anything - which surfaces
            # much later as a CUDA assert inside cross-entropy, with nothing pointing back to here.
            if d.exists() and not args.no_clean:
                n = 0
                for f in d.iterdir():
                    if f.is_file() or f.is_symlink():
                        f.unlink()
                        n += 1
                if n:
                    print(f"[clean] removed {n} stale files from {d}")
            d.mkdir(parents=True, exist_ok=True)

    classes = ["background"] + [object_name(args.dataset, oid) for oid in obj_ids]
    with open(out_dir / "classes.json", "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "class_names": classes,
                "obj_ids": obj_ids,
                "ignore_index": IGNORE_INDEX,
                "obj_to_class": {str(k): v for k, v in obj_to_class.items()},
            },
            f,
            indent=2,
        )

    index: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    class_px: dict[str, np.ndarray] = {sp: np.zeros(len(classes), dtype=np.int64) for sp in index}
    ignore_px: dict[str, int] = {sp: 0 for sp in index}

    for sid in scene_ids:
        scene_dir = split_dir / f"{sid:06d}"
        if not scene_dir.exists():
            print(f"[skip] scene {sid} not found at {scene_dir}")
            continue
        scene = BopScene(scene_dir)
        if not scene.has_visib_masks():
            raise SystemExit(
                f"Scene {scene_dir} has no mask_visib/ folder.\n"
                f"The BOP archive you unzipped does not ship visible masks for this "
                f"split - download the *_test_all.zip variant, or render the masks "
                f"with bop_toolkit/scripts/calc_gt_masks.py."
            )

        im_ids = scene.im_ids[:: max(1, args.frame_stride)]
        if args.max_frames_per_scene > 0:
            im_ids = im_ids[: args.max_frames_per_scene]

        kept = 0
        block_len = max(1, -(-len(im_ids) // max(1, args.n_blocks)))
        for pos, im_id in enumerate(im_ids):
            rgb_path = scene.rgb_path(im_id)
            img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            mask, px_counts = build_class_mask(scene, im_id, obj_to_class, args.min_visib_fract, args.min_px, (h, w))
            present = set(px_counts.keys())
            if not present:
                continue

            # ---- split assignment ------------------------------------------
            # Assignment is by *position in the scanned sequence*, not by a counter over kept frames, so dropping a frame
            # never shifts the split of the frames after it.
            if args.split_by == "scene":
                sp = "test" if sid in test_scenes else "val" if sid in val_scenes else "train"
            else:
                b = (pos // block_len) % 5
                sp = "test" if b == 4 else "val" if b == 3 else "train"

            stem = f"{args.dataset}_s{sid:06d}_i{im_id:06d}"
            dst_img = out_dir / sp / "images" / f"{stem}{rgb_path.suffix}"
            if args.copy_images:
                shutil.copyfile(rgb_path, dst_img)
            else:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                dst_img.symlink_to(rgb_path.resolve())
            cv2.imwrite(str(out_dir / sp / "masks" / f"{stem}.png"), mask)

            counts = np.bincount(mask.reshape(-1), minlength=256)
            class_px[sp] += counts[: len(classes)]
            ignore_px[sp] += int(counts[IGNORE_INDEX])
            index[sp].append(
                {
                    "stem": stem,
                    "scene_id": sid,
                    "im_id": im_id,
                    "objects": sorted(present),
                    "px": {str(k): v for k, v in px_counts.items()},
                }
            )
            kept += 1
        print(f"scene {sid:06d}: {kept} frames kept out of {len(im_ids)} scanned")

    with open(out_dir / "index.json", "w") as f:
        json.dump({"splits": index, "class_names": classes, "obj_ids": obj_ids}, f, indent=2)

    print("\n=== summary ===")
    for sp in ("train", "val", "test"):
        n = len(index[sp])
        print(f"{sp:5s}: {n:5d} frames")
        if n:
            tot = class_px[sp].sum() + ignore_px[sp]
            for ci, cname in enumerate(classes):
                frac = 100.0 * class_px[sp][ci] / max(tot, 1)
                print(f"        {ci}:{cname:<22s} {frac:6.2f}% of pixels")
            print(f"        {IGNORE_INDEX}:{'(ignored)':<22s} {100.0 * ignore_px[sp] / max(tot, 1):6.2f}% of pixels")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
