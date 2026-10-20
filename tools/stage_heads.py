#!/usr/bin/env python3
"""Stage the trained segmentation heads for upload to Hugging Face, after checking every file against configs/heads.json.

    python tools/stage_heads.py --ckpt-dir /path/to/the/checkpoints            # -> workspace/hf_heads
    hf auth login
    hf repos create abramov-de/objeval3d-heads --type model
    hf upload abramov-de/objeval3d-heads workspace/hf_heads . --type model --commit-message "Segmentation heads of the paper"

The checkpoints are the files the paper's experiments used, under the names they were trained with (`trained_as` in heads.json). Each
is checked by size and SHA-256 before anything is staged, so a wrong or damaged file cannot be uploaded by mistake. The staged folder holds
the heads under the names of the release (`file` keys of heads.json, the less important ones in `variants/`), a `SHA256SUMS`, and the
model card as `README.md`. Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(src: Path, dst: Path, copy: bool) -> None:
    """A hard link when the two are on one file system (no second copy of 130 MB), else a copy."""
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copyfile(src, dst)


def stage(ckpt_dir: Path, out: Path, manifest: Path, card: Path, only: str = "all", copy: bool = False) -> list[str]:
    heads = json.loads(manifest.read_text(encoding="utf-8"))["heads"]
    wanted = {k: v for k, v in heads.items() if only == "all" or v["role"] in ("main", "unfamiliar scenes")}
    problems = []
    for name, h in wanted.items():
        src = ckpt_dir / h["trained_as"]
        if not src.is_file():
            problems.append(f"{h['trained_as']}: not in {ckpt_dir}")
        elif src.stat().st_size != h["bytes"]:
            problems.append(f"{h['trained_as']}: {src.stat().st_size} bytes, expected {h['bytes']}")
        elif sha256(src) != h["sha256"]:
            problems.append(f"{h['trained_as']}: SHA-256 differs from configs/heads.json")
    if problems:
        return problems
    for name, h in wanted.items():
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        place((ckpt_dir / h["trained_as"]).resolve(), dst, copy)
    (out / "SHA256SUMS").write_text(
        "".join(f"{h['sha256']}  {n}\n" for n, h in sorted(wanted.items())), encoding="utf-8"
    )
    shutil.copyfile(card, out / "README.md")
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="folder with the checkpoints, under the names they were trained with",
    )
    ap.add_argument("--out", type=Path, default=Path("workspace/hf_heads"))
    ap.add_argument(
        "--only",
        choices=["all", "main"],
        default="all",
        help="main: the three heads of the experiments, without the variants",
    )
    ap.add_argument(
        "--copy", action="store_true", help="copy the files instead of linking them (the folder can then be moved)"
    )
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "heads.json")
    ap.add_argument("--card", type=Path, default=ROOT / "docs" / "MODEL_CARD.md")
    args = ap.parse_args(argv)
    problems = stage(args.ckpt_dir, args.out, args.manifest, args.card, args.only, args.copy)
    if problems:
        print("Nothing staged:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    n = sum(1 for _ in args.out.rglob("*.pt"))
    print(
        f"{n} heads staged in {args.out}; next:\n  hf auth login\n  hf repos create abramov-de/objeval3d-heads --type model\n"
        f"  hf upload abramov-de/objeval3d-heads {args.out} . --type model --commit-message 'Segmentation heads of the paper'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
