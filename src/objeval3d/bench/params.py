"""Parameters of the models in the cost comparison.

    python -m objeval3d.bench.params files FILE [FILE ...] [--json OUT]

`files` opens each weight file on the CPU and adds up the elements of every tensor in its state. A checkpoint can hold more than the
network (an optimiser state, a copy of the weights), so the count is printed by the first level of the key names as well, and what is
reported is the part that is the network. Running statistics of normalisation layers (buffers) and an optimiser state are left out, so
the figure is what `model.parameters()` would give. The DA3 + head model prints its own count in `objeval3d.bench.cost head`; it equals
the file count.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# files: the elements of the tensors in the weight files
# ---------------------------------------------------------------------------
def tensors(obj, prefix=""):
    """Every (name, tensor) in a nested state, however the checkpoint nests it."""
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{prefix}[{i}]")


def count(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            items = [(k, f.get_tensor(k)) for k in f.keys()]
    else:
        items = list(tensors(torch.load(str(path), map_location="cpu", weights_only=False)))
    by = Counter()
    for name, t in items:
        # running statistics of normalisation layers are buffers, not parameters, and the optimiser state of a checkpoint is not the
        # network
        if t.is_floating_point() and not name.endswith(("running_mean", "running_var")) and "optimizer" not in name:
            by[name.split(".")[0]] += t.numel()
    total = sum(by.values())
    print(f"{path}\n  size {path.stat().st_size / 1e6:,.0f} MB, floating-point elements {total / 1e6:,.1f} M")
    for k, v in by.most_common(8):
        print(f"    {k:28s} {v / 1e6:9.1f} M")
    return {"file": str(path), "bytes": path.stat().st_size, "total": total, "by_first_level": dict(by)}


def cmd_files(args) -> None:
    rows = [count(Path(p)) for p in args.files]
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"-> {args.json}")


# ---------------------------------------------------------------------------
# built: model.parameters() of the built networks
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="parameters of the models in the cost comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("files", help="elements of the tensors in weight files")
    f.add_argument("files", nargs="+")
    f.add_argument("--json", default="")
    f.set_defaults(fn=cmd_files)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
