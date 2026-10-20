"""Default locations of the data a command reads, so the commands of the README work from a fresh checkout.

Everything a run produces and everything that is downloaded lives under one workspace directory (ignored by git):

    $OBJEVAL_WORKSPACE            default ./workspace
    $OBJEVAL_BOP_ROOT/<dataset>   default <workspace>/data/bop/<dataset>   (lmo, ycbv: the BOP archives, unzipped)
    <workspace>/data/burde        the benchmark meshes and cameras of Burde et al.
"""

from __future__ import annotations

import os
from pathlib import Path


def workspace() -> Path:
    return Path(os.environ.get("OBJEVAL_WORKSPACE", "workspace"))


def bop_root(dataset: str) -> Path:
    env = os.environ.get("OBJEVAL_BOP_ROOT")
    return Path(env) / dataset if env else workspace() / "data" / "bop" / dataset


def burde_root() -> Path:
    return workspace() / "data" / "burde"
