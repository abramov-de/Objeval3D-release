"""Tables 7 and G.1: does a free fit also understate the error of meshes that are not ours?

The meshes come from the benchmark of Burde et al. (WACV 2025): complete meshes of the 21 YCB-V objects, built by other methods
(BakedSDF, MonoSDF, Nerfacto, VolSDF in the check) from all the images or from three, already in the object frame of the BOP models,
in millimetres. A mesh has no depth model, no mask and no camera poses of its own to be wrong about: only a mesh, a scan and the
true placement. The check asks whether the free fit reaches a lower distance than the true placement on them too.

Statistic: medians over meshes, shares over the number of meshes. A free fit *from an arbitrary start* is the global search of
`objeval3d.align.registration`, *started at the true placement* is the same trimmed ICP started there (`local`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from objeval3d.analysis.common import read_jsonl
from objeval3d.analysis.table import Table

SUBSETS = ("all", "3")  # image counts of the meshes: all images, three images


def load_meshes(results: Path) -> list[dict]:
    recs = {r["key"]: r for r in read_jsonl(results / "burde_main.jsonl")}
    return [r for r in recs.values() if r.get("status") == "ok"]


def _med(x) -> float:
    return float(np.median([v for v in x if v == v]))


def _acc(r, m):
    return r["alignments"][m]["accuracy"]["median"]


def _chamfer(r, m):
    return r["alignments"][m]["chamfer_l1_mm"]


def _within(r, m, tau):
    return r["alignments"][m]["burde_metric"][f"tau_{tau:g}mm"]["acc"]


def column(recs: list[dict], subset: str) -> dict:
    """Everything one image count contributes to the two tables."""
    rs = [r for r in recs if r["subset"] == subset and "auto" in r["alignments"] and "local" in r["alignments"]]
    c = {"n": len(rs), "true": _med(_acc(r, "as_lies") for r in rs)}
    for mode in ("auto", "local"):
        c[mode] = {
            "acc": _med(_acc(r, mode) for r in rs),
            "below": 100 * sum(_chamfer(r, mode) < _chamfer(r, "as_lies") for r in rs) / len(rs),
            "rot": _med(r["vs_as_lies"][mode]["rotation_deg"] for r in rs),
            "shift": 100 * _med(r["vs_as_lies"][mode]["placement_centroid_rel_diam"] for r in rs),
        }
    c["within2_before"] = _med(_within(r, "as_lies", 2.0) for r in rs)
    c["within2_after"] = _med(_within(r, "auto", 2.0) for r in rs)
    c["ratio_of_medians"] = c["true"] / c["auto"]["acc"]
    return c


def columns(results: Path) -> dict[str, dict]:
    recs = load_meshes(results)
    return {s: column(recs, s) for s in SUBSETS}


def meshes_summary(results: Path) -> Table:
    """Table 7: accuracy at the true placement, after the free fit, and how often the free fit reaches a lower Chamfer distance."""
    c = columns(results)

    def cell(f):
        return [f(c["all"]), f(c["3"])]

    return Table(
        "table7",
        "Effect of free-fit alignment on benchmark meshes reconstructed from all images and from three images.",
        [
            "Metric",
            f"Meshes reconstructed from all images (n = {c['all']['n']})",
            f"Meshes reconstructed from 3 images (n = {c['3']['n']})",
        ],
        [
            ["Accuracy at ground-truth pose (mm) ↓"] + cell(lambda x: f"{x['true']:.2f} mm"),
            ["Accuracy after free-fit alignment (mm) ↓"] + cell(lambda x: f"{x['auto']['acc']:.2f} mm"),
            ["Meshes with lower Chamfer distance after free fit (%)"] + cell(lambda x: f"{x['auto']['below']:.0f} %"),
        ],
        ["Medians over meshes; the true placement is the pose of the benchmark, with the depth scale left free."],
    )


def meshes_full(results: Path) -> Table:
    """Table G.1: the full check on the benchmark meshes."""
    c = columns(results)

    def turn(x):
        return f"{x:.0f}°" if x >= 100 else f"{x:.1f}°"

    rows = [
        ["Accuracy at the ground-truth pose (mm) ↓", lambda x: f"{x['true']:.2f}"],
        ["Accuracy after free-fit alignment (mm) ↓", lambda x: f"{x['auto']['acc']:.2f}"],
        ["Accuracy after free fit initialized at the ground-truth pose (mm) ↓", lambda x: f"{x['local']['acc']:.2f}"],
        ["Meshes with lower Chamfer distance after free-fit alignment (%)", lambda x: f"{x['auto']['below']:.0f}"],
        ["Same, with ground-truth pose initialization (%)", lambda x: f"{x['local']['below']:.0f}"],
        [
            "Pose displacement after free fit: rotation (°), translation (% of diameter)",
            lambda x: f"{turn(x['auto']['rot'])}, {x['auto']['shift']:.1f}%",
        ],
        [
            "Same, with ground-truth pose initialization",
            lambda x: f"{turn(x['local']['rot'])}, {x['local']['shift']:.1f}%",
        ],
        [
            "Mesh points within 2 mm of the scan, before → after free fit",
            lambda x: f"{x['within2_before']:.2f} → {x['within2_after']:.2f}",
        ],
    ]
    return Table(
        "tableG1",
        "Full evaluation of benchmark meshes under free-fit alignment.",
        ["Quantity", f"All images ({c['all']['n']} meshes)", f"Three images ({c['3']['n']} meshes)"],
        [[label, f(c["all"]), f(c["3"])] for label, f in rows],
    )
