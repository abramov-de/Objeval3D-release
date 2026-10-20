"""Table 1 (properties of the two datasets) and Table F.1 (the prompts given to the language-driven segmenter)."""

from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

import numpy as np

from objeval3d.analysis.common import load_sets
from objeval3d.analysis.table import Table
from objeval3d.symmetry import models_info

FRAME_PIXELS = 640 * 480  # BOP frame of LM-O and YCB-V
YCBV_TEST_SCENES = 12  # scenes 48-59 of the YCB-V test set
FILES = {"LM-O": ("lmo_gt_full", "lmo_v3", "lmo"), "YCB-V": ("ycbv_gt", "ycbv_v3", "ycbv")}


def manifests_dir(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("OBJEVAL_MANIFESTS") or "configs/manifests")


def _distance_over_diameter(manifest: Path, dataset: str) -> tuple[float, dict[int, float]]:
    """Median over the views of camera-to-object distance divided by the object's diameter, and the median per scene."""
    info = models_info(dataset)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    by_scene: dict[int, list[float]] = {}
    for s in m["sets"]:
        for t in s["t_m2c"]:
            by_scene.setdefault(s["scene_id"], []).append(float(np.linalg.norm(t)) / info[s["obj_id"]]["diameter"])
    allv = [v for vs in by_scene.values() for v in vs]
    return float(np.median(allv)), {sc: float(np.median(vs)) for sc, vs in sorted(by_scene.items())}


def dataset_properties(results: Path, manifests: Path | None = None) -> Table:
    """Table 1: objects, view sets, scenes, how much of the frame the object fills, the diameters and the working distance."""
    col = {}
    for ds, (name, manifest, short) in FILES.items():
        recs = list(load_sets(results, name).values())
        px = [v["gt_px"] for r in recs for v in r["mask_metrics"]["per_view"]]
        diam = {r["obj_name"]: r["diameter_mm"] for r in recs}
        ratio, per_scene = _distance_over_diameter(manifests_dir(manifests) / f"{manifest}.json", short)
        col[ds] = {
            "objects": len(diam),
            "sets": len(recs),
            "scenes": sorted({r["scene_id"] for r in recs}),
            "area": 100 * st.mean(px) / FRAME_PIXELS,
            "dmin": min(diam.values()),
            "dmax": max(diam.values()),
            "ratio": ratio,
            "per_scene": per_scene,
        }
    lo, yc = col["LM-O"], col["YCB-V"]
    rows = [
        ["Number of objects", str(lo["objects"]), str(yc["objects"])],
        ["Number of view sets", str(lo["sets"]), str(yc["sets"])],
        [
            "Scenes in the test set",
            f"{len(lo['scenes'])} (Scene {lo['scenes'][0]}; entire test set)",
            f"{len(yc['scenes'])} of {YCBV_TEST_SCENES} (Scenes {', '.join(map(str, yc['scenes']))})",
        ],
        ["Mean object area (% of frame)", f"{lo['area']:.2f} %", f"{yc['area']:.2f} %"],
        ["Object diameters", f"{lo['dmin']:.0f}–{lo['dmax']:.0f} mm", f"{yc['dmin']:.0f}–{yc['dmax']:.0f} mm"],
        [
            "Distance to object / object diameter, median (per-scene medians)",
            f"{lo['ratio']:.1f}",
            f"{yc['ratio']:.1f} ({min(yc['per_scene'].values()):.1f}–{max(yc['per_scene'].values()):.1f})",
        ],
        [
            "Object types and characteristics",
            "painted figurines without texture, strong mutual occlusion",
            "household items with stable names",
        ],
    ]
    return Table(
        "table1",
        "Properties and differences of the two datasets.",
        ["Parameter", "LM-O", "YCB-V"],
        rows,
        [
            "The area is the mean over all views of the ground-truth mask in pixels over the 640 x 480 frame; the distance is from the "
            "camera to the origin of the object model."
        ],
    )


def sam3_prompts(results: Path, config: Path | None = None) -> Table:
    """Table F.1: the text prompts given to SAM 3, LM-O and YCB-V."""
    from objeval3d.seg.prompts import prompt_for

    def prompt(name):
        return prompt_for(name, config)

    lmo = sorted({r["obj_name"] for r in load_sets(results, "lmo_gt_full").values()})
    ycbv = []
    for r in sorted(load_sets(results, "ycbv_b20_gt").values(), key=lambda r: r["obj_id"]):
        if r["obj_name"] not in ycbv:
            ycbv.append(r["obj_name"])

    def q(s):
        return f'"{s.replace("_", " ")}"'

    rows = []
    for i in range(max(len(lmo), len(ycbv))):
        left = [q(lmo[i]), f'"{prompt(lmo[i])}"'] if i < len(lmo) else ["", ""]
        right = [q(ycbv[i]), f'"{prompt(ycbv[i])}"'] if i < len(ycbv) else ["", ""]
        rows.append(left + right)
    return Table("tableF1", "Prompts for SAM 3.", ["LM-O", "Prompt", "YCB-V", "Prompt"], rows)
