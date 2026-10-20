"""Table 4: the size and the time of the segmentation methods, from the measurement files in `results/cost/`.

    size   the elements of the floating-point tensors in the weight files (`objeval3d.bench.params files`), the same definition for every
           method: the head is one network for all objects of a capture, ZebraPose is one network per object (8 on LM-O, 13 on YCB-V,
           counted from the result files), CNOS is the two models it is made of, SAM 3 is its whole checkpoint.
    time   milliseconds per frame from the image to the cloud and the masks: the reconstruction pass plus what the method adds to it.
           For the head that is its marginal cost on a pass the protocol pays anyway, for ZebraPose the network given a box (and with a
           detector, the detector for the box), for CNOS and SAM 3 a second model over the frame.

Every cell is computed from a file, none is typed in. The ratios are to the head, in size and in the time of the whole pipeline.
"""

from __future__ import annotations

import re
import statistics as st
from pathlib import Path

from objeval3d.analysis.common import load_sets, read_json
from objeval3d.analysis.table import Table

HEADER = [
    "Method",
    "Additional input",
    "LM-O (8 objects)",
    "YCB-V (13 objects)",
    "LM-O",
    "YCB-V",
    "Inference time (ms/frame) ↓",
    "Latency ratio (vs. Ours) ↓",
]


def n_objects(results: Path, name: str) -> int:
    return len({re.search(r"_o(\d+)_", k).group(1) for k in load_sets(results, name)})


def models_cost(results: Path) -> Table:
    c = results / "cost"
    pc = {Path(r["file"]).name: r for r in read_json(c / "parameter_counts.json")}
    rows = {x["method"]: x for x in read_json(c / "ycbv_cost.json")["rows"]}
    da3 = rows["DA3 forward pass (no head)"]["ms_per_frame"]
    with_head = rows["DA3 forward pass + segmentation head"]["ms_per_frame"]
    marginal = rows["segmentation head (marginal)"]["ms_per_frame"]
    sam3_ms = rows["SAM 3"]["ms_per_frame"]
    cnos_ms = rows["CNOS (sam + dinov2_vitl14)"]["ms_per_frame"]
    zms = {
        ds: st.median(
            read_json(p)["cost"]["forward_ms_per_frame"]["median"]
            for p in sorted((c / "zebrapose").glob(f"{ds}_rep*.json"))
        )
        for ds in ("lmo", "ycbv")
    }
    dms = {ds: read_json(c / f"{ds}_detector_cost.json")["median_ms_per_frame"] for ds in ("lmo", "ycbv")}
    z, d = st.mean(zms.values()), st.mean(dms.values())

    head_m = pc["lmo_seg_head.pt"]["total"] / 1e6
    zp_m = pc["ape"]["total"] / 1e6
    sam3_m = pc["sam3.pt"]["total"] / 1e6
    det_m = pc["maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"]["total"] / 1e6
    sam_dino_m = (pc["sam_vit_h_4b8939.pth"]["total"] + pc["dinov2_vitl14_pretrain.pth"]["total"]) / 1e6
    n_obj = {"lmo": n_objects(results, "lmo_gt_full"), "ycbv": n_objects(results, "ycbv_gt")}
    zp_all = {ds: n * zp_m for ds, n in n_obj.items()}  # one network per object of the set

    def row(name, extra, sizes, t):
        return [
            name,
            extra,
            f"{sizes[0]:.1f}",
            f"{sizes[1]:.1f}",
            f"{sizes[0] / head_m:.1f}",
            f"{sizes[1] / head_m:.1f}",
            f"{t:.1f}",
            f"{t / with_head:.1f}",
        ]

    table = [
        row("Our method", "—", (head_m, head_m), with_head),
        row("ZebraPose (given detection)", "Object bounding box", (zp_all["lmo"], zp_all["ycbv"]), da3 + z),
        row("ZebraPose + detector", "—", (zp_all["lmo"] + det_m, zp_all["ycbv"] + det_m), da3 + z + d),
        row("CNOS", "Object CAD model", (sam_dino_m,) * 2, da3 + cnos_ms),
        row("SAM 3", "Text prompt", (sam3_m,) * 2, da3 + sam3_ms),
    ]
    return Table(
        "table4",
        "Model size and inference time of segmentation methods on LM-O and YCB-V.",
        HEADER,
        table,
        [
            f"Time: the reconstruction pass ({da3:.1f} ms per frame) plus what the method adds; the head adds {marginal:.1f} ms. "
            f"ZebraPose is {zp_m:.1f} M parameters per object ({n_obj['lmo']} objects on LM-O, {n_obj['ycbv']} on YCB-V).",
            "Sizes are the elements of the floating-point tensors in the weight files.",
        ],
    )
