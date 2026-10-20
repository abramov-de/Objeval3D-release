"""What each segmenter costs per frame, measured rather than asserted.

The claim of the paper is that the head is cheap: it rides on a forward pass the protocol already pays for, while SAM 3 and CNOS are
second models over the same frame. Every row of the cost table is recomputed from a file on disk by `objeval3d.analysis.costs`; this
module writes those files:

    python -m objeval3d.bench.cost head     --manifest M.json --head CKPT --cnos-record RUN/cnos_run.json --sam3-cost sam3_cost.json --out cost.json
    python -m objeval3d.bench.cost sam3     --manifest M.json --checkpoint-path sam3.pt --out sam3_cost.json     (in the SAM 3 environment)
    python -m objeval3d.bench.cost detector --manifest M.json --out detector_cost.json

Three things the measurements are careful about, because a sloppy cost number is worse than none.

**The head's cost is marginal, not total.** DA3 runs whether or not anything is segmented - the protocol needs the point cloud. So the
head's price is the difference between `infer_seg=True` and `infer_seg=False` on the same view sets, which is the DPT head over features
the backbone has already produced (and the transfer of its logits to the host). Reporting the whole forward pass as the head's cost
would flatter SAM 3 and CNOS; reporting zero would not be true either: 32.65 M parameters are not free.

**Per view set, not per frame, is what DA3 actually computes.** It takes the views jointly; the per-frame figure is a division and is
labelled as one. SAM 3 and CNOS are genuinely per-frame.

**The same rules for every row.** The model is built once, outside the clock; frames are decoded before the clock starts; a warm-up
precedes the clock; the GPU is synchronised around each call; and the repeats of the two arms are alternated rather than run in two
blocks, so that a GPU that warms up or gets shared partway through biases both arms equally. Time one model per process, on an idle GPU
(`nvidia-smi`), and on the same GPU for all rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from objeval3d.data.bop import git_revision
from objeval3d.data.manifest import load_manifest
from objeval3d.models import DA3_REPO, DA3_REVISION, PROCESS_RES
from objeval3d.seg.prompts import prompt_for


def median_ms(xs: list[float]) -> float:
    return round(1000 * statistics.median(xs), 1)


def sam3_row(path: Path) -> dict | None:
    """SAM 3's row out of the SAM 3 benchmark (`cost.py sam3`)."""
    rec = json.loads(Path(path).read_text())
    return {
        "method": "SAM 3",
        "ms_per_frame": rec["ms_per_frame"]["median"],
        "ms_set_image_median": rec.get("ms_set_image_median"),
        "ms_set_text_prompt_median": rec.get("ms_set_text_prompt_median"),
        "n_frames": rec["n_frames"],
        "includes_image_loading": False,
        "includes_model_load": False,
        "gpu": rec.get("gpu"),
        "source": str(path),
        "note": "model built once outside the clock; set_image + set_text_prompt + the best instance to the CPU",
    }


def cnos_row(path: Path) -> dict | None:
    rec = json.loads(Path(path).read_text())
    cost = rec.get("cost")
    if not cost:
        return None
    return {
        "method": f"CNOS ({rec.get('segmentor', '?')} + {rec.get('descriptor_model', '?')})",
        "ms_per_frame": cost["total_ms"]["median"],
        "proposal_stage_ms": cost["proposal_stage_ms"]["median"],
        "matching_stage_ms": cost["matching_stage_ms"]["median"],
        "n_frames": cost["n_frames"],
        "includes_image_loading": False,
        "includes_model_load": False,
        "proposals_per_frame": cost.get("proposals_per_frame"),
        "gpu": cost.get("gpu"),
        "source": str(path),
        "note": "second pass over the frame; reference descriptors are precomputed once per object and not counted here",
    }


# ---------------------------------------------------------------------------
# head: DA3 with and without the segmentation head
# ---------------------------------------------------------------------------
def bench_head(args) -> None:
    import cv2
    import torch

    from objeval3d.recon.backends import DA3Predictor

    manifest = load_manifest(args.manifest, args.bop_root)
    sets = manifest["sets"]
    step = max(1, len(sets) // args.n_sets)
    sets = sets[::step][: args.n_sets]
    n_views = len(sets[0]["images"])
    print(f"{len(sets)} view sets x {n_views} views at process-res {args.process_res}")

    model = DA3Predictor(device=args.device, base_model=args.base_model, revision=args.revision, head_ckpt=args.head)
    device = torch.device(args.device)

    def n_params(m) -> int:
        return sum(p.numel() for p in m.parameters())

    head_params = n_params(model.head)
    total_params = n_params(model.model.model) + head_params
    print(
        f"parameters: {total_params / 1e6:.1f} M total, {head_params / 1e6:.1f} M in the segmentation head "
        f"({100 * head_params / total_params:.1f} %)"
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def run(images, infer_seg: bool) -> float:
        sync()
        t0 = time.time()
        model.inference(images, process_res=args.process_res, infer_seg=infer_seg)
        sync()
        return time.time() - t0

    # Decoded up front: the clock should measure the model, and the reconstruction pays this cost once per view set either way.
    loaded = [[cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in vs["images"]] for vs in sets]

    print(f"warm-up ({args.warmup} passes)...")
    for _ in range(args.warmup):
        run(loaded[0], False)
        run(loaded[0], True)

    t_geom, t_seg = [], []
    for i, images in enumerate(loaded, 1):
        # Alternated rather than run in two blocks: a GPU that warms up or gets shared partway through a benchmark then biases both
        # arms equally instead of whichever one ran second.
        g = [run(images, False) for _ in range(args.repeats)]
        s = [run(images, True) for _ in range(args.repeats)]
        t_geom.append(statistics.median(g))
        t_seg.append(statistics.median(s))
        if i % 5 == 0 or i == len(loaded):
            print(
                f"  [{i}/{len(loaded)}] {1000 * t_geom[-1]:.0f} ms without the head, {1000 * t_seg[-1]:.0f} ms with it"
            )

    marginal = [s - g for s, g in zip(t_seg, t_geom)]
    rows = [
        {
            "method": "DA3 forward pass (no head)",
            "ms_per_view_set": median_ms(t_geom),
            "ms_per_frame": round(median_ms(t_geom) / n_views, 1),
            "note": "the protocol pays this for the point cloud regardless of the mask source; it is not a segmentation cost",
        },
        {
            "method": "DA3 forward pass + segmentation head",
            "ms_per_view_set": median_ms(t_seg),
            "ms_per_frame": round(median_ms(t_seg) / n_views, 1),
        },
        {
            "method": "segmentation head (marginal)",
            "ms_per_view_set": median_ms(marginal),
            "ms_per_frame": round(median_ms(marginal) / n_views, 1),
            "params_M": round(head_params / 1e6, 1),
            "includes_image_loading": False,
            "includes_model_load": False,
            "note": "difference of the two rows above, on the same view sets: the DPT head over features the backbone already computed",
        },
    ]
    for path, fn in ((args.sam3_cost, sam3_row), (args.cnos_record, cnos_row)):
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            print(f"[warn] no record at {p}, that row is left out")
            continue
        row = fn(p)
        if row:
            rows.append(row)

    out = {
        "manifest": args.manifest,
        "git_revision": git_revision(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "process_res": args.process_res,
        "n_sets": len(sets),
        "n_views": n_views,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "head": args.head,
        "params": {
            "total_M": round(total_params / 1e6, 1),
            "seg_head_M": round(head_params / 1e6, 1),
            "seg_head_fraction": round(head_params / total_params, 4),
        },
        "rows": rows,
    }

    print(f"\n{'method':44s} {'ms/frame':>9s} {'ms/set':>8s}")
    for r in rows:
        print(
            f"{r['method']:44s} {r.get('ms_per_frame', float('nan')):9.1f} {r.get('ms_per_view_set', float('nan')):8.1f}"
        )
    print(
        "\nms/set is what DA3 actually computes (the views jointly); ms/frame for those rows is that divided by the view count."
    )

    if args.out:
        o = Path(args.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(out, indent=2))
        print(f"-> {o}")


# ---------------------------------------------------------------------------
# sam3: the model built once, outside the clock
# ---------------------------------------------------------------------------
def bench_sam3(args) -> None:
    from pathlib import Path as _P

    import numpy as np
    import sam3.model_builder as mb
    import torch
    from PIL import Image
    from PIL.ImageOps import exif_transpose
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    def sync() -> None:
        torch.cuda.synchronize()

    sets = load_manifest(args.manifest, args.bop_root)["sets"]
    step = max(1, len(sets) // args.n_sets)
    sets = sets[::step][: args.n_sets]
    frames = []  # (image, prompt, object id)
    for vs in sets:
        for p in vs["images"]:
            frames.append(
                (
                    exif_transpose(Image.open(p)).convert("RGB"),
                    prompt_for(vs["obj_name"], args.prompts_file),
                    int(vs["obj_id"]),
                )
            )
    print(f"{len(sets)} view sets, {len(frames)} frames, {len({f[2] for f in frames})} objects")

    bpe = str(_P(mb.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=bpe, device=args.device, checkpoint_path=args.checkpoint_path, load_from_HF=False
    )
    sync()
    build_s = time.time() - t0
    processor = Sam3Processor(model)
    processor.set_confidence_threshold(args.confidence_threshold)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built in {build_s:.1f} s; parameters of the built model: {n_params / 1e6:.1f} M")

    def one(image, prompt):
        """One object in one frame; returns the seconds of set_image, of set_text_prompt, of the mask to the CPU, the count, and the
        seconds of a SECOND prompt on the same image (the image features are reused), the price of each further object in a frame."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sync()
            a = time.time()
            state = processor.set_image(image)
            sync()
            b = time.time()
            out = processor.set_text_prompt(prompt=prompt, state=state)
            sync()
            c = time.time()
        masks_t, scores_t = out["masks"], out["scores"]
        n = int(masks_t.shape[0])
        if n:
            m = masks_t.detach().float().cpu().numpy()
            m = (m[:, 0] if m.ndim == 4 else m) > 0.5
            _ = m[int(np.argmax(scores_t.detach().float().cpu().numpy()))]
        sync()
        d = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            processor.reset_all_prompts(state)
            sync()
            e0 = time.time()
            processor.set_text_prompt(prompt=prompt, state=state)
            sync()
            e1 = time.time()
        return b - a, c - b, d - c, n, e1 - e0

    print(f"warm-up ({args.warmup} calls)...")
    for i in range(args.warmup):
        one(*frames[i % len(frames)][:2])

    enc, dec, cpu, tot, nd, more = [], [], [], [], [], []
    for i, (image, prompt, _) in enumerate(frames, 1):
        r = [one(image, prompt) for _ in range(args.repeats)]
        enc.append(statistics.median(x[0] for x in r))
        dec.append(statistics.median(x[1] for x in r))
        cpu.append(statistics.median(x[2] for x in r))
        tot.append(statistics.median(x[0] + x[1] + x[2] for x in r))
        nd.append(r[0][3])
        more.append(statistics.median(x[4] for x in r))
        if i % 10 == 0 or i == len(frames):
            print(
                f"  [{i}/{len(frames)}] {1000 * tot[-1]:.0f} ms ({1000 * enc[-1]:.0f} image encoder, {1000 * dec[-1]:.0f} text prompt and decoder)"
            )

    def ms(xs):
        return round(1000 * statistics.median(xs), 1)

    rec = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint_path,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "n_frames": len(frames),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "confidence_threshold": args.confidence_threshold,
        "precision": "bfloat16 autocast",
        "model_build_s": round(build_s, 1),
        "parameters_of_the_built_model_M": round(n_params / 1e6, 1),
        "ms_per_frame": {
            "median": ms(tot),
            "mean": round(1000 * statistics.mean(tot), 1),
            "min": round(1000 * min(tot), 1),
            "max": round(1000 * max(tot), 1),
        },
        "ms_set_image_median": ms(enc),
        "ms_set_text_prompt_median": ms(dec),
        "ms_mask_to_cpu_median": ms(cpu),
        "ms_each_further_object_in_a_frame_median": ms(more),
        "median_instances_per_frame": statistics.median(nd),
        "covers": "set_image + set_text_prompt + best instance to the CPU as a boolean mask, one object per frame; excludes "
        "image decoding and model building",
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=2))
        print(f"-> {args.out}")


# ---------------------------------------------------------------------------
# detector: the missing term in the cost of a network that has to be given a box
# ---------------------------------------------------------------------------
def bench_detector(args) -> None:
    """A per-object network that predicts its mask inside a crop (ZebraPose) needs a box from somewhere, and the released checkpoints
    come with no detector. This puts a number on the omitted term with a representative detector, torchvision's Mask R-CNN (ResNet-50
    FPN, the architecture of the BOP default detections), on the same GPU, at the BOP frame size, with image decoding outside the clock
    and a warm-up before it. It measures the COST of a detector of that class, not the quality of any detector for these objects:
    nothing is trained here, and the weights only matter in that they make the detection count realistic (random weights score below
    the threshold everywhere and skip the mask head)."""
    import cv2
    import torch
    import torchvision

    sets = load_manifest(args.manifest, args.bop_root)["sets"]
    frames = sorted({img for vs in sets for img in vs["images"]})
    step = max(1, len(frames) // args.n_frames)
    frames = frames[::step][: args.n_frames]
    device = torch.device(args.device)
    imgs = []
    for p in frames:
        bgr = cv2.imread(p)
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        imgs.append(rgb.to(device))

    try:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_note = "COCO weights"
    except Exception:
        weights, weights_note = None, "random weights"
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights, min_size=args.min_size, max_size=args.max_size
        )
    except Exception as e:
        print(
            f"[warn] could not load pretrained weights ({type(e).__name__}: {e}); "
            f"falling back to random weights, which understates the mask-head cost"
        )
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, min_size=args.min_size, max_size=args.max_size
        )
        weights_note = "random weights (mask head mostly skipped)"
    model = model.to(device).eval()

    medians, n_det = [], []
    with torch.no_grad():
        for _ in range(args.warmup):
            model([imgs[0]])
        for _ in range(args.repeats):
            ts = []
            for im in imgs:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                out = model([im])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.time() - t0)
                n_det.append(len(out[0]["boxes"]))
            medians.append(1000 * statistics.median(ts))

    rec = {
        "manifest": args.manifest,
        "git_revision": git_revision(),
        "detector": "torchvision Mask R-CNN ResNet-50 FPN",
        "weights": weights_note,
        "input": f"{args.min_size}-{args.max_size} px, batch 1",
        "n_frames": len(imgs),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "median_ms_per_frame_per_repeat": [round(m, 1) for m in medians],
        "median_ms_per_frame": round(statistics.median(medians), 1),
        "median_detections_per_frame": statistics.median(n_det),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "excludes": "image decoding and host-to-device copy of the frame",
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="per-frame cost of the segmenters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--manifest", required=True)
        p.add_argument(
            "--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)"
        )
        p.add_argument("--device", default="cuda")
        p.add_argument("--out", default="")

    h = sub.add_parser("head", help="DA3 with and without the head (+ the records of SAM 3 and CNOS)")
    common(h)
    h.add_argument("--head", required=True, help="without the head there is nothing to measure the head against")
    h.add_argument("--base-model", default=DA3_REPO)
    h.add_argument("--revision", default=DA3_REVISION)
    h.add_argument("--process-res", type=int, default=PROCESS_RES)
    h.add_argument("--n-sets", type=int, default=20, help="view sets to time, spread over the manifest")
    h.add_argument("--repeats", type=int, default=3)
    h.add_argument("--warmup", type=int, default=2)
    h.add_argument("--sam3-cost", default="", help="output of `cost.py sam3`")
    h.add_argument("--cnos-record", default="", help="cnos_run.json of the mask dump")
    h.set_defaults(fn=bench_head)

    s = sub.add_parser("sam3", help="SAM 3, model built once (run in the SAM 3 environment)")
    common(s)
    s.add_argument("--checkpoint-path", required=True)
    s.add_argument("--prompts-file", default=None)
    s.add_argument("--n-sets", type=int, default=20)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--confidence-threshold", type=float, default=0.2)
    s.set_defaults(fn=bench_sam3)

    d = sub.add_parser("detector", help="a standard detector (Mask R-CNN R50-FPN)")
    common(d)
    d.add_argument("--n-frames", type=int, default=60)
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--warmup", type=int, default=10)
    d.add_argument(
        "--min-size",
        type=int,
        default=480,
        help="shorter side the detector resizes to; 480 is the BOP frame's own, torchvision's default of 800 would upsample it",
    )
    d.add_argument("--max-size", type=int, default=640)
    d.set_defaults(fn=bench_detector)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
