"""Build multi-view evaluation sets ("view sets") from a BOP scene, and extend them to more views.

A view set is N frames of the *same static scene* that all show the same target object, with a controlled angular baseline between
them - the input a feed-forward multi-view model is given.

Selection rules
---------------
* only frames from an allowed split (`--seg-index`, `--use-split`: the frames the segmentation head never saw);
* the target object must be visible enough in *every* frame of the set (`--min-visib-fract`, `--min-px`);
* every pair of frames must differ by an angle in [`--min-angle`, `--max-angle`] degrees, measured between the directions from the
  object's origin to the two camera centres. Too small is a degenerate baseline, too large leaves almost no shared surface, which
  conflates reconstruction error with a lack of overlap.

The manifest this writes is the single source of truth for every later stage (reconstruction, ground-truth clouds, alignment,
metrics). The manifests of the paper are in `configs/manifests/`; this module regenerates them from the BOP datasets.

    python -m objeval3d.data.view_sets build --bop-root BOP/lmo --dataset lmo --split test --scenes 2 \\
        --obj-ids 1,5,6,8,9,10,11,12 --seg-index workspace/seg_data/lmo/index.json --use-split test \\
        --n-views 3 --sets-per-object 10 --min-px 1500 --out workspace/manifests/lmo_v3.json

`extend` keeps every set of a manifest and adds frames to it. The paper's six-view measurements use the SAME sets as the three-view
ones with three more frames each, otherwise the comparison would mix the effect of the extra views with that of a different draw of
frames. New frames come from the same scene and obey the manifest's own rules (split, visibility, visible pixels, and the pairwise
angle band against EVERY frame already in the set), drawn like `build` draws them (a random permutation of the candidates, first
feasible wins) with a seed per set. A set that cannot reach the wanted number of views is listed and left out, not relaxed silently.
The band for the added frames can be changed with `--min-angle` / `--max-angle` and is then written into the manifest: a looser band
is a different capture regime and has to be reported as one.

    python -m objeval3d.data.view_sets extend --manifest configs/manifests/lmo_v3.json \\
        --seg-index workspace/seg_data/lmo/index.json --n-views 6 --out workspace/manifests/lmo_v6n.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from objeval3d.data.bop import BopScene, angle_between, object_name, viewing_direction
from objeval3d.data.manifest import load_manifest


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def candidate_frames(
    scene: BopScene,
    obj_id: int,
    allowed: set[int] | None,
    min_visib: float,
    min_px: int,
) -> list[dict]:
    out = []
    for im_id in scene.im_ids:
        if allowed is not None and im_id not in allowed:
            continue
        ann = scene.annotation_for(im_id, obj_id)
        if ann is None:
            continue
        if ann["visib_fract"] < min_visib:
            continue
        if ann["px_count_visib"] >= 0 and ann["px_count_visib"] < min_px:
            continue
        out.append(
            {
                "im_id": im_id,
                "gt_id": ann["gt_id"],
                "dir": viewing_direction(ann["R_m2c"], ann["t_m2c"]),
                "R_m2c": ann["R_m2c"],
                "t_m2c": ann["t_m2c"],
                "visib_fract": ann["visib_fract"],
                "px": ann["px_count_visib"],
            }
        )
    return out


def greedy_sets(
    cands: list[dict],
    n_views: int,
    n_sets: int,
    min_angle: float,
    max_angle: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Pick `n_sets` index-tuples into `cands` obeying the pairwise angle band.

    Anchors are spread over the candidate list rather than drawn at random so
    that the resulting sets cover the whole trajectory instead of clustering.
    """
    if len(cands) < n_views:
        return []
    sets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    anchors = np.linspace(0, len(cands) - 1, num=min(len(cands), n_sets * 4)).astype(int)

    for a in anchors:
        if len(sets) >= n_sets:
            break
        chosen = [int(a)]
        order = rng.permutation(len(cands))
        for j in order:
            j = int(j)
            if j in chosen:
                continue
            ok = True
            for c in chosen:
                ang = angle_between(cands[j]["dir"], cands[c]["dir"])
                if not (min_angle <= ang <= max_angle):
                    ok = False
                    break
            if ok:
                chosen.append(j)
            if len(chosen) == n_views:
                break
        if len(chosen) == n_views:
            key = tuple(sorted(chosen))
            if key not in seen:
                seen.add(key)
                sets.append(sorted(chosen))
    return sets


def _allowed_frames(seg_index: str, use_split: str) -> dict[int, set[int]] | None:
    """The frames of one split of the segmentation dataset, per scene (None: every frame is allowed)."""
    if not seg_index or use_split == "any":
        return None
    with open(seg_index) as f:
        idx = json.load(f)
    allowed: dict[int, set[int]] = {}
    for rec in idx["splits"][use_split]:
        allowed.setdefault(rec["scene_id"], set()).add(rec["im_id"])
    return allowed


def build(args) -> None:
    bop_root = Path(args.bop_root)
    split_dir = bop_root / args.split
    obj_ids = parse_int_list(args.obj_ids)
    rng = np.random.default_rng(args.seed)

    allowed_by_scene = _allowed_frames(args.seg_index, args.use_split)
    if allowed_by_scene is not None:
        n_allowed = sum(len(v) for v in allowed_by_scene.values())
        print(f"Restricted to {n_allowed} frames from split '{args.use_split}'")

    scene_ids = (
        parse_int_list(args.scenes)
        if args.scenes
        else sorted(int(p.name) for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())
    )

    sets_out = []
    for sid in scene_ids:
        scene_dir = split_dir / f"{sid:06d}"
        if not scene_dir.exists():
            continue
        scene = BopScene(scene_dir)
        allowed = None if allowed_by_scene is None else allowed_by_scene.get(sid, set())
        if allowed is not None and not allowed:
            continue

        for obj_id in obj_ids:
            cands = candidate_frames(scene, obj_id, allowed, args.min_visib_fract, args.min_px)
            picks = greedy_sets(
                cands,
                args.n_views,
                args.sets_per_object,
                args.min_angle,
                args.max_angle,
                rng,
            )
            if not picks:
                print(
                    f"scene {sid:06d} obj {obj_id:2d} ({object_name(args.dataset, obj_id)}): "
                    f"{len(cands)} candidate frames -> 0 view sets"
                )
                continue

            for k, idxs in enumerate(picks):
                views = [cands[i] for i in idxs]
                angles = [
                    round(angle_between(views[a]["dir"], views[b]["dir"]), 2)
                    for a in range(len(views))
                    for b in range(a + 1, len(views))
                ]
                sets_out.append(
                    {
                        "set_id": f"{args.dataset}_s{sid:06d}_o{obj_id:02d}_v{k:02d}",
                        "dataset": args.dataset,
                        "split": args.split,
                        "scene_id": sid,
                        "obj_id": obj_id,
                        "obj_name": object_name(args.dataset, obj_id),
                        "im_ids": [v["im_id"] for v in views],
                        "gt_ids": [v["gt_id"] for v in views],
                        "images": [str(scene.rgb_path(v["im_id"])) for v in views],
                        "K": [scene.K(v["im_id"]).tolist() for v in views],
                        "R_m2c": [v["R_m2c"].tolist() for v in views],
                        "t_m2c": [v["t_m2c"].reshape(3).tolist() for v in views],
                        "visib_fract": [round(v["visib_fract"], 4) for v in views],
                        "px_count_visib": [v["px"] for v in views],
                        "pair_angles_deg": angles,
                    }
                )
            print(
                f"scene {sid:06d} obj {obj_id:2d} ({object_name(args.dataset, obj_id)}): "
                f"{len(cands)} candidate frames -> {len(picks)} view sets"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "bop_root": str(bop_root),
                "split": args.split,
                "n_views": args.n_views,
                "obj_ids": obj_ids,
                "selection": {
                    "min_angle": args.min_angle,
                    "max_angle": args.max_angle,
                    "min_visib_fract": args.min_visib_fract,
                    "min_px": args.min_px,
                    "use_split": args.use_split,
                    "seed": args.seed,
                },
                "sets": sets_out,
            },
            f,
            indent=2,
        )

    print(f"\n{len(sets_out)} view sets -> {out_path}")


def extend(args) -> None:
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; this command does not overwrite")
    man = load_manifest(args.manifest, args.bop_root)
    sel = man["selection"]
    lo = sel["min_angle"] if args.min_angle is None else args.min_angle
    hi = sel["max_angle"] if args.max_angle is None else args.max_angle
    split_dir = Path(man["bop_root"]) / man["split"]

    allowed_by_scene = _allowed_frames(args.seg_index, sel["use_split"])

    scenes, cands_cache = {}, {}
    kept, dropped = [], []
    for i, s in enumerate(man["sets"]):
        sid, obj = s["scene_id"], s["obj_id"]
        if sid not in scenes:
            scenes[sid] = BopScene(split_dir / f"{sid:06d}")
        scene = scenes[sid]
        if (sid, obj) not in cands_cache:
            allowed = None if allowed_by_scene is None else allowed_by_scene.get(sid, set())
            cands_cache[(sid, obj)] = candidate_frames(scene, obj, allowed, sel["min_visib_fract"], sel["min_px"])
        cands = cands_cache[(sid, obj)]
        by_im = {c["im_id"]: c for c in cands}
        if any(im not in by_im for im in s["im_ids"]):
            dropped.append((s["set_id"], "an original frame is not a candidate under the manifest's own rules"))
            continue
        chosen = [by_im[im] for im in s["im_ids"]]
        have = {c["im_id"] for c in chosen}
        rng = np.random.default_rng([args.seed, i])
        for j in rng.permutation(len(cands)):
            c = cands[int(j)]
            if c["im_id"] in have:
                continue
            if all(lo <= angle_between(c["dir"], k["dir"]) <= hi for k in chosen):
                chosen.append(c)
                have.add(c["im_id"])
            if len(chosen) == args.n_views:
                break
        if len(chosen) < args.n_views:
            dropped.append((s["set_id"], f"only {len(chosen)} of {args.n_views} frames fit the angle band"))
            continue
        chosen.sort(key=lambda c: c["im_id"])
        angles = [
            round(angle_between(chosen[a]["dir"], chosen[b]["dir"]), 2)
            for a in range(len(chosen))
            for b in range(a + 1, len(chosen))
        ]
        kept.append(
            {
                "set_id": s["set_id"],
                "dataset": s["dataset"],
                "split": s["split"],
                "scene_id": sid,
                "obj_id": obj,
                "obj_name": s["obj_name"],
                "im_ids": [c["im_id"] for c in chosen],
                "gt_ids": [c["gt_id"] for c in chosen],
                "images": [str(scene.rgb_path(c["im_id"])) for c in chosen],
                "K": [scene.K(c["im_id"]).tolist() for c in chosen],
                "R_m2c": [c["R_m2c"].tolist() for c in chosen],
                "t_m2c": [c["t_m2c"].reshape(3).tolist() for c in chosen],
                "visib_fract": [round(c["visib_fract"], 4) for c in chosen],
                "px_count_visib": [c["px"] for c in chosen],
                "pair_angles_deg": angles,
                "base_im_ids": list(s["im_ids"]),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": man["dataset"],
                "bop_root": man["bop_root"],
                "split": man["split"],
                "n_views": args.n_views,
                "obj_ids": man["obj_ids"],
                "selection": sel,
                "extended_from": args.manifest,
                "extension_seed": args.seed,
                "extension_angle_band": [lo, hi],
                "sets": kept,
            },
            indent=2,
        )
    )

    print(
        f"{man['dataset']}: {len(man['sets'])} sets in {args.manifest}; {len(kept)} extended to {args.n_views} views, "
        f"{len(dropped)} could not be -> {out_path}"
    )
    by_obj = {}
    for s in man["sets"]:
        by_obj.setdefault(s["obj_name"], [0, 0])[0] += 1
    for s in kept:
        by_obj[s["obj_name"]][1] += 1
    for name, (n, k) in sorted(by_obj.items()):
        print(f"  {name:22s} {k:2d} of {n:2d}")
    for sid, why in dropped:
        print(f"  left out: {sid}: {why}")
    if kept:
        pa = np.array([a for s in kept for a in s["pair_angles_deg"]])
        print(
            f"pairwise angles of the extended sets: median {np.median(pa):.1f} deg, "
            f"min {pa.min():.1f}, max {pa.max():.1f}"
        )
    # every kept set contains its original frames by construction; say so in the log
    assert all(set(s["base_im_ids"]) <= set(s["im_ids"]) for s in kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build view sets from a BOP scene")
    b.add_argument("--bop-root", required=True, help="root of one BOP dataset, e.g. .../bop/lmo")
    b.add_argument("--dataset", required=True)
    b.add_argument("--split", default="test")
    b.add_argument("--scenes", default="", help="comma separated scene ids; empty = all")
    b.add_argument("--obj-ids", required=True)
    b.add_argument("--out", required=True)
    b.add_argument(
        "--seg-index",
        default="",
        help="index.json from objeval3d.seg.prepare_data - restricts view sets to frames of --use-split so the "
        "segmentation head is evaluated out-of-sample",
    )
    b.add_argument("--use-split", default="test", choices=["test", "val", "train", "any"])
    b.add_argument("--n-views", type=int, default=3)
    b.add_argument("--sets-per-object", type=int, default=10, help="per scene per object")
    b.add_argument("--min-angle", type=float, default=12.0)
    b.add_argument("--max-angle", type=float, default=50.0)
    b.add_argument("--min-visib-fract", type=float, default=0.85)
    b.add_argument("--min-px", type=int, default=1500)
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(run=build)

    e = sub.add_parser("extend", help="add frames to every set of a manifest")
    e.add_argument("--manifest", required=True, help="the manifest to extend (kept as it is)")
    e.add_argument("--bop-root", default=None, help="root of the BOP dataset (default: see objeval3d.data.manifest)")
    e.add_argument("--seg-index", default="", help="index.json of objeval3d.seg.prepare_data, as for `build`")
    e.add_argument("--n-views", type=int, default=6)
    e.add_argument(
        "--min-angle",
        type=float,
        default=None,
        help="lower bound of the pairwise angle band for the ADDED frames against every frame of the set; "
        "default: the manifest's own. Needed where six frames cannot sit in the original band "
        "(YCB-V's cameras travel along one path)",
    )
    e.add_argument("--max-angle", type=float, default=None, help="the same for the upper bound")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    e.set_defaults(run=extend)

    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
