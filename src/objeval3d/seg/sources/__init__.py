"""Masks from segmenters other than the head: SAM 3 (text prompt), CNOS (CAD-model templates), ZebraPose (per-object network).

Each writes `{set_id}_{view:02d}.png` for the views of a manifest, which is the first name `--mask-source dir` looks for. That naming
matters: a frame can belong to view sets of several different objects, so masks named by image id would collide between objects and
silently hand one object's mask to another. Each runs in its own environment (docs/ENVIRONMENTS.md) because the three codebases do not
share one; nothing in this package imports them at module scope.
"""
