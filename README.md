# Objeval3D-release

Code for the paper **Object-level evaluation of feed-forward multi-view reconstruction: from lightweight segmentation to anchored alignment**.
The link to the paper will be added here.

Feed-forward models (DUSt3R, VGGT, Depth Anything 3, …) return the point cloud of a whole scene in an arbitrary coordinate system and at an
arbitrary scale. This code measures what they do to **one object** that fills only 1–5 % of the frame: how accurate and how
complete its reconstruction is, against a ground-truth scan.

* **A protocol** of four stages (segment, clean, align, score) that gives accuracy, completeness and F-score of a single object, for any
  model that returns depth and cameras. The alignment takes the position of the object from an *independent anchor* (the ground-truth
  pose) and searches only the depth scale (the **reference mode**), instead of fitting all seven parameters of a similarity transform to
  the cloud (the **free fit**), which absorbs part of the error it is supposed to measure.
* **A light segmentation head** on the frozen backbone of Depth Anything 3 (32.65 M parameters, about 18 minutes of training on frames of
  the capture), which replaces manual annotation of the masks. The trained heads are on
  [Hugging Face](https://huggingface.co/abramov-de/objeval3d-heads).
* **The result files** of every experiment, and the code that turns them into the tables and figures of the paper.

```
 images of one static scene (2–6 views)           scan of the object  +  its pose in the scene (the independent anchor)
          │                                                   │
          ▼                                                   │
 feed-forward model (DA3, VGGT, your own) ─► depth, confidence, cameras
          │                                                   │
  1 segment  mask of the object in every view   (head, ground truth, SAM 3, CNOS, ZebraPose, your own)
  2 clean    masked depth → point cloud, dropping the 40 % least confident points
  3 align    cloud → frame of the scan: the anchor fixes six of the seven parameters, only the depth scale is searched
  4 score    accuracy, completeness, F-score against the (visible part of the) scan
```

## The tables of the paper in a minute

No GPU, no data and no model are needed: the result files of every experiment are in [`results/`](results).

```bash
git clone https://github.com/abramov-de/Objeval3D-release && cd Objeval3D-release
python -m venv .venv && source .venv/bin/activate          # Python 3.10 or newer
pip install -e ".[analysis,dev]"

python -m objeval3d.analysis.tables table3                 # Table 3 as markdown; the key is the printed number: table5, tableA1, tableG2, …
python -m objeval3d.analysis.tables                        # all of them
python -m objeval3d.figures.alignment                      # Figure 5 -> workspace/figures (Figure 6: objeval3d.figures.rank)
pytest                                                     # the tables against the recorded ones, the code on synthetic data
```

`python -m objeval3d.analysis.tables --help` lists the tables.

## Use the segmentation head on your images

The head is a Depth Anything 3 `DPT` decoder that reads the same four feature maps as the depth head and outputs one logit per object. It
is attached to the **unmodified** upstream model from outside (a forward hook), so nothing of Depth Anything 3 is copied or patched. A
head is trained for one capture setup, the objects and the scene it was annotated on, and works with `depth-anything/DA3-LARGE` at the
pinned revision only ([the model card](docs/MODEL_CARD.md) has the details).

```bash
# environment with a GPU: see docs/ENVIRONMENTS.md (Depth Anything 3 is installed from its own repository at a pinned commit)
pip install -e ".[analysis,gpu]"
hf download abramov-de/objeval3d-heads lmo_seg_head.pt --local-dir workspace/ckpt
```

```python
from pathlib import Path

import numpy as np

from objeval3d.recon.backends import DA3Predictor
from objeval3d.recon.cloud import backproject

# DA3-LARGE at the pinned revision, and the head attached to its backbone
predictor = DA3Predictor(device="cuda", head_ckpt="workspace/ckpt/lmo_seg_head.pt")
frames = sorted(Path("my_frames").glob("*.png"))  # 2-6 views of one static scene
pred = predictor.inference(image=[str(f) for f in frames], infer_seg=True)

labels = pred.seg_logits.argmax(axis=1)  # (views, H, W): 0 = background, 1..K = the objects of the head
names = predictor.ckpt["class_names"]  # ['background', 'ape', 'can', 'cat', …]
depth, conf = pred.depth, pred.conf  # (views, H, W), at 504 px on the long side
K, E = pred.intrinsics, pred.extrinsics  # (views, 3, 3) and (views, 3, 4), world -> camera

# stage 2 for one object: its masked depth as a point cloud in the model's own world frame, without the 40 % least confident points
mask = labels == names.index("duck")
points, colours, view_id = backproject(depth, mask, K, E, pred.processed_images, conf, np.percentile(conf, 40))
```

The cloud is in the arbitrary frame and scale of the reconstruction, which is the point of the protocol: stages 3 and 4 put it into the
frame of the scan and score it.

## Evaluate your own model or segmenter

The protocol needs four things, all of which BOP-format datasets ship:

| input | in the BOP layout |
|---|---|
| the images of the view set | `test/<scene>/rgb/` |
| a scan of the object | `models/obj_*.ply` (millimetres), with `models_info.json` for the diameter and the symmetries |
| the pose of the object in each image, the independent anchor | `scene_gt.json`, `scene_camera.json` |
| a mask of the object in each image | `mask_visib/` for the ground truth; yours as one PNG per view |

**A new segmenter** needs no code in this repository: write one binary PNG per view of every view set into a folder, named
`<scene:06d>_<image:06d>.png` or `<set_id>_<view:02d>.png` (see [`dir_masks_for_set`](src/objeval3d/data/bop.py)), and run stage 2 with
`--mask-source dir --mask-dir <folder>`. The drivers of SAM 3, CNOS and ZebraPose in
[`src/objeval3d/seg/sources/`](src/objeval3d/seg/sources) are examples; `python -m objeval3d.seg.check_masks` scores a folder of masks
against the ground truth on the CPU in seconds.

**A new reconstruction model** is a class with an `inference(image, process_res)` method that returns a
[`Prediction`](src/objeval3d/recon/backends.py) (depth, optional confidence, intrinsics and world-to-camera extrinsics in the OpenCV
convention); `DA3Predictor` and `VGGTPredictor` are the two examples. The model may be scale-free: the reference mode leaves exactly one
scale parameter open.

```bash
# stages 1-2: reconstruct the view sets, segment, clean              (one run folder per view set)
python -m objeval3d.recon.reconstruct --manifest configs/manifests/lmo_v3.json --out-root workspace/runs/lmo \
    --mask-source head --head workspace/ckpt/lmo_seg_head.pt --tag head_full
# stages 3-4: align and score                                         (one JSON, one record per view set)
python -m objeval3d.align.evaluate --runs-root workspace/runs/lmo --manifest configs/manifests/lmo_v3.json \
    --pattern "*__head_full" --modes auto,oracle_raw --out workspace/results/lmo_head_full.json
# the tables of the paper, from your result files
python -m objeval3d.analysis.tables --results workspace/results table2
```

New view sets of any BOP-format dataset: `python -m objeval3d.data.view_sets build --help`.

## Run the experiments of the paper

[docs/REPRODUCING.md](docs/REPRODUCING.md) has the exact command for every result file, from the BOP labels to the tables. The datasets
are described in [docs/DATA.md](docs/DATA.md), the pinned versions of the models in [docs/MODELS.md](docs/MODELS.md), the Python
environments in [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md); [docs/GLOSSARY.md](docs/GLOSSARY.md) maps the words of the paper to the
identifiers of the code and of the result files.

## What is in the repository

```
src/objeval3d/
  data/        BOP readers, view-set construction (view_sets.py), the manifest loader, a small PLY reader/writer
  metrics/     the ground-truth cloud of an object: the scan sampled, its visible part
  seg/         the head (head.py), its training (train.py) and scoring (confusion.py), BOP labels (prepare_data.py),
               drivers of the other segmenters (sources/), the prompts of SAM 3
  recon/       stages 1-2: the two reconstruction models behind one interface (backends.py), cleaning (cloud.py), the driver
  align/       stages 3-4: the reference mode and the free fit (registration.py), third-party searches (baselines.py), the driver
  benchmark/   the external check on the meshes of Burde et al.
  bench/       cost of the segmenters: parameters and time
  analysis/    the tables (tables.py and one module per group of tables)
  figures/     the six figures of the paper
configs/       manifests of the view sets (manifests/), the released heads with their checksums (heads.json)
tools/         the upload of the heads to Hugging Face
results/       one result file per experiment, the training logs, the cost measurements
tests/         the tables against the recorded ones, the code on synthetic data
docs/          reproducing, data, models, environments, glossary, the model card of the heads
```

## Licences

The code is [Apache-2.0](LICENSE). The manifests in `configs/manifests/` and `src/objeval3d/data/models_info_*.json` contain annotations of
the BOP datasets LM-O (CC BY-SA 4.0) and YCB-Video (MIT) and keep their licences. The trained heads on Hugging Face are CC BY-NC 4.0,
because they are trained on the features of DA3-LARGE, which is CC BY-NC 4.0. The models and datasets that the code works with are
downloaded from their own pages and have their own licences ([MODELS.md](docs/MODELS.md), [DATA.md](docs/DATA.md)).
