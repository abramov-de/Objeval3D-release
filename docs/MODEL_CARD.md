---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-segmentation
base_model: depth-anything/DA3-LARGE
tags:
  - depth-anything-3
  - object-segmentation
  - 3d-reconstruction
  - bop
  - linemod-occluded
  - ycb-video
---

# objeval3d segmentation heads

Trained segmentation heads for the paper **Object-level evaluation of feed-forward multi-view reconstruction: from lightweight
segmentation to anchored alignment**. Code, the result files of every experiment and the instructions to run them:
<https://github.com/abramov-de/Objeval3D-release>.

## What a head is

A segmentation head for one capture setup: the decoder of Depth Anything 3's own depth head (`depth_anything_3.model.dpt.DPT`) with a 1×1
final convolution of `num_classes` outputs. It reads the same four intermediate feature maps of the frozen backbone of **DA3-LARGE** as
the depth head does and outputs one logit per object (and one for the background). 32.65 M parameters, about 7 % of the model; the
backbone, the depth head and the camera head are not touched. It is attached to the unmodified upstream model from outside with a forward
hook, and runs in full precision. It adds about 10–12 ms to the forward pass that the reconstruction needs anyway.

It is trained on frames of the very capture it is used on (the same objects in the same scene setup), with the visible masks that BOP
ships as labels, so no manual annotation was done. Within that capture, F@2 % and completeness of the object's reconstruction agree with
those from the ground-truth masks to within 2 % on LM-O; on scenes it was not trained on the head does not transfer (see the third row of
the table).

## Files

| file | trained on | classes | held-out mean IoU | role |
|---|---|---|---|---|
| `lmo_seg_head.pt` | LM-O, scene 2, 732 frames | 8 objects + background | 0.817 | the LM-O head of all experiments |
| `ycbv_b20_seg_head.pt` | YCB-V, scenes 48, 50, 53, 58, 684 frames | 13 objects + background | 0.942 | the YCB-V head of the within-capture experiments |
| `ycbv_cross_scene_seg_head.pt` | YCB-V, scenes 49, 52, 54–57, 1,695 frames | 14 objects + background | 0.287 on scenes 48, 50, 53, 58 | trained on other scenes than it is tested on: the test of the transfer to an unfamiliar scene |
| `variants/lmo_b20_seg_head.pt` | LM-O, 20 blocks | as above | 0.813 | the split check |
| `variants/lmo_b5_seg_head.pt` | LM-O, 5 blocks | | 0.759 | the harder split: the held-out frames form one long segment |
| `variants/ycbv_b5_seg_head.pt` | YCB-V, 5 blocks | | 0.811 | the same for YCB-V |
| `variants/lmo_seg_head_n100.pt`, `…_n250.pt`, `…_n500.pt` | a random subset of 100, 250, 500 of the 732 frames | | | the study of how many annotated frames are needed |

"Held-out mean IoU" is the mean over the objects of the intersection over union on the test frames that were not used for training or for
choosing the epoch (`python -m objeval3d.seg.confusion`). The same list with the classes, the checksums, the validation scores and the
file each score comes from is [`configs/heads.json`](https://github.com/abramov-de/Objeval3D-release/blob/main/configs/heads.json). The `SHA256SUMS` file
in this repository lists the checksums of the files.

A checkpoint is a PyTorch dictionary: `seg_head` (the state dict of the head), `num_classes`, `class_names`, `obj_ids` (BOP object ids,
in the order of the classes), `dataset`, `epoch`, `val_loss`, `val_miou`, `image_size`, `hw`, `n_train_images`.

## How to use

```bash
git clone https://github.com/abramov-de/Objeval3D-release && cd Objeval3D-release
pip install -e ".[analysis,gpu]"        # plus Depth Anything 3 at the pinned commit, see docs/ENVIRONMENTS.md
hf download abramov-de/objeval3d-heads lmo_seg_head.pt --local-dir workspace/ckpt
```

```python
from pathlib import Path
from objeval3d.recon.backends import DA3Predictor

predictor = DA3Predictor(
    device="cuda", head_ckpt="workspace/ckpt/lmo_seg_head.pt"
)  # DA3-LARGE at the pinned revision + the head
frames = sorted(Path("my_frames").glob("*.png"))  # 2-6 views of one static scene
pred = predictor.inference(image=[str(f) for f in frames], infer_seg=True)
labels = pred.seg_logits.argmax(axis=1)  # (views, H, W): 0 = background, 1..K = predictor.ckpt["class_names"][1:]
```

The images are resized to 504 px on the long side (the long side of the processed frame). To reconstruct, clean and score an object with
the protocol of the paper, see the README of the repository.

## Limitations, and what not to use it for

* **Valid only on `depth-anything/DA3-LARGE` at revision `c54c26b16ec04d218e8d584ecf4bce082a9fcc20`.** The head reads the features of
  that backbone; on any other weights (the successor `DA3-LARGE-1.1` included) its output is meaningless.
* **A head is for one capture.** It knows the objects and the scene setup of its training frames. On scenes of the same dataset that it was
  not trained on, the head `ycbv_cross_scene_seg_head.pt` reaches a mean IoU of 0.287. It is a tool for the evaluation protocol of the
  paper, where the capture is annotated by the dataset, and not a general object segmenter.
* The labels are the visible masks of BOP: an object that is more than 70 % occluded or smaller than 400 px (800 px for YCB-V) was
  ignored in training, so the head says little about such objects.
* Not evaluated, and not intended for: safety-relevant use, people, or any object that is not in the table's class lists.

## Training

All heads: 20 epochs, batch size 2, AdamW with learning rate 1e-4 (cosine decay to 1 % of it) and weight decay 1e-4, gradient clipping at
1.0, images at 504 px on the long side, augmentation on, seed 42; loss is cross-entropy (ignored pixels skipped) plus soft Dice over the foreground classes
present; the checkpoint is the epoch with the best validation mean IoU. Only the head is trained. A head trained on the 675–732 frames of one capture takes
17–18 minutes on one GPU (the LM-O heads on an NVIDIA GeForce RTX 5090, the YCB-V heads on an NVIDIA RTX PRO 6000 Blackwell); 100 frames take 6 minutes, and the
cross-scene YCB-V head (1,695 frames, RTX 5090) 43 minutes.

```bash
python -m objeval3d.seg.prepare_data --bop-root workspace/data/bop/lmo --dataset lmo --split test --scenes 2 \
    --obj-ids 1,5,6,8,9,10,11,12 --split-by block --n-blocks 20 --min-visib-fract 0.3 --min-px 400 --out-dir workspace/seg_data/lmo
python -m objeval3d.seg.train --data-dir workspace/seg_data/lmo --out-ckpt workspace/ckpt/lmo_seg_head.pt \
    --epochs 20 --batch-size 2 --lr 1e-4 --image-size 504 --augment
```

The data come from the BOP datasets: LM-O (CC BY-SA 4.0) and YCB-Video (MIT). The training logs of these heads are in the `results/training`
folder of the code repository. Training again gives about the same scores, not the same weights (GPU training is not
bit-deterministic).

## Licence

CC BY-NC 4.0 (non-commercial), because the heads are trained on the features of DA3-LARGE, which is CC BY-NC 4.0. The code that uses
them is Apache-2.0.

## Paper

The link to the paper will be added here.
