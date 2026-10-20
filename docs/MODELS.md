# Models: exact versions, checksums, licences

Every number of the paper was produced with the versions below. They are pinned in code (`src/objeval3d/models.py`): the reconstruction
stage loads the pinned revision unless told otherwise (`--revision`), and writes the revision it used into the `meta.json` of each run.

Nothing of any third-party model is copied or patched into this repository. Code of other projects is installed from its own
repository at the commit given here, weights are downloaded from their own source.

## Reconstruction models

| model | code | weights | licence of the weights | checksum (`model.safetensors`, SHA-256) |
|---|---|---|---|---|
| Depth Anything 3 (**DA3-LARGE**) | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) at `a71fc79123a420678fff8a32034c339f328f44e9` (Apache-2.0) | [`depth-anything/DA3-LARGE`](https://huggingface.co/depth-anything/DA3-LARGE) at revision `c54c26b16ec04d218e8d584ecf4bce082a9fcc20` | CC BY-NC 4.0 | `eaf2ae06df55889ad23eb245c82e2dd2a30c0cbf7e3d873a118fa5ed27a3e421` |
| **VGGT-1B** | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) at `a288dd0f14786c93483e45524328726ab7b1b4ce` (git submodule `third_party/vggt-repo`) | [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) at revision `860abec7937da0a4c03c41d3c269c366e82abdf9` | CC BY-NC 4.0 | `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` |

**Use `DA3-LARGE`, not `DA3-LARGE-1.1`.** The successor is a different network; the segmentation heads below were trained on the
features of the pinned revision and are not valid on any other weights.

Both models are used as published: no fine-tuning, no change of the architecture. The image size given to them is 504 px on the long
side (`PROCESS_RES`), VGGT's own preprocessing resizes to 518.

## Segmentation heads (this paper)

A head is a Depth Anything 3 `DPT` decoder with a 1×1 final convolution of `num_classes` outputs, attached from outside to the frozen
backbone (`src/objeval3d/seg/head.py`): 32,650,697 parameters for the nine classes of LM-O (eight objects and the background),
32,650,862 for the fourteen of YCB-V (thirteen objects and the background); each further class adds 33 (the 1×1 convolution on 32 channels).

The trained heads are on Hugging Face: **[`abramov-de/objeval3d-heads`](https://huggingface.co/abramov-de/objeval3d-heads)**, licence
CC BY-NC 4.0 (they are trained on the features of a CC BY-NC model). The model card there is [`docs/MODEL_CARD.md`](MODEL_CARD.md); the
list with the checksums, the classes and the score of every head is [`configs/heads.json`](../configs/heads.json).

```bash
hf download abramov-de/objeval3d-heads lmo_seg_head.pt ycbv_b20_seg_head.pt --local-dir workspace/ckpt
```

Training one yourself takes some 15–20 minutes on one GPU (commands in [REPRODUCING.md](REPRODUCING.md), step 3); GPU training is not
bit-deterministic, so a head trained again is close to the released one but not identical.

## Segmenters the head is compared with

Each runs in its **own** environment (their dependencies do not agree), writes one PNG mask per view, and the reconstruction stage reads
the masks with `--mask-source dir`. The drivers are in `src/objeval3d/seg/sources/`.

| segmenter | code | weights | licence |
|---|---|---|---|
| **SAM 3** (language-driven) | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) at `5dd401d1c5c1d5c3eedff06d41b77af824517619` | `sam3.pt` from the gated repository [`facebook/sam3`](https://huggingface.co/facebook/sam3): 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`. **Access has to be requested from Meta on Hugging Face**; this repository cannot and does not redistribute it. | SAM License |
| **CNOS** (CAD-model based) | [nv-nguyen/cnos](https://github.com/nv-nguyen/cnos) at `298d1f3366171464ca271659f0e2f7a6eb8e39b4` | SAM ViT-H `sam_vit_h_4b8939.pth` (2,564,550,879 bytes, SHA-256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`) and DINOv2 ViT-L/14 `dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes, SHA-256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`); DINOv2 code at `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | MIT (CNOS), Apache-2.0 (SAM, DINOv2) |
| **ZebraPose** (one network per object) | [suyz526/ZebraPose](https://github.com/suyz526/ZebraPose) at `1cb0a1c717c61d3d23171c72fa3bfd22c17d9a76` | per-object checkpoints of the BOP models from [`dfki-av/ZebraPose`](https://huggingface.co/dfki-av/ZebraPose) at revision `b160f1bff279b83d3f17022a05afcab4647c6488` (folders `zebrapose_ckpts/bop/lmo` and `.../ycbv`) | MIT |
| Mask R-CNN R50-FPN (only the cost of the detector ZebraPose needs, Table 4) | `torchvision.models.detection.maskrcnn_resnet50_fpn`, weights `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` | torchvision | BSD-3-Clause |

The SAM 3 prompts of the paper are in [`src/objeval3d/data/sam3_prompts.json`](../src/objeval3d/data/sam3_prompts.json) (Table F.1);
the confidence threshold is 0.2.

If you cannot get access to SAM 3 you can still reproduce everything else: the SAM 3 row of the tables is recomputed from the released
result files (`results/lmo_dir_sam3_tuned.json.gz`, `results/ycbv_b20_dir_sam3.json.gz`), which need no model.

## What is *not* released

* The masks of the segmenters and the reconstructed clouds. They are produced by the commands in REPRODUCING.md from the datasets and the
  models above (a run folder is 4–9 MB, so one run of the 80 LM-O view sets is 0.3–0.7 GB; the masks of one segmenter are about 1 MB).
* The BOP datasets and the benchmark data of Burde et al.: they are downloaded from their own pages (see [DATA.md](DATA.md)), they are
  large and have their own terms.
* Weights of third-party models.
