# Data

The repository ships the **view-set manifests** and the **result files**. The images, the scans and the benchmark meshes are not
redistributed; they are downloaded from their own pages, under their own licences.

```
workspace/                         # $OBJEVAL_WORKSPACE, default ./workspace, ignored by git
└── data/
    ├── bop/                       # $OBJEVAL_BOP_ROOT, default <workspace>/data/bop
    │   ├── lmo/                   #   LM-O, unzipped BOP archives
    │   └── ycbv/                  #   YCB-V
    └── burde/                     #   benchmark of Burde et al. (only for Tables 7 and G.1 and the check of Appendix G)
```

## BOP datasets: LM-O and YCB-V

Download the three archives of each dataset from the [BOP website](https://bop.felk.cvut.cz/datasets/) or from the
[`bop-benchmark`](https://huggingface.co/bop-benchmark) organisation on Hugging Face, and unzip them into one folder per dataset:

| dataset | archives | size unzipped | licence |
|---|---|---|---|
| LM-O (Linemod-Occluded) | `lmo_base.zip`, `lmo_models.zip`, `lmo_test_all.zip` | 0.8 GB | CC BY-SA 4.0 |
| YCB-V | `ycbv_base.zip`, `ycbv_models.zip`, `ycbv_test_all.zip` | 16 GB | MIT |

Use the `*_test_all.zip` variant: it contains the visible masks (`mask_visib/`), from which the labels of the segmentation heads and the
ground-truth masks of the reference are taken.

```bash
mkdir -p workspace/data/bop/lmo workspace/data/bop/ycbv
for z in lmo_base lmo_models lmo_test_all;    do unzip -q $z.zip -d workspace/data/bop/lmo;  done
for z in ycbv_base ycbv_models ycbv_test_all; do unzip -q $z.zip -d workspace/data/bop/ycbv; done
```

The code reads this layout (the standard BOP one):

```
<bop_root>/<dataset>/
    models/models_info.json, obj_000001.ply …     # the scans, millimetres; diameters and symmetries
    test/<scene, 6 digits>/
        rgb/000000.png
        mask_visib/000000_000000.png              # <image>_<annotation>
        scene_camera.json, scene_gt.json, scene_gt_info.json
```

If the datasets live elsewhere, set `OBJEVAL_BOP_ROOT` (the folder that contains `lmo/` and `ycbv/`) or pass `--bop-root <…/lmo>` to a
single command. The manifests below store the paths of the machine that built them; `load_manifest` replaces them by yours, nothing else
in a manifest is touched.

## View-set manifests (`configs/manifests/`)

A view set is *n* frames of one static scene that all show the same target object, with a controlled angle between the viewing
directions. The manifest lists every set with its frames, intrinsics and ground-truth poses; it is the single source of truth of all
later stages. They were built by `python -m objeval3d.data.view_sets` and are released as they were used, so the sets are identical to
the paper's (the construction is random, with a fixed seed per set, and is reproduced bit for bit by the command in
[REPRODUCING.md](REPRODUCING.md), step 2).

| manifest | dataset | sets | views per set | used for |
|---|---|---|---|---|
| `lmo_v3.json` | LM-O, scene 2 | 80 (8 objects × 10) | 3 | the LM-O experiments: head against ground-truth masks and other segmenters (Tables 2, 3), the alignment (5, 6), the two models (8, 9) |
| `ycbv_b20_v3.json` | YCB-V, scenes 48, 50, 53, 58 | 72 (12 objects have sets) | 3 | YCB-V within one capture: the comparison of the segmenters (Table 3) |
| `ycbv_v3.json` | YCB-V, scenes 48, 50, 53, 58 | 78 (13 objects) | 3 | YCB-V: the alignment (Tables 5, 6) and the two models (8, 9), where the head is the one trained on other scenes |
| `lmo_v2.json`, `lmo_v4.json`, `lmo_v6.json` | LM-O | 80 each | 2, 4, 6 | Table A.1, the number of views |
| `lmo_v6n.json`, `ycbv_v6n.json` | LM-O; YCB-V | 80; 77 | 6 | Table A.2: the *same* sets as the three-view ones with three more frames each |

## Result files (`results/`)

One file per run of the alignment and scoring stage (`*.json.gz`): the arguments, and for each view set the metrics of every alignment
mode, the diagnostics of the placement and the quality of the mask. They are all that is needed to recompute the paper's tables
on a laptop, without a GPU and without the datasets:

```bash
python -m objeval3d.analysis.tables                # every table, as markdown
```

The naming is `<dataset>_<mask source>[_<variant>]`: `lmo_gt_full` is LM-O with the ground-truth masks, `lmo_head_full` with the head,
`lmo_dir_sam3_tuned` with the masks of SAM 3, `ycbv_b20_dir_cnos` YCB-V (within one capture) with CNOS, and so on. Which command wrote which file
is in the table of [REPRODUCING.md](REPRODUCING.md). Other folders:

| folder | content |
|---|---|
| `results/training/` | the log of every head's training (arguments, loss and validation IoU per epoch) |
| `results/cost/` | the measurements behind Table 4: DA3 and the head, SAM 3, CNOS, the detector, ZebraPose, the sizes of the weight files |
| `results/text/` | the small experiments of the paper: the wordings of the SAM 3 prompt and the confidence filter of VGGT |
| `results/burde_*` | the external check on the meshes of Burde et al. (Tables 7 and G.1, and the check of Appendix G) |
| `results/lmo_seg_confusion`, `*_split_*`, `ycbv_seg_confusion` | the confusion matrices of the heads on held-out frames |

The compressed result files are written by Python's `json` and contain its `NaN` literal where a value is undefined (the precision of an empty
mask, say); `json.load` reads them, a strict JSON parser needs to be told to accept `NaN`.

The result files keep the identifiers of the code that wrote them (`oracle_raw` for the *reference mode*, `auto` for the *free fit*);
[GLOSSARY.md](GLOSSARY.md) maps them to the words of the paper.

## Benchmark of Burde et al. (Tables 7 and G.1, and the check of Appendix G)

An external check of the alignment result on meshes reconstructed by other methods (BakedSDF, MonoSDF, Nerfacto, VolSDF) from 3 and from
all images of the 21 YCB-V objects. Burde, Benbihi, Burget, Sattler, *Comparative Evaluation of 3D Reconstruction Methods for Object Pose
Estimation*, WACV 2025, [arXiv:2408.08234](https://arxiv.org/abs/2408.08234). Their data is downloaded from their own server by the
code, and only what the check needs: the mesh zips (0.8 GB for the eight used here) and the camera files of the image archives (a few
hundred KB per object, read with HTTP range requests; the 2.3 GB images are never fetched).

```bash
python -m objeval3d.benchmark.data index                                              # what the server offers
python -m objeval3d.benchmark.data fetch --methods bakedsdf,monosdf,nerfacto,volsdf --subsets 3,all
python -m objeval3d.benchmark.data cameras
python -m objeval3d.benchmark.data check                                              # units and frame of every mesh
```

Their release contains no `colmap_*` and no `capturereality_*` meshes, and the subsets 3 … 300 only for six methods; `index` prints the
current listing. The benchmark's licence is theirs; the files are not part of this repository.

## What the check needs from you, summarised

| to … | you need |
|---|---|
| recompute the tables and Figures 5–6 | nothing but this repository (`pip install -e ".[analysis]"`) |
| use a trained head on your own images | the head (Hugging Face), DA3-LARGE (downloaded by the code) |
| rerun stages 1–4 on LM-O / YCB-V | the BOP archives above, DA3-LARGE (and VGGT-1B), a CUDA GPU |
| rerun the comparison with SAM 3 / CNOS / ZebraPose | their code and weights, [MODELS.md](MODELS.md); SAM 3 is gated |
| rerun the Burde check | `benchmark.data fetch` (above) and the BOP YCB-V models |
