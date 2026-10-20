# Running the experiments of the paper

Three levels, by what you have and how long you want to wait.

| level | what you get | what you need | time |
|---|---|---|---|
| **1. Recompute** | every table of the paper, Figures 5 and 6 | this repository, no GPU, no data | minutes |
| **2. Re-run a part** | stages 1–4 on a few view sets, or one head | the BOP datasets, DA3-LARGE, a CUDA GPU | an hour |
| **3. Re-run everything** | the result files themselves | all of the above plus the other segmenters and VGGT-1B | days on one GPU |

All commands are run from the repository root. Paths are relative to it; `workspace/` is where everything that is downloaded or produced
goes (ignored by git, see [DATA.md](DATA.md)). Each command prints `--help`.

```bash
export OBJEVAL_BOP_ROOT=$PWD/workspace/data/bop      # holds lmo/ and ycbv/ (levels 2 and 3)
```

The environments are separate on purpose, because DA3, VGGT, SAM 3 and CNOS do not agree on their dependencies: [ENVIRONMENTS.md](ENVIRONMENTS.md).
The `# env:` comment at a command says which one.

---

## Level 1: recompute the tables from the released results

```bash
pip install -e ".[analysis,dev]"

python -m objeval3d.analysis.tables                    # every table, as markdown
python -m objeval3d.analysis.tables table3 tableG2     # selected ones; the keys are the printed numbers: table3 = Table 3, tableA1 = Table A.1
python -m objeval3d.figures.alignment                  # Figure 5   -> workspace/figures
python -m objeval3d.figures.rank                       # Figure 6
pytest                                                 # the tables against the recorded ones, the code on synthetic data
```

`pytest` compares every cell of every table with the recorded one, so a change of the code or of the data that moves one number fails
a test.

### Which result files feed which table

| table | result files (`results/<name>.json.gz`) |
|---|---|
| 1 | `lmo_gt_full`, `ycbv_gt`; manifests `lmo_v3`, `ycbv_v3` |
| 2 | `lmo_gt_full`, `lmo_head_full` |
| 3 | `lmo_{gt_full,head_full,dir_zebrapose,dir_cnos,dir_sam3_tuned}`, `ycbv_b20_{gt,head,dir_zebrapose,dir_cnos,dir_sam3}` |
| 4 | `cost/*` (measurements), `lmo_gt_full`, `ycbv_gt` (for the number of objects) |
| 5 | `lmo_gt_full`, `ycbv_gt` |
| 6 | `lmo_gt_full_e9`, `ycbv_gt_full_e9` |
| 7, G.1, G.2 | `burde_main.jsonl.gz`, `burde_controls.json.gz` (G.2 also `lmo_gt_full`, `ycbv_gt`) |
| 8, 9 | `lmo_{gt_da3_gt_maskconf,head_maskconf,vggt_gt,vggt_dir}`, `ycbv_{gt_maskconf,head_maskconf,vggt_gt,vggt_dir}` |
| A.1 | `lmo_head_v2`, `lmo_head_v4`, `lmo_head_v6` |
| A.2 | `lmo_gt_full`, `lmo_gt_full_v6n`, `ycbv_gt`, `ycbv_gt_v6n` |
| A.3 | `lmo_gt_full`, `ycbv_gt` |
| F.1 | `src/objeval3d/data/sam3_prompts.json` |

---

## Level 2 and 3: re-run the pipeline

The order of the stages, with the data each one needs:

```
BOP datasets ──► 1. segmentation labels ──► 2. view sets ──► 3. train a head ──┐
                                                                               ▼
other segmenters' masks (4) ─────────────────────────► 5. reconstruct + clean ──► 6. align + score ──► tables
```

### 1. Segmentation labels from BOP

BOP ships the visible mask of every annotated object, so the labels cost nothing. The class index is the position in `--obj-ids`.
`# env: da3`

```bash
# LM-O: one scene, eight objects. The trajectory is cut into 20 blocks, assigned cyclically to train, train, train, validation, test:
# the held-out frames are four separate segments of the trajectory, not every fifth frame.
python -m objeval3d.seg.prepare_data --bop-root workspace/data/bop/lmo --dataset lmo --split test --scenes 2 \
    --obj-ids 1,5,6,8,9,10,11,12 --split-by block --n-blocks 20 --min-visib-fract 0.3 --min-px 400 \
    --out-dir workspace/seg_data/lmo

# YCB-V within one capture: the same block split inside the four test scenes 48, 50, 53, 58
python -m objeval3d.seg.prepare_data --bop-root workspace/data/bop/ycbv --dataset ycbv --split test --scenes 48,50,53,58 \
    --obj-ids 1,2,4,5,6,9,10,11,13,14,15,19,20 --split-by block --n-blocks 20 --frame-stride 6 \
    --min-visib-fract 0.3 --min-px 800 --out-dir workspace/seg_data/ycbv_b20
```

Three more datasets are only for the checks of the head (the harder split with 5 blocks, and the head that is tested on scenes it never saw;
note that the 5-block LM-O set was prepared with `--min-px 800`, not 400):

```bash
python -m objeval3d.seg.prepare_data --bop-root workspace/data/bop/lmo --dataset lmo --split test --scenes 2 \
    --obj-ids 1,5,6,8,9,10,11,12 --split-by block --n-blocks 5 --min-visib-fract 0.3 --min-px 800 --out-dir workspace/seg_data/lmo_b5
python -m objeval3d.seg.prepare_data --bop-root workspace/data/bop/ycbv --dataset ycbv --split test --scenes 48,50,53,58 \
    --obj-ids 1,2,4,5,6,9,10,11,13,14,15,19,20 --split-by block --n-blocks 5 --frame-stride 6 \
    --min-visib-fract 0.3 --min-px 800 --out-dir workspace/seg_data/ycbv_b5
python -m objeval3d.seg.prepare_data --bop-root workspace/data/bop/ycbv --dataset ycbv --split test \
    --scenes 48,49,50,51,52,53,54,55,56,57,58,59 --obj-ids 1,2,3,4,5,6,9,10,11,13,14,15,19,20 \
    --split-by scene --val-scenes 51,59 --test-scenes 48,50,53,58 --frame-stride 6 \
    --min-visib-fract 0.3 --min-px 800 --out-dir workspace/seg_data/ycbv
```

### 2. View sets

The manifests used by the paper are in `configs/manifests/` and the commands below rebuild them **identically** (the selection is random with
a fixed seed per set), so this step is optional.

```bash
# LM-O, three views per set, ten sets per object
python -m objeval3d.data.view_sets build --bop-root workspace/data/bop/lmo --dataset lmo --split test --scenes 2 \
    --obj-ids 1,5,6,8,9,10,11,12 --seg-index workspace/seg_data/lmo/index.json --use-split test \
    --n-views 3 --sets-per-object 10 --min-angle 12 --max-angle 50 --min-visib-fract 0.85 --min-px 1500 \
    --out workspace/manifests/lmo_v3.json
#   the manifests with 2, 4, 6 views (lmo_v2, lmo_v4, lmo_v6): the same command with --n-views 2 / 4 / 6

# YCB-V within one capture (the segmenters' comparison, Table 3)
python -m objeval3d.data.view_sets build --bop-root workspace/data/bop/ycbv --dataset ycbv --split test --scenes 48,50,53,58 \
    --obj-ids 1,2,4,5,6,9,10,11,13,14,15,19,20 --seg-index workspace/seg_data/ycbv_b20/index.json --use-split test \
    --n-views 3 --sets-per-object 6 --min-angle 12 --max-angle 50 --min-visib-fract 0.85 --min-px 3000 \
    --out workspace/manifests/ycbv_b20_v3.json

# YCB-V on the test frames of the head for unfamiliar scenes (alignment and model comparison, Tables 5, 6, 8, 9)
python -m objeval3d.data.view_sets build --bop-root workspace/data/bop/ycbv --dataset ycbv --split test --scenes 48,50,53,58 \
    --obj-ids 1,2,3,4,5,6,9,10,11,13,14,15,19,20 --seg-index workspace/seg_data/ycbv/index.json --use-split test \
    --n-views 3 --sets-per-object 6 --min-angle 12 --max-angle 50 --min-visib-fract 0.85 --min-px 3000 \
    --out workspace/manifests/ycbv_v3.json

# six views on the SAME sets (Table A.2): the three-view sets with three more frames each
python -m objeval3d.data.view_sets extend --manifest configs/manifests/lmo_v3.json --bop-root workspace/data/bop/lmo \
    --seg-index workspace/seg_data/lmo/index.json --n-views 6 --out workspace/manifests/lmo_v6n.json
python -m objeval3d.data.view_sets extend --manifest configs/manifests/ycbv_v3.json --bop-root workspace/data/bop/ycbv \
    --seg-index workspace/seg_data/ycbv/index.json --n-views 6 --min-angle 5 --out workspace/manifests/ycbv_v6n.json
```

### 3. Train a segmentation head

Only the head is trained (32.65 M parameters); the backbone is frozen. All heads of the paper use the same settings. `# env: da3`

```bash
python -m objeval3d.seg.train --data-dir workspace/seg_data/lmo --out-ckpt workspace/ckpt/lmo_seg_head.pt \
    --epochs 20 --batch-size 2 --lr 1e-4 --image-size 504 --augment

python -m objeval3d.seg.train --data-dir workspace/seg_data/ycbv_b20 --out-ckpt workspace/ckpt/ycbv_b20_seg_head.pt \
    --epochs 20 --batch-size 2 --lr 1e-4 --image-size 504 --augment

# score a head on the frames it has not seen (mean IoU over the objects, confusion matrix)
python -m objeval3d.seg.confusion --data-dir workspace/seg_data/lmo --head workspace/ckpt/lmo_seg_head.pt --split test \
    --out workspace/results/lmo_seg_confusion.json
```

The other heads of the paper are the same command with another data folder and output; `configs/heads.json` lists them all:

| head file | `--data-dir` | extra |
|---|---|---|
| `lmo_b20_seg_head.pt` | `workspace/seg_data/lmo` | (a second run on the data of the main head, the split check) |
| `lmo_b5_seg_head.pt` | `workspace/seg_data/lmo_b5` | |
| `ycbv_b5_seg_head.pt` | `workspace/seg_data/ycbv_b5` | |
| `ycbv_cross_scene_seg_head.pt` (trained as `ycbv_seg_head.pt`) | `workspace/seg_data/ycbv` | |
| `lmo_seg_head_n100.pt`, `_n250.pt`, `_n500.pt` | `workspace/seg_data/lmo` | `--max-train 100` / `250` / `500`: a random subset, seed 42 |

The scores of the split checks are `python -m objeval3d.seg.confusion … --split test` of each head on its own data folder. The training logs of
the paper's heads are in `results/training/`; the checkpoints are on Hugging Face ([MODELS.md](MODELS.md)), so this step can be skipped.
It takes some 15–20 minutes per head on one GPU.

### 4. Masks from the other segmenters

Each in its own environment; each writes one PNG per view of every set into `--out-dir`, which the next stage reads with `--mask-source dir`.
The commands to get the code and the weights are in [MODELS.md](MODELS.md).

```bash
# SAM 3, prompted with the object's name (prompts of Table F.1 are built in; threshold 0.2)       # env: sam3
python -m objeval3d.seg.sources.sam3 --manifest configs/manifests/lmo_v3.json --out-dir workspace/masks/lmo_sam3_tuned \
    --checkpoint-path workspace/ckpt/sam3.pt --confidence-threshold 0.2
python -m objeval3d.seg.sources.sam3 --manifest configs/manifests/ycbv_b20_v3.json --out-dir workspace/masks/ycbv_b20_sam3 \
    --checkpoint-path workspace/ckpt/sam3.pt --confidence-threshold 0.2

# CNOS: render the templates of every object from the BOP models once, then match them             # env: cnos
python -m objeval3d.seg.sources.cnos_templates --cnos-root ~/cnos --models workspace/data/bop/lmo/models \
    --out-dir workspace/masks/cnos_templates/lmo
python -m objeval3d.seg.sources.cnos --manifest configs/manifests/lmo_v3.json --out-dir workspace/masks/lmo_cnos \
    --cnos-root ~/cnos --template-dir workspace/masks/cnos_templates/lmo --weights-root ~/cnos/datasets/bop23_challenge \
    --dinov2-dir ~/dinov2
#   YCB-V: the same two commands with ycbv and ycbv_b20_v3.json -> workspace/masks/ycbv_b20_cnos

# ZebraPose, given the ground-truth box of the object (per-object networks)                         # env: da3
hf download dfki-av/ZebraPose --revision b160f1bff279b83d3f17022a05afcab4647c6488 \
    --include "zebrapose_ckpts/bop/lmo/*" --include "zebrapose_ckpts/bop/ycbv/*" --local-dir workspace/ckpt/zebrapose   # 350 MB per object
python -m objeval3d.seg.sources.zebrapose --manifest configs/manifests/lmo_v3.json --out-dir workspace/masks/lmo_zebrapose \
    --zebrapose-root ~/ZebraPose --ckpt-root workspace/ckpt/zebrapose
python -m objeval3d.seg.sources.zebrapose --manifest configs/manifests/ycbv_b20_v3.json --out-dir workspace/masks/ycbv_b20_zebrapose \
    --zebrapose-root ~/ZebraPose --ckpt-root workspace/ckpt/zebrapose
```

`python -m objeval3d.seg.check_masks --manifest M --mask-dir D` scores a mask folder against the ground truth in seconds on the CPU,
before spending an hour on the reconstruction.

### 5. Reconstruct and clean (stages 1–2)

Writes one run folder per view set, `<out-root>/<set_id>__<tag>/` with `object_points.ply`, `pred.npz` (depth, confidence, cameras,
masks), `meta.json` and the overlays. `# env: da3`

```bash
R="python -m objeval3d.recon.reconstruct --manifest configs/manifests/lmo_v3.json --out-root workspace/runs/lmo"

$R --mask-source gt   --tag gt_full                                                          # ground-truth masks
$R --mask-source head --head workspace/ckpt/lmo_seg_head.pt --tag head_full                  # the head
$R --mask-source dir  --mask-dir workspace/masks/lmo_zebrapose   --tag dir_zebrapose         # an external segmenter
```

The settings that distinguish the other runs of the paper:

| to get | add to the command above |
|---|---|
| confidence percentile inside the mask (the model comparison, Tables 8 and 9) | `--conf-percentile-scope mask` |
| VGGT-1B instead of DA3-LARGE (`# env: vggt`, masks from `--mask-source dir`, see below) | `--backend vggt --vggt-repo third_party/vggt-repo` |
| 2, 4, 6 views (Table A.1), the six-view extension of the same sets (A.2) | `--manifest configs/manifests/lmo_v4.json` etc. |
| no cleaning at all, the floor | `--mask-source none`, and at the next stage `--k-lo 0.1 --k-hi 10` |
| reconstruct from a crop around the object | `--crop-to-object gt --crop-margin 0.35` |
| a head trained on fewer frames | `--mask-source head --head workspace/ckpt/lmo_seg_head_n100.pt` |

The model comparison feeds VGGT the **same masks** as DA3: the head's masks are dumped once and given to both with `--mask-source dir`.

```bash
python -m objeval3d.seg.export_masks --runs-root workspace/runs/lmo --pattern "*__head_full" \
    --manifest configs/manifests/lmo_v3.json --out-dir workspace/masks/lmo_head --allow-mask-source head
python -m objeval3d.recon.reconstruct --backend vggt --vggt-repo third_party/vggt-repo --manifest configs/manifests/lmo_v3.json \
    --out-root workspace/runs/lmo --mask-source dir --mask-dir workspace/masks/lmo_head --conf-percentile-scope mask --tag vggt_dir
```

### 6. Align and score (stages 3–4)

Needs no GPU; one process per run tag. `# env: da3`

```bash
python -m objeval3d.align.evaluate --runs-root workspace/runs/lmo --manifest configs/manifests/lmo_v3.json \
    --pattern "*__gt_full" --modes auto,oracle_raw,oracle --out workspace/results/lmo_gt_full.json
```

`--modes` takes the identifiers of [GLOSSARY.md](GLOSSARY.md): `oracle_raw` is the reference mode, `auto` the free fit, `oracle` the free
fit started at the true pose, `auto_fixedscale` the free fit with the scale locked, and `fpfh_ransac`, `fgr`, `icp_pca`, `teaserpp` the
third-party searches (Open3D 0.19 from the `baselines` extra, `pip install -e ".[baselines]"`, and TEASER++, which is built from source, see
`objeval3d.align.baselines`; with `--ransac-repeats 5` as in Table 6).
Point the analysis at your own results with `--results workspace/results` (or `export OBJEVAL_RESULTS=…`).

The depth scale of the reference mode is searched on a geometric grid of `--n-k` values (default 61, the paper's, 3.1 % apart).

### Which command wrote which result file

`<m>` is the manifest of the row; `modes` as `--modes`. "dir:X" is `--mask-source dir --mask-dir workspace/masks/X`.

| result file | manifest | stage 5 (`--tag`, settings) | `--modes` |
|---|---|---|---|
| `lmo_gt_full` | `lmo_v3` | `gt_full`, `--mask-source gt` | `auto,oracle_raw,oracle` |
| `lmo_head_full` | `lmo_v3` | `head_full`, head `lmo_seg_head.pt` | `auto,oracle_raw,oracle` |
| `lmo_dir_cnos`, `lmo_dir_zebrapose`, `lmo_dir_sam3_tuned` | `lmo_v3` | `dir_cnos` (dir:`lmo_cnos`), `dir_zebrapose` (dir:`lmo_zebrapose`), `dir_sam3_tuned` (dir:`lmo_sam3_tuned`) | `auto,oracle_raw,oracle` |
| `lmo_gt_da3_gt_maskconf`, `lmo_head_maskconf` | `lmo_v3` | `gt_da3_gt_maskconf` (gt), `head_maskconf` (head), `--conf-percentile-scope mask` | `auto,oracle_raw,oracle` |
| `lmo_vggt_gt`, `lmo_vggt_dir` | `lmo_v3` | `vggt_gt` (gt), `vggt_dir` (dir:`lmo_head`), `--backend vggt --conf-percentile-scope mask` | `oracle_raw` |
| `lmo_head_v2`, `_v4`, `_v6` | `lmo_v2`, `lmo_v4`, `lmo_v6` | `head_v2`, … with the head | `oracle_raw` |
| `lmo_gt_full_v6n` | `lmo_v6n` | `gt_full_v6n` | `auto,oracle_raw,oracle` |
| `lmo_head_n100`, `_n250`, `_n500` | `lmo_v3` | `head_n100`, … with `lmo_seg_head_n100.pt`, … | `oracle_raw` |
| `lmo_gt_full_e9`, `lmo_gt_full_v6n_e9` | `lmo_v3`, `lmo_v6n` | the runs `gt_full`, `gt_full_v6n` again | `oracle_raw,auto,fpfh_ransac,fgr,icp_pca,teaserpp --ransac-repeats 5` |
| `lmo_gt_fixedscale` | `lmo_v3` | the runs `gt_full` again | `oracle_raw,auto,auto_fixedscale` |
| `lmo_none_full` | `lmo_v3` | `none_full`, `--mask-source none` | `auto,oracle_raw,oracle --k-lo 0.1 --k-hi 10` |
| `lmo_gt_{crop2,nocrop2}` | `lmo_v3`, sets `lmo_s000002_o01_v00 … v03` (`--set-ids`) | `gt_crop2` with `--crop-to-object gt --crop-margin 0.35`, `gt_nocrop2` without; masks gt | `auto,oracle` |
| `ycbv_b20_{gt,head}` | `ycbv_b20_v3` | `ycbv_b20_gt`, `ycbv_b20_head` (head `ycbv_b20_seg_head.pt`) | `oracle_raw` |
| `ycbv_b20_dir_{cnos,zebrapose,sam3}` | `ycbv_b20_v3` | `ycbv_b20_dir_cnos` (dir:`ycbv_b20_cnos`), … | `oracle_raw` (`auto,oracle_raw` for cnos and sam3) |
| `ycbv_gt`, `ycbv_head` | `ycbv_v3` | `ycbv_gt`, `ycbv_head` (head `ycbv_cross_scene_seg_head.pt`) | `auto,oracle_raw` |
| `ycbv_gt_maskconf`, `ycbv_head_maskconf` | `ycbv_v3` | `--conf-percentile-scope mask` | `auto,oracle_raw` |
| `ycbv_vggt_gt`, `ycbv_vggt_dir` | `ycbv_v3` | `vggt_gt`, `vggt_dir` (dir:`ycbv_head`), VGGT, scope `mask` | `oracle_raw` |
| `ycbv_gt_full_e9`, `ycbv_gt_v6n`, `ycbv_gt_v6n_e9` | `ycbv_v3`, `ycbv_v6n` | the `ycbv_gt` and `gt_v6n` runs again | as for LM-O |

The files record their own arguments (`args`), including the pattern and the modes, so the table is checkable: `zcat results/lmo_gt_full.json.gz | head -c 1500`.

### 7. Cost of the segmenters (Table 4)

One model per process, on an idle GPU, the same GPU for all rows; see the docstring of `objeval3d.bench.cost` for what is and is not timed.
The measurement files behind the table are in `results/cost/`. The time of ZebraPose is the `cost` of the `zebrapose_run.json` that its
driver writes in step 4 (three runs per dataset, `results/cost/zebrapose/<dataset>_rep<k>.json`), the time of CNOS the `cost` of its
`cnos_run.json`.

```bash
python -m objeval3d.bench.cost sam3 --manifest configs/manifests/ycbv_v3.json --checkpoint-path workspace/ckpt/sam3.pt \
    --out workspace/results/cost/ycbv_sam3_cost.json                                                    # env: sam3
python -m objeval3d.bench.cost head --manifest configs/manifests/ycbv_v3.json --head workspace/ckpt/ycbv_cross_scene_seg_head.pt \
    --n-sets 20 --repeats 3 --warmup 2 --sam3-cost workspace/results/cost/ycbv_sam3_cost.json \
    --cnos-record workspace/masks/ycbv_b20_cnos/cnos_run.json --out workspace/results/cost/ycbv_cost.json   # env: da3
python -m objeval3d.bench.cost detector --manifest configs/manifests/ycbv_v3.json --out workspace/results/cost/ycbv_detector_cost.json
python -m objeval3d.bench.cost detector --manifest configs/manifests/lmo_v3.json --out workspace/results/cost/lmo_detector_cost.json

# the sizes: the elements of the tensors of every weight file in the table
python -m objeval3d.bench.params files workspace/ckpt/lmo_seg_head.pt workspace/ckpt/sam3.pt workspace/ckpt/sam_vit_h_4b8939.pth \
    ~/.cache/huggingface/hub/models--depth-anything--DA3-LARGE/snapshots/c54c26b16ec04d218e8d584ecf4bce082a9fcc20/model.safetensors \
    workspace/ckpt/zebrapose/zebrapose_ckpts/bop/lmo/ape workspace/ckpt/zebrapose/zebrapose_ckpts/bop/ycbv/banana \
    ~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth ~/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
    --json workspace/results/cost/parameter_counts.json
```

### 8. The external check on the meshes of Burde et al. (Tables 7, G.1, G.2)

No reconstruction model and no segmentation: a mesh, a scan and a pose. Data: [DATA.md](DATA.md). Uses the CPU only (hours; `--threads`).

```bash
python -m objeval3d.benchmark.bias --root workspace/data/burde --bop-root workspace/data/bop/ycbv \
    --methods bakedsdf,monosdf,nerfacto,volsdf --subsets 3,all --objects all --threads 8 --out workspace/results/burde/main
python -m objeval3d.benchmark.controls --bop-root workspace/data/bop/ycbv --objects 5,10,15 --threads 8 \
    --out workspace/results/burde/controls.json
```

### 9. The small experiments of the paper

```bash
# the wording of the SAM 3 prompt (Table F.1): re-run one object with another prompt, then score the masks on the CPU
python -m objeval3d.seg.sources.sam3 --manifest configs/manifests/lmo_v3.json --out-dir workspace/masks/wording_blue_hole_punch \
    --checkpoint-path workspace/ckpt/sam3.pt --confidence-threshold 0.2 --obj-ids 12 --prompt "12=blue hole punch"
python -m objeval3d.seg.check_masks --manifest configs/manifests/lmo_v3.json --mask-dir workspace/masks/wording_blue_hole_punch --obj-ids 12

# the confidence filter on VGGT: 0 = no filter, 40 = the paper's setting      # env: vggt, then da3
python -m objeval3d.recon.reconstruct --backend vggt --vggt-repo third_party/vggt-repo --manifest configs/manifests/lmo_v3.json \
    --out-root workspace/runs/lmo --set-ids lmo_s000002_o09_v00 --mask-source gt --conf-thresh-percentile 0 --tag vggt_gt_p0
```

### 10. The figures

Figures 5 and 6 need only the result files (level 1). Figures 1–4 are drawn from the run folders, the masks and the result files of your own
runs, named as in the table above (`workspace/runs/lmo/<set>__gt_full`, `…__head_full`, `workspace/masks/lmo_head`, …); the files go to
`workspace/figures`.

```bash
python -m objeval3d.figures.datasets --results-dir workspace/results                       # Figure 1: the overlays of the run folders
python -m objeval3d.figures.qualitative --runs-root workspace/runs/lmo --results-dir workspace/results \
    --set-id lmo_s000002_o05_v03                                                            # Figures 2 and 3: the runs need `evaluate --save-aligned`
python -m objeval3d.figures.segmenters --manifest configs/manifests/lmo_v3.json \
    --runs-root workspace/runs/lmo --masks-root workspace/masks                             # Figure 4: the masks of the head and the three segmenters
python -m objeval3d.figures.alignment                                                      # Figure 5
python -m objeval3d.figures.rank                                                           # Figure 6
```

---

## Notes

* GPU training is not bit-deterministic: a head trained again is close to the released one, but its weights are not the same. Use the
  released heads to get the tables.
* The randomised third-party searches (RANSAC in Open3D has no seed) differ from call to call; Table 6 reports the median of five calls
  (`--ransac-repeats 5`), with Open3D 0.19.0.
* The nearest-neighbour queries use as many threads as the process may run on (`objeval3d.threads`; `taskset` is respected). To run many
  `align.evaluate` processes at once, pin each to one core (`taskset -c N`).
* Tested on Linux with NVIDIA RTX 5090 and RTX PRO 6000 cards; the head and the reconstruction of three views need about 4 GB of GPU memory.
