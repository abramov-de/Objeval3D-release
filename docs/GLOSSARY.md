# Glossary: the words of the paper, the identifiers of the code

The result files keep the identifiers of the code that wrote them. This page maps them to the words of the paper, and describes the
fields of a result record. The same mapping for the alignment modes is in [`src/objeval3d/modes.py`](../src/objeval3d/modes.py).

## Protocol

| paper | in the code and the files |
|---|---|
| **four stages**: segment, clean, align, score | `objeval3d.seg` / `objeval3d.recon.reconstruct` (stages 1–2), `objeval3d.align.evaluate` (stages 3–4) |
| **view set** | *n* frames of one static scene that all show the target object, with a controlled angle between viewing directions; one record of a manifest and of a result file, `set_id` such as `lmo_s000002_o09_v00` (dataset, scene, object, running number) |
| **scan** | the BOP model of the object (`models/obj_*.ply`), in millimetres; 200,000 points sampled on its surface (seed 0) and thinned at random to 60,000 (`--gt-sample`, `--gt-keep`) |
| **visible part of the scan** | the part of the scan that at least one camera of the view set can see (back-face culling plus a z-buffer); completeness is measured against it |
| **ground-truth pose** | the pose that comes with the dataset (`cam_R_m2c`, `cam_t_m2c`); on your own setup the same role is played by a one-time calibration |
| **independent anchor** | the ground-truth pose used as the placement of the object, instead of fitting it to the cloud |
| **cleaning** | back-projection of the masked depth into the reconstruction's world frame, dropping the points below the 40th percentile of the depth confidence (`--conf-thresh-percentile`, `--conf-percentile-scope frame\|mask`) |

## Alignment modes

| paper | mode identifier | what it is allowed to look at |
|---|---|---|
| **reference mode** | `oracle_raw` | the ground-truth pose fixes six of the seven parameters of the similarity transform; only the depth scale is searched (61 values, geometric grid 0.4–2.5 × the initial scale, 3.1 % spacing) |
| **free fit** | `auto` | a deterministic search over all seven parameters (rotation, translation, scale) on the two clouds alone: the usual way of aligning a reconstruction to a scan |
| free fit started at the ground-truth pose | `oracle` | the reference-mode solution followed by a trimmed ICP (appendix) |
| free fit with the scale locked | `auto_fixedscale` | `auto` with the scale equal to the one the reference mode found |
| third-party searches | `fpfh_ransac`, `fgr`, `icp_pca`, `teaserpp`; suffix `_rigid` locks the scale | Table 6; Open3D 0.19 and TEASER++ |
| the free fit **recovered the pose** ("converged") | computed by `analysis/placement.py` from the free fit's transform and `recon_centroid` | the free fit's pose lies within one symmetry element of the true pose: rotation < 15° and cloud-centre shift < 0.10 diameters (`ROT_DEG`, `CENTRE_REL`) |

## Metrics

All distances are in millimetres and, with the suffix `_rel_diam`, as a fraction of the object's diameter.

| paper | key in `alignments[mode]` | definition |
|---|---|---|
| **accuracy** | `accuracy.{mean,median,rmse,p95}` | distance from each point of the cloud to the nearest point of the **full** scan (a point is correct if it lies on the object anywhere) |
| **completeness** | `completeness.{…}` | distance from each point of the **visible** part of the scan to the nearest point of the cloud; `completeness_full` is the same against the full scan |
| **F@1 %, F@2 %, F@5 %** | `fscore.tau_1pct`, `tau_2pct`, `tau_5pct` → `{precision, recall, fscore}` | F-score at the threshold 1, 2, 5 % of the diameter; F@2 % is the one the tables print |
| Chamfer distance | `chamfer_l1_mm` | mean accuracy plus mean completeness |
| scale | `sim3.scale` | the depth scale the search found |
| symmetry element used | `symmetry_index` | the element of the object's symmetry group that matched best (as `bop_toolkit` resolves symmetries for pose error) |

**Statistic convention.** Inside one view set a distance is reduced by the **median over the points** of the set; across
the view sets the per-set numbers are averaged (Tables 2–4) or their median is taken (Tables 5–9), as the caption of each table says.
`accuracy.mean` and the other fields are stored too, for comparison with work that reports means.

## Fields of a result record (`results/*.json.gz`, key `results`)

| key | content |
|---|---|
| `set_id`, `obj_id`, `obj_name`, `scene_id`, `im_ids` | which view set |
| `status` | `ok`, or `too_few_points` when the cloud built from the masks has fewer than 200 points (the set is then not scored and counts as failed) |
| `mask_source`, `tag` | where the masks came from (`gt`, `head`, `dir`, `none`) and the run folder suffix |
| `mask_metrics` | IoU, precision and recall of the mask against the ground truth, pooled over the set (`aggregate`) and per view (`per_view`, with `gt_px`) |
| `alignments` | one entry per mode, with the metrics above |
| `multiview_consistency` | the "inconsistency between views": median distance of the points of each view to the points of the other views, as a fraction of the cloud's bounding-box diagonal; `max_rel` is the worst view |
| `oracle_diagnostics` | the pose each view alone implies and how much they disagree (`pose_disagreement_deg`) |
| `auto_vs_oracle` (and `vs_oracle` per mode in newer files) | the free fit against the reference placement: `rotation_deg`, `translation_mm` and `translation_rel_diam` (displacement of the cloud, per diameter), `scale_ratio`, `delta_fscore_2pct` |
| `recon_centroid` | centroid of the cleaned cloud in the model frame, which is all the placement analysis needs |
| `pair_angles_deg`, `gt_visible_fraction` | geometry of the view set |
| `n_recon_points`, `diameter_mm`, `n_symmetries` | size of the cloud, of the object, symmetry group |

## Segmenters and sources

| paper | in the code |
|---|---|
| **segmentation head**, "our method" | `objeval3d.seg.head`, trained by `objeval3d.seg.train`; mask source `head` |
| **ground-truth masks** | BOP's `mask_visib`; mask source `gt` |
| **SAM 3** (language-driven) | `objeval3d.seg.sources.sam3`; the result files are `*_dir_sam3*`, mask source `dir` |
| **CNOS** (CAD-model based) | `objeval3d.seg.sources.cnos`; `*_dir_cnos` |
| **ZebraPose** (one network per object, given the box) | `objeval3d.seg.sources.zebrapose`; `*_dir_zebrapose` |
| **DA3-LARGE**, **VGGT-1B** | `--backend da3` / `--backend vggt` of `reconstruct` |
