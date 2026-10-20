# Environments

The pipeline uses four models that cannot share one Python environment (Depth Anything 3 needs `numpy<2` and its own stack, VGGT its
`training` package, SAM 3 Python ≥ 3.12 and a recent PyTorch, CNOS an older stack with `pyrender`). Everything shared between them travels
on disk: the masks are written once (`--mask-source dir`) and read by every reconstruction backend, the results are JSON.

| environment | for | Python | needs a GPU |
|---|---|---|---|
| **analysis** | the tables, the statements, the figures, the tests (level 1 of [REPRODUCING.md](REPRODUCING.md)) | 3.10 – 3.13 | no |
| **da3** | the head (training, scoring), DA3-LARGE, ZebraPose, stages 3–4, the cost of the head | 3.10 | yes |
| **vggt** | VGGT-1B | 3.11 | yes |
| **sam3** | SAM 3 | 3.12 | yes |
| **cnos** | CNOS (rendering the templates and matching) | 3.10 | yes |

Run every command from the repository root; in an environment where this package is not installed, `PYTHONPATH=src` does the same.

## analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis,dev]"
pytest
```

Needs `numpy`, `scipy`, `opencv-python-headless`, `pillow`, `tqdm`, `matplotlib` and, for the tests, `pytest`. Works on Python 3.10 to 3.13
with numpy 1.26 or 2.x.

## da3

```bash
conda create -n da3 python=3.10 && conda activate da3
pip install torch==2.10.0 torchvision==0.25.0                      # the default wheels of PyPI, CUDA 12.8

git clone https://github.com/ByteDance-Seed/Depth-Anything-3 third_party/Depth-Anything-3
git -C third_party/Depth-Anything-3 checkout a71fc79123a420678fff8a32034c339f328f44e9
pip install -e third_party/Depth-Anything-3                        # its own requirements: xformers, open3d, trimesh, evo, …

pip install -e ".[analysis,gpu,baselines,dev]"                      # this package; `gpu` adds `addict`, which Depth Anything 3 imports
                                                                    # but does not declare
```

Weights are downloaded on first use from `depth-anything/DA3-LARGE` at the pinned revision ([MODELS.md](MODELS.md)). The packages the
experiments ran with:

| package | version |
|---|---|
| Python | 3.10.19 |
| torch, torchvision | 2.10.0, 0.25.0 |
| xformers | 0.0.35 |
| numpy, scipy | 1.26.4, 1.15.3 (Depth Anything 3 needs `numpy<2`) |
| open3d | 0.19.0 (Table 6; its RANSAC has no seed, so the version matters) |
| opencv-python | 4.11.0.86 |
| huggingface_hub, safetensors, omegaconf | 1.5.0, 0.7.0, 2.3.0 |
| trimesh, matplotlib, scikit-learn | 4.11.2, 3.10.8, 1.7.2 |

TEASER++ (one of the four third-party searches of Table 6) is built from source, see the docstring of `objeval3d.align.baselines`.

Two traps of the combination, both handled by `pyproject.toml`: OpenCV 4.12 and later require numpy 2 and are therefore excluded
(`opencv-python-headless<4.12`); and `pip` places `opencv-python-headless` next to the `opencv-python` that Depth Anything 3 asks for, which
is harmless, both provide the same `cv2`.

## vggt

```bash
conda create -n vggt python=3.11 && conda activate vggt
pip install torch==2.10.0 torchvision==0.25.0
git submodule update --init third_party/vggt-repo                  # facebookresearch/vggt at a288dd0f14786c93483e45524328726ab7b1b4ce
pip install -e third_party/vggt-repo
pip install -e ".[analysis]"
```

The `--vggt-repo third_party/vggt-repo` of `objeval3d.recon.reconstruct` points at the pinned checkout. The experiments ran with torch
2.10.0+cu128, numpy 2.4.6, scipy 1.17.1, opencv 5.0.0.93, kornia 0.8.3, einops 0.8.2, iopath 0.1.10.

## sam3

SAM 3 is installed from its own repository ([facebookresearch/sam3](https://github.com/facebookresearch/sam3) at
`5dd401d1c5c1d5c3eedff06d41b77af824517619`, its README has the steps) into an environment of Python 3.12 with PyTorch ≥ 2.7 and
CUDA ≥ 12.6; then, with the weights `sam3.pt` that Meta grants access to:

```bash
PYTHONPATH=src python -m objeval3d.seg.sources.sam3 --help
```

The experiments ran with Python 3.12.13, torch 2.10.0+cu128, numpy 1.26.4, timm 1.0.27, hydra-core 1.3.3.

## cnos

CNOS ([nv-nguyen/cnos](https://github.com/nv-nguyen/cnos) at `298d1f3366171464ca271659f0e2f7a6eb8e39b4`) and `bop_toolkit` are installed
as their README says, DINOv2 ([facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) at
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`) is checked out next to it and given with `--dinov2-dir`, so that nothing is fetched from GitHub at
run time. The experiments ran with Python 3.10.21, torch 2.14.0, torchvision 0.29.0, numpy 1.26.4, pyrender 0.1.45, trimesh 5.1.0,
hydra-core 1.3.7.

## Hardware of the experiments

AMD Threadripper 2990WX (32 cores), 251 GB RAM, NVIDIA RTX PRO 6000 Blackwell (96 GB) and GeForce RTX 5090 (32 GB), driver 595.99,
Linux. Stages 3–4, the analysis and the Burde check are CPU-only. A reconstruction of three views at 504 px needs a few GB of GPU memory
(the snippet of the README peaked at 3.6 GB). Timings of the cost table (Table 4) were taken on an idle card, one model per process.
