"""The six figures of the paper.

    python -m objeval3d.figures.alignment   Figure 5, from the result files
    python -m objeval3d.figures.rank        Figure 6, from the result files
    python -m objeval3d.figures.datasets    Figure 1, from the run folders of a reconstruction
    python -m objeval3d.figures.qualitative Figures 2 and 3, from the run folders of one view set under two mask sources
    python -m objeval3d.figures.segmenters  Figure 4, from the masks of the four segmenters

Every figure writes a PDF and a PNG into `--out-dir` (default `workspace/figures`). The style is in `objeval3d.figures.style`.
"""
