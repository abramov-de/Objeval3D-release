"""Names of the alignment modes as they appear in the result files, and what the paper calls them.

The result files keep the identifiers of the code that wrote them. The paper uses its own words:

    reference mode  ``oracle_raw``  the position of the object comes from the independent anchor (the ground-truth pose),
                                    only the depth scale is searched, one degree of freedom
    free fit        ``auto``        a deterministic search over the seven parameters of a similarity transform (rotation,
                                    translation, scale) on the two clouds alone, the usual way of aligning

Further identifiers: ``oracle`` is the free fit's trimmed ICP started at the reference placement (appendix); the third-party searches
of Table 6 (``fpfh_ransac``, ``fgr``, ``icp_pca``, ``teaserpp``) are in `objeval3d.align.baselines`, and a ``_rigid`` suffix locks the scale.
"""

REFERENCE = "oracle_raw"
FREE_FIT = "auto"
LOCAL_FIT = "oracle"

PAPER_NAME = {
    REFERENCE: "reference mode",
    FREE_FIT: "free fit",
    LOCAL_FIT: "free fit started at the ground-truth pose",
}
