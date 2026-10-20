"""The tables of the paper, by their printed number, and a command that prints them.

    python -m objeval3d.analysis.tables                    # every table that is implemented
    python -m objeval3d.analysis.tables table3 tableG2     # selected ones
    python -m objeval3d.analysis.tables --results DIR --format json

The numbers are the *printed* numbers of the paper: `table3` is Table 3, `tableA1` Table A.1, `tableG2` Table G.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from objeval3d.analysis import (
    alignment_bias,
    benchmark_meshes,
    costs,
    datasets,
    model_transfer,
    segmenters,
    understatement,
    views,
)
from objeval3d.analysis.common import results_dir
from objeval3d.analysis.table import Table

REGISTRY: dict[str, Callable[[Path], Table]] = {
    "table1": datasets.dataset_properties,
    "table2": segmenters.head_vs_ground_truth,
    "table3": segmenters.quality_by_source,
    "table4": costs.models_cost,
    "table5": alignment_bias.free_fit_vs_reference,
    "table6": alignment_bias.alignment_methods,
    "table7": benchmark_meshes.meshes_summary,
    "table8": model_transfer.rank_agreement,
    "table9": model_transfer.model_comparison,
    "tableA1": views.views_per_set,
    "tableA2": views.three_vs_six,
    "tableA3": views.completeness_visible_vs_full,
    "tableF1": datasets.sam3_prompts,
    "tableG1": benchmark_meshes.meshes_full,
    "tableG2": understatement.understatement_ladder,
}


def compute(key: str, results: Path | None = None) -> Table:
    return REGISTRY[key](results_dir(results))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="*", help="table keys (default: all implemented)")
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help="directory with the result files (default: $OBJEVAL_RESULTS or ./results)",
    )
    ap.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = ap.parse_args(argv)
    keys = args.tables or list(REGISTRY)
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"unknown or not implemented: {unknown}; available: {list(REGISTRY)}", file=sys.stderr)
        return 2
    tables = [compute(k, args.results) for k in keys]
    if args.format == "json":
        print(json.dumps([t.to_dict() for t in tables], ensure_ascii=False, indent=1))
    else:
        print("\n\n".join(t.to_markdown() for t in tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
