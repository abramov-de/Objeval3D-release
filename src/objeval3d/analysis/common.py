"""Readers and statistics shared by the analysis modules.

Statistic convention of the paper (Section 3.5): inside one view set a distance is reduced by the median over the points
of the set, and across view sets the per-set numbers are averaged (Tables 2-4) or their median is taken (Tables 5-9);
each function below says which one it implements.
"""

from __future__ import annotations

import gzip
import json
import os
import statistics as st
from pathlib import Path
from typing import Any, Callable, Iterable

from objeval3d.modes import FREE_FIT, REFERENCE

RESULTS_ENV = "OBJEVAL_RESULTS"


def results_dir(path: str | Path | None = None) -> Path:
    """The directory with the result files: the argument, else $OBJEVAL_RESULTS, else ./results."""
    return Path(path or os.environ.get(RESULTS_ENV) or "results")


def read_json(path: str | Path) -> Any:
    """Read `path`, or `path.gz` / `path` with `.json.gz` when only the compressed file is there."""
    p = Path(path)
    for cand in (p, p.with_name(p.name + ".gz"), p.with_suffix(".json.gz")):
        if cand.exists():
            opener = gzip.open if cand.suffix == ".gz" else open
            with opener(cand, "rt", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"{p} (also tried the .gz variants)")


def load_sets(results: Path, name: str, *, modes: Iterable[str] = (REFERENCE,)) -> dict[str, dict]:
    """The view sets of result file `name` (stem, without extension) that were scored: status `ok` and all `modes` present."""
    modes = tuple(modes)
    data = read_json(results / f"{name}.json")
    return {
        r["set_id"]: r
        for r in data["results"]
        if r.get("status") == "ok" and all(m in r.get("alignments", {}) for m in modes)
    }


def load_all_sets(results: Path, name: str) -> dict[str, dict]:
    """Every record of result file `name`, scored or not (status `too_few_points` records carry the reason)."""
    return {r["set_id"]: r for r in read_json(results / f"{name}.json")["results"]}


# --- per-set quantities -------------------------------------------------------------------------------------------------


def accuracy(r: dict, mode: str = REFERENCE) -> float:
    """Median distance from the cloud to the scan, over the points of one view set (mm)."""
    return r["alignments"][mode]["accuracy"]["median"]


def completeness(r: dict, mode: str = REFERENCE) -> float:
    """Median distance from the visible part of the scan to the cloud, over the points of one view set (mm)."""
    return r["alignments"][mode]["completeness"]["median"]


def fscore(r: dict, mode: str = REFERENCE, tau: str = "tau_2pct") -> float:
    return r["alignments"][mode]["fscore"][tau]["fscore"]


def precision_at(r: dict, mode: str = REFERENCE, tau: str = "tau_2pct") -> float:
    return r["alignments"][mode]["fscore"][tau]["precision"]


def recall_at(r: dict, mode: str = REFERENCE, tau: str = "tau_2pct") -> float:
    return r["alignments"][mode]["fscore"][tau]["recall"]


def chamfer(r: dict, mode: str = REFERENCE) -> float:
    return r["alignments"][mode]["chamfer_l1_mm"]


def object_key(set_id: str) -> str:
    """`lmo_s000002_o05_v03` -> `o05`: the object of a view set."""
    return set_id.split("_")[2]


# --- reductions across sets ---------------------------------------------------------------------------------------------


def mean_over(d: dict[str, dict], ids: Iterable[str], f: Callable[[dict], float]) -> float:
    return st.mean(f(d[s]) for s in sorted(ids))


def median_over(d: dict[str, dict], ids: Iterable[str], f: Callable[[dict], float]) -> float:
    return st.median(f(d[s]) for s in sorted(ids))


__all__ = [
    "FREE_FIT",
    "REFERENCE",
    "results_dir",
    "read_json",
    "load_sets",
    "load_all_sets",
    "accuracy",
    "completeness",
    "fscore",
    "precision_at",
    "recall_at",
    "chamfer",
    "object_key",
    "mean_over",
    "median_over",
]


def read_jsonl(path: str | Path) -> list[dict]:
    """Records of a JSON-lines file, plain or gzip-compressed (`.jsonl` or `.jsonl.gz`)."""
    p = Path(path)
    for cand in (p, p.with_name(p.name + ".gz")):
        if cand.exists():
            opener = gzip.open if cand.suffix == ".gz" else open
            with opener(cand, "rt", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"{p} (also tried the .gz variant)")
