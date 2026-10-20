"""Every table, recomputed from the result files in `results/`, against the recorded one.

`golden/tables.json` holds the rows of every table. A cell matches if it carries the same numbers with the same number of decimals
(words, signs of rounding and TeX are ignored), so "79 / 72", "4 of 80" and "3.14 vs. 6.82" are all compared as their numbers. A change
of the code or of the data that moves one number fails a test.
"""

import json
import re
from pathlib import Path

import pytest

from objeval3d.analysis.tables import REGISTRY, compute

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / "tests" / "golden" / "tables.json").read_text(encoding="utf-8"))["tables"]
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    return [n.replace("−", "-").lstrip("+") for n in NUMBER.findall(text)]


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_table_matches_the_recorded_one(key):
    expected = GOLDEN[key]["rows"]
    table = compute(key, ROOT / "results")
    assert len(table.rows) == len(expected), f"{key}: {len(table.rows)} rows computed, {len(expected)} recorded"
    bad = []
    for i, (got, want) in enumerate(zip(table.rows, expected)):
        g, w = numbers(" ".join(got)), numbers(" ".join(want))
        if g != w:
            bad.append(f"row {i} ({want[0][:40]!r}): computed {g}, recorded {w}")
    assert not bad, f"{key}\n" + "\n".join(bad)


def test_every_table_is_recorded():
    assert set(GOLDEN) == set(REGISTRY)
