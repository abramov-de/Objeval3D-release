"""How many threads a nearest-neighbour query may use.

scipy's `workers=-1` means `os.cpu_count()`, whichever CPUs the process is allowed to run on. On a shared machine, under `taskset`, or in a
container with a CPU quota that starts a pool of dozens of threads on a few cores for every single query, and the scale sweep of the
reference mode makes thousands of small ones: the pools cost a hundred times more than the queries (one view set took 144 s, 99 % of it
in starting threads, on four cores). `workers()` follows the CPU affinity of the process instead; on a machine that is not restricted it is
what `workers=-1` gave.
"""

from __future__ import annotations

import os


def workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not on Linux
        return max(1, os.cpu_count() or 1)
