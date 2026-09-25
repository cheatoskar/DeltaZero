"""Progress on a map WITHOUT a reference run: track cells reached (+ checkpoints).

The ghost driver can drive from the blocks alone (no reference line), but then "how far did
this run get?" still needs an answer: `improve` ranks runs by it, stops runs that are stuck,
and branches from where the best run got stuck. Plain odometer metres reward circling.

Measure: the number of distinct 32 x 8 x 32 m cells the car has been in that lie on the
track, i.e. within one cell sideways (and -2..+3 cells vertically) of a non-filler block.
It only grows by reaching new parts of the track; circling, standing and driving on the
grass away from the road add nothing. Checkpoints (CurCheckpointCount) rank above it.

Measured 2026-09-25 with scripts/course_check.py on 25 TMX maps, 40 top replays without
respawns (no game needed):
  * 97.8% of ghost samples are on track cells (lowest run 80.9%: a cut across the grass);
  * a new cell every 24.8 m of driving (median);
  * the longest time without a new cell is <= 2 s on 24/40 runs, <= 5 s on 38/40; only a
    speed map with long flights (5385842) reaches 16-19 s. Hence the stall rule in improve
    counts only time with a wheel on the ground;
  * the longest stretch off the track cells is <= 1 s on 34/40 runs, at most 3.1 s (jumps,
    landings and cuts next to the road). Hence off track = 4 s on the ground.
The cell grid is the live P_MAP block grid (world = block coord * (32, 8, 32), measured).
Not handled: a wrong route that also covers many track cells (rare on normal maps).
"""
from typing import Iterable

import numpy as np

from .ghost import BLOCK_SIZE, FILLER

CELL_M = 32.0          # progress_m = cells * CELL_M (a rough metre scale for logs)
NEAR_XZ = 1
NEAR_DY = (-2, 3)


def track_cells(blocks: Iterable[dict]) -> frozenset:
    """Plugin P_MAP block dicts -> the set of (x, y, z) cells that count as track."""
    out = set()
    for b in blocks:
        if b['name'] in FILLER:
            continue
        x, y, z = int(b['x']), int(b['y']), int(b['z'])
        for dx in range(-NEAR_XZ, NEAR_XZ + 1):
            for dz in range(-NEAR_XZ, NEAR_XZ + 1):
                for dy in range(NEAR_DY[0], NEAR_DY[1] + 1):
                    out.add((x + dx, y + dy, z + dz))
    return frozenset(out)


def cell_of(pos) -> tuple:
    c = np.floor(np.asarray(pos, dtype=np.float64) / BLOCK_SIZE).astype(np.int64)
    return int(c[0]), int(c[1]), int(c[2])
