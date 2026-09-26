"""A route through any map, found from its geometry alone: no replay, no game, no block catalogue.

The collision track TMNF-C builds from the .Challenge.Gbx (every surface the car can touch) is
rasterised into a 2.5D grid of drivable cells (upward-facing triangles, 4 m cells, several levels
per column). Neighbouring cells connect when the car can get from one to the other: level or up
a ramp, or down any drop. The route is the cheapest path from the start through every
checkpoint (the best order, as the game allows any) to a finish; asphalt is cheaper than dirt,
dirt cheaper than grass.

The result is a HINT (RL v2 can use it as its reference line on a map nobody has driven yet),
not the racing line: it knows nothing of speed, jumps, or wall rides.

    python -m tmdriver.route_planner MAP.Challenge.Gbx [--check]   (--check: compare with the
                                                                     map's fastest replay ghost)
"""
import heapq
import itertools
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import tmnfc

CELL = 4.0                  # m
LEVEL_GAP = 2.5             # m: surfaces further apart in one column are separate levels
CLIMB_PER_CELL = 3.0        # m up per 4 m (steep ramps, ~37 deg)
MAX_DROP = 60.0             # m down in one step (falling)
UP_NY = 0.5                 # |normal.y| of a drivable face
JUMP_CELLS = 10             # a jump over a gap reaches this many cells (40 m)
JUMP_RISE = 1.0             # m a jump may land above its take-off
JUMP_COST = 1.5             # jumps are riskier than driving
MATERIAL_COST = {16: 1.0, 0: 1.0, 1: 1.0, 4: 1.0, 9: 1.1, 7: 0.8, 26: 0.8, 30: 0.9, 8: 1.2, 6: 1.3,
                 2: 1.6, 3: 1.4, 5: 1.6, 14: 1.1, 12: 1.3}
SKIP_MATERIALS = {13, 28}   # water, not collidable


def _trkfile():
    p = str(tmnfc.TMNFC / 'tools' / 'build_track')
    if p not in sys.path:
        sys.path.insert(0, p)
    import trkfile
    return trkfile


def surface_points(track_path: Path):
    """World-space sample points of every collidable face: (points (n,3), up (n,) bool,
    material (n,) int)."""
    trk = _trkfile().load(str(track_path))
    pts, ups, mats = [], [], []
    cache = {}
    for e in trk.entries:
        if not e.active:
            continue
        surf = trk.surfaces[e.surface]
        mesh = trk.meshes[surf.mesh]
        key = surf.mesh
        if key not in cache:
            v = np.frombuffer(mesh.vertices, dtype='<f4').reshape(-1, 3).astype(np.float64)
            nf = len(mesh.faces) // 32
            idx = np.array([struct.unpack_from('<3IH', mesh.faces, f * 32 + 0x10) for f in range(nf)],
                           dtype=np.int64).reshape(-1, 4)
            cache[key] = (v, idx)
        v, idx = cache[key]
        if len(idx) == 0:
            continue
        iso = np.frombuffer(e.iso, dtype='<f4').astype(np.float64)
        m = iso[:9].reshape(3, 3)
        w = v @ m.T + iso[9:12]
        a, b, c = w[idx[:, 0]], w[idx[:, 1]], w[idx[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-9
        ny = np.zeros(len(n))
        ny[ok] = n[ok, 1] / ln[ok]
        mat = np.array([surf.materials[i] if i < len(surf.materials) else 0 for i in idx[:, 3]])
        keep = ok & ~np.isin(mat, list(SKIP_MATERIALS))
        # samples: vertices, centroid and a grid over big faces
        edge = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                  np.linalg.norm(a - c, axis=1)])
        for f in np.flatnonzero(keep):
            k = max(1, int(math.ceil(edge[f] / (CELL / 2))))
            if k > 64:
                k = 64
            ii, jj = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')
            sel = ii + jj <= k
            u, vv = ii[sel] / k, jj[sel] / k
            p = a[f] + np.outer(u, b[f] - a[f]) + np.outer(vv, c[f] - a[f])
            pts.append(p)
            ups.append(np.full(len(p), abs(ny[f]) >= UP_NY))
            mats.append(np.full(len(p), mat[f]))
    return np.concatenate(pts), np.concatenate(ups), np.concatenate(mats)


class Grid:
    def __init__(self, pts, ups, mats):
        P, M = pts[ups], mats[ups]
        ix = np.floor(P[:, 0] / CELL).astype(np.int64)
        iz = np.floor(P[:, 2] / CELL).astype(np.int64)
        order = np.lexsort((P[:, 1], iz, ix))
        ix, iz, y, m = ix[order], iz[order], P[order, 1], M[order]
        self.nodes: List[Tuple[int, int, float]] = []   # (ix, iz, height)
        self.cost: List[float] = []
        self.col: Dict[Tuple[int, int], List[int]] = {}
        start = 0
        n = len(y)
        while start < n:
            end = start
            while end + 1 < n and ix[end + 1] == ix[start] and iz[end + 1] == iz[start]:
                end += 1
            ys, ms = y[start:end + 1], m[start:end + 1]
            lo = 0
            for k in range(1, len(ys) + 1):
                if k == len(ys) or ys[k] - ys[k - 1] > LEVEL_GAP:
                    top = float(ys[k - 1])
                    vals, cnt = np.unique(ms[lo:k], return_counts=True)
                    c = MATERIAL_COST.get(int(vals[np.argmax(cnt)]), 1.2)
                    self.col.setdefault((int(ix[start]), int(iz[start])), []).append(len(self.nodes))
                    self.nodes.append((int(ix[start]), int(iz[start]), top))
                    self.cost.append(c)
                    lo = k
            start = end + 1

    def reach(self, i, cx, cz, rise):
        """The highest surface in column (cx, cz) the car at node i can get onto."""
        h = self.nodes[i][2]
        best = None
        for j in self.col.get((cx, cz), ()):
            hj = self.nodes[j][2]
            if h - MAX_DROP <= hj <= h + rise and (best is None or hj > self.nodes[best][2]):
                best = j
        return best

    def neighbours(self, i):
        x, z, h = self.nodes[i]
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            d = CELL * (1.4142 if dx and dz else 1.0)
            best = self.reach(i, x + dx, z + dz, CLIMB_PER_CELL * d / CELL)
            if best is not None:
                yield best, d * 0.5 * (self.cost[i] + self.cost[best])
                continue
            # a gap: a jump lands on the first surface ahead no higher than the take-off
            for k in range(2, JUMP_CELLS + 1):
                j = self.reach(i, x + dx * k, z + dz * k, JUMP_RISE)
                if j is not None:
                    yield j, d * k * JUMP_COST
                    break

    def nearest(self, p, below: bool = True) -> Optional[int]:
        x, z = int(math.floor(p[0] / CELL)), int(math.floor(p[2] / CELL))
        best, bd = None, 1e9
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for j in self.col.get((x + dx, z + dz), []):
                    hj = self.nodes[j][2]
                    if below and hj > p[1] + 1.0:
                        continue
                    d = abs(dx) + abs(dz) + abs(hj - p[1]) / 8
                    if d < bd:
                        best, bd = j, d
        return best

    def block_nodes(self, bx, by, bz) -> List[int]:
        out = []
        for cx in range(int(bx * 32 // CELL), int((bx * 32 + 32) // CELL)):
            for cz in range(int(bz * 32 // CELL), int((bz * 32 + 32) // CELL)):
                for j in self.col.get((cx, cz), []):
                    if by * 8 - 4 <= self.nodes[j][2] <= by * 8 + 24:
                        out.append(j)
        return out

    def dijkstra(self, sources: List[int]):
        dist = {s: 0.0 for s in sources}
        prev = {s: -1 for s in sources}
        heap = [(0.0, s) for s in sources]
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist.get(i, 1e18):
                continue
            for j, w in self.neighbours(i):
                nd = d + w
                if nd < dist.get(j, 1e18):
                    dist[j], prev[j] = nd, i
                    heapq.heappush(heap, (nd, j))
        return dist, prev


def plan(challenge: Path, track_id: str = 'plan', log=print) -> Optional[np.ndarray]:
    """The route as (n, 3) points (start ... finish), or None when no path was found."""
    from .replaybuild import map_blocks
    from .virtual_game import tmi_waypoint, WP_CP, WP_FINISH, WP_STARTFINISH
    prep = tmnfc.prepare_map(Path(challenge), str(track_id))
    pts, ups, mats = surface_points(prep['track'])
    g = Grid(pts, ups, mats)
    _, names, xyz, _ = map_blocks(Path(challenge))
    cps = sorted({tuple(int(v) for v in p) for n, p in zip(names, xyz) if tmi_waypoint(n) == WP_CP})
    fins = [tuple(int(v) for v in p) for n, p in zip(names, xyz) if tmi_waypoint(n) in (WP_FINISH, WP_STARTFINISH)]
    spawn = prep['spawn'][9:]
    s = g.nearest(spawn)
    if s is None or not fins:
        return None
    cp_nodes = [g.block_nodes(*c) for c in cps]
    fin_nodes = sorted({j for f in fins for j in g.block_nodes(*f)})
    if not fin_nodes or any(not c for c in cp_nodes):
        return None
    groups = [[s]] + cp_nodes                       # sources: start and every checkpoint
    tables = [g.dijkstra(src) for src in groups]

    def cost(i, target_nodes):
        dist = tables[i][0]
        return min((dist.get(j, 1e18), j) for j in target_nodes)

    k = len(cps)
    if k <= 9:
        best = None
        for perm in itertools.permutations(range(k)):
            total, cur = 0.0, 0
            for c in perm:
                total += cost(cur, cp_nodes[c])[0]
                cur = c + 1
            total += cost(cur, fin_nodes)[0]
            if best is None or total < best[0]:
                best = (total, perm)
        order = best[1]
    else:                                            # greedy nearest checkpoint
        order, cur, left = [], 0, set(range(k))
        while left:
            c = min(left, key=lambda c: cost(cur, cp_nodes[c])[0])
            order.append(c)
            left.remove(c)
            cur = c + 1
    path, cur = [], 0
    legs = [cp_nodes[c] for c in order] + [fin_nodes]
    for n_leg, target in enumerate(legs):
        d, end = cost(cur, target)
        if d >= 1e18:
            log(f'route: no path to leg {n_leg}')
            return None
        prev = tables[cur][1]
        seg, j = [], end
        while j != -1:
            seg.append(j)
            j = prev[j]
        path += seg[::-1] if not path else seg[::-1][1:]
        cur = order[n_leg] + 1 if n_leg < len(order) else cur
    route = np.array([((g.nodes[j][0] + 0.5) * CELL, g.nodes[j][2], (g.nodes[j][1] + 0.5) * CELL) for j in path])
    return route


def compare_to_ghost(route: np.ndarray, ghost: np.ndarray) -> dict:
    """Horizontal distance from each ghost sample to the route polyline."""
    a, b = route[:-1, [0, 2]], route[1:, [0, 2]]
    ab = b - a
    ll = np.maximum((ab ** 2).sum(1), 1e-9)
    out = []
    for p in ghost[:, [0, 2]]:
        t = np.clip(((p - a) * ab).sum(1) / ll, 0, 1)
        out.append(np.sqrt(((a + ab * t[:, None] - p) ** 2).sum(1)).min())
    d = np.array(out)
    return {'mean_m': float(d.mean()), 'p90_m': float(np.percentile(d, 90)), 'within_16m': float((d <= 16).mean()),
            'route_m': float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
            'ghost_m': float(np.linalg.norm(np.diff(ghost, axis=0), axis=1).sum())}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('map')
    ap.add_argument('--replay', default='', help='a replay of the map to compare with')
    a = ap.parse_args()
    import time
    t0 = time.time()
    r = plan(Path(a.map))
    if r is None:
        print('no route')
        return
    print(f'route: {len(r)} points, {np.linalg.norm(np.diff(r, axis=0), axis=1).sum():.0f} m, {time.time() - t0:.1f}s')
    if a.replay:
        from . import replay as R
        rep = R.load(a.replay)
        print(compare_to_ghost(r, rep.ghost_pos[rep.ghost_t <= rep.race_time_ms]))


if __name__ == '__main__':
    main()
