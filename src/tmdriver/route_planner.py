"""A route through any map, found from its geometry alone: no replay, no game, no block catalogue.

The collision track TMNF-C builds from the .Challenge.Gbx (every surface the car can touch) is
rasterised into a 2.5D grid of drivable cells (upward-facing triangles, 4 m cells, several levels
per column). Neighbouring cells connect when the car can get from one to the other: level or up
a ramp, or down any drop, unless a wall (a vertical face) stands in between. Across a gap, a jump
lands on the first surface ahead that is not higher than the take-off. The route is the
cheapest path from the start through every checkpoint (the best order: the game allows any) to
a finish; asphalt is cheaper than dirt, dirt cheaper than grass.

The result is a HINT (RL v2 can use it as its reference line on a map nobody has driven yet),
not the racing line: it knows nothing of speed, and jumps and wall rides are guesses.

    python -m tmdriver.route_planner MAP.Challenge.Gbx [--replay R.Replay.Gbx]
"""
import itertools
import math
import struct
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import tmnfc

CELL = 4.0                  # m
LEVEL_GAP = 2.5             # m: surfaces further apart in one column are separate levels
CLIMB_PER_CELL = 3.0        # m up per 4 m (steep ramps, ~37 deg)
MAX_DROP = 60.0             # m down in one step (falling)
UP_NY = 0.5                 # |normal.y| of a drivable face
WALL_NY = 0.3               # |normal.y| below this: a wall
WALL_CELL = 1.0             # m: walls are located on a finer grid
WALL_LOW, WALL_HIGH = 0.5, 2.5   # a wall blocks when it spans this band above the higher cell
JUMP_CELLS = 10             # a jump over a gap reaches this many cells (40 m)
JUMP_RISE = 0.0             # m a jump may land above its take-off
JUMP_COST = 2.0             # jumps are riskier than driving
MATERIAL_COST = {16: 1.0, 0: 1.0, 1: 1.0, 4: 1.0, 9: 1.1, 7: 0.8, 26: 0.8, 30: 0.9, 8: 1.2, 6: 1.3,
                 2: 1.6, 3: 1.4, 5: 1.6, 14: 1.1, 12: 1.3}
SKIP_MATERIALS = (13, 28)   # water, not collidable
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _trkfile():
    p = str(tmnfc.TMNFC / 'tools' / 'build_track')
    if p not in sys.path:
        sys.path.insert(0, p)
    import trkfile
    return trkfile


def _samples(a, b, c, step):
    """Points on triangles (a, b, c: (n, 3)) about `step` apart, vectorised per subdivision."""
    edge = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                              np.linalg.norm(a - c, axis=1)])
    k = np.clip(np.ceil(edge / step).astype(np.int64), 1, 64)
    out, owner = [], []
    for kk in np.unique(k):
        sel = np.flatnonzero(k == kk)
        ii, jj = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing='ij')
        m = ii + jj <= kk
        u, v = ii[m] / kk, jj[m] / kk                  # (s,)
        p = a[sel, None, :] + u[None, :, None] * (b - a)[sel, None, :] + v[None, :, None] * (c - a)[sel, None, :]
        out.append(p.reshape(-1, 3))
        owner.append(np.repeat(sel, len(u)))
    return np.concatenate(out), np.concatenate(owner)


def surface_points(track_path: Path):
    """World-space samples: floors (points, material) and walls (points)."""
    trk = _trkfile().load(str(track_path))
    cache = {}
    tri_a, tri_b, tri_c, tri_ny, tri_m = [], [], [], [], []
    for e in trk.entries:
        if not e.active:
            continue
        surf = trk.surfaces[e.surface]
        if surf.mesh not in cache:
            mesh = trk.meshes[surf.mesh]
            v = np.frombuffer(mesh.vertices, dtype='<f4').reshape(-1, 3).astype(np.float64)
            nf = len(mesh.faces) // 32
            f = np.frombuffer(mesh.faces, dtype=np.uint8).reshape(nf, 32)
            idx = f[:, 0x10:0x1C].copy().view('<u4').astype(np.int64)
            mi = f[:, 0x1C:0x1E].copy().view('<u2').astype(np.int64).ravel()
            cache[surf.mesh] = (v, idx, mi)
        v, idx, mi = cache[surf.mesh]
        if len(idx) == 0:
            continue
        iso = np.frombuffer(e.iso, dtype='<f4').astype(np.float64)
        w = v @ iso[:9].reshape(3, 3).T + iso[9:12]
        a, b, c = w[idx[:, 0]], w[idx[:, 1]], w[idx[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        mats = np.frombuffer(surf.materials, dtype=np.uint8)
        mat = np.where(mi < len(mats), mats[np.minimum(mi, len(mats) - 1)], 0)
        ok = (ln > 1e-9) & ~np.isin(mat, SKIP_MATERIALS)
        tri_a.append(a[ok])
        tri_b.append(b[ok])
        tri_c.append(c[ok])
        tri_ny.append(np.abs(n[ok, 1] / ln[ok]))
        tri_m.append(mat[ok])
    a, b, c = np.concatenate(tri_a), np.concatenate(tri_b), np.concatenate(tri_c)
    ny, mat = np.concatenate(tri_ny), np.concatenate(tri_m)
    floor, wall = ny >= UP_NY, ny < WALL_NY
    fp, fo = _samples(a[floor], b[floor], c[floor], CELL / 2)
    wp, _ = _samples(a[wall], b[wall], c[wall], WALL_CELL / 2)
    return fp, mat[floor][fo], wp


class Grid:
    def __init__(self, floor_pts, floor_mat, wall_pts):
        ix = np.floor(floor_pts[:, 0] / CELL).astype(np.int64)
        iz = np.floor(floor_pts[:, 2] / CELL).astype(np.int64)
        y = floor_pts[:, 1]
        order = np.lexsort((y, iz, ix))
        ix, iz, y, m = ix[order], iz[order], y[order], floor_mat[order]
        # a new node where the column changes or the height jumps by more than LEVEL_GAP
        new = np.ones(len(y), bool)
        new[1:] = (ix[1:] != ix[:-1]) | (iz[1:] != iz[:-1]) | (np.diff(y) > LEVEL_GAP)
        node_of = np.cumsum(new) - 1
        nn = int(node_of[-1]) + 1
        top = np.full(nn, -1e9)
        np.maximum.at(top, node_of, y)
        cost = np.array([MATERIAL_COST.get(int(v), 1.2) for v in range(64)])
        cm = np.zeros(nn)
        np.add.at(cm, node_of, cost[np.clip(m, 0, 63)])
        cnt = np.bincount(node_of, minlength=nn)
        self.x = ix[new]
        self.z = iz[new]
        self.h = top
        self.cost = cm / np.maximum(cnt, 1)
        self.col = {}
        for k in range(nn):
            self.col.setdefault((int(self.x[k]), int(self.z[k])), []).append(k)
        # walls: per 1 m column the sampled wall heights
        wx = np.floor(wall_pts[:, 0] / WALL_CELL).astype(np.int64)
        wz = np.floor(wall_pts[:, 2] / WALL_CELL).astype(np.int64)
        self.walls = {}
        for (a, b), yv in zip(zip(wx.tolist(), wz.tolist()), wall_pts[:, 1].tolist()):
            self.walls.setdefault((a, b), []).append(yv)
        self.walls = {k: np.array(sorted(v)) for k, v in self.walls.items()}

    def blocked(self, x0, z0, x1, z1, h):
        """A wall across the straight line between two cell centres, spanning the band above h."""
        ax, az = (x0 + 0.5) * CELL, (z0 + 0.5) * CELL
        bx, bz = (x1 + 0.5) * CELL, (z1 + 0.5) * CELL
        n = int(math.ceil(math.hypot(bx - ax, bz - az) / WALL_CELL))
        seen = set()
        for k in range(1, n):
            t = k / n
            c = (int(math.floor((ax + (bx - ax) * t) / WALL_CELL)), int(math.floor((az + (bz - az) * t) / WALL_CELL)))
            if c in seen:
                continue
            seen.add(c)
            w = self.walls.get(c)
            if w is not None:
                lo, hi = np.searchsorted(w, [h + WALL_LOW, h + WALL_HIGH])
                if hi > lo:
                    return True
        return False

    def reach(self, i, cx, cz, rise):
        h = self.h[i]
        best = None
        for j in self.col.get((cx, cz), ()):
            hj = self.h[j]
            if h - MAX_DROP <= hj <= h + rise and (best is None or hj > self.h[best]):
                best = j
        return best

    def edges(self):
        src, dst, w = [], [], []
        for i in range(len(self.h)):
            x, z, h = int(self.x[i]), int(self.z[i]), self.h[i]
            for dx, dz in DIRS:
                d = CELL * (1.4142 if dx and dz else 1.0)
                j = self.reach(i, x + dx, z + dz, CLIMB_PER_CELL * d / CELL)
                if j is not None:
                    top = max(h, self.h[j])
                    if not self.blocked(x, z, x + dx, z + dz, top):
                        src.append(i)
                        dst.append(j)
                        w.append(d * 0.5 * (self.cost[i] + self.cost[j]))
                    continue
                for k in range(2, JUMP_CELLS + 1):          # a gap: jump
                    j = self.reach(i, x + dx * k, z + dz * k, JUMP_RISE)
                    if j is not None:
                        src.append(i)
                        dst.append(j)
                        w.append(d * k * JUMP_COST)
                        break
        from scipy.sparse import csr_matrix
        n = len(self.h)
        return csr_matrix((np.array(w), (np.array(src), np.array(dst))), shape=(n, n))

    def nearest(self, p) -> Optional[int]:
        x, z = int(math.floor(p[0] / CELL)), int(math.floor(p[2] / CELL))
        best, bd = None, 1e9
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for j in self.col.get((x + dx, z + dz), []):
                    if self.h[j] > p[1] + 1.0:
                        continue
                    d = abs(dx) + abs(dz) + abs(self.h[j] - p[1]) / 8
                    if d < bd:
                        best, bd = j, d
        return best

    def block_nodes(self, bx, by, bz) -> List[int]:
        out = []
        for cx in range(int(bx * 32 // CELL), int((bx * 32 + 32) // CELL)):
            for cz in range(int(bz * 32 // CELL), int((bz * 32 + 32) // CELL)):
                out += [j for j in self.col.get((cx, cz), []) if by * 8 - 4 <= self.h[j] <= by * 8 + 24]
        return out


def plan(challenge: Path, track_id: str = 'plan', log=print) -> Optional[np.ndarray]:
    """The route as (n, 3) points (start ... finish), or None when no path was found."""
    from scipy.sparse.csgraph import dijkstra
    from .replaybuild import map_blocks
    from .virtual_game import tmi_waypoint, WP_CP, WP_FINISH, WP_STARTFINISH
    prep = tmnfc.prepare_map(Path(challenge), str(track_id))
    g = Grid(*surface_points(prep['track']))
    graph = g.edges()
    _, names, xyz, _ = map_blocks(Path(challenge))
    cps = sorted({tuple(int(v) for v in p) for n, p in zip(names, xyz) if tmi_waypoint(n) == WP_CP})
    fins = [tuple(int(v) for v in p) for n, p in zip(names, xyz) if tmi_waypoint(n) in (WP_FINISH, WP_STARTFINISH)]
    s = g.nearest(prep['spawn'][9:])
    if s is None or not fins:
        return None
    cp_nodes = [g.block_nodes(*c) for c in cps]
    fin_nodes = sorted({j for f in fins for j in g.block_nodes(*f)})
    if not fin_nodes or any(not c for c in cp_nodes):
        return None
    groups = [[s]] + cp_nodes
    tables = [dijkstra(graph, indices=src, min_only=True, return_predecessors=True) for src in groups]

    def cost(i, targets):
        d = tables[i][0][targets]
        k = int(np.argmin(d))
        return float(d[k]), targets[k]

    k = len(cps)
    if k <= 8:
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
    else:
        order, cur, left = [], 0, set(range(k))
        while left:
            c = min(left, key=lambda c: cost(cur, cp_nodes[c])[0])
            order.append(c)
            left.remove(c)
            cur = c + 1
    path, cur = [], 0
    for n_leg, target in enumerate([cp_nodes[c] for c in order] + [fin_nodes]):
        d, end = cost(cur, target)
        if not np.isfinite(d):
            log(f'route: no path to leg {n_leg}')
            return None
        pred = tables[cur][1]
        seg, j = [], end
        while j >= 0:
            seg.append(j)
            j = pred[j]
        path += seg[::-1] if not path else seg[::-1][1:]
        cur = order[n_leg] + 1 if n_leg < len(order) else cur
    return np.array([((g.x[j] + 0.5) * CELL, g.h[j], (g.z[j] + 0.5) * CELL) for j in path])


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
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('map')
    ap.add_argument('--replay', default='', help='a replay of the map to compare with')
    a = ap.parse_args()
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
