"""A route through any map from its geometry, with costs LEARNED from record lines.

1. Geometry: the collision track TMNF-C builds from the .Challenge.Gbx is rasterised into a 2.5D
   grid of drivable cells (upward-facing triangles, 4 m cells, several levels per column), inside
   the map's block volume only (the stadium decoration is left out).
2. Moves: to a neighbouring cell (level, up a ramp, or down a drop no deeper than max_drop)
   unless a wall stands on the line between the cells; across a gap, a jump to the first surface
   ahead no higher than the take-off.
3. Costs: every node has features (surface material, the kind of block it lies in, height above
   the ground, local slope, levels in its column). A logistic model trained on the record lines
   of thousands of re-simulated maps (fit_cost_model) gives the probability that a fast driver is
   there; a node costs its length x (1 + w_learned x -log p). Maps differ, so the costs come from
   what drivers actually do on maps like it rather than from fixed rules.
4. The route is the cheapest path from the start through every checkpoint (best order) to a
   finish (scipy Dijkstra).

The result is a HINT for RL on maps nobody has driven (improve.planned_route), not the racing
line. Measure it with `python -m tmdriver.route_planner eval`.
"""
import itertools
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import tmnfc
from .paths import DATA

CELL = 4.0                  # m
LEVEL_GAP = 2.5             # m: surfaces further apart in one column are separate levels
UP_NY = 0.5                 # |normal.y| of a drivable face
WALL_NY = 0.3               # |normal.y| below this: a wall
WALL_CELL = 1.0             # m: walls are located on a finer grid
WALL_LOW, WALL_HIGH = 0.5, 2.5   # a wall blocks when it spans this band above the higher cell
MAP_SIZE = 32 * 32.0        # m (Stadium: 32 x 32 blocks)
SKIP_MATERIALS = (13, 28)   # water, not collidable
DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
MODEL = DATA / 'route_cost_model.json'

DEFAULT = {'climb_per_cell': 3.0, 'max_drop': 16.0, 'jump_cells': 10, 'jump_rise': 0.0,
           'jump_cost': 2.0, 'drop_cost': 0.3, 'w_learned': 1.0, 'bridge_cost': 20.0}
MATERIAL_BASE = {16: 1.0, 0: 1.0, 1: 1.0, 4: 1.0, 9: 1.1, 7: 0.8, 26: 0.8, 30: 0.9, 8: 1.2, 6: 1.3,
                 2: 1.6, 3: 1.4, 5: 1.6, 14: 1.1, 12: 1.3}


def _trkfile():
    p = str(tmnfc.TMNFC / 'tools' / 'build_track')
    if p not in sys.path:
        sys.path.insert(0, p)
    import trkfile
    return trkfile


def _samples(a, b, c, step):
    """Points on triangles (a, b, c: (n, 3)) about `step` apart."""
    if len(a) == 0:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    edge = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                              np.linalg.norm(a - c, axis=1)])
    k = np.clip(np.ceil(edge / step).astype(np.int64), 1, 64)
    out, owner = [], []
    for kk in np.unique(k):
        sel = np.flatnonzero(k == kk)
        ii, jj = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing='ij')
        m = ii + jj <= kk
        u, v = ii[m] / kk, jj[m] / kk
        p = a[sel, None, :] + u[None, :, None] * (b - a)[sel, None, :] + v[None, :, None] * (c - a)[sel, None, :]
        out.append(p.reshape(-1, 3))
        owner.append(np.repeat(sel, len(u)))
    return np.concatenate(out), np.concatenate(owner)


def surface_points(track_path: Path, y_max: float):
    """Floor samples (points, material) and wall samples inside the map volume."""
    trk = _trkfile().load(str(track_path))
    cache = {}
    tri = {k: [] for k in ('a', 'b', 'c', 'ny', 'm')}
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
        cen = (a + b + c) / 3
        inside = ((cen[:, 0] >= -8) & (cen[:, 0] <= MAP_SIZE + 8) & (cen[:, 2] >= -8) & (cen[:, 2] <= MAP_SIZE + 8)
                  & (cen[:, 1] <= y_max))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        mats = np.frombuffer(surf.materials, dtype=np.uint8)
        mat = np.where(mi < len(mats), mats[np.minimum(mi, len(mats) - 1)], 0)
        ok = inside & (ln > 1e-9) & ~np.isin(mat, SKIP_MATERIALS)
        tri['a'].append(a[ok])
        tri['b'].append(b[ok])
        tri['c'].append(c[ok])
        tri['ny'].append(np.abs(n[ok, 1] / ln[ok]))
        tri['m'].append(mat[ok])
    a, b, c = (np.concatenate(tri[k]) for k in 'abc')
    ny, mat = np.concatenate(tri['ny']), np.concatenate(tri['m'])
    floor, wall = ny >= UP_NY, ny < WALL_NY
    fp, fo = _samples(a[floor], b[floor], c[floor], CELL / 2)
    wp, _ = _samples(a[wall], b[wall], c[wall], WALL_CELL / 2)
    return fp, mat[floor][fo], ny[floor][fo], wp


def block_index(names, xyz):
    """(block column key -> list of (y, name)) for the node features."""
    out: Dict[tuple, list] = {}
    for n, p in zip(names, xyz):
        out.setdefault((int(p[0]), int(p[2])), []).append((int(p[1]), n))
    return out


def block_kind(name: str) -> str:
    """A coarse, map-independent kind of a block name (the feature the model sees)."""
    n = name.replace('Stadium', '')
    for k in ('Checkpoint', 'Finish', 'Start', 'Turbo', 'Loop', 'Wall', 'Tube', 'Circuit', 'Platform',
              'Dirt', 'Grass', 'Water', 'Pool', 'Road', 'Inflatable', 'Sculpt', 'Fabric', 'Control',
              'Pillar', 'Hole', 'Ramp', 'Slope', 'Tilt', 'Bump'):
        if k in n:
            return k
    return 'Other'


KINDS = ['Checkpoint', 'Finish', 'Start', 'Turbo', 'Loop', 'Wall', 'Tube', 'Circuit', 'Platform', 'Dirt',
         'Grass', 'Water', 'Pool', 'Road', 'Inflatable', 'Sculpt', 'Fabric', 'Control', 'Pillar', 'Hole',
         'Ramp', 'Slope', 'Tilt', 'Bump', 'Other', 'None']
MATS = [0, 1, 2, 4, 6, 7, 8, 9, 14, 16, 26, 30]


class Grid:
    def __init__(self, floor_pts, floor_mat, floor_ny, wall_pts, blocks=None):
        ix = np.floor(floor_pts[:, 0] / CELL).astype(np.int64)
        iz = np.floor(floor_pts[:, 2] / CELL).astype(np.int64)
        y = floor_pts[:, 1]
        order = np.lexsort((y, iz, ix))
        ix, iz, y, m, fny = ix[order], iz[order], y[order], floor_mat[order], floor_ny[order]
        new = np.ones(len(y), bool)
        new[1:] = (ix[1:] != ix[:-1]) | (iz[1:] != iz[:-1]) | (np.diff(y) > LEVEL_GAP)
        node_of = np.cumsum(new) - 1
        nn = int(node_of[-1]) + 1
        self.n = nn
        self.h = np.full(nn, -1e9)
        np.maximum.at(self.h, node_of, y)
        self.x, self.z = ix[new], iz[new]
        # majority material, mean flatness of the node
        counts = np.stack([np.bincount(node_of[m == mv], minlength=nn) for mv in MATS], 1)
        self.mat = np.array(MATS)[counts.argmax(1)]
        sny = np.zeros(nn)
        np.add.at(sny, node_of, fny)
        self.flat = sny / np.maximum(np.bincount(node_of, minlength=nn), 1)
        # columns: nodes sorted by (column, height) -> searchsorted per column
        self.col_key = self.x * 4096 + self.z
        self.key = self.col_key.astype(np.float64) * 10000.0 + (self.h + 1000.0)
        assert np.all(np.diff(self.key) >= 0)
        uc, first, cnt = np.unique(self.col_key, return_index=True, return_counts=True)
        self.levels = np.repeat(cnt, cnt)
        ground = np.full(len(uc), np.inf)
        np.minimum.at(ground, np.searchsorted(uc, self.col_key), self.h)
        self.above_ground = self.h - ground[np.searchsorted(uc, self.col_key)]
        # walls: sorted (1 m column, height)
        wx = np.floor(wall_pts[:, 0] / WALL_CELL).astype(np.int64)
        wz = np.floor(wall_pts[:, 2] / WALL_CELL).astype(np.int64)
        self.wall_key = np.sort((wx * 8192 + wz).astype(np.float64) * 10000.0 + (wall_pts[:, 1] + 1000.0))
        # block kind at each node
        self.kind = np.full(nn, KINDS.index('None'))
        if blocks is not None:
            bx, bz, by = self.x * CELL // 32, self.z * CELL // 32, np.floor(self.h / 8)
            for i in range(nn):
                lst = blocks.get((int(bx[i]), int(bz[i])))
                if lst:
                    yb, name = min(lst, key=lambda t: abs(t[0] - by[i]))
                    if abs(yb - by[i]) <= 3:
                        self.kind[i] = KINDS.index(block_kind(name))

    def features(self) -> np.ndarray:
        """Per node: one-hot material, one-hot block kind, height above ground, flatness, levels."""
        f_mat = (self.mat[:, None] == np.array(MATS)[None, :]).astype(np.float64)
        f_kind = (self.kind[:, None] == np.arange(len(KINDS))[None, :]).astype(np.float64)
        ag = np.clip(self.above_ground, 0, 200)[:, None]
        return np.concatenate([f_mat, f_kind, np.log1p(ag), (ag > 0.5).astype(float), self.flat[:, None],
                               np.log(self.levels)[:, None], np.ones((self.n, 1))], 1)

    def _find(self, col, lo, hi):
        """The highest node in column `col` with height in [lo, hi], or -1 (vectorised)."""
        k_hi = col.astype(np.float64) * 10000.0 + (hi + 1000.0)
        k_lo = col.astype(np.float64) * 10000.0 + (lo + 1000.0)
        j = np.searchsorted(self.key, k_hi, side='right') - 1
        ok = (j >= 0) & (self.key[np.clip(j, 0, None)] >= k_lo)
        return np.where(ok, j, -1)

    def _blocked(self, i, j, top):
        """Walls on the line between node centres i and j spanning the band above `top`."""
        ax, az = (self.x[i] + 0.5) * CELL, (self.z[i] + 0.5) * CELL
        bx, bz = (self.x[j] + 0.5) * CELL, (self.z[j] + 0.5) * CELL
        blocked = np.zeros(len(i), bool)
        steps = 6
        for s in range(1, steps):
            t = s / steps
            wx = np.floor((ax + (bx - ax) * t) / WALL_CELL).astype(np.int64)
            wz = np.floor((az + (bz - az) * t) / WALL_CELL).astype(np.int64)
            ck = (wx * 8192 + wz).astype(np.float64) * 10000.0
            lo = np.searchsorted(self.wall_key, ck + top + WALL_LOW + 1000.0)
            hi = np.searchsorted(self.wall_key, ck + top + WALL_HIGH + 1000.0)
            blocked |= hi > lo
        return blocked

    def edges(self, p: dict, node_cost: np.ndarray):
        from scipy.sparse import csr_matrix
        idx = np.arange(self.n)
        src, dst, w = [], [], []
        for dx, dz in DIRS:
            d = CELL * (math.sqrt(2) if dx and dz else 1.0)
            col = (self.x + dx) * 4096 + (self.z + dz)
            j = self._find(col, self.h - p['max_drop'], self.h + p['climb_per_cell'] * d / CELL)
            has = j >= 0
            i1, j1 = idx[has], j[has]
            top = np.maximum(self.h[i1], self.h[j1])
            free = ~self._blocked(i1, j1, top)
            i1, j1 = i1[free], j1[free]
            drop = np.maximum(self.h[i1] - self.h[j1], 0)
            src.append(i1)
            dst.append(j1)
            w.append(d * 0.5 * (node_cost[i1] + node_cost[j1]) + p['drop_cost'] * drop)
            gap = idx[~has]                            # nothing reachable next door: a jump
            for k in range(2, int(p['jump_cells']) + 1):
                if len(gap) == 0:
                    break
                col = (self.x[gap] + dx * k) * 4096 + (self.z[gap] + dz * k)
                jj = self._find(col, self.h[gap] - p['max_drop'] * 2, self.h[gap] + p['jump_rise'])
                hit = jj >= 0
                src.append(gap[hit])
                dst.append(jj[hit])
                w.append(d * k * p['jump_cost'] * 0.5 * (node_cost[gap[hit]] + node_cost[jj[hit]]))
                gap = gap[~hit]
        src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
        return csr_matrix((w, (src, dst)), shape=(self.n, self.n))

    def nearest(self, pos) -> int:
        d = np.abs(self.x - math.floor(pos[0] / CELL)) + np.abs(self.z - math.floor(pos[2] / CELL)) + \
            np.where(self.h > pos[1] + 1.0, 1e6, np.abs(self.h - pos[1]) / 8)
        return int(np.argmin(d))

    def block_nodes(self, bx, by, bz) -> np.ndarray:
        inb = (self.x * CELL // 32 == bx) & (self.z * CELL // 32 == bz) & \
            (self.h >= by * 8 - 4) & (self.h <= by * 8 + 24)
        return np.flatnonzero(inb)


class Planner:
    """The per-map part (geometry, waypoints) is built once; plan() can then be run with any
    parameters (fast: used when tuning)."""

    def __init__(self, challenge: Path, track_id: str = 'plan'):
        from .replaybuild import map_blocks
        from .virtual_game import tmi_waypoint, WP_CP, WP_FINISH, WP_STARTFINISH
        prep = tmnfc.prepare_map(Path(challenge), str(track_id))
        _, names, xyz, _ = map_blocks(Path(challenge))
        y_max = float(np.max(xyz[:, 1]) * 8 + 48) if len(xyz) else 300.0
        fp, fm, fny, wp = surface_points(prep['track'], y_max)
        self.g = Grid(fp, fm, fny, wp, block_index(names, xyz))
        self.start = self.g.nearest(prep['spawn'][9:])
        cps = sorted({tuple(int(v) for v in p) for n, p in zip(names, xyz) if tmi_waypoint(n) == WP_CP})
        fins = [tuple(int(v) for v in p) for n, p in zip(names, xyz)
                if tmi_waypoint(n) in (WP_FINISH, WP_STARTFINISH)]
        self.cp_nodes = [self.g.block_nodes(*c) for c in cps]
        self.fin_nodes = np.unique(np.concatenate([self.g.block_nodes(*f) for f in fins])) if fins else np.zeros(0, int)
        self.feat = self.g.features()

    def node_cost(self, p: dict, model: Optional[dict]) -> np.ndarray:
        base = np.array([MATERIAL_BASE.get(int(m), 1.2) for m in self.g.mat])
        if not model:
            return base
        z = self.feat @ np.array(model['w'])
        prob = 1.0 / (1.0 + np.exp(-z))
        return 1.0 + p['w_learned'] * -np.log(np.clip(prob, 1e-4, 1.0))

    def bridged(self, graph, p: dict):
        """Make every checkpoint and the finish reachable: while a waypoint's nodes cannot be
        reached from the start and the other waypoints, connect the nearest reached node to the
        nearest node of that waypoint with an expensive edge (loops, wall rides and odd jumps are
        not in the grid; the bridge stands for them)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial import cKDTree
        g = self.g
        xyz = np.stack([(g.x + 0.5) * CELL, g.h, (g.z + 0.5) * CELL], 1)
        targets = self.cp_nodes + [self.fin_nodes]
        for _ in range(len(targets) + 1):
            src = np.concatenate([[self.start]] + self.cp_nodes)
            d = dijkstra(graph, indices=src, min_only=True)
            reached = np.flatnonzero(np.isfinite(d))
            missing = [t for t in targets if not np.isfinite(d[t]).any()]
            if not missing:
                return graph
            t = missing[0]
            dd, k = cKDTree(xyz[reached]).query(xyz[t])
            j = int(np.argmin(dd))
            a, b = int(reached[k[j]]), int(t[j])
            extra = csr_matrix(([float(dd[j]) * p['bridge_cost']], ([a], [b])), shape=graph.shape)
            graph = graph + extra
        return graph

    def plan(self, p: dict = None, model: Optional[dict] = None) -> Optional[np.ndarray]:
        from scipy.sparse.csgraph import dijkstra
        p = dict(DEFAULT, **(p or {}))
        if not len(self.fin_nodes) or any(len(c) == 0 for c in self.cp_nodes):
            return None
        graph = self.bridged(self.g.edges(p, self.node_cost(p, model)), p)
        groups = [np.array([self.start])] + self.cp_nodes
        tables = [dijkstra(graph, indices=src, min_only=True, return_predecessors=True) for src in groups]

        def cost(i, targets):
            d = tables[i][0][targets]
            k = int(np.argmin(d))
            return float(d[k]), int(targets[k])

        k = len(self.cp_nodes)
        if k <= 7:
            best = None
            for perm in itertools.permutations(range(k)):
                total, cur = 0.0, 0
                for c in perm:
                    total += cost(cur, self.cp_nodes[c])[0]
                    cur = c + 1
                total += cost(cur, self.fin_nodes)[0]
                if best is None or total < best[0]:
                    best = (total, perm)
            order = best[1]
        else:
            order, cur, left = [], 0, set(range(k))
            while left:
                c = min(left, key=lambda c: cost(cur, self.cp_nodes[c])[0])
                order.append(c)
                left.remove(c)
                cur = c + 1
        path, cur = [], 0
        for n_leg, target in enumerate([self.cp_nodes[c] for c in order] + [self.fin_nodes]):
            d, end = cost(cur, target)
            if not np.isfinite(d):
                return None
            pred = tables[cur][1]
            seg, j = [], end
            while j >= 0:
                seg.append(j)
                j = pred[j]
            path += seg[::-1] if not path else seg[::-1][1:]
            cur = order[n_leg] + 1 if n_leg < len(order) else cur
        g = self.g
        return np.array([((g.x[j] + 0.5) * CELL, g.h[j], (g.z[j] + 0.5) * CELL) for j in path])

    def on_line(self, ghost: np.ndarray) -> np.ndarray:
        """Nodes a record line passes over (the positive examples for the cost model)."""
        g = self.g
        hit = set()
        for q in ghost:
            cx, cz = math.floor(q[0] / CELL), math.floor(q[2] / CELL)
            cand = np.flatnonzero((g.x == cx) & (g.z == cz) & (g.h <= q[1] + 1.0) & (g.h >= q[1] - 6.0))
            if len(cand):
                hit.add(int(cand[np.argmax(g.h[cand])]))
        return np.array(sorted(hit), dtype=np.int64)


PACKAGED = Path(__file__).with_name('route_cost_model.json')   # the model shipped with the code


def load_model() -> Optional[dict]:
    """data/route_cost_model.json (a local route-learn run) or the one shipped with the code."""
    for f in (MODEL, PACKAGED):
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8'))
    return None


def plan(challenge: Path, track_id: str = 'plan', log=print) -> Optional[np.ndarray]:
    """The route as (n, 3) points (start ... finish), or None when no path was found."""
    model = load_model()
    params = dict(DEFAULT, **(model.get('params', {}) if model else {}))
    return Planner(challenge, track_id).plan(params, model)


def compare_to_ghost(route: np.ndarray, ghost: np.ndarray) -> dict:
    """Horizontal distance from each ghost sample to the route polyline, and back."""
    def dist(pts, line):
        a, b = line[:-1, [0, 2]], line[1:, [0, 2]]
        ab = b - a
        ll = np.maximum((ab ** 2).sum(1), 1e-9)
        out = []
        for q in pts[:, [0, 2]]:
            t = np.clip(((q - a) * ab).sum(1) / ll, 0, 1)
            out.append(np.sqrt(((a + ab * t[:, None] - q) ** 2).sum(1)).min())
        return np.array(out)
    d = dist(ghost, route)
    back = dist(route[::3], ghost) if len(ghost) > 1 else d
    return {'mean_m': float(d.mean()), 'p90_m': float(np.percentile(d, 90)), 'within_16m': float((d <= 16).mean()),
            'route_off_m': float(back.mean()),
            'route_m': float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
            'ghost_m': float(np.linalg.norm(np.diff(ghost, axis=0), axis=1).sum())}
