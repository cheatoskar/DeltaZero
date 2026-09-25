"""Features v2 ("ghost features"): computable from car POSITIONS every 100 ms.

Three sources feed the driver, and they must produce identical inputs:
  * HF traces      positions + inputs every 100 ms, no orientation   (pretraining, ~1M runs)
  * TMX replays    ghost positions + orientation every 100 ms, exact inputs (fine-tuning)
  * live driving   full state every 10 ms tick
So the core features use only positions sampled 100 ms apart. Orientation is an optional
group with its own flag (0 in the traces).

Frame ("motion frame"): forward = the horizontal direction of travel over the last
100 ms; up = world y; right = up x forward (the same handedness as the car frame measured
in M0). Below MIN_SPEED the heading is carried over from 100 ms earlier; at the race
start it is the start block's direction. The recursion runs on the 100 ms lattice, so
live (10 ms ticks) and recorded (100 ms samples) runs give the same frame at the same
race time.

Measured facts this relies on (2026-09-24):
  * trace positions == replay ghost positions (0.00 m) and == the game's world frame;
  * world = Gbx block coord * (32, 8, 32); MX parquet x/z = Gbx x/z - 1 (4 maps, all
    blocks match, rotations 100% equal);
  * trace input_steer is in TMInterface's convention (keyboard right = +1; analog = -raw
    replay value) and is the input state 10 ms before the sample.
"""
import math
from typing import Dict, List, Optional

import numpy as np

DT_MS = 100
HIST_K = 10                      # lags of 100 ms .. 1 s
MIN_SPEED = 3.0                  # m/s; below this the travel direction is noise
ROUTE_D = np.array([0, 1, 2, 3, 5, 7, 10, 14, 19, 25, 32, 40, 50, 62, 77, 95, 120, 150], dtype=np.float64)
K_BLOCKS = 32
BLOCK_RANGE = 200.0
BLOCK_SIZE = np.array([32.0, 8.0, 32.0])
FILLER = {'', 'StadiumGrass'}
STEER_BINS = 21
FEATURES_VERSION = 2

# state vector layout
S_HIST = slice(0, 3 * HIST_K)                   # past positions in the frame / 50 m
S_VEL = slice(30, 33)                           # velocity in the frame / 100
S_ACC = slice(33, 36)                           # acceleration in the frame / 100
S_SPEED = 36
S_YAWRATE = 37                                  # heading change over the last 100 ms (rad)
S_ORIENT = slice(38, 44)                        # car forward + up in the frame (optional)
S_ORIENT_FLAG = 44
S_PACE = 45
S_ROUTE_FLAG = 46
STATE_DIM = 47
BLOCK_FEAT = 6

# waypoint ids from the block NAME (MX data has no waypoint field; live uses the same rule)
WP_NONE, WP_START, WP_FINISH, WP_CP, WP_MULTILAP = 1, 2, 3, 4, 5


def waypoint_of(name: str) -> int:
    if 'StartFinish' in name:
        return WP_MULTILAP
    if 'Start' in name and 'Line' in name:
        return WP_START
    if 'Finish' in name:
        return WP_FINISH
    if 'Checkpoint' in name:
        return WP_CP
    return WP_NONE


def dir_heading(d) -> np.ndarray:
    """Block dir (0 North .. 3 West) -> heading atan2(x, z) of its forward direction.
    Measured in M1 on 27 maps: heading = -90 deg * d."""
    return -np.asarray(d, dtype=np.float64) * (math.pi / 2)


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def frame_vectors(psi):
    """heading (...,) -> right, fwd (..., 3); up is world y."""
    s, c = np.sin(psi), np.cos(psi)
    z = np.zeros_like(s)
    fwd = np.stack([s, z, c], -1)
    right = np.stack([c, z, -s], -1)
    return right, fwd


def to_frame(v, psi):
    """World vectors (..., 3) or (..., N, 3) with psi (...,) -> (right, up, fwd)."""
    right, fwd = frame_vectors(psi)
    if v.ndim == right.ndim + 1:
        right, fwd = right[..., None, :], fwd[..., None, :]
    return np.stack([np.sum(v * right, -1), v[..., 1], np.sum(v * fwd, -1)], -1)


# ---------------------------------------------------------------------- heading

def headings(pos: np.ndarray, psi0: float) -> np.ndarray:
    """pos (N, 3) on the 100 ms lattice -> motion heading per sample (the recursion)."""
    n = len(pos)
    psi = np.empty(n)
    psi[0] = psi0
    if n == 1:
        return psi
    d = np.diff(pos, axis=0)
    hs = np.hypot(d[:, 0], d[:, 2]) / (DT_MS / 1000.0)
    cand = np.arctan2(d[:, 0], d[:, 2])
    ok = hs >= MIN_SPEED
    # forward fill: psi[i] = cand[last ok j <= i-1], else psi0
    idx = np.where(ok, np.arange(1, n), 0)
    np.maximum.accumulate(idx, out=idx)
    full = np.concatenate([[psi0], cand])
    psi[1:] = full[idx]
    return psi


def start_heading(blocks_xyz: np.ndarray, blocks_dir: np.ndarray, blocks_wp: np.ndarray, p0: np.ndarray) -> float:
    """Direction of the start block nearest to the spawn (0 if the map has none)."""
    m = (blocks_wp == WP_START) | (blocks_wp == WP_MULTILAP)
    if not m.any():
        return 0.0
    c = blocks_xyz[m] * BLOCK_SIZE + BLOCK_SIZE / 2
    j = int(np.argmin(np.linalg.norm(c - p0, axis=1)))
    return float(dir_heading(blocks_dir[m][j]))


# ---------------------------------------------------------------------- state

def state_features(lags: np.ndarray, psi: np.ndarray, psi_prev: np.ndarray,
                   orient: Optional[np.ndarray] = None, pace=0.0, route_flag=None) -> np.ndarray:
    """lags (T, HIST_K + 1, 3): positions at t, t-100, ..., t-1000 ms (clamped at the race
    start). psi / psi_prev (T,): headings at t and t-100 ms. orient (T, 6) world forward +
    up of the car, or None. -> (T, STATE_DIM) float32."""
    T = len(lags)
    p = lags[:, 0]
    out = np.zeros((T, STATE_DIM), np.float64)
    rel = lags[:, 1:] - p[:, None]
    out[:, S_HIST] = (to_frame(rel, psi) / 50.0).reshape(T, -1)
    dt = DT_MS / 1000.0
    v = (lags[:, 0] - lags[:, 1]) / dt
    a = (lags[:, 0] - 2 * lags[:, 1] + lags[:, 2]) / (dt * dt)
    out[:, S_VEL] = to_frame(v, psi) / 100.0
    out[:, S_ACC] = np.clip(to_frame(a, psi) / 100.0, -5, 5)
    out[:, S_SPEED] = np.linalg.norm(v, axis=1) / 100.0
    out[:, S_YAWRATE] = wrap(psi - psi_prev)
    if orient is not None:
        out[:, S_ORIENT] = np.concatenate([to_frame(orient[:, 0:3], psi), to_frame(orient[:, 3:6], psi)], 1)
        out[:, S_ORIENT_FLAG] = 1.0
    out[:, S_PACE] = pace
    out[:, S_ROUTE_FLAG] = 0.0 if route_flag is None else route_flag
    return out.astype(np.float32)


def lag_positions(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """pos (N, 3) on the 100 ms lattice; idx (T,) sample indices -> (T, HIST_K + 1, 3)."""
    j = np.clip(idx[:, None] - np.arange(HIST_K + 1)[None, :], 0, len(pos) - 1)
    return pos[j]


# ---------------------------------------------------------------------- line

class Line:
    """A reference path resampled every SPACING m by arc length, extended straight
    EXTEND_M past its end (so the route ahead never collapses onto one point)."""
    SPACING = 2.0
    EXTEND_M = 200.0

    def __init__(self, pos: np.ndarray):
        pos = np.asarray(pos, dtype=np.float64)
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-3])
        pos = pos[keep]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        self.length = float(s[-1])
        grid = np.arange(0.0, max(self.length, self.SPACING) + 1e-9, self.SPACING)
        pts = np.stack([np.interp(grid, s, pos[:, k]) for k in range(3)], 1)
        back = min(len(pts) - 1, max(1, int(6.0 / self.SPACING)))
        d = pts[-1] - pts[-1 - back] if len(pts) > 1 else np.array([0.0, 0.0, 1.0])
        d = d / max(np.linalg.norm(d), 1e-9)
        n = int(self.EXTEND_M / self.SPACING)
        ext = pts[-1] + d[None] * (np.arange(1, n + 1)[:, None] * self.SPACING)
        self.pts = np.concatenate([pts, ext])

    def locate(self, p: np.ndarray, hint_s: float, back_m: float = 10.0, ahead_m: float = 60.0) -> float:
        """Arc position of the closest point to p, searched near the previous position so
        that crossings and overlapping sections cannot make it jump."""
        sp = self.SPACING
        lo = max(0, int(hint_s / sp) - int(back_m / sp))
        hi = min(len(self.pts) - 1, int(hint_s / sp) + int(ahead_m / sp) + 1)
        a = self.pts[lo:hi]
        b = self.pts[lo + 1:hi + 1]
        ab = b - a
        t = np.clip(np.sum((p - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-12), 0.0, 1.0)
        q = a + ab * t[:, None]
        k = int(np.argmin(np.sum((q - p) ** 2, 1)))
        return (lo + k + float(t[k])) * sp

    def points_at(self, s: np.ndarray) -> np.ndarray:
        """Arc positions (...,) -> points (..., 3), linear interpolation on the grid."""
        f = np.clip(np.asarray(s) / self.SPACING, 0, len(self.pts) - 1.000001)
        i = np.floor(f).astype(np.int64)
        w = (f - i)[..., None]
        return self.pts[i] * (1 - w) + self.pts[i + 1] * w


def progress(line: Line, pos: np.ndarray, hint_s: float = 0.0) -> np.ndarray:
    """Sequential locate over a run, exactly as the live driver does it."""
    out = np.empty(len(pos))
    h = hint_s
    for i, p in enumerate(pos):
        h = line.locate(p, h)
        out[i] = h
    return out


def route_features(line: Line, s: np.ndarray, pos: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """(T, len(ROUTE_D), 3): path points ahead, in the motion frame / 50 m."""
    pts = line.points_at(s[:, None] + ROUTE_D[None, :])
    return (to_frame(pts - pos[:, None, :], psi) / 50.0).astype(np.float32)


# ---------------------------------------------------------------------- blocks

def block_tiebreak(name_id, xyz) -> np.ndarray:
    """A per-block number in [0, 1) from its name and cell. Blocks stacked in one cell are
    exactly equally far from the car; without this, which of them makes the nearest-K cut
    would depend on array order and float precision (measured: up to 13% of samples on a
    map with stacked blocks differed between two implementations)."""
    xyz = np.asarray(xyz, dtype=np.int64).reshape(-1, 3)
    h = (np.asarray(name_id, dtype=np.int64) * 73856093) ^ (xyz[:, 0] * 19349663) \
        ^ (xyz[:, 1] * 83492791) ^ (xyz[:, 2] * 50331653)
    return (np.abs(h) % 1000) / 1000.0


def selection_key(dist, tb):
    """Nearest-first order: 1/64 m distance buckets, ties broken by block_tiebreak."""
    return np.floor(dist * 64.0) + tb


class MapBlocks:
    """Non-filler blocks of one map in world metres. The same object serves training
    (from the MX parquet, coords + (1, 0, 1)) and live driving (from the plugin's P_MAP)."""

    def __init__(self, names: List[str], xyz: np.ndarray, dirs: np.ndarray, vocab: Dict[str, int]):
        keep = np.array([n not in FILLER for n in names], bool)
        self.names = [n for n, k in zip(names, keep) if k]
        self.xyz = np.asarray(xyz, dtype=np.int64).reshape(-1, 3)[keep]
        self.dir = np.asarray(dirs, dtype=np.int64)[keep]
        self.wp = np.array([waypoint_of(n) for n in self.names], dtype=np.int64)
        self.name_id = np.array([vocab.get(n, 1) for n in self.names], dtype=np.int64)
        self.center = self.xyz * BLOCK_SIZE + BLOCK_SIZE / 2
        self.heading = dir_heading(self.dir)
        self.tb = block_tiebreak(self.name_id, self.xyz)

    @classmethod
    def from_plugin(cls, blocks: List[dict], vocab):
        return cls([b['name'] for b in blocks], np.array([[b['x'], b['y'], b['z']] for b in blocks]).reshape(-1, 3),
                   np.array([b['dir'] for b in blocks]), vocab)

    def start_heading(self, p0: np.ndarray) -> float:
        return start_heading(self.xyz, self.dir, self.wp, p0)


def block_features_np(center, heading, name_id, wp, tb, pos, psi):
    """numpy version for one map: (T,K) names, (T,K) wps, (T,K,BLOCK_FEAT), (T,K) mask.
    Kept identical to ghost_torch.block_features (tested)."""
    T = len(pos)
    names = np.zeros((T, K_BLOCKS), np.int64)
    wps = np.zeros((T, K_BLOCKS), np.int64)
    feats = np.zeros((T, K_BLOCKS, BLOCK_FEAT), np.float32)
    mask = np.zeros((T, K_BLOCKS), bool)
    n = len(center)
    if n == 0:
        return names, wps, feats, mask
    rel = center[None] - pos[:, None]
    dist = np.linalg.norm(rel, axis=-1)
    k = min(K_BLOCKS, n)
    order = np.argsort(selection_key(dist, tb[None, :]), axis=1, kind='stable')[:, :k]
    rows = np.arange(T)[:, None]
    d = dist[rows, order]
    ok = d <= BLOCK_RANGE
    rc = to_frame(rel[rows, order], psi) / 64.0
    rh = heading[order] - psi[:, None]
    names[:, :k] = np.where(ok, name_id[order], 0)
    wps[:, :k] = np.where(ok, wp[order], 0)
    feats[:, :k, 0:3] = rc
    feats[:, :k, 3] = np.sin(rh)
    feats[:, :k, 4] = np.cos(rh)
    feats[:, :k, 5] = d / BLOCK_RANGE
    feats[:, :k] *= ok[..., None]
    mask[:, :k] = ok
    return names, wps, feats, mask


# ---------------------------------------------------------------------- labels

def steer_bin(steer_unit) -> np.ndarray:
    return np.clip(np.round((np.asarray(steer_unit) + 1.0) / 2.0 * (STEER_BINS - 1)), 0, STEER_BINS - 1).astype(np.int64)


def bin_to_steer(b) -> np.ndarray:
    return np.asarray(b, dtype=np.float64) / (STEER_BINS - 1) * 2.0 - 1.0
