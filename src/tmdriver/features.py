"""Model inputs. ONE implementation for training (re-simulated replays) and live driving.

Everything is expressed in the car frame, so a left-hander looks the same on every map
and in every direction. The frame is measured, not assumed (2026-09-24, lolsport
recording): the plugin sends mat3 rows; column 2 is forward (0.999 alignment with the
velocity), column 1 is up (0.996), column 0 = up x forward.

    state  (S,)     velocity / angular velocity / gravity in the car frame, wheels, gear,
                    rpm, surface, and `pace` (log of the run's time / the map's best time;
                    0 = drive like the best). No previous actions: in behaviour cloning
                    they invite the copycat failure (predict "same as last tick").
    route  (R, 3)   points on the reference path at fixed distances ahead, car frame
    blocks (K, ...) nearest non-filler blocks: name id, waypoint type, car-frame offset,
                    orientation relative to the car, distance
"""
from typing import Dict, List, Optional

import numpy as np

from .line import RefLine

FEATURES_VERSION = 1
ROUTE_D = np.array([0, 1, 2, 3, 5, 7, 10, 14, 19, 25, 32, 40, 50, 62, 77, 95, 120, 150], dtype=np.float64)
K_BLOCKS = 32
BLOCK_RANGE = 200.0
N_MATERIAL = 32
FILLER = {'', 'StadiumGrass'}   # 32x32 terrain grid the game reports on every map
BLOCK_SIZE = np.array([32.0, 8.0, 32.0])
STATE_DIM = 3 + 3 + 3 + 1 + 4 + 4 + 4 + N_MATERIAL + 2 + 1
BLOCK_FEAT = 6
STEER_BINS = 21


def car_axes(rot: np.ndarray):
    """rot (..., 9) as sent -> right, up, forward unit vectors in world coordinates."""
    m = np.asarray(rot, dtype=np.float64).reshape(*np.shape(rot)[:-1], 3, 3)
    return m[..., :, 0], m[..., :, 1], m[..., :, 2]


def to_car(v: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """World vectors (..., 3) or (..., N, 3) -> car frame (right, up, forward)."""
    r, u, f = car_axes(rot)
    if v.ndim == r.ndim + 1:            # a set of vectors per tick
        r, u, f = r[..., None, :], u[..., None, :], f[..., None, :]
    return np.stack([np.sum(v * r, -1), np.sum(v * u, -1), np.sum(v * f, -1)], -1)


def heading(fwd: np.ndarray) -> np.ndarray:
    return np.arctan2(fwd[..., 0], fwd[..., 2])


# ---------------------------------------------------------------------- state

def state_features(a: Dict[str, np.ndarray], pace: float = 0.0) -> np.ndarray:
    """a: arrays with a leading tick axis (as saved by re-simulation / built live)."""
    rot = a['rot']
    T = len(rot)
    vel_c = to_car(a['vel'].astype(np.float64), rot) / 100.0
    ang_c = to_car(a['ang_vel'].astype(np.float64), rot)
    g_c = to_car(np.broadcast_to(np.array([0.0, 1.0, 0.0]), (T, 3)), rot)
    speed = np.linalg.norm(a['vel'], axis=-1, keepdims=True) / 100.0
    contact = a['wheel_contact'].astype(np.float64)
    sliding = a['wheel_sliding'].astype(np.float64)
    damper = np.clip(a['wheel_damper'].astype(np.float64) * 10.0, -3, 3) if 'wheel_damper' in a \
        else np.zeros((T, 4))
    mat = np.zeros((T, N_MATERIAL))
    mids = np.clip(a['wheel_material'].astype(np.int64), 0, N_MATERIAL - 1)
    for w in range(4):
        mat[np.arange(T), mids[:, w]] += contact[:, w] * 0.25
    gear = a['gear'].astype(np.float64)[:, None] / 5.0
    rpm = a['rpm'].astype(np.float64)[:, None] / 10000.0
    pace_col = np.full((T, 1), pace)
    out = np.concatenate([vel_c, ang_c, g_c, speed, contact, sliding, damper, mat, gear, rpm, pace_col], 1)
    assert out.shape[1] == STATE_DIM
    return out.astype(np.float32)


# ---------------------------------------------------------------------- route

def progress_indices(line: RefLine, pos: np.ndarray, hint: int = 0) -> np.ndarray:
    """Sequential nearest-point search, exactly as the live driver does it tick by tick."""
    idx = np.empty(len(pos), dtype=np.int64)
    h = hint
    for i, p in enumerate(pos):
        h = line.locate(p, h)
        idx[i] = h
    return idx


def route_features(line: RefLine, pos: np.ndarray, rot: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(T, len(ROUTE_D), 3): reference points ahead of the nearest point, car frame / 50 m."""
    steps = np.round(ROUTE_D / line.spacing).astype(np.int64)
    j = np.minimum(idx[:, None] + steps[None, :], len(line.pts) - 1)
    rel = line.pts[j] - pos[:, None, :].astype(np.float64)
    return (to_car(rel, rot) / 50.0).astype(np.float32)


# ---------------------------------------------------------------------- blocks

class MapGeometry:
    """Blocks of one map as reported by the plugin (P_MAP), filler removed."""

    def __init__(self, blocks: List[dict], vocab: Dict[str, int], dir_sign: int, dir_offset_deg: float):
        keep = [b for b in blocks if b['name'] not in FILLER]
        self.n = len(keep)
        anchor = np.array([[b['x'], b['y'], b['z']] for b in keep], dtype=np.float64).reshape(-1, 3)
        self.center = anchor * BLOCK_SIZE + BLOCK_SIZE / 2.0
        self.name_id = np.array([vocab.get(b['name'], 1) for b in keep], dtype=np.int64)
        self.wp_id = np.array([int(b['waypoint']) + 1 for b in keep], dtype=np.int64)
        d = np.array([int(b['dir']) for b in keep], dtype=np.float64)
        self.dir_heading = np.radians(dir_offset_deg) + dir_sign * d * (np.pi / 2)

    def features(self, pos: np.ndarray, rot: np.ndarray):
        """(T,K) name ids, (T,K) waypoint ids, (T,K,BLOCK_FEAT) floats, (T,K) mask."""
        T = len(pos)
        names = np.zeros((T, K_BLOCKS), np.int64)
        wps = np.zeros((T, K_BLOCKS), np.int64)
        feats = np.zeros((T, K_BLOCKS, BLOCK_FEAT), np.float32)
        mask = np.zeros((T, K_BLOCKS), bool)
        if self.n == 0:
            return names, wps, feats, mask
        rel = self.center[None, :, :] - pos[:, None, :].astype(np.float64)     # (T,B,3)
        dist = np.linalg.norm(rel, axis=-1)
        k = min(K_BLOCKS, self.n)
        order = np.argsort(dist, axis=1)[:, :k]                               # (T,k)
        rows = np.arange(T)[:, None]
        d_sel = dist[rows, order]
        ok = d_sel <= BLOCK_RANGE
        rel_c = to_car(rel[rows, order], rot) / 64.0                            # (T,k,3)
        _, _, fwd = car_axes(rot)
        rel_h = self.dir_heading[order] - heading(fwd)[:, None]
        names[:, :k] = np.where(ok, self.name_id[order], 0)
        wps[:, :k] = np.where(ok, self.wp_id[order], 0)
        feats[:, :k, 0:3] = rel_c
        feats[:, :k, 3] = np.sin(rel_h)
        feats[:, :k, 4] = np.cos(rel_h)
        feats[:, :k, 5] = d_sel / BLOCK_RANGE
        feats[:, :k] *= ok[..., None]
        mask[:, :k] = ok
        return names, wps, feats, mask


# ---------------------------------------------------------------------- labels

def steer_bin(steer_unit: np.ndarray) -> np.ndarray:
    """[-1, 1] -> bin index; the extremes are exactly full lock (keyboard presses)."""
    return np.clip(np.round((steer_unit + 1.0) / 2.0 * (STEER_BINS - 1)), 0, STEER_BINS - 1).astype(np.int64)


def bin_to_steer(b: np.ndarray) -> np.ndarray:
    return np.asarray(b, dtype=np.float64) / (STEER_BINS - 1) * 2.0 - 1.0


def steer_unit(act_steer: np.ndarray, act_bits: np.ndarray, right_sign: int) -> np.ndarray:
    """Applied action -> steer in [-1, 1] in TMInterface's analog convention. Keyboard
    Right equals `right_sign` full lock (measured from data, see dataset.check_steer)."""
    from . import protocol as P
    analog = (act_bits & P.STEER_ANALOG) > 0
    kb = np.where(act_bits & P.RIGHT, right_sign, 0) + np.where(act_bits & P.LEFT, -right_sign, 0)
    return np.where(analog, np.clip(act_steer / P.STEER_MAX, -1, 1), kb).astype(np.float64)
