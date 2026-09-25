"""Frame conventions, measured from data instead of assumed.

Two facts the controller needs are not documented well enough to trust:
  * which vector of TMInterface's rotation matrix (rows x/y/z as the plugin sends them,
    or columns) is the car's forward axis, and its sign;
  * whether a positive analog steer value turns the car left or right.
Both are measured: the forward axis is the candidate best aligned with the velocity at
speed, the steer sign is the direction the car turns when the self test holds +steer.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import protocol as P

AXIS_CANDIDATES = [(kind, idx, sign) for kind in ('row', 'col') for idx in range(3) for sign in (1, -1)]


def axis(rot: np.ndarray, spec) -> np.ndarray:
    """rot: (..., 9) as sent by the plugin. spec: (kind, idx, sign)."""
    kind, idx, sign = spec
    m = np.asarray(rot, dtype=np.float64).reshape(*np.shape(rot)[:-1], 3, 3)
    v = m[..., idx, :] if kind == 'row' else m[..., :, idx]
    return sign * v


def heading(v: np.ndarray) -> np.ndarray:
    """Heading angle of a world vector in the horizontal plane (y is up)."""
    v = np.asarray(v, dtype=np.float64)
    return np.arctan2(v[..., 0], v[..., 2])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def calibrate_forward(rot: np.ndarray, vel: np.ndarray, min_speed: float = 8.0) -> Optional[Dict]:
    """Pick the rotation-matrix axis that best matches the direction of travel."""
    speed = np.linalg.norm(vel, axis=-1)
    m = speed > min_speed
    if m.sum() < 20:
        return None
    vdir = vel[m] / speed[m, None]
    scores = []
    for spec in AXIS_CANDIDATES:
        a = axis(rot[m], spec)
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
        scores.append(float(np.mean(np.sum(a * vdir, axis=-1))))
    order = np.argsort(scores)[::-1]
    best = AXIS_CANDIDATES[order[0]]
    return {'kind': best[0], 'idx': int(best[1]), 'sign': int(best[2]),
            'score': scores[order[0]], 'runner_up': scores[order[1]], 'samples': int(m.sum())}


def turn_rate(vel: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Signed rate of change of the travel heading (rad/s), in the `heading` convention."""
    h = np.unwrap(heading(vel))
    return np.gradient(h, dt)


def steer_sign_from(steer: np.ndarray, omega: np.ndarray, speed: np.ndarray) -> Optional[Dict]:
    """Sign s such that a positive steer value produces sign(omega) == s."""
    m = (np.abs(steer) > 1000) & (speed > 5.0)
    if m.sum() < 10:
        return None
    agree = np.sign(steer[m]) * np.sign(omega[m])
    frac = float(np.mean(agree > 0))
    return {'sign': 1 if frac >= 0.5 else -1, 'agreement': max(frac, 1 - frac), 'samples': int(m.sum())}


def effective_steer(in_steer: np.ndarray, in_bits: np.ndarray) -> np.ndarray:
    """Analog steer if used, else keyboard left/right as -/+ full lock (sign convention of
    the keyboard is itself unverified; only used for diagnostics)."""
    kb = np.where(in_bits & P.RIGHT, P.STEER_MAX, 0) - np.where(in_bits & P.LEFT, P.STEER_MAX, 0)
    return np.where(in_steer != 0, in_steer, kb)


class Calibration:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    @property
    def forward(self):
        f = self.data.get('forward')
        return (f['kind'], f['idx'], f['sign']) if f else None

    @property
    def steer_sign(self) -> Optional[int]:
        s = self.data.get('steer')
        return s['sign'] if s else None

    def ready(self) -> bool:
        return self.forward is not None and self.steer_sign is not None

    def update(self, **kv):
        self.data.update(kv)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
