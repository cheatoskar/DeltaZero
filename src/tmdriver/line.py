"""Reference line: a recorded run resampled by arc length, plus a pure-pursuit follower.

This is M0 scaffolding: it tests the plugin link end to end without any learning. The
line comes from a recorded run (RECORD mode); the follower steers towards a
point ahead on it and matches the recorded speed there.
"""
from pathlib import Path

import numpy as np

from . import protocol as P
from .calib import axis, heading, wrap


class RefLine:
    EXTEND_M = 200.0   # straight continuation past the end (Linesight does the same), so the
                       # route ahead never collapses onto one point near the finish

    def __init__(self, pos: np.ndarray, speed: np.ndarray, spacing: float = 0.5):
        pos = np.asarray(pos, dtype=np.float64)
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-4])      # drop standing-still duplicates
        pos, speed = pos[keep], np.asarray(speed, dtype=np.float64)[keep]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        self.spacing = spacing
        self.s = np.arange(0.0, s[-1], spacing)
        self.pts = np.stack([np.interp(self.s, s, pos[:, k]) for k in range(3)], axis=1)
        self.v = np.interp(self.s, s, speed)
        self.length = float(s[-1])          # the real path; points continue EXTEND_M beyond it
        back = min(len(self.pts) - 1, int(5.0 / spacing))
        if back > 0:
            d = self.pts[-1] - self.pts[-1 - back]
            d = d / max(np.linalg.norm(d), 1e-9)
            n = int(self.EXTEND_M / spacing)
            ext = self.pts[-1] + d[None, :] * (np.arange(1, n + 1)[:, None] * spacing)
            self.pts = np.concatenate([self.pts, ext])
            self.v = np.concatenate([self.v, np.full(n, self.v[-1])])
            self.s = np.concatenate([self.s, self.s[-1] + np.arange(1, n + 1) * spacing])

    @classmethod
    def load(cls, path: Path) -> 'RefLine':
        d = np.load(path)
        return cls(d['pos'], np.linalg.norm(d['vel'], axis=1))

    def locate(self, p: np.ndarray, hint: int, back_m: float = 10.0, ahead_m: float = 60.0) -> int:
        """Nearest line index to p, searched near the previous index so that crossings and
        overlapping sections of the route cannot make the index jump."""
        lo = max(0, hint - int(back_m / self.spacing))
        hi = min(len(self.pts), hint + int(ahead_m / self.spacing) + 1)
        d = np.linalg.norm(self.pts[lo:hi] - p, axis=1)
        return lo + int(np.argmin(d))


class Pursuit:
    """Pure pursuit on the horizontal plane with speed matched to the recording."""

    def __init__(self, line: RefLine, forward_spec, steer_sign: int, gain: float = 2.0,
                 look_time: float = 0.45, look_min: float = 6.0, look_max: float = 40.0):
        self.line = line
        self.forward = forward_spec
        self.steer_sign = steer_sign
        self.gain = gain
        self.look_time, self.look_min, self.look_max = look_time, look_min, look_max
        self.idx = 0

    def reset(self):
        self.idx = 0

    def act(self, step: P.Step):
        line = self.line
        self.idx = line.locate(step.pos.astype(np.float64), self.idx)
        speed = step.speed
        look = float(np.clip(self.look_time * speed + 4.0, self.look_min, self.look_max))
        j = min(self.idx + int(look / line.spacing), len(line.pts) - 1)
        target = line.pts[j] - step.pos
        fwd = axis(step.rot, self.forward)
        alpha = float(wrap(heading(target) - heading(fwd)))
        steer = int(np.clip(self.steer_sign * self.gain * alpha, -1.0, 1.0) * P.STEER_MAX)

        # Speed reference: the slowest recorded speed over the next ~0.6 s of track, so the
        # car is already slow when it reaches the corner.
        k = min(self.idx + int(max(speed * 0.6, 3.0) / line.spacing), len(line.v) - 1)
        v_ref = float(line.v[self.idx:k + 1].min())
        bits = P.STEER_ANALOG
        if speed < v_ref + 0.5:
            bits |= P.UP
        elif speed > v_ref * 1.08 + 2.0:
            bits |= P.DOWN
        return steer, 0, bits

    @property
    def progress_m(self) -> float:
        return self.idx * self.line.spacing
