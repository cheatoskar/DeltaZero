"""Live driver for ghost-feature checkpoints (kind 'ghost2').

Per tick it records the car position; on the 100 ms lattice it runs the same heading
recursion as the training data (ghost.headings), and every `decide_ms` it builds the
features with the same functions as tracebuild and asks the model. Between decisions the
last action is held (Linesight also acts every 50 ms).

The reference line is optional: the model was trained with the line dropped 25% of the
time and on maps that had no second run, so it can drive from the blocks alone.
"""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import ghost as G
from . import protocol as P
from .ghost_torch import DriverNet2, block_features


class NoLine:
    length = float('inf')


class GhostPolicy:
    ghost = True

    def __init__(self, ckpt, threads: int = 4, device: str = 'cpu'):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        if ck.get('kind') != 'ghost2' or ck['features_version'] != G.FEATURES_VERSION:
            raise RuntimeError(f'{ckpt}: not a ghost-feature v{G.FEATURES_VERSION} checkpoint')
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.model = DriverNet2(**ck['cfg']).to(self.device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self.meta = ck['meta']
        self.vocab = self.meta['vocab']
        self.decide_ms = int(self.meta.get('decide_every_ms', 50))
        self.orient_used = bool(self.meta.get('orient_used', False))
        self.val = ck.get('val', {})
        self.ckpt = str(ckpt)
        self.line = None
        self.blocks = None

    # ------------------------------------------------------------------ episode

    def reset(self, blocks: List[dict], line=None):
        """blocks: the plugin's P_MAP block dicts. line: None, positions (N, 3) or a ghost.Line."""
        if getattr(self, '_blocks_src', None) is not blocks:
            self.blocks = G.MapBlocks.from_plugin(blocks, self.vocab)
            self._blocks_src = blocks
            d = self.device
            self._bt = (torch.as_tensor(self.blocks.center, dtype=torch.float32, device=d)[None],
                        torch.as_tensor(self.blocks.heading, dtype=torch.float32, device=d)[None],
                        torch.as_tensor(self.blocks.name_id, device=d)[None],
                        torch.as_tensor(self.blocks.wp, device=d)[None],
                        torch.as_tensor(self.blocks.tb, dtype=torch.float32, device=d)[None],
                        torch.ones(1, len(self.blocks.name_id), dtype=torch.bool, device=d))
        self.set_line(line)
        self.restart()

    def set_line(self, line):
        if line is None:
            self.line = NoLine()
        elif isinstance(line, G.Line):
            self.line = line
        else:
            self.line = G.Line(np.asarray(line))

    def restart(self):
        self.pos = {}
        self.psi = {}
        self.psi0 = None
        self.s = 0.0
        self.odo = 0.0
        self.last_p = None
        self.action = (0, 0, 0)
        self.last_out = None
        self.last_decision_t = None

    @property
    def has_line(self) -> bool:
        return isinstance(self.line, G.Line)

    @property
    def progress_m(self) -> float:
        return self.s if self.has_line else self.odo

    # ------------------------------------------------------------------ features

    def _p(self, t: int) -> np.ndarray:
        t = max(t, 0)
        p = self.pos.get(t)
        if p is None:                     # a skipped tick: nearest earlier sample
            ks = [k for k in self.pos if k <= t]
            p = self.pos[max(ks)] if ks else self.pos[min(self.pos)]
        return p

    def _heading(self, t: int) -> float:
        """The recursion of ghost.headings on the 100 ms lattice through t (memoised)."""
        chain = []
        while t > 0 and t not in self.psi:
            chain.append(t)
            t -= G.DT_MS
        h = self.psi0 if t <= 0 else self.psi[t]
        for u in reversed(chain):
            d = self._p(u) - self._p(u - G.DT_MS)
            if np.hypot(d[0], d[2]) / (G.DT_MS / 1000.0) >= G.MIN_SPEED:
                h = float(np.arctan2(d[0], d[2]))
            self.psi[u] = h
        return h

    def features(self, t: int, orient: Optional[np.ndarray] = None, pace: float = 0.0):
        lags = np.stack([self._p(t - k * G.DT_MS) for k in range(G.HIST_K + 1)])[None]
        psi = np.array([self._heading(t)])
        psi_prev = np.array([self._heading(t - G.DT_MS)])
        state = G.state_features(lags, psi, psi_prev, orient if self.orient_used else None, pace,
                                 1.0 if self.has_line else 0.0)
        if self.has_line:
            route = G.route_features(self.line, np.array([self.s]), lags[:, 0], psi)
        else:
            route = np.zeros((1, len(G.ROUTE_D), 3), np.float32)
        return state, route, lags[0, 0], psi

    @torch.no_grad()
    def forward(self, state, route, pos, psi):
        d = self.device
        center, heading, name, wp, tb, valid = self._bt
        bn, bw, bf, bm = block_features(center, heading, name, wp, tb, valid,
                                        torch.as_tensor(pos, dtype=torch.float32, device=d)[None],
                                        torch.as_tensor(psi, dtype=torch.float32, device=d))
        x = (torch.as_tensor(state, device=d), torch.as_tensor(route, device=d), bn, bw, bf, bm)
        self.last_inputs = x          # kept for self-improvement (improve.py learns from its own runs)
        return self.model(*x)

    # ------------------------------------------------------------------ acting

    def observe(self, step: P.Step):
        t = int(step.race_time)
        p = step.pos.astype(np.float64)
        if t < 0:
            return t
        if self.psi0 is None:
            self.psi0 = self.blocks.start_heading(p)
        self.pos[t] = p
        if self.last_p is not None:
            self.odo += float(np.linalg.norm(p - self.last_p))
        self.last_p = p
        if self.has_line and t % G.DT_MS == 0:
            # on the 100 ms lattice only, exactly like tracebuild's progress()
            self.s = self.line.locate(p, self.s)
        return t

    def decide(self, t: int, pace: float = 0.0, sample: bool = False, temp: float = 1.0, rng=None,
               orient=None):
        state, route, pos, psi = self.features(t, orient, pace)
        out = self.forward(state, route, pos, psi)
        logits = out['steer'][0].float()
        if sample:
            pr = torch.softmax(logits / temp, -1).cpu().numpy()
            rng = rng or np.random.default_rng()
            b = int(rng.choice(len(pr), p=pr / pr.sum()))
            gas = rng.random() < float(torch.sigmoid(out['gas'][0] / temp))
            brake = rng.random() < float(torch.sigmoid(out['brake'][0] / temp))
        else:
            b = int(logits.argmax())
            gas, brake = bool(out['gas'][0] > 0), bool(out['brake'][0] > 0)
        self.last_out = {'bin': b, 'gas': gas, 'brake': brake, 'value_s': float(out['value'][0]) * 10}
        return to_action(b, gas, brake)

    def act(self, step: P.Step, pace: float = 0.0, sample: bool = False, temp: float = 1.0, rng=None):
        t = self.observe(step)
        if t < 0:
            return 0, 0, 0
        if self.last_decision_t is None or t - self.last_decision_t >= self.decide_ms:
            orient = None
            if self.orient_used:
                # car forward / up = columns 2 / 1 of the plugin's matrix (measured in M0 and
                # against the replay ghost decoding, error 1e-4)
                m = np.asarray(step.rot, dtype=np.float64).reshape(3, 3)
                orient = np.concatenate([m[:, 2], m[:, 1]])[None]
            self.action = self.decide(t, pace, sample, temp, rng, orient)
            self.last_decision_t = t
        return self.action


def to_action(b: int, gas: bool, brake: bool):
    """Steer bin (TMInterface analog convention; +1 = the Right key, measured in M1) -> ACTION."""
    bits = (P.UP if gas else 0) | (P.DOWN if brake else 0)
    steer = 0
    if b == 0:
        bits |= P.LEFT
    elif b == G.STEER_BINS - 1:
        bits |= P.RIGHT
    else:
        steer = int(G.bin_to_steer(b) * P.STEER_MAX)
        if steer != 0:
            bits |= P.STEER_ANALOG
    return steer, 0, bits


def load_policy(ckpt: Path, device: str = None):
    """The right live driver for a checkpoint: ghost2 (features v2) or the M1 model.
    device: default TMDRIVER_DEVICE / auto (CUDA if available); the M1 model is CPU only."""
    from .paths import torch_device
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if ck.get('kind') == 'ghost2':
        return GhostPolicy(ckpt, device=device or torch_device())
    from .policy import Policy
    return Policy(ckpt)
