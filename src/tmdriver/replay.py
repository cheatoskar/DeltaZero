"""TMNF .Replay.Gbx -> ghost samples + per-tick input table.

Measured on a 30-replay pilot (2026-09-24): every TMX replay parses; its ghost samples
(100 ms) are identical to the HF `traces` rows (0.00 m), so the traces are a subset of
these files. The replay adds the exact input events (10 ms resolution, analog values),
orientation, respawns and the map UID (from the replay header; `ghost.uid` is a per-ghost
id, not the map - measured: it differs between replays of the same map).

Two conventions are NOT yet verified and are therefore parameters, decided by the in-game
self test (it replays the inputs and checks the finish time):
  * alignment between event time and TMInterface race time (`shift`, in ticks);
  * sign of analog steering relative to TMInterface's Steer input (`steer_sign`).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import protocol as P
from .gbx.reader import Gbx, GbxType

BINARY = {'Accelerate': P.UP, 'Brake': P.DOWN, 'SteerLeft': P.LEFT, 'SteerRight': P.RIGHT}
ANALOG = {'Steer', 'Gas'}
IGNORED = {'_FakeIsRaceRunning', '_FakeFinishLine', 'Horn'}
RESPAWN = P.RESPAWN


@dataclass
class Replay:
    map_uid: str                               # from the replay header (NOT ghost.uid)
    ghost_uid: str
    race_time_ms: int
    events: List[Tuple[int, str, int]]         # (gbx time ms, name, raw 32-bit value)
    ghost_t: np.ndarray                        # (N,) ms
    ghost_pos: np.ndarray                      # (N, 3)
    respawns: int
    cp_times: List[int]
    game_version: str
    login: str
    unknown_events: Dict[str, int] = field(default_factory=dict)
    ghost_orient: np.ndarray = None            # (N, 6) world forward + up per ghost sample

    @property
    def uses_analog_steer(self) -> bool:
        return any(n == 'Steer' for _, n, _ in self.events)


def ghost_orientation(records) -> np.ndarray:
    """Ghost sample rotation -> (N, 6) world forward + up of the car.

    Measured 2026-09-24 against re-simulated runs (11,406 samples, error 1e-4): `angle` is
    the HALF angle (raw / 0xFFFF * pi), axis = (cos H cos P, sin H cos P, sin P) with
    H = raw / 0x7FFF * pi, P = raw / 0x7FFF * pi/2, quaternion (cos a, axis sin a); the
    car's forward / up are columns 2 / 1 of its rotation matrix, as in the plugin's STEP."""
    raw = np.array([(r.angle, r.axis_heading, r.axis_pitch) for r in records], dtype=np.float64).reshape(-1, 3)
    a = raw[:, 0] / 0xFFFF * np.pi
    h = raw[:, 1] / 0x7FFF * np.pi
    p = raw[:, 2] / 0x7FFF * (np.pi / 2)
    ax = np.stack([np.cos(h) * np.cos(p), np.sin(h) * np.cos(p), np.sin(p)], 1)
    w = np.cos(a)
    x, y, z = (ax * np.sin(a)[:, None]).T
    fwd = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], 1)
    up = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], 1)
    return np.concatenate([fwd, up], 1)


def analog_value(raw: int) -> int:
    """Analog inputs are stored as a signed 24-bit integer."""
    v = raw & 0xFFFFFF
    return v - 0x1000000 if v & 0x800000 else v


def load(path) -> Replay:
    g = Gbx(str(path))
    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if not ghosts:
        raise ValueError(f'{path}: no ghost')
    gh = ghosts[0]
    events, unknown = [], {}
    for e in gh.control_entries:
        raw = (e.enabled & 0xFFFF) | ((e.flags & 0xFFFF) << 16)
        events.append((int(e.time), e.event_name, raw))
        if e.event_name not in BINARY and e.event_name not in ANALOG and \
                e.event_name not in IGNORED and e.event_name != 'Respawn':
            unknown[e.event_name] = unknown.get(e.event_name, 0) + 1
    period = gh.sample_period or 100
    pos = np.array([[r.position.x, r.position.y, r.position.z] for r in gh.records], dtype=np.float64)
    ident = g.replay_header().get('map_ident') or ['']
    return Replay(map_uid=ident[0], ghost_uid=gh.uid or '', race_time_ms=int(gh.race_time), events=events,
                  ghost_t=np.arange(len(pos)) * period, ghost_pos=pos.reshape(-1, 3),
                  respawns=int(gh.num_respawns), cp_times=list(gh.cp_times),
                  game_version=gh.game_version, login=gh.login or '', unknown_events=unknown,
                  ghost_orient=ghost_orientation(gh.records))


def input_table(rep: Replay, steer_sign: int = 1, pad_ms: int = 500) -> Dict[str, np.ndarray]:
    """Input state after all events with time <= t, for t = 0, 10, 20, ... ms (gbx time)."""
    end = max(rep.race_time_ms, max((t for t, _, _ in rep.events), default=0)) + pad_ms
    ts = np.arange(0, end + 10, 10, dtype=np.int64)
    steer = np.zeros(len(ts), dtype=np.int64)
    gas = np.zeros(len(ts), dtype=np.int64)
    bits = np.zeros(len(ts), dtype=np.int64)
    cur_steer = cur_gas = cur_bits = 0
    ev = sorted(rep.events, key=lambda e: e[0])   # stable: same-time events keep file order
    j = 0
    for i, t in enumerate(ts):
        respawn = 0
        while j < len(ev) and ev[j][0] <= t:
            _, name, raw = ev[j]
            if name in BINARY:
                b = BINARY[name]
                cur_bits = (cur_bits | b) if (raw & 0xFFFF) else (cur_bits & ~b)
            elif name == 'Steer':
                cur_steer = steer_sign * analog_value(raw)
            elif name == 'Gas':
                cur_gas = analog_value(raw)
            elif name == 'Respawn' and (raw & 0xFFFF):
                respawn = RESPAWN
            j += 1
        steer[i], gas[i] = cur_steer, cur_gas
        b = cur_bits | respawn
        if cur_steer != 0:
            b |= P.STEER_ANALOG
        if cur_gas != 0:
            b |= P.GAS_ANALOG
        bits[i] = b
    return {'t': ts, 'in_steer': steer, 'in_gas': gas, 'in_bits': bits}
