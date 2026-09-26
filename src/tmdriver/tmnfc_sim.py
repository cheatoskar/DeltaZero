"""Many cars on one map in TMNF-C, stepped from Python one 10 ms tick at a time (ctypes over
TMNF-C/build/tmnfc_api.dll, see tmnfc_port/compat/tmnfc_api.c). No game.

    sim = CarSim(map_file, track_id, n=8)
    sim.step([(steer, gas, bits)] * 8)       # DeltaZero actions, one per car
    st = sim.state(0)                        # pos, rot, vel, ang_vel, wheels, rpm, gear
    snap = sim.capture(0); ...; sim.restore(0, snap)

Inputs follow the plugin exactly (tmnfc.schedule semantics): digital keys change on edges,
analog steer is re-sent every tick while valid, the newest source wins. Analog gas is not
mapped yet (same as the re-simulation).
"""
import ctypes
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from . import tmnfc

API_DLL = tmnfc.TMNFC / 'build' / 'tmnfc_api.dll'
ROUTE_PY = tmnfc.TMNFC / 'compat' / 'route.py'
_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(str(API_DLL))
        L.tmc_open.restype = ctypes.c_void_p
        L.tmc_open.argtypes = [ctypes.c_char_p] * 4 + [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                                                        ctypes.c_uint32]
        L.tmc_close.argtypes = [ctypes.c_void_p]
        L.tmc_step.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        L.tmc_race.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        L.tmc_reset.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.tmc_state.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        L.tmc_capture.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p]
        L.tmc_restore.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p]
        for f in ('tmc_state_floats', 'tmc_snapshot_size', 'tmc_input_size'):
            getattr(L, f).restype = ctypes.c_uint32
        assert L.tmc_input_size() == tmnfc.INPUT.size
        _lib = L
    return _lib


class InputMapper:
    """DeltaZero action (steer, gas, bits) -> TMNFRaceInputs for one car, keeping the key and
    analog-steer timestamps the game's mapper compares."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.left = self.right = False
        self.dig_tick = self.ana_tick = -1
        self.analog = 0

    def pack(self, k: int, action) -> bytes:
        s, g, b = action
        up, down = bool(b & 1), bool(b & 2)
        nl, nr = bool(b & 4), bool(b & 8)
        if (nl, nr) != (self.left, self.right):
            self.left, self.right, self.dig_tick = nl, nr, k
        s_now = s if b & 16 else 0
        if b & 16 or s_now != self.analog:
            self.analog, self.ana_tick = s_now, k
        ts = (k + 1) * tmnfc.TICK
        use_analog = self.ana_tick > self.dig_tick or (
            self.ana_tick == self.dig_tick and self.ana_tick >= 0 and not self.left and not self.right
            and abs(self.analog) / 65536.0 > 0.01)
        resp = int(bool(b & 64))
        if use_analog:
            return tmnfc.INPUT.pack(0, 0, 0, 0, 0, 0, ts, 0, float(-self.analog) / 65536.0,
                                    ts, 0, int(up), ts, 0, int(down), 0, resp, 0.0)
        return tmnfc.INPUT.pack(ts, 0, int(self.left), ts, 0, int(self.right), 0, 0, 0.0,
                                ts, 0, int(up), ts, 0, int(down), 0, resp, 0.0)

    def copy(self) -> 'InputMapper':
        m = InputMapper()
        m.left, m.right, m.dig_tick, m.ana_tick, m.analog = \
            self.left, self.right, self.dig_tick, self.ana_tick, self.analog
        return m


class CarSim:
    """n cars on one map. Race time of car i: 10 * (ticks[i] + 1) ms before its next step."""

    def __init__(self, challenge: Path, track_id: int, n: int = 1, threads: int = 1):
        prep = tmnfc.prepare_map(Path(challenge), str(track_id))
        import sys
        p = str(ROUTE_PY.parent)
        if p not in sys.path:
            sys.path.insert(0, p)
        import route as route_mod                            # TMNF-C compat/route.py
        full = prep['track'].with_suffix('.race.tmnfroute')   # the map's real triggers
        start_only = prep['track'].with_suffix('.tmnfroute')  # multilap / unknown trigger
        if not full.exists() and not start_only.exists():
            try:
                full.write_bytes(route_mod.full_route(Path(challenge)))
            except ValueError:
                start_only.write_bytes(route_mod.minimal_route(Path(challenge), prep['spawn']))
        # full_route: the race layer tracks checkpoints and the finish exactly
        self.full_route = full.exists()
        route = full if self.full_route else start_only
        self.n = n
        L = lib()
        spawn = (ctypes.c_float * 12)(*prep['spawn'])
        self.h = L.tmc_open(str(prep['track']).encode(), str(prep['vehicle']).encode(), str(route).encode(),
                            prep['sha'].encode(), spawn, n, threads)
        if not self.h:
            raise RuntimeError('tmc_open failed')
        self.nf = L.tmc_state_floats()
        self.snap_size = L.tmc_snapshot_size()
        self.mappers = [InputMapper() for _ in range(n)]
        self.ticks = [0] * n
        self._buf = (ctypes.c_float * self.nf)()

    def close(self):
        if self.h:
            lib().tmc_close(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def race_time(self, i: int) -> int:
        return 10 * (self.ticks[i] + 1)

    def step(self, actions: Sequence[Tuple[int, int, int]]):
        """One tick for every car; actions[i] = DeltaZero action applied at this race time."""
        data = b''.join(self.mappers[i].pack(self.ticks[i], actions[i]) for i in range(self.n))
        restarted = ctypes.create_string_buffer(self.n)
        lib().tmc_step(self.h, data, restarted)
        for i in range(self.n):
            self.ticks[i] += 1
        again = [i for i in range(self.n) if restarted.raw[i]]
        if again:                        # a respawn before any checkpoint restarts the race
            self.reset(again)
        return again

    def reset(self, cars: Sequence[int] = None):
        cars = range(self.n) if cars is None else cars
        mask = bytearray(self.n)
        for i in cars:
            mask[i] = 1
            self.mappers[i].reset()
            self.ticks[i] = 0
        lib().tmc_reset(self.h, bytes(mask))

    def state(self, i: int) -> dict:
        lib().tmc_state(self.h, i, self._buf)
        v = np.frombuffer(self._buf, dtype=np.float32).copy()
        return {'race_time': self.race_time(i), 'pos': v[0:3], 'rot': v[3:12], 'vel': v[12:15],
                'ang_vel': v[15:18], 'wheel_damper': v[18:22], 'wheel_contact': v[22:26] > 0.5,
                'wheel_sliding': v[26:30] > 0.5, 'wheel_material': v[30:34].astype(np.int32),
                'rpm': float(v[34]), 'gear': int(v[35])}

    def race(self, i: int) -> dict:
        out = (ctypes.c_uint32 * 4)()
        lib().tmc_race(self.h, i, out)
        return {'checkpoints': int(out[0]), 'finished': bool(out[1]), 'finish_ms': int(out[2]),
                'respawn_available': bool(out[3])}

    def capture(self, i: int):
        buf = ctypes.create_string_buffer(self.snap_size)
        lib().tmc_capture(self.h, i, buf)
        return (buf.raw, self.ticks[i], self.mappers[i].copy())

    def restore(self, i: int, snap):
        raw, ticks, mapper = snap
        lib().tmc_restore(self.h, i, raw)
        self.ticks[i] = ticks
        self.mappers[i] = mapper.copy()
