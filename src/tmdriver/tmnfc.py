"""Client-free re-simulation of TMX replays with TMNF-C (github.com/adonis-singh/TMNF-C, MIT).

No game runs. Per map:
  1. the collision track is built from the .Challenge.Gbx (TMNF-C tools/build_track),
  2. the start pose is computed from the start block (bit-exact against the game on every map
     checked so far),
  3. ONE vehicle snapshot captured once in the game (on lolsport) is re-bound to the map,
     respawned at its start and stepped once, which gives the map's race-time-10 state,
  4. each replay's inputs (the same table the in-game re-simulation plays) run through
     tmnfc_batch, loading the track once per map,
  5. a run counts as exact when every ghost sample of the replay (every 100 ms, up to the
     recorded finish) is within 1 mm of the simulated car. No game is needed to check it.

Measured 2026-09-26 against exact in-game re-simulations: lolsport 5/5, Lucky Jump 5/5 and S81
Minis 4/4 at 0.000000 m; Hockolicious fails because the offline track builder orders its
collision entries differently from the game there (the in-game track dump is exact). Such maps
simply come out non-exact and are not used.

Setup (Windows): TMNF-C cloned and built with its compat/build_windows.sh (zig cc), the physics
image extracted from the local packs, a base vehicle snapshot. TMDRIVER_TMNFC points at the
clone (default: ~/Downloads/TMNF-C).
"""
import hashlib
import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .paths import DATA

TMNFC = Path(os.environ.get('TMDRIVER_TMNFC', Path.home() / 'Downloads' / 'TMNF-C'))
BATCH_EXE = TMNFC / 'build' / 'tmnfc_batch.exe'
BUILD_TRACK = TMNFC / 'tools' / 'build_track' / 'build_track.py'
BASE_VEHICLE = TMNFC / 'oracle' / 'vehicles' / 'LOLSPORT-Stadium.tmnfvehicle'
BASE_TRACK_SHA = 'f081a82f2e9ea5d37a54e8f3b5f50962bad8a5a54e59ceff98dabe52ebaa1e29'
PACKS = os.environ.get('TMDRIVER_PACKS', 'C:/Program Files (x86)/TmNationsForever/Packs')
CACHE = DATA / 'tmnfc'                  # built tracks and re-bound vehicles, per map
RESIM_C = DATA / 'resim_c'              # results: <track_id>/<replay>.npz + index.jsonl
# Build maps the exact builder rejects too (our TMNF-C patch, see tmnfc_port/): every run is
# still checked against its ghost, so a wrong track only costs that map. Measured 2026-09-26:
# 85 of 132 rejected maps build, 381 of 949 of their replays exact.
os.environ.setdefault('TMNF_LENIENT', '1')
LENIENT_TAG = 'lenient'
EXACT_TOL_M = 1e-3
STATE_EVERY_MS = 50                     # saved state resolution (actions stay per tick)
TICK = 10
INPUT = struct.Struct('<IIiIIiIIfIIiIIiIIf')     # TMNFRaceInputs (0x48 bytes)
OUT_REC = np.dtype([('t', '<i4'), ('pos', '<f4', 3), ('rot', '<f4', 9), ('vel', '<f4', 3),
                    ('ang_vel', '<f4', 3)])


def available() -> Optional[str]:
    """None when everything needed is present, else what is missing."""
    for p in (BATCH_EXE, BUILD_TRACK, BASE_VEHICLE, TMNFC / 'compat' / 'spawn.py'):
        if not p.exists():
            return f'missing {p}'
    return None


def schedule(actions) -> bytes:
    """TMNFRaceInputs for tick k from DeltaZero's action k = (steer, gas, bits), the action the
    plugin applies at race time k*10 (bits: 1 up, 2 down, 4 left, 8 right, 16 analog steer).
    Mirrors how the plugin sets TMInterface's input state (analog steer re-sent every tick
    while valid, digital keys change on edges) and the game's newest-source-wins rule
    (TMNF-C tools/wr_replay.build_schedule). Offset 0 is measured (0.000000 m)."""
    out = bytearray()
    left = right = False
    dig_tick = ana_tick = -1
    analog = 0
    for k, (s, g, b) in enumerate(actions):
        up, down = bool(b & 1), bool(b & 2)
        nl, nr = bool(b & 4), bool(b & 8)
        if (nl, nr) != (left, right):
            left, right = nl, nr
            dig_tick = k
        s_now = s if b & 16 else 0
        if b & 16 or s_now != analog:
            analog, ana_tick = s_now, k
        ts = (k + 1) * TICK
        use_analog = ana_tick > dig_tick or (ana_tick == dig_tick and ana_tick >= 0 and not left
                                             and not right and abs(analog) / 65536.0 > 0.01)
        if use_analog:
            out += INPUT.pack(0, 0, 0, 0, 0, 0, ts, 0, float(-analog) / 65536.0,
                              ts, 0, int(up), ts, 0, int(down), 0, 0, 0.0)
        else:
            out += INPUT.pack(ts, 0, int(left), ts, 0, int(right), 0, 0, 0.0,
                              ts, 0, int(up), ts, 0, int(down), 0, 0, 0.0)
    return bytes(out)


def _spawn_module():
    p = str(TMNFC / 'compat')
    if p not in sys.path:
        sys.path.insert(0, p)
    import spawn                                   # TMNF-C compat/spawn.py (our Windows port)
    return spawn


def prepare_map(challenge: Path, key: str) -> Dict:
    """Track, re-bound vehicle and start pose for one map (cached under data/tmnfc/<key>)."""
    sha = hashlib.sha256(challenge.read_bytes()).hexdigest()
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    track = d / f'{sha[:16]}.tmnftrack'
    if not track.exists():
        r = subprocess.run([sys.executable, str(BUILD_TRACK), str(challenge), '-o', str(track),
                            '--packs', PACKS], capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip().splitlines()
            raise RuntimeError(f'track build failed ({LENIENT_TAG}): ' + (msg[-1] if msg else '?'))
    vehicle = d / f'{sha[:16]}.tmnfvehicle'
    if not vehicle.exists():
        blob = BASE_VEHICLE.read_bytes()
        old = bytes.fromhex(BASE_TRACK_SHA)
        if blob.count(old) != 1:
            raise RuntimeError('base vehicle: track hash not found exactly once')
        vehicle.write_bytes(blob.replace(old, bytes.fromhex(sha)))
    spawn = _spawn_module().spawn(str(challenge))
    return {'sha': sha, 'track': track, 'vehicle': vehicle, 'spawn': spawn}


class MapSim:
    """One tmnfc_batch process for one map: the track is loaded once, each run gets a fresh
    world. Not thread safe; use one per worker."""

    START_TIMEOUT_S = 60.0
    RUN_TIMEOUT_S = 120.0

    def __init__(self, prep: Dict):
        spawn = ','.join(repr(float(v)) for v in prep['spawn'])
        self.tmp = tempfile.TemporaryDirectory(prefix='tmnfc_')
        self.p = subprocess.Popen([str(BATCH_EXE), str(prep['track']), str(prep['vehicle']), prep['sha'], spawn],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1)
        # drain stderr so the process never blocks on it; stdout lines go through a queue so
        # every wait has a timeout (a crashed or hung simulator must not stop the night)
        self._err: List[str] = []
        self._out: 'queue.Queue[str]' = queue.Queue()
        threading.Thread(target=lambda: self._err.extend(self.p.stderr), daemon=True).start()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        line = self._line(self.START_TIMEOUT_S)
        if line != 'ready':
            self.kill()
            raise RuntimeError(f"tmnfc_batch did not start: {line or ''.join(self._err)[-300:]}")
        self.n = 0

    def _read_stdout(self):
        for line in self.p.stdout:
            self._out.put(line.strip())
        self._out.put('')                          # EOF: the process ended

    def _line(self, timeout: float) -> str:
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return f'timeout after {timeout:.0f} s'

    def kill(self):
        try:
            self.p.kill()
        except OSError:
            pass

    def run(self, actions) -> np.ndarray:
        self.n += 1
        inp = Path(self.tmp.name) / f'in{self.n}.bin'
        out = Path(self.tmp.name) / f'out{self.n}.bin'
        inp.write_bytes(schedule(actions))
        self.p.stdin.write(f'{inp}\t{out}\n')
        self.p.stdin.flush()
        ans = self._line(self.RUN_TIMEOUT_S)
        if not ans.startswith('ok'):
            self.kill()
            raise RuntimeError(f"tmnfc_batch: {ans or ''.join(self._err)[-300:]}")
        rec = np.frombuffer(out.read_bytes(), dtype=OUT_REC).copy()
        inp.unlink()
        out.unlink()
        return rec

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.kill()
        try:
            self.tmp.cleanup()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def ghost_check(rec: np.ndarray, rep) -> Dict:
    """Compare the simulated car with the replay's ghost (every 100 ms up to the finish)."""
    pos = {int(t): p for t, p in zip(rec['t'], rec['pos'].astype(np.float64))}
    ts = [int(t) for t in rep.ghost_t if 10 <= int(t) <= rep.race_time_ms and int(t) in pos]
    if not ts:
        return {'exact': False, 'ghost_max_diff_m': float('inf'), 'ghost_samples': 0, 'first_off_ms': None}
    g = {int(t): p for t, p in zip(rep.ghost_t, rep.ghost_pos)}
    d = np.array([np.linalg.norm(pos[t] - g[t]) for t in ts])
    off = next((t for t, x in zip(ts, d) if x > EXACT_TOL_M), None)
    return {'exact': bool(d.max() <= EXACT_TOL_M), 'ghost_max_diff_m': float(d.max()),
            'ghost_samples': len(ts), 'first_off_ms': off}


def unsupported(rep) -> Optional[str]:
    if rep.respawns:
        return 'respawn'              # needs the race layer (checkpoint spawns): not wired yet
    if any(n == 'Gas' for _, n, _ in rep.events):
        return 'analog gas'           # not mapped to TMNFRaceInputs yet
    return None


def resim_map(challenge: Path, track_id: int, replays: List[Path], out_root: Path = RESIM_C,
              shift: int = 1, sign: int = -1, only_missing: bool = True, log=print) -> List[dict]:
    """Re-simulate a map's replays without the game. Writes <out_root>/<track_id>/<replay>.npz
    (t, pos, rot, vel, ang_vel per tick, the actions, meta) and returns one result per replay."""
    from . import replay as replay_mod
    from .resim import table_action
    out_dir = out_root / str(track_id)
    todo = [r for r in replays if not (only_missing and (out_dir / f'{Path(r).name.split(".")[0]}.npz').exists())]
    if not todo:
        return []
    prep = prepare_map(challenge, str(track_id))
    results = []
    with MapSim(prep) as sim:
        for rp in todo:
            rid = Path(rp).name.split('.')[0]
            base = {'track_id': track_id, 'replay': rid, 'map_sha256': prep['sha']}
            try:
                rep = replay_mod.load(rp)
            except Exception as e:                  # noqa: BLE001
                results.append(dict(base, exact=False, error=f'replay: {e!r}'[:200]))
                continue
            why = unsupported(rep)
            if why:
                results.append(dict(base, exact=False, skipped=why, recorded_ms=rep.race_time_ms))
                continue
            table = replay_mod.input_table(rep, steer_sign=sign)
            n = (rep.race_time_ms + 200) // TICK + 1
            actions = [table_action(table, k * TICK, shift) for k in range(n)]
            rec = sim.run(actions)
            chk = ghost_check(rec, rep)
            meta = dict(base, recorded_ms=rep.race_time_ms, analog=rep.uses_analog_steer, **chk)
            if chk['exact']:
                # State every 50 ms (the driver's decision rate) up to the finish; the actions are
                # kept per tick, so the full 10 ms run can be regenerated bit for bit any time.
                keep = (rec['t'] % STATE_EVERY_MS == 0) & (rec['t'] <= rep.race_time_ms)
                r = rec[keep]
                act = np.array(actions, dtype=np.int32).reshape(-1, 3)
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_dir / f'{rid}.npz', t=r['t'], pos=r['pos'], rot=r['rot'],
                                    vel=r['vel'], ang_vel=r['ang_vel'], act_steer=act[:, 0],
                                    act_gas=act[:, 1].astype(np.int16), act_bits=act[:, 2].astype(np.uint8),
                                    meta=json.dumps(meta))
            results.append(meta)
    return results
