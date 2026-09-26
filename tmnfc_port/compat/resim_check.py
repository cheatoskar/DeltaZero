"""Re-simulate DeltaZero's exact in-game runs (data/resim/<track>/<replay>.npz) in TMNF-C and
compare the car position tick by tick.

    python compat/resim_check.py TRACK.tmnftrack VEHICLE.tmnfvehicle CHALLENGE.Gbx RUN.npz [--offset K]

The npz holds, per race time t = 0, 10, ..., the inputs DeltaZero's plugin applied (act_*:
steer, gas, bits 1 up 2 down 4 left 8 right 16 analog steer 32 analog gas 64 respawn) and the
game's position after that tick. replay_tick's native capture writes one 1668-byte record per
tick (race time, quat, rot, pos, ...).
"""
import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'build' / 'replay_tick.exe'
INPUT = struct.Struct('<IIiIIiIIfIIiIIiIIf')     # TMNFRaceInputs, 0x48 bytes
RECORD = 1668
TICK = 10


def schedule(act_steer, act_gas, act_bits, offset: int, n_ticks: int) -> bytes:
    """Per-tick TMNFRaceInputs. Tick k uses the action DeltaZero applied at race time
    (k + offset) * 10. Same newest-source-wins rule as tools/wr_replay.build_schedule."""
    out = bytearray()
    left = right = False
    dig_tick = ana_tick = -1
    analog = 0
    for k in range(n_ticks):
        i = k + offset
        if 0 <= i < len(act_bits):
            s, g, b = int(act_steer[i]), int(act_gas[i]), int(act_bits[i])
        else:
            s, g, b = 0, 0, 0
        up, down = bool(b & 1), bool(b & 2)
        nl, nr = bool(b & 4), bool(b & 8)
        if (nl, nr) != (left, right):
            left, right = nl, nr
            dig_tick = k
        s_now = s if b & 16 else 0
        if b & 16 or s_now != analog:
            analog = s_now
            ana_tick = k
        ts = (k + 1) * TICK
        use_analog = ana_tick > dig_tick or (ana_tick == dig_tick and ana_tick >= 0 and not left
                                             and not right and abs(analog) / 65536.0 > 0.01)
        resp = int(bool(b & 64))
        if use_analog:
            rec = INPUT.pack(0, 0, 0, 0, 0, 0, ts, 0, float(-analog) / 65536.0,
                             ts, 0, int(up), ts, 0, int(down), 0, resp, 0.0)
        else:
            rec = INPUT.pack(ts, 0, int(left), ts, 0, int(right), 0, 0, 0.0,
                             ts, 0, int(up), ts, 0, int(down), 0, resp, 0.0)
        out += rec
    return bytes(out)


def simulate(track: Path, vehicle: Path, sha: str, sched: bytes):
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d) / 'in.bin', Path(d) / 'out.bin'
        inp.write_bytes(sched)
        r = subprocess.run([str(HARNESS), '--native-capture', str(track), str(vehicle), str(inp),
                            str(outp), sha], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or r.stdout.strip())
        raw = outp.read_bytes()
    n = len(raw) // RECORD
    t = np.array([struct.unpack_from('<i', raw, k * RECORD)[0] for k in range(n)])
    pos = np.array([struct.unpack_from('<3f', raw, k * RECORD + 4 + 16 + 36) for k in range(n)])
    return t, pos


def compare(run: Path, track: Path, vehicle: Path, challenge: Path, offset: int):
    z = np.load(run, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    sha = hashlib.sha256(challenge.read_bytes()).hexdigest()
    n = len(z['t'])
    t_sim, p_sim = simulate(track, vehicle, sha, schedule(z['act_steer'], z['act_gas'], z['act_bits'], offset, n))
    game = {int(t): p for t, p in zip(z['t'], z['pos'])}
    common = [(k, int(t)) for k, t in enumerate(t_sim) if int(t) in game]
    d = np.array([np.linalg.norm(p_sim[k] - game[t]) for k, t in common])
    first_bad = next((t for (k, t), x in zip(common, d) if x > 1e-3), None)
    return meta, d, first_bad, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track', type=Path)
    ap.add_argument('vehicle', type=Path)
    ap.add_argument('challenge', type=Path)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--offset', type=int, default=None, help='tick offset; default: try -2..2')
    a = ap.parse_args()
    offsets = [a.offset] if a.offset is not None else [-2, -1, 0, 1, 2]
    for run in a.runs:
        for off in offsets:
            meta, d, bad, n = compare(run, a.track, a.vehicle, a.challenge, off)
            print(f"{run.name} offset {off:+d}: {n} ticks compared, max {d.max():.6f} m, "
                  f"mean {d.mean():.6f} m, first >1 mm at t={bad} (analog {meta['analog']}, "
                  f"finish {meta['finished_at']})")
            sys.stdout.flush()


if __name__ == '__main__':
    main()
