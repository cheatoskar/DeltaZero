"""Does ONE vehicle snapshot drive every map? Re-bind the lolsport capture to another track, respawn
the car at that track's start and compare DeltaZero's exact in-game runs tick by tick.

    python compat/transfer_check.py TRACK_ID [TRACK_ID ...] [--mode 0|1] [--spawn npz|...]

Start pose: taken from the in-game run (npz row t=0) for now; computing it from the map file
is the next step.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resim_check import schedule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DZ = Path.home() / 'Downloads' / 'DeltaZero'
MAPS = Path.home() / 'Documents' / 'TrackMania' / 'Tracks' / 'Challenges' / 'TMDriver'
PACKS = 'C:/Program Files (x86)/TmNationsForever/Packs'
BASE_VEHICLE = ROOT / 'oracle' / 'vehicles' / 'LOLSPORT-Stadium.tmnfvehicle'
BASE_SHA = 'f081a82f2e9ea5d37a54e8f3b5f50962bad8a5a54e59ceff98dabe52ebaa1e29'
RUN = ROOT / 'build' / 'tmnfc_run.exe'
WORK = ROOT / 'out' / 'transfer'


def challenge_for(track_id: str) -> Path:
    hits = sorted(MAPS.glob(f'*_{track_id}.Challenge.Gbx'))
    if not hits:
        raise FileNotFoundError(f'no map file for {track_id} in {MAPS}')
    return hits[0]


def build_track(challenge: Path, out: Path) -> None:
    if out.exists():
        return
    r = subprocess.run([sys.executable, str(ROOT / 'tools/build_track/build_track.py'), str(challenge),
                        '-o', str(out), '--packs', PACKS], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1])


def rebind_vehicle(sha: str, out: Path) -> None:
    blob = BASE_VEHICLE.read_bytes()
    old, new = bytes.fromhex(BASE_SHA), bytes.fromhex(sha)
    if blob.count(old) != 1:
        raise RuntimeError(f'base track hash found {blob.count(old)} times in the vehicle snapshot')
    out.write_bytes(blob.replace(old, new))


def run(track: Path, vehicle: Path, sha: str, sched: bytes, spawn=None, mode=0):
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d) / 'in.bin', Path(d) / 'out.bin'
        inp.write_bytes(sched)
        cmd = [str(RUN), str(track), str(vehicle), sha, str(inp), str(outp)]
        if spawn is not None:
            cmd += [','.join(repr(float(v)) for v in spawn), str(mode)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip())
        raw = outp.read_bytes()
    rec = np.frombuffer(raw, dtype=np.dtype([('t', '<i4'), ('p', '<f4', 3)]))
    return rec['t'], rec['p'].astype(np.float64)


def check_map(track_id: str, mode: int, respawn_always: bool):
    challenge = challenge_for(track_id)
    sha = hashlib.sha256(challenge.read_bytes()).hexdigest()
    WORK.mkdir(parents=True, exist_ok=True)
    track = WORK / f'{track_id}.tmnftrack'
    build_track(challenge, track)
    vehicle = WORK / f'{track_id}.tmnfvehicle'
    if sha == BASE_SHA:
        vehicle.write_bytes(BASE_VEHICLE.read_bytes())
    else:
        rebind_vehicle(sha, vehicle)
    rows = []
    for npz in sorted((DZ / 'data' / 'resim' / track_id).glob('*.npz')):
        z = np.load(npz, allow_pickle=True)
        meta = json.loads(str(z['meta']))
        if not meta.get('exact'):
            continue
        spawn = None
        if sha != BASE_SHA or respawn_always:
            i0 = int(np.flatnonzero(z['t'] == 0)[0])
            spawn = list(z['rot'][i0]) + list(z['pos'][i0])
        n = len(z['t'])
        t_sim, p_sim = run(track, vehicle, sha, schedule(z['act_steer'], z['act_gas'], z['act_bits'], 0, n),
                           spawn, mode)
        game = {int(t): p for t, p in zip(z['t'], z['pos'])}
        d = np.array([np.linalg.norm(p - game[int(t)]) for t, p in zip(t_sim, p_sim) if int(t) in game])
        bad = next((int(t) for t, p in zip(t_sim, p_sim)
                    if int(t) in game and np.linalg.norm(p - game[int(t)]) > 1e-3), None)
        rows.append((npz.stem, len(d), float(d.max()), bad, meta['finished_at'], meta.get('analog')))
    return challenge.name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track_ids', nargs='+')
    ap.add_argument('--mode', type=int, default=0)
    ap.add_argument('--respawn-always', action='store_true', help='respawn even on the capture track')
    a = ap.parse_args()
    total = exact = 0
    for tid in a.track_ids:
        try:
            name, rows = check_map(tid, a.mode, a.respawn_always)
        except Exception as e:                      # noqa: BLE001
            print(f'{tid}: ERROR {e}')
            continue
        for rep, n, mx, bad, fin, analog in rows:
            total += 1
            exact += mx <= 1e-3
            print(f'{name} {rep}: {n} ticks, max {mx:.6f} m, first >1 mm at t={bad}, finish {fin}, analog {analog}')
        sys.stdout.flush()
    print(f'EXACT {exact}/{total}')


if __name__ == '__main__':
    main()
