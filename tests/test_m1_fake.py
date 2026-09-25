"""End-to-end test of the M1 pipeline against the fake game (no TMNF needed).

    python tests/test_m1_fake.py

manifest -> resim (map loading, replay inputs, exactness) -> dataset (convention checks,
features) -> train (short) -> eval (the model drives) -> "AI fahren" via serve.
Six fake maps face all four block directions; replays are keyboard and pad drivers.
"""
import json
import os
import pickle
import sys
import tempfile
import threading
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix='tmdriver_m1_'))
os.environ['TMDRIVER_DATA'] = str(_TMP / 'data')
os.environ['TMDRIVER_RUNS'] = str(_TMP / 'runs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fake_game as FG  # noqa: E402
from tmdriver import protocol as P  # noqa: E402
from tmdriver import replay as replay_mod  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.paths import CALIBRATION, TMX, safe  # noqa: E402

PORT = 18479
DIRS = [0, 1, 2, 3, 1, 2]
DRIVERS = [(1.0, False), (0.9, False), (1.05, True)]   # (speed scale, analog)


def make_world(game):
    """Fake maps in the game, synthetic replays on disk, manifest + calibration."""
    from tmdriver.collect import M1, MANIFEST
    man = {'maps': {}}
    for k, d in enumerate(DIRS):
        tid = 900000 + k
        f = f'fake_{tid}.Challenge.Gbx'
        m = FG.fake_map(f'FAKEUID{tid}', f'Fake {k} dir{d}', d)
        game.maps[f] = m
        entry = {'ok': True, 'track_id': tid, 'uid': m['uid'], 'name': m['name'], 'map_file': f,
                 'holdout': k == len(DIRS) - 1, 'replays': []}
        folder = TMX / safe(m['uid'])
        folder.mkdir(parents=True, exist_ok=True)
        for j, (scale, analog) in enumerate(DRIVERS):
            r = FG.synth_replay(m, scale, analog)
            rep = replay_mod.Replay(map_uid=m['uid'], ghost_uid='g', race_time_ms=r['race_time_ms'],
                                    events=r['events'], ghost_t=r['ghost_t'], ghost_pos=r['ghost_pos'],
                                    respawns=0, cp_times=[], game_version='fake', login=f'p{j}')
            name = f'{tid}{j}.Replay.Gbx'
            (folder / name).write_bytes(pickle.dumps(rep))
            entry['replays'].append({'file': name, 'time_ms': r['race_time_ms'], 'respawns': 0,
                                     'analog': analog, 'uid_ok': True, 'player': f'p{j}'})
        man['maps'][str(tid)] = entry
    M1.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man), encoding='utf-8')
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps({
        'forward': {'kind': 'col', 'idx': 2, 'sign': 1}, 'steer': {'sign': -1},
        'replay': {'shift': 1, 'steer_sign': -1}}), encoding='utf-8')
    return man


def main():
    game = FG.FakeGame(PORT)
    man = make_world(game)
    game.start()
    link = Link.connect(port=PORT, wait_s=5)

    # 1) re-simulation
    from tmdriver.resim import run_resim
    res = run_resim(link, load_replay=lambda p: pickle.loads(Path(p).read_bytes()), log=print)
    n_exact = sum(r['exact'] for r in res)
    print(f'resim: {n_exact}/{len(res)} exact')
    assert len(res) == len(DIRS) * len(DRIVERS), len(res)
    assert n_exact == len(res), [r for r in res if not r['exact']][:2]
    cal = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    assert cal.get('map_command_template') == r'TMDriver\{f}', cal

    # 1b) batch playback (default) must record exactly what the per-tick path records
    snap = {r['file']: {k: v for k, v in np.load(r['file']).items() if k != 'meta'} for r in res}
    t0 = time.perf_counter()
    res2 = run_resim(link, load_replay=lambda p: pickle.loads(Path(p).read_bytes()), log=print,
                     only_missing=False, batch=False)
    print(f'per-tick resim: {time.perf_counter() - t0:.1f}s')
    assert len(res2) == len(res)
    for r in res2:
        z = np.load(r['file'])
        for k, v in snap[r['file']].items():
            assert np.array_equal(v, z[k]), (r['file'], k)
    print('batch == per-tick: identical recordings')

    # 2) dataset (includes the steer / block-direction convention checks)
    from tmdriver.dataset import build, DATASET
    build(stride=3)
    cal = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    assert cal['keyboard']['right_sign'] == 1, cal['keyboard']       # fake: Right == +analog
    assert cal['block_dir']['sign'] == -1 and cal['block_dir']['offset_deg'] == 0, cal['block_dir']
    z = np.load(DATASET)
    assert z['holdout'].any() and (~z['holdout']).any()
    for k in ('state', 'route', 'bfeat'):
        assert np.isfinite(z[k].astype(np.float32)).all(), k

    # 3) train briefly
    from tmdriver.train import train
    best = train(minutes=1.0)
    print('train best:', {k: round(v, 3) for k, v in best.items() if isinstance(v, float)})

    # 4) closed-loop eval: the model drives every fake map
    from tmdriver.evaluate import run_eval
    ev = run_eval(link, which='all', log=print)
    assert len(ev) == len(DIRS)
    print('eval:', [(r['name'], r['finished'], r['time_ms'], r['progress']) for r in ev])

    # 5) the window button path: serve + "AI fahren" must use the model
    link.close()
    game.stop_flag = True
    from tmdriver import server as S
    game2 = FG.FakeGame(PORT + 1)
    game2.maps.update(game.maps)
    first = next(k for k in game.maps if k != 'default')
    game2.load(first)
    game2.start()
    srv = S.Server(Link.connect(port=PORT + 1, wait_s=5), log=print)
    threading.Thread(target=srv.run, daemon=True).start()
    t0 = time.monotonic()
    while 'verbunden' not in game2.status and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    game2.press(P.MODE_DRIVE, speed=1.0)
    t0 = time.monotonic()
    while not any(w in game2.status for w in ('Ziel!', 'abgebrochen', 'Referenz', 'Bitte zuerst')):
        assert time.monotonic() - t0 < 120 and game2.is_alive(), game2.status
        time.sleep(0.05)
    print('drive button:', game2.status)
    assert '[Modell' in game2.status, game2.status
    game2.stop_flag = True
    print('OK: resim, dataset, train, eval and the drive button ran end to end')


if __name__ == '__main__':
    main()
