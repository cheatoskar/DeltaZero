"""End-to-end test of stage A (ghost features) against the fake game (no TMNF needed).

    python tests/test_ghost_fake.py

synthetic HF-style trace + MX block parquet parts -> ghost-build -> pretrain (short, CPU)
-> eval (the model drives every fake map) -> "AI drive (live)" via serve uses the ghost model.
It checks the plumbing and the conventions, not driving quality (the fake world is tiny).
"""
import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix='tmdriver_ghost_'))
os.environ['TMDRIVER_DATA'] = str(_TMP / 'data')
os.environ['TMDRIVER_RUNS'] = str(_TMP / 'runs')
os.environ['TMDRIVER_CKPT'] = str(_TMP / 'runs' / 'ghost' / 'best.pt')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fake_game as FG  # noqa: E402
from tmdriver import protocol as P  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.paths import CALIBRATION, DATA, LINES, safe  # noqa: E402

PORT = 18489
DIRS = [0, 1, 2, 3, 1, 2, 3, 0]
NAMES = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}


def synth_trace(m, scale, analog):
    """HF-trace rows for one run: position every 100 ms and the input state 10 ms earlier
    (the alignment measured on real traces), steer in TMInterface's convention."""
    car, course, idx = FG.Car(m['h0']), m['course'], 0
    t, prev = 0, FG.Inputs()
    rows = []
    while t < 120000:
        if t % 100 == 0:
            rows.append((t, car.x, 10.0, car.z, prev.s(), 1.0 if prev.up else 0.0, 1.0 if prev.down else 0.0))
        i, idx = FG.human_policy(car, course, idx, scale, analog)
        prev = i
        car.tick(i.up, i.down, i.s())
        t += 10
        if FG.crossed_finish(course, car.x, car.z):
            break
    return rows


def make_world(game):
    from tmdriver.collect import M1, MANIFEST
    hf = DATA / 'hf'
    (hf / 'traces').mkdir(parents=True, exist_ok=True)
    (hf / 'blocks').mkdir(parents=True, exist_ok=True)
    from tmdriver.collect import holdout
    trows, brows, man = [], [], {'maps': {}}
    ids = [t for t in range(900000, 901000) if not holdout(t)][:len(DIRS) - 1]
    ids.append(next(t for t in range(900000, 901000) if holdout(t)))    # the split is by id hash
    for k, (tid, d) in enumerate(zip(ids, DIRS)):
        f = f'fake_{tid}.Challenge.Gbx'
        m = FG.fake_map(f'FAKEUID{tid}', f'Fake {k} dir{d}', d)
        game.maps[f] = m
        runs = [(1.0, False), (0.9, True)] if k % 3 else [(1.0, False)]      # some maps have one run
        for rank, (scale, analog) in enumerate(runs, 1):
            for r in synth_trace(m, scale, analog):
                trows.append((str(tid), rank, r[0], r[0], *r[1:]))
        for b in m['blocks']:
            brows.append((str(tid), b['name'], b['x'] - 1, b['y'], b['z'] - 1, NAMES[b['dir']]))
        pos = np.array([(x, y, z) for _, x, y, z, *_ in synth_trace(m, 1.0, False)])
        LINES.mkdir(parents=True, exist_ok=True)
        np.savez(LINES / f"{safe(m['uid'])}.npz", pos=pos, finish_ms=len(pos) * 100)
        man['maps'][str(tid)] = {'ok': True, 'track_id': tid, 'uid': m['uid'], 'name': m['name'], 'map_file': f,
                                 'holdout': holdout(tid),
                                 'replays': [{'file': 'x', 'time_ms': len(pos) * 100}]}
    pd.DataFrame(trows, columns=['track_id', 'trace_rank', 'timestamp_id', 'time_ms', 'x', 'y', 'z',
                                 'input_steer', 'input_gas', 'input_brake']).to_parquet(
        hf / 'traces' / 'traces_tmnf_part0001.parquet')
    pd.DataFrame(brows, columns=['track_id', 'name', 'x', 'y', 'z', 'dir']).to_parquet(
        hf / 'blocks' / 'blocks_tmnf_part0001.parquet')
    M1.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man), encoding='utf-8')
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps({
        'forward': {'kind': 'col', 'idx': 2, 'sign': 1}, 'steer': {'sign': -1},
        'replay': {'shift': 1, 'steer_sign': -1}, 'map_command_template': 'TMDriver\\{f}'}), encoding='utf-8')


def main():
    game = FG.FakeGame(PORT)
    make_world(game)

    # 1) shards
    from tmdriver.tracebuild import build_all, SHARDS
    st = build_all(DATA / 'hf' / 'traces', DATA / 'hf' / 'blocks', workers=1, stride=1)[0]
    print('build:', {k: st[k] for k in ('maps', 'samples', 'with_line', 'train_samples', 'val_samples',
                                        'start_heading_agree_frac')})
    assert st['maps'] == len(DIRS) and st['val_samples'] > 0
    assert st['start_heading_agree_frac'] == 1.0, 'start block direction disagrees with the driven direction'

    # 2) pretrain briefly on CPU
    from tmdriver.pretrain import pretrain
    best = pretrain(hours=0.03, bs=256, chunk=64, d=64, layers=2, heads=4, workers=0, eval_min=0.5,
                    val_n=5000, route_drop=0.25)
    print('pretrain best:', {k: round(v, 3) for k, v in best.items() if isinstance(v, float)})
    assert best['steer_dir_acc'] > 0.6, best

    # 3) closed-loop eval: the ghost model drives every fake map
    game.start()
    link = Link.connect(port=PORT, wait_s=5)
    from tmdriver.evaluate import run_eval
    ev = run_eval(link, which='all', log=print)
    assert len(ev) == len(DIRS)
    far = [r['progress_m'] for r in ev]
    print('eval progress (m):', far, 'finished:', sum(r['finished'] for r in ev))
    assert max(far) > 60, 'the ghost model does not drive at all'

    # 4) the window button uses the ghost model
    link.close()
    game.stop_flag = True
    from tmdriver import server as S
    game2 = FG.FakeGame(PORT + 1)
    game2.maps.update(game.maps)
    game2.load(next(k for k in game.maps if k != 'default'))
    game2.start()
    srv = S.Server(Link.connect(port=PORT + 1, wait_s=5), log=print)
    threading.Thread(target=srv.run, daemon=True).start()
    t0 = time.monotonic()
    while 'connected' not in game2.status and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    game2.press(P.MODE_DRIVE, speed=1.0)
    t0 = time.monotonic()
    while not any(w in game2.status for w in ('Finish!', 'stopped', 'Please run the self-test')):
        assert time.monotonic() - t0 < 180 and game2.is_alive(), game2.status
        time.sleep(0.05)
    print('drive button:', game2.status)
    assert 'model ghost/best.pt' in game2.status, game2.status
    game2.stop_flag = True
    print('OK: ghost-build, pretrain, eval and the drive button ran end to end')


if __name__ == '__main__':
    main()
