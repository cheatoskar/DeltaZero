"""End-to-end test of the Python side against tests/fake_game.py (no TMNF needed).

    python tests/test_m0_fake.py

Runs the workflow cheatoskar will use in the game: self test -> record -> self test (now with
a recording) -> AI drive, and checks every number the self test reports.
"""
import json
import os
import re
import sys
import tempfile

# isolate all data/runs paths before any tmdriver import (a trained model or recordings in
# the real data folder must not change what this test exercises)
_TMP = tempfile.mkdtemp(prefix='tmdriver_m0_')
os.environ['TMDRIVER_DATA'] = os.path.join(_TMP, 'data')
os.environ['TMDRIVER_RUNS'] = os.path.join(_TMP, 'runs')
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_game import FakeGame  # noqa: E402
from tmdriver import protocol as P, server as S, tasks as T  # noqa: E402
from tmdriver.link import Link  # noqa: E402

PORT = 18478


def wait_status(game, pattern, timeout=120.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if re.search(pattern, game.status):
            return game.status
        if not game.is_alive():
            raise RuntimeError('fake game died')
        time.sleep(0.02)
    raise TimeoutError(f'status never matched {pattern!r}; last: {game.status!r}')


def latest_report(runs: Path):
    return json.loads(sorted(runs.glob('selftest_*.json'))[-1].read_text())


def main():
    tmp = Path(tempfile.mkdtemp(prefix='tmdriver_test_'))
    T.LINES, T.RUNS, S.LIVE_MAPS = tmp / 'lines', tmp / 'runs', tmp / 'live_maps'
    game = FakeGame(PORT)
    game.start()
    link = Link.connect(port=PORT, wait_s=5)
    logs = []
    srv = S.Server(link, log=logs.append, calibration_path=tmp / 'calibration.json')
    threading.Thread(target=lambda: _run(srv), daemon=True).start()
    wait_status(game, 'Python connected')

    # 1) self test without a recording
    game.press(P.MODE_TEST)
    wait_status(game, 'Selbsttest fertig')
    r = latest_report(tmp / 'runs')
    print('selftest 1:', json.dumps({k: r[k] for k in ('determinism', 'forward_axis', 'steer_sign',
                                                        'scripted_python_loop_ticks_per_s', 'bench_ticks_per_s')}))
    assert r['determinism']['max_diff_m'] == 0.0, r['determinism']
    fa = r['forward_axis']
    assert (fa['kind'], fa['idx'], fa['sign']) == ('col', 2, 1), fa
    assert r['steer_sign']['sign'] == -1, r['steer_sign']
    assert set(r['bench_ticks_per_s']) == {'1.0', '10.0', '100.0'}
    time.sleep(1.0)   # jitter-free: the replay files use second-resolution names

    # 2) record a human run
    game.press(P.MODE_RECORD)
    st = wait_status(game, 'Aufnahme gespeichert')
    print('record:', st)
    human_ms = float(re.search(r'([\d.]+) s', st).group(1)) * 1000

    # 3) self test again: now also replays the recorded inputs
    game.press(P.MODE_TEST)
    wait_status(game, 'Selbsttest fertig')
    r = latest_report(tmp / 'runs')
    by = {x['name']: x for x in r['replays']}
    h0, h1 = by['own_recording_shift0'], by['own_recording_shift1']
    print('human replay shift0:', h0)
    print('human replay shift1:', h1)
    assert h0['exact'], h0
    assert not h1['exact'] and h1['max_diff_m'] > 0.1, h1

    # 4) AI drives the recorded line
    game.press(P.MODE_DRIVE, speed=3.0)
    st = wait_status(game, r'Ziel!|abgebrochen')
    print('drive:', st)
    assert 'Ziel!' in st, st
    ai_ms = float(re.search(r'AI ([\d.]+) s', st).group(1)) * 1000
    assert ai_ms < human_ms * 1.25, (ai_ms, human_ms)
    assert abs(game.speed - 1.0) < 1e-6, 'speed must be restored after driving'
    print(f'OK  (AI {ai_ms / 1000:.2f}s vs recorded {human_ms / 1000:.2f}s)')
    game.stop_flag = True


def _run(srv):
    try:
        srv.run()
    except Exception as e:  # surfaced through the status wait timing out
        print('server stopped:', repr(e))


if __name__ == '__main__':
    main()
