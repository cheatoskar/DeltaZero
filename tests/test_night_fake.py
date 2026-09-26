"""The overnight loop (nightly.py) against two fake games: downloads run ahead of the
simulation, a map that cannot be downloaded is logged and skipped, a helper instance that
crashes mid-night does not stop the night, and a second night skips everything already done.

    python tests/test_night_fake.py
"""
import json
import os
import pickle
import sys
import tempfile
import threading
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix='tmdriver_night_'))
os.environ['TMDRIVER_DATA'] = str(_TMP / 'data')
os.environ['TMDRIVER_RUNS'] = str(_TMP / 'runs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_game as FG  # noqa: E402
import test_m1_fake as M1  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.paths import TMX, safe  # noqa: E402

PORT = 18559
BOGUS = 123456789


def main():
    game = FG.FakeGame(PORT)
    man = M1.make_world(game)
    helper = FG.FakeGame(PORT + 1)
    helper.maps = game.maps
    game.start()
    helper.start()
    maps = list(man['maps'].values())

    import tmdriver.nightly as N
    from tmdriver import tmx
    fake_files = _TMP / 'gamefolder'
    fake_files.mkdir()

    def fake_pool(n, log=print):
        return [{'track_id': m['track_id'], 'name': m['name']} for m in maps[:3]] + \
               [{'track_id': BOGUS, 'name': 'gone from TMX'}] + \
               [{'track_id': m['track_id'], 'name': m['name']} for m in maps[3:]]

    def fake_fetch(track_id, n_replays, tmx_root, safe_uid, log=print, reps=None):
        if track_id == BOGUS:
            raise ValueError(f'TMX track {track_id} not found')
        m = man['maps'][str(track_id)]
        f = fake_files / m['map_file']
        f.write_bytes(b'fake map')
        reps = [TMX / safe(m['uid']) / r['file'] for r in m['replays']][:n_replays]
        return {'uid': m['uid'], 'map': f, 'info': {'TrackName': m['name']}, 'replays': reps}

    N.ensure_pool = fake_pool
    tmx.fetch = fake_fetch
    load = lambda p: pickle.loads(Path(p).read_bytes())    # noqa: E731

    from tmdriver.session import GameSession
    GameSession.MAP_TIMEOUT_S = 5.0          # a hung instance is noticed quickly in the test
    links = [Link.connect(port=PORT, wait_s=5), Link.connect(port=PORT + 1, wait_s=5)]
    os.environ['TMDRIVER_DRAW_GAME'] = '0'   # --no-draw
    night = N.Night(links, [PORT, PORT + 1], hours=0.04, n_maps=10, replays=5, load_replay=load)

    def crash_helper():                       # the helper "crashes" once the night is under way
        while night.stats['maps_done'] < 2:
            time.sleep(0.1)
        helper.stop_flag = True
        try:
            helper.client.close()
        except Exception:
            pass

    threading.Thread(target=crash_helper, daemon=True).start()
    t0 = time.monotonic()
    st = night.run()
    print('night 1:', json.dumps(st))
    n_replays = sum(len(m['replays']) for m in maps)
    assert st['maps_done'] == len(maps), st
    assert st['replays'] == n_replays and st['exact'] == n_replays, st
    errors = [json.loads(l) for l in N.ERRORS.read_text(encoding='utf-8').splitlines()]
    assert any(e['track_id'] == BOGUS for e in errors), errors
    assert N.STATUS.exists()
    print(f'OK: {len(maps)} maps, {n_replays}/{n_replays} replays exact in {time.monotonic() - t0:.0f}s; '
          f'the missing map and the crashed helper were logged ({len(errors)} errors) and did not stop the night')

    assert ('draw_game', 'false') in game.var_log and game.vars.get('draw_game') == 'true', game.var_log
    assert game.vars.get('unfocused_fps_limit') == 'false', game.vars
    print('OK: --no-draw turned rendering off for the night and back on at the end')
    del os.environ['TMDRIVER_DRAW_GAME']

    night2 = N.Night(links[:1], [PORT], hours=0.01, n_maps=10, replays=5, load_replay=load)
    st2 = night2.run()
    assert st2['maps_done'] == 0 and st2['replays'] == 0, st2
    game.stop_flag = True
    print('OK: the second night skipped everything already done')


if __name__ == '__main__':
    main()
