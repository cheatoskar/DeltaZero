"""improve (RL v1) against the fake game: rounds run, the elite is kept, the model is
fine-tuned, the best run is exported and played back.

    python tests/test_improve_fake.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix='tmdriver_improve_'))
os.environ['TMDRIVER_DATA'] = str(_TMP / 'data')
os.environ['TMDRIVER_RUNS'] = str(_TMP / 'runs')
os.environ['TMDRIVER_CKPT'] = str(_TMP / 'runs' / 'ghost' / 'best.pt')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_game as FG  # noqa: E402
import test_ghost_fake as TG  # noqa: E402
from tmdriver import protocol as P  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.paths import DATA, RUNS  # noqa: E402

PORT = 18499


def main():
    game = FG.FakeGame(PORT)
    TG.make_world(game)
    from tmdriver.tracebuild import build_all
    build_all(DATA / 'hf' / 'traces', DATA / 'hf' / 'blocks', workers=1, stride=1)
    from tmdriver.pretrain import pretrain
    pretrain(hours=0.02, bs=256, chunk=64, d=64, layers=2, heads=4, workers=0, eval_min=0.5, val_n=5000)
    game.start()
    link = Link.connect(port=PORT, wait_s=5)
    from tmdriver.collect import MANIFEST
    tid = int(next(iter(json.loads(MANIFEST.read_text(encoding='utf-8'))['maps'])))
    import time
    import tmdriver.improve as IM
    orig_ft, calls = IM.fine_tune, []

    def slow_fine_tune(*a, **k):      # longer than the plugin's 5 s step timeout: C_WAIT must hold it
        calls.append(1)
        if len(calls) == 1:
            time.sleep(7)
        return orig_ft(*a, **k)

    IM.fine_tune = slow_fine_tune
    shown = []
    orig_run = IM.GameSession.run

    def run(self, eps, sim_only=True, batch=False, keep=False):
        if any(isinstance(e, IM.Playback) for e in eps):
            shown.append(sim_only)
        return orig_run(self, eps, sim_only, batch, keep)

    IM.GameSession.run = run
    prefixes = []
    orig_pre = IM.PrefixEpisode.result

    def pre_result(self):
        r = orig_pre(self)
        prefixes.append(r)
        return r

    IM.PrefixEpisode.result = pre_result
    hist = IM.improve(link, track_id=tid, rounds=3, episodes=3, steps=5, bs=64, branch=4, use_line=True)
    assert all(h['line'] for h in hist), 'the fake TMX replay must give a reference line'
    IM.fine_tune, IM.GameSession.run, IM.PrefixEpisode.result = orig_ft, orig_run, orig_pre
    assert prefixes and all(r['diverged_m'] is not None and r['diverged_m'] < 1e-3 for r in prefixes), prefixes
    assert all(h['branch_from_s'] for h in hist[1:]), [h['branch_from_s'] for h in hist]
    print(f"branching: rounds 2-3 started {4} runs from {[h['branch_from_s'] for h in hist[1:]]} s; "
          f"replaying the best run to there was exact ({[round(r['diverged_m'], 6) for r in prefixes]} m)")
    assert shown and not any(shown), f'best runs must be shown visibly: {shown}'
    print(f'best run shown {len(shown)}x (every new best + the final one)')
    print('history:', [(h['round'], h['greedy'], h['finished'], h['best_ms'], h['best_progress_m']) for h in hist])
    assert len(hist) == 3
    out = next((RUNS / 'improve').iterdir())
    assert (out / 'model.pt').exists() and (out / 'best_inputs.txt').exists()
    print((out / 'best_inputs.txt').read_text()[:200])
    assert (out / 'best_run.json').exists()
    IM.show_best(link, track_id=tid)          # the "Show best run" button

    # without the line (the default): progress = track cells reached + checkpoints (course.py)
    hist = IM.improve(link, track_id=tid, rounds=2, episodes=3, steps=5, bs=64, branch=4, show=False)
    assert not any(h['line'] for h in hist)
    assert hist[-1]['best_progress_m'] >= 3 * 32, hist[-1]
    print('no line:', [(h['round'], h['greedy'], h['finished'], h['best_ms'], h['best_progress_m'], h['reasons'])
                       for h in hist])

    # stopping early (Ctrl+C): during the learning step (game held) and in the middle of a run
    # (the plugin waits for an ACTION). The best run is saved and shown, the link stays in sync.
    for where in ('fine_tune', 'run'):
        calls = []
        orig_act = IM.ImproveEpisode.act

        def boom_ft(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return orig_ft(*a, **k)

        def boom_act(self, st):
            if len(calls) >= 1 and st.race_time == 2000:
                raise KeyboardInterrupt
            return orig_act(self, st)

        def count_ft(*a, **k):
            calls.append(1)
            return orig_ft(*a, **k)

        if where == 'fine_tune':
            IM.fine_tune = boom_ft
        else:
            IM.fine_tune, IM.ImproveEpisode.act = count_ft, boom_act
        (out / 'best_run.json').unlink()
        shown.clear()
        IM.GameSession.run = run
        hist = IM.improve(link, track_id=tid, rounds=5, episodes=3, steps=5, bs=64, branch=4, show=False)
        IM.fine_tune, IM.ImproveEpisode.act, IM.GameSession.run = orig_ft, orig_act, orig_run
        assert len(hist) == 1, (where, len(hist))
        assert (out / 'best_run.json').exists(), where
        assert shown == [False], (where, shown)            # the best run was shown once, visibly
        IM.show_best(link, track_id=tid)                    # the connection is still in sync
        print(f'OK: stopped in {where}: best run saved and shown, link still in sync')

    # two game instances (fleet.py): the helper loads the map and drives its share of every
    # round, branch runs included; the best run is shown on the main instance only
    helper = FG.FakeGame(PORT + 7)
    helper.maps = game.maps
    steps = {'main': 0, 'helper': 0}
    for g, key in ((game, 'main'), (helper, 'helper')):
        orig_send = g.send_step

        def counted(msg=P.P_STEP, orig_send=orig_send, key=key):
            steps[key] += 1
            return orig_send(msg)

        g.send_step = counted
    helper.start()
    shown.clear()
    IM.GameSession.run = run
    prefixes.clear()
    IM.PrefixEpisode.result = pre_result
    hist = IM.improve(link, track_id=tid, rounds=3, episodes=3, steps=5, bs=64, branch=4,
                      helpers=[Link.connect(port=PORT + 7, wait_s=5)])
    IM.GameSession.run, IM.PrefixEpisode.result = orig_run, orig_pre
    assert len(hist) == 3 and steps['helper'] > 1000 and steps['main'] > 1000, (len(hist), steps)
    # greedy + 3 sampled; with branching greedy + 1 sampled + 4 branch runs (the prefixes are not results)
    assert all(int(h['finished'].split('/')[1]) == (6 if h['branch_from_s'] else 4) for h in hist), \
        [h['finished'] for h in hist]
    assert prefixes and all(r['diverged_m'] < 1e-3 for r in prefixes), prefixes
    assert shown and not any(shown), shown
    helper.stop_flag = True
    print(f"OK: two instances: {[h['finished'] for h in hist]} per round, ticks {steps}, "
          f"prefix replays exact on both ({len(prefixes)})")
    game.stop_flag = True
    print('OK: improve ran rounds, fine-tuned, exported and played back')




def test_preview():
    """drive: plan in simulation-only, draw, then the visible run must be identical."""
    game = FG.FakeGame(PORT + 1)
    TG.make_world(game)
    game.load(next(k for k in game.maps if k != 'default'))
    game.start()
    link = Link.connect(port=PORT + 1, wait_s=5)
    from tmdriver.improve import drive_preview
    plan, real = drive_preview(link)
    assert game.drawn, 'nothing drawn'
    assert plan['ticks'] == real['ticks'], 'the visible run differs from the plan'
    game.stop_flag = True
    print(f'OK: preview drew {len(game.drawn)} points and the drive followed the plan exactly')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    test_preview()
