"""RL v2 (rl.py) against the fake game, on two game instances: iterations run, the critic
warms up, PPO updates the model, branch runs start on both instances, the best run is saved and
shown, and Ctrl+C in the middle leaves every instance in sync.

    python tests/test_rl_fake.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix='tmdriver_rl_'))
os.environ['TMDRIVER_DATA'] = str(_TMP / 'data')
os.environ['TMDRIVER_RUNS'] = str(_TMP / 'runs')
os.environ['TMDRIVER_CKPT'] = str(_TMP / 'runs' / 'ghost' / 'best.pt')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fake_game as FG  # noqa: E402
import test_ghost_fake as TG  # noqa: E402
from tmdriver import protocol as P  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.paths import DATA, RUNS  # noqa: E402

PORT = 18529


def main():
    game = FG.FakeGame(PORT)
    TG.make_world(game)
    from tmdriver.tracebuild import build_all
    build_all(DATA / 'hf' / 'traces', DATA / 'hf' / 'blocks', workers=1, stride=1)
    from tmdriver.pretrain import pretrain
    pretrain(hours=0.02, bs=256, chunk=64, d=64, layers=2, heads=4, workers=0, eval_min=0.5, val_n=5000)
    helper = FG.FakeGame(PORT + 1)
    helper.maps = game.maps
    ticks = {'main': 0, 'helper': 0}
    for g, key in ((game, 'main'), (helper, 'helper')):
        orig_send = g.send_step

        def counted(msg=P.P_STEP, orig_send=orig_send, key=key):
            ticks[key] += 1
            return orig_send(msg)

        g.send_step = counted
    game.start()
    helper.start()
    link = Link.connect(port=PORT, wait_s=5)
    hlink = Link.connect(port=PORT + 1, wait_s=5)
    from tmdriver.collect import MANIFEST
    tid = int(next(iter(json.loads(MANIFEST.read_text(encoding='utf-8'))['maps'])))
    import tmdriver.rl as RL
    before = {k: v.clone() for k, v in torch.load(os.environ['TMDRIVER_CKPT'], weights_only=False)['state_dict'].items()}

    hist = RL.rl_train(link, track_id=tid, iterations=4, runs=4, branch=2, warmup=1, helpers=[hlink],
                       lr=1e-4, seed=7)
    assert len(hist) == 4, len(hist)
    assert hist[0]['critic_only'] and not hist[1]['critic_only'], [h['critic_only'] for h in hist]
    assert all(h['decisions'] > 0 and np.isfinite(h['v_loss']) for h in hist), hist
    assert all(np.isfinite(h['kl_ref']) for h in hist[1:])
    assert ticks['helper'] > 1000, ticks
    out = next((RUNS / 'rl').iterdir())
    ck = torch.load(out / 'model.pt', weights_only=False)
    assert 'critic' in ck and ck['rl_on']
    moved = max(float((ck['state_dict'][k].float() - before[k].float()).abs().max()) for k in before)
    assert moved > 0, 'PPO did not change the model'
    assert (out / 'best_run.json').exists() and (out / 'progress.jsonl').exists()
    print('iterations:', [(h['iteration'], h['greedy'], h['finished'], h['best'], h['return_mean'],
                          h.get('entropy'), h.get('kl_ref')) for h in hist])
    print(f'OK: RL on two instances ({ticks}), critic warm-up then PPO, model changed by up to {moved:.2e}')

    # Ctrl+C in the middle of an iteration: best run saved and shown, instances still usable
    orig_act = RL.ImproveEpisode.act
    seen = []

    def boom(self, st):
        if st.race_time == 1500 and self.temp == 1.0:
            seen.append(1)
            if len(seen) == 3:
                raise KeyboardInterrupt
        return orig_act(self, st)

    RL.ImproveEpisode.act = boom
    hist2 = RL.rl_train(link, track_id=tid, iterations=5, runs=4, branch=2, warmup=1, helpers=[hlink], seed=8)
    RL.ImproveEpisode.act = orig_act
    assert len(hist2) < 5, len(hist2)
    hist3 = RL.rl_train(link, track_id=tid, iterations=1, runs=2, branch=0, warmup=0, helpers=[hlink], seed=9,
                        show=False)
    assert len(hist3) == 1
    game.stop_flag = helper.stop_flag = True
    print('OK: Ctrl+C mid-iteration: saved, shown, and both instances ran the next training normally')


if __name__ == '__main__':
    main()
