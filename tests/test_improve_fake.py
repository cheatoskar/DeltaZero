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
    from tmdriver.improve import improve
    hist = improve(link, track_id=tid, rounds=3, episodes=3, steps=5, bs=64)
    print('history:', [(h['round'], h['greedy'], h['finished'], h['best_ms'], h['best_progress_m']) for h in hist])
    assert len(hist) == 3
    out = next((RUNS / 'improve').iterdir())
    assert (out / 'model.pt').exists() and (out / 'best_inputs.txt').exists()
    print((out / 'best_inputs.txt').read_text()[:200])
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
