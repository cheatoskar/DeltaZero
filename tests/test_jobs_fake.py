"""Tool buttons: the plugin sends P_JOB, `serve` hands the game connection to the job (its
own console on Windows), waits until the job has finished, then reconnects.

    python tests/test_jobs_fake.py
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import fake_game as FG  # noqa: E402
from tmdriver import protocol as P  # noqa: E402
from tmdriver import server as S  # noqa: E402
from tmdriver.link import Link  # noqa: E402

PORT = 18497


class FakeJob:
    """Stands in for the job process: connects to the game like a job would, then finishes."""

    def __init__(self, req, done):
        self.req, self.done, self.rc = req, done, None
        threading.Thread(target=self.work, daemon=True).start()

    def work(self):
        link = Link.connect(port=PORT, wait_s=5)
        kind, payload = link.read_message()
        assert kind == P.P_HELLO and payload[0] == P.PROTOCOL
        time.sleep(0.3)
        link.close()
        self.done.touch()
        self.rc = 0

    def poll(self):
        return self.rc


def main():
    game = FG.FakeGame(PORT)
    game.reconnect = True
    game.start()
    jobs, dirs = [], []

    def spawn(req, done):
        jobs.append(req.command())
        dirs.append(req.game_dir)
        return FakeJob(req, done)

    orig = S.Context.policy
    S.Context.policy = lambda self: None          # no model needed for this test
    t = threading.Thread(target=S.serve, kwargs=dict(port=PORT, wait_s=5, spawn=spawn, max_jobs=2), daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    while game.accepts < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    game.press_job(P.JOB_TRAIN, 10036840, 7, 3, P.JOB_GPU | P.JOB_LINE,   # 'show new best' off
                   game_dir=r'C:\Users\x\Documents\TrackMania')
    deadline = time.monotonic() + 15
    while game.accepts < 3 and time.monotonic() < deadline:   # serve, job, serve again
        time.sleep(0.05)
    assert game.accepts >= 3, f'serve did not reconnect after the job (accepts={game.accepts})'
    time.sleep(0.5)
    game.press_job(P.JOB_RESIM, 0, 20, 90, P.JOB_PER_TICK)   # 90 minutes
    t.join(timeout=15)
    assert not t.is_alive(), 'serve did not return after two jobs'
    S.Context.policy = orig
    assert jobs[0][3:] == ['--device', 'auto', 'improve', '--rounds', '7', '--map', '10036840', '--line',
                           '--no-show'], jobs[0]
    assert dirs == [r'C:\Users\x\Documents\TrackMania', ''], dirs   # tmdriver_game_folder
    assert jobs[1][3:] == ['--device', 'cpu', 'resim', '--source', 'bulk', '--replays', '5', '--hours', '1.5',
                           '--map', '0', '--per-tick'], jobs[1]
    game.stop_flag = True
    print('OK: job buttons hand the connection to the job and serve reconnects afterwards')


if __name__ == '__main__':
    main()
