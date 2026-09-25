"""Progress without a reference line (course.py) and the respawn skip of shown runs.

    python tests/test_course.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tmdriver import course as C  # noqa: E402
from tmdriver import protocol as P  # noqa: E402
from tmdriver.improve import ImproveEpisode, Playback, score  # noqa: E402


def step(t, pos, contact=True, bits=0, cps=0, speed=0.0):
    return P.Step(race_time=t, flags=0, checkpoints=cps, pos=np.asarray(pos, np.float32),
                  rot=np.eye(3, dtype=np.float32).ravel(), vel=np.array([speed, 0, 0], np.float32),
                  ang_vel=np.zeros(3, np.float32), wheel_damper=np.zeros(4, np.float32),
                  wheel_contact=np.full(4, contact), wheel_sliding=np.zeros(4, bool),
                  wheel_material=np.zeros(4, np.int32), gear=1, rpm=0.0, display_speed=0,
                  in_steer=0, in_gas=0, in_bits=bits)


class FakePolicy:
    """Just the progress part of GhostPolicy (the same code path as observe())."""
    has_line = False
    line_progress = False
    last_decision_t = None
    last_out = None

    def __init__(self, blocks):
        self.track = C.track_cells(blocks)
        self.restart()

    def restart(self):
        self.cells, self.cps, self.on_track = set(), 0, True

    @property
    def progress_m(self):
        return len(self.cells) * C.CELL_M

    def act(self, st, **kw):
        self.cps = max(self.cps, int(st.checkpoints))
        c = C.cell_of(st.pos)
        self.on_track = c in self.track
        if self.on_track:
            self.cells.add(c)
        return 0, 0, P.UP


def run(ep, positions, dt=10, **kw):
    ep.begin(None)
    for k, p in enumerate(positions):
        if ep.act(step(k * dt, p, **kw)) is None:
            return ep.result()
    return ep.result()


def main():
    # a straight road of 10 blocks along +z at block height 1 (y = 8..16 m), plus filler grass
    road = [{'name': 'StadiumRoadMain', 'x': 5, 'y': 1, 'z': z} for z in range(10)]
    grass = [{'name': 'StadiumGrass', 'x': x, 'y': 0, 'z': z} for x in range(32) for z in range(32)]
    track = C.track_cells(road + grass)
    assert (5, 1, 3) in track and (6, 1, 3) in track and (7, 1, 3) not in track   # 1 cell sideways
    assert (5, -1, 3) in track and (5, 4, 3) in track and (5, 5, 3) not in track   # dy -2..+3
    assert C.cell_of([5 * 32 + 1, 9.0, 3 * 32 + 31.9]) == (5, 1, 3)

    pol = FakePolicy(road + grass)
    # along the road at 30 m/s: a new cell every ~1 s, never stalled
    along = [(5 * 32 + 16, 9.0, 0.3 * k) for k in range(1000)]
    r = run(ImproveEpisode(pol, 60000, 0.0, 0), along)
    assert r['reason'] is None and r['progress_m'] == 10 * 32, (r['reason'], r['progress_m'])
    # circling on the spot: no new cells -> stalled after 5 s on the ground
    circle = [(5 * 32 + 16 + 5 * np.cos(k / 50), 9.0, 16 + 5 * np.sin(k / 50)) for k in range(2000)]
    r = run(ImproveEpisode(pol, 60000, 0.0, 0), circle)
    assert r['reason'] == 'stalled' and 5.0 < r['race_ms'] / 1000 < 5.2, (r['reason'], r['race_ms'])
    # the same while airborne (no wheel contact) is not a stall: flights do not count
    r = run(ImproveEpisode(pol, 3000, 0.0, 0), circle, contact=False, speed=30.0)
    assert r['reason'] == 'time limit', r['reason']
    # off the road onto the grass, 100 m to the side, driving on there: off track after 4 s
    off = [(5 * 32 + 16, 9.0, 16.0)] * 10 + [(5 * 32 + 116, 9.0, 16 + 0.3 * k) for k in range(3000)]
    r = run(ImproveEpisode(pol, 60000, 0.0, 0), off)
    assert r['reason'] == 'off track' and abs(r['race_ms'] - (100 + 4000)) <= 10, (r['reason'], r['race_ms'])
    # the same in the air is no stall and not off track (a jump beside the road)
    r = run(ImproveEpisode(pol, 6000, 0.0, 0), off, contact=False, speed=30.0)
    assert r['reason'] == 'time limit', r['reason']
    # on its roof / side at a wall: no wheel contact, but slow -> stalled after 5 s, not 180 s
    r = run(ImproveEpisode(pol, 180000, 0.0, 0), circle, contact=False, speed=2.0)
    assert r['reason'] == 'stalled' and 5.0 < r['race_ms'] / 1000 < 5.2, (r['reason'], r['race_ms'])
    # a very long flight without progress still ends after 25 s
    r = run(ImproveEpisode(pol, 180000, 0.0, 0), circle * 2, contact=False, speed=30.0)
    assert r['reason'] == 'stalled' and 25.0 < r['race_ms'] / 1000 < 25.1, (r['reason'], r['race_ms'])
    # ranking: checkpoints before cells, a finish before everything
    a = {'finished': False, 'progress_m': 900.0, 'cps': 1}
    b = {'finished': False, 'progress_m': 300.0, 'cps': 2}
    f = {'finished': True, 'time_ms': 90000, 'progress_m': 100.0, 'cps': 0}
    assert score(f) > score(b) > score(a)
    print('OK: track cells, stall / flight / off-track rules, ranking')

    # a shown run ends when the player respawns / restarts (time back, car jump, respawn key)
    ticks = [(t, 0, 0, P.UP) for t in range(0, 5000, 10)]
    for name, bad in (('restart', lambda k: step(k * 10 if k < 100 else -500, (0, 9, k * 0.3))),
                      ('respawn at a CP', lambda k: step(k * 10, (0, 9, k * 0.3 if k < 100 else 0.0))),
                      ('respawn key', lambda k: step(k * 10, (0, 9, k * 0.3), bits=P.RESPAWN if k == 100 else 0))):
        pb = Playback(ticks)
        pb.begin(None)
        k = 0
        while pb.act(bad(k)) is not None:
            k += 1
        assert pb.result()['aborted'] and k == 100, (name, k)
    pb = Playback(ticks)
    pb.begin(None)
    k = 0
    while pb.act(step(k * 10, (0, 9, k * 0.3))) is not None:
        k += 1
    assert not pb.result()['aborted'] and k > 500
    print('OK: respawn / restart / car jump stop a shown run, a normal run is not stopped')


if __name__ == '__main__':
    main()
