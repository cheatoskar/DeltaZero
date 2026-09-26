"""A virtual game instance: the TMDriver plugin's protocol (plugin_server) over TMNF-C physics.

Every DeltaZero tool that talks to the game (rl, improve, drive, resim, fleets of instances)
works against it unchanged, with no game client, no window and no focus. Start instances with

    python tmdriver.py virtual --instances 11          (ports 8600, 8601, ...)

then point a tool at the first port (`rl --port 8600`); the others are found as helpers.

Physics: exact TMNF (see tmnfc.py for what was measured). Timing follows the real game: an
ACTION answered in the STEP at race time t acts on the tick from t+10 to t+20 (the replay
alignment measured in-game, shift 1). Checkpoints, the finish and respawns come from TMNF-C's
race layer with the map's real trigger boxes (tmnfc_port/compat/route.py): finish times equal
the replays' on every map checked (73/73). Multilap maps fall back to counting waypoint block
cells (the finish then comes about 0.1 s early).
"""
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from . import protocol as P
from .plugin_server import PluginServer

WP_START, WP_FINISH, WP_CP, WP_NONE, WP_STARTFINISH = 0, 1, 2, 3, 4


def tmi_waypoint(name: str) -> int:
    if 'StartFinish' in name:
        return WP_STARTFINISH
    if 'Start' in name and 'Line' in name:
        return WP_START
    if 'Finish' in name:
        return WP_FINISH
    if 'Checkpoint' in name:
        return WP_CP
    return WP_NONE


def map_dirs():
    from .paths import DATA
    dirs = [DATA / 'maps']
    try:
        from .tmx import tracks_dir
        dirs.insert(0, tracks_dir())
    except Exception:                       # noqa: BLE001  (no game folder on a server)
        pass
    return dirs


class VirtualGame(PluginServer):
    def __init__(self, port: int, first_map: Optional[Path] = None):
        super().__init__(port)
        self.reconnect = True
        self.sim = None
        self.cur = None
        self.action = (0, 0, 0)          # set by the last ACTION / playback frame
        self.pending = (0, 0, 0)         # acts on the next physics tick (one-tick delay)
        self.cps = set()
        if first_map is not None:
            self.load(Path(first_map))

    # -------------------------------------------------------------- map
    def map_key(self, file: str):
        for d in map_dirs():
            p = Path(d) / file
            if p.exists():
                return p
        return None

    def load(self, path: Path):
        from .replaybuild import map_blocks
        from .tmnfc_sim import CarSim
        uid, names, xyz, dirs = map_blocks(path)
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        blocks = [{'name': n, 'x': int(p[0]), 'y': int(p[1]), 'z': int(p[2]), 'dir': int(d),
                   'waypoint': tmi_waypoint(n)} for n, p, d in zip(names, xyz, dirs)]
        if self.sim is not None:
            self.sim.close()
        self.sim = CarSim(Path(path), f'v_{sha[:16]}', n=1)
        cps = [(b['x'], b['y'], b['z']) for b in blocks if b['waypoint'] == WP_CP]
        fins = [(b['x'], b['y'], b['z']) for b in blocks if b['waypoint'] in (WP_FINISH, WP_STARTFINISH)]
        self.cur = {'uid': uid, 'name': Path(path).name.split('.')[0], 'blocks': blocks, 'path': Path(path),
                    'cps': cps, 'fins': fins}
        self.reset_race()

    def map_info(self):
        if self.cur is None:
            return '', '', []
        return self.cur['uid'], self.cur['name'], self.cur['blocks']

    # -------------------------------------------------------------- race
    def reset_race(self):
        self.rt = self.COUNTDOWN
        self.finished = False
        self.cps = set()
        self.action = self.pending = (0, 0, 0)
        if self.sim is not None:
            self.sim.reset()

    def set_inputs(self, steer, gas, bits):
        if self.sim is not None and self.sim.full_route:
            self.action = (int(steer), int(gas), int(bits))       # the race layer respawns
        else:
            self.action = (int(steer), int(gas), int(bits) & ~P.RESPAWN)

    def respawn(self):
        if self.sim is None or not self.sim.full_route:
            self.reset_race()            # no checkpoint spawns known: a respawn restarts

    def bench_inputs(self):
        self.action = (0, 0, P.UP)

    def physics_tick(self):
        if self.sim is None or self.finished:
            return
        if self.mode == P.MODE_IDLE and self.rt >= 0:
            return                       # nobody drives: the race waits (the game would idle)
        if self.rt < 10:                 # countdown, and the spawn tick the world already did
            self.rt += 10
            if self.rt == 10:
                self.pending = self.action
            return
        restarted = self.sim.step([self.pending])
        self.pending = self.action
        self.rt += 10
        if restarted:                    # a respawn before any checkpoint restarts the race
            self.reset_race()
            return
        if self.sim.full_route:          # the map's real triggers: exact checkpoints and finish
            race = self.sim.race(0)
            self.cps = set(range(race['checkpoints']))
            if race['finished']:
                self.finished = True
                self.rt = race['finish_ms']
        else:
            self.waypoints()

    def cell_hit(self, pos, cells) -> list:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        cx, cz = int(np.floor(x / 32)), int(np.floor(z / 32))
        return [c for c in cells if c[0] == cx and c[2] == cz and c[1] * 8 - 4 <= y <= c[1] * 8 + 24]

    def waypoints(self):
        pos = self.sim.state(0)['pos']
        for c in self.cell_hit(pos, self.cur['cps']):
            self.cps.add(c)
        if len(self.cps) >= len(set(self.cur['cps'])) and self.cell_hit(pos, self.cur['fins']):
            self.finished = True

    def save_state(self):
        return (self.sim.capture(0), self.rt, self.finished, set(self.cps), self.pending, self.action)

    def load_state(self, state):
        snap, self.rt, self.finished, cps, self.pending, self.action = state
        self.cps = set(cps)
        self.sim.restore(0, snap)

    def step_values(self):
        st = self.sim.state(0)
        flags = (P.F_FINISHED if self.finished else 0) | (P.F_SIM_ONLY if self.sim_only else 0)
        vals = [self.rt, flags, len(self.cps), *st['pos'], *st['rot'], *st['vel'], *st['ang_vel']]
        for k in range(4):
            bits = int(st['wheel_contact'][k]) | (int(st['wheel_sliding'][k]) << 1) | \
                (int(st['wheel_material'][k]) << 8)
            vals += [float(st['wheel_damper'][k]), bits]
        steer, gas, bits = self.action
        vals += [st['gear'], st['rpm'], int(np.linalg.norm(st['vel']) * 3.6), steer, gas, bits]
        return vals

    def idle_sleep(self):
        import time
        time.sleep(0.002)


def main():
    ap = argparse.ArgumentParser(description='one virtual game instance (TMNF-C)')
    ap.add_argument('--port', type=int, default=8600)
    ap.add_argument('--map', default='', help='map file to start on (else: the first `map` command)')
    a = ap.parse_args()
    g = VirtualGame(a.port, Path(a.map) if a.map else None)
    print(f'virtual game on port {a.port}' + (f", map {g.cur['name']}" if g.cur else ''), flush=True)
    g.run()


if __name__ == '__main__':
    main()
