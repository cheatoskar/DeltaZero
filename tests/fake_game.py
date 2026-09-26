"""A stand-in for TMNF + the TMDriver plugin, speaking the same wire protocol.

It exists so the Python side can be tested end to end without the game. It mirrors the
control flow of plugin/TMDriver/TMDriver.as (modes, pending restarts, blocking wait for
ACTION, save/rewind, bench) and simulates a simple deterministic car on a fixed course.

Deliberately awkward conventions, so the Python side has to measure them instead of
assuming them:
  * the forward axis is a *column* of the rotation matrix whose rows are sent;
  * positive analog steer turns the car towards *decreasing* heading;
  * the input reported in a STEP is the one applied on the tick that follows it
    (so the self test's shift-0 alignment must come out exact, shift-1 must not).
"""
import math
import struct
import time

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tmdriver import protocol as P  # noqa: E402
from tmdriver.plugin_server import PluginServer  # noqa: E402

DT = 0.01


START = (500.0, 500.0)


def make_course(h0: float = 0.0):
    """Dense polyline: straight, left arc, straight, right hairpin, straight; the first
    straight points along heading h0 (0 = +z)."""
    pts = []
    x, z, h = START[0], START[1], h0

    def straight(length):
        nonlocal x, z
        for _ in range(int(length)):
            x += math.sin(h)
            z += math.cos(h)
            pts.append((x, z))

    def arc(radius, angle):
        nonlocal x, z, h
        n = int(abs(angle) * radius)
        for _ in range(n):
            h += angle / n
            x += math.sin(h) * abs(angle) * radius / n
            z += math.cos(h) * abs(angle) * radius / n
            pts.append((x, z))

    pts.append((x, z))
    straight(120)
    arc(45, math.pi / 2)
    straight(80)
    arc(30, -math.radians(150))
    straight(150)
    return np.array(pts)


class Car:
    def __init__(self, h0: float = 0.0):
        self.x, self.z, self.h, self.v = START[0], START[1], h0, 0.0

    def copy(self):
        c = Car()
        c.x, c.z, c.h, c.v = self.x, self.z, self.h, self.v
        return c

    def tick(self, up, down, s):
        if up:
            a = 14.0 * (1.0 - self.v / 80.0)
        elif down:
            a = -30.0
        else:
            a = -2.0
        self.v = max(0.0, self.v + a * DT)
        omega = -s * 1.8 * min(1.0, self.v / 15.0)
        self.h += omega * DT
        self.x += self.v * math.sin(self.h) * DT
        self.z += self.v * math.cos(self.h) * DT

    def fwd(self):
        return np.array([math.sin(self.h), 0.0, math.cos(self.h)])


class Inputs:
    def __init__(self):
        self.up = self.down = self.left = self.right = False
        self.steer = 0
        self.gas = 0

    def copy(self):
        i = Inputs()
        i.__dict__.update(self.__dict__)
        return i

    def s(self):
        if self.steer != 0:
            return self.steer / P.STEER_MAX
        return (1.0 if self.right else 0.0) - (1.0 if self.left else 0.0)

    def bits(self):
        return (P.UP if self.up else 0) | (P.DOWN if self.down else 0) | \
               (P.LEFT if self.left else 0) | (P.RIGHT if self.right else 0)


def human_policy(car, course, idx, speed_scale=1.0, analog=False):
    """A driver who knows the course: keyboard (dead band) or pad (proportional)."""
    p = np.array([car.x, car.z])
    lo = max(0, idx - 5)
    d = np.linalg.norm(course[lo:lo + 80] - p, axis=1)
    idx = lo + int(np.argmin(d))
    tgt = course[min(idx + 12, len(course) - 1)] - p
    alpha = math.atan2(tgt[0], tgt[1]) - car.h
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    i = Inputs()
    if analog:
        # positive analog steer turns towards decreasing heading in this fake
        i.steer = int(np.clip(-alpha * 3.0, -1.0, 1.0) * P.STEER_MAX)
    else:
        i.left, i.right = alpha > 0.04, alpha < -0.04
    ahead = course[idx:idx + 40]
    if len(ahead) > 2:
        hd = np.unwrap(np.arctan2(np.diff(ahead[:, 0]), np.diff(ahead[:, 1])))
        curvy = abs(hd[-1] - hd[0]) > 0.35
    else:
        curvy = False
    target = (17.0 if curvy else 35.0) * speed_scale
    i.up = car.v < target
    i.down = car.v > target + 3.0
    return i, idx


def crossed_finish(course, x, z) -> bool:
    """A finish LINE (like a finish block), not a circle: past the last point along the final
    direction, within 16 m sideways."""
    end, prev = course[-1], course[-6]
    d = (end - prev) / np.linalg.norm(end - prev)
    rel = np.array([x, z]) - end
    return float(rel @ d) >= 0.0 and abs(float(rel[0] * d[1] - rel[1] * d[0])) < 16.0


def dir_to_heading(d: int) -> float:
    """Fake convention, chosen to match the one observed in TMNF on lolsport: dir 1 -> -90 deg."""
    return -d * math.pi / 2


def course_blocks(course, d0: int):
    """Plugin-style block list for a course: start, road blocks every 32 m, finish, grass."""
    def anchor(x, z):
        return int(x // 32), 1, int(z // 32)

    blocks = []
    x, y, z = anchor(*course[0])
    blocks.append({'name': 'StadiumRoadMainStartLine', 'x': x, 'y': y, 'z': z, 'dir': d0, 'waypoint': 0})
    for k in range(32, len(course) - 32, 32):
        h = math.atan2(course[k + 1][0] - course[k][0], course[k + 1][1] - course[k][1])
        d = int(round(-h / (math.pi / 2))) % 4
        x, y, z = anchor(*course[k])
        blocks.append({'name': 'StadiumRoadMain', 'x': x, 'y': y, 'z': z, 'dir': d, 'waypoint': 3})
    x, y, z = anchor(*course[-1])
    blocks.append({'name': 'StadiumRoadMainFinishLine', 'x': x, 'y': y, 'z': z, 'dir': 0, 'waypoint': 1})
    for gx in range(0, 32, 8):
        blocks.append({'name': 'StadiumGrass', 'x': gx, 'y': 0, 'z': 0, 'dir': 0, 'waypoint': 3})
    return blocks


def fake_map(uid: str, name: str, d0: int) -> dict:
    h0 = dir_to_heading(d0)
    course = make_course(h0)
    return {'uid': uid, 'name': name, 'dir': d0, 'h0': h0, 'course': course, 'blocks': course_blocks(course, d0)}


def synth_replay(m: dict, speed_scale: float = 1.0, analog: bool = False, limit_ms: int = 120000):
    """Drive the course offline with the game's physics and write it down the way a TMNF
    replay stores it: input events on the 10 ms grid (an input decided at race time t is
    an event at t + 10, the alignment the self test measured), analog steer with the
    replay's sign (negated; calibration steer_sign -1), ghost positions every 100 ms."""
    car, course, idx = Car(m['h0']), m['course'], 0
    t, prev = 0, Inputs()
    events = [(0, '_FakeIsRaceRunning', 1)]
    ghost_t, ghost_pos = [], []
    while t < limit_ms:
        if t % 100 == 0:
            ghost_t.append(t)
            ghost_pos.append((car.x, 10.0, car.z))
        i, idx = human_policy(car, course, idx, speed_scale, analog)
        for name, a, b in (('Accelerate', i.up, prev.up), ('Brake', i.down, prev.down),
                           ('SteerLeft', i.left, prev.left), ('SteerRight', i.right, prev.right)):
            if a != b:
                events.append((t + 10, name, 1 if a else 0))
        if i.steer != prev.steer:
            events.append((t + 10, 'Steer', (-i.steer) & 0xFFFFFF))
        prev = i
        car.tick(i.up, i.down, i.s())
        t += 10
        if crossed_finish(course, car.x, car.z):
            break
    return {'race_time_ms': t, 'events': events, 'ghost_t': np.array(ghost_t), 'ghost_pos': np.array(ghost_pos)}


class FakeGame(PluginServer):
    """The plugin protocol (tmdriver.plugin_server) with the toy car above as its physics."""

    def __init__(self, port: int):
        super().__init__(port)
        self.maps = {'default': fake_map('FAKE_UID_1', 'Fake Course', 0)}
        self.cur = self.maps['default']
        self.course = self.cur['course']
        self.reset_race()

    # -------------------------------------------------------------- physics
    def load(self, key):
        self.cur = self.maps[key]
        self.course = self.cur['course']
        self.reset_race()
        self.after_load()

    def map_key(self, file):
        return file if file in self.maps else None

    def map_info(self):
        return self.cur['uid'], self.cur['name'], self.cur['blocks']

    def reset_race(self):
        self.car = Car(self.cur['h0'])
        self.inp = Inputs()
        self.rt = self.COUNTDOWN
        self.finished = False
        self.human_idx = 0

    def physics_tick(self):
        if self.finished:
            return
        if self.rt >= 0:
            self.car.tick(self.inp.up, self.inp.down, self.inp.s())
            self.rt += 10
            if crossed_finish(self.course, self.car.x, self.car.z):
                self.finished = True
        else:
            self.rt += 10

    def set_inputs(self, steer, gas, bits):
        self.inp.up, self.inp.down = bool(bits & P.UP), bool(bits & P.DOWN)
        self.inp.left, self.inp.right = bool(bits & P.LEFT), bool(bits & P.RIGHT)
        self.inp.steer = steer if bits & P.STEER_ANALOG else 0
        self.inp.gas = gas if bits & P.GAS_ANALOG else 0

    def release_inputs(self):
        self.inp = Inputs()

    def bench_inputs(self):
        self.inp.up = True

    def record_inputs(self):
        self.inp, self.human_idx = human_policy(self.car, self.course, self.human_idx)

    def save_state(self):
        return (self.car.copy(), self.inp.copy(), self.rt, self.finished)

    def load_state(self, state):
        car, inp, rt, fin = state
        self.car, self.inp, self.rt, self.finished = car.copy(), inp.copy(), rt, fin

    def step_values(self):
        c = self.car
        fwd = c.fwd()
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd)
        m = np.stack([right, up, fwd], axis=1)       # forward is column 2
        flags = (P.F_FINISHED if self.finished else 0) | (P.F_SIM_ONLY if self.sim_only else 0)
        vals = [self.rt, flags, 0, c.x, 10.0, c.z, *m[0], *m[1], *m[2],
                *(c.v * fwd), 0.0, 0.0, 0.0]
        for _ in range(4):
            vals += [0.0, 1]
        vals += [3, 5000.0, int(c.v * 3.6), self.inp.steer, self.inp.gas, self.inp.bits()]
        return vals
