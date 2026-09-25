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
import select
import socket
import struct
import threading
import time

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tmdriver import protocol as P  # noqa: E402

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


class FakeGame(threading.Thread):
    COUNTDOWN = -300

    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((P.HOST, port))
        self.server.listen(1)
        self.client = None
        self.buf = bytearray()
        self.maps = {'default': fake_map('FAKE_UID_1', 'Fake Course', 0)}
        self.cur = self.maps['default']
        self.course = self.cur['course']
        self.pending_exec = []   # a queue, like the plugin (a single slot lost commands)
        self.mode = P.MODE_IDLE
        self.status = ''
        self.drawn = []
        self.playing = self.play_armed = False
        self.play_inputs = []
        self.play_last = -10 ** 9
        self.ui_queue = []
        self.job_queue = []      # tool buttons (P_JOB)
        self.reconnect = False   # True: a closed client is dropped and a new one accepted (like the plugin)
        self.accepts = 0
        self.pending_restart = False
        self.pending_release = False
        self.sim_only = False
        self.speed = 1.0
        self.slots = {}
        self.bench_left = 0
        self.sent_uid = ''
        self.finish_sent = False
        self.last_rt = -10 ** 9
        self.stop_flag = False
        self.reset_race()

    # -------------------------------------------------------------- race
    def load(self, key):
        self.cur = self.maps[key]
        self.course = self.cur['course']
        self.reset_race()
        self.finish_sent = False
        self.last_rt = -10 ** 9
        self.sent_uid = ''

    def reset_race(self):
        self.car = Car(self.cur['h0'])
        self.inp = Inputs()
        self.rt = self.COUNTDOWN
        self.finished = False
        self.human_idx = 0

    def human_inputs(self):
        i, self.human_idx = human_policy(self.car, self.course, self.human_idx)
        return i

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

    # -------------------------------------------------------------- socket helpers
    def _fill(self, n, deadline):
        while len(self.buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            r, _, _ = select.select([self.client], [], [], min(left, 0.05))
            if r:
                data = self.client.recv(65536)
                if not data:
                    return False
                self.buf += data
        return True

    def _take(self, fmt):
        n = struct.calcsize(fmt)
        v = struct.unpack(fmt, self.buf[:n])
        del self.buf[:n]
        return v

    def _available(self):
        r, _, _ = select.select([self.client], [], [], 0)
        if r:
            data = self.client.recv(65536)
            if not data:
                raise ConnectionError
            self.buf += data
        return len(self.buf)

    def send(self, data: bytes):
        self.client.sendall(data)

    def read_message(self, in_step: bool, deadline):
        if not self._fill(4, deadline):
            return -1
        (t,) = self._take('<i')
        sizes = {P.C_ACTION: 12, P.C_SPEED: 4, P.C_RESTART: 0, P.C_SIMONLY: 4, P.C_SAVE: 4,
                 P.C_REWIND: 4, P.C_MODE: 4, P.C_BENCH: 4}
        if t == P.C_EXEC:
            if not self._fill(4, deadline):
                return -1
            (n,) = self._take('<i')
            if not self._fill(n, deadline):
                return -1
            self.pending_exec.append(bytes(self.buf[:n]).decode())
            del self.buf[:n]
            return t
        if t == P.C_PLAY:
            if not self._fill(4, deadline):
                return -1
            (n,) = self._take('<i')
            if not self._fill(n * 12, deadline):
                return -1
            self.play_inputs = [self._take('<iii') for _ in range(n)]
            self.play_armed = True
            return t
        if t == P.C_DRAW:
            if not self._fill(4, deadline):
                return -1
            (n,) = self._take('<i')
            if not self._fill(n * 12 + 4, deadline):
                return -1
            self.drawn = [self._take('<fff') for _ in range(n)]
            self._take('<f')
            return t
        if t == P.C_STATUS:
            if not self._fill(4, deadline):
                return -1
            (n,) = self._take('<i')
            if not self._fill(n, deadline):
                return -1
            self.status = bytes(self.buf[:n]).decode()
            del self.buf[:n]
            return t
        if t not in sizes or not self._fill(sizes[t], deadline):
            raise RuntimeError(f'bad message {t}')
        if t == P.C_ACTION:
            steer, gas, bits = self._take('<iii')
            if in_step:
                self.inp.up, self.inp.down = bool(bits & P.UP), bool(bits & P.DOWN)
                self.inp.left, self.inp.right = bool(bits & P.LEFT), bool(bits & P.RIGHT)
                self.inp.steer = steer if bits & P.STEER_ANALOG else 0
                self.inp.gas = gas if bits & P.GAS_ANALOG else 0
                if bits & P.RESPAWN:
                    self.reset_race()   # no checkpoints in the fake course: respawn = restart
        elif t == P.C_SPEED:
            (self.speed,) = self._take('<f')
        elif t == P.C_RESTART:
            if in_step:
                self.reset_race()
                self.finish_sent = False
            else:
                self.pending_restart = True
        elif t == P.C_SIMONLY:
            (on,) = self._take('<i')
            self.sim_only = bool(on)
        elif t == P.C_SAVE:
            (slot,) = self._take('<i')
            assert in_step
            self.slots[slot] = (self.car.copy(), self.inp.copy(), self.rt, self.finished)
        elif t == P.C_REWIND:
            (slot,) = self._take('<i')
            assert in_step and slot in self.slots
            car, inp, rt, fin = self.slots[slot]
            self.car, self.inp, self.rt, self.finished = car.copy(), inp.copy(), rt, fin
            self.finish_sent = False
            self.last_rt = -10 ** 9
        elif t == P.C_MODE:
            (m,) = self._take('<i')
            if m == P.MODE_IDLE and self.mode != P.MODE_IDLE:
                self.pending_release = True
            self.mode = m
        elif t == P.C_BENCH:
            (n,) = self._take('<i')
            self.bench_left = self.bench_ticks = n
            self.bench_start = time.monotonic()
        return t

    # -------------------------------------------------------------- plugin mirror
    def send_step(self, msg=P.P_STEP):
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
        self.send(struct.pack('<i', msg) + P.STEP.pack(*vals))

    def send_map(self):
        def s(text):
            b = text.encode()
            return struct.pack('<i', len(b)) + b
        m = self.cur
        out = struct.pack('<i', P.P_MAP) + s(m['uid']) + s(m['name']) + s('tester') + struct.pack('<i', len(m['blocks']))
        for b in m['blocks']:
            out += s(b['name']) + struct.pack('<5i', b['x'], b['y'], b['z'], b['dir'], b['waypoint'])
        self.send(out)
        self.sent_uid = m['uid']

    def on_run_step(self):
        if self.pending_release:
            self.inp = Inputs()
            self.pending_release = False
        if self.client is None:
            return
        if self.pending_restart:
            self.pending_restart = False
            self.reset_race()
            self.finish_sent = False
            return
        t = self.rt
        if t < self.last_rt:
            self.finish_sent = False
            self.sent_uid = ''
        self.last_rt = t
        if self.sent_uid != self.cur['uid']:
            self.send_map()
        if self.bench_left > 0:
            self.inp.up = True
            self.bench_left -= 1
            if self.bench_left == 0:
                ms = int((time.monotonic() - self.bench_start) * 1000)
                self.send(struct.pack('<iiii', P.P_BENCH, self.bench_ticks, ms, self.rt))
            return
        if self.playing and not self.run_playback(t):
            return
        if self.mode == P.MODE_IDLE or t < -10 or self.finish_sent:
            return
        if self.mode == P.MODE_RECORD and t >= 0:
            self.inp = self.human_inputs()     # the human drives; reported one tick later
        self.send_step()
        if self.finished:
            self.finish_sent = True
        if self.mode == P.MODE_RECORD:
            return
        deadline = time.monotonic() + 5.0
        while True:
            typ = self.read_message(True, deadline)
            if typ == P.C_ACTION:
                break
            if typ == -1:
                raise RuntimeError('python timed out')
        if self.play_armed:
            self.play_armed, self.playing, self.play_last = False, True, -10 ** 9

    def run_playback(self, t):
        """Mirror of RunPlayback in TMDriver.as: True when playback ended."""
        reason = -1
        if t == self.play_last:
            reason = 2
        elif self.finished:
            reason = 1
        elif t < 0 or t // 10 >= len(self.play_inputs):
            reason = 0
        self.play_last = t
        if reason >= 0:
            self.playing = False
            self.send(struct.pack('<iii', P.P_TEND, reason, t))
            return True
        self.send_step(P.P_TREC)
        steer, gas, bits = self.play_inputs[t // 10]
        self.inp.up, self.inp.down = bool(bits & P.UP), bool(bits & P.DOWN)
        self.inp.left, self.inp.right = bool(bits & P.LEFT), bool(bits & P.RIGHT)
        self.inp.steer = steer if bits & P.STEER_ANALOG else 0
        self.inp.gas = gas if bits & P.GAS_ANALOG else 0
        return False

    def accept_new(self):
        """Like the plugin's Render(): a new connection replaces the current one."""
        r, _, _ = select.select([self.server], [], [], 0)
        if not r:
            return
        c, _ = self.server.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.client is not None:
            self.client.close()
        self.client, self.buf, self.sent_uid = c, bytearray(), ''
        self.accepts += 1
        self.send(struct.pack('<ii', P.P_HELLO, P.PROTOCOL))

    def press_job(self, job, track_id=0, rounds=20, hours=3, flags=1):
        self.job_queue.append((job, track_id, rounds, hours, flags))

    def render(self):
        if self.reconnect:
            self.accept_new()
            if self.client is None:
                return
        while self.job_queue:
            self.send(struct.pack('<6i', P.P_JOB, *self.job_queue.pop(0)))
        while self.ui_queue:
            m = self.ui_queue.pop(0)
            if m != P.MODE_IDLE:
                self.pending_restart = True
            self.mode = m
            self.send(struct.pack('<iif', P.P_UI, m, self.ui_speed))
        while self._available() >= 4:
            self.read_message(False, time.monotonic() + 1.0)
        cmds, self.pending_exec = self.pending_exec, []
        for cmd in cmds:
            if cmd.startswith('map '):
                f = cmd[4:].strip().strip('"').replace('\\', '/').split('/')[-1]
                if f in self.maps:
                    self.load(f)

    def press(self, mode, speed=1.0):
        self.ui_speed = speed
        self.ui_queue.append(mode)

    def run(self):
        self.client, _ = self.server.accept()
        self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.accepts = 1
        self.send(struct.pack('<ii', P.P_HELLO, P.PROTOCOL))
        while not self.stop_flag:
            try:
                self.render()
                self.on_run_step()
                self.physics_tick()
                if self.mode in (P.MODE_IDLE,) or self.client is None:
                    time.sleep(0.0005)
            except (ConnectionError, OSError):
                if not self.reconnect:
                    return
                self.client, self.buf = None, bytearray()
