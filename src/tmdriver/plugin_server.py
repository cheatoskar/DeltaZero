"""The TMDriver plugin's side of the wire protocol, in Python, with the physics left to a
subclass. Two physics exist:

  * tests/fake_game.py: a toy car on a fixed course, for testing the Python side quickly;
  * virtual_game.py: TMNF-C, the game's physics without the game (training without a client).

It mirrors the control flow of plugin/TMDriver/TMDriver.as: modes, pending restarts, the
blocking wait for ACTION, save/rewind, C_HOLD, C_PLAY batch playback, bench, map loading
through the `map` console command, and a new connection replacing the old one.

A subclass provides:
  load(key) / reset_race()          the map and the car at its start (race time COUNTDOWN)
  physics_tick()                    one 10 ms tick with self.inp (race time += 10)
  step_values(): list               the STEP payload (see protocol.STEP)
  save_state() / load_state(s)      C_SAVE / C_REWIND
  set_inputs(steer, gas, bits)      what an ACTION or a playback frame sets
  map_info() -> (uid, name, blocks) for P_MAP
  map_key(file) -> key or None      a `map "TMDriver\\file"` command
"""
import select
import socket
import struct
import threading
import time

from . import protocol as P


class PluginServer(threading.Thread):
    COUNTDOWN = -300

    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((P.HOST, port))
        self.server.listen(1)
        self.port = port
        self.client = None
        self.buf = bytearray()
        self.pending_exec = []   # a queue, like the plugin (a single slot lost commands)
        self.vars = {}           # TMInterface variables set through 'set NAME VALUE'
        self.var_log = []        # every (name, value) set, in order
        self.mode = P.MODE_IDLE
        self.status = ''
        self.drawn = []
        self.playing = self.play_armed = False
        self.hold_armed = self.hold_left = 0     # C_HOLD (mirror of the plugin)
        self.play_inputs = []
        self.play_last = -10 ** 9
        self.ui_queue = []
        self.ui_speed = 1.0
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
        self.rt = self.COUNTDOWN
        self.finished = False

    # -------------------------------------------------------------- physics (subclass)
    def load(self, key):
        raise NotImplementedError

    def reset_race(self):
        raise NotImplementedError

    def physics_tick(self):
        raise NotImplementedError

    def step_values(self) -> list:
        raise NotImplementedError

    def save_state(self):
        raise NotImplementedError

    def load_state(self, state):
        raise NotImplementedError

    def set_inputs(self, steer: int, gas: int, bits: int):
        raise NotImplementedError

    def release_inputs(self):
        self.set_inputs(0, 0, 0)

    def respawn(self):
        self.reset_race()      # no checkpoint spawns modelled: a respawn restarts

    def record_inputs(self):
        """MODE_RECORD: the human drives (only the fake game has one)."""

    def bench_inputs(self):
        self.set_inputs(0, 0, P.UP)

    def map_info(self):
        raise NotImplementedError

    def map_key(self, file: str):
        raise NotImplementedError

    def after_load(self):
        self.finish_sent = False
        self.last_rt = -10 ** 9
        self.sent_uid = ''

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
                 P.C_REWIND: 4, P.C_MODE: 4, P.C_BENCH: 4, P.C_WAIT: 0, P.C_HOLD: 4}
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
                self.set_inputs(steer, gas, bits)
                if bits & P.RESPAWN:
                    self.respawn()
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
            self.slots[slot] = self.save_state()
        elif t == P.C_REWIND:
            (slot,) = self._take('<i')
            assert in_step and slot in self.slots
            self.load_state(self.slots[slot])
            self.finish_sent = False
            self.last_rt = -10 ** 9
        elif t == P.C_MODE:
            (m,) = self._take('<i')
            if m == P.MODE_IDLE and self.mode != P.MODE_IDLE:
                self.pending_release = True
            self.mode = m
        elif t == P.C_HOLD:
            (self.hold_armed,) = self._take('<i')
        elif t == P.C_BENCH:
            (n,) = self._take('<i')
            self.bench_left = self.bench_ticks = n
            self.bench_start = time.monotonic()
        return t

    # -------------------------------------------------------------- plugin mirror
    def send_step(self, msg=P.P_STEP):
        self.send(struct.pack('<i', msg) + P.STEP.pack(*self.step_values()))

    def send_map(self):
        def s(text):
            b = text.encode()
            return struct.pack('<i', len(b)) + b
        uid, name, blocks = self.map_info()
        out = struct.pack('<i', P.P_MAP) + s(uid) + s(name) + s('tester') + struct.pack('<i', len(blocks))
        for b in blocks:
            out += s(b['name']) + struct.pack('<5i', b['x'], b['y'], b['z'], b['dir'], b['waypoint'])
        self.send(out)
        self.sent_uid = uid

    def on_run_step(self):
        if self.pending_release:
            self.release_inputs()
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
        if self.sent_uid != self.map_info()[0]:
            self.send_map()
        if self.bench_left > 0:
            self.bench_inputs()
            self.bench_left -= 1
            if self.bench_left == 0:
                ms = int((time.monotonic() - self.bench_start) * 1000)
                self.send(struct.pack('<iiii', P.P_BENCH, self.bench_ticks, ms, self.rt))
            return
        if self.playing and not self.run_playback(t):
            return
        if self.hold_left > 0:            # a held ACTION: the inputs stay, no STEP
            if self.mode != P.MODE_IDLE and t >= 0 and not self.finished:
                self.hold_left -= 1
                return
            self.hold_left = 0
        if self.mode == P.MODE_IDLE or t < -10 or self.finish_sent:
            return
        if self.mode == P.MODE_RECORD and t >= 0:
            self.record_inputs()          # the human drives; reported one tick later
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
            if typ == P.C_WAIT:
                deadline = time.monotonic() + 5.0     # like the plugin: Python is busy, keep waiting
            if typ == -1:
                raise RuntimeError('python timed out')
        self.hold_left = self.hold_armed - 1 if self.hold_armed > 1 else 0
        self.hold_armed = 0
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
        self.set_inputs(steer, gas, bits)
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

    def press_job(self, job, track_id=0, rounds=20, hours=3, flags=1, game_dir=''):
        self.job_queue.append((job, track_id, rounds, hours, flags, game_dir))

    def render(self):
        if self.reconnect:
            self.accept_new()
            if self.client is None:
                return
        while self.job_queue:
            *ints, gdir = self.job_queue.pop(0)
            raw = gdir.encode()
            self.send(struct.pack('<6i', P.P_JOB, *ints) + struct.pack('<i', len(raw)) + raw)
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
            if cmd.startswith('set '):
                parts = cmd.split(None, 2)
                if len(parts) == 3:
                    self.vars[parts[1]] = parts[2]
                    self.var_log.append((parts[1], parts[2]))
            if cmd.startswith('map '):
                f = cmd[4:].strip().strip('"').replace('\\', '/').split('/')[-1]
                key = self.map_key(f)
                if key is not None:
                    self.load(key)
                    self.after_load()

    def press(self, mode, speed=1.0):
        self.ui_speed = speed
        self.ui_queue.append(mode)

    def idle_sleep(self):
        time.sleep(0.0005)

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
                    self.idle_sleep()
            except (ConnectionError, OSError):
                if not self.reconnect:
                    return
                self.client, self.buf = None, bytearray()
