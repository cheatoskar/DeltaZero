"""Socket client for the TMDriver plugin (the plugin listens, Python connects)."""
import socket
import threading
import struct
import time
from dataclasses import dataclass, field
from typing import List

from . import protocol as P


@dataclass
class MapBlock:
    name: str
    x: int
    y: int
    z: int
    dir: int
    waypoint: int   # TM::WayPointType: 0 start, 1 finish, 2 checkpoint, 3 none, 4 start/finish


@dataclass
class MapInfo:
    uid: str
    name: str
    author: str
    blocks: List[MapBlock] = field(default_factory=list)


class Link:
    """Reads plugin messages and queues commands; `flush()` sends the queue at once."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.rfile = sock.makefile('rb', buffering=1 << 16)
        self.out = bytearray()
        # STEPs received minus ACTIONs sent. The plugin waits for exactly one ACTION per STEP;
        # an extra one is taken as the NEXT tick's answer and shifts every later input by a
        # tick. strict=True (GameSession) turns an ACTION without a STEP into an error.
        self.owed = 0
        self.strict = False
        self._owed_lock = threading.Lock()   # the reader thread counts, the caller answers
        self._send_lock = threading.Lock()   # flush vs. the wait-ping thread

    @classmethod
    def connect(cls, host: str = P.HOST, port: int = P.PORT, wait_s: float = 0.0,
                log=print) -> 'Link':
        """Connect, retrying for up to `wait_s` seconds (the game may still be loading)."""
        deadline = time.monotonic() + wait_s
        announced = False
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=5.0)
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return cls(sock)
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                if not announced:
                    log(f'waiting for the TMDriver plugin on {host}:{port} ...')
                    announced = True
                time.sleep(1.0)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------------ reading

    def _read(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if data is None or len(data) < n:
            raise ConnectionError('plugin closed the connection')
        return data

    def _int(self) -> int:
        return struct.unpack('<i', self._read(4))[0]

    def _float(self) -> float:
        return struct.unpack('<f', self._read(4))[0]

    def _str(self) -> str:
        n = self._int()
        return self._read(n).decode('utf-8', errors='replace') if n > 0 else ''

    def read_message(self):
        """Returns (type, payload). Payload: Step, MapInfo, or a tuple of values."""
        t = self._int()
        if t == P.P_STEP:
            st = P.Step.unpack(self._read(P.STEP.size))
            with self._owed_lock:
                self.owed += 1
            return t, st
        if t == P.P_TREC:
            return t, P.Step.unpack(self._read(P.STEP.size))
        if t == P.P_TEND:
            return t, (self._int(), self._int())
        if t == P.P_JOB:
            return t, tuple(self._int() for _ in range(5)) + (self._str(),)
        if t == P.P_PING:
            return t, ()
        if t == P.P_HELLO:
            return t, (self._int(),)
        if t == P.P_UI:
            return t, (self._int(), self._float())
        if t == P.P_BENCH:
            return t, (self._int(), self._int(), self._int())
        if t == P.P_MAP:
            info = MapInfo(self._str(), self._str(), self._str())
            for _ in range(self._int()):
                name = self._str()
                x, y, z, d, w = struct.unpack('<5i', self._read(20))
                info.blocks.append(MapBlock(name, x, y, z, d, w))
            return t, info
        raise ConnectionError(f'unknown message type {t} from plugin (protocol mismatch?)')

    # ------------------------------------------------------------------ commands

    def action(self, steer: int = 0, gas: int = 0, bits: int = 0):
        """Queue the ACTION that ends the plugin's wait, and send everything queued."""
        with self._owed_lock:
            if self.owed <= 0 and self.strict:
                raise RuntimeError('ACTION without a pending STEP: the plugin would apply it one tick late')
            self.owed = max(self.owed - 1, 0)
        self.out += P.action_bytes(steer, gas, bits)
        self.flush()

    def speed(self, s: float):
        self.out += P.float_cmd(P.C_SPEED, s)

    def restart(self):
        self.out += struct.pack('<i', P.C_RESTART)

    def status(self, text: str):
        self.out += P.str_cmd(P.C_STATUS, text)

    def sim_only(self, on: bool):
        self.out += P.int_cmd(P.C_SIMONLY, 1 if on else 0)

    def save(self, slot: int):
        self.out += P.int_cmd(P.C_SAVE, slot)

    def rewind(self, slot: int):
        self.out += P.int_cmd(P.C_REWIND, slot)

    def mode(self, m: int):
        self.out += P.int_cmd(P.C_MODE, m)

    def bench(self, ticks: int):
        self.out += P.int_cmd(P.C_BENCH, ticks)

    def play(self, inputs):
        """Batch playback: queue before the ACTION of a STEP. From the next tick the plugin
        applies these inputs itself (index = race time / 10) and streams P_TREC states, then
        P_TEND, and hands the final tick back as an ordinary STEP."""
        self.out += P.play_cmd(inputs)

    def draw(self, points, size: float = 1.0):
        """Show points in the game as small TMInterface trigger boxes (empty list clears)."""
        self.out += P.draw_cmd(points, size)

    def hold(self, n: int):
        """The next ACTION holds for n ticks (protocol.C_HOLD); send it right before the ACTION."""
        self.out += P.int_cmd(P.C_HOLD, int(n))

    def execute(self, command: str):
        self.out += P.str_cmd(P.C_EXEC, command)

    def flush(self):
        if self.out:
            with self._send_lock:
                self.sock.sendall(self.out)
            self.out.clear()

    def wait_ping(self):
        """C_WAIT, sent at once (from GameSession.hold's thread): keeps the plugin waiting."""
        with self._send_lock:
            self.sock.sendall(struct.pack('<i', P.C_WAIT))
