"""Wire protocol shared with plugin/TMDriver/TMDriver.as. Change both files together.

Every message starts with an int32 type. All values are little endian. A string is an
int32 byte length followed by the bytes.
"""
import struct
from dataclasses import dataclass

import numpy as np

PROTOCOL = 12
HOST = '127.0.0.1'
PORT = 8478

# plugin -> python
P_HELLO = 1   # int protocol
P_MAP = 2     # str uid, str name, str author, int n, n x (str name, int x, y, z, dir, waypoint)
P_STEP = 3    # STEP struct below
P_UI = 4      # int mode, float speed        (a window button was pressed)
P_BENCH = 5   # int ticks, int elapsed_ms, int race_time
P_PING = 6    # -  (heartbeat while idle; ignore)
P_TREC = 7    # STEP struct: one tick of batch playback (no answer)
P_JOB = 9     # int job, int TMX id (0 = current map), int rounds, int minutes, int flags,
              # str game folder (TMI variable tmdriver_game_folder; '' = detect)   (tool button)
JOB_DRIVE, JOB_TRAIN, JOB_RESIM, JOB_SHOW, JOB_LAUNCH = 1, 2, 3, 4, 5   # LAUNCH: rounds = helpers   # P_JOB: ..., int minutes, ...
JOB_GPU, JOB_PER_TICK, JOB_LINE, JOB_SHOW_BEST = 1, 2, 4, 8   # P_JOB flags
P_TEND = 8    # int reason (0 end of inputs, 1 finish, 2 frozen race time), int race time

# python -> plugin
C_ACTION = 10   # int steer, int gas, int bits    (ends the plugin's wait for this tick)
C_SPEED = 11    # float
C_RESTART = 12  # -
C_STATUS = 13   # str
C_SIMONLY = 14  # int 0/1
C_SAVE = 15     # int slot
C_REWIND = 16   # int slot
C_MODE = 17     # int mode
C_BENCH = 18    # int ticks   (plugin runs them with gas held, without talking to Python)
C_EXEC = 19     # str console command, executed in the plugin's Render() (e.g. 'map <file>')
C_DRAW = 20     # int n, n x (float x, y, z), float size: path as TMInterface trigger boxes (n = 0 clears)
MAX_DRAW = 600
C_WAIT = 22     # (during a STEP) Python is busy: the plugin extends its timeout, the game stays frozen
C_HOLD = 23     # int n (during a STEP, before the ACTION): the plugin re-applies that ACTION on the next
                # n - 1 ticks without sending STEPs (stops early at the finish or a restart)
C_PLAY = 21     # int n, n x (int steer, int gas, int bits) for race times 0, 10, ...; send before the ACTION.
                # The plugin plays them from the next tick (P_TREC each), then P_TEND + a normal STEP
MAX_PLAY = 60000

MODE_IDLE, MODE_RECORD, MODE_DRIVE, MODE_TEST = 0, 1, 2, 3
MODE_NAMES = {MODE_IDLE: 'idle', MODE_RECORD: 'record', MODE_DRIVE: 'drive', MODE_TEST: 'test'}

# ACTION bits
UP, DOWN, LEFT, RIGHT, STEER_ANALOG, GAS_ANALOG, RESPAWN = 1, 2, 4, 8, 16, 32, 64
STEER_MAX = 65536

# STEP flags
F_FINISHED, F_SIM_ONLY, F_AFTER_END = 1, 2, 4

# race_time, flags, checkpoints | pos, rot rows x/y/z, linear speed, angular speed |
# 4 wheels (damper, bits: contact | sliding << 1 | material << 8) in order FL, FR, BR, BL |
# gear, rpm, display speed, input steer, input gas, input bits
STEP = struct.Struct('<iii18f' + 'fi' * 4 + 'ifiiii')
assert STEP.size == 140


@dataclass
class Step:
    race_time: int
    flags: int
    checkpoints: int
    pos: np.ndarray        # (3,) world metres, y up
    rot: np.ndarray        # (9,) the three mat3 vectors x, y, z as sent
    vel: np.ndarray        # (3,) m/s, world frame
    ang_vel: np.ndarray    # (3,)
    wheel_damper: np.ndarray    # (4,)
    wheel_contact: np.ndarray   # (4,) bool
    wheel_sliding: np.ndarray   # (4,) bool
    wheel_material: np.ndarray  # (4,) TM::PlugSurfaceMaterialId
    gear: int
    rpm: float
    display_speed: int
    in_steer: int
    in_gas: int
    in_bits: int

    @property
    def finished(self) -> bool:
        return bool(self.flags & F_FINISHED)

    @property
    def sim_only(self) -> bool:
        return bool(self.flags & F_SIM_ONLY)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @classmethod
    def unpack(cls, payload: bytes) -> 'Step':
        v = STEP.unpack(payload)
        f = np.asarray(v[3:21], dtype=np.float32)
        wheels = v[21:29]
        damper = np.asarray(wheels[0::2], dtype=np.float32)
        wbits = np.asarray(wheels[1::2], dtype=np.int32)
        return cls(race_time=v[0], flags=v[1], checkpoints=v[2],
                   pos=f[0:3], rot=f[3:12], vel=f[12:15], ang_vel=f[15:18],
                   wheel_damper=damper, wheel_contact=(wbits & 1) > 0,
                   wheel_sliding=(wbits & 2) > 0, wheel_material=(wbits >> 8) & 255,
                   gear=v[29], rpm=v[30], display_speed=v[31],
                   in_steer=v[32], in_gas=v[33], in_bits=v[34])


def action_bytes(steer: int = 0, gas: int = 0, bits: int = 0) -> bytes:
    return struct.pack('<iiii', C_ACTION, int(steer), int(gas), int(bits))


def play_cmd(inputs) -> bytes:
    """inputs: [(steer, gas, bits)] applied at race time 0, 10, 20, ... ms."""
    if not 0 < len(inputs) <= MAX_PLAY:
        raise ValueError(f'{len(inputs)} inputs (1..{MAX_PLAY})')
    return struct.pack('<ii', C_PLAY, len(inputs)) + b''.join(struct.pack('<iii', *map(int, a)) for a in inputs)


def draw_cmd(points, size: float) -> bytes:
    pts = [tuple(map(float, p)) for p in points][:MAX_DRAW]
    return struct.pack('<ii', C_DRAW, len(pts)) + b''.join(struct.pack('<fff', *p) for p in pts) + \
        struct.pack('<f', float(size))


def int_cmd(cmd: int, value: int) -> bytes:
    return struct.pack('<ii', cmd, int(value))


def float_cmd(cmd: int, value: float) -> bytes:
    return struct.pack('<if', cmd, float(value))


def str_cmd(cmd: int, text: str) -> bytes:
    raw = text.encode('utf-8', errors='replace')
    return struct.pack('<ii', cmd, len(raw)) + raw
