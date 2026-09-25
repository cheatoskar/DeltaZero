"""Several game instances at once.

Every TMNF instance loads the same TMDriver plugin, which listens on the first free port from
P.PORT on (plugin: Main): the instance started first is the main one (P.PORT, the one with the
window you use), helpers get P.PORT + 1, + 2, ... Python finds them by trying those ports.

Careful: the plugin gives its connection to the newest client, so connecting to an instance
takes it away from whoever held it. Only jobs (which own the game while they run) and
standalone commands (serve must not run, see tmdriver.py) connect to helpers.
"""
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import List

from . import protocol as P
from .link import Link

MAX_INSTANCES = 8                 # the plugin tries this many ports as well
TMLOADER = Path(os.path.expandvars(r'%LOCALAPPDATA%\TMLoader\TMLoader.exe'))


def helper_ports(base: int = P.PORT) -> List[int]:
    return list(range(base + 1, base + MAX_INSTANCES))


def listening(port: int, host: str = P.HOST, timeout: float = 0.3) -> bool:
    """Is a plugin listening on this port? (A connect that is closed at once: the plugin logs a
    short connection; only use it on ports nobody else is connected to.)"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def connect_helpers(base: int = P.PORT, log=print) -> List[Link]:
    """Connect to every helper instance that is running (none: [])."""
    links = []
    for port in helper_ports(base):
        try:
            links.append(Link.connect(port=port, wait_s=0.0))
        except OSError:
            continue
        log(f'helper instance on port {port}')
    return links


def launch_command(profile: str = 'DeltaZero') -> List[str]:
    """How a helper instance is started. TMDRIVER_LAUNCH overrides it (the program and its
    arguments, e.g. from a TMLoader desktop shortcut)."""
    custom = os.environ.get('TMDRIVER_LAUNCH', '').strip()
    if custom:
        import shlex
        return shlex.split(custom, posix=False)
    return [str(TMLOADER), 'run', 'TmForever', profile]


def launch_helpers(n: int, base: int = P.PORT, wait_s: float = 300.0, status=None, log=print) -> int:
    """Start n more game instances and wait until their plugins listen. Returns how many do."""
    have = [p for p in helper_ports(base) if listening(p)]
    want = min(len(have) + n, MAX_INSTANCES - 1)
    cmd = launch_command()
    for _ in range(want - len(have)):
        log(f'starting a game instance: {" ".join(cmd)}')
        subprocess.Popen(cmd, cwd=str(TMLOADER.parent) if TMLOADER.exists() else None)
        time.sleep(3.0)                       # TMLoader prepares the profile once per start
    deadline = time.monotonic() + wait_s
    ready = len(have)
    while time.monotonic() < deadline:
        ready = sum(listening(p) for p in helper_ports(base))
        if status:
            status(f'Helpers ready: {ready}/{want}. Log in to each, then press a button here.')
        if ready >= want:
            break
        time.sleep(3.0)
    log(f'helpers ready: {ready}/{want}')
    return ready
