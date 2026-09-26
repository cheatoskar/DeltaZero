"""Scripted control of the game: load maps, run episodes from the race start.

Used by re-simulation (replay inputs -> full 10 ms state) and by closed-loop evaluation
(model drives, in simulation-only mode). Unlike `server.Server`, which reacts to window
buttons, a session drives the game itself: it loads a map with the TMInterface `map`
console command, restarts, saves the start state, and runs one episode after another,
rewinding to the start between them.
"""
import contextlib
import json
import os
import queue
import threading
import time
from typing import List, Optional

import numpy as np

from . import protocol as P
from .link import Link, MapInfo
from .paths import CALIBRATION, LIVE_MAPS, safe
from .calib import Calibration


def pid_listening_on(port: int) -> Optional[int]:
    """Windows: the process that listens on a local TCP port, i.e. the game instance whose plugin
    took that port (instances.py). Get-NetTCPConnection, because netstat's state column is
    translated (German Windows). None if unknown."""
    if os.name != 'nt':
        return None
    import subprocess
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              f'(Get-NetTCPConnection -State Listen -LocalPort {int(port)} '
                              f'-ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None
    return int(out) if out.isdigit() else None


def focus_game_window(pid: Optional[int] = None) -> bool:
    """Bring the TMNF window to the foreground (Windows; best effort): the window of process
    `pid` if given (one of several game instances), else the first TMNF window. A game only
    processes a queued map load while it has focus (seen for background helper instances,
    2026-09-25)."""
    try:
        import ctypes
        import ctypes.wintypes as W
    except ImportError:
        return False
    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        return False
    found = []

    @ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    def cb(h, _):
        n = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(h, n, 256)
        if user32.IsWindowVisible(h) and n.value.startswith('TrackMania') and 'Forever' in n.value:
            if pid is not None:
                owner = W.DWORD()
                user32.GetWindowThreadProcessId(h, ctypes.byref(owner))
                if owner.value != pid:
                    return True
            found.append(h)
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return False
    h = found[0]
    if user32.GetForegroundWindow() == h:
        return True
    if user32.IsIconic(h):
        user32.ShowWindow(h, 9)             # SW_RESTORE: a minimized game gets no focus
    user32.keybd_event(0x12, 0, 0, 0)       # an ALT tap lets a background process take focus
    user32.keybd_event(0x12, 0, 2, 0)
    user32.SetForegroundWindow(h)
    time.sleep(0.3)                         # measured: the switch is not immediate
    return user32.GetForegroundWindow() == h


class Episode:
    """One run from the race start (or, with `start_slot` > 0 and `start_step`, from a state
    saved mid-run in that slot). `act` returns an ACTION tuple, or None when done.

    An episode whose inputs are known in advance may define `plan() -> [(steer, gas, bits)]`
    (index = race time / 10). With `run(..., batch=True)` the plugin then plays them itself
    and `act` only observes the streamed states (its return value is ignored there); the
    final tick arrives as an ordinary STEP, so `act` still ends the episode."""

    def begin(self, start: P.Step):
        raise NotImplementedError

    def act(self, step: P.Step):
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError


class GasProbe(Episode):
    """Full throttle for one second from the race start: does the car move? Judged by the
    distance driven; the speedometer (display_speed) is logged as well."""

    def begin(self, start):
        self.kmh, self.moved, self.p0 = 0, 0.0, np.asarray(start.pos, dtype=np.float64)

    def act(self, st):
        if st.race_time >= 1000 or st.finished:
            self.kmh = int(st.display_speed)
            self.moved = float(np.linalg.norm(np.asarray(st.pos, dtype=np.float64) - self.p0))
            return None
        return 0, 0, P.UP

    def result(self):
        return {'kmh': self.kmh, 'moved_m': self.moved}


class GameSession:
    MAP_TIMEOUT_S = 90.0   # load screens + shadow computation can take long on big maps
    STEP_TIMEOUT_S = 30.0

    def __init__(self, link: Link, log=print):
        self.link = link
        self.log = log
        self.calib = Calibration(CALIBRATION)
        # Exactly one reader per connection: a second session on the same link must share
        # it, or two threads race for the plugin's messages.
        if getattr(link, '_pump_q', None) is None:
            link._pump_q = queue.Queue()
            link._last_map = None
            threading.Thread(target=self._pump, args=(link,), daemon=True).start()
        self.q = link._pump_q
        self._held = None      # the start STEP while the plugin is kept waiting (run(keep=True))
        self.focus = True      # bring this instance's window to the front for map loads
        self._pid = None       # its process (found through the plugin's port), for that
        self.stop = None       # a threading.Event: when set, run() ends its episodes early
        # C_HOLD: episodes that allow it (supports_hold) hold each ACTION this many ticks, so the
        # plugin asks Python once per decision instead of every tick. Off (1) until verified in
        # the real game; TMDRIVER_HOLD=5 (or --hold) turns it on.
        self.hold_ticks = int(os.environ.get('TMDRIVER_HOLD', '1') or 1)
        link.strict = True     # one ACTION per STEP, checked (see Link.owed)
        # No "Press any key to continue" / opponent screens between map loads (TMInterface
        # variable; without it cheatoskar had to press Enter on every map).
        # TMDRIVER_SKIP_LOAD_SCREENS=0 turns it off (suspected of crashing the game under Wine).
        if os.environ.get('TMDRIVER_SKIP_LOAD_SCREENS', '1') != '0':
            link.execute('set skip_map_load_screens true')
        # TMInterface variables (documented at donadigo.com/tminterface/variables, checked
        # 2026-09-26): an unfocused game is frame-limited by default, and in simulation-only mode
        # that throttles the physics (measured: ~57 ticks/s with actions held in the plugin, a
        # background helper instance). MediaTracker intros can block the start of a race.
        link.execute('set unfocused_fps_limit false')
        link.execute('set disable_forced_camera true')
        # draw_game false stops rendering (simulation keeps running); --no-draw sets it for batch
        # jobs. Every other connection turns drawing back on, so a window never stays black.
        draw = os.environ.get('TMDRIVER_DRAW_GAME', '1') != '0'
        link.execute(f"set draw_game {'true' if draw else 'false'}")
        link.flush()

    @property
    def game_pid(self) -> Optional[int]:
        """The process of the game instance behind this connection (Windows), or None."""
        if self._pid is None:
            try:
                port = self.link.sock.getpeername()[1]
            except (OSError, AttributeError):
                return None
            self._pid = pid_listening_on(port) or 0
        return self._pid or None

    def focus_window(self) -> bool:
        return focus_game_window(self.game_pid)

    @property
    def map(self) -> Optional[MapInfo]:
        return self.link._last_map

    @staticmethod
    def _pump(link):
        try:
            while True:
                link._pump_q.put(link.read_message())
        except Exception as e:   # connection closed or protocol error
            link._pump_q.put(('error', e))

    def next(self, timeout: float):
        """Next message; map messages are recorded on the way."""
        try:
            kind, payload = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f'no message from the plugin for {timeout:.0f}s')
        if kind == 'error':
            raise ConnectionError(payload)
        if kind == P.P_HELLO and payload[0] != P.PROTOCOL:
            raise RuntimeError(f'plugin protocol {payload[0]}, python expects {P.PROTOCOL}: '
                               f'run `python tmdriver.py install-plugin` and reload the plugin')
        if kind == P.P_MAP:
            self.link._last_map = payload
            LIVE_MAPS.mkdir(parents=True, exist_ok=True)
            (LIVE_MAPS / f'{safe(payload.uid)}.json').write_text(json.dumps({
                'uid': payload.uid, 'name': payload.name, 'author': payload.author,
                'blocks': [b.__dict__ for b in payload.blocks]}), encoding='utf-8')
        return kind, payload

    def status(self, text: str):
        self.link.status(text)
        self.link.flush()

    # ------------------------------------------------------------------ maps

    def _drain(self):
        """Drop what is already queued (e.g. a MAP from before the load command)."""
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                return
            if kind == 'error':
                raise ConnectionError(payload)
            if kind == P.P_MAP:
                self.link._last_map = payload
            if kind == P.P_STEP:          # plugin waits for an answer: release it
                self.link.action(0, 0, 0)

    def load_map(self, map_file: str, uid: str) -> bool:
        r"""Load Tracks/Challenges/TMDriver/<map_file> with TMInterface's `map` command and wait
        until the new race announces it. Always reloads, so the race starts fresh.

        Measured 2026-09-24: the path is relative to Tracks\Challenges (`TMDriver\<file>`);
        the command is only *queued* and runs up to ~20 s later; in the main menu the queue is
        only processed while the game window has focus. Hence: focus the game, send ONE
        command (a queue of fallbacks would fire later and load the wrong map), wait for the
        MAP that the plugin sends when the new race starts."""
        if self.map is not None and self.map.uid == uid:
            return True                     # already open; the runs restart the race themselves
        known = self.calib.data.get('map_command_template')
        # Only the form measured on 2026-09-24. The fallback forms got queued as well and could load
        # a map much later, in the middle of a run (seen 2026-09-25 on the home PC).
        templates = [known or 'TMDriver\\{f}']
        for tpl in templates:
            if self.focus:
                self.focus_window()
            self._drain()
            form = tpl.format(f=map_file)
            self.link.execute(f'map "{form}"')
            self.link.flush()
            t_sent = time.monotonic()
            deadline = t_sent + self.MAP_TIMEOUT_S
            warned = hinted = False
            while time.monotonic() < deadline:
                try:
                    kind, payload = self.next(timeout=min(3.0, max(0.1, deadline - time.monotonic())))
                except TimeoutError:
                    # Nothing yet. In the menu the game only loads while it has focus, and a job
                    # console that opened meanwhile may have taken it: bring the game back.
                    focused = self.focus_window() if self.focus else True
                    if not focused and not warned:
                        self.log('  waiting for the map: please click into the game window once '
                                 '(Windows did not let Python bring it to the front)')
                        warned = True
                    if not hinted and time.monotonic() - t_sent > 30:
                        self.log('  the map has not loaded after 30 s (TMI only queues it). In the game: '
                                 'close any open dialog, go to the main menu or into a race, and click '
                                 'into the window')
                        hinted = True
                    continue
                if kind == P.P_STEP:
                    self.link.action(0, 0, 0)
                if kind == P.P_MAP:
                    if payload.uid == uid:
                        if known != tpl:
                            self.calib.update(map_command_template=tpl)
                        return True
                    self.log(f'  (map {payload.name!r} started while waiting for {uid}: an older queued command?)')
            self.log(f'map command {form!r} did not load {uid} within {self.MAP_TIMEOUT_S:.0f}s')
        return False

    # ------------------------------------------------------------------ episodes

    @contextlib.contextmanager
    def hold(self, every_s: float = 1.0):
        """While the plugin is kept waiting in a STEP (after run(keep=True)), Python may work
        (e.g. learn between rounds): C_WAIT pings keep the plugin from timing out, and the game
        stays frozen, even when its window is minimized."""
        if self._held is None:
            yield
            return
        stop = threading.Event()

        def ping():
            while not stop.wait(every_s):
                try:
                    self.link.wait_ping()
                except OSError:
                    return

        t = threading.Thread(target=ping, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join()

    def recover(self):
        """After a run was interrupted (Ctrl+C in the console, Stop in the game): rendering on,
        plugin idle, and every STEP it still waits for answered exactly once."""
        link = self.link
        self._held = None
        # Not answered yet: the plugin sends one STEP and then waits, so at most one is open (the
        # one being handled, or one still queued). It may be a finished race: answering it plainly
        # leaves the game on the finish (no more STEPs, the medal screen), so restart the race first.
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == 'error':
                raise ConnectionError(payload)
            if kind == P.P_MAP:
                self.link._last_map = payload
        link.sim_only(False)
        link.speed(1.0)
        link.mode(P.MODE_IDLE)
        if link.owed > 0:
            link.restart()
            for _ in range(link.owed):
                link.action(0, 0, 0)
        link.flush()

    def release(self):
        """End a kept session: back to the start, rendering on, plugin idle."""
        if self._held is None:
            return
        link = self.link
        link.rewind(0)
        link.sim_only(False)
        link.speed(1.0)
        link.mode(P.MODE_IDLE)
        link.action(0, 0, 0)
        self._held = None

    def run(self, episodes: List[Episode], sim_only: bool = True, batch: bool = False,
            keep: bool = False) -> List[dict]:
        """Restart the race, save its start (race time 0), then run each episode from that
        state, rewinding in between. Every STEP is answered with exactly one ACTION.
        batch: episodes with `plan()` are played by the plugin (no round trip per tick).
        keep: do not answer the last STEP: the game stays frozen at it (use hold() while
        working, then run(...) again to continue from the saved start, or release())."""
        link = self.link
        results: List[dict] = []
        i, ep, start = 0, None, None
        uid = self.map.uid if self.map else None
        phase = 'first'
        if self._held is not None:
            if not sim_only:
                self.release()                       # a visible run needs a fresh start
            else:
                start, self._held = self._held, None
                ep = episodes[0]
                a = self._begin(ep, start)
                if batch and a is not None:
                    self._play(ep)
                phase = 'run'
                if a is not None:
                    self._act(ep, a)
                else:
                    # (an episode that ends at once: handled by the loop below on the next STEP
                    # would be wrong; answer it here like the loop does)
                    return self._finish_now(episodes, results, ep, start, keep, batch)
        if phase == 'first':
            link.mode(P.MODE_TEST)
            link.flush()
        while True:
            kind, st = self.next(timeout=self.STEP_TIMEOUT_S)
            if kind == P.P_UI and st[0] == P.MODE_IDLE:
                raise KeyboardInterrupt('stopped in the game window')
            if kind == P.P_MAP and st.uid != uid and phase == 'run':
                raise RuntimeError(f'map changed to {st.name!r} during a run (a queued map command?)')
            if kind == P.P_TREC and phase == 'run':
                ep.act(st)                   # the plugin already applied the planned input
                continue
            if kind != P.P_STEP:
                continue
            if phase == 'first':
                if st.race_time > 0:         # not a fresh race: restart it
                    link.restart()
                    link.action(0, 0, 0)
                    phase = 'restarting'
                    continue
                phase = 'wait0'
            if phase == 'restarting':
                if st.race_time > 0:         # the restart has not happened yet
                    link.action(0, 0, 0)
                    continue
                # Fall through WITHOUT answering: wait0 answers this STEP. Two ACTIONs for one
                # STEP made the plugin take the extra one as the next tick's answer, so every
                # later input (and SAVE) landed one tick late (found 2026-09-25).
                phase = 'wait0'
            if phase == 'wait0':
                if st.race_time < 0:
                    link.action(0, 0, 0)
                    continue
                start = st
                if st.finished or st.race_time > 10:
                    raise RuntimeError(f'race did not start cleanly (t={st.race_time}, flags={st.flags})')
                link.save(0)
                if sim_only:
                    link.sim_only(True)
                ep = episodes[0]
                self._prepare(ep)
                ep.begin(start)
                a = ep.act(start)
                if batch and a is not None:
                    self._play(ep)
                phase = 'run'
            else:
                # stop (fleet.Fleet on Ctrl+C): end every remaining episode at its next STEP
                a = None if self.stop is not None and self.stop.is_set() else ep.act(st)
            while a is None:                 # episode over: next one from the saved start
                results.append(ep.result())
                i += 1
                if i == len(episodes):
                    if keep and sim_only:
                        self._held = start          # leave this STEP unanswered: game stays frozen
                        link.flush()
                        return results
                    # Back to the start before rendering resumes: the game must never see a
                    # finished race, or it shows the medal screen and waits for a key press.
                    link.rewind(0)
                    link.sim_only(False)
                    link.speed(1.0)
                    link.mode(P.MODE_IDLE)
                    link.action(0, 0, 0)
                    return results
                ep = episodes[i]
                a = self._begin(ep, start)
                if batch and a is not None:
                    self._play(ep)
            self._act(ep, a)

    def ensure_drivable(self, tries: int = 6, wait_s: float = 3.0) -> bool:
        """Right after a map load the race can be saved and simulated while the car cannot move
        yet (seen twice on 2026-09-25: every run of the first rounds stayed at 0 m). Probe with
        one second of full throttle; if the car does not move, let the game run normally for a
        few seconds and try again."""
        for k in range(tries):
            r = self.run([GasProbe()], sim_only=True)[0]
            txt = f"{r['moved_m']:.1f} m, {r['kmh']} km/h after 1 s of throttle"
            if r['moved_m'] >= 3.0:
                if k:
                    self.log(f'  the car moves now ({txt})')
                return True
            self.log(f'  the car does not move yet ({txt}): letting the game run for {wait_s:.0f} s ...')
            self.status('Waiting for the race to become drivable ...')
            time.sleep(wait_s)
        self.log('  WARNING: the car still does not move. Click into the game or press Enter, then retry.')
        return False

    def _prepare(self, ep: Episode):
        if getattr(ep, 'supports_hold', False):
            ep.hold = max(1, self.hold_ticks)

    def _act(self, ep: Episode, a):
        """Answer the STEP with the episode's action (held for ep.hold ticks if it allows that)."""
        n = getattr(ep, 'hold', 1) if getattr(ep, 'supports_hold', False) else 1
        if n > 1:
            self.link.hold(n)
        self.link.action(*a)

    def _begin(self, ep: Episode, start: P.Step):
        """Rewind to the episode's start: slot 0 (the race start) or, for an episode with
        `start_slot`/`start_step`, a state saved mid-run (branching)."""
        slot = getattr(ep, 'start_slot', 0)
        st = getattr(ep, 'start_step', None) if slot else None
        self.link.rewind(slot if st is not None else 0)
        s0 = st if st is not None else start
        self._prepare(ep)
        ep.begin(s0)
        return ep.act(s0)

    def _finish_now(self, episodes, results, ep, start, keep, batch):
        """Continue a kept session whose first episode ended on the start STEP itself."""
        self._held = start
        rest = episodes[1:]
        results.append(ep.result())
        if rest:
            return results + self.run(rest, sim_only=True, batch=batch, keep=keep)
        if not keep:
            self.release()
        return results

    def _play(self, ep: Episode):
        plan = getattr(ep, 'plan', None)
        if plan is not None:
            inputs = plan()
            if len(inputs) > 1:
                self.link.play(inputs)
