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

from . import protocol as P
from .link import Link, MapInfo
from .paths import CALIBRATION, LIVE_MAPS, safe
from .calib import Calibration


def focus_game_window() -> bool:
    """Bring the TMNF window to the foreground (Windows; best effort). In the main menu the
    game only processes queued map loads while it has focus."""
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
            found.append(h)
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return False
    user32.keybd_event(0x12, 0, 0, 0)       # an ALT tap lets a background process take focus
    user32.keybd_event(0x12, 0, 2, 0)
    return bool(user32.SetForegroundWindow(found[0]))


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
        link.strict = True     # one ACTION per STEP, checked (see Link.owed)
        # No "Press any key to continue" / opponent screens between map loads (TMInterface
        # variable; without it cheatoskar had to press Enter on every map).
        # TMDRIVER_SKIP_LOAD_SCREENS=0 turns it off (suspected of crashing the game under Wine).
        if os.environ.get('TMDRIVER_SKIP_LOAD_SCREENS', '1') != '0':
            link.execute('set skip_map_load_screens true')
            link.flush()

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
        known = self.calib.data.get('map_command_template')
        templates = [known] if known else ['TMDriver\\{f}', 'TMDriver/{f}', 'Challenges\\TMDriver\\{f}']
        for tpl in templates:
            focus_game_window()
            self._drain()
            form = tpl.format(f=map_file)
            self.link.execute(f'map "{form}"')
            self.link.flush()
            deadline = time.monotonic() + self.MAP_TIMEOUT_S
            while time.monotonic() < deadline:
                try:
                    kind, payload = self.next(timeout=max(0.1, deadline - time.monotonic()))
                except TimeoutError:
                    break
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
                    link.action(*a)
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
                ep.begin(start)
                a = ep.act(start)
                if batch and a is not None:
                    self._play(ep)
                phase = 'run'
            else:
                a = ep.act(st)
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
            link.action(*a)

    def _begin(self, ep: Episode, start: P.Step):
        """Rewind to the episode's start: slot 0 (the race start) or, for an episode with
        `start_slot`/`start_step`, a state saved mid-run (branching)."""
        slot = getattr(ep, 'start_slot', 0)
        st = getattr(ep, 'start_step', None) if slot else None
        self.link.rewind(slot if st is not None else 0)
        s0 = st if st is not None else start
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
