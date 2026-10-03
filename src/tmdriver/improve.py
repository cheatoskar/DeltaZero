"""Per-map self-improvement ("improve"): the model practises one map and gets faster.

    python tmdriver.py improve --map 10030774 --rounds 20        (serve must NOT run)

Version 1, deliberately simple (cross-entropy method / self-imitation):
  1. every round: a few runs in simulation-only mode from the saved race start, steering and
     pedals SAMPLED from the model (temperature = exploration) plus one greedy run;
  2. keep the best runs so far (a finish always beats no finish; then the time; unfinished
     runs rank by distance);
  3. fine-tune the model on the decisions of those best runs (behaviour cloning on its own
     best driving), so the next round starts from there;
  4. log every round; save the best run as a TMInterface input file and the per-map model;
  5. at the end, play the best run back visibly.

Not yet: branching from saved mid-run states (tries only the hard part), the value head as
a critic, many game instances. Those come after this works in the game.
"""
import copy
import json
from collections import Counter
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as TF

from . import protocol as P
from .ghost_policy import GhostPolicy
from .paths import DATA, DRIVER_CKPT, RUNS, safe, tmi_scripts_dir, torch_device
from .session import Episode, GameSession

OUT = RUNS / 'improve'
FETCHED = DATA / 'tmx_maps.json'     # maps that open_map fetched from TMX: TMX id -> file, uid, name


class ImproveEpisode(Episode):
    """Stops a run when it is stuck. With a reference line: no +2 m along it for 3 s. Without
    one (course.py): no new track cell for 5 s "on the ground", or 4 s on the ground off the track
    (the car must not learn to drive across the stadium grass; top TMX runs leave the track cells
    for at most 3.1 s, measured on 40 runs with scripts/course_check.py), or 25 s in any case.
    "On the ground" = a wheel touches it OR slower than 10 m/s: flights do not count (measured
    gaps up to 18.5 s on a speed map), but a car on its roof or side touches nothing with its
    wheels either (seen 2026-09-25: runs stuck at a wall until the 180 s limit)."""
    supports_hold = True     # GameSession may hold each action for `hold` ticks (C_HOLD)
    hold = 1
    STALL_MS = 3000
    CELL_STALL_MS = 5000
    OFF_TRACK_MS = 4000
    MAX_NO_PROGRESS_MS = 25000
    FLIGHT_SPEED = 10.0      # m/s

    def __init__(self, policy: GhostPolicy, limit_ms: int, temp: float, seed: int):
        self.policy, self.limit, self.temp = policy, limit_ms, temp
        self.rng = np.random.default_rng(seed)

    def begin(self, start):
        self.policy.restart()
        self.decisions = []          # (inputs, bin, gas, brake, race_time)
        self.ticks = []              # (race_time, steer, gas, bits)
        self.path = []               # car position every 100 ms (for drawing)
        self.finish = None
        self.reason = None
        self.best, self.best_t, self.last_t = 0.0, 0, 0
        self.ground_ms, self.off_ms = 0, 0
        self.start_kmh = None

    def act(self, st):
        dt = max(0, st.race_time - self.last_t)
        self.last_t = st.race_time
        if st.finished:
            self.finish, self.reason = st.race_time, 'finish'
            return None
        if st.race_time > self.limit:
            self.reason = 'time limit'
            return None
        if st.race_time == 1000:
            self.start_kmh = int(st.display_speed)
        pol = self.policy
        before = pol.last_decision_t
        a = pol.act(st, sample=self.temp > 0, temp=max(self.temp, 1e-3), rng=self.rng)
        if pol.last_decision_t != before and pol.last_out is not None:
            o = pol.last_out
            # (inputs, steer bin, gas, brake, race time, progress so far): rl.py makes rewards from the last two
            self.decisions.append((pol.last_inputs, o['bin'], o['gas'], o['brake'], st.race_time, pol.progress_m))
        return self._after(st, a, dt)

    # two-phase acting for batched driving (rl_vec.py): same rules as act()
    def act_begin(self, st):
        """-> ('end', None) | ('act', action or None) | ('decide', (t, features))."""
        dt = max(0, st.race_time - self.last_t)
        self.last_t = st.race_time
        self._dt = dt
        if st.finished:
            self.finish, self.reason = st.race_time, 'finish'
            return 'end', None
        if st.race_time > self.limit:
            self.reason = 'time limit'
            return 'end', None
        if st.race_time == 1000:
            self.start_kmh = int(st.display_speed)
        t, feats = self.policy.act_begin(st)
        if feats is not None:
            return 'decide', (t, feats)
        a = self.policy.action if t >= 0 else (0, 0, 0)
        return 'act', self._after(st, a, dt)

    def act_end(self, st, t, x_row, out_row):
        pol = self.policy
        a = pol.act_end(t, x_row, out_row, sample=self.temp > 0, temp=max(self.temp, 1e-3), rng=self.rng)
        o = pol.last_out
        self.decisions.append((pol.last_inputs, o['bin'], o['gas'], o['brake'], st.race_time, pol.progress_m))
        return self._after(st, a, self._dt)

    def _after(self, st, a, dt):
        """Record the tick; the stall rules; -> the action, or None when the run ends."""
        pol = self.policy
        if st.race_time >= 0:
            # the held ticks too (C_HOLD): the plugin re-applies this action without asking
            for k in range(max(1, self.hold)):
                self.ticks.append((st.race_time + 10 * k, *a))
            if st.race_time % 100 == 0:
                self.path.append(st.pos.astype(float).tolist())
        prog = pol.progress_m
        if pol.line_progress:
            if prog > self.best + 2.0 or st.race_time <= 0:
                self.best, self.best_t = max(prog, self.best), st.race_time
            elif st.race_time - self.best_t > self.STALL_MS:
                self.reason = 'stalled'
                return None
            return a
        grounded = bool(np.any(st.wheel_contact)) or st.speed < self.FLIGHT_SPEED
        if prog > self.best + 2.0 or st.race_time <= 0:
            self.best, self.best_t, self.ground_ms = max(prog, self.best), st.race_time, 0
        elif st.race_time - self.best_t > self.MAX_NO_PROGRESS_MS:
            self.reason = 'stalled'
            return None
        elif grounded:
            self.ground_ms += dt
            if self.ground_ms > self.CELL_STALL_MS:
                self.reason = 'stalled'
                return None
        if pol.on_track:
            self.off_ms = 0
        elif grounded:
            self.off_ms += dt
            if self.off_ms > self.OFF_TRACK_MS:
                self.reason = 'off track'
                return None
        return a

    def result(self):
        return {'finished': self.finish is not None, 'time_ms': self.finish, 'progress_m': round(self.best, 1),
                'cps': self.policy.cps, 'final_progress': self.policy.progress_m, 'reason': self.reason, 'race_ms': self.last_t, 'temp': self.temp, 'start_kmh': self.start_kmh,
                'decisions': self.decisions, 'ticks': self.ticks, 'stall_t': self.best_t, 'path': self.path}


class PrefixEpisode(Episode):
    """Replays a run's exact inputs up to `branch_t` (deterministic physics: the same states),
    feeding the policy the same observations. At branch_t it saves the game state in `slot`
    and the policy state, so BranchEpisodes can start there instead of at the race start."""

    def __init__(self, link, policy: GhostPolicy, parent: dict, branch_t: int, slot: int = 1, log=print):
        self.link, self.policy, self.parent, self.branch_t, self.slot, self.log = \
            link, policy, parent, branch_t, slot, log
        self.by_t = {t: (s, g, b) for t, s, g, b in parent['ticks']}

    def begin(self, start):
        self.policy.restart()
        self.step, self.snap, self.diverged_m = None, None, None

    def act(self, st):
        t = st.race_time
        if st.finished or t > self.branch_t:
            return None
        self.policy.observe(st)
        if t == self.branch_t:
            k = t // 100
            if k < len(self.parent['path']):
                self.diverged_m = float(np.linalg.norm(st.pos - np.asarray(self.parent['path'][k])))
                if self.diverged_m > 0.01:
                    self.log(f'  WARNING: replaying the best run diverged by {self.diverged_m:.3f} m at '
                             f'{t / 1000:.1f}s (the physics should be deterministic)')
            self.link.save(self.slot)        # answered together with the next episode's REWIND
            self.step, self.snap = st, self.policy.snapshot()
            return None
        return self.by_t.get(t, (0, 0, 0))

    def result(self):
        return {'prefix': True, 'branch_t': self.branch_t, 'diverged_m': self.diverged_m}


class BranchEpisode(ImproveEpisode):
    """An ImproveEpisode that starts where PrefixEpisode saved the game (mid-run), with the
    parent run's history before that point. Falls back to the race start if nothing was saved."""

    def __init__(self, policy: GhostPolicy, limit_ms: int, temp: float, seed: int, prefix: PrefixEpisode):
        super().__init__(policy, limit_ms, temp, seed)
        self.prefix = prefix

    @property
    def start_slot(self):
        return self.prefix.slot if self.prefix.step is not None else 0

    @property
    def start_step(self):
        return self.prefix.step

    def begin(self, start):
        super().begin(start)
        pre = self.prefix
        if pre.step is None:
            return
        bt, parent = pre.branch_t, pre.parent
        self.policy.restore(pre.snap)
        self.ticks = [x for x in parent['ticks'] if x[0] < bt]
        self.decisions = [d for d in parent['decisions'] if d[4] < bt]
        self.path = list(parent['path'][:bt // 100])
        self.best, self.best_t, self.last_t = self.policy.progress_m, bt, bt

    def result(self):
        return dict(super().result(), branch_t=self.prefix.branch_t if self.prefix.step is not None else None)


def branch_point(best: dict, rng) -> Optional[int]:
    """Where to branch from the best run: 1-4 s before it got stuck, or, once it finishes,
    anywhere along the run (to find time). On the 100 ms lattice; None if too early."""
    if best['finished']:
        t = rng.uniform(0.05, 0.95) * best['time_ms']
    else:
        t = best['stall_t'] - rng.uniform(1000, 4000)
    t = int(t // 100 * 100)
    return t if t >= 500 else None


def path_points(path, spacing: float = 4.0, lift: float = 0.6):
    """Car positions -> points every `spacing` m (for the trigger-box drawing)."""
    out, last = [], None
    for p in path:
        p = np.asarray(p, dtype=np.float64)
        if last is None or np.linalg.norm(p - last) >= spacing:
            out.append((p[0], p[1] + lift, p[2]))
            last = p
    return out[:P.MAX_DRAW]


def draw(link, path):
    link.draw(path_points(path) if path else [], size=0.8)
    link.flush()


def run_text(r) -> str:
    if r['finished']:
        return f"{r['time_ms'] / 1000:.2f}s"
    return (f"{r['cps']} CP, " if r.get('cps') else '') + f"{r['progress_m']:.0f} m"


def score(r) -> float:
    """A finish beats everything (then the time); otherwise checkpoints, then progress."""
    return 1e7 - r['time_ms'] if r['finished'] else r.get('cps', 0) * 1e5 + r['progress_m']


def training_decisions(r) -> list:
    """What to imitate from a run: all of a finished run; of an unfinished one only the part
    up to 2 s before it stopped making progress (the mistake itself is not imitated)."""
    if r['finished']:
        return r['decisions']
    cut = r['stall_t'] - 2000
    return [d for d in r['decisions'] if d[4] <= cut]


def fine_tune(policy: GhostPolicy, elite: List[dict], steps: int, bs: int, lr: float, log=print):
    dec = [d for r in elite for d in training_decisions(r)]
    if len(dec) < 16:
        return None
    x = [torch.cat([d[0][k] for d in dec]) for k in range(6)]
    dev = policy.device
    y_steer = torch.tensor([d[1] for d in dec], device=dev)
    y_gas = torch.tensor([float(d[2]) for d in dec], device=dev)
    y_brake = torch.tensor([float(d[3]) for d in dec], device=dev)
    model = policy.model
    threads = torch.get_num_threads()
    torch.set_num_threads(max(threads, (os.cpu_count() or 4)))   # measured: 5.3 -> 1.6 s/step
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(len(dec))
    loss = None
    for _ in range(steps):
        j = torch.randint(0, len(dec), (min(bs, len(dec)),), generator=g).to(dev)
        out = model(*[t[j] for t in x])
        loss = (TF.cross_entropy(out['steer'], y_steer[j]) + 0.5 * TF.binary_cross_entropy_with_logits(
            out['gas'], y_gas[j]) + 0.5 * TF.binary_cross_entropy_with_logits(out['brake'], y_brake[j]))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    torch.set_num_threads(threads)
    return float(loss.detach())


def tmi_script(ticks) -> str:
    """Best run -> TMInterface input script (press / rel / steer at race times in seconds).
    NOTE: the exact syntax TMInterface 2.2 accepts is not verified yet; load it in the TMI
    console and check before relying on it."""
    lines, prev = [], (0, 0, 0)
    names = ((P.UP, 'up'), (P.DOWN, 'down'), (P.LEFT, 'left'), (P.RIGHT, 'right'))
    for t, steer, gas, bits in ticks:
        ts = f'{t / 1000:.2f}'
        _, _, pb = prev
        for bit, name in names:
            if bits & bit and not pb & bit:
                lines.append(f'{ts} press {name}')
            elif pb & bit and not bits & bit:
                lines.append(f'{ts} rel {name}')
        if steer != prev[0]:
            lines.append(f'{ts} steer {steer}')
        prev = (steer, gas, bits)
    return '\n'.join(lines) + '\n'


class Pace:
    """Progress over time of the best plan so far: a plan that is far behind it for a while is
    stopped early (drive only needs the best plan, so hopeless ones need not be finished)."""

    def __init__(self, behind_m: float = 60.0, grace_ms: int = 3000):
        self.curve, self.score, self.behind_m, self.grace_ms = {}, None, behind_m, grace_ms

    def offer(self, r: dict, curve: dict):
        if self.score is None or score(r) > self.score:
            self.score, self.curve = score(r), curve

    def hopeless(self, t: int, progress: float) -> bool:
        ref = self.curve.get(t // 100 * 100)
        return ref is not None and t > self.grace_ms and progress < ref - self.behind_m


class PlanEpisode(ImproveEpisode):
    """An ImproveEpisode for drive: reports itself when done and gives up when hopeless."""

    def __init__(self, policy, limit_ms, temp, seed, k, n, pace: Pace, log=print):
        super().__init__(policy, limit_ms, temp, seed)
        self.k, self.n, self.pace, self.log = k, n, pace, log

    def begin(self, start):
        super().begin(start)
        self.curve, self.behind_since = {}, None

    def act(self, st):
        a = super().act(st)
        if a is None:
            return None
        t, prog = st.race_time, self.policy.progress_m
        if t >= 0 and t % 100 == 0:
            self.curve[t] = prog
            if self.pace.hopeless(t, prog):
                self.behind_since = self.behind_since if self.behind_since is not None else t
                if t - self.behind_since >= 2000:
                    self.reason = 'far behind'
                    return None
            else:
                self.behind_since = None
        return a

    def result(self):
        r = super().result()
        self.pace.offer(r, self.curve)
        txt = f"finish {run_text(r)}" if r['finished'] else f"{run_text(r)} ({r['reason']})"
        kind = 'greedy' if self.temp == 0 else f'temperature {self.temp}'
        self.log(f'  plan {self.k}/{self.n} ({kind}): {txt}')
        return r


class Playback(Episode):
    """Plays a recorded tick sequence back (to show the best run), recording like an
    ImproveEpisode so the result can be compared with the plan."""

    def __init__(self, ticks):
        self.by_t = {t: (s, g, b) for t, s, g, b in ticks}
        self.end = ticks[-1][0] if ticks else 0

    JUMP_M = 15.0     # in one 10 ms tick; the car covers < 3 m per tick even at 1000 km/h

    def begin(self, start):
        self.finish, self.ticks, self.path = None, [], []
        self.aborted, self.prev = False, None

    def act(self, st):
        if st.finished:
            self.finish = st.race_time
            return None
        if st.race_time > self.end + 1000:
            return None
        # The player pressed respawn / restart while the run was shown: stop showing it (the
        # caller carries on, e.g. with the next training round). Seen as the respawn input, the
        # race time jumping back (restart) or the car jumping (respawn at a checkpoint).
        prev, self.prev = self.prev, (st.race_time, st.pos.astype(float))
        if st.in_bits & P.RESPAWN or (prev is not None and (
                st.race_time < prev[0] or float(np.linalg.norm(st.pos - prev[1])) > self.JUMP_M)):
            self.aborted = True
            return None
        a = self.by_t.get(st.race_time, (0, 0, 0))
        if st.race_time >= 0:
            self.ticks.append((st.race_time, *a))
            if st.race_time % 100 == 0:
                self.path.append(st.pos.astype(float).tolist())
        return a

    def result(self):
        return {'finished': self.finish is not None, 'time_ms': self.finish, 'ticks': self.ticks, 'path': self.path,
                'aborted': self.aborted}


def open_map(sess: GameSession, track_id: Optional[int], replays: int, log=print):
    """The map to work on -> (name, uid, best_ms). track_id None: the map open in the game
    (nothing is loaded). Otherwise that TMX map is loaded; a map that is not downloaded yet is
    fetched from TMX first (with `replays` replays, for the reference line)."""
    from .evaluate import map_list
    t0 = time.monotonic()
    while sess.map is None and time.monotonic() - t0 < (10 if track_id is None else 3):
        try:
            sess.next(timeout=1.0)
        except TimeoutError:
            pass
    if track_id is None:
        if sess.map is None:
            raise SystemExit('no map announced: restart the race in the game, or pass --map <TMX id>')
        return sess.map.name, sess.map.uid, None
    from . import tmx
    known = json.loads(FETCHED.read_text(encoding='utf-8')) if FETCHED.exists() else {}
    m = next((m for m in map_list(str(track_id))), None)
    if m is None and str(track_id) in known and (tmx.tracks_dir() / known[str(track_id)]['map_file']).exists():
        m = known[str(track_id)]                    # (not there, e.g. another game folder: fetch again)
    if m is None:
        from .paths import TMX
        log(f'map {track_id} is not downloaded yet: fetching it from TMX ...')
        res = tmx.fetch(track_id, replays, TMX, safe, log=log)
        m = {'map_file': res['map'].name, 'uid': res['uid'], 'name': res['info']['TrackName'],
             'best_ms': res['info'].get('AuthorTime') or None}
        known[str(track_id)] = m
        FETCHED.parent.mkdir(parents=True, exist_ok=True)
        FETCHED.write_text(json.dumps(known, indent=1), encoding='utf-8')
    if sess.map is not None and sess.map.uid == m['uid']:
        log(f"map {m['name']!r} is already loaded")        # runs restart the race themselves
    elif not sess.load_map(m['map_file'], m['uid']):
        raise SystemExit(f'map {track_id} did not load. If the game is in a race, leave it to the '
                         f'main menu (TMI loads queued maps there), or open the map yourself and use '
                         f'map id 0')
    return m['name'], m['uid'], m['best_ms']


def map_file_for(uid: str, track_id: Optional[int], log=print) -> Optional[str]:
    """The map's file name in <game folder>/Tracks/Challenges/TMDriver (what helper instances
    load), fetched from TMX if needed; None if the map is not on TMX (e.g. a campaign map)."""
    from . import tmx
    from .evaluate import map_list
    known = json.loads(FETCHED.read_text(encoding='utf-8')) if FETCHED.exists() else {}
    cands = list(map_list(str(track_id))) if track_id else []
    cands += [m for m in known.values() if m.get('uid') == uid]
    for m in cands:
        if m.get('uid') == uid:
            return m['map_file']            # the name the main instance loads it by, too
    try:
        tid = track_id or tmx.track_id_by_uid(uid)
        if tid:
            from .paths import TMX
            return tmx.fetch(tid, 0, TMX, safe, log=lambda *a: None)['map'].name
    except Exception as e:
        log(f'  TMX: {e}')
    # Check persistent cache first
    from .paths import DATA
    cache_file = DATA / 'map_uids_cache.json'
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding='utf-8'))
            if uid in cache and Path(cache[uid]).exists():
                return cache[uid]
        except Exception:
            pass
    # If not on TMX or offline: search local tracks_dir for the map file containing this uid
    try:
        from .virtual_game import map_dirs
        t0 = time.time()
        for md in map_dirs():
            base = Path(md)
            # 1. Search shallow in TMDriver folder first (fast)
            if base.name == 'TMDriver' and base.exists():
                for f in base.glob('*.Challenge.Gbx'):
                    try:
                        if uid.encode() in f.read_bytes()[:2048]:
                            cache[uid] = str(f.resolve())
                            cache_file.write_text(json.dumps(cache, indent=1), encoding='utf-8')
                            return cache[uid]
                    except Exception:
                        continue
            # 2. Broader search with 2.5s time budget
            search_dirs = [base]
            if base.parent.name == 'Tracks' or base.name == 'TMDriver':
                search_dirs.append(base.parent)  # Tracks/Challenges
            for d in search_dirs:
                if not d.exists():
                    continue
                for f in d.glob('**/*.Challenge.Gbx'):
                    if time.time() - t0 > 2.5:
                        break
                    try:
                        if uid.encode() in f.read_bytes()[:2048]:
                            cache[uid] = str(f.resolve())
                            cache_file.write_text(json.dumps(cache, indent=1), encoding='utf-8')
                            return cache[uid]
                    except Exception:
                        continue
                if time.time() - t0 > 2.5:
                    break
    except Exception as e:
        log(f'  local map search: {e}')
    return None


def helper_sessions(links, uid: str, track_id: Optional[int], log=print) -> list:
    """GameSessions on the helper instances, each with the map loaded and drivable. Instances
    that fail are left out (with a message); no map file: no helpers."""
    links = list(links)
    if not links:
        return []
    f = map_file_for(uid, track_id, log)
    if f is None:
        log(f'{len(links)} helper instance(s) not used: this map is not on TMX, so they cannot load it')
        return []
    out = []
    for k, h in enumerate(links):
        tag = f'[#{k + 2}]'
        hs = GameSession(h, lambda *a, tag=tag: log(tag, *a))
        try:
            if not hs.load_map(f, uid):
                log(f'{tag} did not load the map: not used')
                continue
            hs.ensure_drivable()
        except Exception as e:
            log(f'{tag} not used: {e!r}'[:200])
            continue
        out.append(hs)
    return out


def reference(uid: str, track_id: Optional[int], log=print):
    """The fastest known run of the map as (positions, source), or None: re-simulated > TMX
    replay > recording; if there is none, the map's replays are fetched from TMX (by uid).
    With the line on, the model sees it; with the line off, it only JUDGES progress (see
    reference_setup), so that driving backwards or off the map never counts as progress."""
    from .policy import reference_positions, track_id_for
    tid = track_id or track_id_for(uid)
    if os.environ.get('TMDRIVER_ROUTE') == 'plan':        # as if nobody had driven the map yet
        return planned_route(uid, tid, log)
    ref = reference_positions(uid, tid)
    if ref is None:
        from . import tmx
        from .paths import TMX
        try:
            tid = tid or tmx.track_id_by_uid(uid)
            if tid:
                log(f'no local reference run: fetching the replays of TMX {tid} ...')
                tmx.fetch(tid, 5, TMX, safe, log=log)
                ref = reference_positions(uid, tid)
        except Exception as e:           # offline, TMX down: drive without the line
            log(f'  TMX: {e}')
    if ref is None:
        ref = planned_route(uid, tid, log)
    return ref


def planned_route(uid: str, track_id: Optional[int], log=print):
    """The route planner's line (route_planner.py: from the map's geometry, no run needed), as
    (positions, source), or None."""
    try:
        from . import tmx
        from .route_planner import plan
        name = map_file_for(uid, track_id, log)
        if not name:
            log(f'  route planner: no map file found for uid {uid[:12]}')
            return None
        p = Path(name)
        path = p if p.is_absolute() and p.exists() else tmx.tracks_dir() / name
        if not path.exists():
            log(f'  route planner: {path} not found')
            return None
        log(f'planning a route from the map geometry ({path.name}) ...')
        route = plan(path, f'plan_{safe(uid)}', log=log)
        if route is None:
            return None
        return route, f'planned route ({np.linalg.norm(np.diff(route, axis=0), axis=1).sum():.0f} m)'
    except Exception as e:               # no TMNF-C set up, or no path found
        log(f'  route planner: {e!r}'[:200])
        return None


def reference_setup(uid: str, track_id: Optional[int], use_line: bool, log=print):
    """-> (line for the model or None, judge run or None, text for the log)."""
    route_mode = os.environ.get('TMDRIVER_ROUTE', 'auto')
    if route_mode == 'none':
        return None, None, 'OFF (pure geometry: track cells + checkpoints, zero ghosts)'

    if route_mode == 'plan':
        ref = planned_route(uid, track_id, log)
    elif use_line:
        ref = reference(uid, track_id, log)
    else:
        # Line is OFF: do NOT fetch human TMX ghosts. Use planned geometric centerline if available.
        ref = planned_route(uid, track_id, log=lambda *a: None)

    if use_line and ref is None:
        log('WARNING: line requested, but no reference route available: driving WITHOUT the line')
    if use_line and ref is not None:
        return ref[0], None, f'ON ({ref[1]})'
    if ref is not None:
        return None, ref[0], f'OFF (blocks only; progress judged along {ref[1]}, the model does not see it)'
    return None, None, 'OFF (blocks only; progress = track cells + checkpoints, zero ghosts)'


def improve(link, track_id: Optional[int] = None, rounds: int = 20, episodes: int = 6, elite_k: int = 3,
            steps: int = 12, bs: int = 64, lr: float = 1e-4, temps=(0.4, 0.7, 1.0), show: bool = True,
            ckpt: Path = None, branch: int = 8, seed: Optional[int] = None, use_line: bool = False,
            helpers=(), log=print):
    """branch > 0: from round 2 on, `branch` of the runs start from a state saved shortly before
    the best run got stuck (or anywhere along it once it finishes), so the attempts are spent on
    the hard part instead of re-driving the start every time.
    use_line: give the model the reference line (fastest known run); default: blocks only.
    helpers: Links to more game instances (instances.py): the runs of a round are spread over
    all instances (fleet.py); new best runs are shown in the main one."""
    from .fleet import Fleet, policy_view
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    sess = GameSession(link, log)
    name, uid, best_ms = open_map(sess, track_id, 5, log)
    line, judge, line_txt = reference_setup(uid, track_id, use_line, log)
    policy.reset([b.__dict__ for b in sess.map.blocks], line, judge)
    draw(link, [])                                  # boxes of an earlier map or training
    fleet = Fleet(sess, helper_sessions(helpers, uid, track_id, log))
    pols = [policy] + [policy_view(policy) for _ in fleet.sessions[1:]]
    out = OUT / safe(uid)
    out.mkdir(parents=True, exist_ok=True)
    limit = int(2 * best_ms + 10000) if best_ms else 180000
    line_tag = 'with line' if line is not None else 'no line'
    log(f"improve {name!r}: {rounds} rounds x {episodes + 1} runs on {len(fleet)} game instance(s), "
        f"model {ckpt}")
    log(f"reference line: {line_txt}")
    sess.ensure_drivable()

    elite: List[dict] = []
    best: Optional[dict] = None
    saved = [None]                     # score of the best run on disk
    history = []

    def save_best(rnd):
        """best_run.json, the TMI input file and the per-map model (called on every new best,
        and when the training is stopped early)."""
        if best is None or saved[0] == score(best):
            return
        script_txt = tmi_script(best['ticks'])
        (out / 'best_inputs.txt').write_text(script_txt, encoding='utf-8')
        try:
            (tmi_scripts_dir() / f'{safe(name)}.txt').write_text(script_txt, encoding='utf-8')
        except Exception:
            pass
        (out / 'best_run.json').write_text(json.dumps({
            'map': name, 'uid': uid, 'time_ms': best['time_ms'], 'progress_m': best['progress_m'],
            'cps': best.get('cps'), 'line': line_txt, 'round': rnd + 1, 'ticks': best['ticks'],
            'path': best['path']}), encoding='utf-8')
        policy.model.eval()
        torch.save(dict(torch.load(ckpt, map_location='cpu', weights_only=False),
                        state_dict=policy.model.state_dict(), improved_on=uid,
                        improve_best=run_text(best)), out / 'model.pt')
        saved[0] = score(best)

    seed = int(np.random.default_rng().integers(1_000_000)) if seed is None else seed
    log(f'seed {seed}   (Ctrl+C here or Stop in the game: end the training and show the best run)')
    brng = np.random.default_rng(seed + 12345)
    rnd, stopped = 0, False
    try:
        for rnd in range(rounds):
            t0 = time.perf_counter()
            n = len(fleet)
            groups = [[] for _ in range(n)]
            groups[0].append(ImproveEpisode(pols[0], limit, 0.0, seed))       # greedy: result 0
            bt = branch_point(best, brng) if branch and best is not None else None
            n_start = episodes if bt is None else max(1, episodes // 3)
            for k in range(n_start):
                seed += 1
                i = (k + 1) % n
                groups[i].append(ImproveEpisode(pols[i], limit, temps[k % len(temps)], seed))
            if bt is not None:
                # every instance that gets branch runs replays the best run to bt itself first
                share = [branch // n + (1 if i < branch % n else 0) for i in range(n)]
                for i in range(n):
                    if not share[i]:
                        continue
                    pre = PrefixEpisode(fleet.sessions[i].link, pols[i], best, bt, log=log)
                    groups[i].append(pre)
                    for k in range(share[i]):
                        seed += 1
                        groups[i].append(BranchEpisode(pols[i], limit, temps[k % len(temps)], seed, pre))
            for s_ in fleet.sessions:
                s_.status(f'Training {name} ({line_tag}): round {rnd + 1}/{rounds}' +
                          (f", best {best['time_ms'] / 1000:.2f}s" if best and best['finished'] else ''))
            out_groups = fleet.run(groups, sim_only=True, keep=True)   # the games stay frozen until release()
            res = [r for g in out_groups for r in g if not r.get('prefix')]
            elite = sorted(elite + res, key=score, reverse=True)[:elite_k]
            improved = best is None or score(elite[0]) > score(best)
            best = elite[0]
            with fleet.hold():                                  # learning: games frozen, plugins kept waiting
                loss = fine_tune(policy, elite, steps, bs, lr, log)
            fin = [r['time_ms'] for r in res if r['finished']]
            greedy = res[0]
            rec = {'round': rnd + 1, 'greedy': greedy['time_ms'] if greedy['finished'] else f"{greedy['progress_m']} m",
                   'finished': f'{len(fin)}/{len(res)}', 'fastest_this_round': min(fin) if fin else None,
                   'best_ms': best['time_ms'], 'best_progress_m': best['progress_m'], 'best_cps': best.get('cps'),
                   'line': line is not None, 'loss': loss,
                   'seconds': round(time.perf_counter() - t0, 1),
                   'reasons': [r['reason'] for r in res], 'start_speed_kmh': res[0].get('start_kmh'),
                   'branch_from_s': bt / 1000 if bt is not None else None,
                   'branch_best': max((r['progress_m'] if not r['finished'] else float('inf')
                                       for r in res if r.get('branch_t') is not None), default=None)}
            history.append(rec)
            if max(r['progress_m'] for r in res) < 5:
                log('  WARNING: the car did not move in any run. Is a screen blocking the race start '
                    '("press any key", the map intro)? Click into the game or press Enter once.')
            with open(out / 'progress.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(rec, time=time.strftime('%H:%M:%S'))) + '\n')
            best_txt = run_text(best)
            stops = ', '.join(f'{n}x {k}' for k, n in Counter(rec['reasons']).most_common())
            br = f"; {branch} runs from {bt / 1000:.1f}s" if bt is not None else ''
            log(f"round {rnd + 1}: greedy {rec['greedy']}, {rec['finished']} finished, best {best_txt}"
                f"{' (new)' if improved else ''}, {rec['seconds']}s  [{stops}{br}]")
            if improved:
                with fleet.hold():
                    draw(link, best['path'])          # the best line so far, visible in the game
                    save_best(rnd)
                if show and best['ticks'] and rnd + 1 < rounds:
                    log(f'  showing the new best run ({best_txt}) in the game (respawn skips it) ...')
                    sess.status(f'New best {best_txt} (round {rnd + 1}): showing it. Respawn = skip')
                    with fleet.hold(skip_main=True):              # helpers wait meanwhile
                        shown = sess.run([Playback(best['ticks'])], sim_only=False)[0]  # ends main's frozen state
                    if shown['aborted']:
                        log('  skipped (respawn): next round')
    except KeyboardInterrupt:
        stopped = True
        log(f'\nstopped in round {rnd + 1}' + (f': best so far {run_text(best)}' if best else ': no run yet'))
        fleet.recover()
        if best is not None:
            draw(link, best['path'])
            save_best(rnd)
    fleet.release()
    if best is None:
        sess.status('Training stopped before the first round finished.')
        return history
    best_txt = run_text(best)
    log(f'best {best_txt}; inputs -> {out / "best_inputs.txt"}, model -> {out / "model.pt"}')
    if (show or stopped) and best['ticks']:
        log(f'showing the best run ({best_txt}) ...')
        sess.status(f'Best run ({best_txt}): showing it. Respawn = skip')
        sess.run([Playback(best['ticks'])], sim_only=False)
    sess.status(f'Training {"stopped" if stopped else "done"}: best {best_txt}. '
                f'"Show best run" plays it again.')
    return history


def show_best(link, track_id: Optional[int] = None, speed: float = 1.0, log=print):
    """Play the best run that `improve` or `rl-vec` found on this map."""
    sess = GameSession(link, log)
    open_map(sess, track_id, 0, log)
    u = safe(sess.map.uid)
    cands = [
        (RUNS / 'rl' / f'{u}_vec' / 'best_run.json', 'TMNF-C RL'),
        (RUNS / 'rl' / u / 'best_run.json', 'In-game RL'),
        (OUT / u / 'best_run.json', 'improve'),
    ]
    f, src_tag = None, 'TMNF-C RL'
    best_time = 999999999
    best_prog = -1.0
    for cand, tag in cands:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding='utf-8'))
                t = data.get('time_ms')
                p = data.get('progress_m', 0.0)
                if t is not None and t < best_time:
                    best_time, f, src_tag = t, cand, tag
                elif best_time == 999999999 and p > best_prog:
                    best_prog, f, src_tag = p, cand, tag
            except Exception:
                if f is None:
                    f, src_tag = cand, tag

    if not f or not f.exists():
        sess.status('No best run for this map yet: press "Train" first.')
        raise SystemExit(f'no best run for {sess.map.name!r} yet; train on the map first')
    run = json.loads(f.read_text(encoding='utf-8'))
    txt = f"{run['time_ms'] / 1000:.2f}s" if run.get('time_ms') else f"{run.get('progress_m', 0):.0f} m"
    log(f"best run on {run['map']!r}: {txt} ({src_tag}, line {run.get('line', '?')})")
    sess.ensure_drivable()
    sess.status(f'Best run ({txt}, {src_tag}): showing it')
    link.speed(speed)
    shown = sess.run([Playback([tuple(t) for t in run['ticks']])], sim_only=False)[0]
    sess.status(f'Best run shown ({txt}).' if not shown['aborted'] else 'Best run: stopped (respawn).')


def drive_preview(link, track_id: Optional[int] = None, speed: float = 1.0, ckpt: Path = None,
                  plans: int = 12, temps=(0.4, 0.7, 1.0), use_line: bool = False, log=print):
    """"Route first, then drive": the model drives the map `plans` times without rendering (one
    greedy run, the rest sampled: keyboard steering is tapping, and always taking the most likely
    steering bin understeers), the best plan is drawn as small trigger boxes, then its exact
    inputs are played back visibly. The physics is deterministic, so the visible run is the plan.
    use_line: give the model the reference line (fastest known run); default: blocks only."""
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    sess = GameSession(link, log)
    _, uid, _ = open_map(sess, track_id, 5, log)
    line, judge, line_txt = reference_setup(uid, track_id, use_line, log)
    line_tag = 'with line' if line is not None else 'no line'
    log(f'reference line: {line_txt}')
    policy.reset([b.__dict__ for b in sess.map.blocks], line, judge)
    draw(link, [])
    sess.ensure_drivable()
    sess.status(f'AI is planning its line ({plans} tries, {line_tag}; the game is frozen meanwhile) ...')
    pace, seed = Pace(), int(np.random.default_rng().integers(1_000_000))
    mf = map_file_for(uid, track_id, log)
    used_tmnfc = False
    plan = None
    if mf and Path(mf).exists():
        try:
            from .tmnfc_sim import CarSim
            from .rl_vec import Runner
            from .fleet import policy_view
            sess.status(f'AI is planning instantly in TMNF-C ({plans} runs on GPU, {line_tag}) ...')
            log(f'planning {plans} runs in TMNF-C on GPU ({line_tag}) ...')
            t_plan0 = time.perf_counter()
            sim = CarSim(Path(mf), track_id or 0, n=min(plans, 32))
            runner = Runner(policy, sim)
            plan_eps = [ImproveEpisode(policy_view(policy), 180000, 0.0, seed)] + \
                [ImproveEpisode(policy_view(policy), 180000, temps[k % len(temps)], seed + k + 1)
                 for k in range(plans - 1)]
            res = runner.run(plan_eps)
            sim.close()
            plan = max(res, key=score)
            log(f"TMNF-C planned {plans} runs in {time.perf_counter() - t_plan0:.2f}s: greedy {run_text(res[0])}, "
                f"{sum(r['finished'] for r in res)}/{len(res)} finished, best {run_text(plan)}")
            used_tmnfc = True
        except Exception as e:
            log(f'TMNF-C planning fallback to in-game: {e}')
            used_tmnfc = False

    if not used_tmnfc:
        sess.status(f'AI is planning its line ({plans} tries, {line_tag}; the game is frozen meanwhile) ...')
        log(f'planning {plans} runs (the game is frozen meanwhile) ...')
        eps = [PlanEpisode(policy, 180000, 0.0, seed, 1, plans, pace, log)] + \
            [PlanEpisode(policy, 180000, temps[k % len(temps)], seed + k + 1, k + 2, plans, pace, log)
             for k in range(plans - 1)]
        res = sess.run(eps, sim_only=True)
        plan = max(res, key=score)
        log(f"plans: greedy {run_text(res[0])}, {sum(r['finished'] for r in res)}/{len(res)} finished, "
            f"best {run_text(plan)} (temperature {plan['temp']})")
    txt = run_text(plan) if plan['finished'] else f"{plan['reason']} after {run_text(plan)}"
    log(f'plan: {txt}, {len(plan["path"])} path samples')
    draw(link, plan['path'])
    sess.status(f'Best of {plans} plans ({txt}, {line_tag}) drawn, the AI drives it now')
    link.speed(speed)
    real = sess.run([Playback(plan['ticks'])], sim_only=False)[0]
    if real['aborted']:
        log('drive: stopped (respawn)')
        sess.status(f'AI: stopped by respawn (plan {txt}, {line_tag})')
        return plan, real
    n = min(len(plan['path']), len(real['path']))
    same = real['finished'] == plan['finished'] and real['time_ms'] == plan['time_ms'] and n > 0 and \
        max(float(np.linalg.norm(np.subtract(plan['path'][i], real['path'][i]))) for i in range(n)) < 1e-3
    rtxt = f"{real['time_ms'] / 1000:.2f}s" if real['finished'] else f"ended after {len(real['ticks']) / 100:.1f}s"
    log(f'drive: {rtxt}; identical to the plan: {same}')
    sess.status(f'AI: {rtxt} (plan {txt}, {"identical" if same else "DIFFERENT"}, {line_tag})')
    return plan, real


def hold_check(link, track_id: Optional[int] = None, ckpt: Path = None, log=print) -> bool:
    """Real-game check of C_HOLD: a greedy and a sampled run, once answered every tick and once
    held for 5 ticks by the plugin, must have the same inputs, path and result."""
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    sess = GameSession(link, log)
    _, uid, _ = open_map(sess, track_id, 5, log)
    line, judge, _ = reference_setup(uid, track_id, False, log)
    policy.reset([b.__dict__ for b in sess.map.blocks], line, judge)
    sess.ensure_drivable()
    runs = {}
    for hold in (1, 5):
        sess.hold_ticks = hold
        t0 = time.perf_counter()
        res = sess.run([ImproveEpisode(policy, 60000, 0.0, 1), ImproveEpisode(policy, 60000, 1.0, 5)], sim_only=True)
        runs[hold] = (res, time.perf_counter() - t0)
        log(f"hold {hold}: {', '.join(run_text(r) + ' (' + str(r['reason']) + ')' for r in res)} in {runs[hold][1]:.1f}s")
    same = True
    for a, b in zip(runs[1][0], runs[5][0]):
        end = a['time_ms'] if a['finished'] and b['finished'] else min(a['race_ms'], b['race_ms'])
        ta = [x for x in a['ticks'] if x[0] < end]
        tb = [x for x in b['ticks'] if x[0] < end]
        k = min(len(a['path']), len(b['path']))
        ok = a['finished'] == b['finished'] and a['time_ms'] == b['time_ms'] and ta == tb and \
            a['path'][:k] == b['path'][:k]
        if not ok:
            first = next((x for x, y in zip(ta, tb) if x != y), None)
            log(f'  DIFFERENT: {run_text(a)} vs {run_text(b)}; first differing tick {first}')
        same &= ok
    log(f"held runs identical to per-tick runs: {same}; time {runs[1][1]:.1f}s -> {runs[5][1]:.1f}s")
    sess.status(f'Hold check: identical {same}, {runs[1][1]:.1f}s -> {runs[5][1]:.1f}s')
    return same
