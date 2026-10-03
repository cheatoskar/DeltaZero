"""What Python does in each plugin mode.

Contract with the plugin: in DRIVE and TEST mode every STEP must be answered with exactly
one ACTION (`link.action`), which may be preceded by other commands. In RECORD mode the
plugin does not wait, so nothing is answered.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import protocol as P
from .calib import (Calibration, calibrate_forward, effective_steer, steer_sign_from,
                    turn_rate)
from .line import Pursuit, RefLine
from .link import Link, MapInfo
from . import replay as replay_mod
from .paths import DRIVER_CKPT, LINES, RUNS, TMX, safe


def steps_to_arrays(steps: List[P.Step]) -> Dict[str, np.ndarray]:
    return {
        't': np.array([s.race_time for s in steps], dtype=np.int32),
        'pos': np.stack([s.pos for s in steps]),
        'rot': np.stack([s.rot for s in steps]),
        'vel': np.stack([s.vel for s in steps]),
        'ang_vel': np.stack([s.ang_vel for s in steps]),
        'checkpoints': np.array([s.checkpoints for s in steps], dtype=np.int32),
        'wheel_damper': np.stack([s.wheel_damper for s in steps]),
        'wheel_contact': np.stack([s.wheel_contact for s in steps]),
        'wheel_sliding': np.stack([s.wheel_sliding for s in steps]),
        'wheel_material': np.stack([s.wheel_material for s in steps]).astype(np.int16),
        'gear': np.array([s.gear for s in steps], dtype=np.int16),
        'rpm': np.array([s.rpm for s in steps], dtype=np.float32),
        'in_steer': np.array([s.in_steer for s in steps], dtype=np.int32),
        'in_gas': np.array([s.in_gas for s in steps], dtype=np.int32),
        'in_bits': np.array([s.in_bits for s in steps], dtype=np.int32),
    }


class Task:
    blocking = False

    def __init__(self, link: Link, ctx: 'Context'):
        self.link = link
        self.ctx = ctx
        self.done = False

    def on_step(self, step: P.Step):
        raise NotImplementedError

    def on_bench(self, ticks: int, elapsed_ms: int, race_time: int):
        pass

    def finish(self, message: str):
        """Leave the mode; the caller still owes the ACTION for the current step."""
        self.ctx.log(message)
        self.link.status(message)
        self.link.mode(P.MODE_IDLE)
        self.done = True


class Context:
    """State shared by the tasks: current map, calibration, log."""

    def __init__(self, calibration: Calibration, log=print):
        self.map: Optional[MapInfo] = None
        self.calib = calibration
        self.log = log
        self.ui_speed = 1.0
        self._policy = None
        self._policy_mtime = None

    def policy(self):
        """The trained driver, loaded once and reloaded when the checkpoint changes. Loading
        (torch import + weights) takes seconds, far longer than the plugin waits for an
        answer to a STEP, so it must happen before a race starts, never inside a step."""
        if not DRIVER_CKPT.exists():
            return None
        mtime = DRIVER_CKPT.stat().st_mtime
        if self._policy is None or mtime != self._policy_mtime:
            from .ghost_policy import load_policy
            t0 = time.perf_counter()
            self._policy, self._policy_mtime = load_policy(DRIVER_CKPT), mtime
            self.log(f'driver model loaded in {time.perf_counter() - t0:.1f}s')
        return self._policy

    def line_path(self) -> Optional[Path]:
        return LINES / f'{safe(self.map.uid)}.npz' if self.map else None


# ---------------------------------------------------------------------- RECORD

class Recorder(Task):
    """Stores one clean run (race start to finish) driven manually."""

    def __init__(self, link, ctx):
        super().__init__(link, ctx)
        self.steps: List[P.Step] = []
        link.status('Recording: drive the map once, cleanly, to the finish.')
        link.flush()

    def on_step(self, step: P.Step):
        if self.steps and step.race_time < self.steps[-1].race_time:
            self.steps = []   # restarted: keep only the current attempt
        self.steps.append(step)
        if not step.finished:
            return
        if self.ctx.map is None:
            self.finish('Recording discarded: the plugin did not announce a map.')
            self.link.flush()
            return
        a = steps_to_arrays(self.steps)
        finish = int(step.race_time)
        path = self.ctx.line_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, finish_ms=finish, map_name=self.ctx.map.name, **a)

        notes = []
        fwd = calibrate_forward(a['rot'], a['vel'])
        known = self.ctx.calib.forward
        if fwd and known and (fwd['kind'], fwd['idx'], fwd['sign']) != known:
            notes.append(f'WARNING: the forward axis does not match the calibration: {fwd}')
        analog = float(np.mean(a['in_steer'] != 0))
        pressed = float(np.mean(a['in_bits'] != 0))
        notes.append(f'analog steering {analog:.0%}, keys pressed {pressed:.0%} of the ticks')
        self.ctx.log(f'recorded {len(self.steps)} ticks, finish {finish / 1000:.2f}s -> {path}; ' + '; '.join(notes))
        self.finish(f'Recording saved: {finish / 1000:.2f} s, {len(self.steps)} ticks. ' + notes[-1])
        self.link.flush()


# ---------------------------------------------------------------------- DRIVE

class ModelController:
    """Adapter so the Driver task can use the learned policy like the pursuit follower."""

    def __init__(self, policy, blocks, line):
        self.policy, self.blocks, self.line = policy, blocks, line
        self.reset()

    def reset(self):
        self.policy.reset(self.blocks, self.line)

    def act(self, step):
        return self.policy.act(step)

    @property
    def progress_m(self):
        return self.policy.progress_m


class GhostController:
    """Adapter for the ghost-feature driver (the line is optional for it)."""

    def __init__(self, policy, blocks, line_pos):
        self.policy = policy
        policy.reset(blocks, line_pos)
        self.line = policy.line

    def reset(self):
        self.policy.restart()

    def act(self, step):
        return self.policy.act(step)

    @property
    def progress_m(self):
        return self.policy.progress_m


class Driver(Task):
    """"AI drive (live)": the learned model if one is trained (runs/m1/driver.pt), otherwise
    pure pursuit on human recording."""
    blocking = True
    STALL_S = 3.0
    MAX_ATTEMPTS = 3

    def __init__(self, link, ctx):
        super().__init__(link, ctx)
        self.ctrl = None
        self.error = None
        self.attempt = 1
        self.last_t = None
        self.best_progress = 0.0
        self.progress_time = 0
        self.kind = None

    def _setup(self) -> Optional[str]:
        c = self.ctx
        if not c.calib.ready():
            return 'Please run the self-test first (it measures the steering sign and the car axes).'
        rec = c.line_path()
        self.ref_finish = int(np.load(rec)['finish_ms']) if rec is not None and rec.exists() else None
        policy = c._policy if DRIVER_CKPT.exists() else None   # preloaded on the button press
        if policy is not None and c.map is not None and getattr(policy, 'ghost', False):
            from . import policy as pol
            # the reference line only with `serve --line` (TMDRIVER_LINE=1), as for Drive / Train
            use_line = os.environ.get('TMDRIVER_LINE') == '1'
            ref = pol.reference_positions(c.map.uid, pol.track_id_for(c.map.uid)) if use_line else None
            line_pos, src = ref if ref else (None, 'no line, blocks only')
            self.ctrl = GhostController(policy, [b.__dict__ for b in c.map.blocks], line_pos)
            self.kind = f'model {Path(policy.ckpt).parent.name}/{Path(policy.ckpt).name}, line: {src}'
        elif policy is not None and c.map is not None:
            from . import policy as pol
            ref = pol.reference_line(c.map.uid, pol.track_id_for(c.map.uid))
            if ref is None:
                return 'Model found, but no reference line for this map (no TMX replays, no recording).'
            line, src = ref
            self.ctrl = ModelController(policy, [b.__dict__ for b in c.map.blocks], line)
            self.kind = f'model, line: {src}'
        else:
            if rec is None or not rec.exists():
                return 'No recording for this map. Press "Record" first and drive to the finish once.'
            line = RefLine.load(rec)
            self.ctrl = Pursuit(line, c.calib.forward, c.calib.steer_sign)
            self.kind = 'line follower (no model)'
        self.link.speed(c.ui_speed)
        self.link.status(f'AI faehrt ({self.kind}), Versuch 1')
        return None

    def on_step(self, step: P.Step):
        if self.ctrl is None and not self.done:
            err = self._setup()
            if err:
                self.finish(err)
        if self.done:
            self.link.action(0, 0, 0)
            return

        if self.last_t is not None and step.race_time < self.last_t:
            self.ctrl.reset()
            self.best_progress, self.progress_time = 0.0, step.race_time
        self.last_t = step.race_time

        if step.finished:
            t = step.race_time
            ref = f'  (your recording {self.ref_finish / 1000:.2f} s)' if self.ref_finish else ''
            msg = f'Finish! AI {t / 1000:.2f} s{ref}  [{self.kind}]'
            self._log_attempt(True, t)
            self.link.speed(1.0)
            self.finish(msg)
            self.link.action(0, 0, 0)
            return

        steer, gas, bits = self.ctrl.act(step)
        prog = self.ctrl.progress_m
        if prog > self.best_progress + 2.0 or step.race_time <= 0:
            self.best_progress, self.progress_time = max(prog, self.best_progress), step.race_time
        elif step.race_time - self.progress_time > self.STALL_S * 1000:
            self._log_attempt(False, step.race_time)
            if self.attempt >= self.MAX_ATTEMPTS:
                self.link.speed(1.0)
                total = self.ctrl.line.length
                of = f' of {total:.0f} m' if total != float('inf') else ''
                self.finish(f'AI is stuck (at {prog:.0f} m{of}), stopped. [{self.kind}]')
                self.link.action(0, 0, 0)
                return
            self.attempt += 1
            self.link.status(f'AI stuck at {prog:.0f} m, attempt {self.attempt}')
            self.link.restart()
            self.link.action(0, 0, 0)
            return
        self.link.action(steer, gas, bits)

    def _log_attempt(self, finished: bool, t: int):
        RUNS.mkdir(parents=True, exist_ok=True)
        rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'map': self.ctx.map.name, 'uid': self.ctx.map.uid,
               'attempt': self.attempt, 'finished': finished, 'race_time_ms': t,
               'progress_m': self.ctrl.progress_m,
               'line_m': self.ctrl.line.length if self.ctrl.line.length != float('inf') else None,
               'ref_finish_ms': self.ref_finish, 'speed': self.ctx.ui_speed, 'controller': self.kind}
        with open(RUNS / 'drive_log.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')


# ---------------------------------------------------------------------- TEST

def scripted_action(k: int):
    """Deterministic input sequence for the self test, as a function of the tick index."""
    if k < 150:
        return 0, 0, P.UP                                   # straight, full gas
    if k < 230:
        return 40000, 0, P.UP | P.STEER_ANALOG              # hold +steer: measures its sign
    steer = int(40000 * np.sin(k / 23.0))
    bits = P.UP | P.STEER_ANALOG
    if k % 200 >= 180:
        bits = P.DOWN | P.STEER_ANALOG
    return steer, 0, bits


class SelfTest(Task):
    """Measures what the rest of the project depends on, then restores normal play:

      1. scripted run in simulation-only mode: Python-in-the-loop ticks per second, the
         forward axis of the rotation matrix, and the sign of analog steering;
      2. the same inputs again after rewinding: is the simulation bit-for-bit repeatable?
      3. input replays from the race start, each compared with where the car really went:
         manual recording of this map (both input alignments), and TMX replays of this
         map from data/tmx/<uid>/ (every alignment and steer sign). A variant that reaches
         the finish at exactly the recorded time proves recorded inputs can be re-simulated,
         which the whole replay-based training data plan depends on;
      4. plugin-internal benchmark at game speed 1, 10 and 100: raw physics ticks per second.
    """
    blocking = True
    N = 1500            # scripted ticks (15 s of race time)
    BENCH_TICKS = 10000
    BENCH_SPEEDS = (1.0, 10.0, 100.0)
    MAX_TMX_REPLAYS = 2
    MAX_TMX_MS = 120_000

    def __init__(self, link, ctx):
        super().__init__(link, ctx)
        self.phase = 'wait_start'
        self.report: Dict = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'replays': []}
        self.runs: Dict[str, Dict[int, np.ndarray]] = {}
        self.steps_b: List[P.Step] = []
        self.jobs: List[Dict] = []
        self.job: Optional[Dict] = None
        self.bench_queue = list(self.BENCH_SPEEDS)
        link.status('Self-test running (the picture freezes briefly, that is normal) ...')

    # -- helpers
    def _start_run(self, name: str, step: P.Step):
        self.runs[name] = {}
        self.cur = name
        self.wall0 = time.perf_counter()

    def _record(self, step: P.Step):
        # Race time freezes at the finish, so the last race_time can repeat: keep the first.
        self.runs[self.cur].setdefault(step.race_time, step.pos.astype(np.float64))

    def _compare(self, a: str, b_positions: Dict[int, np.ndarray]) -> Dict:
        ra = self.runs[a]
        common = sorted(set(ra) & set(b_positions))
        if not common:
            return {'ticks': 0}
        d = np.array([np.linalg.norm(ra[t] - b_positions[t]) for t in common])
        bad = np.nonzero(d > 0.01)[0]
        return {'ticks': len(common), 'max_diff_m': float(d.max()), 'final_diff_m': float(d[-1]),
                'first_divergence_ms': int(common[bad[0]]) if len(bad) else None}

    @staticmethod
    def _table_action(table: Dict[str, np.ndarray], t: int, shift: int):
        """Input for the tick after race time t, read `shift` ticks later in the table
        (table times are 0, 10, 20, ...)."""
        i = (t + 10 * shift) // 10
        if i < 0 or i >= len(table['t']):
            return 0, 0, 0
        return int(table['in_steer'][i]), int(table['in_gas'][i]), int(table['in_bits'][i])

    def _plan_replays(self):
        """Input replays to verify, for the map that is loaded."""
        path = self.ctx.line_path()
        if path is not None and path.exists():
            h = dict(np.load(path))
            ts = h['t'].astype(np.int64)
            keep = ts >= 0
            n = int(ts[keep].max() // 10) + 1 if keep.any() else 0
            table = {'t': np.arange(n) * 10, 'in_steer': np.zeros(n, np.int64),
                     'in_gas': np.zeros(n, np.int64), 'in_bits': np.zeros(n, np.int64)}
            idx = ts[keep] // 10
            table['in_steer'][idx] = h['in_steer'][keep]
            table['in_gas'][idx] = h['in_gas'][keep]
            b = h['in_bits'][keep] & (P.UP | P.DOWN | P.LEFT | P.RIGHT)
            b = b | np.where(h['in_steer'][keep] != 0, P.STEER_ANALOG, 0) \
                  | np.where(h['in_gas'][keep] != 0, P.GAS_ANALOG, 0)
            table['in_bits'][idx] = b
            ref = {}
            for tt, pp in zip(ts, h['pos'].astype(np.float64)):
                ref.setdefault(int(tt), pp)
            for shift in (0, 1):
                self.jobs.append({'name': f'own_recording_shift{shift}', 'source': 'recording',
                                  'table': table, 'shift': shift, 'steer_sign': 1, 'ref': ref,
                                  'finish_ms': int(h['finish_ms']), 'need_exact_pos': True})
        else:
            self.report['own_recording'] = 'no recording for this map'

        folder = TMX / safe(self.ctx.map.uid) if self.ctx.map else None
        reps = []
        for f in (sorted(folder.glob('*.Replay.Gbx')) if folder and folder.exists() else []):
            try:
                reps.append((f, replay_mod.load(f)))
            except Exception as e:   # a broken file must not stop the test
                self.report['replays'].append({'name': f.name, 'source': 'tmx', 'error': repr(e)})
        reps = [r for r in reps if r[1].race_time_ms <= self.MAX_TMX_MS]
        reps.sort(key=lambda r: r[1].race_time_ms)
        for f, rep in reps[:self.MAX_TMX_REPLAYS]:
            ref = {int(t): p for t, p in zip(rep.ghost_t, rep.ghost_pos)}
            stem = f.name.split('.')[0]
            for sign in ((1, -1) if rep.uses_analog_steer else (1,)):
                table = replay_mod.input_table(rep, steer_sign=sign)
                for shift in (-1, 0, 1):
                    self.jobs.append({'name': f'tmx_{stem}_shift{shift}_sign{sign}', 'source': 'tmx',
                                      'table': table, 'shift': shift, 'steer_sign': sign, 'ref': ref,
                                      'finish_ms': rep.race_time_ms, 'need_exact_pos': True,
                                      'respawns': rep.respawns, 'analog': rep.uses_analog_steer})

    # -- state machine
    def on_step(self, step: P.Step):
        t = step.race_time
        ph = self.phase

        if ph == 'wait_start':
            if t < 0:
                self.link.action(0, 0, 0)
                return
            self.link.save(0)
            self.link.sim_only(True)
            self.phase = 'scripted'
            self._start_run('scripted', step)
            ph = 'scripted'

        if ph in ('scripted', 'repeat'):
            self._record(step)
            if ph == 'scripted':
                self.steps_b.append(step)
            k = t // 10
            if k < self.N and not step.finished:
                self.link.action(*scripted_action(k))
                return
            wall = time.perf_counter() - self.wall0
            self.report[f'{ph}_ticks'] = len(self.runs[ph])
            self.report[f'{ph}_python_loop_ticks_per_s'] = round(len(self.runs[ph]) / max(wall, 1e-9), 1)
            if ph == 'scripted':
                self._analyse_scripted()
                self.phase = 'repeat'
                self.link.rewind(0)
                self._start_run('repeat', step)
                self.link.action(*scripted_action(0))
                return
            self.report['determinism'] = self._compare('repeat', self.runs['scripted'])
            self._plan_replays()
            self._next_replay_or_bench(step)
            return

        if ph == 'replay':
            job = self.job
            self._record(step)
            if not step.finished and t <= job['finish_ms'] + 200:
                self.link.action(*self._table_action(job['table'], t, job['shift']))
                return
            cmp = self._compare(job['name'], job['ref'])
            same_time = bool(step.finished and t == job['finish_ms'])
            exact = same_time and (not job['need_exact_pos'] or cmp.get('max_diff_m', 1.0) == 0.0)
            res = {k: job[k] for k in ('name', 'source', 'shift', 'steer_sign', 'finish_ms')}
            res.update(cmp)
            res.update(replay_finished=bool(step.finished),
                       replay_time_ms=int(t) if step.finished else None, exact=exact)
            res.update({k: job[k] for k in ('respawns', 'analog') if k in job})
            self.report['replays'].append(res)
            self._next_replay_or_bench(step)
            return

        if ph == 'bench_wait':
            self._next_bench(step)
            return

        # anything unexpected: stay safe
        self.link.action(0, 0, 0)

    def _analyse_scripted(self):
        a = steps_to_arrays(self.steps_b)
        fwd = calibrate_forward(a['rot'], a['vel'])
        k = a['t'] // 10
        omega = turn_rate(a['vel'])
        speed = np.linalg.norm(a['vel'], axis=1)
        m = (k >= 165) & (k < 230)
        steer_cmd = np.where(m, 40000, 0)
        sgn = steer_sign_from(steer_cmd, omega, speed)
        self.report['forward_axis'] = fwd
        self.report['steer_sign'] = sgn
        self.report['max_speed_kmh'] = float(speed.max() * 3.6)
        upd = {}
        if fwd and fwd['score'] > 0.9:
            upd['forward'] = fwd
        if sgn and sgn['agreement'] > 0.8:
            upd['steer'] = sgn
        if upd:
            self.ctx.calib.update(**upd, source='selftest', measured=self.report['time'])

    def _next_replay_or_bench(self, step):
        if self.jobs:
            self.job = self.jobs.pop(0)
            self.phase = 'replay'
            self.link.rewind(0)
            self._start_run(self.job['name'], step)
            self.link.action(*self._table_action(self.job['table'], 0, self.job['shift']))
            return
        self.phase = 'bench_wait'
        self._next_bench(step)

    def _next_bench(self, step):
        if self.bench_queue:
            s = self.bench_queue.pop(0)
            self.bench_speed = s
            self.link.rewind(0)
            self.link.speed(s)
            self.link.bench(self.BENCH_TICKS)
            self.link.action(0, 0, P.UP)
            return
        self._conclude()

    def on_bench(self, ticks, elapsed_ms, race_time):
        self.report.setdefault('bench_ticks_per_s', {})[str(self.bench_speed)] = \
            round(ticks / max(elapsed_ms, 1) * 1000.0, 1)

    def _replay_alignment(self):
        """Persist the TMX input alignment when the evidence is unambiguous: exact (same
        finish time, 0.0 m at every ghost sample) for one shift, and for analog replays one
        steer sign. Measured 2026-09-24: shift 1, steer sign -1."""
        tmx = [x for x in self.report['replays'] if x.get('source') == 'tmx' and 'error' not in x]
        exact = [x for x in tmx if x.get('exact')]
        shifts = {x['shift'] for x in exact}
        signs = {x['steer_sign'] for x in exact if x.get('analog')}
        if len(shifts) == 1 and len(signs) <= 1 and exact:
            upd = {'shift': shifts.pop(), 'measured': self.report['time'], 'exact_runs': len(exact)}
            if signs:
                upd['steer_sign'] = signs.pop()
            old = self.ctx.calib.data.get('replay', {})
            if 'steer_sign' not in upd and 'steer_sign' in old:
                upd['steer_sign'] = old['steer_sign']
            self.ctx.calib.update(replay=upd)
            self.report['replay_alignment'] = upd

    def _conclude(self):
        self._replay_alignment()
        self.link.sim_only(False)
        self.link.speed(1.0)
        self.link.restart()
        r = self.report
        r['map'] = self.ctx.map.name if self.ctx.map else None
        r['calibration_ready'] = self.ctx.calib.ready()
        RUNS.mkdir(parents=True, exist_ok=True)
        out = RUNS / f"selftest_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(r, indent=1))
        det = r.get('determinism', {})
        bench = r.get('bench_ticks_per_s', {})
        parts = [f"Self-test done. Determinism max {det.get('max_diff_m', float('nan')):.4f} m",
                 f"Python loop {r.get('scripted_python_loop_ticks_per_s')} ticks/s",
                 f"physics {bench.get('100.0')} ticks/s",
                 f"calibration {'OK' if r['calibration_ready'] else 'MISSING'}"]
        for src, label in (('recording', 'Own recording'), ('tmx', 'TMX replay')):
            rs = [x for x in r['replays'] if x.get('source') == src and 'error' not in x]
            if rs:
                ok = [x for x in rs if x.get('exact')]
                parts.append(f"{label} exact: " + (', '.join(f"shift {x['shift']} sign {x['steer_sign']}"
                                                              for x in ok) if ok else 'NO'))
        msg = '. '.join(parts) + f'. Report: {out.name}'
        self.ctx.log(json.dumps(r, indent=1))
        self.finish(msg)
        self.link.action(0, 0, 0)
