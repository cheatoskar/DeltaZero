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
from .paths import DRIVER_CKPT, RUNS, safe, torch_device
from .session import Episode, GameSession

OUT = RUNS / 'improve'


class ImproveEpisode(Episode):
    STALL_MS = 3000

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
        self.start_kmh = None

    def act(self, st):
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
            self.decisions.append((pol.last_inputs, o['bin'], o['gas'], o['brake'], st.race_time))
        if st.race_time >= 0:
            self.ticks.append((st.race_time, *a))
            if st.race_time % 100 == 0:
                self.path.append(st.pos.astype(float).tolist())
        prog = pol.progress_m
        if prog > self.best + 2.0 or st.race_time <= 0:
            self.best, self.best_t = max(prog, self.best), st.race_time
        elif st.race_time - self.best_t > self.STALL_MS:
            self.reason = 'stalled'
            return None
        return a

    def result(self):
        return {'finished': self.finish is not None, 'time_ms': self.finish, 'progress_m': round(self.best, 1),
                'reason': self.reason, 'race_ms': self.last_t, 'temp': self.temp, 'start_kmh': self.start_kmh,
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
        self.best, self.best_t = self.policy.progress_m, bt

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


def score(r) -> float:
    return 1e7 - r['time_ms'] if r['finished'] else r['progress_m']


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
    return float(loss)


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
        txt = f"finish {r['time_ms'] / 1000:.2f}s" if r['finished'] else f"{r['progress_m']:.0f} m ({r['reason']})"
        kind = 'greedy' if self.temp == 0 else f'temperature {self.temp}'
        self.log(f'  plan {self.k}/{self.n} ({kind}): {txt}')
        return r


class Playback(Episode):
    """Plays a recorded tick sequence back (to show the best run), recording like an
    ImproveEpisode so the result can be compared with the plan."""

    def __init__(self, ticks):
        self.by_t = {t: (s, g, b) for t, s, g, b in ticks}
        self.end = ticks[-1][0] if ticks else 0

    def begin(self, start):
        self.finish, self.ticks, self.path = None, [], []

    def act(self, st):
        if st.finished:
            self.finish = st.race_time
            return None
        if st.race_time > self.end + 1000:
            return None
        a = self.by_t.get(st.race_time, (0, 0, 0))
        if st.race_time >= 0:
            self.ticks.append((st.race_time, *a))
            if st.race_time % 100 == 0:
                self.path.append(st.pos.astype(float).tolist())
        return a

    def result(self):
        return {'finished': self.finish is not None, 'time_ms': self.finish, 'ticks': self.ticks, 'path': self.path}


def improve(link, track_id: Optional[int] = None, rounds: int = 20, episodes: int = 6, elite_k: int = 3,
            steps: int = 12, bs: int = 64, lr: float = 1e-4, temps=(0.4, 0.7, 1.0), show: bool = True,
            ckpt: Path = None, branch: int = 8, seed: Optional[int] = None, log=print):
    """branch > 0: from round 2 on, `branch` of the runs start from a state saved shortly before
    the best run got stuck (or anywhere along it once it finishes), so the attempts are spent on
    the hard part instead of re-driving the start every time."""
    from .evaluate import map_list
    from .policy import reference_positions
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    sess = GameSession(link, log)
    if track_id is not None:
        m = next((m for m in map_list(str(track_id))), None)
        if m is None:
            raise SystemExit(f'map {track_id} not in the M1 manifest or the bulk download')
        if not sess.load_map(m['map_file'], m['uid']):
            raise SystemExit('the map did not load')
        name, uid, best_ms = m['name'], m['uid'], m['best_ms']
    else:
        t0 = time.monotonic()
        while sess.map is None and time.monotonic() - t0 < 10:
            try:
                sess.next(timeout=1.0)
            except TimeoutError:
                pass
        if sess.map is None:
            raise SystemExit('no map announced: restart the race in the game, or pass --map <TMX id>')
        name, uid, best_ms = sess.map.name, sess.map.uid, None
    from .policy import track_id_for
    ref = reference_positions(uid, track_id or track_id_for(uid))
    policy.reset([b.__dict__ for b in sess.map.blocks], ref[0] if ref else None)
    out = OUT / safe(uid)
    out.mkdir(parents=True, exist_ok=True)
    limit = int(2 * best_ms + 10000) if best_ms else 180000
    log(f"improve {name!r}: {rounds} rounds x {episodes + 1} runs, line: {ref[1] if ref else 'none'}, "
        f"model {ckpt}")
    sess.ensure_drivable()

    elite: List[dict] = []
    best: Optional[dict] = None
    history = []
    seed = int(np.random.default_rng().integers(1_000_000)) if seed is None else seed
    log(f'seed {seed}')
    brng = np.random.default_rng(seed + 12345)
    for rnd in range(rounds):
        t0 = time.perf_counter()
        eps = [ImproveEpisode(policy, limit, 0.0, seed)]
        bt = branch_point(best, brng) if branch and best is not None else None
        n_start = episodes if bt is None else max(1, episodes // 3)
        for k in range(n_start):
            seed += 1
            eps.append(ImproveEpisode(policy, limit, temps[k % len(temps)], seed))
        if bt is not None:
            pre = PrefixEpisode(link, policy, best, bt, log=log)
            eps.append(pre)
            for k in range(branch):
                seed += 1
                eps.append(BranchEpisode(policy, limit, temps[k % len(temps)], seed, pre))
        sess.status(f'Training {name}: round {rnd + 1}/{rounds}' +
                    (f", best {best['time_ms'] / 1000:.2f}s" if best and best['finished'] else ''))
        res = sess.run(eps, sim_only=True, keep=True)      # the game stays frozen until release()
        res = [r for r in res if not r.get('prefix')]
        elite = sorted(elite + res, key=score, reverse=True)[:elite_k]
        improved = best is None or score(elite[0]) > score(best)
        best = elite[0]
        with sess.hold():                                   # learning: game frozen, plugin kept waiting
            loss = fine_tune(policy, elite, steps, bs, lr, log)
        fin = [r['time_ms'] for r in res if r['finished']]
        greedy = res[0]
        rec = {'round': rnd + 1, 'greedy': greedy['time_ms'] if greedy['finished'] else f"{greedy['progress_m']} m",
               'finished': f'{len(fin)}/{len(res)}', 'fastest_this_round': min(fin) if fin else None,
               'best_ms': best['time_ms'], 'best_progress_m': best['progress_m'], 'loss': loss,
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
        best_txt = f"{best['time_ms'] / 1000:.2f}s" if best['finished'] else f"{best['progress_m']:.0f} m"
        stops = ', '.join(f'{n}x {k}' for k, n in Counter(rec['reasons']).most_common())
        br = f"; {branch} runs from {bt / 1000:.1f}s" if bt is not None else ''
        log(f"round {rnd + 1}: greedy {rec['greedy']}, {rec['finished']} finished, best {best_txt}"
            f"{' (new)' if improved else ''}, {rec['seconds']}s  [{stops}{br}]")
        if improved:
            with sess.hold():
                draw(link, best['path'])          # the best line so far, visible in the game
                (out / 'best_inputs.txt').write_text(tmi_script(best['ticks']), encoding='utf-8')
                (out / 'best_run.json').write_text(json.dumps({
                    'map': name, 'uid': uid, 'time_ms': best['time_ms'], 'progress_m': best['progress_m'],
                    'round': rnd + 1, 'ticks': best['ticks'], 'path': best['path']}), encoding='utf-8')
                torch.save(dict(torch.load(ckpt, map_location='cpu', weights_only=False),
                                state_dict=policy.model.state_dict(), improved_on=uid,
                                improve_best=best_txt), out / 'model.pt')
            if show and best['ticks'] and rnd + 1 < rounds:
                log(f'  showing the new best run ({best_txt}) in the game ...')
                sess.status(f'New best {best_txt} (round {rnd + 1}): showing it')
                sess.run([Playback(best['ticks'])], sim_only=False)    # ends the frozen state
    sess.release()
    log(f'best {best_txt}; inputs -> {out / "best_inputs.txt"}, model -> {out / "model.pt"}')
    if show and best['ticks']:
        sess.status(f'Best run ({best_txt}): showing it')
        sess.run([Playback(best['ticks'])], sim_only=False)
    sess.status(f'Training done: best {best_txt}. "Show best run" plays it again.')
    return history


def show_best(link, track_id: Optional[int] = None, speed: float = 1.0, log=print):
    """Play the best run that `improve` found on this map (runs/improve/<uid>/best_run.json)."""
    from .evaluate import map_list
    sess = GameSession(link, log)
    if track_id is not None:
        m = next((m for m in map_list(str(track_id))), None)
        if m is None or not sess.load_map(m['map_file'], m['uid']):
            raise SystemExit(f'map {track_id} did not load')
    else:
        t0 = time.monotonic()
        while sess.map is None and time.monotonic() - t0 < 10:
            try:
                sess.next(timeout=1.0)
            except TimeoutError:
                pass
        if sess.map is None:
            raise SystemExit('no map announced: restart the race in the game, or pass --map <TMX id>')
    f = OUT / safe(sess.map.uid) / 'best_run.json'
    if not f.exists():
        sess.status('No best run for this map yet: press "Train" first.')
        raise SystemExit(f'no best run for {sess.map.name!r} yet ({f}); train on the map first')
    run = json.loads(f.read_text(encoding='utf-8'))
    txt = f"{run['time_ms'] / 1000:.2f}s" if run['time_ms'] else f"{run['progress_m']:.0f} m"
    log(f"best run on {run['map']!r}: {txt} (round {run['round']})")
    sess.ensure_drivable()
    sess.status(f'Best run ({txt}): showing it')
    link.speed(speed)
    sess.run([Playback([tuple(t) for t in run['ticks']])], sim_only=False)
    sess.status(f'Best run shown ({txt}).')


def drive_preview(link, track_id: Optional[int] = None, speed: float = 1.0, ckpt: Path = None,
                  plans: int = 12, temps=(0.4, 0.7, 1.0), log=print):
    """"Route first, then drive": the model drives the map `plans` times without rendering (one
    greedy run, the rest sampled: keyboard steering is tapping, and always taking the most likely
    steering bin understeers), the best plan is drawn as small trigger boxes, then its exact
    inputs are played back visibly. The physics is deterministic, so the visible run is the plan."""
    from .evaluate import map_list
    from .policy import reference_positions, track_id_for
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    sess = GameSession(link, log)
    if track_id is not None:
        m = next((m for m in map_list(str(track_id))), None)
        if m is None or not sess.load_map(m['map_file'], m['uid']):
            raise SystemExit(f'map {track_id} did not load')
    else:
        t0 = time.monotonic()
        while sess.map is None and time.monotonic() - t0 < 10:
            try:
                sess.next(timeout=1.0)
            except TimeoutError:
                pass
        if sess.map is None:
            raise SystemExit('no map announced: restart the race in the game, or pass --map <TMX id>')
    uid = sess.map.uid
    ref = reference_positions(uid, track_id or track_id_for(uid))
    policy.reset([b.__dict__ for b in sess.map.blocks], ref[0] if ref else None)
    draw(link, [])
    sess.ensure_drivable()
    sess.status(f'AI is planning its line ({plans} tries, the game is frozen meanwhile) ...')
    log(f'planning {plans} runs (the game is frozen meanwhile) ...')
    pace, seed = Pace(), int(np.random.default_rng().integers(1_000_000))
    eps = [PlanEpisode(policy, 180000, 0.0, seed, 1, plans, pace, log)] + \
        [PlanEpisode(policy, 180000, temps[k % len(temps)], seed + k + 1, k + 2, plans, pace, log)
         for k in range(plans - 1)]
    res = sess.run(eps, sim_only=True)
    plan = max(res, key=score)
    fmt = lambda r: f"{r['time_ms'] / 1000:.2f}s" if r['finished'] else f"{r['progress_m']:.0f} m"
    log(f"plans: greedy {fmt(res[0])}, {sum(r['finished'] for r in res)}/{len(res)} finished, "
        f"best {fmt(plan)} (temperature {plan['temp']})")
    txt = f"{plan['time_ms'] / 1000:.2f}s" if plan['finished'] else f"{plan['reason']} after {plan['progress_m']:.0f} m"
    log(f'plan: {txt}, {len(plan["path"])} path samples')
    draw(link, plan['path'])
    sess.status(f'Best of {plans} plans ({txt}) drawn, the AI drives it now')
    link.speed(speed)
    real = sess.run([Playback(plan['ticks'])], sim_only=False)[0]
    n = min(len(plan['path']), len(real['path']))
    same = real['finished'] == plan['finished'] and real['time_ms'] == plan['time_ms'] and n > 0 and \
        max(float(np.linalg.norm(np.subtract(plan['path'][i], real['path'][i]))) for i in range(n)) < 1e-3
    rtxt = f"{real['time_ms'] / 1000:.2f}s" if real['finished'] else f"ended after {len(real['ticks']) / 100:.1f}s"
    log(f'drive: {rtxt}; identical to the plan: {same}')
    sess.status(f'AI: {rtxt} (plan {txt}, {"identical" if same else "DIFFERENT"})')
    return plan, real
