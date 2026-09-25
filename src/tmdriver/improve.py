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


class Playback(Episode):
    """Plays a recorded tick sequence back (to show the best run)."""

    def __init__(self, ticks):
        self.by_t = {t: (s, g, b) for t, s, g, b in ticks}
        self.end = ticks[-1][0] if ticks else 0

    def begin(self, start):
        self.done = False

    def act(self, st):
        if st.finished or st.race_time > self.end + 1000:
            return None
        return self.by_t.get(st.race_time, (0, 0, 0))

    def result(self):
        return {}


def improve(link, track_id: Optional[int] = None, rounds: int = 20, episodes: int = 6, elite_k: int = 3,
            steps: int = 12, bs: int = 64, lr: float = 1e-4, temps=(0.4, 0.7, 1.0), show: bool = True,
            ckpt: Path = None, log=print):
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

    elite: List[dict] = []
    best: Optional[dict] = None
    history = []
    seed = 0
    for rnd in range(rounds):
        t0 = time.perf_counter()
        eps = [ImproveEpisode(policy, limit, 0.0, seed)]
        for k in range(episodes):
            seed += 1
            eps.append(ImproveEpisode(policy, limit, temps[k % len(temps)], seed))
        sess.status(f'Training {name}: round {rnd + 1}/{rounds}' +
                    (f", best {best['time_ms'] / 1000:.2f}s" if best and best['finished'] else ''))
        res = sess.run(eps, sim_only=True, keep=True)      # the game stays frozen until release()
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
               'reasons': [r['reason'] for r in res], 'start_speed_kmh': res[0].get('start_kmh')}
        history.append(rec)
        if max(r['progress_m'] for r in res) < 5:
            log('  WARNING: the car did not move in any run. Is a screen blocking the race start '
                '("press any key", the map intro)? Click into the game or press Enter once.')
        with open(out / 'progress.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(rec, time=time.strftime('%H:%M:%S'))) + '\n')
        best_txt = f"{best['time_ms'] / 1000:.2f}s" if best['finished'] else f"{best['progress_m']:.0f} m"
        stops = ', '.join(f'{n}x {k}' for k, n in Counter(rec['reasons']).most_common())
        log(f"round {rnd + 1}: greedy {rec['greedy']}, {rec['finished']} finished, best {best_txt}"
            f"{' (new)' if improved else ''}, {rec['seconds']}s  [{stops}; greedy at 1 s: {rec['start_speed_kmh']} km/h]")
        if improved:
            with sess.hold():
                draw(link, best['path'])          # the best line so far, visible in the game
                (out / 'best_inputs.txt').write_text(tmi_script(best['ticks']), encoding='utf-8')
                (out / 'best_run.json').write_text(json.dumps({
                    'map': name, 'uid': uid, 'time_ms': best['time_ms'], 'progress_m': best['progress_m'],
                    'round': rnd + 1, 'ticks': best['ticks']}), encoding='utf-8')
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
    sess.status(f'Best run ({txt}): showing it')
    link.speed(speed)
    sess.run([Playback([tuple(t) for t in run['ticks']])], sim_only=False)
    sess.status(f'Best run shown ({txt}).')


def drive_preview(link, track_id: Optional[int] = None, speed: float = 1.0, ckpt: Path = None, log=print):
    """"Route first, then drive": the model drives the map once without rendering, its path
    is drawn as small trigger boxes, then it drives visibly. The physics is deterministic and
    the greedy policy too, so the visible run follows the drawn line exactly."""
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
    sess.status('AI is planning its line ...')
    plan = sess.run([ImproveEpisode(policy, 180000, 0.0, 0)], sim_only=True)[0]
    txt = f"{plan['time_ms'] / 1000:.2f}s" if plan['finished'] else f"{plan['reason']} after {plan['progress_m']:.0f} m"
    log(f'plan: {txt}, {len(plan["path"])} path samples')
    draw(link, plan['path'])
    sess.status(f'Planned line ({txt}) drawn, the AI drives it now')
    link.speed(speed)
    real = sess.run([ImproveEpisode(policy, 180000, 0.0, 0)], sim_only=False)[0]
    same = plan['ticks'] == real['ticks']
    rtxt = f"{real['time_ms'] / 1000:.2f}s" if real['finished'] else f"{real['reason']} after {real['progress_m']:.0f} m"
    log(f'drive: {rtxt}; identical to the plan: {same}')
    sess.status(f'AI: {rtxt} (plan {txt}, {"identical" if same else "DIFFERENT"})')
    return plan, real
