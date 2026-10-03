"""RL v2 without the game, batched: N cars of one TMNF-C world, one model call for all cars that
decide at a tick. Same episodes (improve.ImproveEpisode: stall rules, decisions), same rewards
and PPO update (rl.py) as `rl`; only where the runs come from differs.

    python tmdriver.py rl-vec --map 10036840 --iterations 50 --runs 64 --cars 32

Timing follows the game exactly (virtual_game.py): the STEP at race time t shows the state
before the tick from t, the action chosen there acts from t+10. Checkpoints and the finish come
from the race layer with the map's real triggers.
"""
import copy
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import protocol as P
from .fleet import policy_view
from .ghost_policy import GhostPolicy, forward_many
from .improve import ImproveEpisode, reference_setup, run_text, score, tmi_script
from .paths import DRIVER_CKPT, safe, tmi_scripts_dir, torch_device
from .rl import OUT, Batch, Critic, episode_steps, ppo_update


def make_step(sim, i: int, rt: int, race: dict) -> P.Step:
    st = sim.state(i)
    fin = race['finished']
    return P.Step(race_time=race['finish_ms'] if fin else rt, flags=P.F_FINISHED if fin else 0,
                  checkpoints=race['checkpoints'], pos=st['pos'], rot=st['rot'], vel=st['vel'],
                  ang_vel=st['ang_vel'], wheel_damper=st['wheel_damper'], wheel_contact=st['wheel_contact'],
                  wheel_sliding=st['wheel_sliding'], wheel_material=st['wheel_material'], gear=st['gear'],
                  rpm=st['rpm'], display_speed=int(np.linalg.norm(st['vel']) * 3.6), in_steer=0, in_gas=0, in_bits=0)


class Runner:
    """Runs episodes on the cars of a CarSim; a car that ends its episode starts the next one."""

    def __init__(self, policy: GhostPolicy, sim):
        self.policy, self.sim = policy, sim

    def _answer(self, ep, st):
        """The episode's action for one STEP (a single model call if it decides); None = ended."""
        kind, val = ep.act_begin(st)
        if kind == 'end':
            return None
        if kind == 'act':
            return val
        t, feats = val
        (x_row, out_row), = forward_many(self.policy, [feats])
        return ep.act_end(st, t, x_row, out_row)

    def run(self, episodes: List[ImproveEpisode]) -> List[dict]:
        sim, n = self.sim, self.sim.n
        queue = list(range(len(episodes)))
        results = [None] * len(episodes)
        car_ep = [None] * n
        rt = [0] * n
        pending = [(0, 0, 0)] * n

        def start(i, now: bool):
            """Next episode on car i, back at the start (the world is at race time 10 there, the
            car has not moved yet). It answers the STEP at race time 0 (its first action,
            pending); with now=True (mid-tick) also the identical STEP at 10, and returns that
            action for this tick, so the car steps along with the others."""
            while queue:
                e = queue.pop(0)
                sim.reset([i])
                ep = episodes[e]
                ep.begin(None)
                a0 = self._answer(ep, make_step(sim, i, 0, sim.race(i)))
                a1 = a0 if a0 is None or not now else self._answer(ep, make_step(sim, i, 10, sim.race(i)))
                if a0 is None or a1 is None:
                    results[e] = ep.result()
                    continue
                car_ep[i], rt[i], pending[i] = e, 10, a0
                return a1 if now else None
            car_ep[i] = None
            return None

        for i in range(n):
            start(i, now=False)
        while any(e is not None for e in car_ep):
            chosen = [None] * n
            decide = []
            for i in range(n):
                e = car_ep[i]
                if e is None:
                    continue
                st = make_step(sim, i, rt[i], sim.race(i))
                kind, val = episodes[e].act_begin(st)
                if kind == 'end' or (kind == 'act' and val is None):
                    results[e] = episodes[e].result()
                    chosen[i] = start(i, now=True)
                elif kind == 'act':
                    chosen[i] = val
                else:
                    decide.append((i, st, val))
            if decide:
                outs = forward_many(self.policy, [v[1] for _, _, v in decide])
                for (i, st, (t, _)), (x_row, out_row) in zip(decide, outs):
                    e = car_ep[i]
                    a = episodes[e].act_end(st, t, x_row, out_row)
                    if a is None:
                        results[e] = episodes[e].result()
                        chosen[i] = start(i, now=True)
                    else:
                        chosen[i] = a
            if all(e is None for e in car_ep):
                break
            acts = []
            for i in range(n):
                if car_ep[i] is None:
                    acts.append((0, 0, 0))
                    continue
                acts.append(pending[i])          # chosen one tick ago: the game's input delay
                pending[i] = chosen[i]
            sim.step(acts)
            for i in range(n):
                if car_ep[i] is not None:
                    rt[i] += 10
        return results


def rl_vec_train(map_file: Path, track_id: Optional[int] = None, iterations: int = 50, runs: int = 64,
                 cars: int = 32, use_line: bool = False, ckpt: Path = None, lr: float = 1e-5,
                 critic_lr: float = 1e-3, warmup: int = 2, kl_coef: float = 0.1, ent_coef: float = 0.01,
                 kl_max: float = 0.15, eval_temp: float = 0.7, seed: Optional[int] = None, log=print):
    from .replaybuild import map_blocks
    from .tmnfc_sim import CarSim
    from .virtual_game import tmi_waypoint
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    dev = policy.device
    uid, names, xyz, dirs = map_blocks(Path(map_file))
    blocks = [{'name': n, 'x': int(p[0]), 'y': int(p[1]), 'z': int(p[2]), 'dir': int(d), 'waypoint': tmi_waypoint(n)}
              for n, p, d in zip(names, xyz, dirs)]
    line, judge, line_txt = reference_setup(uid, track_id, use_line, log)
    policy.reset(blocks, line, judge)
    sim = CarSim(Path(map_file), f'rl_{safe(uid)}', n=cars)
    runner = Runner(policy, sim)
    name = Path(map_file).name.split('.')[0]
    out_dir = OUT / f'{safe(uid)}_vec'
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = 180000
    model = policy.model
    ref = copy.deepcopy(model).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    critic = Critic(model.cfg['d']).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    copt = torch.optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=0.0)
    seed = int(np.random.default_rng().integers(1_000_000)) if seed is None else seed
    log(f"rl-vec {name!r}: {iterations} iterations x ({runs} sampled + 1 evaluation run at temperature {eval_temp}) on {cars} cars in one TMNF-C world"
        f" ({'full race layer' if sim.full_route else 'start-only route'}), model {ckpt}")
    log(f'reference line: {line_txt}')
    best, history = None, []
    # guard against PPO collapse (seen 2026-09-26: 57/65 finished at iteration 21, 0/65 at 38 with
    # entropy and KL to the start model rising): keep the best policy, and roll back to it with a
    # halved learning rate when the finish rate halves or the policy drifts too far
    best_policy, best_rate = None, -1.0

    def save(best_now):
        torch.save(dict(torch.load(ckpt, map_location='cpu', weights_only=False), state_dict=model.state_dict(),
                        critic=critic.state_dict(), rl_on=uid, rl_best=run_text(best_now) if best_now else None),
                   out_dir / 'model.pt')
        if best_now is not None:
            (out_dir / 'best_run.json').write_text(json.dumps({
                'map': name, 'uid': uid, 'time_ms': best_now['time_ms'], 'progress_m': best_now['progress_m'],
                'line': line_txt, 'ticks': best_now['ticks'], 'path': best_now['path']}), encoding='utf-8')
            script_txt = tmi_script(best_now['ticks'])
            (out_dir / 'best_inputs.txt').write_text(script_txt, encoding='utf-8')
            try:
                (tmi_scripts_dir() / f'{safe(name)}.txt').write_text(script_txt, encoding='utf-8')
            except Exception:
                pass

    try:
        for it in range(iterations):
            t0 = time.perf_counter()
            model.eval()
            # the evaluation run: temperature eval_temp with a fixed seed, not argmax. Measured 2026-09-27 on
            # lolsport: argmax (and 0.3) never finishes, 0.7 finished 12/12 with the RL model (the policy
            # dithers between steer bins every 50 ms as a stand-in for analog steering)
            # Multi-temperature fleet schedule:
            # 1 evaluation run at eval_temp with a fixed seed.
            # Sampled runs scheduled smoothly from exploitation (0.35) to exploration (0.85):
            # - Lower temp runs (0.35-0.45) reliably finish and establish strong baseline lap times.
            # - Mid temp runs (0.50-0.70) refine cornering, drifts, and braking points.
            # - High temp runs (0.75-0.85) explore aggressive cuts and alternative paths.
            eps = [ImproveEpisode(policy_view(policy), limit, eval_temp, 12345)]
            t_min = max(0.25, round(eval_temp - 0.35, 2))
            t_max = min(0.98, round(eval_temp + 0.20, 2))
            for idx in range(runs):
                seed += 1
                frac = idx / max(1, runs - 1)
                temp = round(t_min + (t_max - t_min) * frac, 3)
                eps.append(ImproveEpisode(policy_view(policy), limit, temp, seed))
            res = runner.run(eps)
            t_roll = time.perf_counter() - t0
            evaluation = res[0]
            decisions, rewards, dones = [], [], []
            for r in res:
                st_ = episode_steps(r)
                if st_ is None:
                    continue
                dec, rew = st_
                decisions += dec
                rewards += rew
                dones += [0] * (len(dec) - 1) + [1]
            improved = False
            for r in res:
                if best is None or score(r) > score(best):
                    best, improved = r, True
            stats = ppo_update(model, ref, critic, opt, copt, Batch(decisions, rewards, dones, dev),
                               critic_only=it < warmup, kl_coef=kl_coef, ent_coef=ent_coef, seed=seed) \
                if decisions else {}
            save(best if improved else None)
            fin = [r['time_ms'] for r in res if r['finished']]
            rate = len(fin) / max(len(res), 1) - (float(np.median(fin)) / 1e6 if fin else 0.0)
            rolled = False
            if rate > best_rate:
                best_rate = rate
                best_policy = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(dict(torch.load(ckpt, map_location='cpu', weights_only=False),
                                state_dict=model.state_dict(), rl_on=uid), out_dir / 'model_best.pt')
            elif best_policy is not None and it >= warmup and not improved and (rate < 0.25 * best_rate or stats.get('kl_ref', 0) > kl_max):
                model.load_state_dict(best_policy)
                for gr in opt.param_groups:
                    gr['lr'] *= 0.5
                rolled = True
            ticks = sum(len(r['ticks']) for r in res)
            rec = {'iteration': it + 1, 'eval': run_text(evaluation), 'finished': f'{len(fin)}/{len(res)}',
                   'best': run_text(best), 'best_ms': best['time_ms'], 'decisions': len(decisions),
                   'median_finish_ms': int(np.median(fin)) if fin else None,
                   'rollout_s': round(t_roll, 1), 'seconds': round(time.perf_counter() - t0, 1),
                   'ticks_per_s': int(ticks / max(t_roll, 1e-6)), 'critic_only': it < warmup, **stats}
            history.append(rec)
            with open(out_dir / 'progress.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(rec, time=time.strftime('%H:%M:%S'))) + '\n')
            log(f"iteration {it + 1}: eval {rec['eval']}, {rec['finished']} finished"
                f"{', median ' + str(rec['median_finish_ms'] / 1000) + 's' if fin else ''}, best {rec['best']}"
                f"{' (new)' if improved else ''}, {rec['ticks_per_s']} ticks/s, "
                + (f"kl {stats.get('kl_ref')}, " if stats and not rec['critic_only'] else 'critic warm-up, ')
                + f"{rec['seconds']}s" + (f" | rolled back to the best policy, lr {opt.param_groups[0]['lr']:.2g}"
                                           if rolled else ''))
    except KeyboardInterrupt:
        log('stopped')
        save(best)
    finally:
        sim.close()
    log(f"model -> {out_dir / 'model.pt'}")
    return history
