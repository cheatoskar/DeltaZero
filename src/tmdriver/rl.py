"""RL v2, milestone RL-1: PPO on one map.

    python tmdriver.py rl --map 414041 --iterations 30        (serve must NOT run)

Every iteration:
  1. runs in simulation-only mode on every game instance (fleet.py): one greedy run (only
     measured), `runs` sampled runs at temperature 1 (so the recorded actions follow the policy
     exactly), and from the second iteration on `branch` runs that start shortly before the
     point where the best run got stuck (as in RL v1);
  2. per decision (50 ms) a reward: 0.01 x metres of judged progress - 0.05, +1 at the finish
     (progress along the reference line, the judge run, or track cells: improve.reference_setup);
  3. advantages with GAE from a critic that reads the model's state token (detached: the critic
     does not change the driving features);
  4. PPO update of the whole model: clipped ratio, entropy bonus, and a KL leash to the frozen
     starting model so that the pretrained driving knowledge is kept. The first iterations
     train only the critic (its early advantages would be noise).

The episodes, stall rules, branching, best-run bookkeeping and the stop handling are RL v1's
(improve.py); only what is learned from the runs differs.
"""
import copy
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

from .ghost_policy import GhostPolicy
from .improve import (BranchEpisode, ImproveEpisode, Playback, PrefixEpisode, branch_point, draw,
                      helper_sessions, open_map, reference_setup, run_text, score, tmi_script)
from .paths import DRIVER_CKPT, RUNS, safe, torch_device
from .session import GameSession

OUT = RUNS / 'rl'
PROGRESS_W = 0.01        # reward per metre of judged progress
TIME_W = 0.05            # penalty per 50 ms (1 per second)
FINISH_BONUS = 1.0


class Critic(nn.Module):
    """Value of a decision from the model's state token (detached)."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h):
        return self.net(h.detach().float()).squeeze(-1)


# ---------------------------------------------------------------------- returns

def episode_steps(r: dict, start_t: int = 0):
    """One run -> its own decisions (from start_t on; a branch run also carries its parent's)
    and a reward per decision; None if it made no decision."""
    dec = [d for d in r['decisions'] if d[4] >= start_t]
    if not dec:
        return None
    t = [d[4] for d in dec] + [r['race_ms']]
    p = [d[5] for d in dec] + [r['final_progress']]
    rewards = [PROGRESS_W * (p[i + 1] - p[i]) - TIME_W * (t[i + 1] - t[i]) / 50.0 for i in range(len(dec))]
    if r['finished']:
        rewards[-1] += FINISH_BONUS
    return dec, rewards


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float):
    """Generalised advantage estimation over concatenated episodes (dones marks each episode's
    last decision; the value after it is 0). Returns (advantages, returns)."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for i in range(n - 1, -1, -1):
        nxt = 0.0 if dones[i] else values[i + 1]
        delta = rewards[i] + gamma * nxt - values[i]
        last = delta + gamma * lam * (0.0 if dones[i] else last)
        adv[i] = last
    return adv, adv + values


# ---------------------------------------------------------------------- distribution

def _bern_logp(logit, x):
    return -TF.binary_cross_entropy_with_logits(logit.float(), x, reduction='none')


def _bern_ent(logit):
    return TF.binary_cross_entropy_with_logits(logit.float(), torch.sigmoid(logit.float()), reduction='none')


def _bern_kl(logit_p, logit_q):
    p, lp, lq = torch.sigmoid(logit_p.float()), logit_p.float(), logit_q.float()
    return (p * (TF.logsigmoid(lp) - TF.logsigmoid(lq)) + (1 - p) * (TF.logsigmoid(-lp) - TF.logsigmoid(-lq)))


def dist_terms(out, bins, gas, brake, ref=None):
    """log-probability of the actions, entropy, and KL(current || ref) of the factorised policy
    (21 steering bins, gas and brake as Bernoulli)."""
    ls = TF.log_softmax(out['steer'].float(), -1)
    logp = ls.gather(1, bins[:, None]).squeeze(1) + _bern_logp(out['gas'], gas) + _bern_logp(out['brake'], brake)
    ent = -(ls.exp() * ls).sum(-1) + _bern_ent(out['gas']) + _bern_ent(out['brake'])
    kl = None
    if ref is not None:
        lr_ = TF.log_softmax(ref['steer'].float(), -1)
        kl = (ls.exp() * (ls - lr_)).sum(-1) + _bern_kl(out['gas'], ref['gas']) + _bern_kl(out['brake'], ref['brake'])
    return logp, ent, kl


# ---------------------------------------------------------------------- update

class Batch:
    """The decisions of one iteration, stacked (inputs stay on the model's device)."""

    def __init__(self, decisions: List[tuple], rewards: List[float], dones: List[int], device):
        self.x = [torch.cat([d[0][k] for d in decisions]) for k in range(6)]
        self.bins = torch.tensor([d[1] for d in decisions], device=device)
        self.gas = torch.tensor([float(d[2]) for d in decisions], device=device)
        self.brake = torch.tensor([float(d[3]) for d in decisions], device=device)
        self.rewards = np.asarray(rewards, dtype=np.float64)
        self.dones = np.asarray(dones, dtype=np.int64)
        self.n = len(decisions)

    def inputs(self, idx):
        return [t[idx] for t in self.x]


@torch.no_grad()
def forward_all(model, batch: Batch, mb: int = 512):
    outs = []
    for i in range(0, batch.n, mb):
        idx = torch.arange(i, min(i + mb, batch.n), device=batch.bins.device)
        o = model(*batch.inputs(idx))
        outs.append({k: v.detach() for k, v in o.items()})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def ppo_update(model, ref, critic, opt, copt, batch: Batch, gamma=0.995, lam=0.95, epochs=4, mb=256,
               clip=0.2, ent_coef=0.01, kl_coef=0.1, critic_only=False, seed=0) -> dict:
    """One PPO update on a batch. Returns statistics."""
    dev = batch.bins.device
    was_training = model.training
    model.eval()
    old = forward_all(model, batch)
    with torch.no_grad():
        ref_out = forward_all(ref, batch)
        old_logp, _, _ = dist_terms(old, batch.bins, batch.gas, batch.brake)
        values = critic(old['h']).double().cpu().numpy()
    adv, ret = gae(batch.rewards, values, batch.dones, gamma, lam)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=dev)
    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=dev)
    adv_n = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    g = torch.Generator().manual_seed(seed)
    stats = {'n': batch.n, 'v_loss': 0.0, 'pi_loss': 0.0, 'entropy': 0.0, 'kl_ref': 0.0, 'clipfrac': 0.0,
             'approx_kl': 0.0, 'adv_mean': float(adv.mean()), 'ret_mean': float(ret.mean())}
    k = 0
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(batch.n, generator=g).to(dev)
        for i in range(0, batch.n, mb):
            idx = perm[i:i + mb]
            copt.zero_grad()
            opt.zero_grad()
            if critic_only:
                # warm-up: the policy stays as it is, the critic learns the returns
                v_loss = TF.smooth_l1_loss(critic(old['h'][idx]), ret_t[idx])
                v_loss.backward()
            else:
                out = model(*batch.inputs(idx))
                v_loss = TF.smooth_l1_loss(critic(out['h']), ret_t[idx])   # h is detached in Critic
                sub_ref = {kk: vv[idx] for kk, vv in ref_out.items()}
                logp, ent, kl = dist_terms(out, batch.bins[idx], batch.gas[idx], batch.brake[idx], sub_ref)
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_n[idx]
                pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                loss = pi_loss - ent_coef * ent.mean() + kl_coef * kl.mean() + v_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    stats['pi_loss'] += float(pi_loss)
                    stats['entropy'] += float(ent.mean())
                    stats['kl_ref'] += float(kl.mean())
                    stats['clipfrac'] += float(((ratio - 1).abs() > clip).float().mean())
                    stats['approx_kl'] += float((old_logp[idx] - logp).mean())
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            copt.step()
            stats['v_loss'] += float(v_loss.detach())
            k += 1
    for key in ('v_loss', 'pi_loss', 'entropy', 'kl_ref', 'clipfrac', 'approx_kl'):
        stats[key] = round(stats[key] / max(k, 1), 5)
    model.train(was_training)
    model.eval()
    return stats


# ---------------------------------------------------------------------- training loop

def rl_train(link, track_id: Optional[int] = None, iterations: int = 30, runs: int = 8, branch: int = 4,
             use_line: bool = False, helpers=(), show: bool = True, ckpt: Path = None, lr: float = 1e-5,
             critic_lr: float = 1e-3, warmup: int = 2, kl_coef: float = 0.1, ent_coef: float = 0.01,
             seed: Optional[int] = None, log=print):
    from .fleet import Fleet, policy_view
    ckpt = Path(ckpt or DRIVER_CKPT)
    policy = GhostPolicy(ckpt, device=torch_device())
    dev = policy.device
    sess = GameSession(link, log)
    name, uid, best_ms = open_map(sess, track_id, 5, log)
    line, judge, line_txt = reference_setup(uid, track_id, use_line, log)
    policy.reset([b.__dict__ for b in sess.map.blocks], line, judge)
    draw(link, [])
    fleet = Fleet(sess, helper_sessions(helpers, uid, track_id, log))
    pols = [policy] + [policy_view(policy) for _ in fleet.sessions[1:]]
    out_dir = OUT / safe(uid)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(2 * best_ms + 10000) if best_ms else 180000
    model = policy.model
    ref = copy.deepcopy(model).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    critic = Critic(model.cfg['d']).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    copt = torch.optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=0.0)
    log(f"rl {name!r}: {iterations} iterations x ({runs} sampled + greedy{f' + {branch} branch' if branch else ''}) "
        f"on {len(fleet)} game instance(s), model {ckpt}")
    log(f"reference line: {line_txt}")
    sess.ensure_drivable()
    seed = int(np.random.default_rng().integers(1_000_000)) if seed is None else seed
    brng = np.random.default_rng(seed + 12345)
    log(f'seed {seed}   (Ctrl+C here or Stop in the game: end the training and show the best run)')
    best, history, it = None, [], 0
    line_tag = 'with line' if line is not None else 'no line'

    def save(best_now):
        torch.save(dict(torch.load(ckpt, map_location='cpu', weights_only=False),
                        state_dict=model.state_dict(), critic=critic.state_dict(), rl_on=uid,
                        rl_best=run_text(best_now) if best_now else None), out_dir / 'model.pt')
        if best_now is not None:
            (out_dir / 'best_run.json').write_text(json.dumps({
                'map': name, 'uid': uid, 'time_ms': best_now['time_ms'], 'progress_m': best_now['progress_m'],
                'cps': best_now.get('cps'), 'line': line_txt, 'ticks': best_now['ticks'],
                'path': best_now['path']}), encoding='utf-8')
            (out_dir / 'best_inputs.txt').write_text(tmi_script(best_now['ticks']), encoding='utf-8')

    stopped = False
    try:
        for it in range(iterations):
            t0 = time.perf_counter()
            n = len(fleet)
            groups = [[] for _ in range(n)]
            groups[0].append(ImproveEpisode(pols[0], limit, 0.0, seed))           # greedy: measured only
            for k in range(runs):
                seed += 1
                i = k % n
                groups[i].append(ImproveEpisode(pols[i], limit, 1.0, seed))
            bt = branch_point(best, brng) if branch and best is not None else None
            if bt is not None:
                share = [branch // n + (1 if i < branch % n else 0) for i in range(n)]
                for i in range(n):
                    if share[i]:
                        pre = PrefixEpisode(fleet.sessions[i].link, pols[i], best, bt, log=log)
                        groups[i].append(pre)
                        for _ in range(share[i]):
                            seed += 1
                            groups[i].append(BranchEpisode(pols[i], limit, 1.0, seed, pre))
            for s_ in fleet.sessions:
                s_.status(f'RL {name} ({line_tag}): iteration {it + 1}/{iterations}' +
                          (f', best {run_text(best)}' if best else ''))
            outs = fleet.run(groups, sim_only=True, keep=True)
            res = [r for g in outs for r in g if not r.get('prefix')]
            greedy, sampled = res[0], res[1:]
            decisions, rewards, dones = [], [], []
            for r in sampled:
                st = episode_steps(r, r.get('branch_t') or 0)
                if st is None:
                    continue
                dec, rew = st
                decisions += dec
                rewards += rew
                dones += [0] * (len(dec) - 1) + [1]
            improved = False
            for r in res:
                if best is None or score(r) > score(best):
                    best, improved = r, True
            stats = {}
            if decisions:
                with fleet.hold():                           # learning: games frozen, plugins kept waiting
                    stats = ppo_update(model, ref, critic, opt, copt, Batch(decisions, rewards, dones, dev),
                                       critic_only=it < warmup, kl_coef=kl_coef, ent_coef=ent_coef, seed=seed)
                    save(best if improved else None)
            fin = [r['time_ms'] for r in res if r['finished']]
            rec = {'iteration': it + 1, 'greedy': run_text(greedy), 'finished': f'{len(fin)}/{len(res)}',
                   'best': run_text(best), 'best_ms': best['time_ms'], 'decisions': len(decisions),
                   'return_mean': round(float(np.sum(rewards)) / max(len(sampled), 1), 3),
                   'seconds': round(time.perf_counter() - t0, 1), 'line': line is not None,
                   'critic_only': it < warmup, **stats,
                   'reasons': [r['reason'] for r in res]}
            history.append(rec)
            with open(out_dir / 'progress.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(rec, time=time.strftime('%H:%M:%S'))) + '\n')
            log(f"iteration {it + 1}: greedy {rec['greedy']}, {rec['finished']} finished, best {rec['best']}"
                f"{' (new)' if improved else ''}, return/run {rec['return_mean']}, "
                + (f"entropy {stats.get('entropy')}, kl {stats.get('kl_ref')}, clip {stats.get('clipfrac')}, "
                   if stats and not rec['critic_only'] else 'critic warm-up, ')
                + f"v_loss {stats.get('v_loss')}, {rec['seconds']}s")
            if improved and show and best['ticks'] and it + 1 < iterations:
                with fleet.hold():
                    draw(link, best['path'])
                with fleet.hold(skip_main=True):
                    sess.status(f'New best {run_text(best)} (iteration {it + 1}): showing it. Respawn = skip')
                    sess.run([Playback(best['ticks'])], sim_only=False)
    except KeyboardInterrupt:
        stopped = True
        log(f'\nstopped in iteration {it + 1}' + (f': best so far {run_text(best)}' if best else ''))
        fleet.recover()
        save(best)
    fleet.release()
    if best is not None and best['ticks'] and (show or stopped):
        draw(link, best['path'])
        sess.status(f'Best run ({run_text(best)}): showing it. Respawn = skip')
        sess.run([Playback(best['ticks'])], sim_only=False)
    sess.status(f'RL {"stopped" if stopped else "done"}: best {run_text(best) if best else "-"}.')
    log(f"model -> {out_dir / 'model.pt'}")
    return history
