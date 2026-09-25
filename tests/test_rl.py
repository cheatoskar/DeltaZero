"""RL v2 maths without the game: GAE, the factorised action distribution, and that a PPO update
moves probability towards rewarded actions.

    python tests/test_rl.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tmdriver import ghost as G  # noqa: E402
from tmdriver.ghost_torch import DriverNet2  # noqa: E402
from tmdriver.rl import Batch, Critic, dist_terms, episode_steps, gae, ppo_update  # noqa: E402


def test_gae():
    rng = np.random.default_rng(0)
    r = rng.normal(size=7)
    v = rng.normal(size=7)
    done = np.array([0, 0, 1, 0, 0, 0, 1])
    gamma, lam = 0.9, 0.8
    adv, ret = gae(r, v, done, gamma, lam)
    # by hand: delta_t, then the discounted sum of deltas inside each episode
    for start, end in ((0, 3), (3, 7)):
        for t in range(start, end):
            want = 0.0
            for k in range(t, end):
                nxt = v[k + 1] if k + 1 < end else 0.0
                delta = r[k] + gamma * nxt - v[k]
                want += (gamma * lam) ** (k - t) * delta
            assert abs(adv[t] - want) < 1e-9, (t, adv[t], want)
    assert np.allclose(ret, adv + v)
    # lambda = 1: advantage = discounted return - value
    adv1, _ = gae(r, v, done, gamma, 1.0)
    for t in range(3):
        g = sum(gamma ** (k - t) * r[k] for k in range(t, 3))
        assert abs(adv1[t] - (g - v[t])) < 1e-9
    print('OK: GAE matches the hand computation')


def test_distribution():
    torch.manual_seed(0)
    out = {'steer': torch.randn(1, G.STEER_BINS), 'gas': torch.randn(1), 'brake': torch.randn(1)}
    ref = {'steer': torch.randn(1, G.STEER_BINS), 'gas': torch.randn(1), 'brake': torch.randn(1)}
    acts = list(itertools.product(range(G.STEER_BINS), (0.0, 1.0), (0.0, 1.0)))
    lps = []
    for b, g, br in acts:
        lp, ent, kl = dist_terms(out, torch.tensor([b]), torch.tensor([g]), torch.tensor([br]), ref)
        lps.append(float(lp))
    p = np.exp(lps)
    assert abs(p.sum() - 1.0) < 1e-5, p.sum()
    assert abs(float(ent) - float(-(p * np.array(lps)).sum())) < 1e-4
    lq = [float(dist_terms(ref, torch.tensor([b]), torch.tensor([g]), torch.tensor([br]))[0]) for b, g, br in acts]
    assert abs(float(kl) - float((p * (np.array(lps) - np.array(lq))).sum())) < 1e-4
    _, _, kl0 = dist_terms(out, torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0]), out)
    assert abs(float(kl0)) < 1e-6
    print('OK: factorised policy: probabilities sum to 1, entropy and KL match the enumeration')


def test_episode_steps():
    dec = [(None, 10, True, False, t, p) for t, p in ((0, 0.0), (50, 3.0), (100, 7.0))]
    r = {'decisions': dec, 'race_ms': 150, 'final_progress': 12.0, 'finished': True}
    d, rew = episode_steps(r)
    assert len(d) == 3
    assert np.allclose(rew, [0.03 - 0.05, 0.04 - 0.05, 0.05 - 0.05 + 1.0]), rew
    d2, _ = episode_steps(r, start_t=50)       # a branch run: only its own decisions
    assert len(d2) == 2
    print('OK: rewards = 0.01 x progress - 0.05 per decision, +1 at the finish')


def test_ppo_direction():
    torch.manual_seed(0)
    model = DriverNet2(n_names=8, d=32, layers=1, heads=4)
    ref = DriverNet2(n_names=8, d=32, layers=1, heads=4)
    ref.load_state_dict(model.state_dict())
    critic = Critic(32)
    n = 256
    good, bad = 5, 15
    x = (torch.zeros(1, G.STATE_DIM), torch.zeros(1, len(G.ROUTE_D), 3), torch.zeros(1, G.K_BLOCKS, dtype=torch.long),
         torch.zeros(1, G.K_BLOCKS, dtype=torch.long), torch.zeros(1, G.K_BLOCKS, G.BLOCK_FEAT),
         torch.ones(1, G.K_BLOCKS, dtype=torch.bool))
    decisions, rewards = [], []
    for i in range(n):
        b = good if i % 2 == 0 else bad
        decisions.append((x, b, True, False, 0, 0.0))
        rewards.append(1.0 if b == good else -1.0)
    dones = [1] * n                               # one-step episodes: advantage = reward - value
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    copt = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    def probs():
        with torch.no_grad():
            pr = torch.softmax(model(*x)['steer'][0], -1)
        return float(pr[good]), float(pr[bad])

    g0, b0 = probs()
    for it in range(5):
        stats = ppo_update(model, ref, critic, opt, copt, Batch(decisions, rewards, dones, 'cpu'),
                           epochs=2, mb=64, kl_coef=0.01, seed=it)
    g1, b1 = probs()
    assert g1 > g0 * 1.5 and b1 < b0 / 1.5, (g0, g1, b0, b1)
    assert stats['kl_ref'] > 0 and np.isfinite(stats['v_loss']), stats
    print(f'OK: PPO moves probability to the rewarded action ({g0:.3f} -> {g1:.3f}) and away from the '
          f'punished one ({b0:.3f} -> {b1:.3f}); KL to the start {stats["kl_ref"]:.3f}')


if __name__ == '__main__':
    test_gae()
    test_distribution()
    test_episode_steps()
    test_ppo_direction()
