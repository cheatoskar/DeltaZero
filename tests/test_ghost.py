"""Features v2: the live driver must see exactly what the training data contains.

For real trace runs (part 1 of the HF traces, which must be in data/hf/traces), build
the features twice:
  * the training path  (tracebuild: vectorised over a whole run on the 100 ms lattice)
  * the live path      (GhostPolicy fed one 10 ms tick at a time, with the in-between
                        ticks interpolated, exactly as the plugin would deliver them)
and require them to agree on every lattice sample. Also checks numpy vs torch block tokens.

    python tests/test_ghost.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tmdriver import ghost as G                       # noqa: E402
from tmdriver import protocol as P                    # noqa: E402
from tmdriver.ghost_policy import GhostPolicy         # noqa: E402
from tmdriver.ghost_torch import DriverNet2, block_features  # noqa: E402
from tmdriver.tracebuild import DIRS, MX_TO_GBX      # noqa: E402

TRACES = ROOT / 'data' / 'hf' / 'traces' / 'traces_tmnf_part0001.parquet'
BLOCKS = ROOT / 'data' / 'hf' / 'blocks' / 'blocks_tmnf_part0001.parquet'


def step(t, p):
    z3, z4 = np.zeros(3, np.float32), np.zeros(4, np.float32)
    return P.Step(race_time=t, flags=0, checkpoints=0, pos=np.asarray(p, np.float32),
                  rot=np.eye(3, dtype=np.float32).ravel(), vel=z3, ang_vel=z3, wheel_damper=z4,
                  wheel_contact=z4 > 0, wheel_sliding=z4 > 0, wheel_material=z4.astype(int),
                  gear=1, rpm=0.0, display_speed=0, in_steer=0, in_gas=0, in_bits=0)


def check_blocks_np_vs_torch(rng):
    n = 300
    center = rng.uniform(0, 1000, (n, 3))
    heading = rng.uniform(-3, 3, n)
    name = rng.integers(2, 50, n)
    wp = rng.integers(1, 6, n)
    pos = rng.uniform(100, 900, (7, 3))
    psi = rng.uniform(-3, 3, 7)
    tb = G.block_tiebreak(name, np.floor(center / 32).astype(int))
    a = G.block_features_np(center, heading, name, wp, tb, pos, psi)
    t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
    b = block_features(t(center)[None].expand(7, -1, -1), t(heading)[None].expand(7, -1),
                       t(name, torch.long)[None].expand(7, -1), t(wp, torch.long)[None].expand(7, -1),
                       t(tb)[None].expand(7, -1), torch.ones(7, n, dtype=torch.bool), t(pos), t(psi))
    assert np.array_equal(a[0], b[0].numpy()) and np.array_equal(a[3], b[3].numpy()), 'block ids differ'
    assert np.abs(a[2] - b[2].numpy()).max() < 1e-5, 'block features differ'
    print('block tokens numpy == torch: OK')


def main():
    rng = np.random.default_rng(0)
    check_blocks_np_vs_torch(rng)
    if not TRACES.exists():
        sys.exit(f'{TRACES} missing: copy traces part 1 there to run the live/training comparison')
    tr = pd.read_parquet(TRACES)
    bl = pd.read_parquet(BLOCKS, columns=['track_id', 'name', 'x', 'y', 'z', 'dir'])
    two = tr.groupby('track_id').trace_rank.nunique()
    maps = two[two == 2].index[:6].tolist()
    vocab = {n: i + 2 for i, n in enumerate(sorted(bl.name.unique()))}

    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / 'tiny.pt'
        torch.manual_seed(0)
        m = DriverNet2(len(vocab) + 2, d=32, layers=1, heads=2)
        torch.save({'kind': 'ghost2', 'features_version': G.FEATURES_VERSION, 'state_dict': m.state_dict(),
                    'cfg': m.cfg, 'meta': {'vocab': vocab, 'decide_every_ms': 50}}, ck)
        pol = GhostPolicy(ck)

    worst = {'state': 0.0, 'route': 0.0, 'blocks': 0.0}
    n_cmp = 0
    for tid in maps:
        g = bl[bl.track_id == tid]
        xyz = g[['x', 'y', 'z']].to_numpy(np.int64) + MX_TO_GBX
        dirs = np.array([DIRS[str(d)] for d in g.dir])
        plugin_blocks = [{'name': n, 'x': int(a), 'y': int(b), 'z': int(c), 'dir': int(d), 'waypoint': 0}
                         for n, (a, b, c), d in zip(g.name, xyz, dirs)]
        plugin_blocks += [{'name': 'StadiumGrass', 'x': i, 'y': 1, 'z': 5, 'dir': 0, 'waypoint': 0} for i in range(32)]
        mb = G.MapBlocks(g.name.tolist(), xyz, dirs, vocab)
        runs = {r: x.sort_values('time_ms') for r, x in tr[tr.track_id == tid].groupby('trace_rank')}
        for rank, run in runs.items():
            other = runs[[r for r in runs if r != rank][0]]
            pos = run[['x', 'y', 'z']].to_numpy(np.float64)
            line = G.Line(other[['x', 'y', 'z']].to_numpy(np.float64))
            # ---- training path (as in tracebuild.build_part)
            psi = G.headings(pos, mb.start_heading(pos[0]))
            idx = np.arange(len(pos))
            lags = G.lag_positions(pos, idx)
            st = G.state_features(lags, psi, psi[np.maximum(idx - 1, 0)], None, 0.0, 1.0)
            s = G.progress(line, pos)
            ro = G.route_features(line, s, pos, psi)
            bn, bw, bf, bm = G.block_features_np(mb.center, mb.heading, mb.name_id, mb.wp, mb.tb, pos, psi)
            # ---- live path: 10 ms ticks, lattice ticks carry the exact trace positions
            pol.reset(plugin_blocks, line)
            for i in range(len(pos)):
                for k in range(10):
                    t = i * 100 + k * 10
                    if k == 0:
                        p = pos[i]
                    elif i + 1 < len(pos):
                        p = pos[i] + (pos[i + 1] - pos[i]) * (k / 10)
                    else:
                        continue
                    pol.observe(step(t, p))
                    if k == 0:
                        ls, lr, lp, lpsi = pol.features(t)
                        tb = pol._bt
                        n2, w2, f2, m2 = block_features(*tb, torch.as_tensor(lp, dtype=torch.float32)[None],
                                                        torch.as_tensor(lpsi, dtype=torch.float32))
                        worst['state'] = max(worst['state'], float(np.abs(ls[0] - st[i]).max()))
                        worst['route'] = max(worst['route'], float(np.abs(lr[0] - ro[i]).max()))
                        if sorted(n2[0][m2[0]].tolist()) != sorted(bn[i][bm[i]].tolist()):
                            worst['blocks'] += 1
                        n_cmp += 1
    print(f'compared {n_cmp} lattice samples on {len(maps)} maps; worst abs diff: {json.dumps(worst)}')
    assert worst['state'] < 2e-3, 'state features differ between training and live'
    assert worst['route'] < 2e-3, 'route features differ between training and live'
    # float32 (live, GPU) vs float64 (numpy reference) may flip a block exactly at a 1/64 m
    # bucket edge or at the 200 m range limit; anything beyond a handful is a real bug
    assert worst['blocks'] <= max(2, n_cmp // 1000), 'block token sets differ between training and live'
    print('OK: live features == training features')


if __name__ == '__main__':
    main()
