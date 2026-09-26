"""Learn the route planner's costs from record lines, and measure it on maps it never saw.

    python tmdriver.py route-learn --maps 400       (fit + tune + held-out report)

Maps: sim-night's maps with an exact run; the fastest exact run's positions are the record
line. Split by track id: held-out maps (collect.holdout) are only ever used for the report.

1. For every training map: node features (route_planner.Grid.features) and whether the record
   line passes the node. A logistic model (L2, Newton) gives p(driver there | features).
2. Parameters (max drop, jump and drop costs, weight of the learned cost) are tuned by random
   search on training maps (metric: mean distance record->route + route->record, failures cost).
3. Report on held-out maps: fixed rules vs learned model + tuned parameters.
"""
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .paths import DATA, TMX, safe
from . import route_planner as RP

FAIL_M = 200.0              # a map without a route counts as this far off


def map_list(n: int, holdout: bool) -> List[dict]:
    from .collect import holdout as is_holdout
    from .tmnfc import RESIM_C
    best = {}
    for line in (RESIM_C / 'index.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('exact') and (r['track_id'] not in best or r['recorded_ms'] < best[r['track_id']]['recorded_ms']):
            best[r['track_id']] = r
    maps = {}
    for f in (DATA / 'maps').glob('*_*.Challenge.Gbx'):
        try:
            maps[int(f.name.split('.')[0].rsplit('_', 1)[1])] = f
        except ValueError:
            pass
    out = []
    for tid in sorted(best, key=lambda t: (hash((t, 7)) & 0xFFFF)):        # a fixed pseudo-random order
        if tid in maps and bool(is_holdout(tid)) == holdout:
            b = best[tid]
            out.append({'track_id': tid, 'map': str(maps[tid]), 'npz': str(RESIM_C / str(tid) / f"{b['replay']}.npz"),
                        'replay': str(TMX / safe(b['uid']) / f"{b['replay']}.Replay.Gbx")})
        if len(out) >= n:
            break
    return out


def record_line(m: dict) -> np.ndarray:
    z = np.load(m['npz'], allow_pickle=True)
    t = z['t']
    return z['pos'][t % 100 == 0].astype(np.float64)


def _planner(m):
    return RP.Planner(Path(m['map']), str(m['track_id']))


def _features(m):
    """(features, labels) of one map, negatives subsampled to 10x the positives."""
    try:
        P = _planner(m)
        pos = P.on_line(record_line(m))
        if len(pos) < 10:
            return None
        y = np.zeros(P.g.n, bool)
        y[pos] = True
        neg = np.flatnonzero(~y)
        rng = np.random.default_rng(m['track_id'])
        neg = rng.choice(neg, size=min(len(neg), 10 * len(pos)), replace=False)
        idx = np.concatenate([pos, neg])
        return P.feat[idx].astype(np.float32), y[idx]
    except Exception:                              # noqa: BLE001
        return None


def fit_logistic(X, y, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def score(m, planner, params, model) -> float:
    route = planner.plan(params, model)
    if route is None or len(route) < 2:
        return FAIL_M
    c = RP.compare_to_ghost(route, record_line(m))
    return min(FAIL_M, 0.5 * (c['mean_m'] + c['route_off_m']))


def _score_job(args):
    m, trials, model = args
    try:
        P = _planner(m)
    except Exception:                              # noqa: BLE001
        return [FAIL_M] * len(trials)
    return [score(m, P, p, model) for p in trials]


def run(n_maps: int = 400, n_tune: int = 60, n_eval: int = 150, trials: int = 24, workers: int = 0, log=print):
    workers = workers or max(1, (mp.cpu_count() or 2) - 1)
    ctx = mp.get_context('spawn')
    train = map_list(n_maps, holdout=False)
    held = map_list(n_eval, holdout=True)
    log(f'{len(train)} training maps, {len(held)} held-out maps, {workers} workers')
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        data = [d for d in pool.imap_unordered(_features, train) if d is not None]
    X = np.concatenate([d[0] for d in data]).astype(np.float64)
    y = np.concatenate([d[1] for d in data]).astype(np.float64)
    w = fit_logistic(X, y)
    acc = float(((X @ w > 0) == (y > 0.5)).mean())
    log(f'cost model: {len(data)} maps, {len(y)} nodes ({y.mean():.1%} on a record line), train acc {acc:.3f}, '
        f'{time.time() - t0:.0f}s')
    model = {'w': w.tolist(), 'features': 'route_planner.Grid.features v1'}
    # random search of the parameters on training maps
    rng = np.random.default_rng(0)
    cands = [dict(RP.DEFAULT)] + [{
        'climb_per_cell': float(rng.uniform(2.0, 4.0)), 'max_drop': float(rng.choice([8, 16, 32, 60])),
        'bridge_cost': float(rng.choice([5.0, 10.0, 20.0, 50.0])),
        'jump_cells': int(rng.choice([4, 6, 8, 10, 12])), 'jump_rise': float(rng.choice([0.0, 1.0, 2.0])),
        'jump_cost': float(rng.uniform(1.0, 6.0)), 'drop_cost': float(rng.uniform(0.0, 2.0)),
        'w_learned': float(rng.uniform(0.2, 3.0))} for _ in range(trials - 1)]
    tune = train[:n_tune]
    with ctx.Pool(workers) as pool:
        res = np.array(pool.map(_score_job, [(m, cands, model) for m in tune]))
    means = res.mean(0)
    best = int(np.argmin(means))
    params = cands[best]
    log(f'tuning on {len(tune)} maps: default {means[0]:.1f} m, best {means[best]:.1f} m -> {params}')
    model['params'] = params
    # held-out report: fixed rules (no model, default params) vs learned
    with ctx.Pool(workers) as pool:
        rep = np.array(pool.map(_score_job, [(m, [dict(RP.DEFAULT)], None) for m in held]))[:, 0]
        rep2 = np.array(pool.map(_score_job, [(m, [params], model) for m in held]))[:, 0]
    msg = (f'held-out {len(held)} maps: fixed rules {rep.mean():.1f} m (median {np.median(rep):.1f}, '
           f'{(rep < 16).mean():.0%} under 16 m, {(rep >= FAIL_M).mean():.0%} failed) -> learned {rep2.mean():.1f} m '
           f'(median {np.median(rep2):.1f}, {(rep2 < 16).mean():.0%} under 16 m, {(rep2 >= FAIL_M).mean():.0%} failed)')
    log(msg)
    model['report'] = msg
    RP.MODEL.write_text(json.dumps(model, indent=1), encoding='utf-8')
    log(f'-> {RP.MODEL}')
    return model
