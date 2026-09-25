"""Re-simulated replays -> training arrays (data/m1/dataset.npz).

Two conventions are measured here from the re-simulated runs, and the build refuses to
continue if the evidence is weak or contradictory:
  * which keyboard key equals positive analog steer (turn direction must agree);
  * how a block's `dir` (0..3) maps to a world heading (start block vs car at t=0).

Reference path for a run: the fastest OTHER exact run of the same map. Using the run
itself would hand the model its own future path, which is the answer. Runs of maps with
a single exact run are skipped for that reason.
"""
import collections
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import features as F
from . import protocol as P
from .calib import Calibration, turn_rate
from .collect import M1, MANIFEST
from .line import RefLine
from .paths import CALIBRATION, LIVE_MAPS, safe
from .resim import RESIM

DATASET = M1 / 'dataset.npz'


def load_runs(log=print) -> Dict[int, List[dict]]:
    """Exact re-simulated runs grouped by TMX track id (latest index entry wins)."""
    latest = {}
    for line in (RESIM / 'index.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        latest[(r['track_id'], r['replay'])] = r
    runs = collections.defaultdict(list)
    bad = 0
    for r in latest.values():
        if not r['exact']:
            bad += 1
            continue
        d = dict(np.load(r['file']))
        d.pop('meta', None)
        runs[int(r['track_id'])].append({'meta': r, 'a': d, 'finish': int(r['finished_at'])})
    log(f'{sum(len(v) for v in runs.values())} exact runs on {len(runs)} maps ({bad} not exact, skipped)')
    return runs


def check_steer(runs) -> Dict:
    """Keyboard Right must turn the car the same way as some sign of analog steer."""
    kb, an = [], []
    for rs in runs.values():
        for r in rs:
            a = r['a']
            w = turn_rate(a['vel'].astype(np.float64))
            fast = np.linalg.norm(a['vel'], axis=1) > 10
            bits, st = a['act_bits'], a['act_steer']
            analog = (bits & P.STEER_ANALOG) > 0
            right_only = fast & ~analog & ((bits & P.RIGHT) > 0) & ((bits & P.LEFT) == 0)
            left_only = fast & ~analog & ((bits & P.LEFT) > 0) & ((bits & P.RIGHT) == 0)
            kb += list(np.sign(w[right_only])) + list(-np.sign(w[left_only]))
            strong = fast & analog & (np.abs(st) > 0.3 * P.STEER_MAX)
            an += list(np.sign(w[strong]) * np.sign(st[strong]))
    kb, an = np.array(kb), np.array(an)
    if len(kb) < 200 or len(an) < 200:
        raise RuntimeError(f'too little evidence for the steer convention (keyboard {len(kb)}, analog {len(an)})')
    kb_dir, an_dir = np.sign(kb.mean()), np.sign(an.mean())
    agree_kb, agree_an = float(np.mean(kb == kb_dir)), float(np.mean(an == an_dir))
    if min(agree_kb, agree_an) < 0.7:
        raise RuntimeError(f'steer convention unclear: keyboard agreement {agree_kb:.2f}, analog {agree_an:.2f}')
    return {'right_sign': int(kb_dir * an_dir), 'keyboard_agreement': agree_kb, 'analog_agreement': agree_an,
            'samples': [int(len(kb)), int(len(an))]}


def map_blocks(uid: str) -> List[dict]:
    return json.loads((LIVE_MAPS / f'{safe(uid)}.json').read_text(encoding='utf-8'))['blocks']


def check_block_dir(runs, uids: Dict[int, str]) -> Dict:
    """Fit heading(dir) = offset + sign * 90deg * dir on start blocks vs the car at t=0."""
    obs = []
    for tid, rs in runs.items():
        starts = [b for b in map_blocks(uids[tid]) if b['waypoint'] in (0, 4)]
        if len(starts) != 1:
            continue
        a = rs[0]['a']
        _, _, fwd = F.car_axes(a['rot'][:1])
        obs.append((starts[0]['dir'], float(np.degrees(F.heading(fwd))[0])))
    best = None
    for sign in (1, -1):
        for off in (0, 90, 180, 270):
            err = [abs((h - (off + sign * 90 * d) + 180) % 360 - 180) for d, h in obs]
            score = float(np.mean(np.array(err) < 10))
            if best is None or score > best['agreement']:
                best = {'sign': sign, 'offset_deg': off, 'agreement': score}
    dirs = {d for d, _ in obs}
    best.update(maps=len(obs), distinct_dirs=sorted(dirs))
    if len(obs) < 5 or len(dirs) < 3 or best['agreement'] < 0.9:
        raise RuntimeError(f'block direction convention not determined: {best}')
    return best


def build(stride: int = 3, log=print) -> Path:
    man = json.loads(MANIFEST.read_text(encoding='utf-8'))
    info = {int(m['track_id']): m for m in man['maps'].values() if m.get('ok')}
    runs = load_runs(log)
    uids = {tid: info[tid]['uid'] for tid in runs}

    steer = check_steer(runs)
    bdir = check_block_dir(runs, uids)
    log(f'steer convention: {steer}')
    log(f'block dir convention: {bdir}')
    Calibration(CALIBRATION).update(keyboard=steer, block_dir=bdir)

    names = collections.Counter()
    for tid in runs:
        names.update(b['name'] for b in map_blocks(uids[tid]) if b['name'] not in F.FILLER)
    vocab = {n: i + 2 for i, (n, _) in enumerate(names.most_common())}   # 0 pad, 1 unknown

    cols = collections.defaultdict(list)
    skipped = 0
    for tid, rs in sorted(runs.items()):
        if len(rs) < 2:
            skipped += len(rs)
            continue
        geom = F.MapGeometry(map_blocks(uids[tid]), vocab, bdir['sign'], bdir['offset_deg'])
        best = min(r['finish'] for r in rs)
        for r in rs:
            other = min((o for o in rs if o is not r), key=lambda o: o['finish'])
            oa = other['a']
            line = RefLine(oa['pos'], np.linalg.norm(oa['vel'], axis=1))
            a = r['a']
            t = a['t'].astype(np.int64)
            live = (t >= 0) & (t < r['finish'])
            idx_all = F.progress_indices(line, a['pos'].astype(np.float64))
            off = int(hashlib.sha1(r['meta']['replay'].encode()).hexdigest()[:4], 16) % stride
            sel = np.nonzero(live)[0][off::stride]
            sub = {k: v[sel] for k, v in a.items()}
            pace = float(np.log(r['finish'] / best))
            cols['state'].append(F.state_features(sub, pace))
            cols['route'].append(F.route_features(line, sub['pos'].astype(np.float64), sub['rot'], idx_all[sel]).astype(np.float16))
            bn, bw, bf, bm = geom.features(sub['pos'].astype(np.float64), sub['rot'])
            cols['bname'].append(bn.astype(np.int32))
            cols['bwp'].append(bw.astype(np.int8))
            cols['bfeat'].append(bf.astype(np.float16))
            cols['bmask'].append(bm)
            su = F.steer_unit(sub['act_steer'], sub['act_bits'], steer['right_sign'])
            cols['y_steer'].append(F.steer_bin(su).astype(np.int8))
            cols['y_gas'].append((((sub['act_bits'] & P.UP) > 0) | (sub['act_gas'] > 0)).astype(np.int8))
            cols['y_brake'].append(((sub['act_bits'] & P.DOWN) > 0).astype(np.int8))
            cols['y_value'].append(((r['finish'] - t[sel]) / 1000.0).astype(np.float32))
            cols['map'].append(np.full(len(sel), tid, np.int64))
            cols['holdout'].append(np.full(len(sel), bool(info[tid]['holdout'])))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    meta = {'features_version': F.FEATURES_VERSION, 'stride': stride, 'vocab': vocab,
            'steer': steer, 'block_dir': bdir, 'skipped_single_run': skipped,
            'maps': {str(t): info[t]['name'] for t in runs}}
    np.savez_compressed(DATASET, meta=json.dumps(meta), **out)
    n, nh = len(out['map']), int(out['holdout'].sum())
    log(f'dataset: {n} samples ({n - nh} train, {nh} held-out) from {len(runs)} maps, '
        f'{skipped} single-run replays skipped, vocab {len(vocab)} -> {DATASET}')
    ys = out['y_steer']
    log(f"labels: steer full-left {np.mean(ys == 0):.2f} straight {np.mean(ys == F.STEER_BINS // 2):.2f} "
        f"full-right {np.mean(ys == F.STEER_BINS - 1):.2f}, gas {out['y_gas'].mean():.2f}, brake {out['y_brake'].mean():.2f}")
    return DATASET
