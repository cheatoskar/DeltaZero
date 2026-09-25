"""HF traces (+ MX blocks) -> training shards for the ghost-feature driver (stage A).

One part at a time (trace part N and block part N hold the same ~2000 maps; measured
100% overlap on part 1). Output per part, in data/ghost/shards/partNNNN/:

    <split>_<field>.npy   split = train | val (held-out maps, by track-id hash)
        state  (N, STATE_DIM) f16     route (N, 18, 3) f16     pos (N, 3) f32
        psi    (N,) f32               map   (N,) i32 (index into this part's maps)
        steer  (N,) i8 bin            pedal (N,) u8 (1 gas, 2 brake)
        value  (N,) f16 s to finish   rline (N,) u8 (1 = the route is a real line)
    maps_*.npy: track_id, block CSR (b_off, b_xyz i16, b_dir i8, b_name i32, b_wp i8)
    stats.json
Samples are shuffled within the part, so a contiguous slice is a random sample.

Everything the live driver computes is computed here by the same functions in ghost.py.
Block tokens are NOT precomputed (they would dominate the disk): the trainer builds them
on the GPU from the per-map block table (ghost_torch.block_features).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ghost as G
from .collect import holdout
from .paths import DATA

GHOST = DATA / 'ghost'
SHARDS = GHOST / 'shards'
VOCAB = GHOST / 'vocab.json'
DIRS = {'North': 0, 'East': 1, 'South': 2, 'West': 3, '0': 0, '1': 1, '2': 2, '3': 3,
        '4': 0, '6': 2}   # stray numeric rows seen in MX parts (see TMTrackNN notes)
MAX_JUMP_M = 80.0        # > this between 100 ms samples = respawn / teleport
MIN_RUN_S, MAX_RUN_S = 3.0, 300.0
MAX_BLOCKS = 1024
MX_TO_GBX = np.array([1, 0, 1])


def build_vocab(block_parts, min_count: int = 20, log=print):
    counts = {}
    for p in block_parts:
        vc = pd.read_parquet(p, columns=['name']).name.value_counts()
        for n, c in vc.items():
            counts[n] = counts.get(n, 0) + int(c)
    names = sorted(n for n, c in counts.items() if c >= min_count and n not in G.FILLER)
    vocab = {n: i + 2 for i, n in enumerate(names)}          # 0 pad, 1 unknown
    GHOST.mkdir(parents=True, exist_ok=True)
    VOCAB.write_text(json.dumps(vocab), encoding='utf-8')
    log(f'vocab: {len(vocab)} block names (count >= {min_count}) from {len(block_parts)} parts -> {VOCAB}')
    return vocab


def _runs(tr: pd.DataFrame):
    """-> {track_id: [(rank, t_ms, pos, steer, gas, brake), ...]} of runs that pass the checks."""
    stats = {'runs': 0, 'bad_time': 0, 'jump': 0, 'length': 0, 'nan': 0}
    out = {}
    tr = tr.sort_values(['track_id', 'trace_rank', 'time_ms'], kind='stable')
    for (tid, rank), g in tr.groupby(['track_id', 'trace_rank'], sort=False):
        stats['runs'] += 1
        t = g.time_ms.to_numpy()
        pos = g[['x', 'y', 'z']].to_numpy(np.float64)
        inp = g[['input_steer', 'input_gas', 'input_brake']].to_numpy(np.float64, copy=True)
        if len(t) < 2 or t[0] != 0 or np.any(np.diff(t) != G.DT_MS):
            stats['bad_time'] += 1
            continue
        dur = t[-1] / 1000.0
        if not MIN_RUN_S <= dur <= MAX_RUN_S:
            stats['length'] += 1
            continue
        if not np.isfinite(pos).all() or not np.isfinite(inp).all():
            stats['nan'] += 1
            continue
        if np.linalg.norm(np.diff(pos, axis=0), axis=1).max() > MAX_JUMP_M:
            stats['jump'] += 1
            continue
        # A trace sample carries the input 10 ms BEFORE it, so sample 0 holds the input before
        # the race: "nothing pressed" in 3463/3463 runs of part 1, while at 100 ms 97% hold
        # gas. Left in, the model learns "standing car -> no gas" and never starts. Sample 0
        # takes the first real input instead.
        inp[0] = inp[1]
        out.setdefault(str(tid), []).append((int(rank), t, pos, inp[:, 0], inp[:, 1], inp[:, 2]))
    return out, stats


class PartWriter:
    """Collects maps and runs of one shard part and writes it. Shared by the trace build
    (stage A) and the replay build (stage A2), so both produce identical samples."""
    FIELDS = ('state', 'route', 'pos', 'psi', 'map', 'steer', 'pedal', 'value', 'rline')

    def __init__(self, stride: int, rng):
        self.stride, self.rng = stride, rng
        self.maps_tid, self.maps_hold = [], []
        self.b_off, self.b_xyz, self.b_dir, self.b_name, self.b_wp = [0], [], [], [], []
        self.cols = {k: {'train': [], 'val': []} for k in self.FIELDS}
        self.stats = {'maps': 0, 'samples': 0, 'with_line': 0, 'blocks_capped': 0, 'start_heading_agree': []}

    def add_map(self, tid: int, mb: G.MapBlocks, run_positions) -> int:
        # keep blocks that can ever be within BLOCK_RANGE of this map's runs (lossless)
        allpos = np.concatenate(run_positions)
        dmin = np.full(len(mb.center), np.inf)
        for i in range(0, len(allpos), 2048):
            d = np.linalg.norm(mb.center[:, None, :] - allpos[None, i:i + 2048, :], axis=-1).min(1)
            dmin = np.minimum(dmin, d)
        keep = np.nonzero(dmin <= G.BLOCK_RANGE + 1.0)[0]
        if len(keep) > MAX_BLOCKS:
            keep = keep[np.argsort(dmin[keep], kind='stable')[:MAX_BLOCKS]]
            self.stats['blocks_capped'] += 1
        self.maps_tid.append(int(tid))
        self.maps_hold.append(holdout(int(tid)))
        self.b_xyz.append(mb.xyz[keep])
        self.b_dir.append(mb.dir[keep])
        self.b_name.append(mb.name_id[keep])
        self.b_wp.append(mb.wp[keep])
        self.b_off.append(self.b_off[-1] + len(keep))
        self.stats['maps'] += 1
        return len(self.maps_tid) - 1

    def add_run(self, m_idx: int, mb: G.MapBlocks, t, pos, steer, gas, brake, line_pos=None, pace: float = 0.0,
                orient=None):
        """t (N,) ms on the 100 ms lattice from 0; pos (N,3); steer in TMInterface's convention;
        gas/brake 0/1; line_pos: another run's positions (never this run's: that would leak
        the answer); orient (N,6) world forward + up, or None."""
        split = 'val' if self.maps_hold[m_idx] else 'train'
        st_ = self.stats
        psi0 = mb.start_heading(pos[0])
        psi = G.headings(pos, psi0)
        # check: does the start block's direction match where the car actually goes?
        away = np.nonzero(np.hypot(pos[:, 0] - pos[0, 0], pos[:, 2] - pos[0, 2]) > 5.0)[0]
        if len(away):
            d = pos[away[0]] - pos[0]
            st_['start_heading_agree'].append(float(np.cos(np.arctan2(d[0], d[2]) - psi0)))
        line = G.Line(line_pos) if line_pos is not None else None
        idx = np.arange(int(self.rng.integers(self.stride)), len(t), self.stride)
        lags = G.lag_positions(pos, idx)
        psi_prev = psi[np.maximum(idx - 1, 0)]
        st = G.state_features(lags, psi[idx], psi_prev, None if orient is None else orient[idx], pace,
                              1.0 if line else 0.0)
        if line is not None:
            s = G.progress(line, pos)[idx]
            route = G.route_features(line, s, pos[idx], psi[idx])
            st_['with_line'] += len(idx)
        else:
            route = np.zeros((len(idx), len(G.ROUTE_D), 3), np.float32)
        c = self.cols
        c['state'][split].append(st.astype(np.float16))
        c['route'][split].append(route.astype(np.float16))
        c['pos'][split].append(pos[idx].astype(np.float32))
        c['psi'][split].append(psi[idx].astype(np.float32))
        c['map'][split].append(np.full(len(idx), m_idx, np.int32))
        c['steer'][split].append(G.steer_bin(steer[idx]).astype(np.int8))
        c['pedal'][split].append(((gas[idx] >= 0.5) * 1 + (brake[idx] >= 0.5) * 2).astype(np.uint8))
        c['value'][split].append(((t[-1] - t[idx]) / 1000.0).astype(np.float16))
        c['rline'][split].append(np.full(len(idx), 1 if line else 0, np.uint8))
        st_['samples'] += len(idx)

    def save(self, out_dir: Path, extra_stats: dict, t0: float):
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = dict(extra_stats, **self.stats)
        for split in ('train', 'val'):
            n = sum(len(a) for a in self.cols['map'][split])
            perm = self.rng.permutation(n)
            for k, parts in self.cols.items():
                if n:
                    np.save(out_dir / f'{split}_{k}.npy', np.concatenate(parts[split])[perm])
            stats[f'{split}_samples'] = int(n)
        cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
        np.save(out_dir / 'maps_track_id.npy', np.array(self.maps_tid, np.int64))
        np.save(out_dir / 'maps_holdout.npy', np.array(self.maps_hold, bool))
        np.save(out_dir / 'b_off.npy', np.array(self.b_off, np.int64))
        np.save(out_dir / 'b_xyz.npy', cat(self.b_xyz, (0, 3)).astype(np.int16))
        np.save(out_dir / 'b_dir.npy', cat(self.b_dir, 0).astype(np.int8))
        np.save(out_dir / 'b_name.npy', cat(self.b_name, 0).astype(np.int32))
        np.save(out_dir / 'b_wp.npy', cat(self.b_wp, 0).astype(np.int8))
        agree = stats.pop('start_heading_agree')
        stats['start_heading_cos_mean'] = float(np.mean(agree)) if agree else None
        stats['start_heading_agree_frac'] = float(np.mean(np.array(agree) > 0.7)) if agree else None
        stats['seconds'] = round(time.time() - t0, 1)
        (out_dir / 'stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
        (out_dir / 'DONE').write_text('ok', encoding='utf-8')
        return stats


def build_part(trace_path: Path, block_path: Path, vocab, out_dir: Path, stride: int = 4, seed: int = 0,
               log=print) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(trace_path, columns=['track_id', 'trace_rank', 'time_ms', 'x', 'y', 'z',
                                              'input_steer', 'input_gas', 'input_brake'])
    bl = pd.read_parquet(block_path, columns=['track_id', 'name', 'x', 'y', 'z', 'dir'])
    bl['track_id'] = bl.track_id.astype(str)
    blocks_by_map = {k: g for k, g in bl.groupby('track_id', sort=False)}
    runs, stats = _runs(tr)
    stats['no_blocks'] = 0
    w = PartWriter(stride, rng)
    for tid, rs in runs.items():
        g = blocks_by_map.get(tid)
        if g is None or len(g) == 0:
            stats['no_blocks'] += 1
            continue
        names = g.name.astype(str).tolist()
        xyz = g[['x', 'y', 'z']].to_numpy(np.int64) + MX_TO_GBX
        dirs = np.array([DIRS.get(str(d), 0) for d in g.dir], dtype=np.int64)
        mb = G.MapBlocks(names, xyz, dirs, vocab)
        if mb.center.shape[0] == 0:
            stats['no_blocks'] += 1
            continue
        m_idx = w.add_map(int(tid), mb, [r[2] for r in rs])
        best_t = min(r[1][-1] for r in rs)
        for rank, t, pos, steer, gas, brake in rs:
            others = [r for r in rs if r[0] != rank]
            pace = float(np.clip(np.log(t[-1] / max(best_t, 1)), 0, 1))
            w.add_run(m_idx, mb, t, pos, steer, gas, brake, others[0][2] if others else None, pace)
    stats = w.save(out_dir, stats, t0)
    log(f"{trace_path.name}: {stats['maps']} maps, {stats['samples']} samples "
        f"({stats['with_line'] / max(stats['samples'], 1):.0%} with line), dropped runs: "
        f"time {stats['bad_time']} jump {stats['jump']} length {stats['length']}; "
        f"start heading agrees {stats['start_heading_agree_frac']}; {stats['seconds']}s")
    return stats


def _job(args):
    tp, bp, out, stride, seed = args
    vocab = json.loads(VOCAB.read_text(encoding='utf-8'))
    if (out / 'DONE').exists():
        return json.loads((out / 'stats.json').read_text(encoding='utf-8'))
    return build_part(tp, bp, vocab, out, stride=stride, seed=seed)


def build_all(trace_dir: Path, block_dir: Path, workers: int = 4, stride: int = 4, log=print):
    traces = sorted(trace_dir.glob('traces_tmnf_part*.parquet'))
    jobs = []
    for tp in traces:
        part = tp.stem.split('part')[-1]
        bp = block_dir / f'blocks_tmnf_part{part}.parquet'
        if not bp.exists():
            log(f'no blocks for {tp.name}, skipped')
            continue
        jobs.append((tp, bp, SHARDS / f'part{part}', stride, int(part)))
    if not VOCAB.exists():
        build_vocab([j[1] for j in jobs], log=log)
    log(f'{len(jobs)} parts, {workers} workers')
    if workers <= 1:
        return [_job(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context('fork') if hasattr(mp, 'get_context') and 'fork' in mp.get_all_start_methods() else None
    res = []
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for s in ex.map(_job, jobs):
            res.append(s)
            log(f"  part done: {s.get('maps')} maps, {s.get('samples')} samples, {s.get('seconds')}s "
                f"({len(res)}/{len(jobs)})")
    return res
