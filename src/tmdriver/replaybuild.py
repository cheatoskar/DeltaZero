"""Stage A2 without the game: TMX replays -> shards in the stage-A format, plus orientation.

What a replay adds over the HF traces, all without re-simulation:
  * exact inputs (events on the 10 ms grid, analog values) instead of a 100 ms sample;
  * the car's orientation every 100 ms (ghost samples; decoding verified to 1e-4 against
    re-simulated runs), so the model sees drift / slide angles;
  * ~5 runs per map (top 5), so every run has another run's line as its route.

Maps and replays come from the bulk download (data/bulk/maps.jsonl, data/maps, data/tmx/<id>)
and the M1 manifest. The vocabulary must be the stage-A one (data/ghost/vocab.json), so
build the trace shards first. Output: data/ghost/replay_shards/partNNNN (250 maps each).

Labels follow the trace convention exactly (measured on part 1: trace input == replay input
state at gbx time t - 10 ms, 100% for keyboard runs), including the t = 0 fix.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import ghost as G
from . import protocol as P
from . import replay as replay_mod
from .paths import DATA, TMX, safe
from .tracebuild import GHOST, MAX_JUMP_M, MAX_RUN_S, MIN_RUN_S, VOCAB, PartWriter

REPLAY_SHARDS = GHOST / 'replay_shards'
RESIM_SHARDS = GHOST / 'resim_shards'       # from the client-free re-simulation (sim-night)
MAPS_PER_PART = 250


def map_sources():
    """[(track_id, map path, replay folder)] from the bulk download and the M1 manifest."""
    from .bulk import MAPS_LOG, MAP_DIR, REP_DIR
    from .collect import MANIFEST
    from .tmx import tracks_dir
    out, seen = [], set()
    if MAPS_LOG.exists():
        for line in MAPS_LOG.read_text(encoding='utf-8').splitlines():
            m = json.loads(line)
            if m.get('ok') and m['track_id'] not in seen:
                seen.add(m['track_id'])
                out.append((m['track_id'], MAP_DIR / f"{m['track_id']}.Challenge.Gbx", REP_DIR / str(m['track_id'])))
    if MANIFEST.exists():
        for m in json.loads(MANIFEST.read_text(encoding='utf-8'))['maps'].values():
            if m.get('ok') and m['track_id'] not in seen:
                seen.add(m['track_id'])
                out.append((m['track_id'], tracks_dir() / m['map_file'], TMX / safe(m['uid'])))
    return sorted(out)


def resim_sources():
    """[(track_id, map path, replay folder, exact replay ids)] from sim-night (data/resim_c):
    only replays whose inputs reproduced the ghost exactly in TMNF-C, so every label is known to
    produce the recorded line."""
    from .tmnfc import RESIM_C
    idx = RESIM_C / 'index.jsonl'
    if not idx.exists():
        return []
    exact, uid = {}, {}
    for line in idx.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('exact'):
            exact.setdefault(int(r['track_id']), set()).add(str(r['replay']))
            uid[int(r['track_id'])] = r['uid']
    maps = {}
    for f in (DATA / 'maps').glob('*_*.Challenge.Gbx'):
        try:
            maps[int(f.name.split('.')[0].rsplit('_', 1)[1])] = f
        except ValueError:
            pass
    return sorted((tid, maps[tid], TMX / safe(uid[tid]), ids) for tid, ids in exact.items() if tid in maps)


def load_vocab() -> dict:
    """The stage-A vocabulary; without data/ghost/vocab.json it comes from the driver checkpoint
    (the model can only use block ids it was trained with)."""
    if VOCAB.exists():
        return json.loads(VOCAB.read_text(encoding='utf-8'))
    from .trainpack import PACK
    if (PACK / 'vocab.json').exists():                    # a training server: the pack brings it
        VOCAB.parent.mkdir(parents=True, exist_ok=True)
        shutil_copy = (PACK / 'vocab.json').read_text(encoding='utf-8')
        VOCAB.write_text(shutil_copy, encoding='utf-8')
        return json.loads(shutil_copy)
    import torch
    from .paths import DRIVER_CKPT
    ck = DRIVER_CKPT
    if not Path(ck).exists():
        raise RuntimeError(f'{VOCAB} missing and no driver checkpoint to take the vocabulary from')
    vocab = torch.load(ck, map_location='cpu', weights_only=False)['meta']['vocab']
    VOCAB.parent.mkdir(parents=True, exist_ok=True)
    VOCAB.write_text(json.dumps(vocab), encoding='utf-8')
    return vocab


class NpzRun:
    """A sim-night run (data/resim_c/<track>/<replay>.npz) in the shape build_part reads from a
    replay: ghost samples every 100 ms (exact simulated positions, orientation from the
    rotation matrix: forward / up = columns 2 / 1) and the labels from the per-tick actions.

    The actions are the input table rows the re-simulation played (action k = table row k+1,
    alignment shift 1), so a label taken 10 ms before a sample (see labels()) is action
    t/10 - 2. The t = 0 sample repeats the first simulated state (the car has not moved yet)."""

    def __init__(self, path, map_uid: str):
        if isinstance(path, dict):          # a run of a training pack (trainpack.runs_of)
            z, meta = path, path['meta']
        else:
            z = np.load(path, allow_pickle=True)
            meta = json.loads(str(z['meta']))
        self.map_uid = map_uid
        self.race_time_ms = int(meta['recorded_ms'])
        self.respawns = int(meta.get('respawns') or 0)
        t = z['t'].astype(np.int64)
        if isinstance(path, dict):          # already every 100 ms from t = 0
            self.ghost_t = t
            self.ghost_pos = z['pos'].astype(np.float64)
            rot = z['rot'].astype(np.float64)
        else:
            keep = (t % G.DT_MS == 0) & (t <= self.race_time_ms)
            pos, rot = z['pos'][keep].astype(np.float64), z['rot'][keep].astype(np.float64)
            first_pos, first_rot = z['pos'][:1].astype(np.float64), z['rot'][:1].astype(np.float64)
            self.ghost_t = np.concatenate([[0], t[keep]])
            self.ghost_pos = np.concatenate([first_pos, pos])
            rot = np.concatenate([first_rot, rot])
        self.ghost_orient = np.concatenate([rot[:, [2, 5, 8]], rot[:, [1, 4, 7]]], 1)
        steer = z['act_steer'].astype(np.int64)
        gas = z['act_gas'].astype(np.int64)
        bits = z['act_bits'].astype(np.int64)
        idx = np.clip(self.ghost_t // 10 - 2, 0, len(bits) - 1)
        idx[0] = idx[1] if len(idx) > 1 else idx[0]
        b, st, gs = bits[idx], steer[idx], gas[idx]
        analog = (b & P.STEER_ANALOG) > 0
        kb = np.where(b & P.RIGHT, 1.0, 0.0) - np.where(b & P.LEFT, 1.0, 0.0)
        self.labels = (np.where(analog, np.clip(st / P.STEER_MAX, -1, 1), kb),
                       (((b & P.UP) > 0) | (np.abs(gs) >= P.STEER_MAX // 2)).astype(np.float64),
                       ((b & P.DOWN) > 0).astype(np.float64))


def trainpack_sources():
    """[(track_id, map path, pack file, None, 'pack')] from a training package (trainpack.py)."""
    from .trainpack import PACK
    maps = {}
    for f in (PACK / 'maps').glob('*_*.Challenge.Gbx'):
        try:
            maps[int(f.name.split('.')[0].rsplit('_', 1)[1])] = f
        except ValueError:
            pass
    return sorted((int(f.stem), maps[int(f.stem)], f, None, 'pack') for f in (PACK / 'runs').glob('*.npz')
                  if f.stem.isdigit() and int(f.stem) in maps)


def resim_npz_sources():
    """[(track_id, map path, npz folder, None, 'npz')]: the exact sim-night runs, no replay files
    needed (what goes to a training server)."""
    from .tmnfc import RESIM_C
    maps = {}
    for f in (DATA / 'maps').glob('*_*.Challenge.Gbx'):
        try:
            maps[int(f.name.split('.')[0].rsplit('_', 1)[1])] = f
        except ValueError:
            pass
    out = []
    for d in sorted(RESIM_C.iterdir()) if RESIM_C.exists() else []:
        if d.is_dir() and d.name.isdigit() and int(d.name) in maps and any(d.glob('*.npz')):
            out.append((int(d.name), maps[int(d.name)], d, None, 'npz'))
    return out


def map_blocks(path: Path):
    from .gbx.reader import Gbx
    ch = Gbx(str(path)).get_class_by_id(0x03043000)
    bl = ch.blocks
    names = [b.name for b in bl]
    xyz = np.array([[int(b.position.x), int(b.position.y), int(b.position.z)] for b in bl], np.int64).reshape(-1, 3)
    dirs = np.array([int(b.rotation) % 4 for b in bl], np.int64)
    return ch.map_uid, names, xyz, dirs


def labels(rep) -> tuple:
    """Input state 10 ms before each ghost sample, in TMInterface's steer convention."""
    tab = replay_mod.input_table(rep, steer_sign=-1)
    n = len(rep.ghost_t)
    rows = np.clip(rep.ghost_t // 10 - 1, 0, len(tab['t']) - 1)
    rows[0] = rows[1] if n > 1 else rows[0]          # t = 0 holds the pre-race state: see tracebuild
    bits, st, gs = tab['in_bits'][rows], tab['in_steer'][rows], tab['in_gas'][rows]
    analog = (bits & P.STEER_ANALOG) > 0
    kb = np.where(bits & P.RIGHT, 1.0, 0.0) - np.where(bits & P.LEFT, 1.0, 0.0)   # Right = +1 (M1)
    steer = np.where(analog, np.clip(st / P.STEER_MAX, -1, 1), kb)
    gas = (((bits & P.UP) > 0) | (np.abs(gs) >= P.STEER_MAX // 2)).astype(np.float64)
    brake = ((bits & P.DOWN) > 0).astype(np.float64)
    return steer, gas, brake


def good_run(rep, uid, stats) -> bool:
    if rep.map_uid != uid:
        stats['uid_mismatch'] += 1
        return False
    if rep.respawns:
        stats['respawns'] += 1
        return False
    t, pos = rep.ghost_t, rep.ghost_pos
    if len(t) < 2 or np.any(np.diff(t) != G.DT_MS) or not np.isfinite(pos).all():
        stats['bad_time'] += 1
        return False
    if not MIN_RUN_S <= rep.race_time_ms / 1000 <= MAX_RUN_S:
        stats['length'] += 1
        return False
    if np.linalg.norm(np.diff(pos, axis=0), axis=1).max() > MAX_JUMP_M:
        stats['jump'] += 1
        return False
    return True


def build_part(sources, vocab, out_dir: Path, seed: int, log=print) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    w = PartWriter(stride=1, rng=rng)
    stats = {'replays': 0, 'uid_mismatch': 0, 'respawns': 0, 'bad_time': 0, 'length': 0, 'jump': 0,
             'parse_error': 0, 'no_map': 0, 'runs': 0}
    for src in sources:
        tid, mpath, rdir = src[:3]
        allowed = src[3] if len(src) > 3 else None       # resim_c: only the exact replays
        if not mpath.exists() or not Path(rdir).exists():
            stats['no_map'] += 1
            continue
        try:
            uid, names, xyz, dirs = map_blocks(mpath)
        except Exception:
            stats['no_map'] += 1
            continue
        runs = []
        kind = src[4] if len(src) > 4 else 'replay'
        if kind == 'pack':
            from .trainpack import runs_of
            items = list(runs_of(rdir))
        else:
            items = sorted(rdir.glob('*.npz' if kind == 'npz' else '*.Replay.Gbx'))
        for f in items:
            if allowed is not None and f.name.split('.')[0] not in allowed:
                continue
            stats['replays'] += 1
            try:
                rep = NpzRun(f, uid) if kind in ('npz', 'pack') else replay_mod.load(f)
            except Exception:
                stats['parse_error'] += 1
                continue
            if good_run(rep, uid, stats):
                keep = rep.ghost_t <= rep.race_time_ms
                runs.append((rep, keep))
        if not runs:
            continue
        mb = G.MapBlocks(names, xyz, dirs, vocab)
        if len(mb.center) == 0:
            continue
        m_idx = w.add_map(int(tid), mb, [r.ghost_pos[k] for r, k in runs])
        best = min(r.race_time_ms for r, _ in runs)
        order = sorted(range(len(runs)), key=lambda i: runs[i][0].race_time_ms)
        for i, (rep, keep) in enumerate(runs):
            other = next((runs[j] for j in order if j != i), None)       # the fastest OTHER run
            steer, gas, brake = rep.labels if isinstance(rep, NpzRun) else labels(rep)
            t = rep.ghost_t[keep]
            w.add_run(m_idx, mb, t, rep.ghost_pos[keep], steer[keep], gas[keep], brake[keep],
                      other[0].ghost_pos[other[1]] if other else None,
                      float(np.clip(np.log(rep.race_time_ms / max(best, 1)), 0, 1)), rep.ghost_orient[keep])
            stats['runs'] += 1
    stats = w.save(out_dir, stats, t0)
    log(f"{out_dir.name}: {stats['maps']} maps, {stats['runs']} runs of {stats['replays']} replays, "
        f"{stats['samples']} samples; skipped: respawns {stats['respawns']} uid {stats['uid_mismatch']} "
        f"jump {stats['jump']} parse {stats['parse_error']}; start heading agrees "
        f"{stats['start_heading_agree_frac']}; {stats['seconds']}s")
    return stats


def _job(args):
    sources, out, seed = args
    if (out / 'DONE').exists():
        return json.loads((out / 'stats.json').read_text(encoding='utf-8'))
    return build_part(sources, load_vocab(), out, seed)


def build_all(workers: int = 4, rebuild: bool = False, source: str = 'bulk', log=print):
    """source 'bulk' (the bulk download + M1 manifest) -> replay_shards, 'resim_c' (the exact
    runs of sim-night) -> resim_shards."""
    load_vocab()
    if source == 'resim_c':
        src, root = resim_sources(), RESIM_SHARDS
    elif source == 'resim_npz':
        src, root = resim_npz_sources(), RESIM_SHARDS
    elif source == 'trainpack':
        src, root = trainpack_sources(), RESIM_SHARDS
    else:
        src, root = map_sources(), REPLAY_SHARDS
    jobs = []
    for k in range(0, len(src), MAPS_PER_PART):
        out = root / f'part{k // MAPS_PER_PART:04d}'
        if rebuild and out.exists():
            import shutil
            shutil.rmtree(out)
        jobs.append((src[k:k + MAPS_PER_PART], out, 1000 + k))
    log(f'{len(src)} maps in {len(jobs)} parts, {workers} workers -> {root}')
    if workers <= 1:
        return [_job(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else None
    res = []
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for s in ex.map(_job, jobs):
            res.append(s)
            log(f"  part done: {s.get('maps')} maps, {s.get('runs')} runs, {s.get('samples')} samples "
                f"({len(res)}/{len(jobs)})")
    return res
