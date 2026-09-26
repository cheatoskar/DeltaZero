"""A compact training package from sim-night's exact runs, for a training server (upload once).

Per map one file data/trainpack/runs/<track_id>.npz holding every exact run of the map:
  * the car every 100 ms up to the finish (position + rotation matrix, float32, exact),
  * the per-tick actions run-length encoded (they change rarely),
  * each run's meta (replay id, recorded time, analog, ...).
Plus the map files (data/trainpack/maps/<name>_<id>.Challenge.Gbx) and the vocabulary.

On the server: `python tmdriver.py ghost-replays --source trainpack` builds the shards from it
(replaybuild.NpzRun reads a run of a pack like a sim-night npz), then pretrain as usual.

    python tmdriver.py trainpack            -> data/trainpack/ + data/trainpack.zip
"""
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from .paths import DATA

PACK = DATA / 'trainpack'


def rle(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(start indices, values) of the runs of equal consecutive values."""
    if len(a) == 0:
        return np.zeros(0, np.int32), a
    starts = np.concatenate([[0], np.flatnonzero(np.diff(a) != 0) + 1]).astype(np.int32)
    return starts, a[starts]


def unrle(starts: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(n, dtype=values.dtype)
    ends = np.concatenate([starts[1:], [n]])
    for s, e, v in zip(starts, ends, values):
        out[s:e] = v
    return out


def pack_map(track_id: int, npz_files, out: Path) -> int:
    runs = []
    for f in npz_files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z['meta']))
        t = z['t'].astype(np.int64)
        keep = (t % 100 == 0) & (t <= int(meta['recorded_ms']))
        # the t = 0 sample: the first simulated state (the car has not moved yet)
        pos = np.concatenate([z['pos'][:1], z['pos'][keep]]).astype(np.float32)
        rot = np.concatenate([z['rot'][:1], z['rot'][keep]]).astype(np.float32)
        acts = np.stack([z['act_steer'].astype(np.int64), z['act_gas'].astype(np.int64),
                         z['act_bits'].astype(np.int64)], 1)
        key = acts[:, 0] * 2 ** 24 + (acts[:, 1] & 0xFFFF) * 2 ** 8 + acts[:, 2]   # one code per action
        starts, _ = rle(key)
        runs.append({'meta': meta, 'pos': pos, 'rot': rot, 'n_ticks': len(acts), 'a_start': starts,
                     'a_val': acts[starts]})
    if not runs:
        return 0
    cat = lambda k: np.concatenate([r[k] for r in runs])            # noqa: E731
    np.savez_compressed(
        out, meta=json.dumps([r['meta'] for r in runs]),
        n_samples=np.array([len(r['pos']) for r in runs], np.int32),
        n_ticks=np.array([r['n_ticks'] for r in runs], np.int32),
        n_changes=np.array([len(r['a_start']) for r in runs], np.int32),
        pos=cat('pos'), rot=cat('rot'), a_start=cat('a_start'),
        a_steer=cat('a_val')[:, 0].astype(np.int32), a_gas=cat('a_val')[:, 1].astype(np.int32),
        a_bits=cat('a_val')[:, 2].astype(np.uint8))
    return len(runs)


def runs_of(pack_file: Path) -> Iterator[dict]:
    """Every run of a pack file: {'meta', 't', 'pos', 'rot', 'act_steer', 'act_gas', 'act_bits'}."""
    z = np.load(pack_file, allow_pickle=True)
    metas = json.loads(str(z['meta']))
    s = c = 0
    for k, meta in enumerate(metas):
        ns, nc, nt = int(z['n_samples'][k]), int(z['n_changes'][k]), int(z['n_ticks'][k])
        starts = z['a_start'][c:c + nc]
        yield {'meta': meta, 't': np.arange(ns, dtype=np.int64) * 100, 'pos': z['pos'][s:s + ns],
               'rot': z['rot'][s:s + ns], 'act_steer': unrle(starts, z['a_steer'][c:c + nc], nt),
               'act_gas': unrle(starts, z['a_gas'][c:c + nc], nt),
               'act_bits': unrle(starts, z['a_bits'][c:c + nc], nt)}
        s += ns
        c += nc


def build(zip_it: bool = True, log=print) -> Path:
    from .replaybuild import load_vocab
    from .tmnfc import RESIM_C
    maps = {}
    for f in (DATA / 'maps').glob('*_*.Challenge.Gbx'):
        try:
            maps[int(f.name.split('.')[0].rsplit('_', 1)[1])] = f
        except ValueError:
            pass
    (PACK / 'runs').mkdir(parents=True, exist_ok=True)
    (PACK / 'maps').mkdir(parents=True, exist_ok=True)
    n_maps = n_runs = 0
    for d in sorted(RESIM_C.iterdir()):
        if not (d.is_dir() and d.name.isdigit()) or int(d.name) not in maps:
            continue
        files = sorted(d.glob('*.npz'))
        if not files:
            continue
        out = PACK / 'runs' / f'{d.name}.npz'
        if not out.exists():
            n = pack_map(int(d.name), files, out)
            if not n:
                continue
        m = maps[int(d.name)]
        if not (PACK / 'maps' / m.name).exists():
            shutil.copyfile(m, PACK / 'maps' / m.name)
        n_maps += 1
        n_runs += len(files)
        if n_maps % 1000 == 0:
            log(f'  {n_maps} maps ...')
    (PACK / 'vocab.json').write_text(json.dumps(load_vocab()), encoding='utf-8')
    size = sum(f.stat().st_size for f in PACK.rglob('*') if f.is_file())
    log(f'{n_maps} maps, {n_runs} exact runs -> {PACK} ({size / 1e9:.2f} GB)')
    if not zip_it:
        return PACK
    z = DATA / 'trainpack.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as zf:        # the npz are compressed already
        for f in sorted(PACK.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(PACK.parent))
    log(f'-> {z} ({z.stat().st_size / 1e9:.2f} GB): upload this one file')
    return z
