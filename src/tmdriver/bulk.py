"""Bulk TMX download for fine-tuning: top-awarded maps, their fastest replays.

Resumable: every finished map is appended to data/bulk/maps.jsonl, and a restart skips
maps already listed there. Maps are written to data/maps/<track_id>.Challenge.Gbx (the
copy that travels to the server) and to TmForever/Tracks/Challenges/TMDriver (the copy
the local game loads). Replays go to data/tmx/<track_id>/<ReplayId>.Replay.Gbx.

Per map: 1 replay list + 1 map file + up to N replays. The request rate is set by
--rate (cheatoskar OK'd 3/s for a 1.5 h window on 2026-09-24); nothing here parallelises.
The map UID is read from the map file itself, so no extra metadata request is needed.
"""
import json
import threading
import time
from pathlib import Path

import pandas as pd

from . import tmx
from .collect import DEFAULT_META, holdout
from .paths import DATA

BULK = DATA / 'bulk'
MAPS_LOG = BULK / 'maps.jsonl'
MAP_DIR = DATA / 'maps'
REP_DIR = DATA / 'tmx'
MIN_REPLAYS = 3
# 1 Stunt (score, not time) and 10 PressForward (no steering): nothing to learn for a driver
SKIP_TAGS = {1, 10}


def candidates(meta_path: Path = DEFAULT_META, min_ms=5000, max_ms=120000) -> pd.DataFrame:
    # replay_count is empty in this parquet: the replay list request decides instead
    d = pd.read_parquet(meta_path, columns=['track_id', 'name', 'tags', 'award_count',
                                           'top_leaderboard_entry_time_ms', 'length_ms', 'difficulty'])
    tags = d.tags.fillna('').map(lambda t: {int(x) for x in str(t).replace(' ', '').split(',') if x != ''})
    ok = (d.top_leaderboard_entry_time_ms.between(min_ms, max_ms) & (d.award_count >= 1)
          & ~tags.map(lambda t: bool(t & SKIP_TAGS)))
    c = d[ok].copy()
    c['track_id'] = c.track_id.astype(int)
    return c.sort_values(['award_count', 'track_id'], ascending=[False, True])


def map_uid(path: Path) -> str:
    from .gbx.reader import Gbx
    g = Gbx(str(path))
    ch = g.get_class_by_id(0x03043000)       # CGameCtnChallenge
    return ch.map_uid if ch is not None and getattr(ch, 'map_uid', None) else ''


def done_ids():
    if not MAPS_LOG.exists():
        return set()
    return {json.loads(line)['track_id'] for line in MAPS_LOG.read_text(encoding='utf-8').splitlines() if line}


def run(n_maps: int, n_replays: int = 5, rate: float = 3.0, hours: float = 1.5, threads_n: int = 4, log=print):
    tmx.MIN_GAP_S = 1.0 / rate
    BULK.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    game_dir = tmx.tracks_dir()
    game_dir.mkdir(parents=True, exist_ok=True)
    done = done_ids()
    cand = candidates()
    log(f'{len(cand)} candidates (awarded, 5-120 s); {len(done)} already done; '
        f'rate {rate}/s for at most {hours} h')
    t_end = time.monotonic() + hours * 3600
    todo = [r for r in cand.itertuples() if int(r.track_id) not in done]
    lock = threading.Lock()
    counts = {'new': 0, 'rep': 0}

    def one(row):
        tid = int(row.track_id)
        entry = {'track_id': tid, 'name': row.name, 'tags': row.tags, 'awards': int(row.award_count),
                 'difficulty': row.difficulty, 'holdout': holdout(tid), 'replays': []}
        n_rep = 0
        try:
            reps = tmx.replay_list(tid, count=25)
            if len(reps) < MIN_REPLAYS:
                raise ValueError(f'only {len(reps)} replays')
            reps = reps[:n_replays]
            mp = MAP_DIR / f'{tid}.Challenge.Gbx'
            if not mp.exists():
                mp.write_bytes(tmx.get(f'{tmx.BASE}/trackgbx/{tid}'))
            gp = game_dir / f'{tmx.safe_name(str(row.name))}_{tid}.Challenge.Gbx'
            if not gp.exists():
                gp.write_bytes(mp.read_bytes())
            entry['map_file'] = gp.name
            try:
                entry['uid'] = map_uid(mp)
            except Exception as e:           # the reader is best effort; resim re-checks
                entry['uid'] = ''
                entry['uid_error'] = repr(e)[:120]
            rdir = REP_DIR / str(tid)
            rdir.mkdir(parents=True, exist_ok=True)
            for r in reps:
                rp = rdir / f"{r['ReplayId']}.Replay.Gbx"
                if not rp.exists():
                    rp.write_bytes(tmx.get(f"{tmx.BASE}/recordgbx/{r['ReplayId']}"))
                    n_rep += 1
                entry['replays'].append({'file': rp.name, 'replay_id': r['ReplayId'],
                                         'time_ms': r['ReplayTime'], 'rank': r.get('Position')})
            entry['ok'] = True
        except Exception as e:               # keep going; record why
            entry['ok'] = False
            entry['error'] = repr(e)[:200]
        with lock:
            with MAPS_LOG.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            done.add(tid)
            counts['new'] += 1
            counts['rep'] += n_rep
            if counts['new'] % 25 == 0:
                log(f"{len(done)} maps done ({counts['new']} this run, {counts['rep']} replays)")

    def worker(it):
        for row in it:
            if len(done) >= n_maps or time.monotonic() > t_end:
                return
            one(row)

    shared = iter(todo)
    it_lock = threading.Lock()

    def safe_iter():
        while True:
            with it_lock:
                row = next(shared, None)
            if row is None:
                return
            yield row

    threads = [threading.Thread(target=worker, args=(safe_iter(),)) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"finished: {len(done)} maps, {counts['new']} this run, {counts['rep']} new replays")


def refetch(rate: float = 8.0, threads_n: int = 12, log=print):
    """Download exactly the maps + replays listed in data/bulk/maps.jsonl (e.g. on a server
    that got the list but not the files). No replay-list requests; existing files are kept."""
    tmx.MIN_GAP_S = 1.0 / rate
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for line in MAPS_LOG.read_text(encoding='utf-8').splitlines():
        m = json.loads(line)
        if not m.get('ok'):
            continue
        mp = MAP_DIR / f"{m['track_id']}.Challenge.Gbx"
        if not mp.exists():
            todo.append((mp, f"{tmx.BASE}/trackgbx/{m['track_id']}"))
        for r in m['replays']:
            rp = REP_DIR / str(m['track_id']) / r['file']
            if not rp.exists():
                todo.append((rp, f"{tmx.BASE}/recordgbx/{r['replay_id']}"))
    log(f'{len(todo)} files to fetch at {rate}/s ({len(todo) / rate / 60:.0f} min)')
    lock = threading.Lock()
    it = iter(todo)
    done = {'n': 0, 'err': 0}

    def worker():
        while True:
            with lock:
                job = next(it, None)
            if job is None:
                return
            path, url = job
            try:
                data = tmx.get(url)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(data)
                tmp.replace(path)
            except Exception:
                with lock:
                    done['err'] += 1
            with lock:
                done['n'] += 1
                if done['n'] % 500 == 0:
                    log(f"{done['n']}/{len(todo)} ({done['err']} errors)")

    ts = [threading.Thread(target=worker) for _ in range(threads_n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log(f"refetch done: {done['n']} files, {done['err']} errors")
