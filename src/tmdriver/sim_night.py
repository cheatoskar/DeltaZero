"""Download TMX maps with their replays and re-simulate them WITHOUT the game (TMNF-C), on all
CPU cores, while the next maps download.

Pool: every map on TMNF-X with at least `min_awards` awards, most awarded first, read in pages
of 1000 (`order1=6` = awards descending, `after=<last TrackId>`) and kept in
data/tmx_award_pool.jsonl, so a later night continues the list instead of asking again.
Nadeo's own maps (author "Nadeo", not the [Beta] / [Early Build] uploads) get more replays.

Files:
  data/maps/<name>_<track_id>.Challenge.Gbx    the map (always kept)
  data/tmx/<map uid>/<replay id>.Replay.Gbx    its replays
  data/resim_c/<track_id>/<replay id>.npz      the re-simulated run (t, pos, rot, vel, ang_vel)
  data/resim_c/index.jsonl                     one line per replay (exact against the ghost?)
  data/resim_c/maps.jsonl                      one line per finished map (resume)
  data/resim_c/status.json, errors.jsonl       progress and problems
"""
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from .paths import DATA, TMX, safe

POOL_FILE = DATA / 'tmx_award_pool.jsonl'
POOL_STATE = DATA / 'tmx_award_pool.json'
MAP_DIR = DATA / 'maps'
PAGE = 1000
FETCHERS = 4                       # parallel download threads (one request rate for all)
FIELDS = ('TrackId,TrackName,UId,Authors[],Tags[],AuthorTime,Routes,Difficulty,Environment,Car,'
          'PrimaryType,Mood,Awards,Comments,UploadedAt,WRReplay.ReplayTime,ReplayType')


def is_nadeo(m: dict) -> bool:
    names = [((a or {}).get('User') or {}).get('Name', '') for a in (m.get('Authors') or [])]
    return any(n.strip().lower() == 'nadeo' for n in names) and '[' not in m.get('TrackName', '')


def award_pool(min_awards: int, log=print) -> Iterator[dict]:
    """Maps with >= min_awards awards, most awarded first. Pages already read come from
    POOL_FILE; new pages are appended to it."""
    from . import tmx
    seen = set()
    last = None
    if POOL_FILE.exists():
        for line in POOL_FILE.read_text(encoding='utf-8').splitlines():
            m = json.loads(line)
            if m['TrackId'] in seen:
                continue
            seen.add(m['TrackId'])
            last = m
            if (m.get('Awards') or 0) < min_awards:
                return
            yield m
    state = json.loads(POOL_STATE.read_text(encoding='utf-8')) if POOL_STATE.exists() else {}
    if state.get('complete'):
        return
    after = last['TrackId'] if last else None
    while True:
        url = f'{tmx.BASE}/api/tracks?order1=6&count={PAGE}&fields={FIELDS}' + (f'&after={after}' if after else '')
        d = json.loads(tmx.get(url))
        rs = d.get('Results', [])
        POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(POOL_FILE, 'a', encoding='utf-8') as f:
            for m in rs:
                f.write(json.dumps(m) + '\n')
        log(f'map pool: +{len(rs)} maps (awards {rs[0]["Awards"] if rs else "-"} .. '
            f'{rs[-1]["Awards"] if rs else "-"}) -> {POOL_FILE}')
        for m in rs:
            if m['TrackId'] in seen:
                continue
            seen.add(m['TrackId'])
            if (m.get('Awards') or 0) < min_awards:
                return
            yield m
        if not rs or not d.get('More'):
            POOL_STATE.write_text(json.dumps({'complete': True}), encoding='utf-8')
            return
        after = rs[-1]['TrackId']


def good_replays(track_id: int, n: int, within: float) -> list:
    """The fastest replays of a map that are within `within` x the best time (1.05 = 5 % slower
    than the record at most), at most n: a slow run teaches slow driving."""
    from . import tmx
    reps = [r for r in tmx.replay_list(track_id, max(25, n)) if r.get('ReplayTime')]
    if not reps:
        return []
    best = reps[0]['ReplayTime']
    return [r for r in reps if r['ReplayTime'] <= best * within][:n]


def _work(job):
    """One map in a worker process: returns (job, results or None, error or None)."""
    from . import tmnfc
    try:
        res = tmnfc.resim_map(Path(job['map']), job['track_id'], [Path(p) for p in job['replays']],
                              shift=job['shift'], sign=job['sign'])
        return job, res, None
    except Exception as e:                           # noqa: BLE001
        return job, None, repr(e)[:300]


class SimNight:
    def __init__(self, hours: float = 8.0, min_awards: int = 2, replays: int = 20, nadeo_replays: int = 100,
                 within: float = 1.05, workers: int = 0, prefetch: int = 0, log=print):
        from . import tmnfc
        self.root = tmnfc.RESIM_C
        self.t_end = time.monotonic() + hours * 3600
        self.min_awards, self.replays, self.nadeo_replays = min_awards, replays, nadeo_replays
        self.within = within
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch or self.workers * 3)
        self.log = log
        self.lock = threading.Lock()
        self.fetch_done = threading.Event()
        self.stats = {'maps_done': 0, 'maps_fetched': 0, 'replays': 0, 'exact': 0, 'skipped': 0,
                      'not_exact': 0, 'errors': 0}
        self.t0 = time.monotonic()

    def time_left(self) -> bool:
        return time.monotonic() < self.t_end

    def done_maps(self) -> set:
        """Maps finished or failed on an earlier night (both are skipped)."""
        out = set()
        for name in ('maps.jsonl', 'errors.jsonl'):
            f = self.root / name
            if f.exists():
                out |= {json.loads(l)['track_id'] for l in f.read_text(encoding='utf-8').splitlines() if l.strip()}
        return out

    def error(self, tid, name, msg):
        with self.lock:
            self.stats['errors'] += 1
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.root / 'errors.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'track_id': tid,
                                    'name': name, 'error': msg}) + '\n')
        self.log(f'  ERROR {tid} {name!r}: {msg}'[:240])

    def fetcher(self, shift: int, sign: int):
        """FETCHERS threads share the pool iterator; tmx.get keeps them under the request rate
        (TMX answers take ~0.3-0.5 s each, so one thread alone cannot reach it)."""
        done = self.done_maps()
        pool = award_pool(self.min_awards, self.log)
        pool_lock = threading.Lock()

        def next_map():
            with pool_lock:
                return next(pool, None)

        def fetch_loop():
            from . import tmx
            while self.time_left():
                try:
                    m = next_map()
                except Exception as e:           # noqa: BLE001
                    self.error(0, 'pool', repr(e))
                    return
                if m is None:
                    return
                tid = int(m['TrackId'])
                if tid in done:
                    continue
                n = self.nadeo_replays if is_nadeo(m) else self.replays
                try:
                    reps = good_replays(tid, n, self.within)
                    if not reps:
                        self.error(tid, m.get('TrackName', ''), 'no replays on TMX')
                        continue
                    info = ({'TrackId': tid, 'TrackName': m.get('TrackName', ''), 'UId': m['UId']}
                            if m.get('UId') else None)
                    res = tmx.fetch(tid, len(reps), TMX, safe, log=lambda *a: None, reps=reps, map_dir=MAP_DIR,
                                    info=info)
                except Exception as e:           # noqa: BLE001  (removed from TMX, network)
                    self.error(tid, m.get('TrackName', ''), f'fetch: {e!r}')
                    continue
                with self.lock:
                    self.stats['maps_fetched'] += 1
                self.q.put({'track_id': tid, 'name': m.get('TrackName', ''), 'awards': m.get('Awards'),
                            'uid': res['uid'], 'map': str(res['map']), 'replays': [str(p) for p in res['replays']],
                            'nadeo': is_nadeo(m), 'tags': m.get('Tags'), 'shift': shift, 'sign': sign})

        threads = [threading.Thread(target=fetch_loop, daemon=True) for _ in range(FETCHERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.fetch_done.set()

    def jobs(self) -> Iterator[dict]:
        while True:
            try:
                yield self.q.get(timeout=1.0)
            except queue.Empty:
                if self.fetch_done.is_set() and self.q.empty():
                    return

    def record(self, job, res, err):
        if err is not None:
            self.error(job['track_id'], job['name'], err)
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with open(self.root / 'index.jsonl', 'a', encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(dict(r, uid=job['uid'], name=job['name'])) + '\n')
            ok = sum(1 for r in res if r.get('exact'))
            skipped = sum(1 for r in res if r.get('skipped'))
            s = self.stats
            s['maps_done'] += 1
            s['replays'] += len(res)
            s['exact'] += ok
            s['skipped'] += skipped
            s['not_exact'] += len(res) - ok - skipped
            with open(self.root / 'maps.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'track_id': job['track_id'], 'name': job['name'], 'awards': job['awards'],
                                    'replays': len(res), 'exact': ok, 'skipped': skipped,
                                    'nadeo': job['nadeo']}) + '\n')
            st = self.write_status()
        self.log(f"{job['name']!r} ({job['awards']} awards): {ok}/{len(res)} exact"
                 f"{f', {skipped} skipped' if skipped else ''} | {st['maps_done']} maps, "
                 f"{st['exact']}/{st['replays']} exact, {st['maps_per_hour']} maps/h, {self.q.qsize()} ready")

    def write_status(self) -> dict:
        h = max(1e-6, (time.monotonic() - self.t0) / 3600)
        st = dict(self.stats, maps_per_hour=round(self.stats['maps_done'] / h),
                  replays_per_hour=round(self.stats['replays'] / h), workers=self.workers,
                  updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        (self.root / 'status.json').write_text(json.dumps(st, indent=1), encoding='utf-8')
        return st

    def run(self, shift: int = 1, sign: int = -1) -> dict:
        from . import tmnfc
        missing = tmnfc.available()
        if missing:
            raise SystemExit(f'TMNF-C is not set up: {missing}')
        self.root.mkdir(parents=True, exist_ok=True)
        self.log(f'client-free night: {self.workers} worker processes, maps with >= {self.min_awards} awards, '
                 f'up to {self.replays} replays per map ({self.nadeo_replays} on Nadeo maps) within '
                 f'{(self.within - 1) * 100:.0f} % of the best time -> {self.root}')
        threading.Thread(target=self.fetcher, args=(shift, sign), daemon=True).start()
        ctx = mp.get_context('spawn')
        try:
            with ctx.Pool(self.workers) as pool:
                for job, res, err in pool.imap_unordered(_work, self.jobs()):
                    self.record(job, res, err)
                    if not self.time_left():
                        self.log('time is up')
                        break
        except KeyboardInterrupt:
            self.log('stopped')
        st = self.write_status()
        self.log(f"night done: {st['maps_done']} maps, {st['exact']}/{st['replays']} replays exact, "
                 f"{st['skipped']} skipped, {st['errors']} errors -> {self.root}")
        return st
