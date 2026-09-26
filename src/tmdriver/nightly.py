"""Overnight re-simulation: download the most awarded TMX maps with their fastest replays while
the game instances re-simulate the ones already downloaded, and keep going through errors.

    python tmdriver.py night --hours 8          (serve must NOT run; Night_Resim.bat)

* A fetcher thread walks the map pool (data/pool/maps.json; made or extended from the TMX
  search when it is too short) and downloads each map (into the game folder and
  data/maps/<id>.Challenge.Gbx) and its fastest replays (data/tmx/<uid>/). It stays up to
  PREFETCH maps ahead of the simulation and skips maps that are already done.
* One worker per game instance (main + running helpers) takes the next downloaded map and
  re-simulates it (resim.resim_map): data/resim/<track_id>/<replay>.npz, index.jsonl.
* A map that fails is written to data/resim/night_errors.jsonl and skipped (and not tried again
  on later nights). A lost connection (the game crashed or the plugin reloaded) makes that
  worker reconnect every 15 s and carry on with the next map.
* data/resim/night_status.json is rewritten after every map: maps and replays done, the exact
  share, replays per hour, errors, the time of the last update.
"""
import json
import os
import queue
import threading
import time
from typing import List, Optional

from . import replay as replay_mod
from .collect import holdout
from .link import Link
from .paths import POOL, TMX, safe
from .resim import BULK_RESIM, DEFAULT_REPLAY_ALIGNMENT, MapNotLoaded, pending_replays, resim_map
from .session import GameSession

PREFETCH = 6
ERRORS = BULK_RESIM / 'night_errors.jsonl'
STATUS = BULK_RESIM / 'night_status.json'


def ensure_pool(n: int, log=print) -> List[dict]:
    """The pool's maps; made or extended from the TMX search if it has fewer than n."""
    from . import tmx
    maps = json.loads(POOL.read_text(encoding='utf-8'))['maps'] if POOL.exists() else []
    if len(maps) < n:
        log(f'map pool: {len(maps)} maps, fetching the list of the {n} most awarded from TMX ...')
        found = tmx.search_tracks(n, log=lambda *a: None)
        have = {m['track_id'] for m in maps}
        for m in found:
            if m['track_id'] not in have:
                m['holdout'] = holdout(m['track_id'])
                maps.append(m)
        POOL.parent.mkdir(parents=True, exist_ok=True)
        POOL.write_text(json.dumps({'made': time.strftime('%Y-%m-%d %H:%M'), 'maps': maps}, indent=1),
                        encoding='utf-8')
        log(f'map pool: {len(maps)} maps -> {POOL}')
    return maps


def failed_maps() -> set:
    if not ERRORS.exists():
        return set()
    out = set()
    for line in ERRORS.read_text(encoding='utf-8').splitlines():
        try:
            out.add(int(json.loads(line)['track_id']))
        except (ValueError, KeyError):
            continue
    return out


class Night:
    def __init__(self, links: List[Link], ports: List[int], hours: float, n_maps: int, replays: int,
                 batch: bool = True, load_replay=replay_mod.load, log=print):
        self.links, self.ports, self.log, self.load_replay = links, ports, log, load_replay
        self.t_end = time.monotonic() + hours * 3600.0
        self.n_maps, self.replays, self.batch = n_maps, replays, batch
        self.q: 'queue.Queue' = queue.Queue(maxsize=PREFETCH)
        self.lock = threading.Lock()
        self.fetch_done = threading.Event()
        self.busy = 0                     # maps being re-simulated right now
        self.final = {}                   # instance -> its session when the night ended
        self.t0 = time.monotonic()
        self.stats = {'maps_done': 0, 'maps_skipped': 0, 'replays': 0, 'exact': 0, 'errors': 0,
                      'fetched': 0, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}

    def time_left(self) -> bool:
        return time.monotonic() < self.t_end

    # ---------------------------------------------------------------- fetching
    def fetcher(self):
        from . import tmx
        from .bulk import MAP_DIR
        try:
            done_fail = failed_maps()
            for m in ensure_pool(self.n_maps, self.log):
                if not self.time_left():
                    break
                tid = m['track_id']
                if tid in done_fail:
                    continue
                folder = BULK_RESIM / str(tid)
                if folder.exists() and len(list(folder.glob('*.npz'))) >= self.replays:
                    continue                  # re-simulated on an earlier night
                try:
                    res = tmx.fetch(tid, self.replays, TMX, safe, log=lambda *a: None)
                except Exception as e:        # removed from TMX, network trouble: next map
                    self.error(tid, m.get('name', ''), f'fetch: {e!r}')
                    continue
                MAP_DIR.mkdir(parents=True, exist_ok=True)
                keep = MAP_DIR / f'{tid}.Challenge.Gbx'
                if not keep.exists():
                    keep.write_bytes(res['map'].read_bytes())
                entry = {'ok': True, 'track_id': tid, 'uid': res['uid'], 'name': res['info']['TrackName'],
                         'map_file': res['map'].name, 'holdout': holdout(tid),
                         'replays': [{'file': p.name, 'path': str(p)} for p in res['replays']]}
                if not pending_replays(entry, BULK_RESIM, self.replays):
                    continue
                with self.lock:
                    self.stats['fetched'] += 1
                while self.time_left():
                    try:
                        self.q.put(entry, timeout=5.0)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.log(f'fetcher stopped: {e!r}')
        finally:
            self.fetch_done.set()

    # ---------------------------------------------------------------- bookkeeping
    def error(self, track_id, name, what):
        with self.lock:
            self.stats['errors'] += 1
            ERRORS.parent.mkdir(parents=True, exist_ok=True)
            with open(ERRORS, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'track_id': track_id, 'name': name, 'error': what[:300],
                                    'time': time.strftime('%Y-%m-%d %H:%M:%S')}) + '\n')
        self.log(f'  ERROR {track_id} {name!r}: {what}'[:200])

    def write_status(self):
        el = max(time.monotonic() - self.t0, 1.0)
        st = dict(self.stats, hours=round(el / 3600, 2), replays_per_hour=round(self.stats['replays'] / el * 3600),
                  exact_share=round(self.stats['exact'] / max(self.stats['replays'], 1), 4),
                  updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(st, indent=1), encoding='utf-8')
        return st

    # ---------------------------------------------------------------- simulating
    def connect(self, k: int) -> Optional[GameSession]:
        """(Re)connect to instance k; waits and retries every 15 s while there is time left."""
        tag = f'[#{k + 1}] ' if len(self.ports) > 1 else ''
        while self.time_left():
            try:
                link = Link.connect(port=self.ports[k], wait_s=0.0)
                return GameSession(link, lambda *a: self.log(tag, *a) if tag else self.log(*a))
            except OSError:
                self.log(f'{tag}no game on port {self.ports[k]}; retrying in 15 s')
                time.sleep(15.0)
        return None

    def all_done(self) -> bool:
        """Nothing left: the fetcher finished, the queue is empty and no instance holds a map
        (a map in work can come back to the queue after a crash or a failed load)."""
        with self.lock:
            return self.fetch_done.is_set() and self.q.empty() and self.busy == 0

    def worker(self, k: int, sess: Optional[GameSession], shift: int, sign: int):
        try:
            sess = self._work(k, sess, shift, sign)
        finally:
            self.final[k] = sess              # the connection it ended with (None: lost)

    def _work(self, k: int, sess: Optional[GameSession], shift: int, sign: int) -> Optional[GameSession]:
        tag = f'[#{k + 1}] ' if len(self.ports) > 1 else ''
        load_fails = 0                        # maps in a row this instance did not load
        while self.time_left():
            if sess is None:                  # reconnect BEFORE taking a map, so none is lost
                if self.all_done():
                    return sess
                sess = self.connect(k)
                if sess is None:
                    return None
            try:
                m = self.q.get(timeout=2.0)
            except queue.Empty:
                if self.all_done():
                    return sess
                continue
            if k in m.get('failed_on', ()) and len(self.ports) > 1:
                self.q.put(m)                 # it failed here once: leave it to another instance
                time.sleep(1.0)
                continue
            with self.lock:
                self.busy += 1
            try:
                sess, load_fails = self.one_map(sess, m, shift, sign, tag, load_fails, k)
            finally:
                with self.lock:
                    self.busy -= 1
        return sess

    def one_map(self, sess: GameSession, m: dict, shift: int, sign: int, tag: str, load_fails: int, k: int):
        """Re-simulate one map on one instance. Returns (session or None to reconnect, load_fails)."""
        try:
            res = resim_map(sess, m, shift, sign, BULK_RESIM, max_replays=self.replays, batch=self.batch,
                            load_replay=self.load_replay, log=self.log, tag=tag,
                            status=f"Night re-simulation: {m['name']}")
        except (ConnectionError, TimeoutError, OSError) as e:
            # the game crashed or the plugin reloaded: another instance (or this one after it
            # reconnects) tries the map once more; a second failure marks it as bad
            m['tries'] = m.get('tries', 0) + 1
            m.setdefault('failed_on', []).append(k)
            if m['tries'] >= 2:
                self.error(m['track_id'], m['name'], f'connection, twice: {e!r}')
            else:
                self.log(f"{tag}lost the game during {m['name']!r} ({e!r}); the map goes back to the queue"[:200])
                self.q.put(m)
            try:
                sess.link.close()
            except Exception:
                pass
            return None, load_fails
        except MapNotLoaded as e:
            m['tries'] = m.get('tries', 0) + 1
            m.setdefault('failed_on', []).append(k)
            load_fails += 1
            if m['tries'] >= 2:
                self.error(m['track_id'], m['name'], f'did not load, twice: {e}')
            else:
                self.log(f'{tag}{e}; the map goes back to the queue')
                self.q.put(m)                 # maybe another instance can load it
            if load_fails >= 2:               # this instance seems to hang: reconnect in a minute
                self.log(f'{tag}two maps in a row did not load: reconnecting to this instance in 60 s')
                try:
                    sess.link.close()
                except Exception:
                    pass
                time.sleep(60.0)
                return None, 0
            return sess, load_fails
        except Exception as e:
            self.error(m['track_id'], m['name'], repr(e))
            return sess, load_fails
        if not res:
            with self.lock:
                self.stats['maps_skipped'] += 1
            return sess, 0
        with self.lock:
            with open(BULK_RESIM / 'index.jsonl', 'a', encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(r) + '\n')
            ok = sum(r['exact'] for r in res)
            self.stats['maps_done'] += 1
            self.stats['replays'] += len(res)
            self.stats['exact'] += ok
            st = self.write_status()
        self.log(f"{tag}{m['name']!r}: {ok}/{len(res)} exact | night: {st['maps_done']} maps, "
                 f"{st['exact']}/{st['replays']} exact, {st['replays_per_hour']} replays/h, "
                 f"{st['errors']} errors, {self.q.qsize()} maps ready")
        return sess, 0

    def run(self):
        BULK_RESIM.mkdir(parents=True, exist_ok=True)
        sessions = [GameSession(l, (lambda *a, k=k: self.log(f'[#{k + 1}]', *a)) if len(self.links) > 1 else self.log)
                    for k, l in enumerate(self.links)]
        cal = sessions[0].calib.data.get('replay') or DEFAULT_REPLAY_ALIGNMENT
        shift, sign = int(cal.get('shift', 1)), int(cal.get('steer_sign', -1))
        self.log(f'night re-simulation on {len(sessions)} game instance(s) for '
                 f'{(self.t_end - time.monotonic()) / 3600:.1f} h; alignment shift {shift}, sign {sign}; '
                 f'status: {STATUS}')
        fetch = threading.Thread(target=self.fetcher, daemon=True)
        fetch.start()
        workers = [threading.Thread(target=self.worker, args=(k, s, shift, sign), daemon=True)
                   for k, s in enumerate(sessions)]
        for w in workers:
            w.start()
        try:
            for w in workers:
                while w.is_alive():
                    w.join(1.0)
        except KeyboardInterrupt:
            self.log('stopped')
            self.t_end = time.monotonic()
        if os.environ.get('TMDRIVER_DRAW_GAME') == '0':     # --no-draw: the games render again
            for sess in self.final.values():
                try:
                    sess.link.execute('set draw_game true')
                    sess.link.flush()
                except (AttributeError, OSError):
                    pass
            time.sleep(1.0)                   # let the plugins read it before the process exits
        st = self.write_status()
        self.log(f"night done: {st['maps_done']} maps, {st['exact']}/{st['replays']} replays exact, "
                 f"{st['replays_per_hour']} replays/h, {st['errors']} errors -> {STATUS}")
        return st
