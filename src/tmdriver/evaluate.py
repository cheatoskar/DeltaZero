"""Closed-loop evaluation: load maps, let the model drive from the start, measure.

This is the number that matters. Offline accuracy can look fine while the car leaves the
track after two seconds (errors compound once the car is somewhere no human was).

Works with both checkpoint kinds (M1 `Policy`, ghost-feature `GhostPolicy`). Map sets:
    holdout | train | all          the M1 manifest (80 maps)
    bulk-holdout | bulk            the bulk download (data/bulk/maps.jsonl)
    comma-separated TMX ids        from either list
"""
import json
import time
from typing import List

from .collect import MANIFEST
from .ghost_policy import load_policy
from .paths import DRIVER_CKPT, RUNS
from .policy import reference_line, reference_positions
from .session import Episode, GameSession


class ModelEpisode(Episode):
    STALL_MS = 3000

    def __init__(self, policy, blocks, line, limit_ms: int, meta: dict, pace: float = 0.0):
        self.policy, self.blocks, self.line, self.limit, self.meta, self.pace = policy, blocks, line, limit_ms, meta, pace

    def begin(self, start):
        self.policy.reset(self.blocks, self.line)
        self.best, self.best_t, self.finish, self.reason = 0.0, 0, None, None
        self.last_t = 0
        self.wall0 = time.perf_counter()

    def act(self, st):
        self.last_t = st.race_time
        if st.finished:
            self.finish, self.reason = st.race_time, 'finish'
            return None
        if st.race_time > self.limit:
            self.reason = 'time limit'
            return None
        a = self.policy.act(st, self.pace)
        prog = self.policy.progress_m
        if prog > self.best + 2.0 or st.race_time <= 0:
            self.best, self.best_t = max(prog, self.best), st.race_time
        elif st.race_time - self.best_t > self.STALL_MS:
            self.reason = 'stalled'
            return None
        return a

    def result(self):
        length = getattr(self.policy.line, 'length', float('inf'))
        known = length != float('inf')
        return dict(self.meta, finished=self.finish is not None, time_ms=self.finish, reason=self.reason,
                    progress_m=round(self.best, 1), line_m=round(length, 1) if known else None,
                    progress=round(self.best / max(length, 1e-9), 3) if known else None, race_ms=self.last_t,
                    wall_s=round(time.perf_counter() - self.wall0, 2))


def map_list(which: str) -> List[dict]:
    maps = []
    if MANIFEST.exists():
        for m in json.loads(MANIFEST.read_text(encoding='utf-8'))['maps'].values():
            if m.get('ok'):
                maps.append(dict(m, best_ms=min((r['time_ms'] for r in m['replays'] if 'time_ms' in r),
                                                default=60000)))
    bulk = []
    from .bulk import MAPS_LOG
    if MAPS_LOG.exists():
        for line in MAPS_LOG.read_text(encoding='utf-8').splitlines():
            m = json.loads(line)
            if m.get('ok') and m.get('uid') and m.get('map_file'):
                bulk.append(dict(m, best_ms=min((r['time_ms'] for r in m['replays']), default=60000)))
    if which == 'holdout':
        return [m for m in maps if m['holdout']]
    if which == 'train':
        return [m for m in maps if not m['holdout']]
    if which == 'all':
        return maps
    if which == 'bulk-holdout':
        return [m for m in bulk if m['holdout']]
    if which == 'bulk':
        return bulk
    ids = {int(x) for x in which.split(',')}
    seen, out = set(), []
    for m in maps + bulk:
        if m['track_id'] in ids and m['track_id'] not in seen:
            seen.add(m['track_id'])
            out.append(m)
    return out


def run_eval(link, which: str = 'holdout', watch: bool = False, speed: float = 1.0, limit: int = 0,
             log=print) -> List[dict]:
    maps = map_list(which)
    if limit:
        maps = maps[:limit]
    policy = load_policy(DRIVER_CKPT)
    ghost = getattr(policy, 'ghost', False)
    log(f'{len(maps)} maps, model {DRIVER_CKPT}')
    sess = GameSession(link, log)
    results = []
    for k, m in enumerate(maps):
        if not sess.load_map(m['map_file'], m['uid']):
            log(f"SKIP {m['name']!r}: map did not load")
            continue
        if ghost:
            ref = reference_positions(m['uid'], m['track_id'])
            line, src = (ref[0], ref[1]) if ref else (None, 'no line (blocks only)')
        else:
            ref = reference_line(m['uid'], m['track_id'])
            if ref is None:
                log(f"SKIP {m['name']!r}: no reference path")
                continue
            line, src = ref
        best = m['best_ms']
        sess.status(f"Evaluation {k + 1}/{len(maps)}: {m['name']}")
        if watch:
            link.speed(speed)
        blocks = [b.__dict__ for b in sess.map.blocks]
        if ghost:
            policy.reset(blocks, line)
            line = policy.line if policy.has_line else None   # build the line once per map
        ep = ModelEpisode(policy, blocks, line, limit_ms=2 * best + 10000,
                          meta={'track_id': m['track_id'], 'name': m['name'], 'holdout': m['holdout'],
                                'best_ms': best, 'reference': src})
        r = sess.run([ep], sim_only=not watch)[0]
        results.append(r)
        if r['finished']:
            t = f"{r['time_ms'] / 1000:.2f}s (best {best / 1000:.2f}s)"
        else:
            t = f"{r['reason']} at {r['progress_m']:.0f} m" + (f" ({r['progress']:.0%})" if r['progress'] else '')
        log(f"[{k + 1}/{len(maps)}] {m['name']!r}{' [held-out]' if m['holdout'] else ''}: {t}")
    fin = [r for r in results if r['finished']]
    summary = f"{len(fin)}/{len(results)} finished"
    if fin:
        summary += f", im Mittel {sum(r['time_ms'] / r['best_ms'] for r in fin) / len(fin):.2f}x Bestzeit"
    sess.status(f'Evaluation done: {summary}')
    out = RUNS / 'eval' / f"eval_{which.replace(',', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'model': str(DRIVER_CKPT), 'results': results}, indent=1), encoding='utf-8')
    log(f'{summary} -> {out}')
    return results
