"""Re-simulate TMX replays in the game: recorded inputs -> full physics state every 10 ms.

Each replay is replayed from the race start with the alignment the self test measured
(calibration 'replay': shift 1, analog steer sign -1). A run counts as exact when it
reaches the finish at the recorded time AND matches every ghost sample within 1 mm; only
exact runs are used for training. Output per replay: data/m1/resim/<track_id>/<replay>.npz
with every STEP field (the same fields the live driver sees) plus the ACTION applied at
that tick, which is the training label.
"""
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import protocol as P
from . import replay as replay_mod
from .collect import M1, MANIFEST
from .paths import TMX, safe
from .session import Episode, GameSession
from .tasks import steps_to_arrays

RESIM = M1 / 'resim'
BULK_RESIM = M1.parent / 'resim'      # data/resim: re-simulated bulk replays
# A wrong input alignment is off by > 1 m within seconds (measured: 1.15 m on lolsport);
# float32 transport noise at ~1 km coordinates is ~1e-4 m. The real game gave 0.0 exactly.
EXACT_TOL_M = 1e-3


def table_action(table: Dict[str, np.ndarray], t: int, shift: int):
    i = (t + 10 * shift) // 10
    if i < 0 or i >= len(table['t']):
        return 0, 0, 0
    return int(table['in_steer'][i]), int(table['in_gas'][i]), int(table['in_bits'][i])


class ResimEpisode(Episode):
    def __init__(self, rep: replay_mod.Replay, shift: int, steer_sign: int, meta: dict, out_root: Path = None):
        self.rep, self.shift, self.meta = rep, shift, meta
        self.out_root = out_root or RESIM
        self.table = replay_mod.input_table(rep, steer_sign=steer_sign)
        self.ghost = {int(t): p for t, p in zip(rep.ghost_t, rep.ghost_pos)}

    def begin(self, start):
        self.steps: List[P.Step] = []
        self.actions: List[tuple] = []
        self.finished_at = None
        self.wall0 = time.perf_counter()

    def act(self, st: P.Step):
        repeat = bool(self.steps) and st.race_time == self.steps[-1].race_time
        if not repeat:   # race time can freeze at the finish; a repeat is not a new tick
            self.steps.append(st)
            self.actions.append((0, 0, 0))
        if st.finished:
            self.finished_at = st.race_time
            return None
        if repeat or st.race_time > self.rep.race_time_ms + 200:
            return None
        a = table_action(self.table, st.race_time, self.shift)
        self.actions[-1] = a
        return a

    def plan(self):
        """The replay's inputs for race times 0, 10, ... up to the limit `act` stops at."""
        n = (self.rep.race_time_ms + 200) // 10 + 1
        return [table_action(self.table, k * 10, self.shift) for k in range(n)]

    def result(self) -> dict:
        pos = {}
        for s in self.steps:
            pos.setdefault(s.race_time, s.pos.astype(np.float64))
        common = sorted(set(pos) & set(self.ghost))
        d = np.array([np.linalg.norm(pos[t] - self.ghost[t]) for t in common]) if common else np.array([np.inf])
        finished_at = self.finished_at
        exact = bool(finished_at == self.rep.race_time_ms and d.max() <= EXACT_TOL_M)
        out = dict(self.meta, exact=exact, finished_at=finished_at, recorded_ms=self.rep.race_time_ms,
                   ghost_max_diff_m=float(d.max()), ticks=len(self.steps),
                   wall_s=round(time.perf_counter() - self.wall0, 3))
        a = steps_to_arrays(self.steps)
        act = np.array(self.actions, dtype=np.int64).reshape(-1, 3)
        path = self.out_root / str(self.meta['track_id']) / f"{self.meta['replay']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, act_steer=act[:, 0], act_gas=act[:, 1], act_bits=act[:, 2],
                            meta=json.dumps(out), **a)
        out['file'] = str(path)
        return out


def bulk_maps() -> List[dict]:
    """The bulk download as run_resim map entries (replay paths under data/tmx/<id>/)."""
    from .bulk import MAPS_LOG, REP_DIR
    out = []
    for line in MAPS_LOG.read_text(encoding='utf-8').splitlines():
        m = json.loads(line)
        if m.get('ok') and m.get('uid') and m.get('map_file'):
            out.append({'ok': True, 'track_id': m['track_id'], 'uid': m['uid'], 'name': m['name'],
                        'map_file': m['map_file'], 'holdout': m['holdout'],
                        'replays': [{'file': r['file'], 'path': str(REP_DIR / str(m['track_id']) / r['file']),
                                     'uid_ok': None} for r in m['replays']]})
    return out


def run_resim(link, only_missing: bool = True, max_replays: int = 5, log=print,
              load_replay=replay_mod.load, limit: int = 0, source: str = 'm1', start: int = 0,
              hours: float = 0.0, batch: bool = True) -> List[dict]:
    """source 'm1' (the 80-map manifest) or 'bulk' (data/bulk/maps.jsonl -> data/resim).
    hours > 0 stops after that long (between maps). batch: the plugin plays each replay's
    inputs itself (no Python round trip per tick)."""
    out_root = BULK_RESIM if source == 'bulk' else RESIM
    t_end = time.monotonic() + hours * 3600 if hours else None
    sess = GameSession(link, log)
    cal = sess.calib.data.get('replay')
    if not cal or 'shift' not in cal or 'steer_sign' not in cal:
        raise RuntimeError('calibration has no replay alignment: run the self test on a map with TMX replays')
    shift, sign = int(cal['shift']), int(cal['steer_sign'])
    if source == 'bulk':
        maps = bulk_maps()
    else:
        maps = [m for m in json.loads(MANIFEST.read_text(encoding='utf-8'))['maps'].values() if m.get('ok')]
    maps = maps[start:]
    if limit:
        maps = maps[:limit]
    log(f'{len(maps)} maps; alignment shift {shift}, analog steer sign {sign}')
    index_path = out_root / 'index.jsonl'
    out_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    t_start = time.monotonic()
    for k, m in enumerate(maps):
        if t_end and time.monotonic() > t_end:
            log(f'time budget reached after {k} maps')
            break
        eps = []
        for r in m['replays'][:max_replays]:
            if 'error' in r or r.get('uid_ok') is False or r.get('respawns', 0) > 0:
                continue   # respawn replay-through is not verified yet
            rid = r['file'].split('.')[0]
            if only_missing and (out_root / str(m['track_id']) / f'{rid}.npz').exists():
                continue
            try:
                rep = load_replay(r['path'] if 'path' in r else TMX / safe(m['uid']) / r['file'])
            except Exception as e:
                log(f"  unreadable replay {r['file']}: {e!r}"[:160])
                continue
            if rep.map_uid != m['uid'] or rep.respawns > 0:     # bulk entries: checked here
                continue
            eps.append(ResimEpisode(rep, shift, sign, {
                'track_id': m['track_id'], 'uid': m['uid'], 'replay': rid, 'player': r.get('player'),
                'analog': rep.uses_analog_steer, 'holdout': m['holdout']}, out_root))
        if not eps:
            continue
        sess.status(f"Re-simulation {k + 1}/{len(maps)}: {m['name']} ({len(eps)} replays)")
        if not sess.load_map(m['map_file'], m['uid']):
            log(f"SKIP {m['track_id']} {m['name']!r}: map did not load")
            continue
        res = sess.run(eps, sim_only=True, batch=batch)
        with open(index_path, 'a', encoding='utf-8') as f:
            for r in res:
                f.write(json.dumps({k2: v for k2, v in r.items()}) + '\n')
        ok = sum(r['exact'] for r in res)
        all_results += res
        n_ok = sum(r['exact'] for r in all_results)
        el = time.monotonic() - t_start
        log(f"[{k + 1}/{len(maps)}] {m['name']!r}: {ok}/{len(res)} exact, "
            f"{sum(r['ticks'] for r in res)} ticks in {sum(r['wall_s'] for r in res):.1f}s | total "
            f"{n_ok}/{len(all_results)} exact, {len(all_results) / max(el, 1) * 3600:.0f} replays/h")
    n_ok = sum(r['exact'] for r in all_results)
    sess.status(f'Re-simulation done: {n_ok}/{len(all_results)} replays exact.')
    return all_results
