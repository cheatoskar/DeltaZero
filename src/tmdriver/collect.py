"""M1 data collection: pick TMX maps from the MX metadata, download maps + top replays.

The manifest (data/m1/manifest.json) is the single source of truth for later stages:
which maps, which replays, which split. Maps are split by a hash of the TMX id before
anything else, so no map ever contributes to both training and held-out evaluation.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import replay as replay_mod
from . import tmx
from .paths import DATA, TMX, safe

M1 = DATA / 'm1'
MANIFEST = M1 / 'manifest.json'
DEFAULT_META = Path(__file__).resolve().parents[3] / 'TMTrackNN2.0' / 'data' / 'mx' / 'tracks_details_tmnf.parquet'

# TMNF-X tag ids: 0 Normal 1 Stunt 2 Maze 3 Offroad 4 Laps 5 Fullspeed 6 LOL 7 Tech
# 8 SpeedTech 9 RPG 10 PressForward 11 Trial 12 Grass
SMOKE_TAGS = {0, 5, 6, 7, 8}


def holdout(track_id: int, frac: float = 0.1) -> bool:
    h = int(hashlib.sha1(str(track_id).encode()).hexdigest()[:8], 16)
    return (h % 1000) < frac * 1000


def candidates(meta_path: Path, min_ms=8000, max_ms=25000, min_awards=1) -> pd.DataFrame:
    d = pd.read_parquet(meta_path, columns=['track_id', 'name', 'tags', 'award_count',
                                           'top_leaderboard_entry_time_ms', 'circuit_is_likely_circuit'])
    tags = d.tags.fillna('').map(lambda s: {int(x) for x in str(s).replace(' ', '').split(',') if x != ''})
    ok = (tags.map(lambda t: bool(t) and t <= SMOKE_TAGS)
          & d.top_leaderboard_entry_time_ms.between(min_ms, max_ms)
          & (d.award_count >= min_awards)
          & ~d.circuit_is_likely_circuit.fillna(False))
    c = d[ok].copy()
    c['track_id'] = c.track_id.astype(int)
    return c.sort_values(['award_count', 'track_id'], ascending=[False, True])


def collect(n_maps: int, n_replays: int, meta_path: Path = DEFAULT_META, must: List[int] = (),
            min_replays: int = 3, log=print) -> Dict:
    M1.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'maps': {}}
    cand = candidates(meta_path)
    order = list(must) + [t for t in cand.track_id.tolist() if t not in must]
    log(f'{len(cand)} candidate maps in metadata; want {n_maps}')
    for tid in order:
        have = sum(1 for m in man['maps'].values() if m.get('ok'))
        if have >= n_maps:
            break
        key = str(tid)
        if key in man['maps']:
            continue
        try:
            reps = tmx.replay_list(tid, count=25)
            if len(reps) < min_replays and tid not in must:
                man['maps'][key] = {'ok': False, 'reason': f'{len(reps)} replays'}
                continue
            res = tmx.fetch(tid, n_replays, TMX, safe, log=lambda *a: None, reps=reps)
        except Exception as e:  # keep going; record why
            man['maps'][key] = {'ok': False, 'reason': repr(e)[:200]}
            continue
        entry = {'ok': True, 'track_id': tid, 'uid': res['uid'], 'name': res['info']['TrackName'],
                 'map_file': res['map'].name, 'holdout': holdout(tid) and tid not in must,
                 'tags': res['info'].get('Tags'), 'author_time': res['info'].get('AuthorTime'),
                 'replays': []}
        for rp in res['replays']:
            try:
                r = replay_mod.load(rp)
            except Exception as e:
                entry['replays'].append({'file': rp.name, 'error': repr(e)[:200]})
                continue
            entry['replays'].append({'file': rp.name, 'time_ms': r.race_time_ms, 'respawns': r.respawns,
                                     'analog': r.uses_analog_steer, 'version': r.game_version,
                                     'uid_ok': r.map_uid == res['uid'], 'player': r.login})
        man['maps'][key] = entry
        MANIFEST.write_text(json.dumps(man, indent=1), encoding='utf-8')
        log(f"[{have + 1}/{n_maps}] {tid} {entry['name']!r}: {len(entry['replays'])} replays"
            f"{' (holdout)' if entry['holdout'] else ''}")
    MANIFEST.write_text(json.dumps(man, indent=1), encoding='utf-8')
    return man
