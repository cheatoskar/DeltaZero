"""Small, polite TMX (tmnf.exchange) client: map files and replays.

Verified endpoints (2026-09-24, no auth needed):
    /api/tracks?id={id}&fields=...     track metadata incl. UId
    /api/replays?trackId={id}&count=N  replay list
    /trackgbx/{TrackId}                map file
    /recordgbx/{ReplayId}              replay file
Requests are serialised with a minimum gap, and identify the project in the User-Agent.
Bulk downloading (all 3.5M replays) is a separate decision; see docs/DATA.md.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

BASE = 'https://tmnf.exchange'
UA = 'TMDriverAI-research/0.1 (non-commercial; contact: cheatoskar)'
MIN_GAP_S = 1.0
_last = [0.0]
_lock = threading.Lock()   # the gap is global across threads (bulk.py runs several)


def get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        with _lock:
            wait = _last[0] + MIN_GAP_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError('unreachable')


def track_info(track_id: int) -> Dict:
    fields = 'TrackId,TrackName,UId,AuthorTime,Tags'
    d = json.loads(get(f'{BASE}/api/tracks?id={track_id}&fields={fields}'))
    if not d.get('Results'):
        raise ValueError(f'TMX track {track_id} not found')
    return d['Results'][0]


def track_id_by_uid(uid: str):
    """TMX id of a map uid, or None if TMX does not have the map. `/api/tracks?uid=` checked
    2026-09-25 (414041 <-> RA0Km2Yx1v0wEGZCDIPjv0wmkIh); the answer's UId is verified."""
    d = json.loads(get(f'{BASE}/api/tracks?fields=TrackId,UId&uid={uid}'))
    for r in d.get('Results', []):
        if r.get('UId') == uid:
            return int(r['TrackId'])
    return None


def replay_list(track_id: int, count: int = 25) -> List[Dict]:
    d = json.loads(get(f'{BASE}/api/replays?trackId={track_id}&count={count}'
                       f'&fields=ReplayId,ReplayTime,Position,User.Name'))
    return sorted(d.get('Results', []), key=lambda r: r['ReplayTime'])


def safe_name(name: str) -> str:
    name = re.sub(r'\$[0-9a-fA-F]{3}|\$[a-zA-Z$]', '', name)   # strip TM formatting codes
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', name).strip('_')[:40] or 'map'


_GAME_DIR = []


def game_dir(log=print) -> Path:
    """The game's user folder (holds Tracks/Challenges; the `map` command is relative to it).
    TMDRIVER_GAME_DIR wins (the plugin passes the TMI variable tmdriver_game_folder). Otherwise
    the candidate whose Profiles folder changed last: the laptop's TMNF uses
    Documents/TmForever, the home PC's uses Documents/TrackMania (2026-09-25: its profile was
    written at the game start, the TmForever one not since August)."""
    env = os.environ.get('TMDRIVER_GAME_DIR', '').strip()
    if env:
        return Path(env)
    if _GAME_DIR:
        return _GAME_DIR[0]
    home = Path(os.path.expanduser('~'))
    cands = [home / docs / name for docs in ('Documents', 'OneDrive/Dokumente', 'OneDrive/Documents')
             for name in ('TmForever', 'TrackMania')]
    cands = [c for c in cands if (c / 'Tracks').is_dir()]
    if not cands:
        pick = home / 'Documents' / 'TmForever'
    else:
        def used(c):
            p = c / 'Profiles'
            return p.stat().st_mtime if p.exists() else c.stat().st_mtime
        pick = max(cands, key=used)
        if len(cands) > 1:
            log(f'game folder: {pick} (most recently used of {len(cands)}; to force another: '
                f'`set tmdriver_game_folder <path>` in the TMI console)')
    _GAME_DIR.append(pick)
    return pick


def tracks_dir() -> Path:
    """Where downloaded maps go: <game folder>/Tracks/Challenges/TMDriver."""
    return game_dir() / 'Tracks' / 'Challenges' / 'TMDriver'


def fetch(track_id: int, n_replays: int, tmx_root: Path, safe_uid, log=print, reps=None) -> Dict:
    info = track_info(track_id)
    uid = info['UId']
    name = info['TrackName'].replace('.Challenge', '')
    mdir = tracks_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / f'{safe_name(name)}_{track_id}.Challenge.Gbx'
    if not mpath.exists():
        mpath.write_bytes(get(f'{BASE}/trackgbx/{track_id}'))
    rdir = tmx_root / safe_uid(uid)
    rdir.mkdir(parents=True, exist_ok=True)
    got = []
    for r in (reps if reps is not None else replay_list(track_id))[:n_replays]:
        rp = rdir / f"{r['ReplayId']}.Replay.Gbx"
        if not rp.exists():
            rp.write_bytes(get(f"{BASE}/recordgbx/{r['ReplayId']}"))
        got.append((rp, r['ReplayTime']))
    (rdir / 'track.json').write_text(json.dumps(info, indent=1))
    log(f'{track_id} {name!r}: map -> {mpath}')
    for rp, t in got:
        log(f'    replay {rp.name} ({t / 1000:.2f} s)')
    return {'uid': uid, 'map': mpath, 'replays': [g[0] for g in got], 'info': info}
