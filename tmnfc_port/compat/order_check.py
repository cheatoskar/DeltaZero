"""Offline track builder vs tracks dumped from the game: how many maps come out identical?

    python compat/order_check.py      (every cap/tracks/<sha16>.tmnftrack with a known map file)
"""
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'build_track'))

import trkfile  # noqa: E402
from assets import Assets, cache_dir_for  # noqa: E402
from build_track import build  # noqa: E402

PACKS = 'C:/Program Files (x86)/TmNationsForever/Packs'
SEARCH = [Path.home() / 'Documents/TrackMania/Tracks/Challenges/TMDriver',
          Path.home() / 'Downloads/DeltaZero/data/maps']


def maps_by_sha16():
    out = {}
    for d in SEARCH:
        for f in d.rglob('*.Challenge.Gbx'):
            out.setdefault(hashlib.sha256(f.read_bytes()).hexdigest()[:16], f)
    return out


def main():
    assets = Assets(PACKS, cache_dir_for(PACKS))
    maps = maps_by_sha16()
    same = total = 0
    for dump in sorted((ROOT / 'cap' / 'tracks').glob('*.tmnftrack')):
        f = maps.get(dump.stem)
        if f is None:
            continue
        total += 1
        try:
            built = build(assets, str(f), None, os.environ.get('TMNF_QUALITY', 'low'))
        except Exception as e:                  # noqa: BLE001
            print(f'{f.name}: build error {e}')
            continue
        oracle = trkfile.load(str(dump))
        lines = trkfile.compare(trkfile.canonicalize(built), trkfile.canonicalize(oracle), limit=100000)
        rows = sum(1 for l in lines if l.startswith('  [') and 'built' in l)
        same += not lines
        print(f'{f.name[:40]:40s} {"IDENTICAL" if not lines else f"{rows} differing rows"}')
    print(f'identical: {same}/{total}')


if __name__ == '__main__':
    main()
