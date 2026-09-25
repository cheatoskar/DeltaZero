"""Check the no-line progress measure (src/tmdriver/course.py) on real TMX runs, no game needed.

    python scripts/course_check.py FOLDER [--fetch 414041,10036840,...]

FOLDER holds one sub-folder per map: map.Gbx plus its *.Replay.Gbx. --fetch downloads maps and
their 2 fastest replays from TMX into it first. For every replay without respawns it prints the
share of ghost samples on track cells, metres per new cell, and the longest time without a new
cell (the stall rule in improve.py must tolerate it, except while airborne).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tmdriver import course as C  # noqa: E402
from tmdriver import replay as R  # noqa: E402
from tmdriver import tmx  # noqa: E402
from tmdriver.replaybuild import map_blocks  # noqa: E402


def fetch(folder: Path, ids):
    for tid in ids:
        md = folder / str(tid)
        md.mkdir(parents=True, exist_ok=True)
        if not (md / 'map.Gbx').exists():
            (md / 'map.Gbx').write_bytes(tmx.get(f'{tmx.BASE}/trackgbx/{tid}'))
        for r in tmx.replay_list(tid, 5)[:2]:
            p = md / f"{r['ReplayId']}.Replay.Gbx"
            if not p.exists():
                p.write_bytes(tmx.get(f"{tmx.BASE}/recordgbx/{r['ReplayId']}"))
        print(f'fetched {tid}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--fetch', default='', help='comma-separated TMX ids to download first')
    args = ap.parse_args()
    folder = Path(args.folder)
    if args.fetch:
        fetch(folder, [int(x) for x in args.fetch.split(',')])
    rows = []
    for md in sorted(p for p in folder.iterdir() if (p / 'map.Gbx').exists()):
        uid, names, xyz, dirs = map_blocks(md / 'map.Gbx')
        track = C.track_cells({'name': n, 'x': x, 'y': y, 'z': z} for n, (x, y, z) in zip(names, xyz))
        for rp in sorted(md.glob('*.Replay.Gbx')):
            r = R.load(rp)
            if r.map_uid != uid or r.respawns:
                continue
            cells = [C.cell_of(p) for p in r.ghost_pos]
            dt = float(np.diff(r.ghost_t)[0]) / 1000.0
            seen, last, gaps = set(), 0, []
            for i, c in enumerate(cells):
                if c in track and c not in seen:
                    seen.add(c)
                    gaps.append((i - last) * dt)
                    last = i
            gaps.append((len(cells) - 1 - last) * dt)
            length = float(np.sum(np.linalg.norm(np.diff(r.ghost_pos, axis=0), axis=1)))
            row = {'map': md.name, 'time_s': r.race_time_ms / 1000, 'on_track': float(np.mean([c in track for c in cells])),
                   'cells': len(seen), 'm_per_cell': length / max(len(seen), 1), 'max_gap_s': max(gaps)}
            rows.append(row)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    if rows:
        g = np.array([r['max_gap_s'] for r in rows])
        print(f"{len(rows)} runs on {len({r['map'] for r in rows})} maps: on track {np.mean([r['on_track'] for r in rows]):.1%} "
              f"(min {min(r['on_track'] for r in rows):.1%}), median {np.median([r['m_per_cell'] for r in rows]):.1f} m "
              f"per new cell, longest gap <= 2 s: {np.sum(g <= 2)}, <= 5 s: {np.sum(g <= 5)}, max {g.max():.1f} s")


if __name__ == '__main__':
    main()
