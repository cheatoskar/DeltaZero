"""Pack DeltaZero + TMNF-C into ONE zip for moving to another PC (e.g. a 10 GB USB stick).

    python scripts/portable_zip.py [out.zip] [--replays-per-map 2] [--no-trainpack]
    python scripts/portable_zip.py --full [out.zip]     everything but the caches (~33 GB, e.g. Google Drive)

Left out, because they are rebuilt on first use: data/tmnfc (built tracks), data/ghost (shards),
data/trainpack (the unpacked zip), the exact runs' npz files in data/resim_c/<map>/ (the trainpack
holds night 1; everything is re-simulated again after the TMNF-C fixes), TMNF-C's per-map challenge
JSON dumps (GbxDump regenerates them; needs the .NET 10 runtime). Replays (data/tmx, ~20 GB): only
those of maps without any exact run (for the TMNF-C work, the earliest-deviating first) and all
replays of the test maps; the rest can be fetched from TMX again.
The zip stores without compression (replays and checkpoints are already compressed).
"""
import argparse
import collections
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
TMNFC = Path(os.environ.get('TMDRIVER_TMNFC', Path.home() / 'Downloads' / 'TMNF-C'))
TEST_MAPS = {10036840, 10030774, 2481743, 414041, 175621, 537053, 3131260}
SKIP_DIRS = {'__pycache__', '.git'}


def walk(base: Path, skip):
    for dirpath, dirnames, filenames in os.walk(base):
        d = Path(dirpath)
        dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS and not skip(d / n)]
        for f in filenames:
            p = d / f
            if not skip(p):
                yield p


def replays(per_map: int):
    from tmdriver.paths import TMX, safe
    by = collections.defaultdict(list)
    for line in (ROOT / 'data' / 'resim_c' / 'index.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if 'uid' in r:
            by[r['track_id']].append(r)
    out = []
    for t, rs in by.items():
        if t in TEST_MAPS:
            pick = rs
        elif any(r.get('exact') for r in rs):
            continue
        else:
            pick = sorted(rs, key=lambda r: r.get('first_off_ms') or 1e9)[:per_map]
        out += [TMX / safe(r['uid']) / f"{r['replay']}.Replay.Gbx" for r in pick]
    return [p for p in out if p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default=str(ROOT.parent / 'DeltaZero_portable.zip'))
    ap.add_argument('--replays-per-map', type=int, default=2)
    ap.add_argument('--no-trainpack', action='store_true')
    ap.add_argument('--full', action='store_true', help='all replays and exact runs (only the caches left out)')
    a = ap.parse_args()
    data = ROOT / 'data'
    left_out = {data / 'tmnfc', data / 'ghost', data / 'trainpack'} | (set() if a.full else {data / 'tmx'})
    if a.no_trainpack:
        left_out.add(data / 'trainpack.zip')

    def skip_dz(p: Path):
        if p in left_out:
            return True
        # data/resim_c/<track id>/ holds the npz files
        return not a.full and p.parent == data / 'resim_c' and p.is_dir()

    cache_json = TMNFC / 'third_party' / 'build_track_cache' / 'TmNationsForever' / 'json' / 'challenges'
    files = [(p, Path('DeltaZero') / p.relative_to(ROOT)) for p in walk(ROOT, skip_dz)]
    if not a.full:
        files += [(p, Path('DeltaZero') / p.relative_to(ROOT)) for p in replays(a.replays_per_map)]
    files += [(p, Path('TMNF-C') / p.relative_to(TMNFC)) for p in walk(TMNFC, lambda p: p == cache_json)]
    total = sum(p.stat().st_size for p, _ in files)
    print(f'{len(files):,} files, {total / 1e9:.2f} GB -> {a.out}')
    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, (p, arc) in enumerate(files):
            z.write(p, arc.as_posix())
            if i % 20000 == 0:
                print(f'  {i:,}/{len(files):,}', flush=True)
    print(f'done: {Path(a.out).stat().st_size / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
