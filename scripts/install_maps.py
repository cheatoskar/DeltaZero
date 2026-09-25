"""Copy the bulk-downloaded maps (data/maps/<id>.Challenge.Gbx) into the game's
Tracks/Challenges/TMDriver folder under the names the bulk list expects.

    python scripts/install_maps.py [TRACKS_DIR]

Default TRACKS_DIR: Documents/TmForever/Tracks/Challenges/TMDriver on Windows, the Wine
prefix on the server.
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if len(sys.argv) > 1:
    dst = Path(sys.argv[1])
elif os.name == 'nt':
    from tmdriver.tmx import tracks_dir
    dst = tracks_dir()
else:
    dst = Path('/root/.wine/drive_c/users/root/Documents/TmForever/Tracks/Challenges/TMDriver')
dst.mkdir(parents=True, exist_ok=True)
n = missing = 0
for line in (ROOT / 'data' / 'bulk' / 'maps.jsonl').read_text(encoding='utf-8').splitlines():
    m = json.loads(line)
    if not (m.get('ok') and m.get('map_file')):
        continue
    src = ROOT / 'data' / 'maps' / f"{m['track_id']}.Challenge.Gbx"
    if not src.exists():
        missing += 1
        continue
    if not (dst / m['map_file']).exists():
        shutil.copyfile(src, dst / m['map_file'])
    n += 1
print(f'{n} maps in {dst} ({missing} missing)')
