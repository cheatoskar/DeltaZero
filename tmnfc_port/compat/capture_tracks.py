"""Measure the offline track builder against the game: for each map, dump the track from the running
game (TMNFTrackDump.dll injected into a fresh TMNFC-profile game) and compare it with
tools/build_track (canonical payload).

    python compat/capture_tracks.py MAP.Challenge.Gbx [...]

Maps are copied to Tracks\\Challenges\\TMDriver\\tmnfc\\ so the game can load them.
Results are appended to out/track_check.jsonl.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DZ = Path.home() / 'Downloads' / 'DeltaZero'
CH = Path.home() / 'Documents' / 'TrackMania' / 'Tracks' / 'Challenges'
STAGE = CH / 'TMDriver' / 'tmnfc'
CFG = ROOT / 'build' / 'win32' / 'TMNFTracer.cfg'
DLL = ROOT / 'build' / 'win32' / 'TMNFTrackDump.dll'
INJECT = ROOT / 'build' / 'win32' / 'inject.exe'
PACKS = 'C:/Program Files (x86)/TmNationsForever/Packs'
OUT = ROOT / 'out' / 'track_check.jsonl'
LOADER = Path.home() / 'AppData' / 'Local' / 'TMLoader' / 'TMLoader.exe'

sys.path.insert(0, str(ROOT / 'tools' / 'build_track'))


def ps(cmd: str) -> str:
    return subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True).stdout.strip()


def restart_game() -> int:
    ps('Stop-Process -Name TmForever -Force -ErrorAction SilentlyContinue')
    time.sleep(3)
    subprocess.Popen([str(LOADER), 'run', 'TmForever', 'TMNFC', '/configstring=set custom_port 8490'])
    t0 = time.time()
    while time.time() - t0 < 120:
        if ps('if (Get-NetTCPConnection -State Listen -LocalPort 8490 -ErrorAction SilentlyContinue) { "y" }') == 'y':
            return int(ps('(Get-Process TmForever).Id'))
        time.sleep(2)
    raise RuntimeError('game did not start')


def capture(challenge: Path, out: Path) -> None:
    sha = hashlib.sha256(challenge.read_bytes()).hexdigest()
    STAGE.mkdir(parents=True, exist_ok=True)
    stem = 'm_' + sha[:12]
    shutil.copyfile(challenge, STAGE / f'{stem}.Challenge.Gbx')
    CFG.write_text(f'TMNF_TRACK_PATH={out}\nTMNF_TRACK_SHA256={sha}\n', encoding='utf-8')
    if out.exists():
        out.unlink()
    pid = restart_game()
    subprocess.run([str(INJECT), str(DLL)], check=True, capture_output=True)
    focus = subprocess.Popen([sys.executable, '-c', 'import sys,time; time.sleep(4); sys.path.insert(0, "src"); '
                              f'from tmdriver.session import focus_game_window; focus_game_window({pid})'], cwd=DZ)
    r = subprocess.run([sys.executable, 'determinism_test.py', '--port', '8490', '--map', f'TMDriver/tmnfc/{stem}',
                        '--ticks', '10', '--output-dir', str(ROOT / 'cap' / 'track-probe')],
                       cwd=ROOT / 'oracle', capture_output=True, text=True, timeout=300)
    focus.wait()
    if not out.exists():
        raise RuntimeError(f'no in-game snapshot ({(r.stderr or r.stdout).strip()[-200:]})')


def main():
    import trkfile                                   # noqa: E402
    from assets import Assets, cache_dir_for        # noqa: E402
    from build_track import build                   # noqa: E402
    assets = Assets(PACKS, cache_dir_for(PACKS))
    for arg in sys.argv[1:]:
        challenge = Path(arg)
        row = {'map': challenge.name}
        try:
            game = ROOT / 'cap' / 'tracks' / (hashlib.sha256(challenge.read_bytes()).hexdigest()[:16] + '.tmnftrack')
            game.parent.mkdir(parents=True, exist_ok=True)
            if not game.exists():
                capture(challenge, game)
            built = build(assets, str(challenge))
            oracle = trkfile.load(str(game))
            row['raw_identical'] = (trkfile.build_image(built)[trkfile.HEADER_SIZE:]
                                    == trkfile.build_image(oracle)[trkfile.HEADER_SIZE:])
            lines = trkfile.compare(trkfile.canonicalize(built), trkfile.canonicalize(oracle), limit=3)
            row['canonical_identical'] = not lines
            row['diff'] = [l for l in lines if not l.startswith('  ')][:6]
        except Exception as e:                       # noqa: BLE001
            row['error'] = repr(e)[:300]
        print(json.dumps(row))
        sys.stdout.flush()
        with open(OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
