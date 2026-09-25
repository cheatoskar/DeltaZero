import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Overridable so tests can run the whole pipeline in a temporary directory.
DATA = Path(os.environ.get('TMDRIVER_DATA', ROOT / 'data'))
RUNS = Path(os.environ.get('TMDRIVER_RUNS', ROOT / 'runs'))
LINES = DATA / 'lines'          # recorded runs, one per map uid
LIVE_MAPS = DATA / 'live_maps'  # block lists sent by the plugin
CALIBRATION = DATA / 'calibration.json'
# The driver "AI drive (live)" uses: TMDRIVER_CKPT (serve --model) if set, else the replay
# fine-tune (runs/ghost_ft, sees the car's orientation), else the pretrained ghost-feature
# model (runs/ghost), else the M1 model.
DRIVER_CKPT = Path(os.environ.get('TMDRIVER_CKPT') or next(
    (p for p in (RUNS / 'ghost_ft' / 'best.pt', RUNS / 'ghost' / 'best.pt') if p.exists()),
    RUNS / 'm1' / 'driver.pt'))


def torch_device() -> str:
    """TMDRIVER_DEVICE (tmdriver.py --device) or 'auto': CUDA when available, else CPU."""
    d = os.environ.get('TMDRIVER_DEVICE', 'auto')
    if d == 'auto':
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return d
POOL = DATA / 'pool' / 'maps.json'   # map pool from the TMX search (tmx-pool), for resim and RL
TMX = DATA / 'tmx'              # TMX replays by map uid, for the self test


def safe(uid: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in uid) or 'unknown'
