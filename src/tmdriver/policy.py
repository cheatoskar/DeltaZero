"""Live inference: Step -> the same features as in training -> action."""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import features as F
from . import protocol as P
from . import replay as replay_mod
from .line import RefLine
from .model import DriverNet
from .paths import DRIVER_CKPT, LINES, TMX, safe
from .tasks import steps_to_arrays

CKPT = DRIVER_CKPT


def track_id_for(uid: str) -> Optional[int]:
    """TMX id of a map uid, from the M1 manifest or the bulk download list."""
    from .collect import MANIFEST
    if MANIFEST.exists():
        for m in json.loads(MANIFEST.read_text(encoding='utf-8'))['maps'].values():
            if m.get('ok') and m['uid'] == uid:
                return int(m['track_id'])
    from .bulk import MAPS_LOG
    if MAPS_LOG.exists():
        for line in MAPS_LOG.read_text(encoding='utf-8').splitlines():
            if uid in line:
                m = json.loads(line)
                if m.get('uid') == uid:
                    return int(m['track_id'])
    return None


def reference_positions(uid: str, track_id: Optional[int] = None) -> Optional[tuple]:
    """Positions of the best known run on a map (for the ghost-feature driver's line):
    fastest exact re-simulated run > fastest TMX replay ghost > cheatoskar's recording."""
    from .resim import RESIM
    if track_id is not None and (RESIM / str(track_id)).exists():
        best = None
        for f in (RESIM / str(track_id)).glob('*.npz'):
            m = json.loads(str(np.load(f)['meta']))
            if m.get('exact') and (best is None or m['finished_at'] < best[0]):
                best = (m['finished_at'], f)
        if best:
            return np.load(best[1])['pos'].astype(np.float64), f'resim {best[1].stem} ({best[0] / 1000:.2f}s)'
    reps = []
    for folder in (TMX / safe(uid), TMX / str(track_id)):
        for f in folder.glob('*.Replay.Gbx') if folder.exists() else []:
            try:
                r = replay_mod.load(f)
            except Exception:
                continue
            if r.map_uid == uid and r.respawns == 0:
                reps.append(r)
    if reps:
        r = min(reps, key=lambda r: r.race_time_ms)
        return r.ghost_pos, f'TMX ghost ({r.race_time_ms / 1000:.2f}s)'
    rec = LINES / f'{safe(uid)}.npz'
    if rec.exists():
        return np.load(rec)['pos'].astype(np.float64), 'eigene Aufnahme'
    return None


def reference_line(uid: str, track_id: Optional[int] = None) -> Optional[tuple]:
    """Best available path for a map, and where it came from:
    fastest exact re-simulated run > fastest TMX replay ghost > cheatoskar's recording."""
    from .resim import RESIM
    if track_id is not None and (RESIM / str(track_id)).exists():
        best = None
        for f in (RESIM / str(track_id)).glob('*.npz'):
            m = json.loads(str(np.load(f)['meta']))
            if m.get('exact') and (best is None or m['finished_at'] < best[0]):
                best = (m['finished_at'], f)
        if best:
            a = np.load(best[1])
            return RefLine(a['pos'], np.linalg.norm(a['vel'], axis=1)), f'resim {best[1].stem} ({best[0] / 1000:.2f}s)'
    folder = TMX / safe(uid)
    reps = []
    for f in folder.glob('*.Replay.Gbx') if folder.exists() else []:
        try:
            reps.append(replay_mod.load(f))
        except Exception:
            pass
    if reps:
        r = min(reps, key=lambda r: r.race_time_ms)
        return RefLine(r.ghost_pos, np.zeros(len(r.ghost_pos))), f'TMX ghost ({r.race_time_ms / 1000:.2f}s)'
    rec = LINES / f'{safe(uid)}.npz'
    if rec.exists():
        return RefLine.load(rec), 'eigene Aufnahme'
    return None


class Policy:
    def __init__(self, ckpt: Path = CKPT, threads: int = 4):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        if ck['features_version'] != F.FEATURES_VERSION:
            raise RuntimeError(f"checkpoint features v{ck['features_version']}, code v{F.FEATURES_VERSION}")
        torch.set_num_threads(threads)
        self.model = DriverNet(**ck['cfg'])
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        meta = ck['meta']
        self.vocab = meta['vocab']
        self.right_sign = int(meta['steer']['right_sign'])
        self.bdir = meta['block_dir']
        self.val = ck.get('val', {})
        self.line: Optional[RefLine] = None
        self.idx = 0

    def reset(self, blocks: List[dict], line: RefLine):
        self.geom = F.MapGeometry(blocks, self.vocab, self.bdir['sign'], self.bdir['offset_deg'])
        self.line = line
        self.idx = 0

    @property
    def progress_m(self) -> float:
        return self.idx * self.line.spacing

    @torch.no_grad()
    def act(self, step: P.Step, pace: float = 0.0):
        a = steps_to_arrays([step])
        pos = a['pos'].astype(np.float64)
        self.idx = self.line.locate(pos[0], self.idx)
        state = F.state_features(a, pace)
        route = F.route_features(self.line, pos, a['rot'], np.array([self.idx]))
        bn, bw, bf, bm = self.geom.features(pos, a['rot'])
        out = self.model(torch.from_numpy(state), torch.from_numpy(route),
                         torch.from_numpy(bn), torch.from_numpy(bw),
                         torch.from_numpy(bf), torch.from_numpy(bm))
        b = int(out['steer'][0].argmax())
        bits = 0
        if out['gas'][0] > 0:
            bits |= P.UP
        if out['brake'][0] > 0:
            bits |= P.DOWN
        steer = 0
        if b == 0 or b == F.STEER_BINS - 1:
            unit = -1 if b == 0 else 1          # full lock: press the key, as most players do
            right = (unit == self.right_sign)
            bits |= P.RIGHT if right else P.LEFT
        else:
            steer = int(F.bin_to_steer(b) * P.STEER_MAX)
            if steer != 0:
                bits |= P.STEER_ANALOG
        return steer, 0, bits
