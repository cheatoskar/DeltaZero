"""The race start pose of a challenge, computed from the map file (no game).

CGameCtnBlock::GetSpawnLoc for the start block: the block info's spawn location (ground or air
variant) placed by the block's mobil location. Checked bit for bit against in-game starts.

    python compat/spawn.py MAP.Challenge.Gbx [RUN.npz]
"""
import base64
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'build_track'))

import fp  # noqa: E402
from assets import Assets, cache_dir_for  # noqa: E402
from build_track import challenge_json  # noqa: E402
from scene import Challenge  # noqa: E402

PACKS = 'C:/Program Files (x86)/TmNationsForever/Packs'
START_TYPES = ('Start', 'StartFinish')
_ASSETS = None


def assets() -> Assets:
    global _ASSETS
    if _ASSETS is None:
        _ASSETS = Assets(PACKS, cache_dir_for(PACKS))
    return _ASSETS


def start_blocks(challenge_path: str):
    doc = challenge_json(assets(), challenge_path)
    ch = Challenge(assets(), doc)
    ch.init()
    return ch, [b for b in ch.blocks if b.info.get('wayPointType') in START_TYPES]


def spawn_candidates(challenge_path: str):
    """(name, iso) candidates for the start pose; the first is the game's rule."""
    ch, starts = start_blocks(challenge_path)
    out = []
    for b in starts:
        loc = ch.block_mobil_loc(b)
        key = 'spawnLocGround' if b.ground else 'spawnLocAir'
        for k in (key, 'spawnLocAir' if key == 'spawnLocGround' else 'spawnLocGround'):
            raw = b.info.get(k)
            if raw is None:
                continue
            sp = fp.iso_from_bytes(base64.b64decode(raw))
            out.append((f'{b.name}:{k}:spawn*block', fp.iso_set_mult(sp, loc)))
            out.append((f'{b.name}:{k}:block*spawn', fp.iso_set_mult(loc, sp)))
    return out


def spawn(challenge_path: str) -> tuple:
    c = spawn_candidates(challenge_path)
    if not c:
        raise ValueError('no start block')
    return c[0][1]


def main():
    import numpy as np
    path = sys.argv[1]
    cands = spawn_candidates(path)
    ref = None
    if len(sys.argv) > 2:
        z = np.load(sys.argv[2], allow_pickle=True)
        i0 = int(np.flatnonzero(z['t'] == 0)[0])
        ref = tuple(np.float32(v) for v in list(z['rot'][i0]) + list(z['pos'][i0]))
    for name, iso in cands:
        tag = ''
        if ref is not None:
            same = all(struct.pack('<f', a) == struct.pack('<f', b) for a, b in zip(iso, ref))
            err = max(abs(float(a) - float(b)) for a, b in zip(iso, ref))
            tag = f'  bit-exact={same} maxerr={err:.3g}'
        print(name, [round(float(v), 4) for v in iso], tag)


if __name__ == '__main__':
    main()
