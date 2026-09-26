"""A minimal TMNF-C route file written without the game: the map's start (compat/spawn.py), no
checkpoints and one unreachable finish. TmnfVecEnv needs a route; DeltaZero tracks checkpoints
and the finish itself, so this is enough for raw stepping, reset, capture and restore.

    python compat/route.py MAP.Challenge.Gbx OUT.tmnfroute
"""
import hashlib
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXE_SHA256 = bytes.fromhex('3847cf9f20bfc63914450060ed528c12104f743d96ad23d6e76abd178de8c84f')
HEADER_SIZE = 0xE0
META, START, TRIGGER, REFERENCE = 0x40, 0x114, 0x90, 0x18
WAYPOINT_START, WAYPOINT_FINISH = 0, 1


def align8(v: int) -> int:
    return (v + 7) & ~7


def quat_from_rot(m) -> tuple:
    """(x, y, z, w) of a row-major rotation (only stored, never used by the physics here)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


def minimal_route(challenge: Path, spawn: tuple) -> bytes:
    iso = struct.pack('<12f', *spawn)
    rot, pos = spawn[:9], spawn[9:]
    initial = (struct.pack('<4f', *quat_from_rot(rot)) + struct.pack('<9f', *rot) + struct.pack('<3f', *pos)
               + bytes(12 * 5) + struct.pack('<9f', 0, 0, 0, 0, 0, 0, 0, 0, 0) + bytes(12))
    assert len(initial) == 0xAC
    start = iso + initial + struct.pack('<II', 0, WAYPOINT_START) + iso
    assert len(start) == START
    far = struct.pack('<3f3f', pos[0], pos[1] - 5000.0, pos[2], 1.0, 1.0, 1.0)   # unreachable finish
    ident = struct.pack('<12f', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    finish = struct.pack('<IIII', 0, 0, WAYPOINT_FINISH, 0x1E8CE) + far + ident + iso + struct.pack('<II', 1, 0)
    assert len(finish) == TRIGGER
    fwd = (rot[2], rot[5], rot[8])             # any second point 1 m away
    p1 = (pos[0] + fwd[0], pos[1] + fwd[1], pos[2] + fwd[2])
    length = math.dist(pos, p1)
    reference = struct.pack('<5fI', *pos, 0.0, 16.0, 0) + struct.pack('<5fI', *p1, length, 16.0, 0)
    payloads = [None, start, b'', finish, reference]
    strides = [META, START, TRIGGER, TRIGGER, REFERENCE]
    counts = [1, 1, 0, 1, 2]
    offsets, cursor = [], HEADER_SIZE
    for i in range(5):
        cursor = align8(cursor)
        offsets.append(cursor)
        cursor += counts[i] * strides[i]
    meta = struct.pack('<8I4Q', 1, 0, 1, 2, 1, 1, 0, 2, offsets[1], offsets[2], offsets[3], offsets[4])
    payloads[0] = meta
    image = bytearray(cursor)
    for off, data in zip(offsets, payloads):
        image[off:off + len(data)] = data
    struct.pack_into('<8sIIIIQ', image, 0, b'TMNFROU1', 3, 0x12345678, HEADER_SIZE, 5, len(image))
    image[32:64] = EXE_SHA256
    image[64:96] = hashlib.sha256(challenge.read_bytes()).digest()
    for i in range(5):
        struct.pack_into('<QII', image, 96 + i * 16, offsets[i], counts[i], strides[i])
    image[176:208] = hashlib.sha256(bytes(image[HEADER_SIZE:])).digest()
    return bytes(image)


WAYPOINT_CHECKPOINT, WAYPOINT_START_FINISH = 2, 4


def race_triggers(challenge: Path):
    """(start, checkpoints, finishes) of the map, each trigger a dict: block index, spawn iso,
    root box (centre, half extent) in the block's frame, world transform, tree flags,
    no_respawn. The trigger is the mobil the block's mobil links to (objectLinks), placed at
    relativeLocation x block location; its box is the root tree's surface box."""
    import base64
    import fp
    import spawn as spawn_mod
    from build_track import challenge_json
    from scene import get_mobil
    ch, _ = spawn_mod.start_blocks(str(challenge))
    assets = spawn_mod.assets()
    start = cps = None
    cps, fins = [], []
    for b in ch.blocks:
        wp = b.info.get('wayPointType')
        if wp not in ('Start', 'Finish', 'Checkpoint', 'StartFinish'):
            continue
        loc = ch.block_mobil_loc(b)
        key = 'spawnLocGround' if b.ground else 'spawnLocAir'
        raw = b.info.get(key) or b.info.get('spawnLocAir') or b.info.get('spawnLocGround')
        spawn = fp.iso_set_mult(fp.iso_from_bytes(base64.b64decode(raw)), loc) if raw else loc
        if wp == 'Start':
            start = {'block': b.index, 'spawn': spawn}
            continue
        mob = get_mobil(b.info, b.ground, b.variant, b.sub_variant) or \
            get_mobil(b.info, not b.ground, b.variant, b.sub_variant)
        links = [l for l in (mob['mobil'].get('objectLinks') or []) if l.get('mobil')] if mob else []
        if not links:
            raise ValueError(f'{b}: no trigger linked')
        link = links[0]
        rel = fp.iso_from_bytes(base64.b64decode(link['relativeLocation'])) if link.get('relativeLocation') \
            else fp.IDENTITY
        tree = assets.mobil_tree(link['mobil'], f'{b} trigger', link.get('mobilFile'))
        if tree.surface is None:
            raise ValueError(f'{b}: trigger tree has no surface')
        box = tree.surface.box
        if tree.flags & 4 and tree.loc is not None:
            box = fp.box_set_mult(box, tree.loc)
        trig = {'block': b.index, 'spawn': spawn, 'box': box, 'transform': fp.iso_set_mult(rel, loc),
                'flags': tree.flags, 'no_respawn': 1 if b.info.get('noRespawn') else 0,
                'start_finish': wp == 'StartFinish'}
        (cps if wp == 'Checkpoint' else fins).append(trig)
    return start, cps, fins


def full_route(challenge: Path) -> bytes:
    """A route with the map's real checkpoint and finish triggers (one lap). Multilap maps
    (a StartFinish block) are not supported yet: ValueError."""
    start, cps, fins = race_triggers(challenge)
    if start is None or not fins:
        raise ValueError('no start or no finish')
    if any(f['start_finish'] for f in fins):
        raise ValueError('multilap map')
    spawn = start['spawn']
    rot, pos = spawn[:9], spawn[9:]
    iso = struct.pack('<12f', *spawn)
    initial = (struct.pack('<4f', *quat_from_rot(rot)) + struct.pack('<9f', *rot) + struct.pack('<3f', *pos)
               + bytes(12 * 5) + bytes(36) + bytes(12))
    start_rec = iso + initial + struct.pack('<II', start['block'], WAYPOINT_START) + iso

    def trig(t, index, waypoint):
        return (struct.pack('<IIII', index, t['block'], waypoint, t['flags']) + struct.pack('<6f', *t['box'])
                + struct.pack('<12f', *t['transform']) + struct.pack('<12f', *t['spawn'])
                + struct.pack('<II', t['no_respawn'], 0))

    cp_recs = b''.join(trig(t, i, WAYPOINT_CHECKPOINT) for i, t in enumerate(cps))
    fin_recs = b''.join(trig(t, i, WAYPOINT_FINISH) for i, t in enumerate(fins))

    def centre(t):
        c, m = t['box'][:3], t['transform']
        return (c[0] * m[0] + c[1] * m[1] + c[2] * m[2] + m[9], c[0] * m[3] + c[1] * m[4] + c[2] * m[5] + m[10],
                c[0] * m[6] + c[1] * m[7] + c[2] * m[8] + m[11])

    # A valid reference line (the environment's progress measure, unused by DeltaZero): start,
    # every checkpoint in nearest-next order, the nearest finish; leg i ends at the i-th checkpoint.
    pts, left, cur = [tuple(pos)], list(range(len(cps))), tuple(pos)
    while left:
        k = min(left, key=lambda k: math.dist(cur, centre(cps[k])))
        left.remove(k)
        cur = centre(cps[k])
        pts.append(cur)
    pts.append(centre(min(fins, key=lambda f: math.dist(cur, centre(f)))))
    ref, arc = [], 0.0
    for i, p in enumerate(pts):
        if i:
            d = math.dist(pts[i - 1], p)
            if d <= 0.0:
                continue
            arc += d
        ref.append(struct.pack('<5fI', *p, arc, 16.0, max(0, min(i, len(cps)))))
    ncp = len(cps)
    payloads = [None, start_rec, cp_recs, fin_recs, b''.join(ref)]
    strides = [META, START, TRIGGER, TRIGGER, REFERENCE]
    counts = [1, 1, ncp, len(fins), len(ref)]
    offsets, cursor = [], HEADER_SIZE
    for i in range(5):
        cursor = align8(cursor)
        offsets.append(cursor)
        cursor += counts[i] * strides[i]
    payloads[0] = struct.pack('<8I4Q', 1, ncp, len(fins), ncp + 2, ncp + 1, ncp + 1, 0, len(ref),
                              offsets[1], offsets[2], offsets[3], offsets[4])
    image = bytearray(cursor)
    for off, data in zip(offsets, payloads):
        image[off:off + len(data)] = data
    struct.pack_into('<8sIIIIQ', image, 0, b'TMNFROU1', 3, 0x12345678, HEADER_SIZE, 5, len(image))
    image[32:64] = EXE_SHA256
    image[64:96] = hashlib.sha256(challenge.read_bytes()).digest()
    for i in range(5):
        struct.pack_into('<QII', image, 96 + i * 16, offsets[i], counts[i], strides[i])
    image[176:208] = hashlib.sha256(bytes(image[HEADER_SIZE:])).digest()
    return bytes(image)


def main():
    import spawn as spawn_mod
    challenge, out = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        data = full_route(challenge)
        kind = 'full'
    except ValueError as e:
        data = minimal_route(challenge, spawn_mod.spawn(str(challenge)))
        kind = f'minimal ({e})'
    out.write_bytes(data)
    print(f'{out}: {out.stat().st_size} bytes, {kind}')


if __name__ == '__main__':
    main()
