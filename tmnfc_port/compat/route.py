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


def main():
    import spawn as spawn_mod
    challenge, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.write_bytes(minimal_route(challenge, spawn_mod.spawn(str(challenge))))
    print(f'{out}: {out.stat().st_size} bytes')


if __name__ == '__main__':
    main()
