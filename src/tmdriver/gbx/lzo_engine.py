"""
LZO Compression & Decompression Engine for GBX Files
Provides fast native LZO1X decompression and compression via ctypes.
"""

import os
import sys
import ctypes
from ctypes import CDLL, c_uint32, byref, c_char_p, POINTER, c_void_p
from pathlib import Path

_LIB = None
_IS_64 = sys.maxsize > 2**32

def _init_lzo():
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_dir = Path(__file__).parent / "lib"
    
    if os.name == 'nt':
        dll_name = "lzo1x_64.dll" if _IS_64 else "lzo1x_32.dll"
        dll_path = lib_dir / dll_name
        if dll_path.exists():
            try:
                _LIB = CDLL(str(dll_path))
                _LIB.lzo1x_decompress_safe.restype = c_uint32
                _LIB.lzo1x_decompress_safe.argtypes = [c_char_p, c_uint32, c_char_p, POINTER(c_uint32)]
                
                if hasattr(_LIB, 'lzo1x_999_compress'):
                    _LIB.lzo1x_999_compress.restype = c_uint32
                    _LIB.lzo1x_999_compress.argtypes = [c_char_p, c_uint32, c_char_p, POINTER(c_uint32), c_void_p]
                return _LIB
            except Exception as e:
                print(f"[Warning] Failed to load native LZO DLL ({dll_path}): {e}")

    # Fallback to python-lzo if installed
    try:
        import lzo
        return "python-lzo"
    except ImportError:
        pass

    # Last resort: the pure-Python decompressor below (slower; no compression)
    return "pure"


def _py_decompress(src: bytes, out_len: int) -> bytes:
    """LZO1X decompression in pure Python, following the reference lzo1x_decompress_safe.
    Used only when neither the DLL nor python-lzo is available (e.g. no python-lzo wheel for
    the Python version). Maps and replays are small, so the speed does not matter."""
    src = memoryview(src)
    n = len(src)
    out = bytearray()
    ip = 0

    def ext(base):
        nonlocal ip
        t = 0
        while src[ip] == 0:
            t += 255
            ip += 1
        t += base + src[ip]
        ip += 1
        return t

    def copy_match(dist, count):
        start = len(out) - dist
        if start < 0:
            raise ValueError('LZO: match before the start of the output')
        for k in range(count):              # byte by byte: matches may overlap the output end
            out.append(out[start + k])

    state = 'loop'
    t = 0
    if src[0] > 17:
        t = src[0] - 17
        ip = 1
        if t < 4:
            state = 'match_next'
        else:
            out += src[ip:ip + t]
            ip += t
            state = 'first_literal_run'
    while True:
        if state == 'loop':
            t = src[ip]
            ip += 1
            if t >= 16:
                state = 'match'
                continue
            if t == 0:
                t = ext(15)
            out += src[ip:ip + t + 3]
            ip += t + 3
            state = 'first_literal_run'
            continue
        if state == 'first_literal_run':
            t = src[ip]
            ip += 1
            if t >= 16:
                state = 'match'
                continue
            dist = 1 + 0x0800 + (t >> 2) + (src[ip] << 2)
            ip += 1
            copy_match(dist, 3)
            state = 'match_done'
            continue
        if state == 'match':
            if t >= 64:
                dist = 1 + ((t >> 2) & 7) + (src[ip] << 3)
                ip += 1
                copy_match(dist, (t >> 5) - 1 + 2)
            elif t >= 32:
                t &= 31
                if t == 0:
                    t = ext(31)
                dist = 1 + (src[ip] >> 2) + (src[ip + 1] << 6)
                ip += 2
                copy_match(dist, t + 2)
            elif t >= 16:
                dist = (t & 8) << 11
                t &= 7
                if t == 0:
                    t = ext(7)
                dist += (src[ip] >> 2) + (src[ip + 1] << 6)
                ip += 2
                if dist == 0:
                    break                   # end of stream
                copy_match(dist + 0x4000, t + 2)
            else:
                dist = 1 + (t >> 2) + (src[ip] << 2)
                ip += 1
                copy_match(dist, 2)
            state = 'match_done'
            continue
        if state == 'match_done':
            t = src[ip - 2] & 3
            if t == 0:
                state = 'loop'
                continue
            state = 'match_next'
            continue
        if state == 'match_next':
            out += src[ip:ip + t]
            ip += t
            t = src[ip]
            ip += 1
            state = 'match'
            continue
    if ip != n:
        raise ValueError(f'LZO: {n - ip} input bytes left after the end marker')
    if len(out) != out_len:
        raise ValueError(f'LZO: {len(out)} bytes decompressed, {out_len} expected')
    return bytes(out)

def decompress(data: bytes, *args) -> bytes:
    """
    Decompresses LZO1X compressed data.
    Supports signatures:
      decompress(data, uncompressed_size)
      decompress(data, False, uncompressed_size) (python-lzo compatibility)
    """
    if len(args) == 0:
        raise ValueError("uncompressed_size must be specified")
    uncompressed_size = args[-1]

    lib = _init_lzo()
    if lib is None:
        raise RuntimeError("No LZO decompression engine available! Ensure lzo1x_64.dll is in src/gbx/lib/")

    if lib == "python-lzo":
        import lzo
        return lzo.decompress(data, False, uncompressed_size)
    if lib == "pure":
        return _py_decompress(data, uncompressed_size)

    out_buf = bytes(uncompressed_size)
    c_size = c_uint32(len(data))
    bytes_written = c_uint32(uncompressed_size)
    
    status = lib.lzo1x_decompress_safe(data, c_size, out_buf, byref(bytes_written))
    if status != 0:
        raise ValueError(f"LZO decompression failed with error code {status}")
    return out_buf[:bytes_written.value]

def compress(data: bytes, *args) -> bytes:
    """
    Compresses data using LZO1X-999.
    Supports signatures:
      compress(data)
      compress(data, 1, False) (python-lzo compatibility)
    """
    lib = _init_lzo()
    if lib in (None, "pure"):
        raise RuntimeError("No LZO compression engine available! Ensure lzo1x_64.dll is in src/gbx/lib/")

    if lib == "python-lzo":
        import lzo
        return lzo.compress(data, 1, False)

    out_buffer = bytes(len(data) + (len(data) // 16) + 67)
    work_memory = bytes(524288)
    in_size = c_uint32(len(data))
    bytes_written = c_uint32(0)

    status = lib.lzo1x_999_compress(data, in_size, out_buffer, byref(bytes_written), work_memory)
    if status != 0:
        raise ValueError(f"LZO compression failed with error code {status}")
    return out_buffer[:bytes_written.value]
