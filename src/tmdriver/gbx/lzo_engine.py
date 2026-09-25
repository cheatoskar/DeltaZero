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

    return None

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
    if lib is None:
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
