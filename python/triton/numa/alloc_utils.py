"""
NUMA allocation utilities for Triton-CPU.

Exposes triton_numa_alloc, triton_numa_free, and triton_get_numa_id
from libTritonCPURuntime.so for host-side NUMA-pinned buffer allocation.
Used by distributed algorithms that require symmetric buffers per NUMA node.
"""

import ctypes
import os


def _find_runtime_lib():
    """Find libTritonCPURuntime.so in the triton package."""
    try:
        import triton
        triton_dir = os.path.dirname(triton.__file__)
        candidates = [
            os.path.join(triton_dir, "_C", "libTritonCPURuntime.so"),
            os.path.join(triton_dir, "_C", "libTritonCPURuntime.dylib"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    except ImportError:
        pass

    # Fallback: search relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    fallbacks = [
        os.path.join(this_dir, "..", "..", "_C", "libTritonCPURuntime.so"),
        os.path.join(this_dir, "..", "..", "_C", "libTritonCPURuntime.dylib"),
    ]
    for path in fallbacks:
        path = os.path.normpath(path)
        if os.path.exists(path):
            return path

    return None


_lib = None


def _get_lib():
    """Load and return the NUMA runtime library (lazy)."""
    global _lib
    if _lib is None:
        lib_path = _find_runtime_lib()
        if lib_path is None:
            raise RuntimeError(
                "Could not find libTritonCPURuntime.so. "
                "Build triton-cpu with: pip install -e python/"
            )
        _lib = ctypes.CDLL(lib_path)
        _lib.triton_get_numa_id.restype = ctypes.c_int32
        _lib.triton_get_numa_id.argtypes = []
        _lib.triton_numa_alloc.restype = ctypes.c_void_p
        _lib.triton_numa_alloc.argtypes = [ctypes.c_int64, ctypes.c_int32]
        _lib.triton_numa_free.restype = None
        _lib.triton_numa_free.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    return _lib


def get_numa_id():
    """Return the NUMA node ID of the current CPU."""
    return _get_lib().triton_get_numa_id()


def numa_alloc(size_bytes: int, numa_node: int):
    """
    Allocate size_bytes on the specified NUMA node.
    Returns a ctypes void pointer. Caller must free with numa_free.
    """
    ptr = _get_lib().triton_numa_alloc(ctypes.c_int64(size_bytes), ctypes.c_int32(numa_node))
    if ptr is None or ptr == 0:
        raise RuntimeError(f"triton_numa_alloc failed for {size_bytes} bytes on node {numa_node}")
    return ptr


def numa_free(ptr, size_bytes: int):
    """Free a NUMA allocation from numa_alloc."""
    _get_lib().triton_numa_free(ctypes.c_void_p(ptr), ctypes.c_int64(size_bytes))
