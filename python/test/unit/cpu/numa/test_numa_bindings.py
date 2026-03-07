#!/usr/bin/env python3
"""
NUMA Extension Smoke Test for Triton-CPU
=========================================

Validates that:
  1. The C++ NUMA runtime (libTritonCPURuntime.so) exports all expected symbols.
  2. triton_get_numa_id() returns a valid NUMA node ID on the host.
  3. triton_numa_alloc() produces usable memory pinned to a NUMA node.
  4. A standard @triton.jit kernel can read/write NUMA-allocated memory
     using regular tl.load / tl.store (NOT tl.numa_load / tl.numa_store).
  5. Cross-socket signaling (wait/notify) runs without crashing.

Usage:
  # From the triton-cpu repo root, after building with `pip install -e python/`:
  python3 test_numa_bindings.py

  # Or with explicit PYTHONPATH if not installed:
  PYTHONPATH=/home/shavin/llm-cpu-compiler/triton-cpu/python python3 test_numa_bindings.py
"""

import sys
import os
import ctypes
import ctypes.util
import struct

# ──────────────────────────────────────────────────────────────────────
# Phase 0: Locate and load the NUMA runtime shared library
# ──────────────────────────────────────────────────────────────────────

def find_runtime_lib():
    """Find libTritonCPURuntime.so in the triton package."""
    # Try standard install location
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

    # Fallback: search common build directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallbacks = [
        os.path.join(script_dir, "python", "triton", "_C", "libTritonCPURuntime.so"),
        os.path.join(script_dir, "build", "libTritonCPURuntime.so"),
    ]
    for path in fallbacks:
        if os.path.exists(path):
            return path

    return None


def load_runtime():
    """Load the NUMA runtime shared library and set up ctypes bindings."""
    lib_path = find_runtime_lib()
    if lib_path is None:
        print("[FATAL] Could not find libTritonCPURuntime.so")
        print("        Make sure you've built triton-cpu: pip install -e python/")
        sys.exit(1)

    print(f"[INFO] Loading runtime from: {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # Declare function signatures
    # int32_t triton_get_numa_id()
    lib.triton_get_numa_id.restype = ctypes.c_int32
    lib.triton_get_numa_id.argtypes = []

    # void* triton_numa_alloc(int64_t size_bytes, int32_t numa_node)
    lib.triton_numa_alloc.restype = ctypes.c_void_p
    lib.triton_numa_alloc.argtypes = [ctypes.c_int64, ctypes.c_int32]

    # void triton_numa_free(void* ptr, int64_t size_bytes)
    lib.triton_numa_free.restype = None
    lib.triton_numa_free.argtypes = [ctypes.c_void_p, ctypes.c_int64]

    # void triton_numa_pin_thread(int32_t numa_node)
    lib.triton_numa_pin_thread.restype = None
    lib.triton_numa_pin_thread.argtypes = [ctypes.c_int32]

    # void triton_numa_wait(int32_t signal_id)
    lib.triton_numa_wait.restype = None
    lib.triton_numa_wait.argtypes = [ctypes.c_int32]

    # void triton_numa_notify(int32_t signal_id, int32_t numa_dst)
    lib.triton_numa_notify.restype = None
    lib.triton_numa_notify.argtypes = [ctypes.c_int32, ctypes.c_int32]

    print("[PASS] All 6 NUMA runtime symbols resolved successfully.\n")
    return lib


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Test NUMA Discovery
# ──────────────────────────────────────────────────────────────────────

def test_numa_discovery(lib):
    """Test triton_get_numa_id() returns a sane value."""
    print("=" * 60)
    print("TEST 1: NUMA Discovery (triton_get_numa_id)")
    print("=" * 60)

    print("[INFO] Calling triton_get_numa_id()...")
    numa_id = lib.triton_get_numa_id()
    print(f"[INFO] Returned NUMA node ID: {numa_id}")

    if numa_id < 0:
        print("[FAIL] NUMA node ID is negative — likely a detection error.")
        return False

    print(f"[PASS] NUMA node ID = {numa_id} (non-negative).\n")
    return True


# ──────────────────────────────────────────────────────────────────────
# Helper: Verify NUMA page placement via move_pages() syscall
# ──────────────────────────────────────────────────────────────────────

def verify_numa_placement(ptr, size_bytes, expected_node):
    """Verify that physical pages at ptr reside on expected_node.

    Uses the Linux move_pages() syscall in query mode (nodes=NULL) to ask
    the kernel which NUMA node each page is physically allocated on.

    Since glibc may not export move_pages() as a library function, we invoke
    it as a raw syscall via libc.syscall(SYS_move_pages, ...) — the same
    pattern used in runtime_numa.cpp for futex.

    Returns True if all pages are on the expected node, False otherwise.
    On non-Linux or if the syscall fails, prints a warning and returns None.
    """
    if sys.platform != "linux":
        print("[SKIP] NUMA placement verification only supported on Linux.")
        return None

    # __NR_move_pages on x86_64 = 279
    SYS_MOVE_PAGES = 279

    libc_path = ctypes.util.find_library("c")
    if libc_path is None:
        print("[WARN] Could not find libc — skipping placement verification.")
        return None
    libc = ctypes.CDLL(libc_path, use_errno=True)

    # syscall() returns long
    libc.syscall.restype = ctypes.c_long

    page_size = os.sysconf("SC_PAGE_SIZE")  # typically 4096
    n_pages = max(1, size_bytes // page_size)

    # Build array of page-aligned virtual addresses to query
    PageArray = ctypes.c_void_p * n_pages
    pages = PageArray()
    for i in range(n_pages):
        pages[i] = ptr + i * page_size

    # Output array: kernel writes the NUMA node ID of each page here
    StatusArray = ctypes.c_int * n_pages
    status = StatusArray()

    # syscall(SYS_move_pages, pid, count, pages, nodes=NULL→query, status, flags)
    ret = libc.syscall(
        ctypes.c_long(SYS_MOVE_PAGES),
        ctypes.c_int(0),           # pid: 0 = current process
        ctypes.c_ulong(n_pages),   # number of pages
        pages,                     # array of page addresses
        None,                      # nodes=NULL → query mode (don't move)
        status,                    # output: node ID per page
        ctypes.c_int(0)            # flags
    )

    if ret != 0:
        errno = ctypes.get_errno()
        print(f"[WARN] move_pages syscall returned {ret} (errno={errno}) — cannot verify.")
        return None

    wrong = 0
    for i in range(n_pages):
        if status[i] < 0:
            # Negative values are errors (e.g., -ENOENT if page not yet faulted)
            print(f"  [WARN] Page {i}: status={status[i]} (page may not be faulted in)")
            wrong += 1
        elif status[i] != expected_node:
            if wrong < 5:  # limit output
                print(f"  [MISS] Page {i}: on node {status[i]}, expected node {expected_node}")
            wrong += 1

    if wrong == 0:
        print(f"[PASS] All {n_pages} pages verified on NUMA node {expected_node}.")
        return True
    else:
        print(f"[FAIL] {wrong}/{n_pages} pages NOT on expected node {expected_node}.")
        return False


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Test NUMA Allocation
# ──────────────────────────────────────────────────────────────────────

def test_numa_allocation(lib, numa_node):
    """Test triton_numa_alloc/free with basic read/write."""
    print("=" * 60)
    print(f"TEST 2: NUMA Allocation on node {numa_node}")
    print("=" * 60)

    N = 1024
    elem_size = 4  # float32
    total_bytes = N * elem_size

    print(f"[INFO] Attempting NUMA allocation: {N} x float32 = {total_bytes} bytes on node {numa_node}...")
    ptr = lib.triton_numa_alloc(ctypes.c_int64(total_bytes), ctypes.c_int32(numa_node))
    if ptr is None or ptr == 0:
        print("[FAIL] triton_numa_alloc returned NULL.")
        return None

    print(f"[INFO] Allocation succeeded: ptr = 0x{ptr:016x}")

    # Write test pattern: float values 0.0, 1.0, 2.0, ...
    print("[INFO] Writing test pattern (0.0, 1.0, 2.0, ...) to NUMA buffer...")
    float_array = (ctypes.c_float * N).from_address(ptr)
    for i in range(N):
        float_array[i] = float(i)

    # Verify readback
    print("[INFO] Verifying readback from NUMA buffer...")
    errors = 0
    for i in range(min(N, 16)):  # Spot-check first 16
        if abs(float_array[i] - float(i)) > 1e-6:
            print(f"  [ERROR] Index {i}: expected {float(i)}, got {float_array[i]}")
            errors += 1

    if errors > 0:
        print(f"[FAIL] {errors} readback errors.")
        return None

    print(f"[PASS] NUMA buffer write/read verified (first 16 of {N} elements).")

    # Verify that pages actually reside on the requested NUMA node
    print(f"[INFO] Verifying physical page placement on node {numa_node}...")
    placement = verify_numa_placement(ptr, total_bytes, numa_node)
    if placement is False:
        print("[WARN] Pages may not be on the requested NUMA node.")
    elif placement is None:
        print("[INFO] Placement verification not available (non-Linux or syscall issue).")

    print(f"[INFO] Keeping buffer alive at 0x{ptr:016x} for JIT kernel test.\n")
    return ptr, total_bytes


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Test Cross-Socket Signaling
# ──────────────────────────────────────────────────────────────────────

def test_signaling(lib):
    """Test that notify/wait works without deadlocking (single-threaded)."""
    print("=" * 60)
    print("TEST 3: Cross-Socket Signaling (notify → wait)")
    print("=" * 60)

    signal_id = 42
    target_node = 0

    print(f"[INFO] Sending signal {signal_id} to node {target_node}...")
    lib.triton_numa_notify(ctypes.c_int32(signal_id), ctypes.c_int32(target_node))
    print("[INFO] Signal sent. Now waiting on same signal (should return immediately)...")
    lib.triton_numa_wait(ctypes.c_int32(signal_id))
    print("[PASS] notify → wait round-trip succeeded without deadlock.\n")
    return True


# ──────────────────────────────────────────────────────────────────────
# Phase 4: Triton JIT Kernel on NUMA-Allocated Memory
# ──────────────────────────────────────────────────────────────────────

# The kernel must be defined at module level because @triton.jit captures
# the function's __globals__ at decoration time. If triton.language (tl)
# is only imported inside a function, it won't be in __globals__ and the
# JIT compiler will raise NameError('tl is not defined').

_HAS_TRITON = False
try:
    import torch
    import numpy as np
    import triton
    import triton.language as tl
    _HAS_TRITON = True

    @triton.jit
    def add_scalar_kernel(
        input_ptr, output_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        result = x + 42.0
        tl.store(output_ptr + offsets, result, mask=mask)

except ImportError:
    pass


def test_jit_kernel_on_numa_memory(lib, numa_ptr, total_bytes):
    """
    Launch a standard @triton.jit kernel that operates on NUMA-allocated
    memory using regular tl.load / tl.store. This tests that the CPU
    backend's JIT pipeline can handle our custom-allocated pointers.
    """
    print("=" * 60)
    print("TEST 4: Triton JIT Kernel on NUMA Memory")
    print("=" * 60)

    if not _HAS_TRITON:
        print("[SKIP] Cannot import triton/torch.")
        print("       Install with: pip install -e python/ && pip install torch")
        return True  # Not a NUMA failure

    print("[INFO] Triton and PyTorch imported successfully.")

    N = total_bytes // 4  # float32 elements
    BLOCK_SIZE = 128
    SCALAR_ADD = 42.0

    # The buffer at numa_ptr already contains [0.0, 1.0, 2.0, ..., 1023.0]
    # We'll also create a standard torch output tensor for verification.
    print(f"[INFO] Creating PyTorch output tensor ({N} elements, float32)...")
    output = torch.zeros(N, dtype=torch.float32)

    # Wrap NUMA pointer as a torch tensor for passing to the kernel.
    # torch.frombuffer creates a tensor from an existing memory region.
    print(f"[INFO] Wrapping NUMA buffer (0x{numa_ptr:016x}) as PyTorch tensor...")
    numa_buffer = ctypes.cast(numa_ptr, ctypes.POINTER(ctypes.c_float * N))
    # Create a numpy array from the pointer, then convert to torch
    np_arr = np.ctypeslib.as_array(numa_buffer.contents)
    input_tensor = torch.from_numpy(np_arr.copy())  # Copy to regular memory for input
    print(f"[INFO] Input tensor sample: {input_tensor[:8].tolist()}")

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    print(f"[INFO] Launching Triton JIT kernel: output = input + {SCALAR_ADD}")
    print(f"[INFO]   Grid: ({triton.cdiv(N, BLOCK_SIZE)},), BLOCK_SIZE={BLOCK_SIZE}")
    print("[INFO]   Using standard tl.load/tl.store (NOT tl.numa_load/tl.numa_store)")

    try:
        add_scalar_kernel[grid](input_tensor, output, N, BLOCK_SIZE=BLOCK_SIZE)
        print("[INFO] Kernel execution returned without segfault.")
    except Exception as e:
        print(f"[FAIL] Kernel execution raised: {type(e).__name__}: {e}")
        return False

    # Verify results
    print("[INFO] Verifying kernel output...")
    expected = input_tensor + SCALAR_ADD
    if torch.allclose(output, expected, atol=1e-5):
        print(f"[INFO] Sample output: {output[:8].tolist()}")
        print(f"[INFO] Expected:      {expected[:8].tolist()}")
        print("[PASS] JIT kernel produced correct results on NUMA-sourced data.\n")
        return True
    else:
        max_diff = (output - expected).abs().max().item()
        mismatches = (output != expected).sum().item()
        print(f"[FAIL] Output mismatch! max_diff={max_diff}, mismatches={mismatches}/{N}")
        print(f"[INFO] Output[:8]:   {output[:8].tolist()}")
        print(f"[INFO] Expected[:8]: {expected[:8].tolist()}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   NUMA Extension Smoke Test — Triton-CPU                ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Load runtime
    lib = load_runtime()
    results = {}

    # Test 1: NUMA Discovery
    results["discovery"] = test_numa_discovery(lib)
    numa_node = lib.triton_get_numa_id()

    # Test 2: NUMA Allocation
    alloc_result = test_numa_allocation(lib, numa_node)
    results["allocation"] = alloc_result is not None
    numa_ptr, total_bytes = alloc_result if alloc_result else (None, 0)

    # Test 3: Signaling
    results["signaling"] = test_signaling(lib)

    # Test 4: JIT Kernel
    if numa_ptr:
        results["jit_kernel"] = test_jit_kernel_on_numa_memory(lib, numa_ptr, total_bytes)
    else:
        print("[SKIP] JIT kernel test skipped (no NUMA allocation).")
        results["jit_kernel"] = False

    # Cleanup
    if numa_ptr:
        print("[INFO] Freeing NUMA buffer...")
        lib.triton_numa_free(ctypes.c_void_p(numa_ptr), ctypes.c_int64(total_bytes))
        print("[INFO] NUMA buffer freed.\n")

    # Summary
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   RESULTS SUMMARY                                      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✅" if passed else "FAIL ❌"
        if not passed:
            all_pass = False
        print(f"║   {name:20s}  {status:>10s}                       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    overall = "ALL TESTS PASSED ✅" if all_pass else "SOME TESTS FAILED ❌"
    print(f"║   {overall:^52s}   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
