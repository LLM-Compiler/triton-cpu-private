#!/usr/bin/env python3
"""
Benchmark script for distributed GEMM with NUMA-aware allocation.

Allocates A, B, C_partial, and C_out on the appropriate NUMA nodes:
- A, B: on NUMA 0 (shared input; all ranks read - single contiguous buffer)
- C_partial[0..3]: on NUMA 0, 1, 2, 3 respectively (each rank writes to local DRAM)
- C_out: on NUMA 0 (reduce writes here)

Requires 4 NUMA nodes. Run from triton-cpu root:
  PYTHONPATH=python python python/test/unit/cpu/numa/benchmark_distributed_gemm.py

  # Or from triton-cpu/python:
  python test/unit/cpu/numa/benchmark_distributed_gemm.py [--batch 8] [--validate]
"""

import argparse
import ctypes
import os
import sys
import time

import torch

# Ensure triton-cpu python is on path when run from repo (script in .../python/test/unit/cpu/numa/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.normpath(os.path.join(_script_dir, *([".."] * 4)))  # numa->cpu->unit->test->python
if os.path.isdir(_python_dir) and "triton" in os.listdir(_python_dir):
    sys.path.insert(0, _python_dir)

from triton.numa import alloc_utils
from triton.numa.distributed_gemm import (
    distributed_bmm,
    M,
    N,
    K,
    NUM_RANKS,
)


def _num_numa_nodes():
    """Return number of NUMA nodes (best-effort)."""
    try:
        with open("/sys/devices/system/node/online") as f:
            s = f.read().strip()
        if "-" in s:
            lo, hi = s.split("-")
            return int(hi) - int(lo) + 1
        return 1
    except Exception:
        pass
    return 1


def numa_tensor(shape, dtype, numa_node):
    """
    Allocate a tensor on the specified NUMA node using triton_numa_alloc.

    Returns (tensor, ptr, size_bytes, buffer_ref).
    Caller MUST call alloc_utils.numa_free(ptr, size_bytes) when done.
    The buffer_ref must be kept alive while using the tensor.
    """
    numel = 1
    for d in shape:
        numel *= d

    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float16:
        elem_size = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    size_bytes = numel * elem_size
    ptr = alloc_utils.numa_alloc(size_bytes, numa_node)
    addr = int(ptr) if not isinstance(ptr, int) else ptr

    # Create ctypes array view (must keep alive)
    if dtype == torch.float32:
        buf = (ctypes.c_float * numel).from_address(addr)
        tensor = torch.frombuffer(buf, dtype=torch.float32).view(*shape)
    else:
        buf = (ctypes.c_uint16 * numel).from_address(addr)
        tensor = torch.frombuffer(buf, dtype=torch.float16).view(*shape)

    return tensor, ptr, size_bytes, buf


def allocate_numa_buffers(batch, dtype=torch.float32):
    """
    Allocate A, B, C_partial, C_out on appropriate NUMA nodes.

    Returns (A, B, C_partial_list, C_out, allocations) where allocations
    is a list of (ptr, size_bytes) for cleanup.
    """
    num_numa = _num_numa_nodes()
    if num_numa < NUM_RANKS:
        raise RuntimeError(
            f"Requires {NUM_RANKS} NUMA nodes, found {num_numa}. "
            "Run on a multi-socket system."
        )

    allocations = []  # (ptr, size_bytes) for cleanup

    # A: (batch, M, K) - shared input, allocate on NUMA 0
    A, ptr_a, sz_a, _ = numa_tensor((batch, M, K), dtype, 0)
    allocations.append((ptr_a, sz_a))

    # B: (batch, K, N) - shared input, allocate on NUMA 0
    B, ptr_b, sz_b, _ = numa_tensor((batch, K, N), dtype, 0)
    allocations.append((ptr_b, sz_b))

    # C_partial: one per NUMA node - each rank writes to its local buffer
    C_partial = []
    for r in range(NUM_RANKS):
        c, ptr, sz, _ = numa_tensor((batch, M, N), dtype, r)
        C_partial.append(c)
        allocations.append((ptr, sz))

    # C_out: reduce destination, allocate on NUMA 0
    C_out, ptr_c, sz_c, _ = numa_tensor((batch, M, N), dtype, 0)
    allocations.append((ptr_c, sz_c))

    return A, B, C_partial, C_out, allocations


def free_numa_buffers(allocations):
    """Free all NUMA-allocated buffers."""
    for ptr, size_bytes in allocations:
        alloc_utils.numa_free(ptr, size_bytes)


def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed GEMM with NUMA allocation")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--validate", action="store_true", help="Validate against torch.bmm")
    args = parser.parse_args()

    num_numa = _num_numa_nodes()
    print(f"NUMA nodes: {num_numa}")
    if num_numa < NUM_RANKS:
        print(f"ERROR: Requires {NUM_RANKS} NUMA nodes. Exiting.")
        sys.exit(1)

    batch = args.batch
    dtype = torch.float32
    print(f"\nDistributed GEMM benchmark: batch={batch}, M={M}, N={N}, K={K}")
    print(f"  A: ({batch}, {M}, {K}), B: ({batch}, {K}, {N}), C: ({batch}, {M}, {N})")
    print(f"  NUMA allocation: A,B,C_out on node 0; C_partial[r] on node r\n")

    # Allocate with NUMA placement
    A, B, C_partial, C_out, allocations = allocate_numa_buffers(batch, dtype)

    try:
        # Initialize with random data (in-place on NUMA buffers)
        A.copy_(torch.randn(batch, M, K, dtype=dtype))
        B.copy_(torch.randn(batch, K, N, dtype=dtype))
        for c in C_partial:
            c.zero_()
        C_out.zero_()

        if args.validate:
            C_ref = torch.bmm(A, B)
            distributed_bmm(A, B, C_partial, C_out)
            max_diff = torch.max(torch.abs(C_out - C_ref)).item()
            print(f"Validation: max_diff={max_diff:.2e}")
            if max_diff > 1e-3:
                print("  WARNING: Mismatch vs reference!")
            else:
                print("  OK\n")

        # Warmup
        for _ in range(args.warmup):
            for c in C_partial:
                c.zero_()
            C_out.zero_()
            distributed_bmm(A, B, C_partial, C_out)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(args.iterations):
            for c in C_partial:
                c.zero_()
            C_out.zero_()
            distributed_bmm(A, B, C_partial, C_out)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        elapsed = end - start
        per_iter = elapsed / args.iterations
        flops = 2 * batch * M * N * K  # per BMM
        tflops = (flops * args.iterations) / elapsed / 1e12
        print(f"Time: {elapsed:.4f}s total, {per_iter*1000:.3f} ms/iter")
        print(f"Throughput: {tflops:.4f} TFLOPS")
        print(f"  ({args.iterations} iterations)")

    finally:
        free_numa_buffers(allocations)

    print("\nDone.")


if __name__ == "__main__":
    main()
