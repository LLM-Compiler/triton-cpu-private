#!/usr/bin/env python3
"""
Benchmark script for standard (non-distributed) Triton GEMM.

Uses the same input dimensions as distributed GEMM for comparison:
  A: (batch, 2048, 64), B: (batch, 64, 2048) -> C: (batch, 2048, 2048)

No NUMA/distributed scheduling - standard torch allocation and single-node Triton BMM.

Run from triton-cpu root:
  PYTHONPATH=python python python/test/unit/cpu/numa/benchmark_standard_gemm.py

  # Or from triton-cpu/python:
  python test/unit/cpu/numa/benchmark_standard_gemm.py [--batch 8] [--validate]
"""

import argparse
import os
import sys
import time

import torch
import triton
import triton.language as tl

# Ensure triton-cpu python is on path when run from repo
_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.normpath(os.path.join(_script_dir, *([".."] * 4)))
if os.path.isdir(_python_dir) and "triton" in os.listdir(_python_dir):
    sys.path.insert(0, _python_dir)

# Match distributed_gemm dimensions
M, N, K = 2048, 2048, 64
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 16


@triton.jit
def standard_bmm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_aq: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bq: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard BMM: batch of (M, K) @ (K, N) -> (M, N).
    Same tiling as distributed_gemm reference (triton_tem_fused_bmm_2).
    """
    pid = tl.program_id(0)
    idx_q = tl.program_id(1)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = 8 * grid_n  # GROUP_M = 8
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = rm % M
    rbn = rn % N

    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex = idx_n + N * idx_m + (M * N) * idx_q
    tl.store(C_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)


def _bmm_grid(batch, meta):
    """Grid for standard BMM: (num_tiles, batch)."""
    grid_m = (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
    grid_n = (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
    num_tiles = grid_m * grid_n
    return (num_tiles, batch)


def standard_bmm(A, B, C_out):
    """
    Standard BMM: (batch, M, K) @ (batch, K, N) -> (batch, M, N).
    A: (batch, 2048, 64), B: (batch, 64, 2048), C_out: (batch, 2048, 2048).
    """
    batch = A.shape[0]
    meta = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
    grid = _bmm_grid(batch, meta)

    stride_aq = A.stride(0)
    stride_am = A.stride(1)
    stride_ak = A.stride(2)
    stride_bq = B.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(2)

    standard_bmm_kernel[grid](
        A, B, C_out,
        M=M, N=N, K=K,
        stride_aq=stride_aq, stride_am=stride_am, stride_ak=stride_ak,
        stride_bq=stride_bq, stride_bk=stride_bk, stride_bn=stride_bn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark standard Triton GEMM")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--validate", action="store_true", help="Validate against torch.bmm")
    args = parser.parse_args()

    batch = args.batch
    dtype = torch.float32
    print(f"\nStandard GEMM benchmark: batch={batch}, M={M}, N={N}, K={K}")
    print(f"  A: ({batch}, {M}, {K}), B: ({batch}, {K}, {N}), C: ({batch}, {M}, {N})")
    print(f"  (Same dimensions as distributed GEMM, no NUMA)\n")

    # Standard allocation
    A = torch.randn(batch, M, K, dtype=dtype)
    B = torch.randn(batch, K, N, dtype=dtype)
    C_out = torch.empty(batch, M, N, dtype=dtype)

    if args.validate:
        C_ref = torch.bmm(A, B)
        standard_bmm(A, B, C_out)
        max_diff = torch.max(torch.abs(C_out - C_ref)).item()
        print(f"Validation: max_diff={max_diff:.2e}")
        if max_diff > 1e-3:
            print("  WARNING: Mismatch vs reference!")
        else:
            print("  OK\n")

    # Warmup
    for _ in range(args.warmup):
        standard_bmm(A, B, C_out)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        standard_bmm(A, B, C_out)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    per_iter = elapsed / args.iterations
    flops = 2 * batch * M * N * K
    tflops = (flops * args.iterations) / elapsed / 1e12
    print(f"Time: {elapsed:.4f}s total, {per_iter*1000:.3f} ms/iter")
    print(f"Throughput: {tflops:.4f} TFLOPS")
    print(f"  ({args.iterations} iterations)")

    print("\nDone.")


if __name__ == "__main__":
    main()
