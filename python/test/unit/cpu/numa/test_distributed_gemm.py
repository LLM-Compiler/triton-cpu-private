"""
Test distributed GEMM against reference BMM (torch.bmm and triton_tem_fused_bmm_2).

Compares output of distributed_bmm (K-split across 4 NUMA ranks) to reference.
Requires 4 NUMA nodes for full testing; on single-node systems, partial correctness
may still be verified.
"""

import pytest
import torch

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
        path = "/sys/devices/system/node/online"
        with open(path) as f:
            s = f.read().strip()
        if "-" in s:
            lo, hi = s.split("-")
            return int(hi) - int(lo) + 1
        return 1
    except Exception:
        pass
    return 1


def _skip_if_insufficient_numa():
    if _num_numa_nodes() < 4:
        pytest.skip("Requires 4 NUMA nodes for distributed GEMM tests")


def test_distributed_gemm_vs_torch_bmm():
    """Compare distributed GEMM output to torch.bmm (reference)."""
    _skip_if_insufficient_numa()

    batch = 8
    # A: (batch, M, K) = (8, 2048, 64), B: (batch, K, N) = (8, 64, 2048)
    A = torch.randn(batch, M, K, dtype=torch.float32)
    B = torch.randn(batch, K, N, dtype=torch.float32)

    # Reference: torch.bmm
    C_ref = torch.bmm(A, B)

    # Distributed: 4 C_partial buffers
    C_partial = [
        torch.zeros(batch, M, N, dtype=torch.float32)
        for _ in range(NUM_RANKS)
    ]
    C_out = torch.zeros(batch, M, N, dtype=torch.float32)

    distributed_bmm(A, B, C_partial, C_out)

    assert torch.allclose(C_out, C_ref, rtol=1e-4, atol=1e-3), (
        f"Distributed GEMM mismatch: max_diff={torch.max(torch.abs(C_out - C_ref))}"
    )


