"""
Tests for tl.numa_load / tl.numa_store lowering to LLVM IR.

Verifies:
  1. Functional correctness of numa_load and numa_store.
  2. Streaming access patterns produce !nontemporal loads in LLVM IR.
  3. Loop-invariant access patterns do NOT produce !nontemporal loads.
  4. Remote stores always produce !nontemporal stores.

Note: nontemporal propagation to LLVM works via vector::LoadOp / vector::StoreOp
which have native nontemporal attributes. The masked variants (vector::MaskedLoadOp,
vector::MaskedStoreOp) do NOT support nontemporal, so tests use unmasked patterns
for LLVM IR verification.
"""
import pytest
import torch
import re

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Streaming numa_load — simple one-shot (no loop, no mask)
#
# A single block load from a remote NUMA region with no enclosing loop.
# The analysis sees no loop => no reuse => CacheModifier::CG => nontemporal.
# No mask means it goes through vector::LoadOp (which carries nontemporal).
# ---------------------------------------------------------------------------

@triton.jit
def streaming_oneshot_kernel(
    src_ptr, dst_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    tile = tl.numa_load(src_ptr + offsets, numa_src=0)
    tl.store(dst_ptr + offsets, tile)


# ---------------------------------------------------------------------------
# Kernel 2: Streaming numa_load inside a loop (IV-dependent, with mask)
#
# For functional correctness testing only. The masked load path uses
# vector::MaskedLoadOp which cannot carry nontemporal in this MLIR version.
# ---------------------------------------------------------------------------

@triton.jit
def streaming_loop_kernel(
    src_ptr, dst_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE):
        ptrs = src_ptr + k + offsets
        mask = (k + offsets) < N
        tile = tl.numa_load(ptrs, numa_src=0, mask=mask, other=0.0)
        acc += tile
    tl.store(dst_ptr + pid * BLOCK_SIZE + offsets, acc)


# ---------------------------------------------------------------------------
# Kernel 3: Loop-invariant numa_load (ptr defined outside the loop)
#
# The bias pointer is loaded ONCE before the loop. Its result is consumed
# inside the loop every iteration => high temporal reuse => CA => no nontemporal.
# No mask on the bias load => goes through vector::LoadOp.
# ---------------------------------------------------------------------------

@triton.jit
def invariant_numa_load_kernel(
    bias_ptr, src_ptr, dst_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    bias = tl.numa_load(bias_ptr + offsets, numa_src=0)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE):
        ptrs = src_ptr + k + offsets
        mask = (k + offsets) < N
        x = tl.load(ptrs, mask=mask, other=0.0)
        acc += x + bias
    tl.store(dst_ptr + pid * BLOCK_SIZE + offsets, acc)


# ---------------------------------------------------------------------------
# Kernel 4: numa_store (remote writes should always be nontemporal)
# No mask => goes through vector::StoreOp (which carries nontemporal).
# ---------------------------------------------------------------------------

@triton.jit
def numa_store_kernel(
    src_ptr, dst_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src_ptr + offsets)
    tl.numa_store(dst_ptr + offsets, x, numa_dst=0)


# ===========================================================================
# Functional correctness tests
# ===========================================================================

def test_streaming_oneshot_correctness():
    """One-shot numa_load without mask produces correct results."""
    BLOCK_SIZE = 128
    src = torch.arange(BLOCK_SIZE, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    streaming_oneshot_kernel[(1,)](src, dst, BLOCK_SIZE=BLOCK_SIZE)
    assert torch.equal(src, dst), f"dst={dst[:8]}, expected={src[:8]}"


def test_streaming_loop_correctness():
    """Masked numa_load in a streaming loop produces correct results."""
    N = 256
    BLOCK_SIZE = 64
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    streaming_loop_kernel[(1,)](src, dst, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
    for k in range(0, N, BLOCK_SIZE):
        expected += src[k:k + BLOCK_SIZE]
    assert torch.allclose(dst, expected), f"dst={dst[:8]}, expected={expected[:8]}"


def test_invariant_numa_load_correctness():
    """Loop-invariant numa_load produces correct results."""
    N = 256
    BLOCK_SIZE = 64
    bias = torch.ones(BLOCK_SIZE, dtype=torch.float32) * 10.0
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    invariant_numa_load_kernel[(1,)](bias, src, dst, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
    for k in range(0, N, BLOCK_SIZE):
        expected += src[k:k + BLOCK_SIZE] + bias
    assert torch.allclose(dst, expected), f"dst={dst[:8]}, expected={expected[:8]}"


def test_numa_store_correctness():
    """numa_store writes correct values."""
    BLOCK_SIZE = 128
    src = torch.arange(BLOCK_SIZE, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    numa_store_kernel[(1,)](src, dst, BLOCK_SIZE=BLOCK_SIZE)
    assert torch.equal(src, dst), f"src={src[:8]}, dst={dst[:8]}"


# ===========================================================================
# LLVM IR nontemporal metadata tests
# ===========================================================================

def _has_nontemporal_load(llir: str) -> bool:
    """Check if any load instruction (plain or vector) has !nontemporal."""
    return bool(re.search(r'\bload\b.*!nontemporal', llir))


def _has_nontemporal_store(llir: str) -> bool:
    """Check if any store instruction has !nontemporal."""
    return bool(re.search(r'\bstore\b.*!nontemporal', llir))


def _count_nontemporal_loads(llir: str) -> int:
    """Count load instructions with !nontemporal metadata."""
    return len(re.findall(r'\bload\b.*!nontemporal', llir))


def test_streaming_oneshot_has_nontemporal():
    """One-shot numa_load (no loop, no mask) should produce nontemporal LLVM load.

    Pattern: a single remote block fetch with no surrounding loop. The
    reuse analysis sees no loop and no downstream loop consumer =>
    CacheModifier::CG => nontemporal=true on the vector::LoadOp.
    """
    BLOCK_SIZE = 128
    src = torch.arange(BLOCK_SIZE, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    meta = streaming_oneshot_kernel[(1,)](src, dst, BLOCK_SIZE=BLOCK_SIZE)
    llir = meta.asm["llir"]

    assert _has_nontemporal_load(llir), (
        f"Expected nontemporal load for streaming one-shot numa_load.\n"
        f"LLVM IR:\n{llir}"
    )


def test_invariant_numa_load_no_nontemporal():
    """Loop-invariant numa_load (ptr outside loop, result consumed in loop)
    should NOT produce nontemporal loads.

    The bias load happens before the loop but its result is used in every
    iteration. The polyhedral analysis detects the downstream loop consumer
    and assigns CA (cache-all) => no !nontemporal.
    """
    N = 256
    BLOCK_SIZE = 64
    bias = torch.ones(BLOCK_SIZE, dtype=torch.float32) * 10.0
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    meta = invariant_numa_load_kernel[(1,)](bias, src, dst, N, BLOCK_SIZE=BLOCK_SIZE)
    llir = meta.asm["llir"]

    assert not _has_nontemporal_load(llir), (
        f"Loop-invariant numa_load should NOT have !nontemporal.\n"
        f"LLVM IR:\n{llir}"
    )


def test_numa_store_has_nontemporal():
    """numa_store should always produce nontemporal LLVM stores."""
    BLOCK_SIZE = 128
    src = torch.arange(BLOCK_SIZE, dtype=torch.float32)
    dst = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    meta = numa_store_kernel[(1,)](src, dst, BLOCK_SIZE=BLOCK_SIZE)
    llir = meta.asm["llir"]

    assert _has_nontemporal_store(llir), (
        f"Expected nontemporal store for numa_store.\n"
        f"LLVM IR:\n{llir}"
    )
