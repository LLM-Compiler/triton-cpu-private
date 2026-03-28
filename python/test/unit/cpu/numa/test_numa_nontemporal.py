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

================================================================================
CACHE EVICTION POLICIES & NUMA MEMORY FLOW — Architecture Diagrams
================================================================================

Cache Eviction Policies (CacheModifier) — Detailed Explanation:
---------------------------------------------------------------
  • CG (Cache Global / Streaming):
    - Semantics: Data bypasses all cache levels. Loads use vmovntdqa (streaming
      load hint); stores use vmovntps/vmovntdq (non-temporal store).
    - Data path: CPU ←→ DRAM directly. No cache fill on load; no cache allocate
      on store. Reduces cache pollution.
    - When to use: One-shot reads, sequential streaming (e.g., weight matrix
      tiles read once per layer), or any remote store (consumer is on another
      socket and won't read from our cache).

  • CA (Cache Allocate / Temporal):
    - Semantics: Data is allocated in cache. Loads use vmovdqa/vmovaps; stores
      use vmovaps/vmovdq. Normal cache behavior.
    - Data path: On first access: DRAM → cache → CPU. On reuse: cache → CPU.
    - When to use: Loop-invariant data (bias, scale), KV-cache heads reused
      across iterations, small lookup tables.

ML Advantages in Machine Learning Context:
------------------------------------------
  - Weight streaming (CG): Transformer weights are read once per layer. Using
    CG prevents 100MB+ of weights from evicting useful L3 data (e.g., activations).
    L3 is typically 32–64MB; streaming weights avoids thrashing.

  - Bias / layer norm params (CA): Small vectors (e.g., 4096 floats) reused
    every matmul iteration. CA keeps them in L3; without it, each iteration
    would fetch from remote DRAM (~140ns vs ~80ns local, ~12ns L3).

  - KV-cache reuse (CA): Attention keys/values reused across query positions.
    Loop-invariant analysis assigns CA so the same tile stays cached.

  - Remote stores (CG): Writing activations to a buffer consumed by another
    NUMA node. Caching that write in our L3 wastes capacity—the consumer
    on the other socket cannot read our cache. CG writes directly to remote
    DRAM.

--------------------------------------------------------------------------------
Diagram 1: make_numa_buffer — NUMA-Pinned Allocation
--------------------------------------------------------------------------------

    tl.make_numa_buffer((N,), dtype, numa_node)  allocates memory on a
    specific NUMA node's DRAM. The OS places pages on that node's physical RAM.

    ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
    │  NUMA Node 0 (e.g., numa_node=0)│     │  NUMA Node 1 (e.g., numa_node=1) │
    │  ┌─────┐  ┌─────┐  ┌─────────┐ │     │  ┌─────┐  ┌─────┐  ┌─────────┐   │
    │  │Core │  │Core │  │ L3 Cache│ │     │  │Core │  │Core │  │ L3 Cache│   │
    │  │ 0   │  │ 1   │  │ (32MB)  │ │     │  │ 2   │  │ 3   │  │ (32MB)  │   │
    │  └──┬──┘  └──┬──┘  └────┬────┘ │     │  └──┬──┘  └──┬──┘  └────┬────┘   │
    │     │        │          │      │     │     │        │          │        │
    │     ▼        ▼          ▼      │     │     ▼        ▼          ▼        │
    │  ┌──────────────────────────┐ │     │  ┌──────────────────────────┐   │
    │  │   Local DRAM (64GB)       │ │     │  │   Local DRAM (64GB)       │   │
    │  │  ┌────────────────────┐   │ │     │  │  ┌────────────────────┐   │   │
    │  │  │ make_numa_buffer(   │   │ │     │  │  │ make_numa_buffer(   │   │   │
    │  │  │   ..., numa_node=0) │   │ │     │  │  │   ..., numa_node=1) │   │   │
    │  │  │   → pages here      │   │ │     │  │  │   → pages here      │   │   │
    │  │  └────────────────────┘   │ │     │  │  └────────────────────┘   │   │
    │  └──────────────────────────┘ │     │  └──────────────────────────┘   │
    └─────────────────────────────────┘     └─────────────────────────────────┘
                        │                                    │
                        └──────────── Interconnect ──────────┘
                              (~1.7× latency for remote)

--------------------------------------------------------------------------------
Diagram 2: Kernel 1 — streaming_oneshot_kernel (numa_load, CG / nontemporal)
--------------------------------------------------------------------------------

    One-shot load from remote NUMA node. No loop, no reuse → CG → bypass cache.

    ┌─ NUMA Node 1 (executing core) ─────┐    ┌─ NUMA Node 0 (remote) ────────┐
    │  ┌─────────┐                        │    │  ┌──────────────────────────┐  │
    │  │  Core   │  tl.numa_load(ptr,    │    │  │  Remote DRAM             │  │
    │  │         │    numa_src=0)         │    │  │  src_ptr [0..127]       │  │
    │  └────┬────┘                        │    │  └────────────▲─────────────┘  │
    │       │                             │    │               │                 │
    │       │  CG: vmovntdqa              │    │               │ 1. fetch        │
    │       │  (bypass L1/L2/L3)          │    │               │ 2. no cache fill│
    │       │                             │    │               │                 │
    │  ┌────┴────┐  L3: skipped           │    │               │                 │
    │  │   L3    │ ───────────────────────┼────┼───────────────┘                 │
    │  │ (skip)  │     Interconnect       │    │  (data flows directly to core)   │
    │  └─────────┘                        │    └─────────────────────────────────┘
    │       │                             │
    │       ▼  tl.store(dst_ptr)          │
    │  Local DRAM (dst_ptr)               │
    └────────────────────────────────────┘

--------------------------------------------------------------------------------
Diagram 3: Kernel 2 — streaming_loop_kernel (numa_load in loop, CG)
--------------------------------------------------------------------------------

    IV-dependent ptr: each iteration loads different remote block. Sequential
    streaming, no reuse → CG. (Masked path may not emit nontemporal in IR.)

    ┌─ NUMA Node 1 (executing core) ─────┐    ┌─ NUMA Node 0 (remote) ────────┐
    │  for k in 0, BLOCK_SIZE, 2*BLOCK:   │    │  src_ptr + k + offsets        │
    │    tile = numa_load(src_ptr+k, 0)   │    │  ┌─────┬─────┬─────┬─────┐    │
    │    acc += tile                      │    │  │ k=0 │ k=64│ k=128│ ... │    │
    │                                     │    │  └──▲──┴──▲──┴──▲───┴─────┘    │
    │  Each load: different block         │    │     │     │     │              │
    │  → no temporal reuse                │    │     │     │     │ CG: direct   │
    │  → CG (streaming)                   │    │     └─────┴─────┴──────────────┼─►
    │                                     │    │       (bypass cache each iter) │
    │  ┌─────────┐                        │    └────────────────────────────────┘
    │  │  Core   │  acc in registers      │
    │  └─────────┘                        │
    └────────────────────────────────────┘

--------------------------------------------------------------------------------
Diagram 4: Kernel 3 — invariant_numa_load_kernel (numa_load, CA / cached)
--------------------------------------------------------------------------------

    Bias loaded ONCE before loop, used every iteration → high reuse → CA.

    ┌─ NUMA Node 1 (executing core) ─────┐    ┌─ NUMA Node 0 (remote) ────────┐
    │  bias = numa_load(bias_ptr, 0)  ───┼───►│  bias_ptr in Remote DRAM      │
    │       │  CA: allocate in L3         │    │                                │
    │       ▼                             │    └────────────────────────────────┘
    │  ┌─────────────────────────┐        │
    │  │  L3 Cache               │        │
    │  │  ┌───────────────────┐  │        │
    │  │  │ bias block (cached)│  │        │  for k in ...:
    │  │  └─────────┬─────────┘  │        │    x = load(src_ptr + k)
    │  └────────────┼────────────┘        │    acc += x + bias  ◄── reuse bias
    │               │                     │         ▲
    │               │  reused every iter  │         │ from L3 (no remote fetch)
    │               ▼                     │         │
    │  ┌─────────┐                        │
    │  │  Core   │  acc += x + bias       │
    │  └─────────┘                        │
    └────────────────────────────────────┘

--------------------------------------------------------------------------------
Diagram 5: Kernel 4 — numa_store_kernel (numa_store, always CG)
--------------------------------------------------------------------------------

    Store to remote NUMA node. Always CG: bypass cache, write directly to
    remote DRAM. Caching remote writes wastes local L3; consumer is on another
    socket and won't read from our cache.

    ┌─ NUMA Node 1 (executing core) ─────┐    ┌─ NUMA Node 0 (remote) ────────┐
    │  ┌─────────┐                        │    │                                │
    │  │  Core   │  x = load(src_ptr)     │    │  ┌──────────────────────────┐ │
    │  │         │  numa_store(dst_ptr,   │    │  │  Remote DRAM              │ │
    │  │         │    x, numa_dst=0)      │    │  │  dst_ptr ◄────────────────┼─┼─┐
    │  └────┬────┘                        │    │  │  vmovntps (direct write)  │ │ │
    │       │                             │    │  └──────────────────────────┘ │ │
    │       │  CG: bypass L1/L2/L3        │    │            ▲                   │ │
    │       │  vmovntps / vmovntdq        │    │            │                   │ │
    │       │  (no cache allocate)       │    │            │                   │ │
    │       └─────────────────────────────┼────┼────────────┘                   │ │
    │                                     │    │  Interconnect                  │ │
    │  Local L3: NOT polluted by          │    └──────────────────────────────┘ │
    │  remote store data                   │                                      │
    └────────────────────────────────────┘                                      │
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
#
# Memory flow: Remote DRAM → (bypass cache) → Core. See Diagram 2 above.
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
#
# Memory flow: Each iteration loads different remote block (src_ptr+k).
# No reuse across iterations → CG. See Diagram 3 above.
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
#
# Memory flow: Remote DRAM → L3 (cached) → Core reuses from L3 each iter.
# See Diagram 4 above. ML use case: bias vector in matmul, layer norm params.
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
#
# Memory flow: Core → (bypass cache) → Remote DRAM. See Diagram 5 above.
# ML use case: writing activations/gradients to remote buffer for next stage.
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
