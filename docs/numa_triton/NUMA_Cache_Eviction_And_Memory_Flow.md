# NUMA Cache Eviction Policies & Memory Flow — Detailed Guide

> **Purpose**: This document provides a comprehensive explanation of cache eviction policies (CG vs CA), NUMA memory flow diagrams, and their advantages in machine learning workloads. It complements the tests in `python/test/unit/cpu/numa/test_numa_nontemporal.py`.

---

## Table of Contents

1. [Cache Eviction Policies — Detailed Explanation](#1-cache-eviction-policies--detailed-explanation)
2. [Machine Learning Advantages](#2-machine-learning-advantages)
3. [NUMA Topology & make_numa_buffer](#3-numa-topology--make_numa_buffer)
4. [Memory Flow Diagrams by Kernel](#4-memory-flow-diagrams-by-kernel)
5. [Summary Table](#5-summary-table)

---

## 1. Cache Eviction Policies — Detailed Explanation

The Triton-CPU NUMA extension uses two cache modifiers when lowering `tl.numa_load` and `tl.numa_store` to LLVM IR. The compiler's polyhedral reuse analysis (`analyzeReuseDistance`) selects the appropriate modifier based on loop structure and access patterns.

### 1.1 CG (Cache Global / Streaming)

**Semantics**:
- Data **bypasses all cache levels** (L1, L2, L3).
- **Loads**: Use `vmovntdqa` (streaming load hint). Data flows from DRAM directly to CPU registers; the CPU may still bring it into L1 for the current instruction but signals low reuse to the prefetcher.
- **Stores**: Use `vmovntps` / `vmovntdq` (non-temporal store). Data is written directly to DRAM without allocating a cache line.

**Data Path**:
```
Load:  Remote/Local DRAM ──(interconnect)──► CPU registers  (cache NOT filled)
Store: CPU registers ──(interconnect)──► Remote/Local DRAM  (cache NOT allocated)
```

**When to Use**:
- One-shot reads (data used once, never again).
- Sequential streaming (e.g., weight matrix tiles read once per layer).
- IV-dependent pointers in a loop (each iteration loads a different block).
- **All remote stores** — the consumer is on another NUMA node and cannot read from our local cache.

**LLVM IR**: Load/store instructions carry `!nontemporal` metadata.

---

### 1.2 CA (Cache Allocate / Temporal)

**Semantics**:
- Data **is allocated in cache** for potential reuse.
- **Loads**: Use `vmovdqa` / `vmovaps` (normal load). Data is fetched into cache; subsequent accesses hit the cache.
- **Stores**: Use `vmovaps` / `vmovdq` (normal store). Data is written to cache (write-allocate) and may be written back to DRAM later.

**Data Path**:
```
First access:  DRAM ──► Cache ──► CPU
Reuse:         Cache ──► CPU  (no DRAM fetch)
```

**When to Use**:
- Loop-invariant data (same pointer, same block every iteration).
- High temporal reuse (e.g., bias vector added to every matmul tile).
- Small lookup tables or scale/gamma vectors in layer norm.
- KV-cache heads reused across attention query positions.

**LLVM IR**: Load/store instructions do **not** have `!nontemporal` metadata.

---

### 1.3 How the Compiler Chooses

The `analyzeReuseDistance()` function in `ConvertNUMAOps.cpp` applies these heuristics in order:

| Step | Condition | Result |
|------|-----------|--------|
| 1 | `eviction_policy="evict_first"` attribute | CG |
| 2 | `eviction_policy="evict_last"` attribute | CA |
| 3 | No enclosing loops; result consumed inside a downstream loop | CA |
| 4 | No enclosing loops; result not consumed in loop | CG |
| 5 | Pointer defined outside outermost loop (loop-invariant) | CA |
| 6 | Footprint (tripCount × tileBytes) ≤ 30 MB | CA |
| 7 | Default | CG |

**NumaStoreOp** is always lowered with **CG** — remote stores never use cache.

---

## 2. Machine Learning Advantages

### 2.1 Weight Streaming (CG)

**Scenario**: Transformer forward pass. Each layer reads a large weight matrix (e.g., 4096×4096 per attention head). Weights are read **once per layer** and never reused within that layer.

**Without CG**: Each weight tile would be cached in L3. L3 is typically 32–64 MB per socket. A single layer's weights can exceed 100 MB. Caching would evict useful data (e.g., activations, intermediate results) and cause thrashing.

**With CG**: Weights stream directly from DRAM to the core. L3 remains available for activations and other high-reuse data. No cache pollution.

**Latency trade-off**: A single streaming load may be slightly slower than a cached load on first access, but the overall workload benefits because L3 is preserved for data that actually benefits from caching.

---

### 2.2 Bias / Layer Norm Parameters (CA)

**Scenario**: Matmul with bias: `Y = X @ W + bias`. The bias vector (e.g., 4096 floats) is added to every output tile. Same bias, many iterations.

**Without CA**: Each iteration would fetch the bias from remote DRAM (~140 ns for remote, ~80 ns for local). With 64 iterations, that's 64 × 140 ns = 9 µs just for bias fetches.

**With CA**: First access fetches bias into L3 (~140 ns). Subsequent 63 iterations read from L3 (~12 ns). Total: 140 ns + 63 × 12 ns ≈ 0.9 µs — **~10× faster**.

**Same applies to**: Layer norm scale/gamma, small lookup tables.

---

### 2.3 KV-Cache Reuse (CA)

**Scenario**: Attention over a KV-cache. For each query position, we attend to all previous key/value heads. The same KV tiles are read for many query positions.

**With CA**: KV tiles are cached. As we iterate over query positions, we reuse cached K and V. Reduces remote DRAM traffic significantly.

**Loop-invariance**: The KV pointer is often defined outside the inner loop over query positions. The analysis detects this and assigns CA.

---

### 2.4 Remote Stores (CG)

**Scenario**: Producer on NUMA node 1 writes activations to a buffer. Consumer on NUMA node 0 will read that buffer. The buffer resides on node 0's DRAM (allocated via `make_numa_buffer(..., numa_node=0)`).

**Without CG**: Our store would allocate a line in our local L3. The consumer on node 0 cannot read our L3 — it would fetch from DRAM anyway. Our cache line is wasted.

**With CG**: We write directly to remote DRAM. No local cache pollution. The consumer reads from its local DRAM (same node as the buffer). Optimal.

---

## 3. NUMA Topology & make_numa_buffer

### 3.1 Two-Socket NUMA Topology

```
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│  NUMA Node 0 (e.g., numa_node=0)        │     │  NUMA Node 1 (e.g., numa_node=1)        │
│                                         │     │                                         │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │     │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
│  │Core │  │Core │  │Core │  │Core │   │     │  │Core │  │Core │  │Core │  │Core │   │
│  │  0  │  │  1  │  │  2  │  │  3  │   │     │  │  4  │  │  5  │  │  6  │  │  7  │   │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │     │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │
│     │        │        │        │      │     │     │        │        │        │      │
│     ▼        ▼        ▼        ▼      │     │     ▼        ▼        ▼        ▼      │
│  ┌─────────────────────────────────┐  │     │  ┌─────────────────────────────────┐  │
│  │  L1/L2 (per-core)               │  │     │  │  L1/L2 (per-core)               │  │
│  └──────────────┬──────────────────┘  │     │  └──────────────┬──────────────────┘  │
│                 ▼                     │     │                 ▼                     │
│  ┌─────────────────────────────────┐  │     │  ┌─────────────────────────────────┐  │
│  │  L3 Cache (shared, ~32MB)        │  │     │  │  L3 Cache (shared, ~32MB)        │  │
│  └──────────────┬──────────────────┘  │     │  └──────────────┬──────────────────┘  │
│                 ▼                     │     │                 ▼                     │
│  ┌─────────────────────────────────┐  │     │  ┌─────────────────────────────────┐  │
│  │  Local DRAM (64GB)              │  │     │  │  Local DRAM (64GB)              │  │
│  │  make_numa_buffer(..., node=0)  │  │     │  │  make_numa_buffer(..., node=1)  │  │
│  │  → pages allocated here        │  │     │  │  → pages allocated here          │  │
│  └─────────────────────────────────┘  │     │  └─────────────────────────────────┘  │
└─────────────────┬─────────────────────┘     └─────────────────┬─────────────────────┘
                   │                                             │
                   └──────────────── Interconnect ───────────────┘
                         (~1.7× latency for remote vs local)
```

### 3.2 make_numa_buffer

`tl.make_numa_buffer((N,), dtype, numa_node)` allocates memory on a specific NUMA node's physical RAM.

- **Implementation**: Calls `triton_numa_alloc(size, node)` which uses `numa_alloc_onnode` (with libnuma) or `mmap` + `mbind` (Linux fallback).
- **Effect**: The OS places the allocated pages on the requested node's DRAM. Cores on that node have local access (~80 ns); cores on other nodes have remote access (~140 ns).
- **Use case**: Place weights on the node that will read them; place output buffers on the node that will consume them.

---

## 4. Memory Flow Diagrams by Kernel

### 4.1 Diagram 1: streaming_oneshot_kernel (numa_load, CG)

**Kernel**: Single block load from remote NUMA node. No loop, no reuse.

**Cache policy**: CG (streaming). Data bypasses cache.

```
┌─ NUMA Node 1 (executing core) ─────────────┐    ┌─ NUMA Node 0 (remote) ────────────────┐
│                                             │    │                                       │
│  tl.numa_load(src_ptr, numa_src=0)         │    │  src_ptr in Remote DRAM               │
│  tl.store(dst_ptr, tile)                    │    │  [block 0..127]                       │
│                                             │    │                                       │
│  ┌─────────┐                                │    │  ┌─────────────────────────────┐     │
│  │  Core   │  ◄────────────────────────────┼────┼──│  Remote DRAM                │     │
│  │         │     CG: vmovntdqa              │    │  │  src_ptr [0..127]           │     │
│  └────┬────┘     (bypass L1/L2/L3)          │    │  └─────────────────────────────┘     │
│       │                                     │    │            ▲                          │
│       │  L3: skipped                        │    │            │ 1. fetch                  │
│  ┌────┴────┐                                │    │            │ 2. no cache fill          │
│  │   L3    │  (cache NOT used)              │    │            │ 3. data to core           │
│  │ (skip)  │ ──────────────────────────────┼────┼────────────┘                          │
│  └─────────┘     Interconnect               │    │                                       │
│       │                                     │    └───────────────────────────────────────┘
│       ▼  tl.store(dst_ptr)                 │
│  Local DRAM (dst_ptr)                      │
└────────────────────────────────────────────┘
```

**Key point**: Data flows directly from remote DRAM to the core. L3 on the executing node is not used. Ideal for one-shot reads.

---

### 4.2 Diagram 2: streaming_loop_kernel (numa_load in loop, CG)

**Kernel**: Loop over k; each iteration loads `src_ptr + k`. IV-dependent pointer.

**Cache policy**: CG. Each iteration loads a different block; no temporal reuse.

```
┌─ NUMA Node 1 (executing core) ─────────────┐    ┌─ NUMA Node 0 (remote) ────────────────┐
│                                            │    │                                       │
│  for k in 0, BLOCK_SIZE, 2*BLOCK_SIZE:    │    │  src_ptr + k + offsets                │
│    tile = numa_load(src_ptr + k, numa_src=0)   │  ┌─────┬─────┬─────┬─────┐            │
│    acc += tile                               │  │ k=0 │ k=64│k=128│ ... │            │
│                                            │    │  └──▲──┴──▲──┴──▲───┴─────┘            │
│  Each load: different block                │    │     │     │     │                     │
│  → no temporal reuse                       │    │     │     │     │ CG: direct fetch   │
│  → CG (streaming)                          │    │     └─────┴─────┴────────────────────┼─►
│                                            │    │       (bypass cache each iteration)  │
│  ┌─────────┐                               │    └───────────────────────────────────────┘
│  │  Core   │  acc in registers             │
│  └─────────┘                               │
└────────────────────────────────────────────┘
```

**Key point**: Sequential streaming. Each iteration fetches a new block; caching would evict the previous block before reuse. CG avoids cache thrashing.

---

### 4.3 Diagram 3: invariant_numa_load_kernel (numa_load, CA)

**Kernel**: Bias loaded once before loop; used every iteration inside the loop.

**Cache policy**: CA (cache allocate). High temporal reuse.

```
┌─ NUMA Node 1 (executing core) ─────────────┐    ┌─ NUMA Node 0 (remote) ────────────────┐
│                                            │    │                                       │
│  bias = numa_load(bias_ptr, numa_src=0) ───┼───►│  bias_ptr in Remote DRAM              │
│       │                                    │    │  (first access only)                   │
│       │  CA: allocate in L3                │    └───────────────────────────────────────┘
│       ▼                                    │
│  ┌─────────────────────────────┐           │
│  │  L3 Cache                   │           │
│  │  ┌───────────────────────┐   │           │
│  │  │ bias block (cached)   │   │           │  for k in ...:
│  │  └───────────┬───────────┘   │           │    x = load(src_ptr + k)
│  └──────────────┼───────────────┘           │    acc += x + bias  ◄── reuse from L3
│                 │                           │         ▲
│                 │  reused every iteration   │         │ no remote fetch
│                 ▼                           │         │
│  ┌─────────┐                                │
│  │  Core   │  acc += x + bias               │
│  └─────────┘                                │
└────────────────────────────────────────────┘
```

**Key point**: First access fetches from remote DRAM into L3. All subsequent iterations read from L3. Massive speedup for small, frequently reused data.

---

### 4.4 Diagram 4: numa_store_kernel (numa_store, always CG)

**Kernel**: Store to remote NUMA node. Always CG.

**Cache policy**: CG. Bypass cache; write directly to remote DRAM.

```
┌─ NUMA Node 1 (executing core) ─────────────┐    ┌─ NUMA Node 0 (remote) ────────────────┐
│                                            │    │                                       │
│  x = load(src_ptr)                         │    │  ┌─────────────────────────────┐     │
│  numa_store(dst_ptr, x, numa_dst=0)        │    │  │  Remote DRAM                │     │
│                                            │    │  │  dst_ptr ◄──────────────────┼─────┼─┐
│  ┌─────────┐                               │    │  │  vmovntps (direct write)   │     │ │
│  │  Core   │  ─────────────────────────────┼────┼──┼─────────────────────────────┘     │ │
│  │         │  CG: vmovntps / vmovntdq      │    │            ▲                          │ │
│  └────┬────┘  (bypass L1/L2/L3)            │    │            │                         │ │
│       │                                    │    │            │                         │ │
│  ┌────┴────┐  L3: NOT allocated            │    │            │                         │ │
│  │   L3    │  (no cache pollution)         │    │            │                         │ │
│  │ (skip)  │                               │    │            │                         │ │
│  └─────────┘                               │    │  Interconnect                        │ │
│                                            │    └───────────────────────────────────────┘
└────────────────────────────────────────────┘
```

**Key point**: Caching this store would waste L3 capacity. The consumer on node 0 cannot read our cache. Direct write to remote DRAM is optimal.

---

## 5. Summary Table

| Kernel | Operation | Cache Policy | x86 Instruction | Rationale |
|--------|------------|--------------|-----------------|-----------|
| streaming_oneshot_kernel | numa_load | CG | vmovntdqa | One-shot; no reuse |
| streaming_loop_kernel | numa_load | CG | vmovntdqa | IV-dependent; each iter different block |
| invariant_numa_load_kernel | numa_load | CA | vmovdqa/vmovaps | Loop-invariant; reused every iter |
| numa_store_kernel | numa_store | CG | vmovntps/vmovntdq | Remote store; consumer on other socket |

---

## Related Documents

- [NUMA_Extension_Developer_Guide.md](NUMA_Extension_Developer_Guide.md) — Implementation and pipeline
- [numa_runtime_developer_guide.md](numa_runtime_developer_guide.md) — Runtime API reference
- [Nontemporal_Instruction_Analysis.md](Nontemporal_Instruction_Analysis.md) — Where nontemporal is generated
- `python/test/unit/cpu/numa/test_numa_nontemporal.py` — Test kernels and verification
