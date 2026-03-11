# NUMA Distributed Algorithms — Mapping from Triton-distributed

> **Purpose**: Maps standard distributed algorithms from the Triton-distributed paper (arXiv 2504.19442) to NUMA Triton primitives and documents the implementations in `triton.numa.distributed`.

---

## 1. Concept Mapping

The Triton-distributed paper targets multi-GPU systems with OpenSHMEM (NVSHMEM/ROCSHMEM). NUMA Triton targets multi-socket CPU servers. The following mapping applies:

| Triton-distributed (GPU) | NUMA Triton (CPU) |
|--------------------------|-------------------|
| **Rank** (GPU / PE) | NUMA node; `tl.program_id(0)` as rank index |
| **Symmetric memory** | `tl.make_numa_buffer(shape, dtype, numa_node)` per node |
| **Remote put** | `tl.numa_store(ptr, value, numa_dst=N)` |
| **Remote get** | `tl.numa_load(ptr, numa_src=N)` |
| **Signal set** | `tl.notify(signal_id, numa_dst)` |
| **Signal wait** | `tl.wait(signal_id)` |
| **putmem_nbi / getmem** | `numa_store` / `numa_load` with `wait`/`notify` for sync |

**Key difference**: Triton-distributed uses separate processes and OpenSHMEM; NUMA Triton uses a single process with explicit NUMA placement. Each kernel block (`program_id`) acts as one "rank"; block-to-NUMA mapping depends on scheduler/affinity.

---

## 2. Implemented Algorithms

### 2.1 Broadcast (2-node)

**Paper**: Single source broadcasts to all ranks.

**NUMA implementation**: `broadcast_2node_kernel` in `triton.numa.distributed`.

- **Rank 0**: Load from `src_ptr`, `numa_store` to `dst_ptr` (rank 1's buffer), `notify` rank 1.
- **Rank 1**: `wait` for signal, then data is in `dst_ptr` (written by rank 0).

```python
from triton.numa.distributed import broadcast_2node_kernel, SIGNAL_BROADCAST

broadcast_2node_kernel[(2,)](src, dst, BLOCK_SIZE=128, signal_base=SIGNAL_BROADCAST)
```

---

### 2.2 AllGather (push mode, 2-node)

**Paper**: Algorithm 1 — each rank pushes its local shard to all others.

**NUMA implementation**: `allgather_push_2node_kernel`.

- Each rank has a local shard. Layout: `buf0` on node 0 = [shard0, slot_for_shard1]; `buf1` on node 1 = [slot_for_shard0, shard1].
- Rank 0: Push to `buf1[0:BLOCK]`, notify rank 1, wait for rank 1.
- Rank 1: Push to `buf0[BLOCK:2*BLOCK]`, notify rank 0, wait for rank 0.

```python
from triton.numa.distributed import allgather_push_2node_kernel, SIGNAL_ALLGATHER

allgather_push_2node_kernel[(2,)](buf0, buf1, BLOCK_SIZE=64, signal_base=SIGNAL_ALLGATHER)
```

---

### 2.3 ReduceScatter (push mode, 2-node)

**Paper**: Algorithm 3 — producer pushes partials; consumer waits and reduces.

**NUMA implementation**: `reducescatter_push_2node_kernel`.

- Layout: `buf0` = [partial0, recv_from_1], `buf1` = [recv_from_0, partial1].
- Each rank sends its partial to the other's recv slot, waits, then reduces (adds) and stores.

```python
from triton.numa.distributed import reducescatter_push_2node_kernel, SIGNAL_REDUCESCATTER

reducescatter_push_2node_kernel[(2,)](buf0, buf1, BLOCK_SIZE=64, signal_base=SIGNAL_REDUCESCATTER)
```

---

### 2.4 Producer-Consumer

**Paper**: Async-task pattern — producer computes, stores to consumer's buffer, notifies.

**NUMA implementation**: `producer_consumer_kernel`.

- **Rank 0 (producer)**: Load from `producer_data_ptr`, `numa_store` to `consumer_buf_ptr`, `notify` rank 1.
- **Rank 1 (consumer)**: `wait`, then data is in `consumer_buf_ptr`.

```python
from triton.numa.distributed import producer_consumer_kernel, SIGNAL_PRODUCER_CONSUMER

producer_consumer_kernel[(2,)](producer_data, consumer_buf, BLOCK_SIZE=64, signal_base=SIGNAL_PRODUCER_CONSUMER)
```

---

## 3. 4-Node Collectives and Distributed GEMM

For 4 NUMA nodes, `triton.numa.distributed_gemm` provides:

| Collective | Kernel | Purpose |
|------------|--------|---------|
| Broadcast | `broadcast_4node_kernel` | Rank 0 → ranks 1, 2, 3 |
| Scatter | `scatter_4node_kernel` | Rank 0 scatters chunks to each rank |
| Gather | `gather_4node_kernel` | Ranks 1, 2, 3 → rank 0 |
| AllGather | `allgather_4node_kernel` | Each rank pushes shard to shared dst |
| Reduce | `reduce_4node_kernel` | Sum from all ranks to rank 0 |
| AllReduce | `allreduce_4node_kernel` | Reduce + Broadcast |
| ReduceScatter | `reducescatter_4node_kernel` | Each rank gets sum of one shard |

**Distributed GEMM** (`distributed_bmm`): K-split matmul `(batch, 2048, 64) @ (batch, 64, 2048)` matching `triton_tem_fused_bmm_2`. Uses Scatter (logical K-split), Partial GEMM, and Reduce (batched `numa_load` sum).

```python
from triton.numa.distributed_gemm import distributed_bmm, NUM_RANKS

A = torch.randn(8, 2048, 64, dtype=torch.float32)
B = torch.randn(8, 64, 2048, dtype=torch.float32)
C_partial = [torch.zeros(8, 2048, 2048, dtype=torch.float32) for _ in range(NUM_RANKS)]
C_out = torch.zeros(8, 2048, 2048, dtype=torch.float32)
distributed_bmm(A, B, C_partial, C_out)
# C_out == torch.bmm(A, B)
```

---

## 4. Signal ID Ranges

To avoid cross-test collision, use disjoint ranges:

| Algorithm | Signal base | Signals used |
|-----------|-------------|--------------|
| Broadcast | 100 | 101 |
| AllGather | 200 | 200, 201 |
| ReduceScatter | 300 | 300, 301 |
| Producer-Consumer | 400 | 400 |
| Distributed GEMM | 500 | 500+ |

---

## 5. Host-Side Allocation

NUMA Triton lacks symmetric virtual addressing. For multi-rank algorithms:

1. Use `triton.numa.alloc_utils.numa_alloc(size, node)` for per-node buffers.
2. Pass base pointers to the kernel; each block uses `program_id` to select its buffers.

Example:

```python
from triton.numa.alloc_utils import numa_alloc, numa_free, get_numa_id

size = 128 * 4  # 128 floats
ptr0 = numa_alloc(size, 0)
ptr1 = numa_alloc(size, 1)
# Wrap as torch tensors and pass to kernel...
numa_free(ptr0, size)
numa_free(ptr1, size)
```

---

## 6. Limitations

| Limitation | Mitigation |
|------------|------------|
| `numa_dst` must be compile-time literal | Branch in kernel with literal 0/1 for 2-node |
| No symmetric addressing | Host allocates and passes pointers |
| Block-to-NUMA mapping | Document CPU affinity; use `triton_numa_pin_thread` if available |
| Single-node correctness | ReduceScatter may have memory-ordering issues when both blocks run on same node |
| Single-node testing | Tests skip when `num_numa_nodes < 2` |

---

## 7. Related Documents

- [NUMA_Extension_Developer_Guide.md](NUMA_Extension_Developer_Guide.md) — Primitives and pipeline
- [numa_runtime_developer_guide.md](numa_runtime_developer_guide.md) — Runtime API
- `python/triton/numa/distributed.py` — Kernel implementations
- `python/test/unit/cpu/numa/test_numa_distributed.py` — Tests
