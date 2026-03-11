# NUMA Triton Documentation — Catalog

This directory contains documentation for the NUMA-aware extension to Triton-CPU. Use this catalog to find the right document for your needs.

---

## Document Index

| Document | Audience | When to Read |
|----------|----------|--------------|
| [NUMA_Extension_Developer_Guide.md](NUMA_Extension_Developer_Guide.md) | Developers extending or integrating NUMA | **Start here** for implementation details, pipeline ordering, and API usage |
| [NUMA_Triton_Developer_Document.md](NUMA_Triton_Developer_Document.md) | Architects, designers | Feature overview, design rationale, ASCII data-flow diagrams |
| [NUMA_Cache_Eviction_And_Memory_Flow.md](NUMA_Cache_Eviction_And_Memory_Flow.md) | ML engineers, performance tuners | CG vs CA semantics, memory-flow diagrams, ML workload advantages |
| [numa_runtime_developer_guide.md](numa_runtime_developer_guide.md) | Runtime / systems developers | C++ runtime API, `triton_numa_alloc`, `triton_numa_wait`/`notify`, platform fallbacks |
| [Nontemporal_Instruction_Analysis.md](Nontemporal_Instruction_Analysis.md) | Compiler / low-level debuggers | Where `vmovntdqa`/`vmovntps` are emitted, code paths, commit history |
| [NUMA_Distributed_Algorithms.md](NUMA_Distributed_Algorithms.md) | Distributed / ML engineers | Mapping from Triton-distributed paper, Broadcast/AllGather/ReduceScatter |

---

## Quick Reference by Topic

| Topic | Document |
|-------|----------|
| **What is the NUMA extension?** | [NUMA_Triton_Developer_Document.md](NUMA_Triton_Developer_Document.md) §1 |
| **How do I use `tl.numa_load` / `tl.numa_store`?** | [NUMA_Extension_Developer_Guide.md](NUMA_Extension_Developer_Guide.md) §1, §4.2 |
| **When does the compiler choose CG vs CA?** | [NUMA_Cache_Eviction_And_Memory_Flow.md](NUMA_Cache_Eviction_And_Memory_Flow.md) §1.3 |
| **Why use CG for weights, CA for bias?** | [NUMA_Cache_Eviction_And_Memory_Flow.md](NUMA_Cache_Eviction_And_Memory_Flow.md) §2 |
| **Compiler pipeline order (ConvertNUMAOps, etc.)** | [NUMA_Extension_Developer_Guide.md](NUMA_Extension_Developer_Guide.md) §2, §4.9 |
| **Runtime API (`triton_numa_alloc`, `triton_numa_wait`)** | [numa_runtime_developer_guide.md](numa_runtime_developer_guide.md) §4 |
| **Where does nontemporal get generated?** | [Nontemporal_Instruction_Analysis.md](Nontemporal_Instruction_Analysis.md) §2 |
| **Memory-flow diagrams (cores, cache, DRAM)** | [NUMA_Cache_Eviction_And_Memory_Flow.md](NUMA_Cache_Eviction_And_Memory_Flow.md) §3, §4 |
| **Distributed algorithms (Broadcast, AllGather, ReduceScatter)** | [NUMA_Distributed_Algorithms.md](NUMA_Distributed_Algorithms.md) |
| **Distributed GEMM (7 collectives, K-split)** | [NUMA_Distributed_Algorithms.md](NUMA_Distributed_Algorithms.md) §3 |

---

## Related Tests

| Test File | Purpose |
|-----------|---------|
| `python/test/unit/cpu/numa/test_numa_nontemporal.py` | Verifies CG/CA lowering and nontemporal LLVM IR for 4 kernel patterns |
| `python/test/unit/cpu/numa/test_numa_kernels.py` | JIT kernel execution tests |
| `python/test/unit/cpu/numa/test_numa_bindings.py` | Runtime and Python bindings |
| `python/test/unit/cpu/numa/test_numa_distributed.py` | Broadcast, AllGather, ReduceScatter, Producer-Consumer |
| `python/test/unit/cpu/numa/test_distributed_gemm.py` | Distributed BMM vs torch.bmm (4 NUMA nodes) |
