# NUMA Megatron Runtime Contract (Triton CPU)

This file captures the runtime-side ABI used by the NUMA Megatron prototype.

## Launcher Metadata ABI

`launch_metadata` may provide:

- `numa_socket_id`: integer socket shard id for this launch (`-1` means no partition).
- `numa_num_sockets`: total number of sockets (>= 1).
- `numa_enable_pin`: enable NUMA thread pinning in launcher.

The launcher uses these values to:

1. Slice grid iteration range by socket (`numa_socket_id`).
2. Pin worker threads to socket cores through `triton_numa_pin_thread`.
3. Keep OpenMP static scheduling inside socket-local execution regions.

## Runtime Primitives

`third_party/cpu/runtime/runtime_numa.cpp` exports:

- `triton_numa_alloc` / `triton_numa_free`
- `triton_numa_pin_thread`
- `triton_numa_wait` / `triton_numa_notify`
- `triton_numa_allreduce_sum_f32` (new prototype)

`triton_numa_allreduce_sum_f32` performs chunked shared-memory reduction to
avoid per-element atomic coherence thrash.

## MLIR Metadata Pass

When `TRITON_CPU_NUMA_SHARDS > 1`, pass
`triton-cpu-annotate-numa-shard-attrs` is inserted before dot backend lowering.
It annotates `ttcpu.dot` with:

- `ttcpu.numa.shard_count`
- `ttcpu.numa.partition_axis`

These attrs preserve shard intent through lowering stages.

## Validation Targets

1. Correctness: kernel outputs match non-NUMA path.
2. Affinity: per-thread CPU affinity maps to expected socket.
3. Reduction: all-reduce parity for fp32 shard buffers.
4. Performance: reduced remote DRAM traffic and controlled interconnect bandwidth.
