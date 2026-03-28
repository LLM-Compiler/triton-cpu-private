"""
NUMA-distributed algorithms implemented with Triton-CPU primitives.

Maps concepts from the Triton-distributed paper (arXiv 2504.19442) to NUMA:
- Rank -> NUMA node (program_id)
- Remote access -> numa_load / numa_store
- Signal exchange -> wait / notify
"""

import triton
import triton.language as tl

# Signal ID ranges to avoid cross-test collision
SIGNAL_BROADCAST = 100
SIGNAL_ALLGATHER = 200
SIGNAL_REDUCESCATTER = 300
SIGNAL_PRODUCER_CONSUMER = 400


# ---------------------------------------------------------------------------
# Broadcast: Rank 0 sends data to all other ranks (2-node version)
# ---------------------------------------------------------------------------

@triton.jit
def broadcast_2node_kernel(
    src_ptr,
    dst_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """
    Rank 0 broadcasts to rank 1.
    - Rank 0: numa_store to rank 1's buffer, then notify
    - Rank 1: wait, then data is ready in local buffer
    """
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if rank == 0:
        data = tl.load(src_ptr + offsets)
        tl.numa_store(dst_ptr + offsets, data, numa_dst=1)
        tl.notify(signal_id=signal_base + 1, numa_dst=1)
    else:
        tl.wait(signal_id=signal_base + 1)
        # Data already in dst_ptr (written by rank 0 via numa_store)


# ---------------------------------------------------------------------------
# AllGather (push mode, 2-node): Each rank pushes its shard to the other
# ---------------------------------------------------------------------------

@triton.jit
def allgather_push_2node_kernel(
    buf0_ptr,
    buf1_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """
    Each rank has a local shard. Each pushes to the other's buffer.
    buf0 on node 0: [shard0, slot_for_shard1]. buf1 on node 1: [slot_for_shard0, shard1].
    Rank 0: local=buf0[0:BLOCK], remote=buf1. Writes to buf1[0:BLOCK], waits for buf0[BLOCK:].
    Rank 1: local=buf1[BLOCK:], remote=buf0. Writes to buf0[BLOCK:], waits for buf1[0:BLOCK].
    Note: numa_dst must be a compile-time literal, so we branch.
    """
    my_rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if my_rank == 0:
        local_ptr = buf0_ptr
        remote_ptr = buf1_ptr
        my_shard = tl.load(local_ptr + offsets)
        tl.numa_store(remote_ptr + offsets, my_shard, numa_dst=1)
        tl.notify(signal_id=signal_base + 0, numa_dst=1)
        tl.wait(signal_id=signal_base + 1)
    else:
        local_ptr = buf1_ptr + BLOCK_SIZE
        remote_ptr = buf0_ptr
        my_shard = tl.load(local_ptr + offsets)
        tl.numa_store(remote_ptr + BLOCK_SIZE + offsets, my_shard, numa_dst=0)
        tl.notify(signal_id=signal_base + 1, numa_dst=0)
        tl.wait(signal_id=signal_base + 0)


# ---------------------------------------------------------------------------
# ReduceScatter (push mode, 2-node): Producer pushes partial sums; consumer reduces
# ---------------------------------------------------------------------------

@triton.jit
def reducescatter_push_2node_kernel(
    buf0_ptr,
    buf1_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """
    buf0 on node 0: [partial0, recv_from_1]. buf1 on node 1: [recv_from_0, partial1].
    Each rank sends its partial to the other's recv slot, waits, then reduces.
    Note: numa_dst must be a compile-time literal, so we branch.
    """
    my_rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if my_rank == 0:
        # Rank 0: partial in buf0[0:BLOCK], recv from rank 1 in buf0[BLOCK:2*BLOCK]
        local_partial_ptr = buf0_ptr
        remote_recv_ptr = buf0_ptr + BLOCK_SIZE  # rank 1 writes here (to our node)
        output_ptr = buf0_ptr
        my_partial = tl.load(local_partial_ptr + offsets)
        tl.numa_store(buf1_ptr + offsets, my_partial, numa_dst=1)  # send to rank 1's recv
        tl.notify(signal_id=signal_base + 0, numa_dst=1)
        tl.wait(signal_id=signal_base + 1)
        received = tl.load(remote_recv_ptr + offsets)
        result = my_partial + received
        tl.store(output_ptr + offsets, result)
    else:
        # Rank 1: partial in buf1[BLOCK:2*BLOCK], recv from rank 0 in buf1[0:BLOCK]
        local_partial_ptr = buf1_ptr + BLOCK_SIZE
        remote_recv_ptr = buf1_ptr  # rank 0 writes here (to our node)
        output_ptr = buf1_ptr + BLOCK_SIZE
        my_partial = tl.load(local_partial_ptr + offsets)
        tl.numa_store(buf0_ptr + BLOCK_SIZE + offsets, my_partial, numa_dst=0)  # send to rank 0's recv
        tl.notify(signal_id=signal_base + 1, numa_dst=0)
        tl.wait(signal_id=signal_base + 0)
        received = tl.load(remote_recv_ptr + offsets)
        result = my_partial + received
        tl.store(output_ptr + offsets, result)


# ---------------------------------------------------------------------------
# Producer-Consumer: Producer on node 0, consumer on node 1
# ---------------------------------------------------------------------------

@triton.jit
def producer_consumer_kernel(
    producer_data_ptr,
    consumer_buf_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """
    Producer (rank 0): load data, numa_store to consumer's buffer, notify
    Consumer (rank 1): wait, then data is ready (or numa_load to use it)
    """
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if rank == 0:
        data = tl.load(producer_data_ptr + offsets)
        tl.numa_store(consumer_buf_ptr + offsets, data, numa_dst=1)
        tl.notify(signal_id=signal_base, numa_dst=1)
    else:
        tl.wait(signal_id=signal_base)
        # Data is in consumer_buf_ptr (written by producer)
        # Consumer can optionally load and process:
        result = tl.load(consumer_buf_ptr + offsets)
        tl.store(consumer_buf_ptr + offsets, result)  # in-place or to output
