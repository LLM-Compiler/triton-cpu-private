"""
Distributed GEMM using 7 collective primitives: Broadcast, Scatter, Gather,
AllGather, Reduce, AllReduce, ReduceScatter.

Implements tensor-parallel GEMM across NUMA nodes (K-split) that produces
the same result as a single-node BMM. Designed for 4 NUMA nodes.

Reference: triton_tem_fused_bmm_2 in test.py - computes (batch, 2048, 64) @ (batch, 64, 2048)

Collective usage in the pipeline:
- Scatter: (logical) K-dimension split - each rank reads A[:, r*16:(r+1)*16], B[r*16:(r+1)*16, :]
- Partial GEMM: each rank computes C_partial_r
- Reduce: sum C_partial across ranks (reduce_tiles_batched_kernel uses numa_load; reduce_tile_4node_kernel uses wait/notify)
- Broadcast, Gather, AllGather, AllReduce, ReduceScatter: available as 4-node primitives for custom pipelines
"""

import triton
import triton.language as tl

# Signal ID base for distributed GEMM collectives
SIGNAL_DIST_GEMM = 500

# Reference BMM dimensions
M, N, K = 2048, 2048, 64
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 16
NUM_RANKS = 4
K_PER_RANK = K // NUM_RANKS  # 16


# ---------------------------------------------------------------------------
# 4-node collective primitives (numa_dst must be literal 0,1,2,3)
# ---------------------------------------------------------------------------

@triton.jit
def broadcast_4node_kernel(
    src_ptr,
    dst0_ptr,
    dst1_ptr,
    dst2_ptr,
    dst3_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """Rank 0 broadcasts to ranks 1, 2, 3."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if rank == 0:
        data = tl.load(src_ptr + offsets)
        tl.numa_store(dst1_ptr + offsets, data, numa_dst=1)
        tl.numa_store(dst2_ptr + offsets, data, numa_dst=2)
        tl.numa_store(dst3_ptr + offsets, data, numa_dst=3)
        tl.notify(signal_id=signal_base + 1, numa_dst=1)
        tl.notify(signal_id=signal_base + 2, numa_dst=2)
        tl.notify(signal_id=signal_base + 3, numa_dst=3)
    elif rank == 1:
        tl.wait(signal_id=signal_base + 1)
    elif rank == 2:
        tl.wait(signal_id=signal_base + 2)
    else:
        tl.wait(signal_id=signal_base + 3)


@triton.jit
def scatter_4node_kernel(
    src_ptr,
    dst0_ptr,
    dst1_ptr,
    dst2_ptr,
    dst3_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """Rank 0 scatters different chunks to ranks 0,1,2,3."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    src_offset = rank * BLOCK_SIZE

    if rank == 0:
        data = tl.load(src_ptr + src_offset + offsets)
        tl.store(dst0_ptr + offsets, data)
        tl.numa_store(dst1_ptr + offsets, tl.load(src_ptr + BLOCK_SIZE + offsets), numa_dst=1)
        tl.numa_store(dst2_ptr + offsets, tl.load(src_ptr + 2 * BLOCK_SIZE + offsets), numa_dst=2)
        tl.numa_store(dst3_ptr + offsets, tl.load(src_ptr + 3 * BLOCK_SIZE + offsets), numa_dst=3)
        tl.notify(signal_id=signal_base + 1, numa_dst=1)
        tl.notify(signal_id=signal_base + 2, numa_dst=2)
        tl.notify(signal_id=signal_base + 3, numa_dst=3)
    elif rank == 1:
        tl.wait(signal_id=signal_base + 1)
    elif rank == 2:
        tl.wait(signal_id=signal_base + 2)
    else:
        tl.wait(signal_id=signal_base + 3)


@triton.jit
def gather_4node_kernel(
    src0_ptr,
    src1_ptr,
    src2_ptr,
    src3_ptr,
    dst_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """Ranks 0,1,2,3 gather their chunks to rank 0's dst. dst[rank*BLOCK:(rank+1)*BLOCK] = src_rank."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    dst_offset = rank * BLOCK_SIZE

    if rank == 0:
        tl.store(dst_ptr + offsets, tl.load(src0_ptr + offsets))
        tl.wait(signal_id=signal_base + 1)
        tl.wait(signal_id=signal_base + 2)
        tl.wait(signal_id=signal_base + 3)
    elif rank == 1:
        tl.numa_store(dst_ptr + dst_offset + offsets, tl.load(src1_ptr + offsets), numa_dst=0)
        tl.notify(signal_id=signal_base + 1, numa_dst=0)
    elif rank == 2:
        tl.numa_store(dst_ptr + dst_offset + offsets, tl.load(src2_ptr + offsets), numa_dst=0)
        tl.notify(signal_id=signal_base + 2, numa_dst=0)
    else:
        tl.numa_store(dst_ptr + dst_offset + offsets, tl.load(src3_ptr + offsets), numa_dst=0)
        tl.notify(signal_id=signal_base + 3, numa_dst=0)


@triton.jit
def allgather_4node_kernel(
    src_ptr,
    dst_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """Each rank has src shard. AllGather: each writes its shard to dst[rank*BLOCK:(rank+1)*BLOCK].
    dst is on node 0; ranks 1,2,3 use numa_store to write their slot."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    my_shard = tl.load(src_ptr + offsets)
    dst_offset = rank * BLOCK_SIZE

    if rank == 0:
        tl.store(dst_ptr + dst_offset + offsets, my_shard)
    else:
        tl.numa_store(dst_ptr + dst_offset + offsets, my_shard, numa_dst=0)

    for r in range(4):
        if r != rank:
            tl.notify(signal_id=signal_base + rank, numa_dst=r)
    for r in range(4):
        if r != rank:
            tl.wait(signal_id=signal_base + r)


@triton.jit
def reduce_4node_kernel(
    src0_ptr,
    src1_ptr,
    src2_ptr,
    src3_ptr,
    dst_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """Reduce (sum) from all ranks to rank 0. Rank 0 waits, then numa_loads and sums."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if rank == 0:
        tl.wait(signal_id=signal_base + 1)
        tl.wait(signal_id=signal_base + 2)
        tl.wait(signal_id=signal_base + 3)
        acc = tl.load(src0_ptr + offsets)
        acc = acc + tl.numa_load(src1_ptr + offsets, numa_src=1)
        acc = acc + tl.numa_load(src2_ptr + offsets, numa_src=2)
        acc = acc + tl.numa_load(src3_ptr + offsets, numa_src=3)
        tl.store(dst_ptr + offsets, acc)
    else:
        tl.notify(signal_id=signal_base + rank, numa_dst=0)


@triton.jit
def allreduce_4node_kernel(
    partial0_ptr,
    partial1_ptr,
    partial2_ptr,
    partial3_ptr,
    output0_ptr,
    output1_ptr,
    output2_ptr,
    output3_ptr,
    BLOCK_SIZE: tl.constexpr,
    signal_base: tl.constexpr,
):
    """AllReduce (sum): Reduce to rank 0, then Broadcast to all."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if rank == 0:
        tl.wait(signal_id=signal_base + 1)
        tl.wait(signal_id=signal_base + 2)
        tl.wait(signal_id=signal_base + 3)
        acc = tl.load(partial0_ptr + offsets)
        acc = acc + tl.numa_load(partial1_ptr + offsets, numa_src=1)
        acc = acc + tl.numa_load(partial2_ptr + offsets, numa_src=2)
        acc = acc + tl.numa_load(partial3_ptr + offsets, numa_src=3)
        tl.store(output0_ptr + offsets, acc)
        tl.numa_store(output1_ptr + offsets, acc, numa_dst=1)
        tl.numa_store(output2_ptr + offsets, acc, numa_dst=2)
        tl.numa_store(output3_ptr + offsets, acc, numa_dst=3)
        tl.notify(signal_id=signal_base + 10, numa_dst=1)
        tl.notify(signal_id=signal_base + 11, numa_dst=2)
        tl.notify(signal_id=signal_base + 12, numa_dst=3)
    elif rank == 1:
        tl.notify(signal_id=signal_base + 1, numa_dst=0)
        tl.wait(signal_id=signal_base + 10)
    elif rank == 2:
        tl.notify(signal_id=signal_base + 2, numa_dst=0)
        tl.wait(signal_id=signal_base + 11)
    else:
        tl.notify(signal_id=signal_base + 3, numa_dst=0)
        tl.wait(signal_id=signal_base + 12)


@triton.jit
def reducescatter_4node_kernel(
    partial0_ptr,
    partial1_ptr,
    partial2_ptr,
    partial3_ptr,
    output0_ptr,
    output1_ptr,
    output2_ptr,
    output3_ptr,
    BLOCK_SIZE: tl.constexpr,
    shard_rank: tl.constexpr,
    signal_base: tl.constexpr,
):
    """ReduceScatter: rank r gets sum of partial[s][r*BLOCK:(r+1)*BLOCK] for all s.
    Each rank handles one output shard. shard_rank indicates which shard this block handles."""
    rank = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    shard_offset = shard_rank * BLOCK_SIZE

    if rank == 0:
        tl.wait(signal_id=signal_base + 1)
        tl.wait(signal_id=signal_base + 2)
        tl.wait(signal_id=signal_base + 3)
        acc = tl.load(partial0_ptr + shard_offset + offsets)
        acc = acc + tl.numa_load(partial1_ptr + shard_offset + offsets, numa_src=1)
        acc = acc + tl.numa_load(partial2_ptr + shard_offset + offsets, numa_src=2)
        acc = acc + tl.numa_load(partial3_ptr + shard_offset + offsets, numa_src=3)
        tl.store(output0_ptr + offsets, acc)
    elif rank == 1:
        tl.numa_store(output1_ptr + offsets, tl.load(partial1_ptr + shard_offset + offsets), numa_dst=1)
        tl.notify(signal_id=signal_base + 1, numa_dst=0)
    elif rank == 2:
        tl.numa_store(output2_ptr + offsets, tl.load(partial2_ptr + shard_offset + offsets), numa_dst=2)
        tl.notify(signal_id=signal_base + 2, numa_dst=0)
    else:
        tl.numa_store(output3_ptr + offsets, tl.load(partial3_ptr + shard_offset + offsets), numa_dst=3)
        tl.notify(signal_id=signal_base + 3, numa_dst=0)


# ---------------------------------------------------------------------------
# Distributed GEMM: K-split partial matmul per rank
# ---------------------------------------------------------------------------

@triton.jit
def distributed_gemm_partial_kernel(
    A_ptr,
    B_ptr,
    C0_ptr,
    C1_ptr,
    C2_ptr,
    C3_ptr,
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
    Each block computes one tile of C_partial for one (batch, rank) pair.
    program_id(0) = tile index, program_id(1) = batch, program_id(2) = rank.
    Rank r computes A[:, K_offset:K_offset+K_rank] @ B[K_offset:K_offset+K_rank, :]
    """
    GROUP_M = 8
    pid = tl.program_id(0)
    idx_q = tl.program_id(1)
    rank = tl.program_id(2)

    K_offset = rank * (K // 4)
    K_rank = K // 4

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = rm % M
    rbn = rn % N

    A = A_ptr + (ram[:, None] * stride_am + (rk[None, :] + K_offset) * stride_ak + idx_q * stride_aq)
    B = B_ptr + ((rk[:, None] + K_offset) * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K_rank, 0, -BLOCK_K):
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

    if rank == 0:
        tl.store(C0_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)
    elif rank == 1:
        tl.store(C1_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)
    elif rank == 2:
        tl.store(C2_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)
    else:
        tl.store(C3_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)


# ---------------------------------------------------------------------------
# Tiled reduce: sum C_partial across ranks for each output tile
# ---------------------------------------------------------------------------

@triton.jit
def reduce_tile_4node_kernel(
    c0_ptr,
    c1_ptr,
    c2_ptr,
    c3_ptr,
    out_ptr,
    batch_idx: tl.constexpr,
    pid_m: tl.constexpr,
    pid_n: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    signal_base: tl.constexpr,
):
    """Reduce one tile: out = c0 + c1 + c2 + c3. Tile at (batch, pid_m, pid_n).
    Uses Reduce collective (wait/notify). Grid (4,) - one block per rank."""
    rank = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    base = batch_idx * M * N
    offsets = idx_m * N + idx_n + base

    if rank == 0:
        tl.wait(signal_id=signal_base + 1)
        tl.wait(signal_id=signal_base + 2)
        tl.wait(signal_id=signal_base + 3)
        acc = tl.load(c0_ptr + offsets, mask=mask, other=0.0)
        acc = acc + tl.numa_load(c1_ptr + offsets, numa_src=1, mask=mask, other=0.0)
        acc = acc + tl.numa_load(c2_ptr + offsets, numa_src=2, mask=mask, other=0.0)
        acc = acc + tl.numa_load(c3_ptr + offsets, numa_src=3, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, acc, mask=mask)
    elif rank == 1:
        tl.notify(signal_id=signal_base + 1, numa_dst=0)
    elif rank == 2:
        tl.notify(signal_id=signal_base + 2, numa_dst=0)
    else:
        tl.notify(signal_id=signal_base + 3, numa_dst=0)


@triton.jit
def reduce_tiles_batched_kernel(
    c0_ptr,
    c1_ptr,
    c2_ptr,
    c3_ptr,
    out_ptr,
    tile_start: tl.constexpr,
    num_tiles: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce num_tiles starting at tile_start. Single block, no sync (launch boundary)."""
    for t in range(num_tiles):
        tile_idx = tile_start + t
        pid_n = tile_idx % grid_n
        pid_m = (tile_idx // grid_n) % grid_m
        batch_idx = tile_idx // (grid_m * grid_n)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        base = batch_idx * M * N
        offsets = idx_m * N + idx_n + base

        acc = tl.load(c0_ptr + offsets, mask=mask, other=0.0)
        acc = acc + tl.numa_load(c1_ptr + offsets, numa_src=1, mask=mask, other=0.0)
        acc = acc + tl.numa_load(c2_ptr + offsets, numa_src=2, mask=mask, other=0.0)
        acc = acc + tl.numa_load(c3_ptr + offsets, numa_src=3, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, acc, mask=mask)


def _bmm_grid(batch, meta):
    """Match torch._inductor.kernel.bmm.bmm_grid(8, 2048, 2048, meta)."""
    grid_m = (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
    grid_n = (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
    GROUP_M = 8
    num_tiles = grid_m * grid_n
    return (num_tiles, batch, NUM_RANKS)


def distributed_bmm(A, B, C_partial_list, C_out, stream=None):
    """
    Distributed BMM: (batch, M, K) @ (batch, K, N) = (batch, M, N).
    A: (batch, 2048, 64), B: (batch, 64, 2048), C_out: (batch, 2048, 2048).
    C_partial_list: [C0, C1, C2, C3] - one per NUMA node.
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

    distributed_gemm_partial_kernel[grid](
        A, B,
        C_partial_list[0], C_partial_list[1], C_partial_list[2], C_partial_list[3],
        M=M, N=N, K=K,
        stride_aq=stride_aq, stride_am=stride_am, stride_ak=stride_ak,
        stride_bq=stride_bq, stride_bk=stride_bk, stride_bn=stride_bn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles_per_batch = grid_m * grid_n
    total_tiles = num_tiles_per_batch * batch

    # Batched reduce: process 256 tiles per launch (avoids 8192 kernel launches)
    TILES_PER_BATCH = 256
    for tile_start in range(0, total_tiles, TILES_PER_BATCH):
        num_tiles = min(TILES_PER_BATCH, total_tiles - tile_start)
        reduce_tiles_batched_kernel[(1,)](
            C_partial_list[0], C_partial_list[1], C_partial_list[2], C_partial_list[3],
            C_out,
            tile_start=tile_start,
            num_tiles=num_tiles,
            grid_m=grid_m,
            grid_n=grid_n,
            M=M, N=N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
