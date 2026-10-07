# Exact six-neighbour frontier search on the GPU: multi-source breadth-first search over the pore voxels,
# ties to the lower site index. gpu/code/flow_runner.py takes the ownership labels of the GPU forward runs
# from exact_frontier_dijkstra_gpu_6.

import numpy as np
import cupy as cp
import time
import math


EXACT_FRONTIER_LAST_WALL_TIME = 0.0
EXACT_FRONTIER_LAST_TINIT_WALL = 0.0
EXACT_FRONTIER_LAST_TITER_WALL = 0.0
EXACT_FRONTIER_LAST_ITERS = 0
EXACT_FRONTIER_LAST_MAX_FRONTIER = 0
EXACT_FRONTIER_LAST_MAX_DIST = 0
_EXACT_FRONTIER_BFS6_KERNEL = None
_EXACT_FRONTIER_BFS6_INIT_KERNEL = None
_EXACT_FRONTIER_BFS6_SEED_KERNEL = None


def build_exact_frontier_bfs6_init_kernel():
    """Build the initialization kernel for exact frontier BFS on a 6-neighbour grid."""
    code = r'''
    extern "C" __global__
    void exact_frontier_bfs6_init(
        const unsigned char* mask,
        const int* seeds,
        int* dist,
        int* label,
        int* frontier,
        int* bad_seed,
        int n_total,
        int n_seed,
        int D,
        int H,
        int W,
        int inf_label
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_total) {
            dist[tid] = -1;
            label[tid] = mask[tid] ? inf_label : -1;
        }
        if (tid < n_seed) {
            int z = seeds[3 * tid + 0];
            int y = seeds[3 * tid + 1];
            int x = seeds[3 * tid + 2];
            if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) {
                atomicExch(bad_seed, 1);
                return;
            }
            int idx = (z * H + y) * W + x;
            if (!mask[idx]) {
                atomicExch(bad_seed, 1);
                return;
            }
            dist[idx] = 0;
            label[idx] = tid;
            frontier[tid] = idx;
        }
    }
    '''
    return cp.RawKernel(code, "exact_frontier_bfs6_init")


def build_exact_frontier_bfs6_kernel():
    """Build the exact multi-source frontier kernel for a 6-neighbour unit graph."""
    code = r'''
    extern "C" __global__
    void exact_frontier_bfs6_step(
        const unsigned char* mask,
        int* dist,
        int* label,
        const int* frontier,
        int frontier_n,
        int* next_frontier,
        int* next_n,
        int D,
        int H,
        int W,
        int step
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= frontier_n) return;

        int idx = frontier[tid];
        int lab = label[idx];
        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int neigh[6];
        int ncnt = 0;
        if (z > 0)     neigh[ncnt++] = idx - HW;
        if (z + 1 < D) neigh[ncnt++] = idx + HW;
        if (y > 0)     neigh[ncnt++] = idx - W;
        if (y + 1 < H) neigh[ncnt++] = idx + W;
        if (x > 0)     neigh[ncnt++] = idx - 1;
        if (x + 1 < W) neigh[ncnt++] = idx + 1;

        for (int q = 0; q < ncnt; ++q) {
            int nb = neigh[q];
            if (!mask[nb]) continue;
            int old = atomicCAS(&dist[nb], -1, step);
            if (old == -1) {
                atomicMin(&label[nb], lab);
                int pos = atomicAdd(next_n, 1);
                next_frontier[pos] = nb;
            } else if (old == step) {
                atomicMin(&label[nb], lab);
            }
        }
    }
    '''
    return cp.RawKernel(code, "exact_frontier_bfs6_step")


def build_exact_frontier_bfs6_seed_kernel():
    """Build the seed kernel for exact frontier BFS."""
    code = r'''
    extern "C" __global__
    void exact_frontier_bfs6_seed(
        const unsigned char* mask,
        const int* seeds,
        int* dist,
        int* label,
        int* frontier,
        int* bad_seed,
        int n_seed,
        int D,
        int H,
        int W
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_seed) return;
        int z = seeds[3 * tid + 0];
        int y = seeds[3 * tid + 1];
        int x = seeds[3 * tid + 2];
        if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) {
            atomicExch(bad_seed, 1);
            return;
        }
        int idx = (z * H + y) * W + x;
        if (!mask[idx]) {
            atomicExch(bad_seed, 1);
            return;
        }
        dist[idx] = 0;
        label[idx] = tid;
        frontier[tid] = idx;
    }
    '''
    return cp.RawKernel(code, "exact_frontier_bfs6_seed")


def exact_frontier_dijkstra_gpu_6(
    mask,
    seeds_zyx,
    *,
    kernel=None,
    init_kernel=None,
    block_size=256,
    validate=True,
    collect_stats=True,
    return_float64=True,
):
    """Exact multi-source Dijkstra for the reported 6-neighbour unit graph.

    For unit edge weights, Dijkstra is equivalent to multi-source BFS. This
    backend uses a sparse frontier list and deterministic minimum-source tie
    handling.
    """
    global _EXACT_FRONTIER_BFS6_KERNEL, _EXACT_FRONTIER_BFS6_INIT_KERNEL, _EXACT_FRONTIER_BFS6_SEED_KERNEL
    global EXACT_FRONTIER_LAST_WALL_TIME, EXACT_FRONTIER_LAST_TINIT_WALL
    global EXACT_FRONTIER_LAST_TITER_WALL, EXACT_FRONTIER_LAST_ITERS
    global EXACT_FRONTIER_LAST_MAX_FRONTIER, EXACT_FRONTIER_LAST_MAX_DIST

    EXACT_FRONTIER_LAST_WALL_TIME = 0.0
    EXACT_FRONTIER_LAST_TINIT_WALL = 0.0
    EXACT_FRONTIER_LAST_TITER_WALL = 0.0
    EXACT_FRONTIER_LAST_ITERS = 0
    EXACT_FRONTIER_LAST_MAX_FRONTIER = 0
    EXACT_FRONTIER_LAST_MAX_DIST = 0

    if kernel is None:
        if _EXACT_FRONTIER_BFS6_KERNEL is None:
            _EXACT_FRONTIER_BFS6_KERNEL = build_exact_frontier_bfs6_kernel()
        kernel = _EXACT_FRONTIER_BFS6_KERNEL
    if init_kernel is None:
        if _EXACT_FRONTIER_BFS6_INIT_KERNEL is None:
            _EXACT_FRONTIER_BFS6_INIT_KERNEL = build_exact_frontier_bfs6_init_kernel()
        init_kernel = _EXACT_FRONTIER_BFS6_INIT_KERNEL
    if _EXACT_FRONTIER_BFS6_SEED_KERNEL is None:
        _EXACT_FRONTIER_BFS6_SEED_KERNEL = build_exact_frontier_bfs6_seed_kernel()
    seed_kernel = _EXACT_FRONTIER_BFS6_SEED_KERNEL

    total_t0 = time.perf_counter()
    init_t0 = time.perf_counter()
    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    if mask_u8.ndim != 3:
        raise ValueError("exact_frontier_dijkstra_gpu_6 expects a 3D mask")
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")

    D, H, W = [int(v) for v in mask_u8.shape]
    n_total = int(mask_u8.size)
    n_seed = int(seeds.shape[0])
    flat_mask = mask_u8.ravel()

    inf_label = np.int32(1_073_741_823)
    dist_i = cp.empty(n_total, dtype=cp.int32)
    label_i = cp.empty(n_total, dtype=cp.int32)
    frontier = cp.empty(n_total, dtype=cp.int32)
    next_frontier = cp.empty(n_total, dtype=cp.int32)
    next_n = cp.empty(1, dtype=cp.int32)
    bad_seed = cp.zeros(1, dtype=cp.int32)

    block = int(block_size)
    init_grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    init_kernel(
        init_grid,
        (block,),
        (
            flat_mask,
            seeds,
            dist_i,
            label_i,
            frontier,
            bad_seed,
            int(n_total),
            int(0),
            int(D),
            int(H),
            int(W),
            int(inf_label),
        ),
    )
    seed_grid = (max(1, int(math.ceil(float(n_seed) / float(block)))),)
    seed_kernel(
        seed_grid,
        (block,),
        (
            flat_mask,
            seeds,
            dist_i,
            label_i,
            frontier,
            bad_seed,
            int(n_seed),
            int(D),
            int(H),
            int(W),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    if validate and int(bad_seed.get()[0]) != 0:
        raise RuntimeError("exact_frontier_dijkstra_gpu_6 received an invalid or non-fluid seed")
    EXACT_FRONTIER_LAST_TINIT_WALL = float(time.perf_counter() - init_t0)

    iter_t0 = time.perf_counter()
    current_n = n_seed
    max_frontier = current_n
    step = 1
    while current_n > 0:
        next_n.fill(0)
        grid = (max(1, int(math.ceil(float(current_n) / float(block)))),)
        kernel(
            grid,
            (block,),
            (
                flat_mask,
                dist_i,
                label_i,
                frontier,
                int(current_n),
                next_frontier,
                next_n,
                D,
                H,
                W,
                int(step),
            ),
        )
        current_n = int(next_n.get()[0])
        max_frontier = max(max_frontier, current_n)
        frontier, next_frontier = next_frontier, frontier
        step += 1
        if step > n_total:
            raise RuntimeError("exact_frontier_dijkstra_gpu_6 exceeded node count")
    cp.cuda.Stream.null.synchronize()
    EXACT_FRONTIER_LAST_TITER_WALL = float(time.perf_counter() - iter_t0)
    EXACT_FRONTIER_LAST_ITERS = int(step - 1)
    EXACT_FRONTIER_LAST_MAX_FRONTIER = int(max_frontier)

    if validate:
        assigned = int(cp.count_nonzero((flat_mask != 0) & (dist_i >= 0)).get())
        expected = int(cp.count_nonzero(flat_mask != 0).get())
        if assigned != expected:
            raise RuntimeError(f"exact frontier assigned {assigned} fluid voxels, expected {expected}")
    if collect_stats:
        max_dist = int(cp.max(dist_i[flat_mask != 0]).get())
    else:
        max_dist = max(0, int(step) - 2)
    EXACT_FRONTIER_LAST_MAX_DIST = int(max_dist)

    label_i = cp.where(label_i == inf_label, cp.int32(-1), label_i)
    dist_dtype = cp.float64 if return_float64 else cp.float32
    dist = cp.where(dist_i >= 0, dist_i.astype(dist_dtype), cp.inf)
    EXACT_FRONTIER_LAST_WALL_TIME = float(time.perf_counter() - total_t0)
    return label_i.reshape(mask_u8.shape).astype(cp.int32, copy=False), dist.reshape(mask_u8.shape)
