from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


INF_DISTANCE = np.uint32(0x3FFFFFFF)
INF_LABEL = np.uint32(0xFFFFFFFF)
INF_PACK = np.uint64((int(INF_DISTANCE) << 32) | int(INF_LABEL))
UNIT_PACK = np.uint64(1 << 32)


EXACT_FRONTIER_LAST_ITERS = 0
EXACT_FRONTIER_LAST_MAX_DIST = 0
EXACT_PERSISTENT_BFS6_LAST_STATS: dict[str, Any] = {}
EXACT_PERSISTENT6_LAST_STATS: dict[str, Any] = {}

CERTIFIED_ROI6_LAST_STATS: dict[str, Any] = {}

COOPERATIVE_RELAX_CHUNK = 512

_KERNELS: dict[tuple[str, bool], Any] = {}


def _kernel(name: str, code: str, *, cooperative: bool = False):
    import cupy as cp

    key = (name, cooperative)
    cached = _KERNELS.get(key)
    if cached is None:
        cached = cp.RawKernel(
            code,
            name,
            options=("-std=c++17",) if cooperative else ("-std=c++11",),
            enable_cooperative_groups=cooperative,
        )
        _KERNELS[key] = cached
    return cached


def _exact_init_kernel():
    return _kernel(
        "ptv_exact_bfs6_init",
        r'''
        extern "C" __global__
        void ptv_exact_bfs6_init(
            const unsigned char* mask,
            int* distance,
            int* label,
            const int nvox,
            const int inf_label
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            distance[idx] = -1;
            label[idx] = mask[idx] ? inf_label : -1;
        }
        ''',
    )


def _exact_seed_kernel():
    return _kernel(
        "ptv_exact_bfs6_seed",
        r'''
        extern "C" __global__
        void ptv_exact_bfs6_seed(
            const unsigned char* mask,
            const int* seeds,
            int* distance,
            int* label,
            int* frontier,
            int* bad_seed,
            const int nseed,
            const int D,
            const int H,
            const int W
        ) {
            int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (tid >= nseed) return;
            int z = seeds[3 * tid + 0];
            int y = seeds[3 * tid + 1];
            int x = seeds[3 * tid + 2];
            if ((unsigned)z >= (unsigned)D ||
                (unsigned)y >= (unsigned)H ||
                (unsigned)x >= (unsigned)W) {
                atomicExch(bad_seed, 1);
                return;
            }
            int idx = (z * H + y) * W + x;
            if (!mask[idx]) {
                atomicExch(bad_seed, 1);
                return;
            }
            distance[idx] = 0;
            atomicMin(&label[idx], tid);
            frontier[tid] = idx;
        }
        ''',
    )


def _exact_step_kernel():
    return _kernel(
        "ptv_exact_bfs6_step",
        r'''
        extern "C" __global__
        void ptv_exact_bfs6_step(
            const unsigned char* mask,
            int* distance,
            int* label,
            const int* frontier,
            const int frontier_n,
            int* next_frontier,
            int* next_n,
            const int D,
            const int H,
            const int W,
            const int step
        ) {
            int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (tid >= frontier_n) return;

            int idx = frontier[tid];
            int owner = label[idx];
            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;

            int neighbor[6];
            int count = 0;
            if (z > 0) neighbor[count++] = idx - HW;
            if (z + 1 < D) neighbor[count++] = idx + HW;
            if (y > 0) neighbor[count++] = idx - W;
            if (y + 1 < H) neighbor[count++] = idx + W;
            if (x > 0) neighbor[count++] = idx - 1;
            if (x + 1 < W) neighbor[count++] = idx + 1;

            for (int q = 0; q < count; ++q) {
                int nb = neighbor[q];
                if (!mask[nb]) continue;
                int old = atomicCAS(&distance[nb], -1, step);
                if (old == -1) {
                    atomicMin(&label[nb], owner);
                    int pos = atomicAdd(next_n, 1);
                    next_frontier[pos] = nb;
                } else if (old == step) {
                    atomicMin(&label[nb], owner);
                }
            }
        }
        ''',
    )


def exact_frontier_dijkstra_gpu_6(
    mask,
    seeds_zyx,
    *,
    block_size: int = 256,
    validate: bool = True,
    return_float64: bool = False,
):
    """Exact deterministic multi-source BFS on the unit six-neighbour pore graph."""
    import cupy as cp

    global EXACT_FRONTIER_LAST_ITERS, EXACT_FRONTIER_LAST_MAX_DIST

    mask_u8 = cp.asarray(mask, dtype=cp.uint8)
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")

    D, H, W = (int(value) for value in mask_u8.shape)
    nvox = int(mask_u8.size)
    nseed = int(seeds.shape[0])
    if nseed == 0:
        raise ValueError("at least one prescribed site is required")

    flat_mask = mask_u8.ravel()
    distance = cp.empty(nvox, dtype=cp.int32)
    label = cp.empty(nvox, dtype=cp.int32)
    frontier = cp.empty(nvox, dtype=cp.int32)
    next_frontier = cp.empty_like(frontier)
    next_n = cp.empty(1, dtype=cp.int32)
    bad_seed = cp.zeros(1, dtype=cp.int32)

    block = int(block_size)
    grid_vox = (max(1, math.ceil(nvox / block)),)
    grid_seed = (max(1, math.ceil(nseed / block)),)
    inf_label = np.int32(1_073_741_823)
    _exact_init_kernel()(
        grid_vox,
        (block,),
        (flat_mask, distance, label, np.int32(nvox), inf_label),
    )
    _exact_seed_kernel()(
        grid_seed,
        (block,),
        (
            flat_mask,
            seeds,
            distance,
            label,
            frontier,
            bad_seed,
            np.int32(nseed),
            np.int32(D),
            np.int32(H),
            np.int32(W),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    if validate and int(bad_seed.get()[0]) != 0:
        raise RuntimeError("an input site is outside the pore mask")

    current_n = nseed
    step = 1
    while current_n > 0:
        next_n.fill(0)
        grid = (max(1, math.ceil(current_n / block)),)
        _exact_step_kernel()(
            grid,
            (block,),
            (
                flat_mask,
                distance,
                label,
                frontier,
                np.int32(current_n),
                next_frontier,
                next_n,
                np.int32(D),
                np.int32(H),
                np.int32(W),
                np.int32(step),
            ),
        )
        current_n = int(next_n.get()[0])
        frontier, next_frontier = next_frontier, frontier
        step += 1
        if step > nvox:
            raise RuntimeError("six-neighbour frontier exceeded the voxel count")

    EXACT_FRONTIER_LAST_ITERS = int(step - 1)
    EXACT_FRONTIER_LAST_MAX_DIST = max(0, int(step - 2))
    if validate:
        assigned = int(cp.count_nonzero((flat_mask != 0) & (distance >= 0)).get())
        expected = int(cp.count_nonzero(flat_mask != 0).get())
        if assigned != expected:
            raise RuntimeError(
                f"six-neighbour frontier assigned {assigned} of {expected} pore voxels"
            )

    label = cp.where(label == inf_label, cp.int32(-1), label)
    distance_dtype = cp.float64 if return_float64 else cp.float32
    distance_out = cp.where(distance >= 0, distance.astype(distance_dtype), cp.inf)
    return label.reshape(mask_u8.shape), distance_out.reshape(mask_u8.shape)


def _packed_init_kernel():
    return _kernel(
        "ptv_packed_init",
        r'''
        extern "C" __global__
        void ptv_packed_init(
            unsigned long long* state,
            const int nvox,
            const unsigned long long inf_pack
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx < nvox) state[idx] = inf_pack;
        }
        ''',
    )


def _packed_seed_kernel():
    return _kernel(
        "ptv_packed_seed",
        r'''
        extern "C" __global__
        void ptv_packed_seed(
            const unsigned char* mask,
            const int* seeds,
            unsigned long long* state,
            int* seed_label,
            int* bad_seed,
            const int nseed,
            const int D,
            const int H,
            const int W
        ) {
            int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (tid >= nseed) return;
            int z = seeds[3 * tid + 0];
            int y = seeds[3 * tid + 1];
            int x = seeds[3 * tid + 2];
            if ((unsigned)z >= (unsigned)D ||
                (unsigned)y >= (unsigned)H ||
                (unsigned)x >= (unsigned)W) {
                atomicExch(bad_seed, 1);
                return;
            }
            int idx = (z * H + y) * W + x;
            if (!mask[idx]) {
                atomicExch(bad_seed, 1);
                return;
            }
            unsigned long long packed = (unsigned long long)(unsigned int)tid;
            atomicMin(state + idx, packed);
            atomicMin(seed_label + idx, tid);
        }
        ''',
    )


def _packed_seed_frontier_kernel():
    return _kernel(
        "ptv_packed_seed_frontier6",
        r'''
        extern "C" __global__
        void ptv_packed_seed_frontier6(
            const unsigned char* mask,
            const int* seeds,
            unsigned long long* state,
            int* seed_label,
            int* frontier,
            int* frontier_n,
            int* bad_seed,
            const int nseed,
            const int D,
            const int H,
            const int W,
            const unsigned long long inf_pack
        ) {
            int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (tid >= nseed) return;
            int z = seeds[3 * tid + 0];
            int y = seeds[3 * tid + 1];
            int x = seeds[3 * tid + 2];
            if ((unsigned)z >= (unsigned)D ||
                (unsigned)y >= (unsigned)H ||
                (unsigned)x >= (unsigned)W) {
                atomicExch(bad_seed, 1);
                return;
            }
            int idx = (z * H + y) * W + x;
            if (!mask[idx]) {
                atomicExch(bad_seed, 1);
                return;
            }
            unsigned long long packed = (unsigned long long)(unsigned int)tid;
            unsigned long long old = atomicMin(state + idx, packed);
            atomicMin(seed_label + idx, tid);
            if (old == inf_pack) {
                int pos = atomicAdd(frontier_n, 1);
                frontier[pos] = idx;
            }
        }
        ''',
    )


def _l1_scan_x_kernel():
    return _kernel(
        "ptv_l1_scan_x",
        r'''
        extern "C" __global__
        void ptv_l1_scan_x(
            unsigned long long* state,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            int line = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            int nline = D * H;
            if (line >= nline) return;
            int base = line * W;
            const unsigned long long unit = 1ULL << 32;
            for (int x = 1; x < W; ++x) {
                unsigned long long prev = state[base + x - 1];
                if ((unsigned int)(prev >> 32) < inf_distance) {
                    unsigned long long cand = prev + unit;
                    if (cand < state[base + x]) state[base + x] = cand;
                }
            }
            for (int x = W - 2; x >= 0; --x) {
                unsigned long long prev = state[base + x + 1];
                if ((unsigned int)(prev >> 32) < inf_distance) {
                    unsigned long long cand = prev + unit;
                    if (cand < state[base + x]) state[base + x] = cand;
                }
            }
        }
        ''',
    )


def _l1_scan_y_kernel():
    return _kernel(
        "ptv_l1_scan_y",
        r'''
        extern "C" __global__
        void ptv_l1_scan_y(
            unsigned long long* state,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            int line = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            int nline = D * W;
            if (line >= nline) return;
            int z = line / W;
            int x = line - z * W;
            int base = z * H * W + x;
            const unsigned long long unit = 1ULL << 32;
            for (int y = 1; y < H; ++y) {
                int idx = base + y * W;
                unsigned long long prev = state[idx - W];
                if ((unsigned int)(prev >> 32) < inf_distance) {
                    unsigned long long cand = prev + unit;
                    if (cand < state[idx]) state[idx] = cand;
                }
            }
            for (int y = H - 2; y >= 0; --y) {
                int idx = base + y * W;
                unsigned long long prev = state[idx + W];
                if ((unsigned int)(prev >> 32) < inf_distance) {
                    unsigned long long cand = prev + unit;
                    if (cand < state[idx]) state[idx] = cand;
                }
            }
        }
        ''',
    )


def _l1_scan_z_kernel():
    return _kernel(
        "ptv_l1_scan_z",
        r'''
        extern "C" __global__
        void ptv_l1_scan_z(
            unsigned long long* state,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            int line = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            int HW = H * W;
            if (line >= HW) return;
            const unsigned long long unit = 1ULL << 32;
            for (int z = 1; z < D; ++z) {
                int idx = z * HW + line;
                unsigned long long prev = state[idx - HW];
                if ((unsigned int)(prev >> 32) < inf_distance) {
                    unsigned long long cand = prev + unit;
                    if (cand < state[idx]) state[idx] = cand;
                }
            }
            for (int z = D - 2; z >= 0; --z) {
                int idx = z * HW + line;
                unsigned long long prev = state[idx + HW];
                if ((unsigned int)(prev >> 32) < inf_distance) {
                    unsigned long long cand = prev + unit;
                    if (cand < state[idx]) state[idx] = cand;
                }
            }
        }
        ''',
    )


def _l1_jump_axis_kernel():
    return _kernel(
        "ptv_l1_jump_axis",
        r'''
        extern "C" __global__
        void ptv_l1_jump_axis(
            const unsigned long long* state_in,
            unsigned long long* state_out,
            const int nvox,
            const int D,
            const int H,
            const int W,
            const int axis,
            const int jump,
            const unsigned int inf_distance
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;
            unsigned long long best = state_in[idx];
            unsigned long long increment = ((unsigned long long)(unsigned int)jump) << 32;

            for (int sign = -1; sign <= 1; sign += 2) {
                int nz = z;
                int ny = y;
                int nx = x;
                if (axis == 0) nz += sign * jump;
                else if (axis == 1) ny += sign * jump;
                else nx += sign * jump;
                if ((unsigned)nz >= (unsigned)D ||
                    (unsigned)ny >= (unsigned)H ||
                    (unsigned)nx >= (unsigned)W) continue;
                int source_idx = (nz * H + ny) * W + nx;
                unsigned long long source = state_in[source_idx];
                if ((unsigned int)(source >> 32) >= inf_distance) continue;
                unsigned long long candidate = source + increment;
                if (candidate < best) best = candidate;
            }
            state_out[idx] = best;
        }
        ''',
    )


def _l1_fused_jump_axis_kernel():
    return _kernel(
        "ptv_l1_fused_jump_axis",
        r'''
        extern "C" __global__
        void ptv_l1_fused_jump_axis(
            unsigned long long* state,
            const int D,
            const int H,
            const int W,
            const int axis,
            const unsigned int inf_distance,
            const unsigned long long inf_pack
        ) {
            extern __shared__ unsigned long long line_state[];
            int position = (int)threadIdx.x;
            int line = (int)blockIdx.x;
            int HW = H * W;
            int length = axis == 0 ? D : axis == 1 ? H : W;
            int nline = axis == 0 ? H * W : axis == 1 ? D * W : D * H;
            if (line >= nline) return;

            int idx = -1;
            if (position < length) {
                if (axis == 0) {
                    idx = position * HW + line;
                } else if (axis == 1) {
                    int z = line / W;
                    int x = line - z * W;
                    idx = z * HW + position * W + x;
                } else {
                    int z = line / H;
                    int y = line - z * H;
                    idx = (z * H + y) * W + position;
                }
                line_state[position] = state[idx];
            } else {
                line_state[position] = inf_pack;
            }
            __syncthreads();

            for (int jump = 1; jump < length; jump <<= 1) {
                unsigned long long best = line_state[position];
                unsigned long long increment =
                    ((unsigned long long)(unsigned int)jump) << 32;
                int lower = position - jump;
                int upper = position + jump;
                if (lower >= 0) {
                    unsigned long long source = line_state[lower];
                    if ((unsigned int)(source >> 32) < inf_distance) {
                        unsigned long long candidate = source + increment;
                        if (candidate < best) best = candidate;
                    }
                }
                if (upper < length) {
                    unsigned long long source = line_state[upper];
                    if ((unsigned int)(source >> 32) < inf_distance) {
                        unsigned long long candidate = source + increment;
                        if (candidate < best) best = candidate;
                    }
                }
                __syncthreads();
                line_state[position] = best;
                __syncthreads();
            }

            if (position < length) state[idx] = line_state[position];
        }
        ''',
    )


def _l1_cooperative_fused_jump_kernel():
    return _kernel(
        "ptv_l1_cooperative_fused_jump",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_l1_cooperative_fused_jump(
            unsigned long long* state,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance,
            const unsigned long long inf_pack
        ) {
            cg::grid_group grid = cg::this_grid();
            extern __shared__ unsigned long long line_state[];
            int position = (int)threadIdx.x;
            int HW = H * W;

            for (int axis = 0; axis < 3; ++axis) {
                int length = axis == 0 ? D : axis == 1 ? H : W;
                int nline = axis == 0 ? H * W : axis == 1 ? D * W : D * H;

                for (int line = (int)blockIdx.x;
                     line < nline;
                     line += (int)gridDim.x) {
                    int idx = -1;
                    if (position < length) {
                        if (axis == 0) {
                            idx = position * HW + line;
                        } else if (axis == 1) {
                            int z = line / W;
                            int x = line - z * W;
                            idx = z * HW + position * W + x;
                        } else {
                            int z = line / H;
                            int y = line - z * H;
                            idx = (z * H + y) * W + position;
                        }
                        line_state[position] = state[idx];
                    } else {
                        line_state[position] = inf_pack;
                    }
                    __syncthreads();

                    for (int jump = 1; jump < length; jump <<= 1) {
                        unsigned long long best = line_state[position];
                        unsigned long long increment =
                            ((unsigned long long)(unsigned int)jump) << 32;
                        int lower = position - jump;
                        int upper = position + jump;
                        if (lower >= 0) {
                            unsigned long long source = line_state[lower];
                            if ((unsigned int)(source >> 32) < inf_distance) {
                                unsigned long long candidate = source + increment;
                                if (candidate < best) best = candidate;
                            }
                        }
                        if (upper < length) {
                            unsigned long long source = line_state[upper];
                            if ((unsigned int)(source >> 32) < inf_distance) {
                                unsigned long long candidate = source + increment;
                                if (candidate < best) best = candidate;
                            }
                        }
                        __syncthreads();
                        line_state[position] = best;
                        __syncthreads();
                    }

                    if (position < length) state[idx] = line_state[position];
                    __syncthreads();
                }
                grid.sync();
            }
        }
        ''',
        cooperative=True,
    )


def _l1_cooperative_warp64_init_jump_kernel():
    return _kernel(
        "ptv_l1_cooperative_warp64_init_jump",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        __device__ __forceinline__
        unsigned long long ptv_warp64_add_jump(
            const unsigned long long source,
            const unsigned int jump,
            const unsigned int inf_distance
        ) {
            if ((unsigned int)(source >> 32) >= inf_distance) {
                return 0xFFFFFFFFFFFFFFFFULL;
            }
            return source + ((unsigned long long)jump << 32);
        }

        extern "C" __global__
        void ptv_l1_cooperative_warp64_init_jump(
            const unsigned char* mask,
            const int* seeds,
            unsigned long long* lower_state,
            unsigned long long* closure_state,
            int* bad_seed,
            const int nvox,
            const int nseed,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance,
            const unsigned long long inf_pack
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();

            for (unsigned long long idx = rank;
                 idx < (unsigned long long)nvox;
                 idx += stride) {
                lower_state[idx] = inf_pack;
                closure_state[idx] = inf_pack;
            }
            grid.sync();

            for (unsigned long long tid = rank;
                 tid < (unsigned long long)nseed;
                 tid += stride) {
                int z = seeds[3 * tid + 0];
                int y = seeds[3 * tid + 1];
                int x = seeds[3 * tid + 2];
                if ((unsigned)z >= (unsigned)D ||
                    (unsigned)y >= (unsigned)H ||
                    (unsigned)x >= (unsigned)W) {
                    atomicExch(bad_seed, 1);
                    continue;
                }
                int idx = (z * H + y) * W + x;
                if (!mask[idx]) {
                    atomicExch(bad_seed, 1);
                    continue;
                }
                atomicMin(
                    lower_state + idx,
                    (unsigned long long)(unsigned int)tid
                );
            }
            grid.sync();

            const int lane = (int)threadIdx.x & 31;
            const int warp_in_block = (int)threadIdx.x >> 5;
            const int warps_per_block = (int)blockDim.x >> 5;
            const int global_warp =
                (int)blockIdx.x * warps_per_block + warp_in_block;
            const int warp_stride = (int)gridDim.x * warps_per_block;
            const int HW = H * W;
            const unsigned int full_mask = 0xFFFFFFFFU;

            for (int axis = 0; axis < 3; ++axis) {
                int nline = axis == 0 ? H * W : axis == 1 ? D * W : D * H;
                for (int line = global_warp;
                     line < nline;
                     line += warp_stride) {
                    int p0 = lane;
                    int p1 = lane + 32;
                    int idx0;
                    int idx1;
                    if (axis == 0) {
                        idx0 = p0 * HW + line;
                        idx1 = p1 * HW + line;
                    } else if (axis == 1) {
                        int z = line / W;
                        int x = line - z * W;
                        idx0 = z * HW + p0 * W + x;
                        idx1 = z * HW + p1 * W + x;
                    } else {
                        int z = line / H;
                        int y = line - z * H;
                        idx0 = (z * H + y) * W + p0;
                        idx1 = (z * H + y) * W + p1;
                    }

                    unsigned long long value0 = lower_state[idx0];
                    unsigned long long value1 = lower_state[idx1];

                    #pragma unroll
                    for (int jump = 1; jump < 32; jump <<= 1) {
                        unsigned long long old0 = value0;
                        unsigned long long old1 = value1;
                        unsigned long long source;
                        unsigned long long candidate;
                        unsigned long long old0_up = __shfl_up_sync(
                            full_mask, old0, jump
                        );
                        unsigned long long old0_down = __shfl_down_sync(
                            full_mask, old0, jump
                        );
                        unsigned long long old1_up = __shfl_up_sync(
                            full_mask, old1, jump
                        );
                        unsigned long long old1_down = __shfl_down_sync(
                            full_mask, old1, jump
                        );
                        unsigned long long old1_cross_upper = __shfl_sync(
                            full_mask, old1, (lane + jump) & 31
                        );
                        unsigned long long old0_cross_lower = __shfl_sync(
                            full_mask, old0, (lane + 32 - jump) & 31
                        );

                        if (lane >= jump) {
                            source = old0_up;
                            candidate = ptv_warp64_add_jump(
                                source, (unsigned int)jump, inf_distance
                            );
                            if (candidate < value0) value0 = candidate;
                        }
                        if (lane + jump < 32) {
                            source = old0_down;
                        } else {
                            source = old1_cross_upper;
                        }
                        candidate = ptv_warp64_add_jump(
                            source, (unsigned int)jump, inf_distance
                        );
                        if (candidate < value0) value0 = candidate;

                        if (lane >= jump) {
                            source = old1_up;
                        } else {
                            source = old0_cross_lower;
                        }
                        candidate = ptv_warp64_add_jump(
                            source, (unsigned int)jump, inf_distance
                        );
                        if (candidate < value1) value1 = candidate;
                        if (lane + jump < 32) {
                            source = old1_down;
                            candidate = ptv_warp64_add_jump(
                                source, (unsigned int)jump, inf_distance
                            );
                            if (candidate < value1) value1 = candidate;
                        }
                    }

                    unsigned long long old0 = value0;
                    unsigned long long old1 = value1;
                    unsigned long long candidate0 = ptv_warp64_add_jump(
                        old1, 32U, inf_distance
                    );
                    unsigned long long candidate1 = ptv_warp64_add_jump(
                        old0, 32U, inf_distance
                    );
                    if (candidate0 < value0) value0 = candidate0;
                    if (candidate1 < value1) value1 = candidate1;
                    lower_state[idx0] = value0;
                    lower_state[idx1] = value1;
                }
                grid.sync();
            }
        }
        ''',
        cooperative=True,
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _monotone_certificate_kernel():
    return _kernel(
        "ptv_monotone_path_certificate6",
        r'''
        __device__ __forceinline__
        bool ptv_walk_axis(
            const unsigned char* mask,
            int& z,
            int& y,
            int& x,
            const int tz,
            const int ty,
            const int tx,
            const int axis,
            const int H,
            const int W
        ) {
            int target = axis == 0 ? tz : axis == 1 ? ty : tx;
            int* current = axis == 0 ? &z : axis == 1 ? &y : &x;
            int direction = target > *current ? 1 : -1;
            while (*current != target) {
                *current += direction;
                int idx = (z * H + y) * W + x;
                if (!mask[idx]) return false;
            }
            return true;
        }

        __device__ __forceinline__
        bool ptv_axis_order_clear(
            const unsigned char* mask,
            const int z0,
            const int y0,
            const int x0,
            const int z1,
            const int y1,
            const int x1,
            const int a0,
            const int a1,
            const int a2,
            const int H,
            const int W
        ) {
            int z = z0;
            int y = y0;
            int x = x0;
            return ptv_walk_axis(mask, z, y, x, z1, y1, x1, a0, H, W) &&
                   ptv_walk_axis(mask, z, y, x, z1, y1, x1, a1, H, W) &&
                   ptv_walk_axis(mask, z, y, x, z1, y1, x1, a2, H, W);
        }

        __device__ __forceinline__
        bool ptv_greedy_monotone_clear(
            const unsigned char* mask,
            const int z0,
            const int y0,
            const int x0,
            const int z1,
            const int y1,
            const int x1,
            const int a0,
            const int a1,
            const int a2,
            const int H,
            const int W
        ) {
            int z = z0;
            int y = y0;
            int x = x0;
            int remaining = abs(z1 - z0) + abs(y1 - y0) + abs(x1 - x0);
            int order[3] = {a0, a1, a2};
            for (int step = 0; step < remaining; ++step) {
                bool moved = false;
                #pragma unroll
                for (int rank = 0; rank < 3; ++rank) {
                    int axis = order[rank];
                    int nz = z;
                    int ny = y;
                    int nx = x;
                    if (axis == 0 && z != z1) nz += z1 > z ? 1 : -1;
                    else if (axis == 1 && y != y1) ny += y1 > y ? 1 : -1;
                    else if (axis == 2 && x != x1) nx += x1 > x ? 1 : -1;
                    else continue;
                    int next_idx = (nz * H + ny) * W + nx;
                    if (!mask[next_idx]) continue;
                    z = nz;
                    y = ny;
                    x = nx;
                    moved = true;
                    break;
                }
                if (!moved) return false;
            }
            return z == z1 && y == y1 && x == x1;
        }

        extern "C" __global__
        void ptv_monotone_path_certificate6(
            const unsigned char* mask,
            const unsigned long long* lower_state,
            unsigned long long* closure_state,
            const int* seeds,
            int* certified,
            int* roi_ids,
            int* roi_n,
            int* pore_n,
            const int nseed,
            const int nvox,
            const int bidirectional_greedy,
            const int count_pore,
            const int write_closure_state,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance,
            const unsigned long long inf_pack
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            if (!mask[idx]) {
                certified[idx] = 1;
                if (write_closure_state) closure_state[idx] = inf_pack;
                return;
            }
            if (count_pore) atomicAdd(pore_n, 1);

            unsigned long long packed = lower_state[idx];
            unsigned int distance = (unsigned int)(packed >> 32);
            int owner = (int)(unsigned int)packed;
            if (distance >= inf_distance || owner < 0 || owner >= nseed) {
                certified[idx] = 0;
                if (write_closure_state) closure_state[idx] = inf_pack;
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
                return;
            }

            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;
            int sz = seeds[3 * owner + 0];
            int sy = seeds[3 * owner + 1];
            int sx = seeds[3 * owner + 2];

            bool clear =
                ptv_axis_order_clear(mask, sz, sy, sx, z, y, x, 0, 1, 2, H, W) ||
                ptv_axis_order_clear(mask, sz, sy, sx, z, y, x, 0, 2, 1, H, W) ||
                ptv_axis_order_clear(mask, sz, sy, sx, z, y, x, 1, 0, 2, H, W) ||
                ptv_axis_order_clear(mask, sz, sy, sx, z, y, x, 1, 2, 0, H, W) ||
                ptv_axis_order_clear(mask, sz, sy, sx, z, y, x, 2, 0, 1, H, W) ||
                ptv_axis_order_clear(mask, sz, sy, sx, z, y, x, 2, 1, 0, H, W);
            if (!clear) {
                clear =
                    ptv_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 0, 1, 2, H, W) ||
                    ptv_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 0, 2, 1, H, W) ||
                    ptv_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 1, 0, 2, H, W) ||
                    ptv_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 1, 2, 0, H, W) ||
                    ptv_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 2, 0, 1, H, W) ||
                    ptv_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 2, 1, 0, H, W);
                if (!clear && bidirectional_greedy) {
                    clear =
                        ptv_greedy_monotone_clear(mask, z, y, x, sz, sy, sx, 0, 1, 2, H, W) ||
                        ptv_greedy_monotone_clear(mask, z, y, x, sz, sy, sx, 0, 2, 1, H, W) ||
                        ptv_greedy_monotone_clear(mask, z, y, x, sz, sy, sx, 1, 0, 2, H, W) ||
                        ptv_greedy_monotone_clear(mask, z, y, x, sz, sy, sx, 1, 2, 0, H, W) ||
                        ptv_greedy_monotone_clear(mask, z, y, x, sz, sy, sx, 2, 0, 1, H, W) ||
                        ptv_greedy_monotone_clear(mask, z, y, x, sz, sy, sx, 2, 1, 0, H, W);
                }
            }
            certified[idx] = clear ? 1 : 0;
            if (write_closure_state) {
                closure_state[idx] = clear ? packed : inf_pack;
            }
            if (!clear) {
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
            }
        }
        ''',
    )


def _mask_line_bits_kernel():
    return _kernel(
        "ptv_mask_line_bits64",
        r'''
        extern "C" __global__
        void ptv_mask_line_bits64(
            const unsigned char* mask,
            unsigned long long* x_bits,
            unsigned long long* y_bits,
            unsigned long long* z_bits,
            const int D,
            const int H,
            const int W
        ) {
            int line = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            int nx_line = D * H;
            int ny_line = D * W;
            int nz_line = H * W;
            if (line < nx_line) {
                int z = line / H;
                int y = line - z * H;
                int base = (z * H + y) * W;
                unsigned long long bits = 0ULL;
                for (int x = 0; x < W; ++x) {
                    if (mask[base + x]) bits |= 1ULL << x;
                }
                x_bits[line] = bits;
                return;
            }
            line -= nx_line;
            if (line < ny_line) {
                int z = line / W;
                int x = line - z * W;
                int base = z * H * W + x;
                unsigned long long bits = 0ULL;
                for (int y = 0; y < H; ++y) {
                    if (mask[base + y * W]) bits |= 1ULL << y;
                }
                y_bits[line] = bits;
                return;
            }
            line -= ny_line;
            if (line < nz_line) {
                int y = line / W;
                int x = line - y * W;
                int base = y * W + x;
                int HW = H * W;
                unsigned long long bits = 0ULL;
                for (int z = 0; z < D; ++z) {
                    if (mask[base + z * HW]) bits |= 1ULL << z;
                }
                z_bits[line] = bits;
            }
        }
        ''',
    )


def _bitset_certificate_kernel():
    return _kernel(
        "ptv_bitset_path_certificate6",
        r'''
        __device__ __forceinline__
        unsigned long long ptv_interval_bits(const int a, const int b) {
            int lo = a < b ? a : b;
            int hi = a < b ? b : a;
            unsigned long long upper =
                hi == 63 ? ~0ULL : ((1ULL << (hi + 1)) - 1ULL);
            unsigned long long lower =
                lo == 0 ? 0ULL : ((1ULL << lo) - 1ULL);
            return upper & ~lower;
        }

        __device__ __forceinline__
        bool ptv_bitset_segment_clear(
            const unsigned long long* x_bits,
            const unsigned long long* y_bits,
            const unsigned long long* z_bits,
            const int z,
            const int y,
            const int x,
            const int target,
            const int axis,
            const int H,
            const int W
        ) {
            unsigned long long line;
            int current;
            if (axis == 0) {
                line = z_bits[y * W + x];
                current = z;
            } else if (axis == 1) {
                line = y_bits[z * W + x];
                current = y;
            } else {
                line = x_bits[z * H + y];
                current = x;
            }
            unsigned long long required = ptv_interval_bits(current, target);
            return (line & required) == required;
        }

        __device__ __forceinline__
        bool ptv_bitset_axis_order_clear(
            const unsigned long long* x_bits,
            const unsigned long long* y_bits,
            const unsigned long long* z_bits,
            const int z0,
            const int y0,
            const int x0,
            const int z1,
            const int y1,
            const int x1,
            const int a0,
            const int a1,
            const int a2,
            const int H,
            const int W
        ) {
            int z = z0;
            int y = y0;
            int x = x0;
            int order[3] = {a0, a1, a2};
            #pragma unroll
            for (int rank = 0; rank < 3; ++rank) {
                int axis = order[rank];
                int target = axis == 0 ? z1 : axis == 1 ? y1 : x1;
                if (!ptv_bitset_segment_clear(
                        x_bits, y_bits, z_bits, z, y, x, target, axis, H, W)) {
                    return false;
                }
                if (axis == 0) z = z1;
                else if (axis == 1) y = y1;
                else x = x1;
            }
            return true;
        }

        __device__ __forceinline__
        bool ptv_bitset_greedy_monotone_clear(
            const unsigned char* mask,
            const int z0,
            const int y0,
            const int x0,
            const int z1,
            const int y1,
            const int x1,
            const int a0,
            const int a1,
            const int a2,
            const int H,
            const int W
        ) {
            int z = z0;
            int y = y0;
            int x = x0;
            int remaining = abs(z1 - z0) + abs(y1 - y0) + abs(x1 - x0);
            int order[3] = {a0, a1, a2};
            for (int step = 0; step < remaining; ++step) {
                bool moved = false;
                #pragma unroll
                for (int rank = 0; rank < 3; ++rank) {
                    int axis = order[rank];
                    int nz = z;
                    int ny = y;
                    int nx = x;
                    if (axis == 0 && z != z1) nz += z1 > z ? 1 : -1;
                    else if (axis == 1 && y != y1) ny += y1 > y ? 1 : -1;
                    else if (axis == 2 && x != x1) nx += x1 > x ? 1 : -1;
                    else continue;
                    int next_idx = (nz * H + ny) * W + nx;
                    if (!mask[next_idx]) continue;
                    z = nz;
                    y = ny;
                    x = nx;
                    moved = true;
                    break;
                }
                if (!moved) return false;
            }
            return z == z1 && y == y1 && x == x1;
        }

        extern "C" __global__
        void ptv_bitset_path_certificate6(
            const unsigned char* mask,
            const unsigned long long* lower_state,
            unsigned long long* closure_state,
            const unsigned long long* x_bits,
            const unsigned long long* y_bits,
            const unsigned long long* z_bits,
            const int* seeds,
            int* certified,
            int* roi_ids,
            int* roi_n,
            int* pore_n,
            const int nseed,
            const int nvox,
            const int count_pore,
            const int write_closure_state,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance,
            const unsigned long long inf_pack
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            if (!mask[idx]) {
                certified[idx] = 1;
                if (write_closure_state) closure_state[idx] = inf_pack;
                return;
            }
            if (count_pore) atomicAdd(pore_n, 1);

            unsigned long long packed = lower_state[idx];
            unsigned int distance = (unsigned int)(packed >> 32);
            int owner = (int)(unsigned int)packed;
            if (distance >= inf_distance || owner < 0 || owner >= nseed) {
                certified[idx] = 0;
                if (write_closure_state) closure_state[idx] = inf_pack;
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
                return;
            }

            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;
            int sz = seeds[3 * owner + 0];
            int sy = seeds[3 * owner + 1];
            int sx = seeds[3 * owner + 2];

            bool clear =
                ptv_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 0, 1, 2, H, W) ||
                ptv_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 0, 2, 1, H, W) ||
                ptv_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 1, 0, 2, H, W) ||
                ptv_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 1, 2, 0, H, W) ||
                ptv_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 2, 0, 1, H, W) ||
                ptv_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 2, 1, 0, H, W);
            if (!clear) {
                clear =
                    ptv_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 0, 1, 2, H, W) ||
                    ptv_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 0, 2, 1, H, W) ||
                    ptv_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 1, 0, 2, H, W) ||
                    ptv_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 1, 2, 0, H, W) ||
                    ptv_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 2, 0, 1, H, W) ||
                    ptv_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 2, 1, 0, H, W);
            }
            certified[idx] = clear ? 1 : 0;
            if (write_closure_state) {
                closure_state[idx] = clear ? packed : inf_pack;
            }
            if (!clear) {
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
            }
        }
        ''',
    )


def _bitset_compact_certificate_kernel(*, specialized_64: bool = False):
    if specialized_64:
        kernel_name = "ptv_bitset_compact_path_certificate6_64"
    else:
        kernel_name = "ptv_bitset_compact_path_certificate6"
    preamble = (
        f"#define PTV_CERTIFICATE_KERNEL {kernel_name}\n"
        f"#define PTV_SPECIALIZED_64 {1 if specialized_64 else 0}\n"
    )
    return _kernel(
        kernel_name,
        preamble
        + r'''
        #if PTV_SPECIALIZED_64
        #define PTV_H_VALUE(H) 64
        #define PTV_W_VALUE(W) 64
        #else
        #define PTV_H_VALUE(H) (H)
        #define PTV_W_VALUE(W) (W)
        #endif

        __device__ __forceinline__
        unsigned long long ptv_pore_interval_bits(const int a, const int b) {
            int lo = a < b ? a : b;
            int hi = a < b ? b : a;
            unsigned long long upper =
                hi == 63 ? ~0ULL : ((1ULL << (hi + 1)) - 1ULL);
            unsigned long long lower =
                lo == 0 ? 0ULL : ((1ULL << lo) - 1ULL);
            return upper & ~lower;
        }

        __device__ __forceinline__
        bool ptv_pore_bitset_segment_clear(
            const unsigned long long* x_bits,
            const unsigned long long* y_bits,
            const unsigned long long* z_bits,
            const int z,
            const int y,
            const int x,
            const int target,
            const int axis,
            const int H,
            const int W
        ) {
            unsigned long long line;
            int current;
            if (axis == 0) {
                line = z_bits[y * PTV_W_VALUE(W) + x];
                current = z;
            } else if (axis == 1) {
                line = y_bits[z * PTV_W_VALUE(W) + x];
                current = y;
            } else {
                line = x_bits[z * PTV_H_VALUE(H) + y];
                current = x;
            }
            unsigned long long required =
                ptv_pore_interval_bits(current, target);
            return (line & required) == required;
        }

        __device__ __forceinline__
        bool ptv_pore_bitset_axis_order_clear(
            const unsigned long long* x_bits,
            const unsigned long long* y_bits,
            const unsigned long long* z_bits,
            const int z0,
            const int y0,
            const int x0,
            const int z1,
            const int y1,
            const int x1,
            const int a0,
            const int a1,
            const int a2,
            const int H,
            const int W
        ) {
            int z = z0;
            int y = y0;
            int x = x0;
            int order[3] = {a0, a1, a2};
            #pragma unroll
            for (int rank = 0; rank < 3; ++rank) {
                int axis = order[rank];
                int target = axis == 0 ? z1 : axis == 1 ? y1 : x1;
                if (!ptv_pore_bitset_segment_clear(
                        x_bits, y_bits, z_bits, z, y, x, target, axis, H, W)) {
                    return false;
                }
                if (axis == 0) z = z1;
                else if (axis == 1) y = y1;
                else x = x1;
            }
            return true;
        }

        __device__ __forceinline__
        bool ptv_pore_bitset_greedy_monotone_clear(
            const unsigned char* mask,
            const int z0,
            const int y0,
            const int x0,
            const int z1,
            const int y1,
            const int x1,
            const int a0,
            const int a1,
            const int a2,
            const int H,
            const int W
        ) {
            int z = z0;
            int y = y0;
            int x = x0;
            int remaining = abs(z1 - z0) + abs(y1 - y0) + abs(x1 - x0);
            int order[3] = {a0, a1, a2};
            for (int step = 0; step < remaining; ++step) {
                bool moved = false;
                #pragma unroll
                for (int rank = 0; rank < 3; ++rank) {
                    int axis = order[rank];
                    int nz = z;
                    int ny = y;
                    int nx = x;
                    if (axis == 0 && z != z1) nz += z1 > z ? 1 : -1;
                    else if (axis == 1 && y != y1) ny += y1 > y ? 1 : -1;
                    else if (axis == 2 && x != x1) nx += x1 > x ? 1 : -1;
                    else continue;
                    int next_idx =
                        (nz * PTV_H_VALUE(H) + ny) *
                        PTV_W_VALUE(W) + nx;
                    if (!mask[next_idx]) continue;
                    z = nz;
                    y = ny;
                    x = nx;
                    moved = true;
                    break;
                }
                if (!moved) return false;
            }
            return z == z1 && y == y1 && x == x1;
        }

        extern "C" __global__
        void PTV_CERTIFICATE_KERNEL(
            const unsigned char* mask,
            const unsigned long long* lower_state,
            unsigned long long* closure_state,
            const unsigned long long* x_bits,
            const unsigned long long* y_bits,
            const unsigned long long* z_bits,
            const int* seeds,
            int* roi_ids,
            int* roi_n,
            const int nseed,
            const int nvox,
            const int D,
            const int H,
            const int W,
            const int power2_coordinates,
            const int w_shift,
            const int hw_shift,
            const unsigned int inf_distance,
            const unsigned long long inf_pack
        ) {
            int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (tid >= nvox) return;
            int idx = tid;
            if (!mask[idx]) return;
            unsigned long long packed = lower_state[idx];
            unsigned int distance = (unsigned int)(packed >> 32);
            int owner = (int)(unsigned int)packed;
            if (distance >= inf_distance || owner < 0 || owner >= nseed) {
                closure_state[idx] = inf_pack;
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
                return;
            }

            int z;
            int y;
            int x;
            #if PTV_SPECIALIZED_64
            z = idx >> 12;
            y = (idx >> 6) & 63;
            x = idx & 63;
            #else
            if (power2_coordinates) {
                int HW = H * W;
                z = idx >> hw_shift;
                int rem = idx & (HW - 1);
                y = rem >> w_shift;
                x = rem & (W - 1);
            } else {
                int HW = H * W;
                z = idx / HW;
                int rem = idx - z * HW;
                y = rem / W;
                x = rem - y * W;
            }
            #endif
            int sz = seeds[3 * owner + 0];
            int sy = seeds[3 * owner + 1];
            int sx = seeds[3 * owner + 2];

            bool clear =
                ptv_pore_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 0, 1, 2, H, W) ||
                ptv_pore_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 0, 2, 1, H, W) ||
                ptv_pore_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 1, 0, 2, H, W) ||
                ptv_pore_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 1, 2, 0, H, W) ||
                ptv_pore_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 2, 0, 1, H, W) ||
                ptv_pore_bitset_axis_order_clear(x_bits, y_bits, z_bits, sz, sy, sx, z, y, x, 2, 1, 0, H, W);
            if (!clear) {
                clear =
                    ptv_pore_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 0, 1, 2, H, W) ||
                    ptv_pore_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 0, 2, 1, H, W) ||
                    ptv_pore_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 1, 0, 2, H, W) ||
                    ptv_pore_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 1, 2, 0, H, W) ||
                    ptv_pore_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 2, 0, 1, H, W) ||
                    ptv_pore_bitset_greedy_monotone_clear(mask, sz, sy, sx, z, y, x, 2, 1, 0, H, W);
            }
            closure_state[idx] = clear ? packed : inf_pack;
            if (!clear) {
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
            }
        }
        ''',
    )


def prepare_mask_line_bits64(mask, *, block_size: int = 256):
    """Build reusable pore-line bitsets for masks with axis lengths at most 64."""
    import cupy as cp

    mask_u8 = cp.asarray(mask, dtype=cp.uint8)
    if mask_u8.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    D, H, W = (int(value) for value in mask_u8.shape)
    if max(D, H, W) > 64:
        raise ValueError("64-bit pore-line masks require every axis length to be at most 64")
    x_bits = cp.empty(D * H, dtype=cp.uint64)
    y_bits = cp.empty(D * W, dtype=cp.uint64)
    z_bits = cp.empty(H * W, dtype=cp.uint64)
    line_count = D * H + D * W + H * W
    block = int(block_size)
    grid = (max(1, math.ceil(line_count / block)),)
    _mask_line_bits_kernel()(
        grid,
        (block,),
        (
            mask_u8.ravel(),
            x_bits,
            y_bits,
            z_bits,
            np.int32(D),
            np.int32(H),
            np.int32(W),
        ),
    )
    return x_bits, y_bits, z_bits


def prepare_mask_pore_ids(mask):
    """Build a reusable compact list of pore-voxel flat indices."""
    import cupy as cp

    mask_u8 = cp.asarray(mask, dtype=cp.uint8)
    if mask_u8.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    return cp.flatnonzero(mask_u8.ravel()).astype(cp.int32, copy=False)


def _certificate_seed_kernel():
    return _kernel(
        "ptv_certificate_seed6",
        r'''
        extern "C" __global__
        void ptv_certificate_seed6(
            const unsigned char* mask,
            const unsigned long long* lower_state,
            const int* seed_label,
            int* certified,
            int* frontier,
            int* frontier_n,
            int* certified_n,
            const int nvox,
            const int nseed
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox || !mask[idx]) return;
            int seed = seed_label[idx];
            if (seed < 0 || seed >= nseed) return;
            unsigned long long expected = (unsigned long long)(unsigned int)seed;
            if (lower_state[idx] != expected) return;
            if (atomicCAS(certified + idx, 0, 1) == 0) {
                int pos = atomicAdd(frontier_n, 1);
                frontier[pos] = idx;
                atomicAdd(certified_n, 1);
            }
        }
        ''',
    )


def _cooperative_certificate_kernel():
    return _kernel(
        "ptv_cooperative_certificate6",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_cooperative_certificate6(
            const unsigned char* mask,
            const unsigned long long* lower_state,
            int* certified,
            int* frontier_a,
            int* frontier_b,
            int* current_n,
            int* next_n,
            int* certified_n,
            int* iterations,
            int* converged,
            int* overflow,
            const int nvox,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();
            const unsigned long long unit = 1ULL << 32;
            const int HW = H * W;

            if (rank == 0) {
                *iterations = 0;
                *converged = 0;
                *overflow = 0;
            }
            grid.sync();

            for (int level = 1; level <= nvox; ++level) {
                if (rank == 0) *next_n = 0;
                grid.sync();

                int count = *current_n;
                int* frontier_in = (level & 1) ? frontier_a : frontier_b;
                int* frontier_out = (level & 1) ? frontier_b : frontier_a;
                for (unsigned long long pos = rank;
                     pos < (unsigned long long)count;
                     pos += stride) {
                    int idx = frontier_in[pos];
                    unsigned long long current = lower_state[idx];
                    if ((unsigned int)(current >> 32) >= inf_distance) continue;
                    unsigned long long successor = current + unit;

                    int z = idx / HW;
                    int rem = idx - z * HW;
                    int y = rem / W;
                    int x = rem - y * W;
                    int neighbor[6];
                    int neighbor_count = 0;
                    if (z > 0) neighbor[neighbor_count++] = idx - HW;
                    if (z + 1 < D) neighbor[neighbor_count++] = idx + HW;
                    if (y > 0) neighbor[neighbor_count++] = idx - W;
                    if (y + 1 < H) neighbor[neighbor_count++] = idx + W;
                    if (x > 0) neighbor[neighbor_count++] = idx - 1;
                    if (x + 1 < W) neighbor[neighbor_count++] = idx + 1;

                    #pragma unroll
                    for (int q = 0; q < 6; ++q) {
                        if (q >= neighbor_count) break;
                        int nb = neighbor[q];
                        if (!mask[nb] || lower_state[nb] != successor) continue;
                        if (atomicCAS(certified + nb, 0, 1) == 0) {
                            int output_pos = atomicAdd(next_n, 1);
                            if (output_pos < nvox) frontier_out[output_pos] = nb;
                            else atomicExch(overflow, 1);
                            atomicAdd(certified_n, 1);
                        }
                    }
                }

                grid.sync();
                if (rank == 0) {
                    *current_n = *next_n;
                    *iterations = level;
                    if (*next_n == 0) *converged = 1;
                }
                grid.sync();
                if (*converged || *overflow) break;
            }
        }
        ''',
        cooperative=True,
    )


def _roi_compact_kernel():
    return _kernel(
        "ptv_roi_compact6",
        r'''
        extern "C" __global__
        void ptv_roi_compact6(
            const unsigned char* mask,
            const int* certified,
            int* roi_ids,
            int* roi_n,
            int* pore_n,
            const int nvox
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox || !mask[idx]) return;
            atomicAdd(pore_n, 1);
            if (!certified[idx]) {
                int pos = atomicAdd(roi_n, 1);
                roi_ids[pos] = idx;
            }
        }
        ''',
    )


def _roi_state_init_kernel():
    return _kernel(
        "ptv_roi_state_init",
        r'''
        extern "C" __global__
        void ptv_roi_state_init(
            const unsigned char* mask,
            const int* certified,
            const unsigned long long* lower_state,
            unsigned long long* state,
            const int nvox,
            const unsigned long long inf_pack
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            state[idx] = (mask[idx] && certified[idx]) ? lower_state[idx] : inf_pack;
        }
        ''',
    )


def _packed_output_f32_kernel():
    return _kernel(
        "ptv_packed_output_f32",
        r'''
        extern "C" __global__
        void ptv_packed_output_f32(
            const unsigned char* mask,
            const unsigned long long* state,
            int* label,
            float* distance,
            const int nvox
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            if (!mask[idx]) {
                label[idx] = -1;
                distance[idx] = __int_as_float(0x7f800000);
                return;
            }
            unsigned long long packed = state[idx];
            label[idx] = (int)(unsigned int)packed;
            distance[idx] = (float)(unsigned int)(packed >> 32);
        }
        ''',
    )


def _packed_output_f64_kernel():
    return _kernel(
        "ptv_packed_output_f64",
        r'''
        extern "C" __global__
        void ptv_packed_output_f64(
            const unsigned char* mask,
            const unsigned long long* state,
            int* label,
            double* distance,
            const int nvox
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox) return;
            if (!mask[idx]) {
                label[idx] = -1;
                distance[idx] = __longlong_as_double(0x7ff0000000000000ULL);
                return;
            }
            unsigned long long packed = state[idx];
            label[idx] = (int)(unsigned int)packed;
            distance[idx] = (double)(unsigned int)(packed >> 32);
        }
        ''',
    )


def _roi_relax_kernel():
    return _kernel(
        "ptv_roi_relax6",
        r'''
        extern "C" __global__
        void ptv_roi_relax6(
            const unsigned char* mask,
            const int* roi_ids,
            const int nroi,
            const unsigned long long* state_in,
            unsigned long long* state_out,
            int* any_changed,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (tid >= nroi) return;
            int idx = roi_ids[tid];
            unsigned long long current = state_in[idx];
            unsigned long long best = current;
            const unsigned long long unit = 1ULL << 32;

            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;
            int neighbor[6];
            int count = 0;
            if (z > 0) neighbor[count++] = idx - HW;
            if (z + 1 < D) neighbor[count++] = idx + HW;
            if (y > 0) neighbor[count++] = idx - W;
            if (y + 1 < H) neighbor[count++] = idx + W;
            if (x > 0) neighbor[count++] = idx - 1;
            if (x + 1 < W) neighbor[count++] = idx + 1;

            for (int q = 0; q < count; ++q) {
                int nb = neighbor[q];
                if (!mask[nb]) continue;
                unsigned long long source = state_in[nb];
                if ((unsigned int)(source >> 32) >= inf_distance) continue;
                unsigned long long candidate = source + unit;
                if (candidate < best) best = candidate;
            }
            state_out[idx] = best;
            if (best < current) atomicExch(any_changed, 1);
        }
        ''',
    )


def _cooperative_relax_kernel():
    return _kernel(
        "ptv_cooperative_relax6",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_cooperative_relax6(
            const unsigned char* mask,
            const int* active_ids,
            const int nactive,
            const int* seed_label,
            unsigned long long* state,
            int* changed,
            int* iterations,
            int* converged,
            const int max_iterations,
            const int clamp_seeds,
            const int nseed,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();
            const unsigned long long unit = 1ULL << 32;
            const int HW = H * W;

            if (rank == 0) {
                *iterations = 0;
                *converged = 0;
            }
            grid.sync();

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                if (rank == 0) *changed = 0;
                grid.sync();

                for (unsigned long long pos = rank;
                     pos < (unsigned long long)nactive;
                     pos += stride) {
                    int idx = active_ids[pos];
                    if (clamp_seeds) {
                        int seed = seed_label[idx];
                        if (seed >= 0 && seed < nseed) continue;
                    }

                    unsigned long long current = state[idx];
                    unsigned long long best = current;
                    int z = idx / HW;
                    int rem = idx - z * HW;
                    int y = rem / W;
                    int x = rem - y * W;

                    int neighbor[6];
                    int count = 0;
                    if (z > 0) neighbor[count++] = idx - HW;
                    if (z + 1 < D) neighbor[count++] = idx + HW;
                    if (y > 0) neighbor[count++] = idx - W;
                    if (y + 1 < H) neighbor[count++] = idx + W;
                    if (x > 0) neighbor[count++] = idx - 1;
                    if (x + 1 < W) neighbor[count++] = idx + 1;

                    #pragma unroll
                    for (int q = 0; q < 6; ++q) {
                        if (q >= count) break;
                        int nb = neighbor[q];
                        if (!mask[nb]) continue;
                        unsigned long long source = state[nb];
                        if ((unsigned int)(source >> 32) >= inf_distance) continue;
                        unsigned long long candidate = source + unit;
                        if (candidate < best) best = candidate;
                    }

                    if (best < current) {
                        unsigned long long old = atomicMin(state + idx, best);
                        if (best < old) atomicExch(changed, 1);
                    }
                }

                grid.sync();
                int iteration_changed = *changed;
                if (rank == 0) {
                    *iterations = iteration + 1;
                    if (!iteration_changed) *converged = 1;
                }
                grid.sync();
                if (!iteration_changed) break;
            }
        }
        ''',
        cooperative=True,
    )


def _cooperative_roi_relax_kernel():
    return _kernel(
        "ptv_cooperative_roi_relax6",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_cooperative_roi_relax6(
            const int* active_ids,
            const int nactive,
            unsigned long long* state,
            int* changed,
            int* iterations,
            int* converged,
            const int max_iterations,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();
            const unsigned long long unit = 1ULL << 32;
            const int HW = H * W;

            if (rank == 0) {
                *iterations = 0;
                *converged = 0;
            }
            grid.sync();

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                if (rank == 0) *changed = 0;
                grid.sync();

                for (unsigned long long pos = rank;
                     pos < (unsigned long long)nactive;
                     pos += stride) {
                    int idx = active_ids[pos];
                    unsigned long long current = state[idx];
                    unsigned long long best = current;
                    int z = idx / HW;
                    int rem = idx - z * HW;
                    int y = rem / W;
                    int x = rem - y * W;

                    #define PTV_TRY_ROI_NEIGHBOR(NB) do { \
                        unsigned long long source = state[(NB)]; \
                        if ((unsigned int)(source >> 32) < inf_distance) { \
                            unsigned long long candidate = source + unit; \
                            if (candidate < best) best = candidate; \
                        } \
                    } while (0)

                    if (z > 0) PTV_TRY_ROI_NEIGHBOR(idx - HW);
                    if (z + 1 < D) PTV_TRY_ROI_NEIGHBOR(idx + HW);
                    if (y > 0) PTV_TRY_ROI_NEIGHBOR(idx - W);
                    if (y + 1 < H) PTV_TRY_ROI_NEIGHBOR(idx + W);
                    if (x > 0) PTV_TRY_ROI_NEIGHBOR(idx - 1);
                    if (x + 1 < W) PTV_TRY_ROI_NEIGHBOR(idx + 1);
                    #undef PTV_TRY_ROI_NEIGHBOR

                    if (best < current) {
                        state[idx] = best;
                        atomicExch(changed, 1);
                    }
                }

                grid.sync();
                int iteration_changed = *changed;
                if (rank == 0) {
                    *iterations = iteration + 1;
                    if (!iteration_changed) *converged = 1;
                }
                grid.sync();
                if (!iteration_changed) break;
            }
        }
        ''',
        cooperative=True,
    )


def _cooperative_roi_relax_power2_kernel():
    return _kernel(
        "ptv_cooperative_roi_relax6_power2",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_cooperative_roi_relax6_power2(
            const int* active_ids,
            const int nactive,
            unsigned long long* state,
            int* changed,
            int* iterations,
            int* converged,
            const int max_iterations,
            const int D,
            const int H,
            const int W,
            const int w_shift,
            const int hw_shift,
            const unsigned int inf_distance
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();
            const unsigned long long unit = 1ULL << 32;
            const int HW = H * W;
            const int w_mask = W - 1;
            const int hw_mask = HW - 1;

            if (rank == 0) {
                *iterations = 0;
                *converged = 0;
            }
            grid.sync();

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                if (rank == 0) *changed = 0;
                grid.sync();

                for (unsigned long long pos = rank;
                     pos < (unsigned long long)nactive;
                     pos += stride) {
                    int idx = active_ids[pos];
                    unsigned long long current = state[idx];
                    unsigned long long best = current;
                    int z = idx >> hw_shift;
                    int rem = idx & hw_mask;
                    int y = rem >> w_shift;
                    int x = rem & w_mask;

                    #define PTV_TRY_ROI_NEIGHBOR(NB) do { \
                        unsigned long long source = state[(NB)]; \
                        if ((unsigned int)(source >> 32) < inf_distance) { \
                            unsigned long long candidate = source + unit; \
                            if (candidate < best) best = candidate; \
                        } \
                    } while (0)

                    if (z > 0) PTV_TRY_ROI_NEIGHBOR(idx - HW);
                    if (z + 1 < D) PTV_TRY_ROI_NEIGHBOR(idx + HW);
                    if (y > 0) PTV_TRY_ROI_NEIGHBOR(idx - W);
                    if (y + 1 < H) PTV_TRY_ROI_NEIGHBOR(idx + W);
                    if (x > 0) PTV_TRY_ROI_NEIGHBOR(idx - 1);
                    if (x + 1 < W) PTV_TRY_ROI_NEIGHBOR(idx + 1);
                    #undef PTV_TRY_ROI_NEIGHBOR

                    if (best < current) {
                        state[idx] = best;
                        atomicExch(changed, 1);
                    }
                }

                grid.sync();
                int iteration_changed = *changed;
                if (rank == 0) {
                    *iterations = iteration + 1;
                    if (!iteration_changed) *converged = 1;
                }
                grid.sync();
                if (!iteration_changed) break;
            }
        }
        ''',
        cooperative=True,
    )


def _cooperative_roi_relax_power2_warp_vote_kernel():
    return _kernel(
        "ptv_cooperative_roi_relax6_power2_warp_vote",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_cooperative_roi_relax6_power2_warp_vote(
            const int* active_ids,
            const int nactive,
            unsigned long long* state,
            int* changed,
            int* iterations,
            int* converged,
            const int max_iterations,
            const int D,
            const int H,
            const int W,
            const int w_shift,
            const int hw_shift,
            const unsigned int inf_distance
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();
            const unsigned long long unit = 1ULL << 32;
            const int HW = H * W;
            const int w_mask = W - 1;
            const int hw_mask = HW - 1;

            if (rank == 0) {
                *iterations = 0;
                *converged = 0;
                *changed = 0;
            }
            grid.sync();

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                int thread_changed = 0;
                const int warp_base = threadIdx.x & ~31;
                const int warp_width =
                    min(32, (int)blockDim.x - warp_base);
                const unsigned int warp_mask =
                    warp_width == 32
                    ? 0xFFFFFFFFU
                    : ((1U << warp_width) - 1U);
                for (unsigned long long pos = rank;
                     pos < (unsigned long long)nactive;
                     pos += stride) {
                    int idx = active_ids[pos];
                    unsigned long long current = state[idx];
                    unsigned long long best = current;
                    int z = idx >> hw_shift;
                    int rem = idx & hw_mask;
                    int y = rem >> w_shift;
                    int x = rem & w_mask;

                    #define PTV_TRY_ROI_NEIGHBOR(NB) do { \
                        unsigned long long source = state[(NB)]; \
                        if ((unsigned int)(source >> 32) < inf_distance) { \
                            unsigned long long candidate = source + unit; \
                            if (candidate < best) best = candidate; \
                        } \
                    } while (0)

                    if (z > 0) PTV_TRY_ROI_NEIGHBOR(idx - HW);
                    if (z + 1 < D) PTV_TRY_ROI_NEIGHBOR(idx + HW);
                    if (y > 0) PTV_TRY_ROI_NEIGHBOR(idx - W);
                    if (y + 1 < H) PTV_TRY_ROI_NEIGHBOR(idx + W);
                    if (x > 0) PTV_TRY_ROI_NEIGHBOR(idx - 1);
                    if (x + 1 < W) PTV_TRY_ROI_NEIGHBOR(idx + 1);
                    #undef PTV_TRY_ROI_NEIGHBOR

                    if (best < current) {
                        state[idx] = best;
                        thread_changed = 1;
                    }
                }

                __syncwarp(warp_mask);
                int warp_changed = __any_sync(warp_mask, thread_changed);
                if ((threadIdx.x & 31) == 0 && warp_changed) {
                    atomicExch(changed, 1);
                }
                grid.sync();
                if (rank == 0) {
                    int iteration_changed = *changed;
                    *iterations = iteration + 1;
                    if (!iteration_changed) *converged = 1;
                    *changed = 0;
                }
                grid.sync();
                if (*converged) break;
            }
        }
        ''',
        cooperative=True,
    )


def _cooperative_bfs_kernel():
    return _kernel(
        "ptv_cooperative_bfs6",
        r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        extern "C" __global__
        void ptv_cooperative_bfs6(
            const unsigned char* mask,
            unsigned long long* state,
            int* frontier_a,
            int* frontier_b,
            int* current_n,
            int* next_n,
            int* iterations,
            int* converged,
            int* overflow,
            const int nvox,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            cg::grid_group grid = cg::this_grid();
            const unsigned long long rank = grid.thread_rank();
            const unsigned long long stride = grid.size();
            const unsigned long long unit = 1ULL << 32;
            const int HW = H * W;

            if (rank == 0) {
                *iterations = 0;
                *converged = 0;
                *overflow = 0;
            }
            grid.sync();

            for (int level = 1; level <= nvox; ++level) {
                if (rank == 0) *next_n = 0;
                grid.sync();

                int count = *current_n;
                int* frontier_in = (level & 1) ? frontier_a : frontier_b;
                int* frontier_out = (level & 1) ? frontier_b : frontier_a;
                for (unsigned long long pos = rank;
                     pos < (unsigned long long)count;
                     pos += stride) {
                    int idx = frontier_in[pos];
                    unsigned long long current = state[idx];
                    unsigned int current_distance = (unsigned int)(current >> 32);
                    if (current_distance >= inf_distance) continue;
                    unsigned long long candidate = current + unit;
                    unsigned int candidate_distance = current_distance + 1U;

                    int z = idx / HW;
                    int rem = idx - z * HW;
                    int y = rem / W;
                    int x = rem - y * W;
                    int neighbor[6];
                    int neighbor_count = 0;
                    if (z > 0) neighbor[neighbor_count++] = idx - HW;
                    if (z + 1 < D) neighbor[neighbor_count++] = idx + HW;
                    if (y > 0) neighbor[neighbor_count++] = idx - W;
                    if (y + 1 < H) neighbor[neighbor_count++] = idx + W;
                    if (x > 0) neighbor[neighbor_count++] = idx - 1;
                    if (x + 1 < W) neighbor[neighbor_count++] = idx + 1;

                    #pragma unroll
                    for (int q = 0; q < 6; ++q) {
                        if (q >= neighbor_count) break;
                        int nb = neighbor[q];
                        if (!mask[nb]) continue;
                        unsigned long long old = atomicMin(state + nb, candidate);
                        unsigned int old_distance = (unsigned int)(old >> 32);
                        if (candidate < old && candidate_distance < old_distance) {
                            int output_pos = atomicAdd(next_n, 1);
                            if (output_pos < nvox) frontier_out[output_pos] = nb;
                            else atomicExch(overflow, 1);
                        }
                    }
                }

                grid.sync();
                if (rank == 0) {
                    *current_n = *next_n;
                    *iterations = level;
                    if (*next_n == 0) *converged = 1;
                }
                grid.sync();
                if (*converged || *overflow) break;
            }
        }
        ''',
        cooperative=True,
    )


def _cooperative_grid_size(
    kernel,
    *,
    nitems: int,
    block_size: int,
    blocks_per_sm: int,
    dynamic_shared_bytes: int = 0,
):
    import cupy as cp

    device = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device)
    multiprocessors = int(properties["multiProcessorCount"])
    kernel.compile()
    resident_limit = int(
        cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
            kernel.kernel.ptr, int(block_size), int(dynamic_shared_bytes)
        )
    )
    requested = max(1, int(blocks_per_sm))
    resident = max(1, min(requested, resident_limit))
    blocks = max(1, min(math.ceil(int(nitems) / int(block_size)), multiprocessors * resident))
    return blocks, resident, resident_limit


def _run_cooperative_fused_l1_jump(
    *,
    state,
    shape: tuple[int, int, int],
    blocks_per_sm: int,
):
    import cupy as cp

    device = int(cp.cuda.runtime.getDevice())
    cooperative = int(
        cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrCooperativeLaunch, device
        )
    )
    if not cooperative:
        return None

    D, H, W = shape
    threads = _next_power_of_two(max(D, H, W))
    if threads > 1024:
        return None
    shared_bytes = threads * np.dtype(np.uint64).itemsize
    max_lines = max(H * W, D * W, D * H)
    kernel = _l1_cooperative_fused_jump_kernel()
    blocks, resident, resident_limit = _cooperative_grid_size(
        kernel,
        nitems=max_lines * threads,
        block_size=threads,
        blocks_per_sm=blocks_per_sm,
        dynamic_shared_bytes=shared_bytes,
    )
    kernel(
        (blocks,),
        (threads,),
        (
            state,
            np.int32(D),
            np.int32(H),
            np.int32(W),
            INF_DISTANCE,
            INF_PACK,
        ),
        shared_mem=shared_bytes,
    )
    return {
        "grid_blocks": int(blocks),
        "resident_blocks_per_sm": int(resident),
        "resident_limit_per_sm": int(resident_limit),
        "threads_per_block": int(threads),
    }


def _run_cooperative_warp64_init_l1_jump(
    *,
    mask,
    seeds,
    lower_state,
    closure_state,
    bad_seed,
    shape: tuple[int, int, int],
    block_size: int,
    blocks_per_sm: int,
):
    import cupy as cp

    device = int(cp.cuda.runtime.getDevice())
    cooperative = int(
        cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrCooperativeLaunch, device
        )
    )
    if not cooperative:
        return None

    D, H, W = shape
    if (D, H, W) != (64, 64, 64):
        return None
    threads = int(block_size)
    if threads < 32 or threads > 1024 or threads % 32:
        raise ValueError(
            "warp64 lower-bound block size must be a multiple of 32 in [32, 1024]"
        )
    nvox = int(mask.size)
    nseed = int(seeds.shape[0])
    max_lines = max(H * W, D * W, D * H)
    kernel = _l1_cooperative_warp64_init_jump_kernel()
    blocks, resident, resident_limit = _cooperative_grid_size(
        kernel,
        nitems=max_lines * 32,
        block_size=threads,
        blocks_per_sm=blocks_per_sm,
    )
    kernel(
        (blocks,),
        (threads,),
        (
            mask,
            seeds,
            lower_state,
            closure_state,
            bad_seed,
            np.int32(nvox),
            np.int32(nseed),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            INF_DISTANCE,
            INF_PACK,
        ),
    )
    return {
        "grid_blocks": int(blocks),
        "resident_blocks_per_sm": int(resident),
        "resident_limit_per_sm": int(resident_limit),
        "threads_per_block": int(threads),
        "warps_per_block": int(threads // 32),
    }


def _run_cooperative_certificate(
    *,
    mask,
    lower_state,
    seed_label,
    nseed: int,
    shape: tuple[int, int, int],
    block_size: int,
    blocks_per_sm: int,
):
    import cupy as cp

    D, H, W = shape
    nvox = int(mask.size)
    block = int(block_size)
    grid_vox = (max(1, math.ceil(nvox / block)),)
    certified = cp.zeros(nvox, dtype=cp.int32)
    frontier_a = cp.empty(nvox, dtype=cp.int32)
    frontier_b = cp.empty_like(frontier_a)
    current_n = cp.zeros(1, dtype=cp.int32)
    next_n = cp.empty(1, dtype=cp.int32)
    certified_n = cp.zeros(1, dtype=cp.int32)
    iterations = cp.empty(1, dtype=cp.int32)
    converged = cp.empty(1, dtype=cp.int32)
    overflow = cp.empty(1, dtype=cp.int32)

    _certificate_seed_kernel()(
        grid_vox,
        (block,),
        (
            mask,
            lower_state,
            seed_label,
            certified,
            frontier_a,
            current_n,
            certified_n,
            np.int32(nvox),
            np.int32(nseed),
        ),
    )
    kernel = _cooperative_certificate_kernel()
    blocks, resident_blocks_per_sm, resident_limit = _cooperative_grid_size(
        kernel,
        nitems=nvox,
        block_size=block,
        blocks_per_sm=blocks_per_sm,
    )
    kernel(
        (blocks,),
        (block,),
        (
            mask,
            lower_state,
            certified,
            frontier_a,
            frontier_b,
            current_n,
            next_n,
            certified_n,
            iterations,
            converged,
            overflow,
            np.int32(nvox),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            INF_DISTANCE,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    status = {
        "iterations": int(iterations.get()[0]),
        "converged": bool(int(converged.get()[0])),
        "overflow": bool(int(overflow.get()[0])),
        "certified_voxels": int(certified_n.get()[0]),
        "grid_blocks": int(blocks),
        "resident_blocks_per_sm": int(resident_blocks_per_sm),
        "resident_limit_per_sm": int(resident_limit),
    }
    if status["overflow"]:
        raise RuntimeError("six-neighbour certificate frontier exceeded its voxel capacity")
    if not status["converged"]:
        raise RuntimeError("six-neighbour certificate frontier exceeded the voxel count")
    return certified, status


def _run_cooperative_relax(
    *,
    mask,
    active_ids,
    seed_label,
    state,
    clamp_seeds: bool,
    nseed: int,
    shape: tuple[int, int, int],
    block_size: int,
    max_iterations: int,
    blocks_per_sm: int = 1,
    wait_for_status: bool = True,
):
    import cupy as cp

    nactive = int(active_ids.size)
    if nactive == 0:
        return 0, True
    device = int(cp.cuda.runtime.getDevice())
    cooperative = int(
        cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrCooperativeLaunch, device
        )
    )
    if not cooperative:
        return 0, False

    D, H, W = shape
    block = int(block_size)
    kernel = _cooperative_relax_kernel()
    blocks, _, _ = _cooperative_grid_size(
        kernel,
        nitems=nactive,
        block_size=block,
        blocks_per_sm=blocks_per_sm,
    )
    changed = cp.empty(1, dtype=cp.int32)
    iterations = cp.empty(1, dtype=cp.int32)
    converged = cp.empty(1, dtype=cp.int32)
    kernel(
        (blocks,),
        (block,),
        (
            mask,
            active_ids,
            np.int32(nactive),
            seed_label,
            state,
            changed,
            iterations,
            converged,
            np.int32(max_iterations),
            np.int32(1 if clamp_seeds else 0),
            np.int32(nseed),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            INF_DISTANCE,
        ),
    )
    if not wait_for_status:
        return -1, None
    cp.cuda.Stream.null.synchronize()
    return int(iterations.get()[0]), bool(int(converged.get()[0]))


def _run_cooperative_roi_relax(
    *,
    active_ids,
    state,
    shape: tuple[int, int, int],
    block_size: int,
    max_iterations: int,
    blocks_per_sm: int = 1,
    wait_for_status: bool = True,
    status_buffer=None,
    coordinate_method: str = "divide",
    launch_info: dict | None = None,
):
    import cupy as cp

    nactive = int(active_ids.size)
    if nactive == 0:
        return 0, True
    device = int(cp.cuda.runtime.getDevice())
    cooperative = int(
        cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrCooperativeLaunch, device
        )
    )
    if not cooperative:
        return 0, False

    D, H, W = shape
    block = int(block_size)
    coordinate_backend = str(coordinate_method).strip().lower()
    if coordinate_backend in {
        "power2",
        "power2_warp_vote",
    }:
        if H <= 0 or W <= 0 or (H & (H - 1)) or (W & (W - 1)):
            raise ValueError("power2 closure coordinates require power-of-two H and W")
        if coordinate_backend == "power2_warp_vote":
            kernel = _cooperative_roi_relax_power2_warp_vote_kernel()
        else:
            kernel = _cooperative_roi_relax_power2_kernel()
    elif coordinate_backend == "divide":
        kernel = _cooperative_roi_relax_kernel()
    else:
        raise ValueError(
            "coordinate_method must be 'divide', 'power2', or "
            "'power2_warp_vote'"
        )
    blocks, resident, resident_limit = _cooperative_grid_size(
        kernel,
        nitems=nactive,
        block_size=block,
        blocks_per_sm=blocks_per_sm,
    )
    if launch_info is not None:
        launch_info.update(
            {
                "grid_blocks": int(blocks),
                "resident_blocks_per_sm": int(resident),
                "resident_limit_per_sm": int(resident_limit),
                "threads_per_block": int(block),
            }
        )
    if status_buffer is None:
        status = cp.empty(3, dtype=cp.int32)
    else:
        status = cp.asarray(status_buffer, dtype=cp.int32)
        if int(status.size) < 3:
            raise ValueError("status_buffer must contain at least three int32 values")
    arguments = [
        active_ids,
        np.int32(nactive),
        state,
        status[0:1],
        status[1:2],
        status[2:3],
        np.int32(max_iterations),
        np.int32(D),
        np.int32(H),
        np.int32(W),
    ]
    if coordinate_backend in {
        "power2",
        "power2_warp_vote",
    }:
        arguments.extend(
            [
                np.int32(int(math.log2(W))),
                np.int32(int(math.log2(H)) + int(math.log2(W))),
            ]
        )
    arguments.append(INF_DISTANCE)
    kernel((blocks,), (block,), tuple(arguments))
    if not wait_for_status:
        return -1, None
    host_status = status.get()
    return int(host_status[1]), bool(int(host_status[2]))


def _fixed_point_kernel():
    return _kernel(
        "ptv_fixed_point_check6",
        r'''
        extern "C" __global__
        void ptv_fixed_point_check6(
            const unsigned char* mask,
            const int* seed_label,
            const unsigned long long* state,
            int* bad,
            const int nvox,
            const int nseed,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox || !mask[idx]) return;
            unsigned long long current = state[idx];
            if ((unsigned int)(current >> 32) >= inf_distance) {
                atomicAdd(bad, 1);
                return;
            }
            int seed = seed_label[idx];
            if (seed >= 0 && seed < nseed) {
                unsigned long long expected = (unsigned long long)(unsigned int)seed;
                if (current != expected) atomicAdd(bad, 1);
                return;
            }

            unsigned long long best = 0xFFFFFFFFFFFFFFFFULL;
            const unsigned long long unit = 1ULL << 32;
            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;
            int neighbor[6];
            int count = 0;
            if (z > 0) neighbor[count++] = idx - HW;
            if (z + 1 < D) neighbor[count++] = idx + HW;
            if (y > 0) neighbor[count++] = idx - W;
            if (y + 1 < H) neighbor[count++] = idx + W;
            if (x > 0) neighbor[count++] = idx - 1;
            if (x + 1 < W) neighbor[count++] = idx + 1;
            for (int q = 0; q < count; ++q) {
                int nb = neighbor[q];
                if (!mask[nb]) continue;
                unsigned long long source = state[nb];
                if ((unsigned int)(source >> 32) >= inf_distance) continue;
                unsigned long long candidate = source + unit;
                if (candidate < best) best = candidate;
            }
            if (current != best) atomicAdd(bad, 1);
        }
        ''',
    )


def _fixed_point_seed_coordinates_kernel():
    return _kernel(
        "ptv_fixed_point_seed_coordinates_check6",
        r'''
        extern "C" __global__
        void ptv_fixed_point_seed_coordinates_check6(
            const unsigned char* mask,
            const int* seeds,
            const unsigned long long* state,
            int* bad,
            const int nvox,
            const int nseed,
            const int D,
            const int H,
            const int W,
            const unsigned int inf_distance
        ) {
            int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
            if (idx >= nvox || !mask[idx]) return;
            unsigned long long current = state[idx];
            unsigned int distance = (unsigned int)(current >> 32);
            int owner = (int)(unsigned int)current;
            if (distance >= inf_distance || owner < 0 || owner >= nseed) {
                atomicAdd(bad, 1);
                return;
            }
            if (distance == 0U) {
                int z = seeds[3 * owner + 0];
                int y = seeds[3 * owner + 1];
                int x = seeds[3 * owner + 2];
                int seed_idx = (z * H + y) * W + x;
                unsigned long long expected =
                    (unsigned long long)(unsigned int)owner;
                if (seed_idx != idx || current != expected) atomicAdd(bad, 1);
                return;
            }

            unsigned long long best = 0xFFFFFFFFFFFFFFFFULL;
            const unsigned long long unit = 1ULL << 32;
            int HW = H * W;
            int z = idx / HW;
            int rem = idx - z * HW;
            int y = rem / W;
            int x = rem - y * W;
            int neighbor[6];
            int count = 0;
            if (z > 0) neighbor[count++] = idx - HW;
            if (z + 1 < D) neighbor[count++] = idx + HW;
            if (y > 0) neighbor[count++] = idx - W;
            if (y + 1 < H) neighbor[count++] = idx + W;
            if (x > 0) neighbor[count++] = idx - 1;
            if (x + 1 < W) neighbor[count++] = idx + 1;
            for (int q = 0; q < count; ++q) {
                int nb = neighbor[q];
                if (!mask[nb]) continue;
                unsigned long long source = state[nb];
                if ((unsigned int)(source >> 32) >= inf_distance) continue;
                unsigned long long candidate = source + unit;
                if (candidate < best) best = candidate;
            }
            if (current != best) atomicAdd(bad, 1);
        }
        ''',
    )


def exact_persistent_frontier_gpu_6(
    mask,
    seeds_zyx,
    *,
    block_size: int = 256,
    blocks_per_sm: int = 1,
    validate: bool = True,
    return_float64: bool = False,
    return_stats: bool = False,
):
    """Exact six-neighbour ownership using one cooperative device-side BFS."""
    import cupy as cp

    global EXACT_PERSISTENT_BFS6_LAST_STATS

    total_start = time.perf_counter()
    mask_u8 = cp.asarray(mask, dtype=cp.uint8)
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")

    D, H, W = (int(value) for value in mask_u8.shape)
    nvox = int(mask_u8.size)
    nseed = int(seeds.shape[0])
    if nseed == 0:
        raise ValueError("at least one prescribed site is required")

    device = int(cp.cuda.runtime.getDevice())
    cooperative = int(
        cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrCooperativeLaunch, device
        )
    )
    if not cooperative:
        labels, distances = exact_frontier_dijkstra_gpu_6(
            mask_u8,
            seeds,
            block_size=block_size,
            validate=validate,
            return_float64=return_float64,
        )
        EXACT_PERSISTENT_BFS6_LAST_STATS = {
            "backend": "frontier_fallback_no_cooperative_launch",
            "iterations": int(EXACT_FRONTIER_LAST_ITERS),
            "max_distance": int(EXACT_FRONTIER_LAST_MAX_DIST),
            "fixed_point_residuals": -1,
            "total_s": float(time.perf_counter() - total_start),
        }
        if return_stats:
            return labels, distances, dict(EXACT_PERSISTENT_BFS6_LAST_STATS)
        return labels, distances

    flat_mask = mask_u8.ravel()
    block = int(block_size)
    if block < 32 or block > 1024 or block % 32 != 0:
        raise ValueError("block_size must be a warp-aligned value in [32, 1024]")
    grid_vox = (max(1, math.ceil(nvox / block)),)
    grid_seed = (max(1, math.ceil(nseed / block)),)

    state = cp.empty(nvox, dtype=cp.uint64)
    seed_label = cp.full(nvox, np.int32(1_073_741_823), dtype=cp.int32)
    frontier_a = cp.empty(nvox, dtype=cp.int32)
    frontier_b = cp.empty_like(frontier_a)
    current_n = cp.zeros(1, dtype=cp.int32)
    next_n = cp.empty(1, dtype=cp.int32)
    bad_seed = cp.zeros(1, dtype=cp.int32)
    iterations = cp.empty(1, dtype=cp.int32)
    converged = cp.empty(1, dtype=cp.int32)
    overflow = cp.empty(1, dtype=cp.int32)

    _packed_init_kernel()(grid_vox, (block,), (state, np.int32(nvox), INF_PACK))
    _packed_seed_frontier_kernel()(
        grid_seed,
        (block,),
        (
            flat_mask,
            seeds,
            state,
            seed_label,
            frontier_a,
            current_n,
            bad_seed,
            np.int32(nseed),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            INF_PACK,
        ),
    )

    kernel = _cooperative_bfs_kernel()
    blocks, resident_blocks_per_sm, resident_limit = _cooperative_grid_size(
        kernel,
        nitems=nvox,
        block_size=block,
        blocks_per_sm=blocks_per_sm,
    )
    kernel(
        (blocks,),
        (block,),
        (
            flat_mask,
            state,
            frontier_a,
            frontier_b,
            current_n,
            next_n,
            iterations,
            converged,
            overflow,
            np.int32(nvox),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            INF_DISTANCE,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    if int(bad_seed.get()[0]) != 0:
        raise RuntimeError("an input site is outside the pore mask")
    if int(overflow.get()[0]) != 0:
        raise RuntimeError("persistent six-neighbour frontier exceeded its voxel capacity")
    converged_value = bool(int(converged.get()[0]))
    iteration_count = int(iterations.get()[0])
    if not converged_value:
        raise RuntimeError("persistent six-neighbour frontier exceeded the voxel count")

    fixed_point_residuals = -1
    if validate:
        bad_fixed_point = cp.zeros(1, dtype=cp.int32)
        _fixed_point_kernel()(
            grid_vox,
            (block,),
            (
                flat_mask,
                seed_label,
                state,
                bad_fixed_point,
                np.int32(nvox),
                np.int32(nseed),
                np.int32(D),
                np.int32(H),
                np.int32(W),
                INF_DISTANCE,
            ),
        )
        fixed_point_residuals = int(bad_fixed_point.get()[0])
        if fixed_point_residuals != 0:
            raise RuntimeError(
                f"persistent six-neighbour frontier has {fixed_point_residuals} fixed-point residuals"
            )

    distance_u32 = (state >> cp.uint64(32)).astype(cp.uint32)
    label_u32 = (state & cp.uint64(0xFFFFFFFF)).astype(cp.uint32)
    label = label_u32.view(cp.int32)
    distance_dtype = cp.float64 if return_float64 else cp.float32
    distance = distance_u32.astype(distance_dtype)
    pore = flat_mask != 0
    label = cp.where(pore, label, cp.int32(-1))
    distance = cp.where(pore, distance, cp.inf)
    if return_stats:
        cp.cuda.Stream.null.synchronize()

    EXACT_PERSISTENT_BFS6_LAST_STATS = {
        "backend": "cooperative_persistent_frontier6",
        "iterations": iteration_count,
        "max_distance": max(0, iteration_count - 1),
        "fixed_point_residuals": int(fixed_point_residuals),
        "block_size": block,
        "grid_blocks": int(blocks),
        "requested_blocks_per_sm": int(blocks_per_sm),
        "resident_blocks_per_sm": int(resident_blocks_per_sm),
        "resident_limit_per_sm": int(resident_limit),
        "total_s": float(time.perf_counter() - total_start),
    }
    result = (label.reshape(mask_u8.shape), distance.reshape(mask_u8.shape))
    if return_stats:
        return (*result, dict(EXACT_PERSISTENT_BFS6_LAST_STATS))
    return result


def exact_persistent_relax_gpu_6(
    mask,
    seeds_zyx,
    *,
    block_size: int = 256,
    validate: bool = True,
    return_float64: bool = False,
    return_stats: bool = False,
):
    """Exact six-neighbour ownership using a device-side persistent fixed point."""
    import cupy as cp

    global EXACT_PERSISTENT6_LAST_STATS

    total_start = time.perf_counter()
    mask_u8 = cp.asarray(mask, dtype=cp.uint8)
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")

    D, H, W = (int(value) for value in mask_u8.shape)
    nvox = int(mask_u8.size)
    nseed = int(seeds.shape[0])
    if nseed == 0:
        raise ValueError("at least one prescribed site is required")

    flat_mask = mask_u8.ravel()
    block = int(block_size)
    grid_vox = (max(1, math.ceil(nvox / block)),)
    grid_seed = (max(1, math.ceil(nseed / block)),)
    state = cp.empty(nvox, dtype=cp.uint64)
    seed_label = cp.full(nvox, np.int32(1_073_741_823), dtype=cp.int32)
    bad_seed = cp.zeros(1, dtype=cp.int32)
    _packed_init_kernel()(grid_vox, (block,), (state, np.int32(nvox), INF_PACK))
    _packed_seed_kernel()(
        grid_seed,
        (block,),
        (
            flat_mask,
            seeds,
            state,
            seed_label,
            bad_seed,
            np.int32(nseed),
            np.int32(D),
            np.int32(H),
            np.int32(W),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    if int(bad_seed.get()[0]) != 0:
        raise RuntimeError("an input site is outside the pore mask")

    pore_ids = cp.flatnonzero(flat_mask != 0).astype(cp.int32)
    npore = int(pore_ids.size)
    iterations = 0
    converged = False
    while iterations < nvox and not converged:
        chunk = min(COOPERATIVE_RELAX_CHUNK, nvox - iterations)
        completed, converged = _run_cooperative_relax(
            mask=flat_mask,
            active_ids=pore_ids,
            seed_label=seed_label,
            state=state,
            clamp_seeds=True,
            nseed=nseed,
            shape=(D, H, W),
            block_size=block,
            max_iterations=chunk,
        )
        if completed == 0 and not converged:
            labels, distances = exact_frontier_dijkstra_gpu_6(
                mask_u8,
                seeds,
                block_size=block,
                validate=validate,
                return_float64=return_float64,
            )
            EXACT_PERSISTENT6_LAST_STATS = {
                "backend": "frontier_fallback_no_cooperative_launch",
                "pore_voxels": npore,
                "iterations": 0,
                "fixed_point_residuals": -1,
                "total_s": float(time.perf_counter() - total_start),
            }
            if return_stats:
                return labels, distances, dict(EXACT_PERSISTENT6_LAST_STATS)
            return labels, distances
        iterations += completed
    if not converged:
        raise RuntimeError("persistent six-neighbour relaxation exceeded the voxel count")

    fixed_point_residuals = -1
    if validate:
        bad_fixed_point = cp.zeros(1, dtype=cp.int32)
        _fixed_point_kernel()(
            grid_vox,
            (block,),
            (
                flat_mask,
                seed_label,
                state,
                bad_fixed_point,
                np.int32(nvox),
                np.int32(nseed),
                np.int32(D),
                np.int32(H),
                np.int32(W),
                INF_DISTANCE,
            ),
        )
        fixed_point_residuals = int(bad_fixed_point.get()[0])
        if fixed_point_residuals != 0:
            raise RuntimeError(
                f"persistent six-neighbour result has {fixed_point_residuals} fixed-point residuals"
            )

    distance_u32 = (state >> cp.uint64(32)).astype(cp.uint32)
    label_u32 = (state & cp.uint64(0xFFFFFFFF)).astype(cp.uint32)
    label = label_u32.view(cp.int32)
    distance_dtype = cp.float64 if return_float64 else cp.float32
    distance = distance_u32.astype(distance_dtype)
    pore = flat_mask != 0
    label = cp.where(pore, label, cp.int32(-1))
    distance = cp.where(pore, distance, cp.inf)
    cp.cuda.Stream.null.synchronize()

    EXACT_PERSISTENT6_LAST_STATS = {
        "backend": "cooperative_persistent_relax6",
        "pore_voxels": npore,
        "iterations": int(iterations),
        "fixed_point_residuals": int(fixed_point_residuals),
        "total_s": float(time.perf_counter() - total_start),
    }
    result = (label.reshape(mask_u8.shape), distance.reshape(mask_u8.shape))
    if return_stats:
        return (*result, dict(EXACT_PERSISTENT6_LAST_STATS))
    return result


def certified_l1_roi_frontier_gpu_6(
    mask,
    seeds_zyx,
    *,
    block_size: int = 256,
    validate: bool = True,
    return_float64: bool = False,
    return_stats: bool = False,
    lower_bound_method: str = "jump_cooperative",
    closure_method: str = "cooperative",
    cooperative_blocks_per_sm: int = 4,
    certificate_method: str = "auto",
    certificate_blocks_per_sm: int = 1,
    profile_stages: bool = False,
    output_method: str = "fused",
    certificate_line_bits=None,
    pore_count_method: str = "auto",
    pore_voxel_count: int | None = None,
    defer_closure_status: bool | None = None,
    closure_kernel_method: str = "roi_single_writer",
    closure_block_size: int | None = None,
    status_transfer_method: str = "packed",
    fuse_state_init: bool | None = None,
    reuse_status_buffer: bool = True,
    closure_coordinate_method: str = "auto",
    lower_cooperative_blocks_per_sm: int = 8,
    lower_warp_block_size: int = 256,
):
    """Exact arbitrary-site ownership using an L1 certificate and six-neighbour ROI closure.

    The production path uses a cooperative three-axis L1 jump lower bound,
    a cached 64-bit line certificate when supplied, a single-writer
    cooperative ROI closure, and fused packed-state output. Diagnostic
    alternatives remain selectable for same-version audits. A voxel is frozen
    only when the winning lower-bound site has a certified monotone
    face-neighbour path. Remaining pore voxels are relaxed to the unique
    six-neighbour fixed point. Packed states order first by distance and then
    by site id, matching the deterministic exact-frontier tie rule.
    """
    import cupy as cp

    global CERTIFIED_ROI6_LAST_STATS

    total_start = time.perf_counter()
    mask_u8 = cp.asarray(mask, dtype=cp.uint8)
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")

    D, H, W = (int(value) for value in mask_u8.shape)
    nvox = int(mask_u8.size)
    nseed = int(seeds.shape[0])
    if nseed == 0:
        raise ValueError("at least one prescribed site is required")
    flat_mask = mask_u8.ravel()
    block = int(block_size)
    closure_block = block if closure_block_size is None else int(closure_block_size)
    if block <= 0 or closure_block <= 0:
        raise ValueError("block sizes must be positive")
    grid_vox = (max(1, math.ceil(nvox / block)),)
    grid_seed = (max(1, math.ceil(nseed / block)),)

    profile = bool(profile_stages)
    closure_status_deferred = (
        not (validate or return_stats or profile)
        if defer_closure_status is None
        else bool(defer_closure_status)
    )
    if validate or return_stats or profile:
        closure_status_deferred = False
    lower_start = time.perf_counter()
    lower_state = cp.empty(nvox, dtype=cp.uint64)
    lower_method = str(lower_bound_method).strip().lower()
    integrated_warp64_lower = lower_method == "jump_cooperative_warp64"
    seed_label = (
        None
        if integrated_warp64_lower
        else cp.full(nvox, np.int32(1_073_741_823), dtype=cp.int32)
    )
    state = cp.empty(nvox, dtype=cp.uint64) if integrated_warp64_lower else None
    status_transfer_backend = str(status_transfer_method).strip().lower()
    if status_transfer_backend not in {"separate", "packed"}:
        raise ValueError("status_transfer_method must be 'separate' or 'packed'")
    if status_transfer_backend == "packed":
        certificate_status = cp.zeros(3, dtype=cp.int32)
        bad_seed = certificate_status[0:1]
    else:
        certificate_status = None
        bad_seed = cp.zeros(1, dtype=cp.int32)
    lower_backend = lower_method
    jump_passes = 0
    jump_kernel_launches = 0
    lower_cooperative_status = None
    if integrated_warp64_lower:
        lower_cooperative_status = _run_cooperative_warp64_init_l1_jump(
            mask=flat_mask,
            seeds=seeds,
            lower_state=lower_state,
            closure_state=state,
            bad_seed=bad_seed,
            shape=(D, H, W),
            block_size=lower_warp_block_size,
            blocks_per_sm=lower_cooperative_blocks_per_sm,
        )
        if lower_cooperative_status is None:
            raise RuntimeError(
                "jump_cooperative_warp64 requires a cooperative CUDA device "
                "and a 64 x 64 x 64 domain"
            )
        lower_backend = "jump_cooperative_warp64_init"
        jump_passes = 18
        jump_kernel_launches = 1
    else:
        _packed_init_kernel()(
            grid_vox, (block,), (lower_state, np.int32(nvox), INF_PACK)
        )
        _packed_seed_kernel()(
            grid_seed,
            (block,),
            (
                flat_mask,
                seeds,
                lower_state,
                seed_label,
                bad_seed,
                np.int32(nseed),
                np.int32(D),
                np.int32(H),
                np.int32(W),
            ),
        )
    if lower_method == "scan":
        lower_backend = "scan_three_axis"
        grid_x = (max(1, math.ceil((D * H) / block)),)
        grid_y = (max(1, math.ceil((D * W) / block)),)
        grid_z = (max(1, math.ceil((H * W) / block)),)
        _l1_scan_x_kernel()(
            grid_x,
            (block,),
            (lower_state, np.int32(D), np.int32(H), np.int32(W), INF_DISTANCE),
        )
        _l1_scan_y_kernel()(
            grid_y,
            (block,),
            (lower_state, np.int32(D), np.int32(H), np.int32(W), INF_DISTANCE),
        )
        _l1_scan_z_kernel()(
            grid_z,
            (block,),
            (lower_state, np.int32(D), np.int32(H), np.int32(W), INF_DISTANCE),
        )
    elif lower_method in {"jump", "jump_cooperative"}:
        if lower_method == "jump_cooperative":
            lower_cooperative_status = _run_cooperative_fused_l1_jump(
                state=lower_state,
                shape=(D, H, W),
                blocks_per_sm=lower_cooperative_blocks_per_sm,
            )
        if lower_cooperative_status is not None:
            lower_backend = "jump_cooperative_fused"
            jump_passes = sum(
                int(math.ceil(math.log2(max(axis_length, 1))))
                for axis_length in (D, H, W)
            )
            jump_kernel_launches = 1
        elif max(D, H, W) <= 1024:
            lower_backend = "jump_three_axis_fused"
            fused_kernel = _l1_fused_jump_axis_kernel()
            for axis, axis_length in enumerate((D, H, W)):
                threads = _next_power_of_two(axis_length)
                nline = (H * W, D * W, D * H)[axis]
                fused_kernel(
                    (nline,),
                    (threads,),
                    (
                        lower_state,
                        np.int32(D),
                        np.int32(H),
                        np.int32(W),
                        np.int32(axis),
                        INF_DISTANCE,
                        INF_PACK,
                    ),
                    shared_mem=threads * np.dtype(np.uint64).itemsize,
                )
                jump_passes += int(math.ceil(math.log2(max(axis_length, 1))))
                jump_kernel_launches += 1
        else:
            lower_backend = "jump_global_passes"
            lower_scratch = cp.empty_like(lower_state)
            for axis, axis_length in enumerate((D, H, W)):
                jump = 1
                while jump < axis_length:
                    _l1_jump_axis_kernel()(
                        grid_vox,
                        (block,),
                        (
                            lower_state,
                            lower_scratch,
                            np.int32(nvox),
                            np.int32(D),
                            np.int32(H),
                            np.int32(W),
                            np.int32(axis),
                            np.int32(jump),
                            INF_DISTANCE,
                        ),
                    )
                    lower_state, lower_scratch = lower_scratch, lower_state
                    jump <<= 1
                    jump_passes += 1
                    jump_kernel_launches += 1
    elif not integrated_warp64_lower:
        raise ValueError(
            "lower_bound_method must be 'jump', 'jump_cooperative', "
            "'jump_cooperative_warp64', or 'scan'"
        )
    if profile:
        cp.cuda.Stream.null.synchronize()
        lower_s = float(time.perf_counter() - lower_start)
    else:
        lower_s = float("nan")

    certificate_start = time.perf_counter()
    certificate_backend = str(certificate_method).strip().lower()
    if certificate_backend == "auto":
        if certificate_line_bits is not None and max(D, H, W) <= 64:
            certificate_backend = "path_bitset"
        else:
            certificate_backend = "path_forward"
    certificate_state_fused = (
        certificate_backend != "wavefront"
        if fuse_state_init is None
        else bool(fuse_state_init)
    )
    if certificate_state_fused and certificate_backend == "wavefront":
        raise ValueError("fuse_state_init is not supported by the wavefront certificate")
    if integrated_warp64_lower and not certificate_state_fused:
        raise ValueError(
            "jump_cooperative_warp64 requires a fused path certificate"
        )
    if integrated_warp64_lower and certificate_backend not in {
        "path_bitset",
        "path_bitset_compact",
        "path_bitset_compact64",
    }:
        raise ValueError(
            "jump_cooperative_warp64 requires a bitset path certificate"
        )
    if state is None and certificate_state_fused:
        state = cp.empty(nvox, dtype=cp.uint64)
    certificate_state = state if state is not None else lower_state
    certificate_iterations = 0
    roi_ids = cp.empty(nvox, dtype=cp.int32)
    if certificate_status is not None:
        roi_n = certificate_status[1:2]
        pore_n = certificate_status[2:3]
    else:
        roi_n = cp.zeros(1, dtype=cp.int32)
        pore_n = cp.zeros(1, dtype=cp.int32)
    pore_count_request = str(pore_count_method).strip().lower()
    if pore_count_request not in {"auto", "atomic", "skip"}:
        raise ValueError("pore_count_method must be 'auto', 'atomic', or 'skip'")
    if pore_voxel_count is not None:
        supplied_pore_count = int(pore_voxel_count)
        if supplied_pore_count < 0 or supplied_pore_count > nvox:
            raise ValueError("pore_voxel_count is outside the valid voxel range")
    else:
        supplied_pore_count = None
    if supplied_pore_count is not None and pore_count_request != "atomic":
        pore_count_backend = "external"
    elif pore_count_request == "auto":
        pore_count_backend = "atomic" if return_stats or profile else "skip"
    else:
        pore_count_backend = pore_count_request
    count_pore = np.int32(1 if pore_count_backend == "atomic" else 0)
    if certificate_backend in {"path", "path_forward"}:
        certified = cp.empty(nvox, dtype=cp.int32)
        _monotone_certificate_kernel()(
            grid_vox,
            (block,),
            (
                flat_mask,
                lower_state,
                certificate_state,
                seeds,
                certified,
                roi_ids,
                roi_n,
                pore_n,
                np.int32(nseed),
                np.int32(nvox),
                np.int32(1 if certificate_backend == "path" else 0),
                count_pore,
                np.int32(1 if certificate_state_fused else 0),
                np.int32(D),
                np.int32(H),
                np.int32(W),
                INF_DISTANCE,
                INF_PACK,
            ),
        )
    elif certificate_backend in {
        "path_bitset",
        "path_bitset_compact",
        "path_bitset_compact64",
    }:
        if max(D, H, W) > 64:
            raise ValueError(
                "bitset certificate methods require every axis length to be at most 64"
            )
        if certificate_line_bits is None:
            x_bits, y_bits, z_bits = prepare_mask_line_bits64(mask_u8)
            certificate_bits_cached = False
        else:
            if len(certificate_line_bits) != 3:
                raise ValueError("certificate_line_bits must contain x, y, and z arrays")
            x_bits, y_bits, z_bits = certificate_line_bits
            if (
                int(x_bits.size) != D * H
                or int(y_bits.size) != D * W
                or int(z_bits.size) != H * W
            ):
                raise ValueError("certificate_line_bits do not match the mask shape")
            certificate_bits_cached = True
        compact_certificate = certificate_backend in {
            "path_bitset_compact",
            "path_bitset_compact64",
        }
        specialized_compact_certificate = (
            certificate_backend == "path_bitset_compact64"
        )
        if compact_certificate and not integrated_warp64_lower:
            raise ValueError(
                "path_bitset_compact requires preinitialized closure state"
            )
        if specialized_compact_certificate and (D, H, W) != (64, 64, 64):
            raise ValueError(
                "path_bitset_compact64 requires a 64-cubed domain"
            )
        if compact_certificate:
            certified = None
            power2_coordinates = (
                H > 0
                and W > 0
                and not (H & (H - 1))
                and not (W & (W - 1))
            )
            _bitset_compact_certificate_kernel(
                specialized_64=specialized_compact_certificate
            )(
                grid_vox,
                (block,),
                (
                    flat_mask,
                    lower_state,
                    certificate_state,
                    x_bits,
                    y_bits,
                    z_bits,
                    seeds,
                    roi_ids,
                    roi_n,
                    np.int32(nseed),
                    np.int32(nvox),
                    np.int32(D),
                    np.int32(H),
                    np.int32(W),
                    np.int32(1 if power2_coordinates else 0),
                    np.int32(int(math.log2(W)) if power2_coordinates else 0),
                    np.int32(
                        int(math.log2(H)) + int(math.log2(W))
                        if power2_coordinates
                        else 0
                    ),
                    INF_DISTANCE,
                    INF_PACK,
                ),
            )
        else:
            certified = cp.empty(nvox, dtype=cp.int32)
            _bitset_certificate_kernel()(
                grid_vox,
                (block,),
                (
                    flat_mask,
                    lower_state,
                    certificate_state,
                    x_bits,
                    y_bits,
                    z_bits,
                    seeds,
                    certified,
                    roi_ids,
                    roi_n,
                    pore_n,
                    np.int32(nseed),
                    np.int32(nvox),
                    count_pore,
                    np.int32(1 if certificate_state_fused else 0),
                    np.int32(D),
                    np.int32(H),
                    np.int32(W),
                    INF_DISTANCE,
                    INF_PACK,
                ),
            )
    elif certificate_backend == "wavefront":
        certified, certificate_status = _run_cooperative_certificate(
            mask=flat_mask,
            lower_state=lower_state,
            seed_label=seed_label,
            nseed=nseed,
            shape=(D, H, W),
            block_size=block,
            blocks_per_sm=certificate_blocks_per_sm,
        )
        certificate_iterations = int(certificate_status["iterations"])
        _roi_compact_kernel()(
            grid_vox,
            (block,),
            (flat_mask, certified, roi_ids, roi_n, pore_n, np.int32(nvox)),
        )
    else:
        raise ValueError(
            "certificate_method must be 'path', 'path_forward', "
            "'path_bitset', 'path_bitset_compact', "
            "'path_bitset_compact64', or 'wavefront'"
        )
    if certificate_backend not in {
        "path_bitset",
        "path_bitset_compact",
        "path_bitset_compact64",
    }:
        certificate_bits_cached = False
    if certificate_status is not None:
        host_certificate_status = certificate_status.get()
        bad_seed_count = int(host_certificate_status[0])
        nroi = int(host_certificate_status[1])
        npore_atomic = int(host_certificate_status[2])
    else:
        cp.cuda.Stream.null.synchronize()
        bad_seed_count = int(bad_seed.get()[0])
        nroi = int(roi_n.get()[0])
        npore_atomic = int(pore_n.get()[0])
    if bad_seed_count != 0:
        raise RuntimeError("an input site is outside the pore mask")
    if pore_count_backend == "atomic":
        npore = npore_atomic
    elif pore_count_backend == "external":
        npore = int(supplied_pore_count)
    else:
        npore = -1
    roi_ids = roi_ids[:nroi]
    ncertified = npore - nroi if npore >= 0 else -1
    certificate_s = float(time.perf_counter() - certificate_start)

    closure_start = time.perf_counter()
    if state is None:
        state = cp.empty(nvox, dtype=cp.uint64)
        _roi_state_init_kernel()(
            grid_vox,
            (block,),
            (flat_mask, certified, lower_state, state, np.int32(nvox), INF_PACK),
        )
    closure_iterations = 0
    closure_launch_status = {} if return_stats or profile else None
    closure_backend = str(closure_method).strip().lower()
    if closure_backend not in {"host", "cooperative"}:
        raise ValueError("closure_method must be 'host' or 'cooperative'")
    closure_kernel_backend = str(closure_kernel_method).strip().lower()
    if closure_kernel_backend not in {"generic_atomic", "roi_single_writer"}:
        raise ValueError(
            "closure_kernel_method must be 'generic_atomic' or 'roi_single_writer'"
        )
    closure_coordinate_backend = str(closure_coordinate_method).strip().lower()
    if closure_coordinate_backend == "auto":
        closure_coordinate_backend = (
            "power2"
            if H > 0
            and W > 0
            and not (H & (H - 1))
            and not (W & (W - 1))
            else "divide"
        )
    if closure_coordinate_backend not in {
        "divide",
        "power2",
        "power2_warp_vote",
    }:
        raise ValueError(
            "closure_coordinate_method must be 'auto', 'divide', 'power2', "
            "or 'power2_warp_vote'"
        )
    if nroi > 0 and closure_backend == "cooperative":
        if closure_kernel_backend == "roi_single_writer":
            closure_runner = _run_cooperative_roi_relax
            closure_runner_kwargs = {
                "active_ids": roi_ids,
                "state": state,
                "shape": (D, H, W),
                "block_size": closure_block,
                "blocks_per_sm": cooperative_blocks_per_sm,
                "status_buffer": (
                    certificate_status
                    if reuse_status_buffer and certificate_status is not None
                    else None
                ),
                "coordinate_method": closure_coordinate_backend,
                "launch_info": closure_launch_status,
            }
        else:
            if seed_label is None:
                raise ValueError(
                    "generic_atomic closure requires a materialized seed-label field"
                )
            closure_runner = _run_cooperative_relax
            closure_runner_kwargs = {
                "mask": flat_mask,
                "active_ids": roi_ids,
                "seed_label": seed_label,
                "state": state,
                "clamp_seeds": False,
                "nseed": nseed,
                "shape": (D, H, W),
                "block_size": closure_block,
                "blocks_per_sm": cooperative_blocks_per_sm,
            }
        if closure_status_deferred:
            completed, converged = closure_runner(
                max_iterations=nvox,
                wait_for_status=False,
                **closure_runner_kwargs,
            )
            if completed == 0 and converged is False:
                closure_backend = "host_fallback_no_cooperative_launch"
            else:
                closure_iterations = -1
        else:
            converged = False
            while closure_iterations < nvox and not converged:
                chunk = min(COOPERATIVE_RELAX_CHUNK, nvox - closure_iterations)
                completed, converged = closure_runner(
                    max_iterations=chunk,
                    **closure_runner_kwargs,
                )
                if completed == 0 and not converged:
                    closure_backend = "host_fallback_no_cooperative_launch"
                    break
                closure_iterations += completed
            if closure_backend == "cooperative" and not converged:
                raise RuntimeError(
                    "cooperative six-neighbour ROI closure exceeded the voxel count"
                )
    if nroi > 0 and closure_backend != "cooperative":
        scratch = state.copy()
        any_changed = cp.empty(1, dtype=cp.int32)
        grid_roi = (max(1, math.ceil(nroi / closure_block)),)
        start_iteration = int(closure_iterations)
        for closure_iterations in range(start_iteration + 1, nvox + 1):
            any_changed.fill(0)
            _roi_relax_kernel()(
                grid_roi,
                (closure_block,),
                (
                    flat_mask,
                    roi_ids,
                    np.int32(nroi),
                    state,
                    scratch,
                    any_changed,
                    np.int32(D),
                    np.int32(H),
                    np.int32(W),
                    INF_DISTANCE,
                ),
            )
            state, scratch = scratch, state
            if int(any_changed.get()[0]) == 0:
                break
        else:
            raise RuntimeError("six-neighbour ROI closure exceeded the voxel count")
    if profile:
        cp.cuda.Stream.null.synchronize()
        closure_s = float(time.perf_counter() - closure_start)
    else:
        closure_s = float("nan")

    validation_start = time.perf_counter()
    bad_fixed_point = cp.zeros(1, dtype=cp.int32)
    if validate:
        if seed_label is None:
            _fixed_point_seed_coordinates_kernel()(
                grid_vox,
                (block,),
                (
                    flat_mask,
                    seeds,
                    state,
                    bad_fixed_point,
                    np.int32(nvox),
                    np.int32(nseed),
                    np.int32(D),
                    np.int32(H),
                    np.int32(W),
                    INF_DISTANCE,
                ),
            )
        else:
            _fixed_point_kernel()(
                grid_vox,
                (block,),
                (
                    flat_mask,
                    seed_label,
                    state,
                    bad_fixed_point,
                    np.int32(nvox),
                    np.int32(nseed),
                    np.int32(D),
                    np.int32(H),
                    np.int32(W),
                    INF_DISTANCE,
                ),
            )
        fixed_point_residuals = int(bad_fixed_point.get()[0])
        if fixed_point_residuals != 0:
            raise RuntimeError(
                f"six-neighbour ROI result has {fixed_point_residuals} fixed-point residuals"
            )
    else:
        fixed_point_residuals = -1
    if profile:
        cp.cuda.Stream.null.synchronize()
        validation_s = float(time.perf_counter() - validation_start)
    else:
        validation_s = float("nan")

    output_backend = str(output_method).strip().lower()
    if output_backend == "cupy":
        distance_u32 = (state >> cp.uint64(32)).astype(cp.uint32)
        label_u32 = (state & cp.uint64(0xFFFFFFFF)).astype(cp.uint32)
        label = label_u32.view(cp.int32)
        distance_dtype = cp.float64 if return_float64 else cp.float32
        distance = distance_u32.astype(distance_dtype)
        pore = flat_mask != 0
        label = cp.where(pore, label, cp.int32(-1))
        distance = cp.where(pore, distance, cp.inf)
    elif output_backend == "fused":
        label = cp.empty(nvox, dtype=cp.int32)
        if return_float64:
            distance = cp.empty(nvox, dtype=cp.float64)
            output_kernel = _packed_output_f64_kernel()
        else:
            distance = cp.empty(nvox, dtype=cp.float32)
            output_kernel = _packed_output_f32_kernel()
        output_kernel(
            grid_vox,
            (block,),
            (flat_mask, state, label, distance, np.int32(nvox)),
        )
    else:
        raise ValueError("output_method must be 'cupy' or 'fused'")
    if return_stats or profile:
        cp.cuda.Stream.null.synchronize()

    CERTIFIED_ROI6_LAST_STATS = {
        "pore_voxels": npore,
        "certified_voxels": ncertified,
        "roi_voxels": nroi,
        "certified_fraction": (
            float(ncertified / max(npore, 1)) if npore >= 0 else float("nan")
        ),
        "roi_fraction": float(nroi / max(npore, 1)) if npore >= 0 else float("nan"),
        "pore_count_method": pore_count_backend,
        "closure_iterations": int(closure_iterations),
        "fixed_point_residuals": int(fixed_point_residuals),
        "lower_bound_method": lower_method,
        "lower_bound_backend": lower_backend,
        "lower_bound_jump_passes": int(jump_passes),
        "lower_bound_jump_kernel_launches": int(jump_kernel_launches),
        "lower_bound_cooperative_status": lower_cooperative_status,
        "lower_bound_cooperative_blocks_per_sm": int(
            lower_cooperative_blocks_per_sm
        ),
        "lower_bound_warp_block_size": int(lower_warp_block_size),
        "lower_bound_initialization_fused": bool(integrated_warp64_lower),
        "closure_backend": closure_backend,
        "closure_status_deferred": bool(closure_status_deferred),
        "closure_kernel_method": closure_kernel_backend,
        "closure_block_size": int(closure_block),
        "cooperative_blocks_per_sm": int(cooperative_blocks_per_sm),
        "closure_launch_status": closure_launch_status,
        "certificate_method": certificate_backend,
        "certificate_bits_cached": bool(certificate_bits_cached),
        "certificate_state_init_fused": bool(certificate_state_fused),
        "status_transfer_method": status_transfer_backend,
        "closure_status_buffer_reused": bool(
            reuse_status_buffer
            and closure_kernel_backend == "roi_single_writer"
            and certificate_status is not None
        ),
        "closure_coordinate_method": closure_coordinate_backend,
        "certificate_iterations": int(certificate_iterations),
        "certificate_blocks_per_sm": int(certificate_blocks_per_sm),
        "output_method": output_backend,
        "l1_lower_bound_s": lower_s,
        "path_certificate_s": certificate_s,
        "roi_closure_s": closure_s,
        "fixed_point_check_s": validation_s,
        "total_s": float(time.perf_counter() - total_start),
    }
    result = (label.reshape(mask_u8.shape), distance.reshape(mask_u8.shape))
    if return_stats:
        return (*result, dict(CERTIFIED_ROI6_LAST_STATS))
    return result


def certified_roi_jfa_gpu_6(mask, seeds_zyx, **kwargs):
    """Six-neighbour arbitrary-site ROI-JFA with an exact L1 jump lower bound."""
    shape = tuple(int(value) for value in getattr(mask, "shape", ()))
    if shape == (64, 64, 64):
        kwargs.setdefault("lower_bound_method", "jump_cooperative_warp64")
        kwargs.setdefault("lower_warp_block_size", 256)
        kwargs.setdefault("lower_cooperative_blocks_per_sm", 8)
        kwargs.setdefault("certificate_method", "path_bitset_compact64")
        kwargs.setdefault("closure_coordinate_method", "power2_warp_vote")
        kwargs.setdefault("cooperative_blocks_per_sm", 8)
    else:
        kwargs.setdefault("lower_bound_method", "jump_cooperative")
    kwargs.setdefault("closure_block_size", 64)
    return certified_l1_roi_frontier_gpu_6(mask, seeds_zyx, **kwargs)
