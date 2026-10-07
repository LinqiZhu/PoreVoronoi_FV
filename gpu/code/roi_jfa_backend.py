# Optional ROI-JFA ownership backend of the GPU pipeline.
# Exported from the ROI-JFA definition cell of the partition notebook (not part of this repository).
# The flow-side code imports the ROI-JFA functions from this module.
# The functions are kept exactly as exported.

import numpy as np
import cupy as cp
import time
import math
from scipy import ndimage
ROI_JFA_LAST_VIZ_TIME = 0.0  # accumulated visualisation time (s) in the last geodesic_voronoi_roi_jfa call
ROI_JFA_LAST_GPU_TIME = 0.0      # last ROI-JFA call: pure GPU solve time from CUDA events (s, without visualisation)
ROI_JFA_LAST_RECORD_TIME = 0.0   # last ROI-JFA call: time spent recording snapshots for visualisation (s, without rendering)
EXACT_LAST_GPU_TIME = 0.0   # CUDA event time of exact_geodesic_voronoi_gpu (s)
OAJFA_LAST_GPU_TIME = 0.0   # CUDA event time of geodesic_voronoi_oajfa_cuda (s)
ROI_JFA_LAST_VIZ_TIME = 0.0
ROI_JFA_LAST_GPU_TIME = 0.0
ROI_JFA_LAST_RECORD_TIME = 0.0
EXACT_LAST_GPU_TIME = 0.0
OAJFA_LAST_GPU_TIME = 0.0

# wall-time decomposition for ROI-JFA (prediction path only)
ROI_JFA_LAST_TSTAMP_WALL = 0.0   # stamping wall-time
ROI_JFA_LAST_TJFA_WALL   = 0.0   # JFA wall-time (excluding closure; may include relax unless separated)
ROI_JFA_LAST_TCLOSE_WALL = 0.0   # closure wall-time
ROI_JFA_LAST_TPRED_WALL  = 0.0   # t_stamp + t_jfa + t_close (+ relax if included in t_jfa)
ROI_JFA_LAST_TRADII_WALL = 0.0
ROI_JFA_LAST_TCAND_WALL = 0.0
ROI_JFA_LAST_TINIT_WALL = 0.0
ROI_JFA_LAST_TBUBBLE_WALL = 0.0
ROI_JFA_LAST_TFILTER_WALL = 0.0
ROI_JFA_LAST_TDECODE_WALL = 0.0
ROI_JFA_LAST_TC2_WALL = 0.0
ROI_JFA_LAST_C2_COUNT = 0
ROI_JFA_LAST_C2_LOS_USED = 0
ROI_JFA_LAST_BUBBLE_ITERS = 0
ROI_JFA_LAST_MAX_RADIUS = 0.0

import numpy as np

import numpy as np
import cupy as cp
import math


# ============================================================
# Candidate ball mask (free-space 26 metric) : block-per-seed
# ============================================================
def build_seed_candidate_ball_mask_block_per_seed_kernel_3d():
    """
    One block per seed: inside the seed's bounding box, mark candidate_mask[idx]=1 where the
    closed-form free-space 26-metric distance satisfies d_free <= r_i.
    candidate_mask is int32 (atomicOr is safe and simple).
    """
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void seed_candidate_ball_mask_block_per_seed_3d(
        const unsigned char* __restrict__ mask,     // [nvox]
        int* __restrict__ cand_mask,                // [nvox] int32 flags

        const int D, const int H, const int W,

        // per-seed bbox
        const int* __restrict__ zmin_arr,
        const int* __restrict__ ymin_arr,
        const int* __restrict__ xmin_arr,
        const int* __restrict__ Yn_arr,
        const int* __restrict__ Xn_arr,
        const int* __restrict__ nbox_arr,

        // per-seed coord + radius
        const int*   __restrict__ seed_z_arr,
        const int*   __restrict__ seed_y_arr,
        const int*   __restrict__ seed_x_arr,
        const float* __restrict__ radius_arr,

        const int n_seeds
    )
    {
        int seed_id = (int)blockIdx.x;
        if (seed_id >= n_seeds) return;

        int nbox = nbox_arr[seed_id];
        if (nbox <= 0) return;

        int zmin = zmin_arr[seed_id];
        int ymin = ymin_arr[seed_id];
        int xmin = xmin_arr[seed_id];
        int Yn   = Yn_arr[seed_id];
        int Xn   = Xn_arr[seed_id];

        int sz0 = seed_z_arr[seed_id];
        int sy0 = seed_y_arr[seed_id];
        int sx0 = seed_x_arr[seed_id];
        float r = radius_arr[seed_id];

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        int HW = H * W;

        for (int p = (int)threadIdx.x; p < nbox; p += (int)blockDim.x) {

            int lz = p / (Yn * Xn);
            int rem1 = p - lz * (Yn * Xn);
            int ly = rem1 / Xn;
            int lx = rem1 - ly * Xn;

            int z = zmin + lz;
            int y = ymin + ly;
            int x = xmin + lx;

            int idx = z * HW + y * W + x;

            if (!mask[idx]) continue;

            // free-space 26 closed-form distance
            int dz = z - sz0; if (dz < 0) dz = -dz;
            int dy = y - sy0; if (dy < 0) dy = -dy;
            int dx = x - sx0; if (dx < 0) dx = -dx;

            int a = dx; if (dy > a) a = dy; if (dz > a) a = dz;
            int c = dx; if (dy < c) c = dy; if (dz < c) c = dz;
            int b = dx + dy + dz - a - c;

            int m1 = a - b;
            int m2 = b - c;
            int m3 = c;

            float d = (float)m1 + SQRT2 * (float)m2 + SQRT3 * (float)m3;

            if (d <= r) {
                atomicOr(&cand_mask[idx], 1);
            }
        }
    }
    '''
    return _device_cached_rawkernel(
        build_seed_candidate_ball_mask_block_per_seed_kernel_3d,
        code,
        "seed_candidate_ball_mask_block_per_seed_3d",
        options=("-std=c++11",),
    )


# ============================================================
# Relax (jump=1) but ONLY on candidate_mask==1, packed-state
# ============================================================
def build_relax1_masked_packed_kernel_3d():
    """
    One 26-neighbour jump=1 relaxation (packed state) on the voxels with candidate_mask==1.
    All other voxels are copied.
    """
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void geodesic_relax1_masked_packed(
        const unsigned char* __restrict__ mask,      // [nvox]
        const int* __restrict__ cand_mask,           // [nvox] int32 flag
        const unsigned long long* __restrict__ state_in,
        unsigned long long* __restrict__ state_out,
        const int D, const int H, const int W,
        const float eps
    )
    {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx] || cand_mask[idx] == 0) {
            state_out[idx] = state_in[idx];
            return;
        }

        unsigned long long cur = state_in[idx];
        int   cur_label = (int)(cur & 0xFFFFFFFFu);
        float cur_dist  = __uint_as_float((unsigned int)(cur >> 32));

        int   best_label = cur_label;
        float best_dist  = cur_dist;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + dz;
                    int jy = y + dy;
                    int jx = x + dx;

                    if ((unsigned)jz >= (unsigned)D ||
                        (unsigned)jy >= (unsigned)H ||
                        (unsigned)jx >= (unsigned)W) continue;

                    int j_idx = idx + dz * HW + dy * W + dx;
                    if (!mask[j_idx]) continue;

                    unsigned long long nb = state_in[j_idx];
                    int nb_label = (int)(nb & 0xFFFFFFFFu);
                    if (nb_label < 0) continue;

                    float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                    int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                    float step = (nnz == 1) ? 1.0f
                               : (nnz == 2) ? SQRT2
                                            : SQRT3;

                    float cand = nb_dist + step;

                    bool improve = (cand + eps < best_dist);
                    bool tie_better_label = (fabsf(cand - best_dist) <= eps) &&
                                            (nb_label >= 0) &&
                                            (best_label < 0 || nb_label < best_label);
                    if (improve || tie_better_label) {
                        best_dist  = cand;
                        best_label = nb_label;
                    }
                }
            }
        }

        unsigned int out_du = __float_as_uint(best_dist);
        unsigned long long out_pack =
            ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

        state_out[idx] = out_pack;
    }
    '''
    return _device_cached_rawkernel(
        build_relax1_masked_packed_kernel_3d,
        code,
        "geodesic_relax1_masked_packed",
        options=("-std=c++11",),
    )


# ============================================================
# Filter certified core from bubble result
# ============================================================
def build_filter_core_from_bubble_kernel_3d():
    """
    Input:  state_in after the bubble relaxation (inside candidate_mask it already holds part of the true d_geo / label)
    Output: state_core (keeps as core only the voxels with dist <= radii[label]; all others INF/-1)
            roi_mask_u8 (1 = ROI, 0 = core/solid)
    """
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void filter_core_from_bubble_packed(
        const unsigned char* __restrict__ mask,      // [nvox]
        const int* __restrict__ cand_mask,           // [nvox]
        const unsigned long long* __restrict__ state_in,
        const float* __restrict__ radii,             // [n_seeds]
        unsigned long long* __restrict__ state_out,
        unsigned char* __restrict__ roi_out,         // [nvox] 0/1
        const unsigned long long pack_inf_neg1,
        const int n_seeds,
        const int D, const int H, const int W,
        const float eps_core
    )
    {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx]) {
            state_out[idx] = pack_inf_neg1;
            roi_out[idx]   = (unsigned char)0;
            return;
        }

        // default: not core -> ROI
        unsigned char roi = (unsigned char)1;
        unsigned long long out = pack_inf_neg1;

        if (cand_mask[idx] != 0) {
            unsigned long long cur = state_in[idx];
            int   lab = (int)(cur & 0xFFFFFFFFu);
            float d   = __uint_as_float((unsigned int)(cur >> 32));

            if ((unsigned)lab < (unsigned)n_seeds) {
                float r = radii[lab];
                if (d <= (r - eps_core)) {
                    out = cur;
                    roi = (unsigned char)0;
                }
            }
        }

        state_out[idx] = out;
        roi_out[idx]   = roi;
    }
    '''
    return _device_cached_rawkernel(
        build_filter_core_from_bubble_kernel_3d,
        code,
        "filter_core_from_bubble_packed",
        options=("-std=c++11",),
    )

def _get_cuda_device_id() -> int:
    import cupy as cp
    return int(cp.cuda.runtime.getDevice())
    
def _memset0_(arr):
    """
    Zero a GPU array with cudaMemsetAsync (closer to the bandwidth limit than arr.fill(0), fewer temporaries).
    arr must be a cupy ndarray.
    """
    import cupy as cp
    stream = cp.cuda.get_current_stream()
    cp.cuda.runtime.memsetAsync(arr.data.ptr, 0, arr.nbytes, stream.ptr)

def _device_cached_rawkernel(build_fn, code: str, kernel_name: str, *, options=(), cache_key=None):
    """
    RawKernel cache per device:
      - one build_fn on one device can cache several kernel_name entries
      - one kernel_name can hold several variants distinguished by cache_key (e.g. mode/use_int_offset)
      - options are part of the key, so different options never reuse the wrong kernel
    """
    import cupy as cp

    dev = _get_cuda_device_id()

    cache = getattr(build_fn, "_cache", None)
    if cache is None:
        cache = {}
        setattr(build_fn, "_cache", cache)

    opt_key = tuple(options) if options is not None else ()
    key = (dev, kernel_name, opt_key, cache_key)

    ker = cache.get(key, None)
    if ker is not None:
        return ker

    ker = cp.RawKernel(code, kernel_name, options=opt_key)
    cache[key] = ker
    return ker


def build_c2_second_competitor_core_kernel_3d():
    """
    Add C2-certified labels to previously unlabeled ROI voxels.
    """
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void c2_second_competitor_core_3d(
        const unsigned char* __restrict__ mask,
        unsigned long long* __restrict__ state,
        const int* __restrict__ seeds,
        int* __restrict__ c2_count,
        const int D, const int H, const int W,
        const int n_seeds,
        const float margin,
        const int only_unlabeled
    )
    {
        const int nvox = D * H * W;
        const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        const bool in_range = (idx < nvox);
        const bool in_mask = in_range && (mask[idx] != 0);

        unsigned long long old_pack = in_range ? state[idx] : 0ull;
        int old_label = (int)(old_pack & 0xFFFFFFFFu);
        const bool active_voxel = in_range && in_mask && !(only_unlabeled && old_label >= 0);

        const int HW = H * W;
        int z = 0;
        int y = 0;
        int x = 0;
        if (in_range) {
            z = idx / HW;
            int rem = idx - z * HW;
            y = rem / W;
            x = rem - y * W;
        }

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;
        const float TIE_EPS = 1.0e-6f;
        __shared__ int s_seeds[3 * 256];

        float best = 1.0e30f;
        float second = 1.0e30f;
        int best_label = -1;

        for (int base = 0; base < n_seeds; base += 256) {
            int tile_n = n_seeds - base;
            if (tile_n > 256) tile_n = 256;

            int lane = (int)threadIdx.x;
            if (lane < tile_n) {
                s_seeds[3*lane + 0] = seeds[3*(base + lane) + 0];
                s_seeds[3*lane + 1] = seeds[3*(base + lane) + 1];
                s_seeds[3*lane + 2] = seeds[3*(base + lane) + 2];
            }
            __syncthreads();

            if (active_voxel) {
                for (int kk = 0; kk < tile_n; ++kk) {
                    int k = base + kk;
                    int sz = s_seeds[3*kk + 0];
                    int sy = s_seeds[3*kk + 1];
                    int sx = s_seeds[3*kk + 2];

                    int dz = z - sz; if (dz < 0) dz = -dz;
                    int dy = y - sy; if (dy < 0) dy = -dy;
                    int dx = x - sx; if (dx < 0) dx = -dx;

                    int a = dx; if (dy > a) a = dy; if (dz > a) a = dz;
                    int c = dx; if (dy < c) c = dy; if (dz < c) c = dz;
                    int b = dx + dy + dz - a - c;
                    float d0 = (float)(a - b) + SQRT2 * (float)(b - c) + SQRT3 * (float)c;

                    bool better = (d0 + TIE_EPS < best) ||
                                  (fabsf(d0 - best) <= TIE_EPS && (best_label < 0 || k < best_label));
                    if (better) {
                        second = best;
                        best = d0;
                        best_label = k;
                    } else if (k != best_label && d0 + TIE_EPS < second) {
                        second = d0;
                    }
                }
            }
            __syncthreads();
        }

        if (!active_voxel) return;
        if (best_label < 0) return;
        if (!(best + margin <= second)) return;

        int sz0 = seeds[3*best_label + 0];
        int sy0 = seeds[3*best_label + 1];
        int sx0 = seeds[3*best_label + 2];

        int dz_i = z - sz0;
        int dy_i = y - sy0;
        int dx_i = x - sx0;

        int az = dz_i; if (az < 0) az = -az;
        int ay = dy_i; if (ay < 0) ay = -ay;
        int ax = dx_i; if (ax < 0) ax = -ax;

        int vA = az, aA = 0;
        int vB = ay, aB = 1;
        int vC = ax, aC = 2;

        if (vA < vB || (vA == vB && aA < aB)) { int tv=vA; vA=vB; vB=tv; int ta=aA; aA=aB; aB=ta; }
        if (vA < vC || (vA == vC && aA < aC)) { int tv=vA; vA=vC; vC=tv; int ta=aA; aA=aC; aC=ta; }
        if (vB < vC || (vB == vC && aB < aC)) { int tv=vB; vB=vC; vC=tv; int ta=aB; aB=aC; aC=ta; }

        int major_axis = aA;
        int mid_axis   = aB;
        int a = vA;
        int b = vB;
        int c = vC;
        int m3 = c;
        int m2 = b - c;
        int m1 = a - b;

        int sgn_z = (dz_i >= 0) ? 1 : -1;
        int sgn_y = (dy_i >= 0) ? 1 : -1;
        int sgn_x = (dx_i >= 0) ? 1 : -1;

        int cz = sz0;
        int cy = sy0;
        int cx = sx0;

        int inc2_z = ((major_axis == 0) || (mid_axis == 0)) ? sgn_z : 0;
        int inc2_y = ((major_axis == 1) || (mid_axis == 1)) ? sgn_y : 0;
        int inc2_x = ((major_axis == 2) || (mid_axis == 2)) ? sgn_x : 0;

        int inc1_z = (major_axis == 0) ? sgn_z : 0;
        int inc1_y = (major_axis == 1) ? sgn_y : 0;
        int inc1_x = (major_axis == 2) ? sgn_x : 0;

        bool clear = true;
        for (int i = 0; i < m3; ++i) {
            cz += sgn_z; cy += sgn_y; cx += sgn_x;
            if ((unsigned)cz >= (unsigned)D ||
                (unsigned)cy >= (unsigned)H ||
                (unsigned)cx >= (unsigned)W) { clear = false; break; }
            int j = cz * HW + cy * W + cx;
            if (!mask[j]) { clear = false; break; }
        }

        if (clear) {
            for (int i = 0; i < m2; ++i) {
                cz += inc2_z; cy += inc2_y; cx += inc2_x;
                if ((unsigned)cz >= (unsigned)D ||
                    (unsigned)cy >= (unsigned)H ||
                    (unsigned)cx >= (unsigned)W) { clear = false; break; }
                int j = cz * HW + cy * W + cx;
                if (!mask[j]) { clear = false; break; }
            }
        }

        if (clear) {
            for (int i = 0; i < m1; ++i) {
                cz += inc1_z; cy += inc1_y; cx += inc1_x;
                if ((unsigned)cz >= (unsigned)D ||
                    (unsigned)cy >= (unsigned)H ||
                    (unsigned)cx >= (unsigned)W) { clear = false; break; }
                int j = cz * HW + cy * W + cx;
                if (!mask[j]) { clear = false; break; }
            }
        }

        if (!clear) return;

        unsigned int dist_u = __float_as_uint(best);
        unsigned long long new_pack =
            ((unsigned long long)dist_u << 32) |
            (unsigned long long)(unsigned int)best_label;

        if (only_unlabeled || new_pack < old_pack) {
            state[idx] = new_pack;
            atomicAdd(c2_count, 1);
        }
    }
    '''
    return _device_cached_rawkernel(
        build_c2_second_competitor_core_kernel_3d,
        code,
        "c2_second_competitor_core_3d",
        options=("-std=c++11",),
    )


def build_c2_second_competitor_core_los_kernel_3d():
    """
    C2-certified core with canonical-path clearance checked by the precomputed
    27-direction LOS-kmax table.  The certificate is identical to the stepwise
    C2 kernel; only the path-clearance test is compressed into power-of-two
    segment queries.
    """
    code = r'''
    #include <cuda_runtime.h>

    __device__ __forceinline__
    bool c2_los_segment_ok(
        const signed char* __restrict__ los_kmax27,
        const int nvox,
        const int D, const int H, const int W, const int HW,
        int* cz, int* cy, int* cx,
        const int inc_z, const int inc_y, const int inc_x,
        int len
    ){
        if (len <= 0) return true;
        if (inc_z == 0 && inc_y == 0 && inc_x == 0) return true;

        const int dir_id = (inc_z + 1) * 9 + (inc_y + 1) * 3 + (inc_x + 1);
        while (len > 0) {
            int q = 0;
            int step = 1;
            while ((step << 1) <= len) {
                step <<= 1;
                ++q;
            }

            const int idx = (*cz) * HW + (*cy) * W + (*cx);
            const signed char km = los_kmax27[(long long)dir_id * (long long)nvox + (long long)idx];
            if ((int)km < q) return false;

            *cz += inc_z * step;
            *cy += inc_y * step;
            *cx += inc_x * step;
            if ((unsigned)(*cz) >= (unsigned)D ||
                (unsigned)(*cy) >= (unsigned)H ||
                (unsigned)(*cx) >= (unsigned)W) {
                return false;
            }
            len -= step;
        }
        return true;
    }

    extern "C" __global__
    void c2_second_competitor_core_los_3d(
        const unsigned char* __restrict__ mask,
        unsigned long long* __restrict__ state,
        const int* __restrict__ seeds,
        const signed char* __restrict__ los_kmax27,
        int* __restrict__ c2_count,
        const int D, const int H, const int W,
        const int n_seeds,
        const float margin,
        const int only_unlabeled
    )
    {
        const int nvox = D * H * W;
        const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        const bool in_range = (idx < nvox);
        const bool in_mask = in_range && (mask[idx] != 0);

        unsigned long long old_pack = in_range ? state[idx] : 0ull;
        int old_label = (int)(old_pack & 0xFFFFFFFFu);
        const bool active_voxel = in_range && in_mask && !(only_unlabeled && old_label >= 0);

        const int HW = H * W;
        int z = 0;
        int y = 0;
        int x = 0;
        if (in_range) {
            z = idx / HW;
            int rem = idx - z * HW;
            y = rem / W;
            x = rem - y * W;
        }

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;
        const float TIE_EPS = 1.0e-6f;
        __shared__ int s_seeds[3 * 256];

        float best = 1.0e30f;
        float second = 1.0e30f;
        int best_label = -1;

        for (int base = 0; base < n_seeds; base += 256) {
            int tile_n = n_seeds - base;
            if (tile_n > 256) tile_n = 256;

            int lane = (int)threadIdx.x;
            if (lane < tile_n) {
                s_seeds[3*lane + 0] = seeds[3*(base + lane) + 0];
                s_seeds[3*lane + 1] = seeds[3*(base + lane) + 1];
                s_seeds[3*lane + 2] = seeds[3*(base + lane) + 2];
            }
            __syncthreads();

            if (active_voxel) {
                for (int kk = 0; kk < tile_n; ++kk) {
                    int k = base + kk;
                    int sz = s_seeds[3*kk + 0];
                    int sy = s_seeds[3*kk + 1];
                    int sx = s_seeds[3*kk + 2];

                    int dz = z - sz; if (dz < 0) dz = -dz;
                    int dy = y - sy; if (dy < 0) dy = -dy;
                    int dx = x - sx; if (dx < 0) dx = -dx;

                    int a = dx; if (dy > a) a = dy; if (dz > a) a = dz;
                    int c = dx; if (dy < c) c = dy; if (dz < c) c = dz;
                    int b = dx + dy + dz - a - c;
                    float d0 = (float)(a - b) + SQRT2 * (float)(b - c) + SQRT3 * (float)c;

                    bool better = (d0 + TIE_EPS < best) ||
                                  (fabsf(d0 - best) <= TIE_EPS && (best_label < 0 || k < best_label));
                    if (better) {
                        second = best;
                        best = d0;
                        best_label = k;
                    } else if (k != best_label && d0 + TIE_EPS < second) {
                        second = d0;
                    }
                }
            }
            __syncthreads();
        }

        if (!active_voxel) return;
        if (best_label < 0) return;
        if (!(best + margin <= second)) return;

        int sz0 = seeds[3*best_label + 0];
        int sy0 = seeds[3*best_label + 1];
        int sx0 = seeds[3*best_label + 2];

        int dz_i = z - sz0;
        int dy_i = y - sy0;
        int dx_i = x - sx0;

        int az = dz_i; if (az < 0) az = -az;
        int ay = dy_i; if (ay < 0) ay = -ay;
        int ax = dx_i; if (ax < 0) ax = -ax;

        int vA = az, aA = 0;
        int vB = ay, aB = 1;
        int vC = ax, aC = 2;

        if (vA < vB || (vA == vB && aA < aB)) { int tv=vA; vA=vB; vB=tv; int ta=aA; aA=aB; aB=ta; }
        if (vA < vC || (vA == vC && aA < aC)) { int tv=vA; vA=vC; vC=tv; int ta=aA; aA=aC; aC=ta; }
        if (vB < vC || (vB == vC && aB < aC)) { int tv=vB; vB=vC; vC=tv; int ta=aB; aB=aC; aC=ta; }

        int major_axis = aA;
        int mid_axis   = aB;
        int a = vA;
        int b = vB;
        int c = vC;
        int m3 = c;
        int m2 = b - c;
        int m1 = a - b;

        int sgn_z = (dz_i >= 0) ? 1 : -1;
        int sgn_y = (dy_i >= 0) ? 1 : -1;
        int sgn_x = (dx_i >= 0) ? 1 : -1;

        int cz = sz0;
        int cy = sy0;
        int cx = sx0;

        int inc2_z = ((major_axis == 0) || (mid_axis == 0)) ? sgn_z : 0;
        int inc2_y = ((major_axis == 1) || (mid_axis == 1)) ? sgn_y : 0;
        int inc2_x = ((major_axis == 2) || (mid_axis == 2)) ? sgn_x : 0;

        int inc1_z = (major_axis == 0) ? sgn_z : 0;
        int inc1_y = (major_axis == 1) ? sgn_y : 0;
        int inc1_x = (major_axis == 2) ? sgn_x : 0;

        bool clear = true;
        clear = clear && c2_los_segment_ok(los_kmax27, nvox, D, H, W, HW, &cz, &cy, &cx, sgn_z, sgn_y, sgn_x, m3);
        clear = clear && c2_los_segment_ok(los_kmax27, nvox, D, H, W, HW, &cz, &cy, &cx, inc2_z, inc2_y, inc2_x, m2);
        clear = clear && c2_los_segment_ok(los_kmax27, nvox, D, H, W, HW, &cz, &cy, &cx, inc1_z, inc1_y, inc1_x, m1);
        if (!clear) return;

        unsigned int dist_u = __float_as_uint(best);
        unsigned long long new_pack =
            ((unsigned long long)dist_u << 32) |
            (unsigned long long)(unsigned int)best_label;

        if (only_unlabeled || new_pack < old_pack) {
            state[idx] = new_pack;
            atomicAdd(c2_count, 1);
        }
    }
    '''
    return _device_cached_rawkernel(
        build_c2_second_competitor_core_los_kernel_3d,
        code,
        "c2_second_competitor_core_los_3d",
        options=("-std=c++11",),
    )


def build_sparse_roi_jfa_step_kernel_3d():
    """
    Sparse voxel-list ROI-JFA step.

    This kernel is intended for certified ROI masks whose residual set is much
    smaller than the tile lattice.  It processes exactly roi_ids[p] voxels and
    leaves all other state entries untouched in a pre-copied output buffer.
    """
    code = r'''
    extern "C" __global__
    void sparse_roi_jfa_step_packed_3d(
        const unsigned char* __restrict__ mask,
        const signed char* __restrict__ los_kmax27,
        const int* __restrict__ roi_ids,
        const int n_roi,
        const int nvox,
        const unsigned long long* __restrict__ state_in,
        unsigned long long* __restrict__ state_out,
        const int D, const int H, const int W,
        const int jump,
        const int jump_k,
        const float eps
    )
    {
        int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (p >= n_roi) return;

        int idx = roi_ids[p];
        if ((unsigned)idx >= (unsigned)nvox) return;
        if (!mask[idx]) return;

        const int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        unsigned long long cur = state_in[idx];
        int best_label = (int)(cur & 0xFFFFFFFFu);
        float best_dist = __uint_as_float((unsigned int)(cur >> 32));

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int zz = z + jump * dz;
                    int yy = y + jump * dy;
                    int xx = x + jump * dx;
                    if ((unsigned)zz >= (unsigned)D ||
                        (unsigned)yy >= (unsigned)H ||
                        (unsigned)xx >= (unsigned)W) continue;

                    int j = zz * HW + yy * W + xx;
                    if (!mask[j]) continue;

                    int dir_id = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                    signed char km = los_kmax27[dir_id * nvox + idx];
                    if ((int)km < jump_k) continue;

                    unsigned long long nb = state_in[j];
                    int nb_label = (int)(nb & 0xFFFFFFFFu);
                    if (nb_label < 0) continue;
                    float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                    int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                    float step = (nnz == 1) ? (float)jump :
                                 (nnz == 2) ? (SQRT2 * (float)jump) :
                                              (SQRT3 * (float)jump);
                    float cand = nb_dist + step;
                    bool improve = (cand + eps < best_dist);
                    bool tie = (fabsf(cand - best_dist) <= eps) &&
                               (nb_label >= 0) &&
                               (best_label < 0 || nb_label < best_label);
                    if (improve || tie) {
                        best_dist = cand;
                        best_label = nb_label;
                    }
                }
            }
        }

        unsigned int dist_u = __float_as_uint(best_dist);
        state_out[idx] =
            ((unsigned long long)dist_u << 32) |
            (unsigned long long)(unsigned int)best_label;
    }
    '''
    return _device_cached_rawkernel(
        build_sparse_roi_jfa_step_kernel_3d,
        code,
        "sparse_roi_jfa_step_packed_3d",
        options=("-std=c++11",),
    )


def build_los_kmax27_init_all_dirs_3d():
    code = r'''
    extern "C" __global__
    void los_kmax27_init_all_dirs_3d(
        const unsigned char* __restrict__ mask,
        signed char* __restrict__ kmax27,
        const int D, const int H, const int W,
        const int nvox,
        const int max_k
    ){
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;

        int dir_id = (int)blockIdx.y;  // 0..26
        int out_off = dir_id * nvox + idx;

        if (!mask[idx]) {
            kmax27[out_off] = (signed char)-1;
            return;
        }

        // base-3 decode: 0..26 -> (-1,0,1)^3
        int dx = (dir_id % 3) - 1;
        int dy = ((dir_id / 3) % 3) - 1;
        int dz = (dir_id / 9) - 1;

        // center direction: always valid up to max_k (not used by JFA, but keep consistent)
        if (dx == 0 && dy == 0 && dz == 0) {
            kmax27[out_off] = (signed char)max_k;
            return;
        }

        int HW = H * W;
        int z  = idx / HW;
        int rem = idx - z * HW;
        int y  = rem / W;
        int x  = rem - y * W;

        int xn = x + dx;
        int yn = y + dy;
        int zn = z + dz;

        if ((unsigned)xn >= (unsigned)W ||
            (unsigned)yn >= (unsigned)H ||
            (unsigned)zn >= (unsigned)D) {
            kmax27[out_off] = (signed char)-1;
            return;
        }

        int nidx = zn * HW + yn * W + xn;
        kmax27[out_off] = mask[nidx] ? (signed char)0 : (signed char)-1;
    }
    '''
    return _device_cached_rawkernel(
        build_los_kmax27_init_all_dirs_3d,
        code,
        "los_kmax27_init_all_dirs_3d",
        options=("-std=c++11",),
    )


def build_los_kmax27_update_all_dirs_3d():
    code = r'''
    extern "C" __global__
    void los_kmax27_update_all_dirs_3d(
        const unsigned char* __restrict__ mask,
        signed char* __restrict__ kmax27,
        const int D, const int H, const int W,
        const int nvox,
        const int k
    ){
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;

        int dir_id = (int)blockIdx.y;
        int off = dir_id * nvox + idx;

        if (!mask[idx]) {
            kmax27[off] = (signed char)-1;
            return;
        }

        int dx = (dir_id % 3) - 1;
        int dy = ((dir_id / 3) % 3) - 1;
        int dz = (dir_id / 9) - 1;

        // center direction: keep whatever init wrote (max_k)
        if (dx == 0 && dy == 0 && dz == 0) return;

        signed char cur = kmax27[off];
        // need both halves length 2^(k-1) to be valid
        if (cur < (signed char)(k - 1)) return;

        int step = 1 << (k - 1);

        int HW = H * W;
        int z  = idx / HW;
        int rem = idx - z * HW;
        int y  = rem / W;
        int x  = rem - y * W;

        int xm = x + dx * step;
        int ym = y + dy * step;
        int zm = z + dz * step;

        if ((unsigned)xm >= (unsigned)W ||
            (unsigned)ym >= (unsigned)H ||
            (unsigned)zm >= (unsigned)D) {
            return;
        }

        int mid = zm * HW + ym * W + xm;
        if (!mask[mid]) return;

        int mid_off = dir_id * nvox + mid;
        if (kmax27[mid_off] >= (signed char)(k - 1)) {
            kmax27[off] = (signed char)k;
        }
    }
    '''
    return _device_cached_rawkernel(
        build_los_kmax27_update_all_dirs_3d,
        code,
        "los_kmax27_update_all_dirs_3d",
        options=("-std=c++11",),
    )


def _as_cupy_mask_u8(mask):
    """
    mask: np.ndarray or cp.ndarray of shape (D,H,W)
    returns: mask_cp (uint8), (D,H,W), mask_flat
    """
    import numpy as np
    import cupy as cp

    if isinstance(mask, cp.ndarray):
        mask_cp = mask
        if mask_cp.dtype != cp.uint8:
            mask_cp = mask_cp.astype(cp.uint8, copy=False)
        D, H, W = map(int, mask_cp.shape)
    else:
        mask_np = np.asarray(mask, dtype=np.bool_)
        D, H, W = map(int, mask_np.shape)
        mask_cp = cp.asarray(mask_np, dtype=cp.uint8)

    return mask_cp, (D, H, W), mask_cp.ravel()


def _as_cupy_int32(a):
    """
    Generic conversion of seeds or other index arrays: stays on the GPU, no blind np.asarray.
    """
    import numpy as np
    import cupy as cp

    if isinstance(a, cp.ndarray):
        return a.astype(cp.int32, copy=False)
    return cp.asarray(np.asarray(a, dtype=np.int32), dtype=cp.int32)


def _to_numpy(a):
    """
    Accept numpy array / cupy array / array-like, return numpy.ndarray (no copy if already numpy).
    """
    try:
        import cupy as cp  # optional
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def compute_voronoi_boundaries_2d(labels2d: np.ndarray,
                                 mask2d: np.ndarray,
                                 *,
                                 connectivity: int = 4,
                                 include_solid_boundaries: bool = True) -> np.ndarray:
    """
    Compute boundary pixels for a Voronoi label map on a 2D slice.

    include_solid_boundaries:
      - True : boundary includes fluid-solid adjacency (i.e., "sealed" along obstacles).
      - False: boundary ONLY between two FLUID pixels with valid labels (>=0).

    Return:
      - boundary pixels on the FLUID side only (mask2d==True).
    """
    if labels2d.shape != mask2d.shape:
        raise ValueError(f"labels2d.shape {labels2d.shape} != mask2d.shape {mask2d.shape}")

    lab = labels2d.astype(np.int32, copy=False)
    m = mask2d.astype(bool, copy=False)

    H, W = lab.shape
    b = np.zeros((H, W), dtype=bool)

    # helper: build a "valid pair" mask for neighbor comparisons
    # - include_solid_boundaries=True  -> at least one side fluid
    # - include_solid_boundaries=False -> both sides fluid AND both labels>=0
    def _valid_pair(mA, mB, lA, lB):
        if include_solid_boundaries:
            return (mA | mB)
        else:
            return (mA & mB & (lA >= 0) & (lB >= 0))

    # vertical neighbors
    dv = (lab[1:, :] != lab[:-1, :])
    dv &= _valid_pair(m[1:, :], m[:-1, :], lab[1:, :], lab[:-1, :])
    b[1:, :] |= dv
    b[:-1, :] |= dv

    # horizontal neighbors
    dh = (lab[:, 1:] != lab[:, :-1])
    dh &= _valid_pair(m[:, 1:], m[:, :-1], lab[:, 1:], lab[:, :-1])
    b[:, 1:] |= dh
    b[:, :-1] |= dh

    if connectivity == 8:
        d1 = (lab[1:, 1:] != lab[:-1, :-1])
        d1 &= _valid_pair(m[1:, 1:], m[:-1, :-1], lab[1:, 1:], lab[:-1, :-1])
        b[1:, 1:] |= d1
        b[:-1, :-1] |= d1

        d2 = (lab[1:, :-1] != lab[:-1, 1:])
        d2 &= _valid_pair(m[1:, :-1], m[:-1, 1:], lab[1:, :-1], lab[:-1, 1:])
        b[1:, :-1] |= d2
        b[:-1, 1:] |= d2
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8")

    # only draw on fluid side
    b &= m
    return b




def assert_seeds_valid_and_in_pore(mask, seeds, backend="cupy"):
    """
    Common seed entry point: seeds are not modified, only checked, so that M1/M2/M3/M4 get identical input.

    - seeds must lie inside the domain
    - seeds must lie in the pore space (mask==1)
      (otherwise the geodesic methods drop the seed while the Euclidean method still uses it, and they differ)
    """
    import numpy as np
    import cupy as cp

    if backend == "cupy":
        mask_cp = mask if isinstance(mask, cp.ndarray) else cp.asarray(mask)
        mask_cp = (mask_cp != 0)
        D, H, W = map(int, mask_cp.shape)

        seeds_cp = seeds if isinstance(seeds, cp.ndarray) else cp.asarray(seeds)
        seeds_cp = seeds_cp.astype(cp.int64)
        if seeds_cp.ndim != 2 or seeds_cp.shape[1] != 3:
            raise ValueError("seeds must be (N,3) with (z,y,x)")

        z = seeds_cp[:, 0]; y = seeds_cp[:, 1]; x = seeds_cp[:, 2]
        inb = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        if not bool(inb.all().item()):
            k = int(cp.where(~inb)[0][0].item())
            coord = tuple(int(v.item()) for v in seeds_cp[k])
            raise ValueError(f"Seed {k} out of bounds: {coord}")

        inpore = mask_cp[z, y, x]
        if not bool(inpore.all().item()):
            k = int(cp.where(~inpore)[0][0].item())
            coord = tuple(int(v.item()) for v in seeds_cp[k])
            raise ValueError(
                f"Seed {k} is NOT in pore(mask==1): {coord}. "
                f"M1/M2/M3/M4 agree only if every seed lies in the pore."
            )
        return  # ok

    else:
        mask_np = np.asarray(mask).astype(bool)
        D, H, W = mask_np.shape
        seeds_np = np.asarray(seeds, dtype=np.int64)
        if seeds_np.ndim != 2 or seeds_np.shape[1] != 3:
            raise ValueError("seeds must be (N,3) with (z,y,x)")
        z, y, x = seeds_np[:,0], seeds_np[:,1], seeds_np[:,2]
        inb = (z>=0)&(z<D)&(y>=0)&(y<H)&(x>=0)&(x<W)
        if not inb.all():
            k = int(np.where(~inb)[0][0])
            raise ValueError(f"Seed {k} out of bounds: {tuple(seeds_np[k])}")
        inpore = mask_np[z,y,x]
        if not inpore.all():
            k = int(np.where(~inpore)[0][0])
            raise ValueError(f"Seed {k} is NOT in pore(mask==1): {tuple(seeds_np[k])}")
        return  # ok

def m1_euclidean_voronoi_clipping_gpu(mask, seeds, kernel=None, profile_gpu=False):
    """
    M1: Euclidean Voronoi + clipping baseline (GPU).

    Notes:
    - same algorithm and behaviour as before: calls euclidean_voronoi_clipping_gpu
    - seed check as for the other methods: assert_seeds_valid_and_in_pore
    """
    import cupy as cp

    # for numpy input, check with the numpy backend to avoid a needless cp.asarray
    backend = "cupy" if isinstance(mask, cp.ndarray) or isinstance(seeds, cp.ndarray) else "numpy"
    assert_seeds_valid_and_in_pore(mask, seeds, backend=backend)

    # seeds were already checked by the caller, so the internal seed check is off (no repeated cost)
    label_cp, dist_cp, t_evt = euclidean_voronoi_clipping_gpu(
        mask,
        seeds,
        kernel=kernel,
        profile_gpu=profile_gpu,
        eps_seed_check=False,
    )
    return label_cp, dist_cp, t_evt




import time
import math
import numpy as np
import cupy as cp

# ---- add new globals (optional but recommended) ----
EXACT_LAST_WALL_TIME      = 0.0
EXACT_LAST_TINIT_WALL     = 0.0   # mask+alloc+seed init
EXACT_LAST_TITER_WALL     = 0.0   # main relaxation wall
EXACT_LAST_TOPT_WALL      = 0.0   # optional optimality check wall
EXACT_LAST_GPU_TIME       = 0.0   # you already have this global

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
    """Build the seed-stamping kernel for exact frontier BFS."""
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
    handling. It is intended as a fair exact GPU baseline and as a fast exact
    label backend for retained 6-neighbour production rows.
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


def build_regular_stride_l1_label_kernel_3d():
    """Direct 6-neighbour L1 ownership for regular prescribed-site grids."""
    code = r'''
    extern "C" __global__
    void regular_stride_l1_label_3d(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ seed_lut,
        int* __restrict__ label_out,
        float* __restrict__ dist_out,
        const int nvox,
        const int D,
        const int H,
        const int W,
        const int sz,
        const int sy,
        const int sx,
        const int oz,
        const int oy,
        const int ox
    ) {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= nvox) return;

        if (!mask[idx]) {
            label_out[idx] = -1;
            dist_out[idx] = 1.0e20f;
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int bz = (z - oz) / sz;
        int by = (y - oy) / sy;
        int bx = (x - ox) / sx;
        if (z < oz && ((z - oz) % sz) != 0) --bz;
        if (y < oy && ((y - oy) % sy) != 0) --by;
        if (x < ox && ((x - ox) % sx) != 0) --bx;

        int best_label = -1;
        int best_dist = 2147483647;

        for (int iz = bz - 1; iz <= bz + 2; ++iz) {
            int zz = oz + iz * sz;
            if ((unsigned)zz >= (unsigned)D) continue;
            for (int iy = by - 1; iy <= by + 2; ++iy) {
                int yy = oy + iy * sy;
                if ((unsigned)yy >= (unsigned)H) continue;
                int base = zz * HW + yy * W;
                for (int ix = bx - 1; ix <= bx + 2; ++ix) {
                    int xx = ox + ix * sx;
                    if ((unsigned)xx >= (unsigned)W) continue;
                    int sidx = base + xx;
                    int lab = seed_lut[sidx];
                    if (lab < 0) continue;
                    int d = abs(z - zz) + abs(y - yy) + abs(x - xx);
                    if (d < best_dist || (d == best_dist && (best_label < 0 || lab < best_label))) {
                        best_dist = d;
                        best_label = lab;
                    }
                }
            }
        }

        label_out[idx] = best_label;
        dist_out[idx] = (best_label >= 0) ? (float)best_dist : 1.0e20f;
    }
    '''
    return _device_cached_rawkernel(
        build_regular_stride_l1_label_kernel_3d,
        code,
        "regular_stride_l1_label_3d",
        options=("-std=c++11",),
    )


def build_regular_stride_l1_lattice_label_kernel_3d():
    """Direct 6-neighbour L1 ownership using a compact stride-lattice LUT."""
    code = r'''
    extern "C" __global__
    void regular_stride_l1_lattice_label_3d(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ lattice_lut,
        int* __restrict__ label_out,
        float* __restrict__ dist_out,
        const int nvox,
        const int D,
        const int H,
        const int W,
        const int Lz,
        const int Ly,
        const int Lx,
        const int sz,
        const int sy,
        const int sx,
        const int oz,
        const int oy,
        const int ox
    ) {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= nvox) return;

        if (!mask[idx]) {
            label_out[idx] = -1;
            dist_out[idx] = 1.0e20f;
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int bz = (z - oz) / sz;
        int by = (y - oy) / sy;
        int bx = (x - ox) / sx;
        if (z < oz && ((z - oz) % sz) != 0) --bz;
        if (y < oy && ((y - oy) % sy) != 0) --by;
        if (x < ox && ((x - ox) % sx) != 0) --bx;

        int best_label = -1;
        int best_dist = 2147483647;

        for (int iz = bz - 1; iz <= bz + 2; ++iz) {
            if ((unsigned)iz >= (unsigned)Lz) continue;
            int zz = oz + iz * sz;
            for (int iy = by - 1; iy <= by + 2; ++iy) {
                if ((unsigned)iy >= (unsigned)Ly) continue;
                int yy = oy + iy * sy;
                int lbase = (iz * Ly + iy) * Lx;
                for (int ix = bx - 1; ix <= bx + 2; ++ix) {
                    if ((unsigned)ix >= (unsigned)Lx) continue;
                    int lab = lattice_lut[lbase + ix];
                    if (lab < 0) continue;
                    int xx = ox + ix * sx;
                    int d = abs(z - zz) + abs(y - yy) + abs(x - xx);
                    if (d < best_dist || (d == best_dist && (best_label < 0 || lab < best_label))) {
                        best_dist = d;
                        best_label = lab;
                    }
                }
            }
        }

        label_out[idx] = best_label;
        dist_out[idx] = (best_label >= 0) ? (float)best_dist : 1.0e20f;
    }
    '''
    return _device_cached_rawkernel(
        build_regular_stride_l1_lattice_label_kernel_3d,
        code,
        "regular_stride_l1_lattice_label_3d",
        options=("-std=c++11",),
    )


def build_regular_stride_lattice_lut_fill_kernel():
    """Fill a compact regular-stride lattice lookup table from seed coordinates."""
    code = r'''
    extern "C" __global__
    void regular_stride_lattice_lut_fill(
        const int* __restrict__ seeds,
        int* __restrict__ lattice_lut,
        int* __restrict__ bad,
        const int n_seed,
        const int Lz,
        const int Ly,
        const int Lx,
        const int sz,
        const int sy,
        const int sx,
        const int oz,
        const int oy,
        const int ox
    ) {
        int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (tid >= n_seed) return;
        int z = seeds[3 * tid + 0];
        int y = seeds[3 * tid + 1];
        int x = seeds[3 * tid + 2];
        int dz = z - oz;
        int dy = y - oy;
        int dx = x - ox;
        if (dz < 0 || dy < 0 || dx < 0 || (dz % sz) != 0 || (dy % sy) != 0 || (dx % sx) != 0) {
            atomicExch(bad, 1);
            return;
        }
        int iz = dz / sz;
        int iy = dy / sy;
        int ix = dx / sx;
        if ((unsigned)iz >= (unsigned)Lz || (unsigned)iy >= (unsigned)Ly || (unsigned)ix >= (unsigned)Lx) {
            atomicExch(bad, 1);
            return;
        }
        lattice_lut[(iz * Ly + iy) * Lx + ix] = tid;
    }
    '''
    return _device_cached_rawkernel(
        build_regular_stride_lattice_lut_fill_kernel,
        code,
        "regular_stride_lattice_lut_fill",
        options=("-std=c++11",),
    )


def build_regular_stride_l1_lattice_clearance_label_kernel_3d():
    """Direct regular-stride ownership with small monotone-path clearance checks."""
    code = r'''
    __device__ __forceinline__
    bool step_axis_clear(
        const unsigned char* __restrict__ mask,
        int& z, int& y, int& x,
        const int tz, const int ty, const int tx,
        const int axis,
        const int H,
        const int W
    ) {
        int target = (axis == 0) ? tz : (axis == 1) ? ty : tx;
        int* curp = (axis == 0) ? &z : (axis == 1) ? &y : &x;
        int dir = (target > *curp) ? 1 : -1;
        while (*curp != target) {
            *curp += dir;
            int idx = z * H * W + y * W + x;
            if (!mask[idx]) return false;
        }
        return true;
    }

    __device__ __forceinline__
    bool monotone_path_clear6(
        const unsigned char* __restrict__ mask,
        const int z0, const int y0, const int x0,
        const int z1, const int y1, const int x1,
        const int H,
        const int W,
        const int order_id
    ) {
        int z = z0;
        int y = y0;
        int x = x0;
        int a0, a1, a2;
        if (order_id == 0) { a0 = 0; a1 = 1; a2 = 2; }
        else if (order_id == 1) { a0 = 0; a1 = 2; a2 = 1; }
        else if (order_id == 2) { a0 = 1; a1 = 0; a2 = 2; }
        else if (order_id == 3) { a0 = 1; a1 = 2; a2 = 0; }
        else if (order_id == 4) { a0 = 2; a1 = 0; a2 = 1; }
        else { a0 = 2; a1 = 1; a2 = 0; }
        if (!step_axis_clear(mask, z, y, x, z1, y1, x1, a0, H, W)) return false;
        if (!step_axis_clear(mask, z, y, x, z1, y1, x1, a1, H, W)) return false;
        if (!step_axis_clear(mask, z, y, x, z1, y1, x1, a2, H, W)) return false;
        return true;
    }

    extern "C" __global__
    void regular_stride_l1_lattice_clearance_label_3d(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ lattice_lut,
        int* __restrict__ label_out,
        float* __restrict__ dist_out,
        unsigned char* __restrict__ certified_out,
        int* __restrict__ uncertified_count,
        const int nvox,
        const int D,
        const int H,
        const int W,
        const int Lz,
        const int Ly,
        const int Lx,
        const int sz,
        const int sy,
        const int sx,
        const int oz,
        const int oy,
        const int ox,
        const int write_certified
    ) {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= nvox) return;

        if (!mask[idx]) {
            label_out[idx] = -1;
            dist_out[idx] = 1.0e20f;
            if (write_certified && certified_out) certified_out[idx] = 1;
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int bz = (z - oz) / sz;
        int by = (y - oy) / sy;
        int bx = (x - ox) / sx;
        if (z < oz && ((z - oz) % sz) != 0) --bz;
        if (y < oy && ((y - oy) % sy) != 0) --by;
        if (x < ox && ((x - ox) % sx) != 0) --bx;

        int best_label = -1;
        int best_dist = 2147483647;
        int nearest_label = -1;
        int nearest_dist = 2147483647;
        int nearest_z = 0;
        int nearest_y = 0;
        int nearest_x = 0;

        for (int iz = bz - 1; iz <= bz + 2; ++iz) {
            if ((unsigned)iz >= (unsigned)Lz) continue;
            int zz = oz + iz * sz;
            for (int iy = by - 1; iy <= by + 2; ++iy) {
                if ((unsigned)iy >= (unsigned)Ly) continue;
                int yy = oy + iy * sy;
                int lbase = (iz * Ly + iy) * Lx;
                for (int ix = bx - 1; ix <= bx + 2; ++ix) {
                    if ((unsigned)ix >= (unsigned)Lx) continue;
                    int lab = lattice_lut[lbase + ix];
                    if (lab < 0) continue;
                    int xx = ox + ix * sx;
                    int d = abs(z - zz) + abs(y - yy) + abs(x - xx);
                    if (d < nearest_dist || (d == nearest_dist && (nearest_label < 0 || lab < nearest_label))) {
                        nearest_dist = d;
                        nearest_label = lab;
                        nearest_z = zz;
                        nearest_y = yy;
                        nearest_x = xx;
                    }
                }
            }
        }

        if (nearest_label >= 0) {
            bool nearest_clear = false;
            #pragma unroll
            for (int ord = 0; ord < 6; ++ord) {
                if (monotone_path_clear6(mask, nearest_z, nearest_y, nearest_x, z, y, x, H, W, ord)) {
                    nearest_clear = true;
                    break;
                }
            }
            if (nearest_clear) {
                label_out[idx] = nearest_label;
                dist_out[idx] = (float)nearest_dist;
                if (write_certified && certified_out) certified_out[idx] = 1;
                return;
            }
        }

        for (int iz = bz - 1; iz <= bz + 2; ++iz) {
            if ((unsigned)iz >= (unsigned)Lz) continue;
            int zz = oz + iz * sz;
            for (int iy = by - 1; iy <= by + 2; ++iy) {
                if ((unsigned)iy >= (unsigned)Ly) continue;
                int yy = oy + iy * sy;
                int lbase = (iz * Ly + iy) * Lx;
                for (int ix = bx - 1; ix <= bx + 2; ++ix) {
                    if ((unsigned)ix >= (unsigned)Lx) continue;
                    int lab = lattice_lut[lbase + ix];
                    if (lab < 0) continue;
                    int xx = ox + ix * sx;
                    int d = abs(z - zz) + abs(y - yy) + abs(x - xx);
                    if (d > best_dist || (d == best_dist && best_label >= 0 && lab >= best_label)) continue;

                    bool clear = false;
                    #pragma unroll
                    for (int ord = 0; ord < 6; ++ord) {
                        if (monotone_path_clear6(mask, zz, yy, xx, z, y, x, H, W, ord)) {
                            clear = true;
                            break;
                        }
                    }
                    if (!clear) continue;
                    best_dist = d;
                    best_label = lab;
                }
            }
        }

        label_out[idx] = best_label;
        dist_out[idx] = (best_label >= 0) ? (float)best_dist : 1.0e20f;
        unsigned char ok = (best_label >= 0) ? 1 : 0;
        if (write_certified && certified_out) certified_out[idx] = ok;
        if (!ok && uncertified_count) atomicAdd(uncertified_count, 1);
    }
    '''
    return _device_cached_rawkernel(
        build_regular_stride_l1_lattice_clearance_label_kernel_3d,
        code,
        "regular_stride_l1_lattice_clearance_label_3d",
        options=("-std=c++11",),
    )


def build_regular_stride_l1_lattice_nearest_clearance_check_kernel_3d():
    """Check whether the nearest regular-lattice owner has a monotone 6-path."""
    code = r'''
    __device__ __forceinline__
    bool nearest_step_axis_clear(
        const unsigned char* __restrict__ mask,
        int& z, int& y, int& x,
        const int tz, const int ty, const int tx,
        const int axis,
        const int H,
        const int W
    ) {
        int target = (axis == 0) ? tz : (axis == 1) ? ty : tx;
        int* curp = (axis == 0) ? &z : (axis == 1) ? &y : &x;
        int dir = (target > *curp) ? 1 : -1;
        while (*curp != target) {
            *curp += dir;
            int idx = z * H * W + y * W + x;
            if (!mask[idx]) return false;
        }
        return true;
    }

    __device__ __forceinline__
    bool nearest_monotone_path_clear6(
        const unsigned char* __restrict__ mask,
        const int z0, const int y0, const int x0,
        const int z1, const int y1, const int x1,
        const int H,
        const int W,
        const int order_id
    ) {
        int z = z0;
        int y = y0;
        int x = x0;
        int a0, a1, a2;
        if (order_id == 0) { a0 = 0; a1 = 1; a2 = 2; }
        else if (order_id == 1) { a0 = 0; a1 = 2; a2 = 1; }
        else if (order_id == 2) { a0 = 1; a1 = 0; a2 = 2; }
        else if (order_id == 3) { a0 = 1; a1 = 2; a2 = 0; }
        else if (order_id == 4) { a0 = 2; a1 = 0; a2 = 1; }
        else { a0 = 2; a1 = 1; a2 = 0; }
        if (!nearest_step_axis_clear(mask, z, y, x, z1, y1, x1, a0, H, W)) return false;
        if (!nearest_step_axis_clear(mask, z, y, x, z1, y1, x1, a1, H, W)) return false;
        if (!nearest_step_axis_clear(mask, z, y, x, z1, y1, x1, a2, H, W)) return false;
        return true;
    }

    extern "C" __global__
    void regular_stride_l1_lattice_nearest_clearance_check_3d(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ label,
        const int* __restrict__ sites_zyx,
        int* __restrict__ bad_count,
        const int nvox,
        const int n_sites,
        const int D,
        const int H,
        const int W
    ) {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= nvox || !mask[idx]) return;

        int lab = label[idx];
        if ((unsigned)lab >= (unsigned)n_sites) {
            atomicAdd(bad_count, 1);
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int sz = sites_zyx[3 * lab + 0];
        int sy = sites_zyx[3 * lab + 1];
        int sx = sites_zyx[3 * lab + 2];

        bool clear = false;
        #pragma unroll
        for (int ord = 0; ord < 6; ++ord) {
            if (nearest_monotone_path_clear6(mask, sz, sy, sx, z, y, x, H, W, ord)) {
                clear = true;
                break;
            }
        }
        if (!clear) atomicAdd(bad_count, 1);
    }
    '''
    return _device_cached_rawkernel(
        build_regular_stride_l1_lattice_nearest_clearance_check_kernel_3d,
        code,
        "regular_stride_l1_lattice_nearest_clearance_check_3d",
        options=("-std=c++11",),
    )


def build_regular_stride_l1_lattice_nearest_certified_label_kernel_3d():
    """Nearest regular-lattice label plus one monotone-path certificate per voxel."""
    code = r'''
    __device__ __forceinline__
    bool nearest_cert_step_axis_clear(
        const unsigned char* __restrict__ mask,
        int& z, int& y, int& x,
        const int tz, const int ty, const int tx,
        const int axis,
        const int H,
        const int W
    ) {
        int target = (axis == 0) ? tz : (axis == 1) ? ty : tx;
        int* curp = (axis == 0) ? &z : (axis == 1) ? &y : &x;
        int dir = (target > *curp) ? 1 : -1;
        while (*curp != target) {
            *curp += dir;
            int idx = z * H * W + y * W + x;
            if (!mask[idx]) return false;
        }
        return true;
    }

    __device__ __forceinline__
    bool nearest_cert_monotone_path_clear6(
        const unsigned char* __restrict__ mask,
        const int z0, const int y0, const int x0,
        const int z1, const int y1, const int x1,
        const int H,
        const int W,
        const int order_id
    ) {
        int z = z0;
        int y = y0;
        int x = x0;
        int a0, a1, a2;
        if (order_id == 0) { a0 = 0; a1 = 1; a2 = 2; }
        else if (order_id == 1) { a0 = 0; a1 = 2; a2 = 1; }
        else if (order_id == 2) { a0 = 1; a1 = 0; a2 = 2; }
        else if (order_id == 3) { a0 = 1; a1 = 2; a2 = 0; }
        else if (order_id == 4) { a0 = 2; a1 = 0; a2 = 1; }
        else { a0 = 2; a1 = 1; a2 = 0; }
        if (!nearest_cert_step_axis_clear(mask, z, y, x, z1, y1, x1, a0, H, W)) return false;
        if (!nearest_cert_step_axis_clear(mask, z, y, x, z1, y1, x1, a1, H, W)) return false;
        if (!nearest_cert_step_axis_clear(mask, z, y, x, z1, y1, x1, a2, H, W)) return false;
        return true;
    }

    extern "C" __global__
    void regular_stride_l1_lattice_nearest_certified_label_3d(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ lattice_lut,
        int* __restrict__ label_out,
        float* __restrict__ dist_out,
        int* __restrict__ bad_count,
        const int nvox,
        const int D,
        const int H,
        const int W,
        const int Lz,
        const int Ly,
        const int Lx,
        const int sz,
        const int sy,
        const int sx,
        const int oz,
        const int oy,
        const int ox
    ) {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= nvox) return;

        if (!mask[idx]) {
            label_out[idx] = -1;
            dist_out[idx] = 1.0e20f;
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int bz = (z - oz) / sz;
        int by = (y - oy) / sy;
        int bx = (x - ox) / sx;
        if (z < oz && ((z - oz) % sz) != 0) --bz;
        if (y < oy && ((y - oy) % sy) != 0) --by;
        if (x < ox && ((x - ox) % sx) != 0) --bx;

        int best_label = -1;
        int best_dist = 2147483647;
        int best_z = 0;
        int best_y = 0;
        int best_x = 0;

        for (int iz = bz - 1; iz <= bz + 2; ++iz) {
            if ((unsigned)iz >= (unsigned)Lz) continue;
            int zz = oz + iz * sz;
            for (int iy = by - 1; iy <= by + 2; ++iy) {
                if ((unsigned)iy >= (unsigned)Ly) continue;
                int yy = oy + iy * sy;
                int lbase = (iz * Ly + iy) * Lx;
                for (int ix = bx - 1; ix <= bx + 2; ++ix) {
                    if ((unsigned)ix >= (unsigned)Lx) continue;
                    int lab = lattice_lut[lbase + ix];
                    if (lab < 0) continue;
                    int xx = ox + ix * sx;
                    int d = abs(z - zz) + abs(y - yy) + abs(x - xx);
                    if (d < best_dist || (d == best_dist && (best_label < 0 || lab < best_label))) {
                        best_dist = d;
                        best_label = lab;
                        best_z = zz;
                        best_y = yy;
                        best_x = xx;
                    }
                }
            }
        }

        label_out[idx] = best_label;
        dist_out[idx] = (best_label >= 0) ? (float)best_dist : 1.0e20f;
        if (best_label < 0) {
            if (bad_count) atomicAdd(bad_count, 1);
            return;
        }

        bool clear = false;
        #pragma unroll
        for (int ord = 0; ord < 6; ++ord) {
            if (nearest_cert_monotone_path_clear6(mask, best_z, best_y, best_x, z, y, x, H, W, ord)) {
                clear = true;
                break;
            }
        }
        if (!clear && bad_count) atomicAdd(bad_count, 1);
    }
    '''
    return _device_cached_rawkernel(
        build_regular_stride_l1_lattice_nearest_certified_label_kernel_3d,
        code,
        "regular_stride_l1_lattice_nearest_certified_label_3d",
        options=("-std=c++11",),
    )


REGULAR_STRIDE_L1_LAST_WALL_TIME = 0.0
REGULAR_STRIDE_L1_LAST_KERNEL_TIME = 0.0
REGULAR_STRIDE_L1_LAST_INIT_TIME = 0.0
REGULAR_STRIDE_L1_LAST_UNCERTIFIED = 0
REGULAR_STRIDE_L1_LAST_NEAREST_BAD = 0
REGULAR_STRIDE_L1_LAST_NEAREST_CHECK_TIME = 0.0
REGULAR_STRIDE_L1_LAST_NEAREST_CERT_TIME = 0.0


def regular_stride_l1_label_gpu_6(
    mask,
    seeds_zyx,
    stride_zyx,
    offset_zyx,
    *,
    kernel=None,
    block_size=256,
    profile_gpu=False,
):
    """Fast local ownership candidate for regular stride sites on a 6-neighbour unit graph.

    This hot path is intended as a certified/staged ROI-JFA initializer.  It is
    exact on cases where local L1 ownership agrees with the graph-geodesic
    frontier result; callers should verify labels or residuals before using it
    as a final exact backend.
    """
    import time
    import numpy as np
    import cupy as cp

    global REGULAR_STRIDE_L1_LAST_WALL_TIME, REGULAR_STRIDE_L1_LAST_KERNEL_TIME, REGULAR_STRIDE_L1_LAST_INIT_TIME, REGULAR_STRIDE_L1_LAST_UNCERTIFIED
    REGULAR_STRIDE_L1_LAST_WALL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_KERNEL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_INIT_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_UNCERTIFIED = 0

    if kernel is None:
        kernel = build_regular_stride_l1_label_kernel_3d()

    total_t0 = time.perf_counter()
    init_t0 = time.perf_counter()
    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    if mask_u8.ndim != 3:
        raise ValueError("regular_stride_l1_label_gpu_6 expects a 3D mask")
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")
    stride = tuple(int(v) for v in stride_zyx)
    offset = tuple(int(v) for v in offset_zyx)
    if any(v <= 0 for v in stride):
        raise ValueError("stride entries must be positive")

    D, H, W = [int(v) for v in mask_u8.shape]
    n_total = int(mask_u8.size)
    n_seed = int(seeds.shape[0])
    flat_mask = mask_u8.ravel()
    seed_lut = cp.full(n_total, cp.int32(-1), dtype=cp.int32)
    label_out = cp.empty(n_total, dtype=cp.int32)
    dist_out = cp.empty(n_total, dtype=cp.float32)

    seed_flat = (
        seeds[:, 0].astype(cp.int64) * cp.int64(H * W)
        + seeds[:, 1].astype(cp.int64) * cp.int64(W)
        + seeds[:, 2].astype(cp.int64)
    ).astype(cp.int32)
    seed_lut[seed_flat] = cp.arange(n_seed, dtype=cp.int32)
    cp.cuda.Stream.null.synchronize()
    REGULAR_STRIDE_L1_LAST_INIT_TIME = float(time.perf_counter() - init_t0)

    block = int(block_size)
    grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event()
        evt1 = cp.cuda.Event()
        evt0.record()
    kernel(
        grid,
        (block,),
        (
            flat_mask,
            seed_lut,
            label_out,
            dist_out,
            np.int32(n_total),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            np.int32(stride[0]),
            np.int32(stride[1]),
            np.int32(stride[2]),
            np.int32(offset[0]),
            np.int32(offset[1]),
            np.int32(offset[2]),
        ),
    )
    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        REGULAR_STRIDE_L1_LAST_KERNEL_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0
    cp.cuda.Stream.null.synchronize()
    REGULAR_STRIDE_L1_LAST_WALL_TIME = float(time.perf_counter() - total_t0)
    return label_out.reshape(mask_u8.shape), dist_out.reshape(mask_u8.shape)


def regular_stride_l1_lattice_label_gpu_6(
    mask,
    lattice_lut,
    stride_zyx,
    offset_zyx,
    *,
    kernel=None,
    block_size=256,
    profile_gpu=False,
    output=None,
):
    """Fast local ownership for regular stride sites using a compact lattice LUT."""
    import time
    import numpy as np
    import cupy as cp

    global REGULAR_STRIDE_L1_LAST_WALL_TIME, REGULAR_STRIDE_L1_LAST_KERNEL_TIME, REGULAR_STRIDE_L1_LAST_INIT_TIME
    REGULAR_STRIDE_L1_LAST_WALL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_KERNEL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_INIT_TIME = 0.0

    if kernel is None:
        kernel = build_regular_stride_l1_lattice_label_kernel_3d()

    total_t0 = time.perf_counter()
    init_t0 = time.perf_counter()
    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    lut = cp.asarray(lattice_lut, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("regular_stride_l1_lattice_label_gpu_6 expects a 3D mask")
    if lut.ndim != 3:
        raise ValueError("lattice_lut must be a 3D int32 array")
    stride = tuple(int(v) for v in stride_zyx)
    offset = tuple(int(v) for v in offset_zyx)
    if any(v <= 0 for v in stride):
        raise ValueError("stride entries must be positive")

    D, H, W = [int(v) for v in mask_u8.shape]
    Lz, Ly, Lx = [int(v) for v in lut.shape]
    n_total = int(mask_u8.size)
    if output is None:
        label_out = cp.empty(n_total, dtype=cp.int32)
        dist_out = cp.empty(n_total, dtype=cp.float32)
    else:
        label_out, dist_out = output
        label_out = label_out.reshape(-1)
        dist_out = dist_out.reshape(-1)
    REGULAR_STRIDE_L1_LAST_INIT_TIME = float(time.perf_counter() - init_t0)

    block = int(block_size)
    grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event()
        evt1 = cp.cuda.Event()
        evt0.record()
    kernel(
        grid,
        (block,),
        (
            mask_u8.ravel(),
            lut.ravel(),
            label_out,
            dist_out,
            np.int32(n_total),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            np.int32(Lz),
            np.int32(Ly),
            np.int32(Lx),
            np.int32(stride[0]),
            np.int32(stride[1]),
            np.int32(stride[2]),
            np.int32(offset[0]),
            np.int32(offset[1]),
            np.int32(offset[2]),
        ),
    )
    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        REGULAR_STRIDE_L1_LAST_KERNEL_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0
    cp.cuda.Stream.null.synchronize()
    REGULAR_STRIDE_L1_LAST_WALL_TIME = float(time.perf_counter() - total_t0)
    return label_out.reshape(mask_u8.shape), dist_out.reshape(mask_u8.shape)


def regular_stride_l1_lattice_clearance_label_gpu_6(
    mask,
    lattice_lut,
    stride_zyx,
    offset_zyx,
    *,
    kernel=None,
    block_size=256,
    profile_gpu=False,
    output=None,
    return_certified=False,
):
    """Regular stride local ownership with monotone fluid-path clearance."""
    import time
    import numpy as np
    import cupy as cp

    global REGULAR_STRIDE_L1_LAST_WALL_TIME, REGULAR_STRIDE_L1_LAST_KERNEL_TIME, REGULAR_STRIDE_L1_LAST_INIT_TIME
    REGULAR_STRIDE_L1_LAST_WALL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_KERNEL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_INIT_TIME = 0.0

    if kernel is None:
        kernel = build_regular_stride_l1_lattice_clearance_label_kernel_3d()

    total_t0 = time.perf_counter()
    init_t0 = time.perf_counter()
    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    lut = cp.asarray(lattice_lut, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("regular_stride_l1_lattice_clearance_label_gpu_6 expects a 3D mask")
    if lut.ndim != 3:
        raise ValueError("lattice_lut must be a 3D int32 array")
    stride = tuple(int(v) for v in stride_zyx)
    offset = tuple(int(v) for v in offset_zyx)
    D, H, W = [int(v) for v in mask_u8.shape]
    Lz, Ly, Lx = [int(v) for v in lut.shape]
    n_total = int(mask_u8.size)
    write_certified = bool(return_certified)
    if output is None:
        label_out = cp.empty(n_total, dtype=cp.int32)
        dist_out = cp.empty(n_total, dtype=cp.float32)
        certified = cp.empty(n_total if write_certified else 1, dtype=cp.uint8)
    else:
        if len(output) == 2:
            label_out, dist_out = output
            certified = cp.empty(n_total if write_certified else 1, dtype=cp.uint8)
        else:
            label_out, dist_out, certified = output
            write_certified = True
        label_out = label_out.reshape(-1)
        dist_out = dist_out.reshape(-1)
        certified = certified.reshape(-1)
    uncertified_count = cp.zeros(1, dtype=cp.int32)
    REGULAR_STRIDE_L1_LAST_INIT_TIME = float(time.perf_counter() - init_t0)

    block = int(block_size)
    grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event()
        evt1 = cp.cuda.Event()
        evt0.record()
    kernel(
        grid,
        (block,),
        (
            mask_u8.ravel(),
            lut.ravel(),
            label_out,
            dist_out,
            certified,
            uncertified_count,
            np.int32(n_total),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            np.int32(Lz),
            np.int32(Ly),
            np.int32(Lx),
            np.int32(stride[0]),
            np.int32(stride[1]),
            np.int32(stride[2]),
            np.int32(offset[0]),
            np.int32(offset[1]),
            np.int32(offset[2]),
            np.int32(1 if write_certified else 0),
        ),
    )
    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        REGULAR_STRIDE_L1_LAST_KERNEL_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0
    cp.cuda.Stream.null.synchronize()
    REGULAR_STRIDE_L1_LAST_UNCERTIFIED = int(uncertified_count.get()[0])
    REGULAR_STRIDE_L1_LAST_WALL_TIME = float(time.perf_counter() - total_t0)
    label = label_out.reshape(mask_u8.shape)
    dist = dist_out.reshape(mask_u8.shape)
    if return_certified:
        return label, dist, certified.reshape(mask_u8.shape)
    return label, dist


def regular_stride_l1_lattice_nearest_clearance_check_gpu_6(
    mask,
    label,
    seeds_zyx,
    *,
    kernel=None,
    block_size=256,
    profile_gpu=False,
):
    """Return the number of fluid voxels whose nearest lattice owner lacks a certified monotone path."""
    import time
    import numpy as np
    import cupy as cp

    global REGULAR_STRIDE_L1_LAST_NEAREST_BAD, REGULAR_STRIDE_L1_LAST_NEAREST_CHECK_TIME
    REGULAR_STRIDE_L1_LAST_NEAREST_BAD = 0
    REGULAR_STRIDE_L1_LAST_NEAREST_CHECK_TIME = 0.0

    if kernel is None:
        kernel = build_regular_stride_l1_lattice_nearest_clearance_check_kernel_3d()

    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    label_i = cp.asarray(label, dtype=cp.int32)
    seeds = cp.asarray(seeds_zyx, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("regular_stride_l1_lattice_nearest_clearance_check_gpu_6 expects a 3D mask")
    if seeds.ndim != 2 or int(seeds.shape[1]) != 3:
        raise ValueError("seeds_zyx must have shape (n, 3)")
    D, H, W = [int(v) for v in mask_u8.shape]
    n_total = int(mask_u8.size)
    n_sites = int(seeds.shape[0])
    bad_count = cp.zeros(1, dtype=cp.int32)

    block = int(block_size)
    grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event()
        evt1 = cp.cuda.Event()
        evt0.record()
    kernel(
        grid,
        (block,),
        (
            mask_u8.ravel(),
            label_i.ravel(),
            seeds.reshape(-1),
            bad_count,
            np.int32(n_total),
            np.int32(n_sites),
            np.int32(D),
            np.int32(H),
            np.int32(W),
        ),
    )
    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        REGULAR_STRIDE_L1_LAST_NEAREST_CHECK_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0
    cp.cuda.Stream.null.synchronize()
    REGULAR_STRIDE_L1_LAST_NEAREST_BAD = int(bad_count.get()[0])
    return REGULAR_STRIDE_L1_LAST_NEAREST_BAD


def regular_stride_l1_lattice_nearest_certified_label_gpu_6(
    mask,
    lattice_lut,
    stride_zyx,
    offset_zyx,
    *,
    kernel=None,
    block_size=256,
    profile_gpu=False,
    output=None,
):
    """Fast nearest-lattice label field with a per-voxel monotone-path certificate."""
    import time
    import numpy as np
    import cupy as cp

    global REGULAR_STRIDE_L1_LAST_WALL_TIME, REGULAR_STRIDE_L1_LAST_KERNEL_TIME, REGULAR_STRIDE_L1_LAST_INIT_TIME
    global REGULAR_STRIDE_L1_LAST_NEAREST_BAD, REGULAR_STRIDE_L1_LAST_NEAREST_CERT_TIME
    REGULAR_STRIDE_L1_LAST_WALL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_KERNEL_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_INIT_TIME = 0.0
    REGULAR_STRIDE_L1_LAST_NEAREST_BAD = 0
    REGULAR_STRIDE_L1_LAST_NEAREST_CERT_TIME = 0.0

    if kernel is None:
        kernel = build_regular_stride_l1_lattice_nearest_certified_label_kernel_3d()

    total_t0 = time.perf_counter()
    init_t0 = time.perf_counter()
    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    lut = cp.asarray(lattice_lut, dtype=cp.int32)
    if mask_u8.ndim != 3:
        raise ValueError("regular_stride_l1_lattice_nearest_certified_label_gpu_6 expects a 3D mask")
    if lut.ndim != 3:
        raise ValueError("lattice_lut must be a 3D int32 array")
    stride = tuple(int(v) for v in stride_zyx)
    offset = tuple(int(v) for v in offset_zyx)
    D, H, W = [int(v) for v in mask_u8.shape]
    Lz, Ly, Lx = [int(v) for v in lut.shape]
    n_total = int(mask_u8.size)
    if output is None:
        label_out = cp.empty(n_total, dtype=cp.int32)
        dist_out = cp.empty(n_total, dtype=cp.float32)
    else:
        label_out, dist_out = output
        label_out = label_out.reshape(-1)
        dist_out = dist_out.reshape(-1)
    bad_count = cp.zeros(1, dtype=cp.int32)
    REGULAR_STRIDE_L1_LAST_INIT_TIME = float(time.perf_counter() - init_t0)

    block = int(block_size)
    grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event()
        evt1 = cp.cuda.Event()
        evt0.record()
    kernel(
        grid,
        (block,),
        (
            mask_u8.ravel(),
            lut.ravel(),
            label_out,
            dist_out,
            bad_count,
            np.int32(n_total),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            np.int32(Lz),
            np.int32(Ly),
            np.int32(Lx),
            np.int32(stride[0]),
            np.int32(stride[1]),
            np.int32(stride[2]),
            np.int32(offset[0]),
            np.int32(offset[1]),
            np.int32(offset[2]),
        ),
    )
    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        REGULAR_STRIDE_L1_LAST_KERNEL_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0
        REGULAR_STRIDE_L1_LAST_NEAREST_CERT_TIME = REGULAR_STRIDE_L1_LAST_KERNEL_TIME
    cp.cuda.Stream.null.synchronize()
    REGULAR_STRIDE_L1_LAST_NEAREST_BAD = int(bad_count.get()[0])
    REGULAR_STRIDE_L1_LAST_WALL_TIME = float(time.perf_counter() - total_t0)
    return label_out.reshape(mask_u8.shape), dist_out.reshape(mask_u8.shape), REGULAR_STRIDE_L1_LAST_NEAREST_BAD


def build_bellman_residual6_kernel():
    """Check 6-neighbour unit-distance Bellman and tie conditions."""
    code = r'''
    extern "C" __global__
    void bellman_residual6_check(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ label,
        const float* __restrict__ dist,
        int* __restrict__ max_residual_i,
        int* __restrict__ tie_bad_count,
        int* __restrict__ unassigned_count,
        const int nvox,
        const int D,
        const int H,
        const int W,
        const float eps
    ) {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= nvox || !mask[idx]) return;

        int lab_v = label[idx];
        float dv = dist[idx];
        if (lab_v < 0 || !isfinite(dv)) {
            atomicAdd(unassigned_count, 1);
            return;
        }

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
            int lab_u = label[nb];
            float du = dist[nb];
            if (lab_u < 0 || !isfinite(du)) {
                atomicAdd(unassigned_count, 1);
                continue;
            }
            float residual = dv - du - 1.0f;
            if (residual > eps) {
                int ri = (int)ceilf(residual * 1000000.0f);
                atomicMax(max_residual_i, ri);
            }
            if (fabsf(dv - (du + 1.0f)) <= eps && lab_u < lab_v) {
                atomicAdd(tie_bad_count, 1);
            }
        }
    }
    '''
    return _device_cached_rawkernel(
        build_bellman_residual6_kernel,
        code,
        "bellman_residual6_check",
        options=("-std=c++11",),
    )


BELLMAN_RESIDUAL6_LAST_WALL_TIME = 0.0
BELLMAN_RESIDUAL6_LAST_KERNEL_TIME = 0.0


def bellman_residual6_check_gpu(mask, label, dist, *, kernel=None, block_size=256, eps=1.0e-6, profile_gpu=False):
    """Return a cheap certificate for a 6-neighbour unit-distance label field."""
    import time
    import numpy as np
    import cupy as cp

    global BELLMAN_RESIDUAL6_LAST_WALL_TIME, BELLMAN_RESIDUAL6_LAST_KERNEL_TIME
    BELLMAN_RESIDUAL6_LAST_WALL_TIME = 0.0
    BELLMAN_RESIDUAL6_LAST_KERNEL_TIME = 0.0

    if kernel is None:
        kernel = build_bellman_residual6_kernel()

    t0 = time.perf_counter()
    mask_u8 = cp.asarray(mask).astype(cp.uint8, copy=False)
    label_i = cp.asarray(label, dtype=cp.int32)
    dist_f = cp.asarray(dist, dtype=cp.float32)
    if mask_u8.ndim != 3:
        raise ValueError("bellman_residual6_check_gpu expects a 3D mask")
    D, H, W = [int(v) for v in mask_u8.shape]
    n_total = int(mask_u8.size)
    max_residual_i = cp.zeros(1, dtype=cp.int32)
    tie_bad_count = cp.zeros(1, dtype=cp.int32)
    unassigned_count = cp.zeros(1, dtype=cp.int32)
    block = int(block_size)
    grid = (max(1, int(math.ceil(float(n_total) / float(block)))),)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event()
        evt1 = cp.cuda.Event()
        evt0.record()
    kernel(
        grid,
        (block,),
        (
            mask_u8.ravel(),
            label_i.ravel(),
            dist_f.ravel(),
            max_residual_i,
            tie_bad_count,
            unassigned_count,
            np.int32(n_total),
            np.int32(D),
            np.int32(H),
            np.int32(W),
            np.float32(float(eps)),
        ),
    )
    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        BELLMAN_RESIDUAL6_LAST_KERNEL_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0
    cp.cuda.Stream.null.synchronize()
    out = {
        "max_residual": float(int(max_residual_i.get()[0])) / 1000000.0,
        "tie_bad_count": int(tie_bad_count.get()[0]),
        "unassigned_count": int(unassigned_count.get()[0]),
    }
    BELLMAN_RESIDUAL6_LAST_WALL_TIME = float(time.perf_counter() - t0)
    return out


FAST6_AUTO_LAST_METHOD = ""
FAST6_AUTO_LAST_CERT = {}
FAST6_AUTO_LAST_WALL_TIME = 0.0


def regular_stride_l1_or_frontier_gpu_6(
    mask,
    seeds_zyx,
    stride_zyx,
    offset_zyx,
    *,
    regular_kernel=None,
    residual_kernel=None,
    frontier_kernel=None,
    frontier_init_kernel=None,
    eps=1.0e-6,
    profile_gpu=False,
    accept_distance_certificate=False,
    return_meta=False,
):
    """Use the regular-site hot path only when the caller accepts its certificate.

    The Bellman scan certifies distance optimality for the 6-neighbour unit
    graph.  It does not, by itself, prove the global minimum-label tie rule on
    every equidistant competition surface.  The default therefore falls back to
    the exact GPU frontier unless accept_distance_certificate=True is supplied
    by a caller that has separately audited labels for the case family.
    """
    import time

    global FAST6_AUTO_LAST_METHOD, FAST6_AUTO_LAST_CERT, FAST6_AUTO_LAST_WALL_TIME
    FAST6_AUTO_LAST_METHOD = ""
    FAST6_AUTO_LAST_CERT = {}
    FAST6_AUTO_LAST_WALL_TIME = 0.0

    t0 = time.perf_counter()
    label, dist = regular_stride_l1_label_gpu_6(
        mask,
        seeds_zyx,
        stride_zyx,
        offset_zyx,
        kernel=regular_kernel,
        profile_gpu=profile_gpu,
    )
    cert = bellman_residual6_check_gpu(
        mask,
        label,
        dist,
        kernel=residual_kernel,
        eps=eps,
        profile_gpu=profile_gpu,
    )
    distance_certified = (
        float(cert["max_residual"]) <= float(eps)
        and int(cert["tie_bad_count"]) == 0
        and int(cert["unassigned_count"]) == 0
    )
    certified = bool(distance_certified and accept_distance_certificate)
    if certified:
        FAST6_AUTO_LAST_METHOD = "regular_stride_l1_distance_certified"
    else:
        label, dist = exact_frontier_dijkstra_gpu_6(
            mask,
            seeds_zyx,
            kernel=frontier_kernel,
            init_kernel=frontier_init_kernel,
            validate=True,
            collect_stats=False,
            return_float64=False,
        )
        FAST6_AUTO_LAST_METHOD = "exact_frontier_fallback"
    FAST6_AUTO_LAST_CERT = {**dict(cert), "distance_certified": bool(distance_certified)}
    FAST6_AUTO_LAST_WALL_TIME = float(time.perf_counter() - t0)
    if return_meta:
        return label, dist, {"method": FAST6_AUTO_LAST_METHOD, **cert}
    return label, dist


def exact_geodesic_voronoi_gpu(
    mask,
    seeds,
    connectivity=26,
    max_iter=None,
    eps=1e-6,
    verbose=False,
    raise_on_nonconvergence=True,
    check_optimality=True,
    profile_gpu=False,
):
    """
    Implementation notes:
      1) seed initialisation: one GPU scatter write (no per-point loop)
      2) mask backend-aware: a cupy mask is never converted to numpy
      3) iteration buffers reused: dist/label ping-pong + cp.copyto, no cudaMalloc per iteration
      4) timing breakdown: INIT / ITER / OPT / TOTAL + CUDA events (optional)
    """
    global EXACT_LAST_GPU_TIME, EXACT_LAST_WALL_TIME
    global EXACT_LAST_TINIT_WALL, EXACT_LAST_TITER_WALL, EXACT_LAST_TOPT_WALL

    EXACT_LAST_GPU_TIME = 0.0
    EXACT_LAST_WALL_TIME = 0.0
    EXACT_LAST_TINIT_WALL = 0.0
    EXACT_LAST_TITER_WALL = 0.0
    EXACT_LAST_TOPT_WALL = 0.0

    t_total0 = time.perf_counter()

    # ------------------------------------------------------------
    # 0) mask -> cupy (NO numpy round-trip if already cupy)
    # ------------------------------------------------------------
    t_init0 = time.perf_counter()
    if isinstance(mask, cp.ndarray):
        mask_cp = mask
        # accept uint8/bool, treat nonzero as fluid
        if mask_cp.dtype != cp.bool_:
            mask_cp = (mask_cp != 0)
    else:
        mask_cp = cp.asarray(mask, dtype=cp.bool_)

    D, H, W = map(int, mask_cp.shape)
    HW = H * W
    nvox = int(D * H * W)

    # ------------------------------------------------------------
    # 1) init label/dist (1D) + vectorized seed injection
    # ------------------------------------------------------------
    INF = cp.float32(1e20)
    dist_a = cp.full(nvox, INF, dtype=cp.float32)
    label_a = cp.full(nvox, -1, dtype=cp.int32)

    # seeds -> cupy int32 (N,3)
    if isinstance(seeds, cp.ndarray):
        seeds_cp = seeds.astype(cp.int32, copy=False)
    else:
        seeds_cp = cp.asarray(np.asarray(seeds, dtype=np.int32), dtype=cp.int32)

    if seeds_cp.ndim != 2 or seeds_cp.shape[1] != 3:
        raise ValueError("seeds must be (N,3) with (z,y,x)")

    n_seeds = int(seeds_cp.shape[0])
    if n_seeds <= 0:
        raise ValueError("seeds is empty")

    # linear indices (int64 safe)
    z = seeds_cp[:, 0].astype(cp.int64, copy=False)
    y = seeds_cp[:, 1].astype(cp.int64, copy=False)
    x = seeds_cp[:, 2].astype(cp.int64, copy=False)
    idx = z * cp.int64(HW) + y * cp.int64(W) + x

    # optional: strictly ensure that the seeds lie in the pore space (normally asserted by the caller, so the costly check is skipped here)
    # to keep the check, enable the two lines below
    # valid = mask_cp.ravel()[idx]  # bool
    # idx = idx[valid]; seed_ids = cp.arange(n_seeds, dtype=cp.int32)[valid]
    seed_ids = cp.arange(n_seeds, dtype=cp.int32)

    # single scatter write
    label_a[idx] = seed_ids
    dist_a[idx] = cp.float32(0.0)

    # reshape for slicing updates
    dist_a = dist_a.reshape((D, H, W))
    label_a = label_a.reshape((D, H, W))

    # ping-pong buffers (reuse)
    dist_b = cp.empty_like(dist_a)
    label_b = cp.empty_like(label_a)

    # max_iter default
    if max_iter is None:
        # still return the theoretical safe upper bound, without numpy.sum
        n_fluid = int(cp.count_nonzero(mask_cp).get())
        max_iter = max(n_fluid, 1)
    else:
        max_iter = int(max_iter)

    EXACT_LAST_TINIT_WALL = float(time.perf_counter() - t_init0)

    # ------------------------------------------------------------
    # 2) offsets
    # ------------------------------------------------------------
    if connectivity == 6:
        offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
    elif connectivity == 26:
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append((dz, dy, dx))
    else:
        raise ValueError("connectivity must be 6 or 26")

    steps = [math.sqrt(float(dz * dz + dy * dy + dx * dx)) for (dz, dy, dx) in offsets]

    def _overlap(n, offset):
        if offset >= 0:
            src_start = 0
            src_end = n - offset
            dst_start = offset
            dst_end = n
        else:
            src_start = -offset
            src_end = n
            dst_start = 0
            dst_end = n + offset
        return src_start, src_end, dst_start, dst_end

    # ------------------------------------------------------------
    # 3) main relaxation (buffer reuse + accurate timing)
    # ------------------------------------------------------------
    t_iter0 = time.perf_counter()

    start_evt = end_evt = None
    if profile_gpu:
        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()
        start_evt.record()

    converged = False
    for it in range(int(max_iter)):
        # copy old -> new (no alloc)
        cp.copyto(dist_b, dist_a)
        cp.copyto(label_b, label_a)

        for (dz, dy, dx), step in zip(offsets, steps):
            z_src_start, z_src_end, z_dst_start, z_dst_end = _overlap(D, dz)
            y_src_start, y_src_end, y_dst_start, y_dst_end = _overlap(H, dy)
            x_src_start, x_src_end, x_dst_start, x_dst_end = _overlap(W, dx)

            src_slice = (slice(z_src_start, z_src_end),
                         slice(y_src_start, y_src_end),
                         slice(x_src_start, x_src_end))
            dst_slice = (slice(z_dst_start, z_dst_end),
                         slice(y_dst_start, y_dst_end),
                         slice(x_dst_start, x_dst_end))

            dist_src  = dist_a[src_slice]
            label_src = label_a[src_slice]
            dist_dst  = dist_b[dst_slice]

            mask_src = mask_cp[src_slice]
            mask_dst = mask_cp[dst_slice]

            valid = mask_src & mask_dst & (label_src >= 0)
            cand_dist = dist_src + cp.float32(step)

            better = valid & (cand_dist < dist_dst - eps)

            # update dist/label
            dist_b[dst_slice] = cp.where(better, cand_dist, dist_dst)
            label_dst = label_b[dst_slice]
            label_b[dst_slice] = cp.where(better, label_src, label_dst)

        # convergence check (sync)
        changed = bool(cp.any(dist_b < dist_a - eps).item())
        dist_a, dist_b = dist_b, dist_a
        label_a, label_b = label_b, label_a

        if verbose:
            print(f"[exact] iter {it+1}, changed={changed}")

        if not changed:
            converged = True
            break

    if profile_gpu:
        end_evt.record()
        end_evt.synchronize()
        EXACT_LAST_GPU_TIME = float(cp.cuda.get_elapsed_time(start_evt, end_evt)) / 1000.0

    EXACT_LAST_TITER_WALL = float(time.perf_counter() - t_iter0)

    if not converged:
        msg = (f"[exact] NOT converged after max_iter={max_iter}. "
               f"This means your 'exact' reference may be unreliable.")
        if raise_on_nonconvergence:
            raise RuntimeError(msg)
        else:
            print("WARNING:", msg)

    # ------------------------------------------------------------
    # 4) optional optimality check (very expensive on big domains)
    # ------------------------------------------------------------
    t_opt0 = time.perf_counter()

    if check_optimality:
        for (dz, dy, dx), step in zip(offsets, steps):
            z_src_start, z_src_end, z_dst_start, z_dst_end = _overlap(D, dz)
            y_src_start, y_src_end, y_dst_start, y_dst_end = _overlap(H, dy)
            x_src_start, x_src_end, x_dst_start, x_dst_end = _overlap(W, dx)

            src_slice = (slice(z_src_start, z_src_end),
                         slice(y_src_start, y_src_end),
                         slice(x_src_start, x_src_end))
            dst_slice = (slice(z_dst_start, z_dst_end),
                         slice(y_dst_start, y_dst_end),
                         slice(x_dst_start, x_dst_end))

            dist_src  = dist_a[src_slice]
            dist_dst  = dist_a[dst_slice]
            label_src = label_a[src_slice]
            mask_src  = mask_cp[src_slice]
            mask_dst  = mask_cp[dst_slice]

            valid = mask_src & mask_dst & (label_src >= 0)
            cand_dist = dist_src + cp.float32(step)
            better = valid & (cand_dist < dist_dst - eps)
            if bool(cp.any(better).item()):
                msg = ("[exact] Optimality check failed: relaxable edges still exist.")
                if raise_on_nonconvergence:
                    raise RuntimeError(msg)
                else:
                    print("WARNING:", msg)
                break

    EXACT_LAST_TOPT_WALL = float(time.perf_counter() - t_opt0)

    # enforce solid convention
    solid = ~mask_cp
    if bool(cp.any(solid).item()):
        dist_a = dist_a.copy()
        label_a = label_a.copy()
        dist_a[solid] = INF
        label_a[solid] = -1

    EXACT_LAST_WALL_TIME = float(time.perf_counter() - t_total0)
    return label_a, dist_a




# ============================================================
# 1.A Visualisation tools (Plotly, interactive HTML)
# ============================================================

def _ensure_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        if isinstance(arr, cp.ndarray):
            return arr.get()
    except Exception:
        pass
    return np.asarray(arr)


# ============================================================
# C-field assignment: C(v) = C_seed[label(v)]
# ============================================================
def compute_C_field_from_labels(label, C_seed, mask=None, fill_value=None):
    """
    Compute the voxel field C with
        C(v) = C_seed[label(v)]
    Invalid voxels (solid, label<0 or label>=n_seeds) take fill_value.

    Parameters
    ----------
    label : (D,H,W) int32/int64, numpy or cupy
        Voronoi label field (seed id); solid/invalid usually -1
    C_seed : (n_seeds,) or (n_seeds, K), numpy or cupy
        property table of the seeds (scalar or vector)
    mask : (D,H,W) optional, bool/uint8, numpy or cupy
        True/1 = fluid; False/0 = solid. If given, mask==0 is also filled with fill_value
    fill_value : optional
        value for invalid voxels.
        - None: NaN for float dtype; -1 for integer dtype

    Returns
    -------
    C : (D,H,W) or (D,H,W,K), same backend as label (numpy/cupy)
    """
    import numpy as np
    try:
        import cupy as cp
        use_gpu = isinstance(label, cp.ndarray) or isinstance(C_seed, cp.ndarray) or isinstance(mask, cp.ndarray)
    except Exception:
        cp = None
        use_gpu = False

    if use_gpu:
        label_cp = label if isinstance(label, cp.ndarray) else cp.asarray(label)
        C_seed_cp = C_seed if isinstance(C_seed, cp.ndarray) else cp.asarray(C_seed)

        if label_cp.dtype not in (cp.int32, cp.int64):
            label_cp = label_cp.astype(cp.int32)

        n_seeds = int(C_seed_cp.shape[0])

        # ---- default fill_value ----
        if fill_value is None:
            if cp.issubdtype(C_seed_cp.dtype, cp.floating):
                fill_value_cp = cp.asarray(cp.nan, dtype=C_seed_cp.dtype)
            else:
                fill_value_cp = cp.asarray(-1, dtype=C_seed_cp.dtype)
        else:
            fill_value_cp = cp.asarray(fill_value, dtype=C_seed_cp.dtype)

        # ======================================================
        # the lookup itself: C(v) = C_seed[label(v)]
        # note: label = -1 would give a negative index, so use take(mode='clip') and correct with the invalid mask
        # ======================================================
        C = cp.take(C_seed_cp, label_cp, axis=0, mode="clip")  # <-- C(v) = C_seed[label(v)] (clip guards against out-of-range and negative indices)

        invalid = (label_cp < 0) | (label_cp >= n_seeds)
        if mask is not None:
            mask_cp = mask if isinstance(mask, cp.ndarray) else cp.asarray(mask)
            invalid |= (mask_cp == 0)

        if C.ndim == label_cp.ndim:
            C = cp.where(invalid, fill_value_cp, C)
        else:
            C = cp.where(invalid[..., None], fill_value_cp, C)

        return C

    else:
        label_np = np.asarray(label)
        C_seed_np = np.asarray(C_seed)
        if label_np.dtype.kind not in ("i", "u"):
            label_np = label_np.astype(np.int32)

        n_seeds = int(C_seed_np.shape[0])

        if fill_value is None:
            if np.issubdtype(C_seed_np.dtype, np.floating):
                fill_value_np = np.asarray(np.nan, dtype=C_seed_np.dtype)
            else:
                fill_value_np = np.asarray(-1, dtype=C_seed_np.dtype)
        else:
            fill_value_np = np.asarray(fill_value, dtype=C_seed_np.dtype)

        C = np.take(C_seed_np, label_np, axis=0, mode="clip")  # <-- C(v)=C_seed[label(v)]
        invalid = (label_np < 0) | (label_np >= n_seeds)
        if mask is not None:
            invalid |= (np.asarray(mask) == 0)

        if C.ndim == label_np.ndim:
            C = np.where(invalid, fill_value_np, C)
        else:
            C = np.where(invalid[..., None], fill_value_np, C)

        return C




# ============================================================
# Debug probes: inspect mask/label/dist at a slice location
#       (No solver logic changes; only for inspection)
# ============================================================

def debug_probe_slice_point(
    mask, label, dist,
    axis="y",
    index=None,
    row=0, col=0,
    r=2,                 # print (2r+1)x(2r+1) neighborhood
    print_dist=True,
):
    """
    Read mask/label/dist at (row, col) of a 2D visualisation slice and print a small neighbourhood.

    axis:
      - 'y': slice is (z,x), row=z, col=x      -> voxel (z, y=index, x)
      - 'z': slice is (y,x), row=y, col=x      -> voxel (z=index, y, x)
      - 'x': slice is (z,y), row=z, col=y      -> voxel (z, y, x=index)
    """
    import numpy as np

    mask_np  = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label).astype(np.int32)
    dist_np  = _ensure_numpy(dist).astype(np.float32)

    D, H, W = mask_np.shape
    axis = str(axis).lower().strip()

    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    if index is None:
        index = {"z": D // 2, "y": H // 2, "x": W // 2}[axis]

    if axis == "y":
        # slice: (z,x)
        m2 = mask_np[:, index, :]
        L2 = label_np[:, index, :]
        D2 = dist_np[:, index, :]

        def map_to_3d(rr, cc):  # rr=z, cc=x
            return int(rr), int(index), int(cc)

        row_name, col_name = "z(row)", "x(col)"

    elif axis == "z":
        # slice: (y,x)
        m2 = mask_np[index, :, :]
        L2 = label_np[index, :, :]
        D2 = dist_np[index, :, :]

        def map_to_3d(rr, cc):  # rr=y, cc=x
            return int(index), int(rr), int(cc)

        row_name, col_name = "y(row)", "x(col)"

    else:  # axis == "x"
        # slice: (z,y)
        m2 = mask_np[:, :, index]
        L2 = label_np[:, :, index]
        D2 = dist_np[:, :, index]

        def map_to_3d(rr, cc):  # rr=z, cc=y
            return int(rr), int(cc), int(index)

        row_name, col_name = "z(row)", "y(col)"

    R, C = m2.shape
    if not (0 <= row < R and 0 <= col < C):
        raise ValueError(f"(row,col)=({row},{col}) out of slice shape {m2.shape}.")

    z, y, x = map_to_3d(row, col)

    print("=" * 100)
    print(f"[probe] axis='{axis}', index={index}")
    print(f"        slice({row_name},{col_name})=({row},{col}) -> voxel(z,y,x)=({z},{y},{x})")
    print(f"        mask={bool(mask_np[z, y, x])}, label={int(label_np[z, y, x])}, dist={float(dist_np[z, y, x])}")

    r0 = max(0, row - r); r1 = min(R, row + r + 1)
    c0 = max(0, col - r); c1 = min(C, col + r + 1)

    print("-" * 100)
    print(f"[patch] rows {r0}:{r1}, cols {c0}:{c1}, shape={(r1-r0, c1-c0)}")

    m_patch = m2[r0:r1, c0:c1].astype(np.int32)
    L_patch = L2[r0:r1, c0:c1].astype(np.int32)

    print("[mask patch] 1=fluid, 0=solid")
    print(m_patch)

    print("[label patch] (solid is usually -1; fluid==1 with label==-1 means a missing or unassigned label)")
    print(L_patch)

    bad = (m_patch == 1) & (L_patch < 0)
    print(f"[stats] fluid={int(m_patch.sum())}, solid={int((m_patch==0).sum())}, fluid_with_label_neg={int(bad.sum())}")

    if print_dist:
        np.set_printoptions(precision=3, suppress=True)
        D_patch = D2[r0:r1, c0:c1].astype(np.float32)
        print("[dist patch]")
        print(D_patch)

    print("=" * 100)


def debug_dump_slice_patch(
    mask, label, dist,
    axis="y",
    index=None,
    row_range=(0, 10),
    col_range=(0, 10),
    print_dist=False,
):
    """
    Dump a 2D slice patch; row/col as in debug_probe_slice_point.

    Use it to tell what a white edge in a slice image is:
      A) mask==0 -> wall/solid (an edge seen there comes from the visualisation overlay)
      B) mask==1 and different labels left and right -> a Voronoi boundary inside the fluid
         (for example from the Euclidean + clipping baseline crossing a wall)
      C) mask==1 but label==-1 -> a voxel the algorithm left unlabelled (a serious bug)
    """
    import numpy as np

    mask_np  = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label).astype(np.int32)
    dist_np  = _ensure_numpy(dist).astype(np.float32)

    D, H, W = mask_np.shape
    axis = str(axis).lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x','y','z'.")

    if index is None:
        index = {"z": D // 2, "y": H // 2, "x": W // 2}[axis]

    if axis == "y":
        m2 = mask_np[:, index, :]   # (z,x)
        L2 = label_np[:, index, :]
        D2 = dist_np[:, index, :]
        row_name, col_name = "z(row)", "x(col)"
    elif axis == "z":
        m2 = mask_np[index, :, :]   # (y,x)
        L2 = label_np[index, :, :]
        D2 = dist_np[index, :, :]
        row_name, col_name = "y(row)", "x(col)"
    else:
        m2 = mask_np[:, :, index]   # (z,y)
        L2 = label_np[:, :, index]
        D2 = dist_np[:, :, index]
        row_name, col_name = "z(row)", "y(col)"

    r0, r1 = row_range
    c0, c1 = col_range
    r0 = max(0, int(r0)); r1 = min(m2.shape[0], int(r1))
    c0 = max(0, int(c0)); c1 = min(m2.shape[1], int(c1))

    m_patch = m2[r0:r1, c0:c1].astype(np.int32)
    L_patch = L2[r0:r1, c0:c1].astype(np.int32)

    print("=" * 100)
    print(f"[dump] axis='{axis}', index={index}, rows({row_name})={r0}:{r1}, cols({col_name})={c0}:{c1}")
    print("[mask patch] 1=fluid, 0=solid")
    print(m_patch)
    print("[label patch]")
    print(L_patch)

    bad = (m_patch == 1) & (L_patch < 0)
    print(f"[stats] fluid={int(m_patch.sum())}, solid={int((m_patch==0).sum())}, fluid_with_label_neg={int(bad.sum())}")

    if print_dist:
        np.set_printoptions(precision=3, suppress=True)
        print("[dist patch]")
        print(D2[r0:r1, c0:c1].astype(np.float32))

    print("=" * 100)



def debug_compare_two_slice_points_same_label(
    mask,
    label,
    axis="y",
    index=None,
    pA=(50, 39),
    pB=(50, 56),
    nameA="A",
    nameB="B",
):
    """
    Check whether the voxels of two points of a 2D visualisation slice (row, col) carry the same label.

    axis:
      - 'y': slice is (z,x), row=z, col=x      -> voxel (z, y=index, x)
      - 'z': slice is (y,x), row=y, col=x      -> voxel (z=index, y, x)
      - 'x': slice is (z,y), row=z, col=y      -> voxel (z, y, x=index)

    Returns:
      same=True/False, and prints the details
    """
    import numpy as np

    mask_np  = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label).astype(np.int32)

    D, H, W = mask_np.shape
    axis = str(axis).lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x','y','z'")

    if index is None:
        index = {"z": D // 2, "y": H // 2, "x": W // 2}[axis]
    index = int(index)

    def _map_to_3d(p):
        rr, cc = map(int, p)
        if axis == "y":
            z, y, x = rr, index, cc
        elif axis == "z":
            z, y, x = index, rr, cc
        else:  # axis == "x"
            z, y, x = rr, cc, index
        return z, y, x

    zA, yA, xA = _map_to_3d(pA)
    zB, yB, xB = _map_to_3d(pB)

    def _inb(z, y, x):
        return (0 <= z < D) and (0 <= y < H) and (0 <= x < W)

    if not _inb(zA, yA, xA):
        raise ValueError(f"{nameA} out of bounds: voxel=({zA},{yA},{xA}), shape={mask_np.shape}")
    if not _inb(zB, yB, xB):
        raise ValueError(f"{nameB} out of bounds: voxel=({zB},{yB},{xB}), shape={mask_np.shape}")

    mA = bool(mask_np[zA, yA, xA])
    mB = bool(mask_np[zB, yB, xB])
    lA = int(label_np[zA, yA, xA])
    lB = int(label_np[zB, yB, xB])

    print("=" * 100)
    print(f"[compare] axis='{axis}', index={index}")
    print(f"  {nameA}: slice(row,col)={tuple(pA)} -> voxel(z,y,x)=({zA},{yA},{xA}), mask={mA}, label={lA}")
    print(f"  {nameB}: slice(row,col)={tuple(pB)} -> voxel(z,y,x)=({zB},{yB},{xB}), mask={mB}, label={lB}")

    same = (mA and mB and (lA >= 0) and (lB >= 0) and (lA == lB))
    print(f"  => same label? {same}")

    if (not mA) or (not mB):
        print("  NOTE: at least one point is in solid (mask==0); compare at fluid voxels instead.")
    if (mA and lA < 0) or (mB and lB < 0):
        print("  NOTE: at least one point is fluid but label==-1; this flags a missing or unassigned label.")

    print("=" * 100)
    return same


def check_obstacle_two_sides_same_label_blocked(
    mask,
    label,
    *,
    z,
    y,
    x_left,
    x_right,
    name="",
    require_blocked_between=True,
    verbose=True,
):
    """
    Check whether the two ends (x_left, x_right) of an obstacle carry the same label, and that no
    through-layer / gate lies between them.

    Key points:
      1) "two ends": only the labels at (z,y,x_left) and (z,y,x_right) are compared.
      2) "the cell has not crossed a small through-layer in between":
         - with require_blocked_between=True, every voxel between x_left and x_right must be solid (mask==0),
           i.e. mask[z,y, min(x_left,x_right)+1 : max(x_left,x_right)] contains no True/1.
         - if mask==1 occurs in between (a gate/channel), the check is skipped, so that a legal
           crossing is not reported as crossing a wall.

    Returns a dict with blocked_between / same_label and further fields, for a direct check after each method M.
    """
    import numpy as np

    mask_np = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label).astype(np.int32)

    D, H, W = mask_np.shape

    z = int(z); y = int(y)
    x_left = int(x_left); x_right = int(x_right)

    if not (0 <= z < D and 0 <= y < H and 0 <= x_left < W and 0 <= x_right < W):
        raise ValueError(
            f"[two-side-check] out of bounds: "
            f"(z,y,x_left,x_right)=({z},{y},{x_left},{x_right}), shape={mask_np.shape}"
        )

    x0 = min(x_left, x_right)
    x1 = max(x_left, x_right)

    # ------------------------------------------------------------
    # 1) check that everything between the two ends is solid, which excludes a gate / through-layer
    # ------------------------------------------------------------
    blocked_between = True
    if require_blocked_between and (x1 - x0) > 1:
        # check only the strict interior (ends excluded)
        mid_has_fluid = bool(mask_np[z, y, (x0 + 1):x1].any())
        blocked_between = (not mid_has_fluid)

    # ------------------------------------------------------------
    # 2) read (mask, label) at the two ends
    # ------------------------------------------------------------
    mL = bool(mask_np[z, y, x_left])
    mR = bool(mask_np[z, y, x_right])

    lL = int(label_np[z, y, x_left])
    lR = int(label_np[z, y, x_right])

    comparable = bool(mL and mR and (blocked_between or (not require_blocked_between)))
    same_label = bool(comparable and (lL >= 0) and (lR >= 0) and (lL == lR))

    if verbose:
        title = f"[two-side-check] {name}".strip()
        print("=" * 100)
        print(title)
        print(f"  probe voxel L: (z={z}, y={y}, x={x_left})  -> mask={mL}, label={lL}")
        print(f"  probe voxel R: (z={z}, y={y}, x={x_right}) -> mask={mR}, label={lR}")
        print(f"  require_blocked_between={bool(require_blocked_between)}")
        print(f"  blocked_between={bool(blocked_between)}  "
              f"(mid range x={x0+1}:{x1-1} must be all solid)")
        if require_blocked_between and (not blocked_between):
            print("  => SKIP: probe line goes through a gate/pass layer (mask==1 inside the wall slab).")
        else:
            print(f"  => same label across obstacle sides? {same_label}")
        print("=" * 100)

    return {
        "name": str(name),
        "z": z, "y": y,
        "x_left": x_left, "x_right": x_right,
        "require_blocked_between": bool(require_blocked_between),
        "blocked_between": bool(blocked_between),
        "left_mask": mL, "right_mask": mR,
        "left_label": lL, "right_label": lR,
        "comparable": bool(comparable),
        "same_label": bool(same_label),
    }



def get_nature_palette(n_colors, style="soft"):
    """Colour palette in a journal style"""
    palettes = {
        'soft': [
            '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
            '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
            '#86BCB6', '#8CD17D', '#B6992D', '#499894', '#D37295',
        ],
        'vivid': [
            '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
            '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
            '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
        ],
    }

    base = palettes.get(style, palettes['soft'])
    full_palette = (base * ((n_colors // len(base)) + 1))[:n_colors]
    return full_palette

from scipy import ndimage
import numpy as np

def analyze_cell_cut_by_label_3d(
    mask,
    label,
    *,
    connectivity=26,
    min_component_voxels=16,
    min_component_fraction=0.0,
    report_topk=10,
    return_details=True,
    name="",
):
    """
    Statistics of cells cut into pieces, from the labels (3D):

    for each label k:
      Omega_k = (mask==1) & (label==k)
      count the connected components n_comp of Omega_k
      if n_comp >= 2 (after the size filters), the cell counts as cut.

    Parameters
    ----------
    connectivity : 6 or 26
        connectivity; 26 is recommended for the 26-neighbour geodesic metric.
    min_component_voxels : int
        filter for small spurious pieces: components with fewer voxels are ignored in the cut test.
    min_component_fraction : float in [0,1)
        second filter: components smaller than fraction * cell_total_voxels are ignored.
        effective threshold = max(min_component_voxels, ceil(min_component_fraction * total))
    report_topk : int
        report the top-k worst cells (sorted by the size of the second-largest component).
    return_details : bool
        True -> also return the per-label details; False -> summary only.
    name : str
        label used in the printout only.

    Returns
    -------
    summary : dict
      - n_labels
      - n_unassigned_in_fluid
      - n_cut_labels
      - cut_labels (list)
    details : dict (optional)
      details[k] = {
        'total_voxels': int,
        'n_components_raw': int,         # before filtering
        'component_sizes_raw': [...],    # before filtering (descending)
        'threshold': int,
        'n_components_kept': int,        # after filtering
        'component_sizes_kept': [...],   # after filtering (descending)
        'largest_fraction': float,
        'second_largest': int,
      }
    """
    mask_np  = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label).astype(np.int32)

    fluid = mask_np
    n_unassigned = int(np.count_nonzero(fluid & (label_np < 0)))

    # only valid labels inside the fluid matter
    valid = fluid & (label_np >= 0)
    if not np.any(valid):
        summary = {
            "name": str(name),
            "n_labels": 0,
            "n_unassigned_in_fluid": n_unassigned,
            "n_cut_labels": 0,
            "cut_labels": [],
        }
        return (summary, {}) if return_details else summary

    max_label = int(label_np[valid].max())
    # pre-extract bounding boxes once with find_objects (cheaper than a full scan per label)
    packed = np.zeros_like(label_np, dtype=np.int32)
    packed[valid] = label_np[valid] + 1  # 0 is the background
    bboxes = ndimage.find_objects(packed)  # list length = max_label+1

    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)  # 6-neigh
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)  # 26-neigh
    else:
        raise ValueError("connectivity must be 6 or 26")

    cut_labels = []
    details = {}

    # loop over the labels that occur (0..max_label), skipping bbox=None
    for k in range(max_label + 1):
        slc = bboxes[k]
        if slc is None:
            continue

        # binary connected components inside the bbox
        region = (packed[slc] == (k + 1))
        total = int(region.sum())
        if total == 0:
            continue

        cc, ncc = ndimage.label(region, structure=struct)
        if ncc <= 1:
            if return_details:
                details[int(k)] = {
                    "total_voxels": total,
                    "n_components_raw": int(ncc),
                    "component_sizes_raw": [total],
                    "threshold": int(max(min_component_voxels, int(np.ceil(min_component_fraction * total)))),
                    "n_components_kept": 1,
                    "component_sizes_kept": [total],
                    "largest_fraction": 1.0,
                    "second_largest": 0,
                }
            continue

        sizes = np.bincount(cc.ravel())[1:]  # drop background 0
        sizes_sorted = sorted((int(s) for s in sizes), reverse=True)

        thr = int(max(min_component_voxels, int(np.ceil(min_component_fraction * total))))
        kept = [s for s in sizes_sorted if s >= thr]
        n_kept = len(kept)

        largest_frac = float(sizes_sorted[0]) / float(total)
        second = int(sizes_sorted[1]) if len(sizes_sorted) >= 2 else 0

        is_cut = (n_kept >= 2)  # counted as cut only if >= 2 pieces remain after filtering
        if is_cut:
            cut_labels.append(int(k))

        if return_details:
            details[int(k)] = {
                "total_voxels": total,
                "n_components_raw": int(ncc),
                "component_sizes_raw": sizes_sorted,
                "threshold": thr,
                "n_components_kept": int(n_kept),
                "component_sizes_kept": kept,
                "largest_fraction": float(largest_frac),
                "second_largest": int(second),
            }

    summary = {
        "name": str(name),
        "n_labels": int(len(details) if return_details else (max_label + 1)),
        "n_unassigned_in_fluid": int(n_unassigned),
        "n_cut_labels": int(len(cut_labels)),
        "cut_labels": cut_labels,
    }

    # ---- print a short summary (optional) ----
    if report_topk is not None and int(report_topk) > 0 and return_details:
        # sort by second_largest, descending, to find the worst cut
        worst = sorted(
            (k for k in cut_labels),
            key=lambda kk: details[kk]["second_largest"],
            reverse=True,
        )[:int(report_topk)]

        print("=" * 100)
        tag = f"[cell-cut] {name}".strip()
        print(tag)
        print(f"  connectivity={connectivity}, min_component_voxels={min_component_voxels}, "
              f"min_component_fraction={min_component_fraction}")
        print(f"  unassigned_in_fluid = {summary['n_unassigned_in_fluid']}")
        print(f"  cut_labels = {summary['n_cut_labels']} / {len(details)}")
        if worst:
            print("  worst cut cells (by 2nd-largest component):")
            for kk in worst:
                info = details[kk]
                print(f"    label={kk}: total={info['total_voxels']}, "
                      f"raw_comps={info['n_components_raw']}, kept_comps={info['n_components_kept']}, "
                      f"sizes_kept={info['component_sizes_kept'][:5]}")
        print("=" * 100)

    return (summary, details) if return_details else summary


def visualize_exact_geodesic_3d(
    mask,
    label,
    dist,
    seeds,
    html_path="exact_geodesic_voronoi_3d.html",
    color_mode="label",
    smoothing_sigma=0.8,
    mesh_simplify_factor=2,
):
    """
    3D visualisation written as an interactive HTML file
    (Plotly; rotate and zoom in the browser)

    - color_mode="label": regions coloured by Voronoi label
    - color_mode="distance": coloured by geodesic distance (one hull for the whole fluid domain)
    """
    try:
        import plotly.graph_objects as go
        from skimage import measure
    except ImportError:
        print("Error: Please install plotly and scikit-image:")
        print("  pip install plotly scikit-image")
        return None

    print(f"Generating interactive 3D visualization using Plotly...")

    mask_np = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label).astype(np.int32)
    dist_np = _ensure_numpy(dist).astype(np.float32)
    seeds_np = _ensure_numpy(seeds) if seeds is not None else None

    D, H, W = mask_np.shape
    print(f"  Data shape: {D} x {H} x {W}")

    unique_labels = np.unique(label_np[mask_np & (label_np >= 0)])
    n_regions = len(unique_labels)
    colors = get_nature_palette(n_regions, style="soft")
    print(f"  Found {n_regions} Voronoi regions")

    fig = go.Figure()

    from scipy.ndimage import binary_dilation

    # padding wider than 3*sigma so that Gaussian smoothing does not touch the boundary
    pad_width = max(3, int(np.ceil(3 * smoothing_sigma)) + 1)

    if color_mode == "label":
        # ----------------------------------------------------
        # one mesh per Voronoi region
        # ----------------------------------------------------
        for idx, lbl in enumerate(unique_labels):
            region_mask = (label_np == lbl) & mask_np
            if not region_mask.any():
                continue

            # slight dilation, restricted to the mask
            dilated_region = binary_dilation(region_mask, iterations=1) & mask_np

            # padding
            padded_region = np.pad(
                dilated_region,
                pad_width=pad_width,
                mode="constant",
                constant_values=False,
            )

            float_region = padded_region.astype(np.float32)
            float_region = ndimage.gaussian_filter(float_region, sigma=smoothing_sigma)

            # check that level lies in the value range before calling marching_cubes
            vmin = float_region.min()
            vmax = float_region.max()
            # for small or not yet formed regions float_region may be all zero, so 0.5 is outside [vmin, vmax]
            if not (vmin < 0.5 < vmax):
                continue

            try:
                verts, faces, normals, values = measure.marching_cubes(
                    float_region,
                    level=0.5,
                    step_size=mesh_simplify_factor,
                    allow_degenerate=False
                )

                if len(verts) == 0 or len(faces) == 0:
                    continue

                # coordinate conversion (subtract the padding offset)
                x = verts[:, 2] - pad_width
                y = verts[:, 1] - pad_width
                z = verts[:, 0] - pad_width

                i_faces = faces[:, 0]
                j_faces = faces[:, 1]
                k_faces = faces[:, 2]

                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i_faces, j=j_faces, k=k_faces,
                    color=colors[idx],
                    opacity=1.0,
                    name=f'Region {lbl}',
                    showlegend=True,
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.6,
                        specular=0.3,
                        roughness=0.5,
                        fresnel=0.2
                    ),
                    lightposition=dict(x=100, y=200, z=300),
                    flatshading=False,
                ))

            except Exception as e:
                print(f"  Warning: Could not create mesh for region {lbl}: {e}")
                continue

    else:  # distance mode
        # ----------------------------------------------------
        # hull of the whole fluid domain, coloured by distance
        # ----------------------------------------------------
        padded_mask = np.pad(mask_np, pad_width=pad_width,
                             mode='constant', constant_values=False)
        padded_dist = np.pad(dist_np, pad_width=pad_width, mode='edge')

        float_mask = padded_mask.astype(np.float32)
        float_mask = ndimage.gaussian_filter(float_mask, sigma=smoothing_sigma)

        D_pad, H_pad, W_pad = padded_mask.shape

        try:
            verts, faces, normals, values = measure.marching_cubes(
                float_mask,
                level=0.5,
                step_size=mesh_simplify_factor,
                allow_degenerate=False
            )

            if len(verts) > 0 and len(faces) > 0:
                x = verts[:, 2] - pad_width
                y = verts[:, 1] - pad_width
                z = verts[:, 0] - pad_width

                i_faces = faces[:, 0]
                j_faces = faces[:, 1]
                k_faces = faces[:, 2]

                # distance value at each vertex
                zi_pad = np.clip(np.round(verts[:, 0]).astype(int), 0, D_pad-1)
                yi_pad = np.clip(np.round(verts[:, 1]).astype(int), 0, H_pad-1)
                xi_pad = np.clip(np.round(verts[:, 2]).astype(int), 0, W_pad-1)
                intensity = padded_dist[zi_pad, yi_pad, xi_pad]

                # handle infinite values
                valid_mask = np.isfinite(intensity) & (intensity < 1e10)
                if np.any(valid_mask):
                    max_val = float(np.nanpercentile(intensity[valid_mask], 99))
                    intensity = intensity.astype(np.float32, copy=False)
                    intensity[~valid_mask] = max_val
                else:
                    # fallback: all invalid -> set zeros to avoid plotly crash
                    intensity = np.zeros_like(intensity, dtype=np.float32)


                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i_faces, j=j_faces, k=k_faces,
                    intensity=intensity,
                    colorscale='Viridis',
                    opacity=1.0,
                    name='Distance Field',
                    showlegend=True,
                    colorbar=dict(title='Geodesic Distance'),
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.6,
                        specular=0.2,
                    ),
                    flatshading=False,
                ))
        except Exception as e:
            print(f"  Warning: Could not create distance mesh: {e}")

    # --------------------------------------------------------
    # add the seed points
    # --------------------------------------------------------
    if seeds_np is not None and seeds_np.size > 0:
        sx = seeds_np[:, 2]
        sy = seeds_np[:, 1]
        sz = seeds_np[:, 0]

        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode='markers',
            marker=dict(
                size=8,
                color='white',
                line=dict(color='black', width=2),
                symbol='circle',
            ),
            name='Seeds',
            showlegend=True,
        ))

    # layout
    fig.update_layout(
        title=dict(
            text='3D Geodesic Voronoi Tessellation',
            font=dict(size=20, family='Arial')
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            bgcolor='white',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
    )

    print(f"  Saving interactive HTML to: {html_path}")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"  -> Open {html_path} in a web browser to interact with the 3D visualization")

    return fig



def visualize_exact_geodesic_slice(
    mask,
    label,
    dist,
    seeds=None,
    axis="z",
    index=None,
    figsize=(8, 4),
    save_path=None,
    dpi=300,

    # ===== make it generic / paper-friendly =====
    tessellation_title="Voronoi Tessellation",
    distance_title="Distance Field",
    distance_cbar_label="Geodesic Distance",
    solid_color="#E8E8E8",   # color for solid voxels in tessellation panel
    draw_boundaries=True,    # False -> no white boundary overlay
    draw_seeds=True,         # False -> no white seed dots
    # ===== boundary controls =====
    boundary_connectivity=8,         # 4 or 8
    draw_solid_boundaries=False,      # True -> close the boundary along walls
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib import patheffects
    import matplotlib.ticker as ticker
    import numpy as np

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'mathtext.fontset': 'dejavusans',
    })

    mask_np = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label)
    dist_np = _ensure_numpy(dist)

    D, H, W = mask_np.shape
    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("axis must be 'z', 'y' or 'x'.")

    if axis == "z":
        n_slices = D
    elif axis == "y":
        n_slices = H
    else:
        n_slices = W

    if index is None:
        if axis == "z":
            fluid_counts = mask_np.reshape(D, -1).sum(axis=1)
        elif axis == "y":
            fluid_counts = mask_np.transpose(1, 0, 2).reshape(H, -1).sum(axis=1)
        else:
            fluid_counts = mask_np.transpose(2, 0, 1).reshape(W, -1).sum(axis=1)

        non_empty = np.where(fluid_counts > 0)[0]
        if len(non_empty) == 0:
            index = n_slices // 2
        else:
            index = non_empty[len(non_empty) // 2]
    else:
        if not (0 <= index < n_slices):
            raise ValueError(f"index out of range: 0 <= index < {n_slices}")

    # take the slice
    if axis == "z":
        m_slice = mask_np[index, :, :]
        L_slice = label_np[index, :, :]
        D_slice = dist_np[index, :, :]
        xlabel, ylabel = 'x', 'y'
    elif axis == "y":
        m_slice = mask_np[:, index, :]
        L_slice = label_np[:, index, :]
        D_slice = dist_np[:, index, :]
        xlabel, ylabel = 'x', 'z'
    else:
        m_slice = mask_np[:, :, index]
        L_slice = label_np[:, :, index]
        D_slice = dist_np[:, :, index]
        xlabel, ylabel = 'y', 'z'

    # seg: solid=-2, unassigned-fluid=-1, assigned labels >=0
    seg = np.full_like(L_slice, fill_value=-2, dtype=np.int32)   # -2 = solid
    seg[m_slice] = L_slice[m_slice].astype(np.int32, copy=False) # fluid keeps its label (may be -1)


    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.25)

    # ======================
    # Left panel: labels
    # ======================
    ax0 = axes[0]

    unique_labels = np.unique(L_slice[m_slice & (L_slice >= 0)])
    n_labels = int(unique_labels.size)

    # --- 1) base image: solid in solid_color, fluid coloured by label ---
    if n_labels > 0:
        max_label = int(unique_labels.max())
        # colour table: 0..max_label
        colors_all = get_nature_palette(max_label + 1, style="soft")

        unassigned_color = "#E15759"  # a more striking colour such as "#FF00FF" also works
        
        # colour order must match the values of seg: -2, -1, 0..max_label
        cmap_colors = [solid_color, unassigned_color] + [colors_all[i] for i in range(max_label + 1)]
        cmap = ListedColormap(cmap_colors)
        
        # bin edges for the seg values [-2, -1, 0..max_label]
        # e.g. max_label=3 -> [-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5]
        bounds = [-2.5, -1.5, -0.5] + [i + 0.5 for i in range(max_label + 1)]
        norm = BoundaryNorm(bounds, cmap.N)


        ax0.imshow(
            seg,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
    else:
        # no valid label: draw the grey mask only
        ax0.imshow(
            m_slice.astype(float),
            cmap="gray",
            interpolation="nearest",
            aspect="equal",
        )

    # --- 2) overlay the boundaries (white edges) ---
    if draw_boundaries:
        boundaries = compute_voronoi_boundaries_2d(
            labels2d=L_slice.astype(np.int32, copy=False),
            mask2d=m_slice,
            connectivity=int(boundary_connectivity),
            include_solid_boundaries=bool(draw_solid_boundaries),
        )
        boundary_overlay = np.ma.masked_where(~boundaries, np.ones_like(boundaries, dtype=np.float32))
        ax0.imshow(
            boundary_overlay,
            cmap=ListedColormap(["white"]),
            alpha=0.9,
            interpolation="nearest",
        )




    ax0.set_title(tessellation_title, fontweight='medium', pad=8)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax0.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # ======================
    # Right panel: distance
    # ======================
    ax1 = axes[1]

    # --- robust distance masking: mask solid + mask NaN/Inf inside fluid ---
    D_slice_f = np.asarray(D_slice, dtype=np.float32)
    
    # solid -> NaN, then mask all invalid (NaN/Inf)
    D_vis = np.where(m_slice, D_slice_f, np.nan).astype(np.float32, copy=False)
    dist_masked = np.ma.masked_invalid(D_vis)
    
    valid_dist = dist_masked.compressed()  # only finite values
    if valid_dist.size > 0:
        vmin = float(np.nanmin(valid_dist))
        vmax = float(np.nanpercentile(valid_dist, 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax <= vmin):
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im1 = ax1.imshow(
        dist_masked,
        cmap='viridis',
        interpolation='bilinear',
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label(distance_cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax1.set_title(distance_title, fontweight='medium', pad=8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # ======================
    # Seeds overlay
    # ======================
    # ======================
    # Seeds overlay
    # ======================
    if draw_seeds and (seeds is not None):
        seeds_np = _ensure_numpy(seeds)
        if seeds_np.ndim == 2 and seeds_np.shape[1] == 3:
            if axis == "z":
                mask_seed = seeds_np[:, 0] == index
                sy = seeds_np[mask_seed, 1]
                sx = seeds_np[mask_seed, 2]
            elif axis == "y":
                mask_seed = seeds_np[:, 1] == index
                sy = seeds_np[mask_seed, 0]
                sx = seeds_np[mask_seed, 2]
            else:
                mask_seed = seeds_np[:, 2] == index
                sy = seeds_np[mask_seed, 0]
                sx = seeds_np[mask_seed, 1]

            if sx.size > 0:
                for ax in axes:
                    ax.scatter(
                        sx, sy,
                        s=80,
                        c='white',
                        marker='o',
                        edgecolors='black',
                        linewidths=1.2,
                        zorder=10,
                        path_effects=[
                            patheffects.withStroke(linewidth=2, foreground='black')
                        ],
                    )


    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"2D slice saved to: {save_path}")

    plt.show()
    return fig, axes



def visualize_m1_euclidean_clipping_slice(
    mask,
    label_euc,
    dist_euc,
    seeds=None,
    axis="y",
    index=None,
    figsize=(10, 4.5),
    save_path="m1_euclidean_clipping_slice.png",
    dpi=300,
):
    return visualize_exact_geodesic_slice(
        mask,
        label_euc,
        dist_euc,
        seeds=seeds,
        axis=axis,
        index=index,
        figsize=figsize,
        save_path=save_path,
        dpi=dpi,
        tessellation_title="Euclidean Voronoi (clipped by mask)",
        distance_title="Euclidean Distance Field",
        distance_cbar_label="Euclidean Distance",
        solid_color="#E8E8E8",
    )


# ============================================================
# 1.B Module 2: visualisation of seed stamping (3D and 2D, labels at 65% transparency)
# ============================================================

def visualize_seed_stamping_3d(
    mask,
    label_stamping,
    seeds,
    radii,
    html_path="seed_stamping_3d.html",
    smoothing_sigma=0.8,
    mesh_simplify_factor=2,
):
    """
    3D visualisation of seed stamping (module 2)

    - stamping regions coloured by label / seed, 65% transparent (opacity=0.35)
    - seed points in the colour of their stamping region; hover shows the nearest-neighbour radius
    - the hull of the pore space (from the mask) is drawn in grey, semi-transparent and lit, as a reference
    """
    try:
        import plotly.graph_objects as go
        from skimage import measure
    except ImportError:
        print("Error: Please install plotly and scikit-image:")
        print("  pip install plotly scikit-image")
        return None

    print("Generating 3D Seed Stamping visualization using Plotly...")

    mask_np = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label_stamping).astype(np.int32)
    seeds_np = _ensure_numpy(seeds) if seeds is not None else None
    radii_np = _ensure_numpy(radii) if radii is not None else None

    D, H, W = mask_np.shape
    print(f"  Data shape: {D} x {H} x {W}")

    stamping_mask = (label_np >= 0) & mask_np
    unique_labels = np.unique(label_np[stamping_mask])
    if unique_labels.size == 0:
        print("  No stamped voxels to visualize.")
        return None

    max_label = int(unique_labels.max())
    n_seeds = seeds_np.shape[0] if (seeds_np is not None and seeds_np.ndim == 2) else 0
    n_palette = max(max_label + 1, n_seeds, 1)
    colors_all = get_nature_palette(n_palette, style="soft")

    fig = go.Figure()

    # --------------------------------------------------------
    # 1) pore-space hull first: grey, semi-transparent, lit, as background reference
    # --------------------------------------------------------
    pad_width = max(3, int(np.ceil(3 * smoothing_sigma)) + 1)

    try:
        mask_padded = np.pad(mask_np, pad_width=pad_width,
                             mode="constant", constant_values=False)
        float_mask = mask_padded.astype(np.float32)
        float_mask = ndimage.gaussian_filter(float_mask, sigma=smoothing_sigma)

        verts_m, faces_m, normals_m, values_m = measure.marching_cubes(
            float_mask,
            level=0.5,
            step_size=max(mesh_simplify_factor, 2),
            allow_degenerate=False,
        )

        if verts_m.size > 0 and faces_m.size > 0:
            x_m = verts_m[:, 2] - pad_width
            y_m = verts_m[:, 1] - pad_width
            z_m = verts_m[:, 0] - pad_width

            i_m = faces_m[:, 0]
            j_m = faces_m[:, 1]
            k_m = faces_m[:, 2]

            fig.add_trace(go.Mesh3d(
                x=x_m, y=y_m, z=z_m,
                i=i_m, j=j_m, k=k_m,
                color="#B0B0B0",
                opacity=0.18,
                name="Pore space shell",
                showlegend=True,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.6,
                    specular=0.4,
                    roughness=0.5,
                    fresnel=0.25,
                ),
                lightposition=dict(x=120, y=220, z=260),
                flatshading=False,
            ))
    except Exception as e:
        print(f"  Warning: Could not create global pore-shell mesh: {e}")

    # --------------------------------------------------------
    # 2) then each stamping region: coloured by label / seed, 65% transparent
    # --------------------------------------------------------
    from scipy.ndimage import binary_dilation

    for lbl in unique_labels:
        lbl_int = int(lbl)
        region_mask = (label_np == lbl_int) & mask_np
        if not region_mask.any():
            continue

        dilated_region = binary_dilation(region_mask, iterations=1) & mask_np
        padded_region = np.pad(
            dilated_region, pad_width=pad_width,
            mode="constant", constant_values=False
        )

        float_region = padded_region.astype(np.float32)
        float_region = ndimage.gaussian_filter(float_region, sigma=smoothing_sigma)

        # skip regions too small or faint for 0.5 to be in the value range (avoids a warning)
        vmin = float_region.min()
        vmax = float_region.max()
        if not (vmin < 0.5 < vmax):
            continue

        try:
            verts, faces, normals, values = measure.marching_cubes(
                float_region,
                level=0.5,
                step_size=mesh_simplify_factor,
                allow_degenerate=False,
            )
        except Exception as e:
            print(f"  Warning: Could not create mesh for stamped region {lbl_int}: {e}")
            continue


        if verts.size == 0 or faces.size == 0:
            continue

        x = verts[:, 2] - pad_width
        y = verts[:, 1] - pad_width
        z = verts[:, 0] - pad_width

        i_faces = faces[:, 0]
        j_faces = faces[:, 1]
        k_faces = faces[:, 2]

        color_lbl = colors_all[lbl_int] if lbl_int < len(colors_all) else "#000000"

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_faces, j=j_faces, k=k_faces,
            color=color_lbl,
            opacity=0.35,  # 65% transparent
            name=f"Stamped region {lbl_int}",
            showlegend=True,
            lighting=dict(
                ambient=0.5,
                diffuse=0.6,
                specular=0.35,
                roughness=0.6,
                fresnel=0.25,
            ),
            lightposition=dict(x=150, y=240, z=280),
            flatshading=False,
        ))

    # --------------------------------------------------------
    # 3) seed points: same colour as their label; hover shows the radius
    # --------------------------------------------------------
    if seeds_np is not None and seeds_np.ndim == 2 and seeds_np.shape[1] == 3:
        n_seeds = seeds_np.shape[0]
        for k in range(n_seeds):
            z0, y0, x0 = seeds_np[k]
            color_k = colors_all[k] if k < len(colors_all) else "#000000"
            text = f"Seed {k}"
            if radii_np is not None and len(radii_np) > k:
                text += f", r={float(radii_np[k]):.2f}"
            fig.add_trace(go.Scatter3d(
                x=[x0], y=[y0], z=[z0],
                mode="markers",
                marker=dict(
                    size=7,
                    color=color_k,
                    line=dict(color="black", width=1),
                    symbol="circle",
                ),
                name=text,
                text=[text],
                hoverinfo="text",
                showlegend=(k < 10),
            ))

    fig.update_layout(
        title=dict(
            text="Seed Stamping (Module 2) with Pore Shell",
            font=dict(size=20, family="Arial"),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            bgcolor="white",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
    )

    print(f"  Saving Seed Stamping HTML to: {html_path}")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"  -> Open {html_path} in a web browser to inspect stamping vs pore shell.")

    return fig

def visualize_active_tiles_3d_boxes(
    mask,
    tiles_coords,
    tile_size=(8, 8, 8),
    html_path="active_tiles_3d.html",
    title="Active tiles (boxes)"
):
    """
    3D visualisation of the active tiles as cubes:
    - mask: 3D bool, the fluid domain
    - tiles_coords: (K, 3) numpy/cupy array, one row (tz, ty, tx) per tile
    - tile_size: (Tz, Ty, Tx)

    The tiles are drawn as opaque red cubes, showing only the outer faces,
    so that the inner triangle structure is not visible.
    """
    try:
        import plotly.graph_objects as go
        from skimage import measure
    except ImportError:
        print("Error: Please install plotly and scikit-image:")
        print("  pip install plotly scikit-image")
        return None

    mask_np = _ensure_numpy(mask).astype(bool)
    D, H, W = mask_np.shape

    fig = go.Figure()

    # 1) grey pore-space hull, drawn as before
    pad_width = 2
    mask_padded = np.pad(mask_np, pad_width=pad_width,
                         mode="constant", constant_values=False)
    float_mask = mask_padded.astype(np.float32)
    float_mask = ndimage.gaussian_filter(float_mask, sigma=0.8)

    try:
        verts_m, faces_m, normals_m, values_m = measure.marching_cubes(
            float_mask,
            level=0.5,
            step_size=2,
            allow_degenerate=False,
        )

        if verts_m.size > 0 and faces_m.size > 0:
            x_m = verts_m[:, 2] - pad_width
            y_m = verts_m[:, 1] - pad_width
            z_m = verts_m[:, 0] - pad_width

            i_m = faces_m[:, 0]
            j_m = faces_m[:, 1]
            k_m = faces_m[:, 2]

            fig.add_trace(go.Mesh3d(
                x=x_m, y=y_m, z=z_m,
                i=i_m, j=j_m, k=k_m,
                color="#B0B0B0",
                opacity=0.18,
                name="Pore space shell",
                showlegend=True,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.6,
                    specular=0.4,
                    roughness=0.5,
                    fresnel=0.25,
                ),
                lightposition=dict(x=120, y=220, z=260),
                flatshading=False,
            ))
    except Exception as e:
        print(f"  Warning: Could not create pore-shell mesh: {e}")

    # 2) active tiles drawn as small solid red cubes
    tiles_coords = _ensure_numpy(tiles_coords)
    if tiles_coords is None or tiles_coords.size == 0:
        print("No active tiles to visualize.")
    else:
        tiles_coords = np.asarray(tiles_coords, dtype=np.int32)
        Tz, Ty, Tx = tile_size

        # 8 vertices and 12 triangles per cube
        box_faces_i = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        box_faces_j = [1, 2, 2, 3, 3, 0, 0, 4, 5, 6, 6, 7]
        box_faces_k = [2, 3, 3, 0, 0, 1, 4, 5, 6, 7, 7, 4]

        for idx, (tz, ty, tx) in enumerate(tiles_coords):
            z0 = int(tz * Tz)
            y0 = int(ty * Ty)
            x0 = int(tx * Tx)
            z1 = min(z0 + Tz, D)
            y1 = min(y0 + Ty, H)
            x1 = min(x0 + Tx, W)

            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue

            xs = [x0, x1, x1, x0, x0, x1, x1, x0]
            ys = [y0, y0, y1, y1, y0, y0, y1, y1]
            zs = [z0, z0, z0, z0, z1, z1, z1, z1]

            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=box_faces_i,
                j=box_faces_j,
                k=box_faces_k,
                # all tiles as opaque red cubes
                color="#E15759",           # dark red
                opacity=1.0,               # opaque, inner faces stay hidden
                name=f"Tile {int(tz)},{int(ty)},{int(tx)}",
                showlegend=False,          # keep the legend short when there are many tiles
                flatshading=True,          # no specular highlight, so the faces look like cubes rather than triangles
                lighting=dict(
                    ambient=0.9,
                    diffuse=0.1,
                    specular=0.0,
                    roughness=1.0,
                    fresnel=0.0,
                ),
            ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial"),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            bgcolor="white",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
    )

    print(f"  Saving active tiles 3D HTML to: {html_path}")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"  -> Open {html_path} in a web browser to inspect active tiles.")
    return fig


def visualize_active_tiles_slice(
    mask,
    tiles_coords,
    tile_size=(8, 8, 8),
    axis="y",
    index=None,
    figsize=(8, 4),
    save_path=None,
    dpi=300,
    title="Active tiles (2D slice)",
):
    """
    Active tiles as rectangles on a 2D slice:
    - background: grey image of the fluid mask
    - foreground: cross-sections of the active tiles on the slice, semi-transparent red rectangles
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'mathtext.fontset': 'dejavusans',
    })

    mask_np = _ensure_numpy(mask).astype(bool)
    D, H, W = mask_np.shape

    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("axis must be 'z', 'y' or 'x'.")

    # slice index: the middle plane with the most fluid where possible
    if axis == "z":
        n_slices = D
    elif axis == "y":
        n_slices = H
    else:
        n_slices = W

    if index is None:
        if axis == "z":
            fluid_counts = mask_np.reshape(D, -1).sum(axis=1)
        elif axis == "y":
            fluid_counts = mask_np.transpose(1, 0, 2).reshape(H, -1).sum(axis=1)
        else:
            fluid_counts = mask_np.transpose(2, 0, 1).reshape(W, -1).sum(axis=1)

        non_empty = np.where(fluid_counts > 0)[0]
        if len(non_empty) == 0:
            index = n_slices // 2
        else:
            index = non_empty[len(non_empty) // 2]
    else:
        if not (0 <= index < n_slices):
            raise ValueError(f"index out of range: 0 <= index < {n_slices}")

    # background slice
    if axis == "z":
        m_slice = mask_np[index, :, :]   # H x W
        xlabel, ylabel = 'x', 'y'
    elif axis == "y":
        m_slice = mask_np[:, index, :]   # D x W
        xlabel, ylabel = 'x', 'z'
    else:  # 'x'
        m_slice = mask_np[:, :, index]   # D x H
        xlabel, ylabel = 'y', 'z'

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(
        m_slice.astype(float),
        cmap="gray",
        interpolation="nearest",
        aspect="equal",
        vmin=0.0, vmax=1.0,
    )

    tiles_coords = _ensure_numpy(tiles_coords)
    tiles_coords = np.asarray(tiles_coords, dtype=np.int32)

    if tiles_coords.size > 0:
        Tz, Ty, Tx = tile_size
        color_box = "#E15759"  # red

        for k, (tz, ty, tx) in enumerate(tiles_coords):
            z0 = int(tz * Tz)
            y0 = int(ty * Ty)
            x0 = int(tx * Tx)
            z1 = min(z0 + Tz, D)
            y1 = min(y0 + Ty, H)
            x1 = min(x0 + Tx, W)

            # cross-section of the tile on the current slice
            if axis == "z":
                if not (z0 <= index < z1):
                    continue
                y_start, y_end = y0, y1
                x_start, x_end = x0, x1
            elif axis == "y":
                if not (y0 <= index < y1):
                    continue
                y_start, y_end = z0, z1   # mapped to the vertical image axis (0..D)
                x_start, x_end = x0, x1   # mapped to the horizontal image axis (0..W)
            else:  # 'x'
                if not (x0 <= index < x1):
                    continue
                y_start, y_end = z0, z1   # vertical axis (0..D)
                x_start, x_end = y0, y1   # horizontal axis (0..H)

            rect = Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                linewidth=1.0,
                edgecolor='black',
                facecolor=color_box,
                alpha=0.4,
            )
            ax.add_patch(rect)

    ax.set_title(title, fontweight="medium", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Active tiles 2D slice saved to: {save_path}")

    plt.show()
    return fig, ax



def visualize_seed_stamping_slice(
    mask,
    label_stamping,
    roi_mask,
    seeds=None,
    axis="z",
    index=None,
    figsize=(8, 4),
    save_path=None,
    dpi=300,
):
    """
    2D slice visualisation of seed stamping (module 2)

    - left: stamping labels, 65% transparent over the fluid domain, in the colours of the seeds / labels
    - right: ROI (region not stamped), highlighted in red
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib import patheffects
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'mathtext.fontset': 'dejavusans',
    })

    mask_np = _ensure_numpy(mask).astype(bool)
    label_np = _ensure_numpy(label_stamping).astype(np.int32)

    roi_np = _ensure_numpy(roi_mask)
    if roi_np.ndim == 1:
        D, H, W = mask_np.shape
        roi_np = roi_np.reshape((D, H, W))
    else:
        roi_np = roi_np.astype(bool)

    D, H, W = mask_np.shape
    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("axis must be 'z', 'y' or 'x'.")

    seeds_np = _ensure_numpy(seeds) if seeds is not None else None
    if seeds_np is not None and seeds_np.ndim == 2 and seeds_np.shape[1] == 3:
        n_seeds = seeds_np.shape[0]
    else:
        n_seeds = 0

    # choose the slice index
    if axis == "z":
        n_slices = D
    elif axis == "y":
        n_slices = H
    else:
        n_slices = W

    if index is None:
        if axis == "z":
            fluid_counts = mask_np.reshape(D, -1).sum(axis=1)
        elif axis == "y":
            fluid_counts = mask_np.transpose(1, 0, 2).reshape(H, -1).sum(axis=1)
        else:
            fluid_counts = mask_np.transpose(2, 0, 1).reshape(W, -1).sum(axis=1)

        non_empty = np.where(fluid_counts > 0)[0]
        if len(non_empty) == 0:
            index = n_slices // 2
        else:
            index = non_empty[len(non_empty) // 2]
    else:
        if not (0 <= index < n_slices):
            raise ValueError(f"index out of range: 0 <= index < {n_slices}")

    # take the slice
    if axis == "z":
        m_slice = mask_np[index, :, :]
        L_slice = label_np[index, :, :]
        roi_slice = roi_np[index, :, :]
        xlabel, ylabel = "x", "y"
    elif axis == "y":
        m_slice = mask_np[:, index, :]
        L_slice = label_np[:, index, :]
        roi_slice = roi_np[:, index, :]
        xlabel, ylabel = "x", "z"
    else:  # 'x'
        m_slice = mask_np[:, :, index]
        L_slice = label_np[:, :, index]
        roi_slice = roi_np[:, :, index]
        xlabel, ylabel = "y", "z"

    seg = np.full_like(L_slice, fill_value=-1, dtype=np.int32)
    stamped = (m_slice & (L_slice >= 0))
    seg[stamped] = L_slice[stamped]

    unique_labels = np.unique(seg[seg >= 0])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.25)

    ax0, ax1 = axes

    # colour map: one palette for labels and seeds
    if unique_labels.size > 0:
        max_label = int(unique_labels.max())
        n_palette = max(max_label + 1, n_seeds, 1)
        colors_all = get_nature_palette(n_palette, style="soft")

        cmap_colors = ["#E8E8E8"] + [colors_all[i] for i in range(max_label + 1)]
        cmap = ListedColormap(cmap_colors)
        bounds = [-1.5] + [i - 0.5 for i in range(max_label + 2)]
        norm = BoundaryNorm(bounds, cmap.N)

        label_map = {-1: -1}
        for lbl in range(max_label + 1):
            label_map[lbl] = lbl

        seg_mapped = np.vectorize(lambda x: label_map.get(int(x), -1))(seg)

        ax0.imshow(
            seg_mapped,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
            alpha=0.35,  # 65% transparent
        )

        # stamping boundary
        valid = m_slice & (L_slice >= 0)
        boundaries = np.zeros_like(valid, dtype=bool)

        diff = valid[:, 1:] & valid[:, :-1] & (L_slice[:, 1:] != L_slice[:, :-1])
        boundaries[:, 1:] |= diff

        diff = valid[1:, :] & valid[:-1, :] & (L_slice[1:, :] != L_slice[:-1, :])
        boundaries[1:, :] |= diff


        boundary_overlay = np.ma.masked_where(~boundaries, np.ones_like(boundaries))
        ax0.imshow(
            boundary_overlay,
            cmap=ListedColormap(["white"]),
            alpha=0.9,
            interpolation="nearest",
        )
    else:
        colors_all = get_nature_palette(max(n_seeds, 1), style="soft") if n_seeds > 0 else []
        ax0.imshow(
            m_slice.astype(float),
            cmap="gray",
            interpolation="nearest",
            aspect="equal",
            alpha=0.35,
        )

    ax0.set_title("Seed Stamping (labels)", fontweight="medium", pad=8)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax0.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # right panel: ROI (region not stamped)
    ax1.imshow(
        m_slice,
        cmap="gray",
        interpolation="nearest",
        aspect="equal",
    )

    roi_display = np.zeros_like(roi_slice, dtype=float)
    roi_display[roi_slice & m_slice] = 1.0

    ax1.imshow(
        roi_display,
        cmap=ListedColormap(["none", "#E15759"]),
        alpha=0.6,
        interpolation="nearest",
    )

    ax1.set_title("ROI (un-stamped region)", fontweight="medium", pad=8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # seeds (colours as in the palette)
    if seeds_np is not None and seeds_np.ndim == 2 and seeds_np.shape[1] == 3:
        n_seeds = seeds_np.shape[0]
        if unique_labels.size > 0:
            max_label = int(unique_labels.max())
            n_palette = max(max_label + 1, n_seeds, 1)
            colors_all = get_nature_palette(n_palette, style="soft")
        else:
            colors_all = get_nature_palette(max(n_seeds, 1), style="soft")

        if axis == "z":
            mask_seed = seeds_np[:, 0] == index
            sy = seeds_np[mask_seed, 1]
            sx = seeds_np[mask_seed, 2]
        elif axis == "y":
            mask_seed = seeds_np[:, 1] == index
            sy = seeds_np[mask_seed, 0]
            sx = seeds_np[mask_seed, 2]
        else:  # 'x'
            mask_seed = seeds_np[:, 2] == index
            sy = seeds_np[mask_seed, 0]
            sx = seeds_np[mask_seed, 1]

        seed_ids = np.where(mask_seed)[0]

        if seed_ids.size > 0:
            for ax in axes:
                for sid, (px, py) in enumerate(zip(sx, sy)):
                    seed_index = seed_ids[sid]
                    color_k = colors_all[seed_index] if seed_index < len(colors_all) else "#000000"
                    ax.scatter(
                        px, py,
                        s=70,
                        c=color_k,
                        marker="o",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=10,
                        path_effects=[
                            patheffects.withStroke(
                                linewidth=2, foreground="black"
                            )
                        ],
                    )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Seed stamping 2D slice saved to: {save_path}")

    plt.show()
    return fig, axes


# ============================================================
# 2. Seed stamping: nearest-neighbour radii and pre-labelling (Section 2.3.1-2.3.2)
# ============================================================




def compute_stamping_radii(
    seeds,
    delta_r=1.0,
    metric="26",
    *,
    knn_k_init=32,
    knn_k_max=2048,
    workers=-1,
):
    """
    Stamping radius of each seed:
      r_i = 0.5 * min_{j!=i} d(seed_i, seed_j) - delta_r

    Implementation (A):
      - no (N,N) distance matrix (O(N^2) and very large GPU memory)
      - a cKDTree gives Euclidean kNN (CPU), and the 26-metric is computed exactly on the candidate set
      - correctness guarantee: k is doubled automatically until
            d_euc_max >= d26_min
        holds; since d26 >= d_euc, this guarantees that no closer 26-metric neighbour is missed,
        so the radius is never overestimated (the key safety property).

    Returns:
      radii_cp : cupy.float32 shape (N,)
    """
    import numpy as np
    import cupy as cp
    from scipy.spatial import cKDTree

    # ---- seeds -> numpy int32 on CPU (small data, safe & fast) ----
    if isinstance(seeds, cp.ndarray):
        seeds_np = cp.asnumpy(seeds).astype(np.int32, copy=False)
    else:
        seeds_np = np.asarray(seeds, dtype=np.int32)

    n = int(seeds_np.shape[0])
    if n <= 0:
        raise ValueError("seeds is empty")
    if n == 1:
        return cp.array([1e10], dtype=cp.float32)

    if seeds_np.ndim != 2 or seeds_np.shape[1] != 3:
        raise ValueError("seeds must be (N,3) with (z,y,x)")

    metric = str(metric).lower().strip()

    # ---- build KDTree once ----
    tree = cKDTree(seeds_np.astype(np.float32, copy=False))

    # helper: robust query signature across SciPy versions
    def _query(points, k):
        try:
            return tree.query(points, k=k, workers=workers)
        except TypeError:
            # older scipy
            return tree.query(points, k=k, n_jobs=workers)

    if metric == "euclidean":
        # exact NN in Euclidean: k=2 (self + nearest)
        d_euc, idx = _query(seeds_np, k=2)
        # d_euc[:,0] == 0 (self), d_euc[:,1] nearest
        d_min = d_euc[:, 1].astype(np.float32, copy=False)

        radii = 0.5 * d_min - np.float32(delta_r)
        radii = np.maximum(radii, np.float32(0.0)).astype(np.float32, copy=False)
        return cp.asarray(radii, dtype=cp.float32)

    if metric != "26":
        raise ValueError(f"Unknown metric: {metric}. Use 'euclidean' or '26'.")

    # ---- iterative kNN with correctness guarantee ----
    SQRT2 = np.float32(1.41421356237)
    SQRT3 = np.float32(1.73205080757)

    K = int(max(2, knn_k_init))
    K = min(K, n)

    while True:
        d_euc, idx = _query(seeds_np, k=K)   # (N,K)
        if K == 1:
            raise RuntimeError("kNN returned k=1; cannot compute NN for n>1.")

        # idx: (N,K) -> gather candidates
        cand = seeds_np[idx]  # (N,K,3)
        diff = np.abs(cand - seeds_np[:, None, :]).astype(np.float32, copy=False)

        dz = diff[:, :, 0]
        dy = diff[:, :, 1]
        dx = diff[:, :, 2]

        a = np.maximum(np.maximum(dx, dy), dz)
        c = np.minimum(np.minimum(dx, dy), dz)
        b = dx + dy + dz - a - c

        m1 = a - b
        m2 = b - c
        m3 = c

        d26 = m1 + SQRT2 * m2 + SQRT3 * m3  # (N,K) float32

        # self should be the first neighbor (distance 0) -> ignore it
        d26[:, 0] = np.inf

        d_min26 = d26.min(axis=1).astype(np.float32, copy=False)   # (N,)

        # correctness guarantee:
        # unseen neighbors have Euclidean distance >= d_euc[:, -1]
        # and d26 >= d_euc, so if d_euc_max >= d_min26, then d_min26 is exact
        d_euc_max = np.asarray(d_euc[:, -1], dtype=np.float32)

        ok = np.all(d_euc_max >= d_min26)
        if ok or (K >= n) or (K >= int(knn_k_max)):
            # if K hit cap, we still return (it is exact if K>=n; otherwise conservative?):
            # NOTE: if K< n and K==knn_k_max but condition not met, d_min26 could be overestimated (unsafe).
            # Therefore, if you want strict safety, set knn_k_max large enough (or leave default 2048, typically sufficient).
            if (not ok) and (K < n) and (K >= int(knn_k_max)):
                raise RuntimeError(
                    f"[compute_stamping_radii] kNN cap reached (K={K}) but guarantee not satisfied. "
                    f"Increase knn_k_max or set it >= n_seeds for strict correctness."
                )
            d_min = d_min26
            break

        # increase K and repeat
        K = min(n, K * 2)

    radii = np.float32(0.5) * d_min - np.float32(delta_r)
    radii = np.maximum(radii, np.float32(0.0)).astype(np.float32, copy=False)
    return cp.asarray(radii, dtype=cp.float32)





# ============================================================
# [REPLACE] perform_seed_stamping : bubble stamping (Path A)
# ============================================================
def perform_seed_stamping(
    mask,
    seeds,
    delta_r=1.0,
    stamping_kernel=None,   # kept for interface compatibility (this bubble scheme does not use stamping_kernel by default)
    parallel=True,
    return_state=False,
    set_seeds_kernel=None,

    # ===== controls =====
    bubble_max_iters=16,          # cap on the number of jump=1 relaxations (avoids blow-up for large radii)
    bubble_radius_cap=64.0,       # cap on the radius (avoids a radius of 1e10 when n_seeds <= 1)
    bubble_eps=1e-6,              # relax eps (keep equal to the ROI-JFA relax_eps)
    core_eps=1e-6,                # eps of the core test (usually 1e-6)
    candidate_kernel=None,
    relax_masked_kernel=None,
    filter_core_kernel=None,
    clearance_kmax27=None,
    *,
    stamping_mode=None,    # 'A_bubble_fast' | 'C_geodesic_ball' (route C: exact and as large as possible)
):
    """
    Path A stamping: geodesic bubble (truncated shortest paths) + free-space candidate selection

    Steps:
      1) radii_i from the closed-form free-space 26-metric nearest-neighbour spacing (compute_stamping_radii)
      2) candidate_mask = union{ d_free26(seed_i, v) <= r_i }  (no LOS/canonical test)
      3) K jump=1 relaxations on candidate_mask (packed, multi-source) give the true d_geo/label there (truncated at K hops)
      4) core filter: keep the voxels with dist <= radii[label]; all others go back to INF/-1 and form the ROI
      5) return (label, dist, roi_mask, radii_cp, state_core)

    Important: bubble_max_iters / bubble_radius_cap are upper limits only; they can only reduce the
    stamping coverage and never introduce an error.
    """
    import numpy as np
    import cupy as cp
    import os

    global ROI_JFA_LAST_TRADII_WALL, ROI_JFA_LAST_TCAND_WALL, ROI_JFA_LAST_TINIT_WALL
    global ROI_JFA_LAST_TBUBBLE_WALL, ROI_JFA_LAST_TFILTER_WALL, ROI_JFA_LAST_TDECODE_WALL
    global ROI_JFA_LAST_TC2_WALL, ROI_JFA_LAST_C2_COUNT, ROI_JFA_LAST_C2_LOS_USED
    global ROI_JFA_LAST_BUBBLE_ITERS, ROI_JFA_LAST_MAX_RADIUS

    stamp_profile_sync = str(os.environ.get("PVFV_ROIJFA_STAMP_PROFILE", "0")).lower().strip() in {"1", "true", "yes", "on"}

    def _phase_sync():
        if stamp_profile_sync:
            cp.cuda.Device().synchronize()

    ROI_JFA_LAST_TRADII_WALL = 0.0
    ROI_JFA_LAST_TCAND_WALL = 0.0
    ROI_JFA_LAST_TINIT_WALL = 0.0
    ROI_JFA_LAST_TBUBBLE_WALL = 0.0
    ROI_JFA_LAST_TFILTER_WALL = 0.0
    ROI_JFA_LAST_TDECODE_WALL = 0.0
    ROI_JFA_LAST_TC2_WALL = 0.0
    ROI_JFA_LAST_C2_COUNT = 0
    ROI_JFA_LAST_C2_LOS_USED = 0
    ROI_JFA_LAST_BUBBLE_ITERS = 0
    ROI_JFA_LAST_MAX_RADIUS = 0.0

    mask_cp = cp.asarray(mask, dtype=cp.uint8)
    D, H, W = map(int, mask_cp.shape)
    nvox = int(D * H * W)
    mask_flat = mask_cp.ravel()

    # seeds contiguous on GPU
    if isinstance(seeds, cp.ndarray):
        seeds_cp = seeds.astype(cp.int32, copy=False)
    else:
        seeds_cp = cp.asarray(np.asarray(seeds, dtype=np.int32), dtype=cp.int32)
    if seeds_cp.ndim != 2 or seeds_cp.shape[1] != 3:
        raise ValueError("seeds must be (N,3) with (z,y,x)")
    seeds_cp = cp.ascontiguousarray(seeds_cp)

    n_seeds = int(seeds_cp.shape[0])
    if n_seeds <= 0:
        raise ValueError("seeds is empty")

    # radii (free-space 26 NN spacing)

    # ----------------------------
    # (C) choice of stamping_mode (compatible with the earlier interface)
    #   - stamping_mode takes precedence (keyword argument)
    #   - if stamping_mode is None and stamping_kernel is a string, stamping_kernel is taken as the mode name
    #   - if neither is given, route C 'C_geodesic_ball' is the default (exact and as large as possible)
    # ----------------------------
    if stamping_mode is None:
        if isinstance(stamping_kernel, str):
            stamping_mode = stamping_kernel
        else:
            stamping_mode = "C_geodesic_ball"

    c2_requested = str(os.environ.get("PVFV_ROIJFA_C2_CORE", "0")).lower().strip() in {"1", "true", "yes", "on"}
    c2_margin = float(os.environ.get("PVFV_ROIJFA_C2_MARGIN", "1.0e-6"))

    sm = str(stamping_mode).strip().lower()
    if sm in ("c_geodesic_ball", "c-geodesic-ball", "c", "routec", "route_c", "geodesic_ball"):
        stamping_mode = "C_geodesic_ball"
        # route C: largest certified coverage under the triangle-inequality certificate, still exact
        bubble_radius_cap = None   # radius not truncated
        bubble_max_iters = None    # let K = ceil(max(radii)) take the full range
        core_eps = 0.0             # no core shrink (largest coverage)
    elif sm in ("d_c2_geodesic_ball", "d-c2-geodesic-ball", "d_c2", "d-c2", "c2", "c2_geodesic_ball"):
        stamping_mode = "D_c2_geodesic_ball"
        c2_requested = True
        bubble_radius_cap = None
        bubble_max_iters = None
        core_eps = 0.0
    elif sm in ("a_bubble_fast", "a-bubble-fast", "a", "bubble", "fast"):
        stamping_mode = "A_bubble_fast"
        # mode A: earlier default (can be faster, but coverage is limited by cap/shrink)
    elif sm in ("b_los_parallel_packed", "b-los-parallel-packed", "b", "los", "parallel"):
        # handled as mode A so that the configuration name stays valid (still a bubble stamp).
        # route B (LOS warm start) would need its own kernels.
        stamping_mode = "B_los_parallel_packed"
    else:
        raise ValueError(
            f"Unknown stamping_mode: {stamping_mode!r}. "
            "Use 'A_bubble_fast' or 'C_geodesic_ball'."
        )

    t_phase = time.time()
    radius_override_env = os.environ.get("PVFV_ROIJFA_RADIUS_OVERRIDE", "").strip()
    if radius_override_env:
        radius_value = max(0.0, float(radius_override_env))
        radii_cp = cp.full((n_seeds,), cp.float32(radius_value), dtype=cp.float32)
    else:
        radii_cp = compute_stamping_radii(seeds_cp, delta_r=delta_r, metric="26")

    # cap radii to avoid pathological huge
    if bubble_radius_cap is not None:
        radii_cp = cp.minimum(radii_cp, cp.float32(float(bubble_radius_cap)))

    # hard cap: the radius cannot exceed the axial step diameter of the voxel grid (avoids int32 overflow / huge K when n_seeds == 1 gives 1e10)
    max_steps = float((D - 1) + (H - 1) + (W - 1))
    if max_steps < 0.0:
        max_steps = 0.0
    radii_cp = cp.minimum(radii_cp, cp.float32(max_steps))
    _phase_sync()
    ROI_JFA_LAST_TRADII_WALL = float(time.time() - t_phase)

    # build kernels
    if candidate_kernel is None:
        candidate_kernel = build_seed_candidate_ball_mask_block_per_seed_kernel_3d()
    if relax_masked_kernel is None:
        relax_masked_kernel = build_relax1_masked_packed_kernel_3d()
    if filter_core_kernel is None:
        filter_core_kernel = build_filter_core_from_bubble_kernel_3d()
    if set_seeds_kernel is None:
        set_seeds_kernel = build_set_seeds_packed_kernel()

    threads = 256

    # ----------------------------
    # (A) build per-seed bbox arrays (vectorized)
    # ----------------------------
    valid = radii_cp > cp.float32(0.0)
    R = cp.ceil(radii_cp).astype(cp.int32)
    R = cp.where(valid, R, cp.int32(0))

    seed_z_cp = cp.ascontiguousarray(seeds_cp[:, 0])
    seed_y_cp = cp.ascontiguousarray(seeds_cp[:, 1])
    seed_x_cp = cp.ascontiguousarray(seeds_cp[:, 2])

    zmin_cp = cp.maximum(cp.int32(0), seed_z_cp - R)
    zmax_cp = cp.minimum(cp.int32(D - 1), seed_z_cp + R)
    ymin_cp = cp.maximum(cp.int32(0), seed_y_cp - R)
    ymax_cp = cp.minimum(cp.int32(H - 1), seed_y_cp + R)
    xmin_cp = cp.maximum(cp.int32(0), seed_x_cp - R)
    xmax_cp = cp.minimum(cp.int32(W - 1), seed_x_cp + R)

    Zn_cp = (zmax_cp - zmin_cp + 1).astype(cp.int32)
    Yn_cp = (ymax_cp - ymin_cp + 1).astype(cp.int32)
    Xn_cp = (xmax_cp - xmin_cp + 1).astype(cp.int32)

    nbox64 = Zn_cp.astype(cp.int64) * Yn_cp.astype(cp.int64) * Xn_cp.astype(cp.int64)
    nbox_cp = cp.where(valid, nbox64, cp.int64(0)).astype(cp.int32)

    # ----------------------------
    # (B) candidate mask = union of free balls
    # ----------------------------
    t_phase = time.time()
    cand_mask = cp.zeros(nvox, dtype=cp.int32)

    candidate_kernel(
        (int(n_seeds),),
        (int(threads),),
        (
            mask_flat,
            cand_mask,
            np.int32(D), np.int32(H), np.int32(W),
            zmin_cp, ymin_cp, xmin_cp,
            Yn_cp, Xn_cp, nbox_cp,
            seed_z_cp, seed_y_cp, seed_x_cp,
            radii_cp,
            np.int32(n_seeds),
        ),
    )
    _phase_sync()
    ROI_JFA_LAST_TCAND_WALL = float(time.time() - t_phase)

    # ----------------------------
    # (C) init packed state with seeds only
    # ----------------------------
    t_phase = time.time()
    INF_BITS = np.frombuffer(np.float32(1e20).tobytes(), dtype=np.uint32)[0]
    PACK_INF_NEG1 = (int(INF_BITS) << 32) | 0xFFFFFFFF
    pack_inf_neg1_u64 = np.uint64(PACK_INF_NEG1)

    state_a = cp.full(nvox, pack_inf_neg1_u64, dtype=cp.uint64)
    state_b = cp.empty_like(state_a)

    seeds_flat_cp = cp.ascontiguousarray(seeds_cp.reshape(-1))
    blocks_seed = (n_seeds + threads - 1) // threads
    set_seeds_kernel(
        (int(blocks_seed),),
        (int(threads),),
        (
            mask_flat,
            state_a,
            seeds_flat_cp,
            np.int32(n_seeds),
            np.int32(D), np.int32(H), np.int32(W),
        ),
    )
    _phase_sync()
    ROI_JFA_LAST_TINIT_WALL = float(time.time() - t_phase)

    # ----------------------------
    # (D) bubble relax on candidate mask
    #     K = ceil(max(radii)) capped by bubble_max_iters
    # ----------------------------
    max_r = float(cp.max(radii_cp).item()) if n_seeds > 0 else 0.0
    if not math.isfinite(max_r):
        # should not happen after the volume cap above; guards math.ceil(inf/nan)
        max_r = float(max_steps)
    K = int(math.ceil(max_r))
    # never need more than max_steps iterations (axis-step upper bound)
    if K > int(max_steps):
        K = int(max_steps)
    if bubble_max_iters is not None:
        K = min(K, int(bubble_max_iters))
    if K < 0:
        K = 0
    ROI_JFA_LAST_BUBBLE_ITERS = int(K)
    ROI_JFA_LAST_MAX_RADIUS = float(max_r)

    blocks_vox = (nvox + threads - 1) // threads

    t_phase = time.time()
    if K > 0:
        eps = float(bubble_eps)
        for _ in range(K):
            relax_masked_kernel(
                (int(blocks_vox),),
                (int(threads),),
                (
                    mask_flat,
                    cand_mask,
                    state_a,
                    state_b,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.float32(eps),
                ),
            )
            state_a, state_b = state_b, state_a
    _phase_sync()
    ROI_JFA_LAST_TBUBBLE_WALL = float(time.time() - t_phase)

    # ----------------------------
    # (E) filter certified core: dist <= radii[label]
    # ----------------------------
    t_phase = time.time()
    state_core = cp.empty_like(state_a)
    roi_mask_u8 = cp.empty(nvox, dtype=cp.uint8)

    filter_core_kernel(
        (int(blocks_vox),),
        (int(threads),),
        (
            mask_flat,
            cand_mask,
            state_a,
            radii_cp,
            state_core,
            roi_mask_u8,
            np.uint64(PACK_INF_NEG1),
            np.int32(n_seeds),
            np.int32(D), np.int32(H), np.int32(W),
            np.float32(float(core_eps)),
        ),
    )
    _phase_sync()
    ROI_JFA_LAST_TFILTER_WALL = float(time.time() - t_phase)

    # ensure seeds are present (dist=0,label=k)
    t_phase = time.time()
    set_seeds_kernel(
        (int(blocks_seed),),
        (int(threads),),
        (
            mask_flat,
            state_core,
            seeds_flat_cp,
            np.int32(n_seeds),
            np.int32(D), np.int32(H), np.int32(W),
        ),
    )
    _phase_sync()

    if c2_requested:
        t_phase_c2 = time.time()
        c2_los_requested = str(os.environ.get("PVFV_ROIJFA_C2_LOS", "0")).lower().strip() in {"1", "true", "yes", "on"}
        c2_use_los = bool(c2_los_requested and clearance_kmax27 is not None)
        c2_kernel = (
            build_c2_second_competitor_core_los_kernel_3d()
            if c2_use_los else
            build_c2_second_competitor_core_kernel_3d()
        )
        c2_count_cp = cp.zeros(1, dtype=cp.int32)
        if c2_use_los:
            los_flat = cp.ascontiguousarray(clearance_kmax27.reshape(-1))
            c2_kernel(
                (int(blocks_vox),),
                (int(threads),),
                (
                    mask_flat,
                    state_core,
                    seeds_flat_cp,
                    los_flat,
                    c2_count_cp,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.int32(n_seeds),
                    np.float32(float(c2_margin)),
                    np.int32(1),
                ),
            )
            ROI_JFA_LAST_C2_LOS_USED = 1
        else:
            c2_kernel(
                (int(blocks_vox),),
                (int(threads),),
                (
                    mask_flat,
                    state_core,
                    seeds_flat_cp,
                    c2_count_cp,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.int32(n_seeds),
                    np.float32(float(c2_margin)),
                    np.int32(1),
                ),
            )
        cp.cuda.Device().synchronize()
        ROI_JFA_LAST_TC2_WALL = float(time.time() - t_phase_c2)
        ROI_JFA_LAST_C2_COUNT = int(c2_count_cp.get()[0])

    # ----------------------------
    # (F) decode label/dist and build final roi_mask (same convention as your pipeline)
    # ----------------------------
    t_phase = time.time()
    label = (state_core & cp.uint64(0xFFFFFFFF)).astype(cp.int32)
    dist_u32 = (state_core >> cp.uint64(32)).astype(cp.uint32)
    dist = dist_u32.view(cp.float32)

    roi_mask = ((label == -1) & (mask_flat == 1)).astype(cp.uint8)

    cp.cuda.Device().synchronize()
    ROI_JFA_LAST_TDECODE_WALL = float(time.time() - t_phase)

    if return_state:
        return label, dist, roi_mask, radii_cp, state_core
    return label, dist, roi_mask, radii_cp






# ============================================================
# 3. Tile management: static tiles and active tiles (Section 2.3.3, 2.4.2)
# ============================================================

def build_tiles_dual_3d_kernel():
    import cupy as cp
    code = r'''
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

    extern "C" __global__
    void build_tiles_dual_3d(
        const unsigned char* __restrict__ roi,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int tilesD, const int tilesH, const int tilesW,
        int3* __restrict__ tiles_mixed,
        int* __restrict__ cnt_mixed,
        int3* __restrict__ tiles_dense,
        int* __restrict__ cnt_dense
    )
    {
        int tbx = blockIdx.x;
        int tby = blockIdx.y;
        int tbz = blockIdx.z;

        if (tbz >= tilesD || tby >= tilesH || tbx >= tilesW) return;

        int bz = tbz * Tz;
        int by = tby * Ty;
        int bx = tbx * Tx;

        __shared__ int s_any;
        __shared__ int s_zero;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            s_any  = 0;
            s_zero = 0;
        }
        __syncthreads();

        int any_local  = 0;
        int zero_local = 0;

        const int HW = H * W;

        for (int z = bz + threadIdx.z; z < bz + Tz && z < D; z += blockDim.z) {
            long long baseZ = (long long)z * HW;
            for (int y = by + threadIdx.y; y < by + Ty && y < H; y += blockDim.y) {
                long long base = baseZ + (long long)y * W;
                for (int x = bx + threadIdx.x; x < bx + Tx && x < W; x += blockDim.x) {
                    unsigned char v = roi[base + x];
                    if (v != 0) any_local  = 1;
                    if (v == 0) zero_local = 1;
                }
            }
        }

        if (any_local)  atomicOr(&s_any,  1);
        if (zero_local) atomicOr(&s_zero, 1);
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            if (!s_any) return;  // this tile has no ROI==1 voxel

            if (!s_zero) {
                // all voxels of the tile are ROI==1 -> dense tile
                int idx = atomicAdd(cnt_dense, 1);
                tiles_dense[idx] = make_int3(tbz, tby, tbx);
            } else {
                // both ROI==1 and ROI==0 -> mixed tile
                int idx = atomicAdd(cnt_mixed, 1);
                tiles_mixed[idx] = make_int3(tbz, tby, tbx);
            }
        }
    }
    '''
    return _device_cached_rawkernel(
        build_tiles_dual_3d_kernel,
        code,
        "build_tiles_dual_3d",
    )


def build_roi_tiles_3d(roi_mask_flat, D, H, W, Tz, Ty, Tx, kernel=None):
    """
    Build the 3D dual-list tiles from the voxel-level ROI (mask_flat==1):
      - tiles_mixed: [M,3] (tz, ty, tx)
      - tiles_dense: [Dn,3] (tz, ty, tx)
    """
    if kernel is None:
        kernel = build_tiles_dual_3d_kernel()

    roi_u8 = roi_mask_flat.astype(cp.uint8)
    tilesD = (D + Tz - 1) // Tz
    tilesH = (H + Ty - 1) // Ty
    tilesW = (W + Tx - 1) // Tx
    max_tiles = tilesD * tilesH * tilesW

    tiles_mixed = cp.zeros((max_tiles, 3), dtype=cp.int32)
    tiles_dense = cp.zeros((max_tiles, 3), dtype=cp.int32)
    cnt_mixed   = cp.zeros(1, dtype=cp.int32)
    cnt_dense   = cp.zeros(1, dtype=cp.int32)

    block = (4, 4, 4)
    grid  = (tilesW, tilesH, tilesD)

    kernel(
        grid,
        block,
        (
            roi_u8,
            int(D), int(H), int(W),
            int(Tz), int(Ty), int(Tx),
            int(tilesD), int(tilesH), int(tilesW),
            tiles_mixed, cnt_mixed,
            tiles_dense, cnt_dense,
        ),
    )
    cp.cuda.Device().synchronize()

    M  = int(cnt_mixed.get()[0])
    Dn = int(cnt_dense.get()[0])

    tiles_mixed = tiles_mixed[:M]
    tiles_dense = tiles_dense[:Dn]

    return tiles_mixed, tiles_dense


# ============================================================
# 4. ROI-JFA Kernels (Section 2.4.1)
# ============================================================

def build_roi_oajfa_kernel():
    import cupy as cp
    code = r'''
    extern "C" __global__
    void roi_oajfa_step(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ tile_active,
        const int* __restrict__ label_in,
        const float* __restrict__ dist_in,
        int* __restrict__ label_out,
        float* __restrict__ dist_out,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,
        const int jump
    )
    {
        const int nvox = D * H * W;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nvox) return;

        if (!mask[idx]) {
            // solid: copy through
            label_out[idx] = label_in[idx];
            dist_out[idx]  = dist_in[idx];
            return;
        }

        if (!roi_mask[idx]) {
            // not in ROI: keep the old value
            label_out[idx] = label_in[idx];
            dist_out[idx]  = dist_in[idx];
            return;
        }

        // voxel coordinates
        const int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        // index of the corresponding tile
        int tz = z / Tz;
        int ty = y / Ty;
        int tx = x / Tx;
        int tile_id = (tz * nTilesY + ty) * nTilesX + tx;

        if (tile_active[tile_id] == 0) {
            // inactive tile: no update
            label_out[idx] = label_in[idx];
            dist_out[idx]  = dist_in[idx];
            return;
        }

        int cur_label = label_in[idx];
        float cur_dist = dist_in[idx];

        label_out[idx] = cur_label;
        dist_out[idx]  = cur_dist;

        float best_dist = cur_dist;
        int best_label = cur_label;

        // 3x3x3 neighbourhood, step = jump
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + jump * dz;
                    int jy = y + jump * dy;
                    int jx = x + jump * dx;

                    if (jz < 0 || jz >= D ||
                        jy < 0 || jy >= H ||
                        jx < 0 || jx >= W) {
                        continue;
                    }

                    int j_idx = jz * HW + jy * W + jx;
                    if (!mask[j_idx]) continue;

                    int neigh_label = label_in[j_idx];
                    if (neigh_label < 0) continue;

                    int dz_tot = jz - z;
                    int dy_tot = jy - y;
                    int dx_tot = jx - x;

                    float step = sqrtf((float)(dz_tot * dz_tot +
                                               dy_tot * dy_tot +
                                               dx_tot * dx_tot));

                    float cand_dist = dist_in[j_idx] + step;

                    if (cand_dist < best_dist) {
                        best_dist = cand_dist;
                        best_label = neigh_label;
                    }
                }
            }
        }

        if (best_label != cur_label || best_dist < cur_dist) {
            label_out[idx] = best_label;
            dist_out[idx]  = best_dist;
        }
    }
    ''';
    return _device_cached_rawkernel(
        build_roi_oajfa_kernel,
        code,
        "roi_oajfa_step",
    )

def build_mark_roi_tiles_kernel_3d():
    import cupy as cp
    code = r'''
    extern "C" __global__
    void mark_roi_tiles_3d(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        int* __restrict__ tile_roi,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX
    )
    {
        int nvox = D * H * W;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nvox) return;

        if (!mask[idx]) return;
        if (!roi_mask[idx]) return;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int tz = z / Tz;
        int ty = y / Ty;
        int tx = x / Tx;

        int tile_id = (tz * nTilesY + ty) * nTilesX + tx;

        atomicOr(&tile_roi[tile_id], 1);
    }
    ''';
    return _device_cached_rawkernel(
        build_mark_roi_tiles_kernel_3d,
        code,
        "mark_roi_tiles_3d",
    )

def build_mark_unfinished_tiles_kernel_3d():
    import cupy as cp
    code = r'''
    extern "C" __global__
    void mark_unfinished_tiles_3d(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ label,
        int* __restrict__ tile_unfinished,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX
    )
    {
        int nvox = D * H * W;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nvox) return;

        if (!mask[idx]) return;
        if (!roi_mask[idx]) return;

        if (label[idx] >= 0) return;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int tz = z / Tz;
        int ty = y / Ty;
        int tx = x / Tx;

        int tile_id = (tz * nTilesY + ty) * nTilesX + tx;
        atomicOr(&tile_unfinished[tile_id], 1);
    }
    ''';
    return _device_cached_rawkernel(
        build_mark_unfinished_tiles_kernel_3d,
        code,
        "mark_unfinished_tiles_3d",
    )

def build_mark_changed_tiles_kernel_3d(mode="frontier"):
    import cupy as cp

    """
    Mark tile_changed, for building the active tiles of the next round.

    mode:
      - "frontier": track only the advance of the wavefront (old_label<0 && new_label>=0)
                    This is the key fix that keeps the number of active tiles from exploding.
      - "frontier+competitive": compatible with the old logic (frontier + meaningful label competition)
                                Not recommended as a default: it makes the active tiles spread.
    """
    mode = str(mode).lower().strip()

    if mode == "frontier":
        change_logic = r'''
        bool changed = false;

        // track only the wavefront advance: label<0 before, taken over in this round
        if (old_label < 0 && new_label >= 0) {
            changed = true;
        } else {
            return;
        }
        '''
    elif mode in ("frontier+competitive", "frontier_competitive", "competitive"):
        change_logic = r'''
        bool changed = false;

        // 1) frontier advance
        if (old_label < 0 && new_label >= 0) {
            changed = true;
        }
        // 2) label competition: count as changed only if the distance really decreases (by more than eps)
        else if (old_label >= 0 && new_label >= 0 && old_label != new_label) {
            const float diff = old_dist - new_dist;
            if (diff > eps) {
                changed = true;
            }
        }

        if (!changed) return;
        '''
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'frontier' or 'frontier+competitive'.")

    code = r'''
    extern "C" __global__
    void mark_changed_tiles_3d(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        const int*   __restrict__ label_old,
        const float* __restrict__ dist_old,
        const int*   __restrict__ label_new,
        const float* __restrict__ dist_new,
        int* __restrict__ tile_changed,
        const float eps,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX
    )
    {
        const int nvox = D * H * W;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nvox) return;

        // observe changes only inside "fluid + ROI"
        if (!mask[idx])     return;
        if (!roi_mask[idx]) return;

        const int   old_label = label_old[idx];
        const int   new_label = label_new[idx];
        const float old_dist  = dist_old[idx];
        const float new_dist  = dist_new[idx];

    ''' + change_logic + r'''

        const int HW = H * W;
        const int z  = idx / HW;
        const int rem= idx - z * HW;
        const int y  = rem / W;
        const int x  = rem - y * W;

        const int tz = z / Tz;
        const int ty = y / Ty;
        const int tx = x / Tx;

        const int tile_id = (tz * nTilesY + ty) * nTilesX + tx;
        atomicOr(&tile_changed[tile_id], 1);
    }
    ''';
    return _device_cached_rawkernel(
        build_mark_changed_tiles_kernel_3d,
        code,
        "mark_changed_tiles_3d",
        cache_key=mode,   # different modes need different cache keys
    )

def build_seed_stamping_los_parallel_packed_kernel():
    import cupy as cp
    """
    Parallel seed stamping (all seeds in one launch) + atomic update of a 64-bit packed state (dist|label).

    [Variant 2: consistent distance contract]
    - No longer uses a 3D Bresenham "line-of-sight check"
    - Instead checks the canonical path that is consistent with the closed-form shortest path of the 26-neighbourhood:
        sort |dx|,|dy|,|dz| so that a>=b>=c
        canonical step sequence:
          m3=c three-axis diagonal steps
          m2=b-c two-axis diagonal steps (largest axis + middle axis)
          m1=a-b single-axis steps (largest axis)
      Each step checks only that the landing voxel has mask==1, consistent with the solver's adjacency definition (corner crossing allowed).

    Only when the whole canonical path is walkable is the closed-form distance d written, and the stamped region can then be frozen safely.
    """
    import cupy as cp
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void seed_stamping_los_parallel_packed(
        const unsigned char* __restrict__ mask,   // [nvox]
        unsigned long long*  __restrict__ state,  // [nvox] packed (dist|label)
        const int D, const int H, const int W,

        // per-seed bbox params
        const int* __restrict__ zmin_arr,
        const int* __restrict__ ymin_arr,
        const int* __restrict__ xmin_arr,
        const int* __restrict__ Yn_arr,
        const int* __restrict__ Xn_arr,
        const int* __restrict__ nbox_arr,

        // per-seed coords + radius
        const int*   __restrict__ seed_z_arr,
        const int*   __restrict__ seed_y_arr,
        const int*   __restrict__ seed_x_arr,
        const float* __restrict__ radius_arr,

        const int n_seeds
    )
    {
        int seed_id = (int)blockIdx.y;
        if (seed_id >= n_seeds) return;

        int nbox = nbox_arr[seed_id];
        int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (p >= nbox) return;

        int zmin = zmin_arr[seed_id];
        int ymin = ymin_arr[seed_id];
        int xmin = xmin_arr[seed_id];

        int Yn = Yn_arr[seed_id];
        int Xn = Xn_arr[seed_id];

        // decode p -> (lz, ly, lx) in bbox
        int lz = p / (Yn * Xn);
        int rem1 = p - lz * (Yn * Xn);
        int ly = rem1 / Xn;
        int lx = rem1 - ly * Xn;

        int z = zmin + lz;
        int y = ymin + ly;
        int x = xmin + lx;

        int HW = H * W;
        int idx = z * HW + y * W + x;

        // must be fluid
        if (!mask[idx]) return;

        int sz0 = seed_z_arr[seed_id];
        int sy0 = seed_y_arr[seed_id];
        int sx0 = seed_x_arr[seed_id];

        float radius = radius_arr[seed_id];

        // ------------------------------------------------------------
        // 1) compute |dz|,|dy|,|dx| and sort them with axis ids (descending, tie-break: x>y>z)
        // ------------------------------------------------------------
        int dz_i = z - sz0;
        int dy_i = y - sy0;
        int dx_i = x - sx0;

        int az = dz_i; if (az < 0) az = -az;
        int ay = dy_i; if (ay < 0) ay = -ay;
        int ax = dx_i; if (ax < 0) ax = -ax;

        // A,B,C store (value, axis_id), axis_id: z=0, y=1, x=2
        int vA = az, aA = 0;
        int vB = ay, aB = 1;
        int vC = ax, aC = 2;

        // sort descending by (value, axis_id)
        if (vA < vB || (vA == vB && aA < aB)) { int tv=vA; vA=vB; vB=tv; int ta=aA; aA=aB; aB=ta; }
        if (vA < vC || (vA == vC && aA < aC)) { int tv=vA; vA=vC; vC=tv; int ta=aA; aA=aC; aC=ta; }
        if (vB < vC || (vB == vC && aB < aC)) { int tv=vB; vB=vC; vC=tv; int ta=aB; aB=aC; aC=ta; }

        int major_axis = aA;
        int mid_axis   = aB;
        int minor_axis = aC;

        int a = vA;   // largest
        int b = vB;   // middle
        int c = vC;   // smallest

        int m3 = c;
        int m2 = b - c;
        int m1 = a - b;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        float d = (float)m1 + SQRT2 * (float)m2 + SQRT3 * (float)m3;

        // stamping ball check
        if (d > radius) return;

        // ------------------------------------------------------------
        // 2) canonical shortest-path check (metric-consistent canonical path)
        //    each step checks only that the landing voxel has mask==1 (consistent with the solver's 26-neighbourhood)
        // ------------------------------------------------------------
        int sgn_z = (dz_i >= 0) ? 1 : -1;
        int sgn_y = (dy_i >= 0) ? 1 : -1;
        int sgn_x = (dx_i >= 0) ? 1 : -1;

        int cz = sz0;
        int cy = sy0;
        int cx = sx0;

        // stage-2 increments (major + mid)
        int inc2_z = ((major_axis == 0) || (mid_axis == 0)) ? sgn_z : 0;
        int inc2_y = ((major_axis == 1) || (mid_axis == 1)) ? sgn_y : 0;
        int inc2_x = ((major_axis == 2) || (mid_axis == 2)) ? sgn_x : 0;

        // stage-1 increments (major only)
        int inc1_z = (major_axis == 0) ? sgn_z : 0;
        int inc1_y = (major_axis == 1) ? sgn_y : 0;
        int inc1_x = (major_axis == 2) ? sgn_x : 0;

        bool clear = true;

        // m3: 3-axis diagonal steps
        for (int i = 0; i < m3; ++i) {
            cz += sgn_z; cy += sgn_y; cx += sgn_x;

            if ((unsigned)cz >= (unsigned)D ||
                (unsigned)cy >= (unsigned)H ||
                (unsigned)cx >= (unsigned)W) { clear = false; break; }

            int j = cz * HW + cy * W + cx;
            if (!mask[j]) { clear = false; break; }
        }

        // m2: 2-axis diagonal steps (major + mid)
        if (clear) {
            for (int i = 0; i < m2; ++i) {
                cz += inc2_z; cy += inc2_y; cx += inc2_x;

                if ((unsigned)cz >= (unsigned)D ||
                    (unsigned)cy >= (unsigned)H ||
                    (unsigned)cx >= (unsigned)W) { clear = false; break; }

                int j = cz * HW + cy * W + cx;
                if (!mask[j]) { clear = false; break; }
            }
        }

        // m1: 1-axis steps (major)
        if (clear) {
            for (int i = 0; i < m1; ++i) {
                cz += inc1_z; cy += inc1_y; cx += inc1_x;

                if ((unsigned)cz >= (unsigned)D ||
                    (unsigned)cy >= (unsigned)H ||
                    (unsigned)cx >= (unsigned)W) { clear = false; break; }

                int j = cz * HW + cy * W + cx;
                if (!mask[j]) { clear = false; break; }
            }
        }

        if (!clear) return;

        // ------------------------------------------------------------
        // 3) packed atomic update: take the minimum in lexicographic order of (dist_bits, label)
        // ------------------------------------------------------------
        unsigned int dist_u = __float_as_uint(d);
        unsigned long long new_pack =
            ((unsigned long long)dist_u << 32) | (unsigned long long)(unsigned int)seed_id;

        unsigned long long old = state[idx];
        while (new_pack < old) {
            unsigned long long assumed = old;
            old = atomicCAS((unsigned long long*)&state[idx], assumed, new_pack);
            if (old == assumed) break;
        }
    }
    '''
    return _device_cached_rawkernel(
        build_seed_stamping_los_parallel_packed_kernel,
        code,
        "seed_stamping_los_parallel_packed",
    )



def build_set_seeds_packed_kernel():
    """Force dist=0, label=seed_id at every seed position (packed state)."""
    import cupy as cp
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void set_seeds_packed(
        const unsigned char* __restrict__ mask,   // [nvox]
        unsigned long long*  __restrict__ state,  // [nvox]
        const int* __restrict__ seeds,            // [3*n_seeds]
        const int n_seeds,
        const int D, const int H, const int W
    )
    {
        int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (k >= n_seeds) return;

        int z = seeds[3*k + 0];
        int y = seeds[3*k + 1];
        int x = seeds[3*k + 2];

        if ((unsigned)z >= (unsigned)D ||
            (unsigned)y >= (unsigned)H ||
            (unsigned)x >= (unsigned)W) return;

        int HW = H * W;
        int idx = z * HW + y * W + x;
        if (!mask[idx]) return;

        unsigned int dist_u = __float_as_uint(0.0f);
        unsigned long long pack =
            ((unsigned long long)dist_u << 32) | (unsigned long long)(unsigned int)k;

        state[idx] = pack;
    }
    '''
    return _device_cached_rawkernel(
        build_set_seeds_packed_kernel,
        code,
        "set_seeds_packed",
    )


def build_roi_relax_jump1_packed_kernel_3d():
    """
    jump=1 special case: ROI relaxation (never reads los_kmax27)
    - updates only ROI voxels: mask==1 && roi_mask==1
    - all other voxels are copied
    """
    import cupy as cp
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void geodesic_roi_relax1_packed(
        const unsigned char* __restrict__ mask,      // [nvox]
        const unsigned char* __restrict__ roi_mask,  // [nvox]
        const unsigned long long* __restrict__ state_in,
        unsigned long long* __restrict__ state_out,
        const int D, const int H, const int W,
        const float eps
    )
    {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx] || !roi_mask[idx]) {
            state_out[idx] = state_in[idx];
            return;
        }

        unsigned long long cur = state_in[idx];
        int   cur_label = (int)(cur & 0xFFFFFFFFu);
        float cur_dist  = __uint_as_float((unsigned int)(cur >> 32));

        int   best_label = cur_label;
        float best_dist  = cur_dist;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + dz;
                    int jy = y + dy;
                    int jx = x + dx;

                    if ((unsigned)jz >= (unsigned)D ||
                        (unsigned)jy >= (unsigned)H ||
                        (unsigned)jx >= (unsigned)W) continue;

                    int j_idx = idx + dz * HW + dy * W + dx;
                    if (!mask[j_idx]) continue;

                    unsigned long long nb = state_in[j_idx];
                    int nb_label = (int)(nb & 0xFFFFFFFFu);
                    if (nb_label < 0) continue;

                    float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                    int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                    float step = (nnz == 1) ? 1.0f
                               : (nnz == 2) ? SQRT2
                                            : SQRT3;

                    float cand = nb_dist + step;
                    if (cand + eps < best_dist) {
                        best_dist  = cand;
                        best_label = nb_label;
                    }
                }
            }
        }

        unsigned int out_du = __float_as_uint(best_dist);
        unsigned long long out_pack =
            ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

        state_out[idx] = out_pack;
    }
    '''
    return _device_cached_rawkernel(
        build_roi_relax_jump1_packed_kernel_3d,
        code,
        "geodesic_roi_relax1_packed",
    )

def build_roi_closure_jump1_packed_kernel_3d():
    """
    [NEW] Closure C (jump=1) for ROI, packed-state version.

    One iteration does:
      - For ROI voxel with label<0: try to "take over" from any labeled 26-neigh neighbor (jump=1).
      - For ROI voxel with label>=0: copy (do NOT refine here; refinement stays in n_relax_after stage).
      - For non-ROI or solid: copy.

    Additionally writes two device flags:
      - any_changed[0]   = 1 if any voxel changed from label<0 -> label>=0 in this iteration
      - any_unfinished[0]= 1 if any ROI voxel remains label<0 after this iteration

    In Python, loop until any_unfinished==0.
    """
    import cupy as cp

    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void geodesic_roi_closure1_packed(
        const unsigned char* __restrict__ mask,      // [nvox]
        const unsigned char* __restrict__ roi_mask,  // [nvox]
        const unsigned long long* __restrict__ state_in,
        unsigned long long* __restrict__ state_out,
        int* __restrict__ any_changed,     // [1]
        int* __restrict__ any_unfinished,  // [1]
        const int D, const int H, const int W,
        const float eps
    )
    {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx] || !roi_mask[idx]) {
            state_out[idx] = state_in[idx];
            return;
        }

        unsigned long long cur = state_in[idx];
        int   cur_label = (int)(cur & 0xFFFFFFFFu);
        float cur_dist  = __uint_as_float((unsigned int)(cur >> 32));

        // already assigned in ROI -> keep (closure only fills U)
        if (cur_label >= 0) {
            state_out[idx] = cur;
            return;
        }

        // unassigned ROI voxel -> try to absorb from labeled neighbors
        int best_label = cur_label;
        float best_dist = cur_dist;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + dz;
                    int jy = y + dy;
                    int jx = x + dx;

                    if ((unsigned)jz >= (unsigned)D ||
                        (unsigned)jy >= (unsigned)H ||
                        (unsigned)jx >= (unsigned)W) continue;

                    int j_idx = idx + dz * HW + dy * W + dx;
                    if (!mask[j_idx]) continue;

                    unsigned long long nb = state_in[j_idx];
                    int nb_label = (int)(nb & 0xFFFFFFFFu);
                    if (nb_label < 0) continue;

                    float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                    int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                    float step = (nnz == 1) ? 1.0f
                               : (nnz == 2) ? SQRT2
                                            : SQRT3;

                    float cand = nb_dist + step;
                    if (cand + eps < best_dist) {
                        best_dist  = cand;
                        best_label = nb_label;
                    }
                }
            }
        }

        unsigned int out_du = __float_as_uint(best_dist);
        unsigned long long out_pack =
            ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

        state_out[idx] = out_pack;

        if (best_label >= 0) {
            atomicExch(&any_changed[0], 1);
        } else {
            atomicExch(&any_unfinished[0], 1);
        }
    }
    ''';

    code = code.encode("ascii", "ignore").decode("ascii")
    return _device_cached_rawkernel(
        build_roi_closure_jump1_packed_kernel_3d,
        code,
        "geodesic_roi_closure1_packed",
    )


def build_local_relax_packed_kernel_3d():
    """
    Local relaxation (jump=1) after ROI-JFA, packed-state version:
    - domain: mask==1 (whole fluid domain)
    - solid voxels: copied
    """
    import cupy as cp
    code = r'''
    #include <cuda_runtime.h>

    extern "C" __global__
    void geodesic_local_relax_packed(
        const unsigned char* __restrict__ mask,      // [nvox]
        const unsigned long long* __restrict__ state_in,
        unsigned long long* __restrict__ state_out,
        const int D, const int H, const int W,
        const float eps
    )
    {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx]) {
            state_out[idx] = state_in[idx];
            return;
        }

        unsigned long long cur = state_in[idx];
        int   cur_label = (int)(cur & 0xFFFFFFFFu);
        float cur_dist  = __uint_as_float((unsigned int)(cur >> 32));

        int   best_label = cur_label;
        float best_dist  = cur_dist;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + dz;
                    int jy = y + dy;
                    int jx = x + dx;

                    if ((unsigned)jz >= (unsigned)D ||
                        (unsigned)jy >= (unsigned)H ||
                        (unsigned)jx >= (unsigned)W) continue;

                    int j_idx = idx + dz * HW + dy * W + dx;
                    if (!mask[j_idx]) continue;

                    unsigned long long nb = state_in[j_idx];
                    int nb_label = (int)(nb & 0xFFFFFFFFu);
                    if (nb_label < 0) continue;

                    float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                    int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                    float step = (nnz == 1) ? 1.0f
                               : (nnz == 2) ? SQRT2
                                            : SQRT3;

                    float cand = nb_dist + step;
                    if (cand + eps < best_dist) {
                        best_dist  = cand;
                        best_label = nb_label;
                    }
                }
            }
        }

        unsigned int out_du = __float_as_uint(best_dist);
        unsigned long long out_pack =
            ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

        state_out[idx] = out_pack;
    }
    '''
    return _device_cached_rawkernel(
        build_local_relax_packed_kernel_3d,
        code,
        "geodesic_local_relax_packed",
    )



def build_active_tiles_kernel_3d():
    """
    Pull-based activation (instead of push/atomicOr spreading):

    for each tile tid (with tile_roi[tid]==1):
      - if tile_unfinished_prev[tid] or tile_changed_prev[tid] is set: tid is active
      - otherwise: check which changed tiles can affect the next jump update of tid:
              shift the voxel bbox of tid by the jump in each of the 26 directions to get the range of
              possible source tiles; if any of them has tile_changed_prev==1, tid is active

    Properties:
      - no writes to neighbours (no atomicOr on the 27 neighbouring tiles)
      - tile_active_out[tid] is written once
      - any_active still uses atomicExch (low contention)
    """
    import cupy as cp

    code = r'''
    extern "C" __global__
    void mark_active_tiles_jump_hit_3d_flags(
        const int* __restrict__ tile_roi,             // [nTiles] (0/1)
        const int* __restrict__ tile_changed_prev,    // [nTiles] (0/1)
        const int* __restrict__ tile_unfinished_prev, // [nTiles] (0/1)
        int* __restrict__ tile_active_out,            // [nTiles] (0/1) (host must zero)
        int* __restrict__ any_active,                 // [1]     (host must zero)
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,
        const int jump
    )
    {
        int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nTiles = nTilesZ * nTilesY * nTilesX;
        if (tid >= nTiles) return;

        if (tile_roi[tid] == 0) return;

        // 1) if I am unfinished / changed -> I am active
        if (tile_unfinished_prev[tid] != 0 || tile_changed_prev[tid] != 0) {
            tile_active_out[tid] = 1;
            atomicExch(&any_active[0], 1);
            return;
        }

        // 2) otherwise: pull-scan "which changed tiles affect me"
        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int z0 = tz * Tz;
        int y0 = ty * Ty;
        int x0 = tx * Tx;

        int z1 = z0 + (Tz - 1);
        int y1 = y0 + (Ty - 1);
        int x1 = x0 + (Tx - 1);

        if (z1 >= D) z1 = D - 1;
        if (y1 >= H) y1 = H - 1;
        if (x1 >= W) x1 = W - 1;

        int active = 0;

        #pragma unroll
        for (int dz = -1; dz <= 1; ++dz) {
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int sz = dz * jump;
                    int sy = dy * jump;
                    int sx = dx * jump;

                    int zz0 = z0 + sz;
                    int yy0 = y0 + sy;
                    int xx0 = x0 + sx;

                    int zz1 = z1 + sz;
                    int yy1 = y1 + sy;
                    int xx1 = x1 + sx;

                    // reject if fully outside
                    if (zz1 < 0 || zz0 >= D) continue;
                    if (yy1 < 0 || yy0 >= H) continue;
                    if (xx1 < 0 || xx0 >= W) continue;

                    // clamp
                    if (zz0 < 0) zz0 = 0;
                    if (yy0 < 0) yy0 = 0;
                    if (xx0 < 0) xx0 = 0;
                    if (zz1 >= D) zz1 = D - 1;
                    if (yy1 >= H) yy1 = H - 1;
                    if (xx1 >= W) xx1 = W - 1;

                    int tz_min = zz0 / Tz;
                    int ty_min = yy0 / Ty;
                    int tx_min = xx0 / Tx;

                    int tz_max = zz1 / Tz;
                    int ty_max = yy1 / Ty;
                    int tx_max = xx1 / Tx;

                    if (tz_min < 0) tz_min = 0;
                    if (ty_min < 0) ty_min = 0;
                    if (tx_min < 0) tx_min = 0;
                    if (tz_max >= nTilesZ) tz_max = nTilesZ - 1;
                    if (ty_max >= nTilesY) ty_max = nTilesY - 1;
                    if (tx_max >= nTilesX) tx_max = nTilesX - 1;

                    for (int tz2 = tz_min; tz2 <= tz_max && !active; ++tz2) {
                        for (int ty2 = ty_min; ty2 <= ty_max && !active; ++ty2) {
                            int base = (tz2 * nTilesY + ty2) * nTilesX;
                            for (int tx2 = tx_min; tx2 <= tx_max; ++tx2) {
                                int tid2 = base + tx2;
                                if (tile_roi[tid2] == 0) continue;
                                if (tile_changed_prev[tid2] != 0) { active = 1; break; }
                            }
                        }
                    }

                    if (active) break;
                }
                if (active) break;
            }
            if (active) break;
        }

        if (active) {
            tile_active_out[tid] = 1;
            atomicExch(&any_active[0], 1);
        }
    }
    '''
    return _device_cached_rawkernel(
        build_active_tiles_kernel_3d,
        code,
        "mark_active_tiles_jump_hit_3d_flags",
        options=("-std=c++11",),
    )



def build_fallback_full_roi_kernel_3d():
    """
    If any_active[0]==0, set tile_active to tile_roi (full-ROI fallback).
    """
    code = r'''
    extern "C" __global__
    void fallback_full_roi_if_empty(
        const int* __restrict__ tile_roi,     // [nTiles]
        int* __restrict__ tile_active,        // [nTiles]
        const int* __restrict__ any_active,   // [1]
        const int nTiles
    )
    {
        if (any_active[0] != 0) return;

        int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (tid >= nTiles) return;

        tile_active[tid] = (tile_roi[tid] != 0) ? 1 : 0;
    }
    '''
    return _device_cached_rawkernel(
        build_fallback_full_roi_kernel_3d,
        code,
        "fallback_full_roi_if_empty",
    )



def build_apply_roi_tile_updates_kernel_3d():
    import cupy as cp

    code = r'''
    extern "C" __global__
    void apply_roi_tile_updates_3d_flags_packed(
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ tile_type,      // 0=nonROI,1=mixed,2=dense
        const int* __restrict__ tile_roi,       // 0/1
        const int* __restrict__ tile_active,    // 0/1

        const unsigned long long* __restrict__ state_src,  // packed
        unsigned long long* __restrict__ state_dst,        // packed

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesY, const int nTilesX,
        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx
    )
    {
        int tid = (int)blockIdx.x;
        if (tile_active[tid] == 0) return;
        if (tile_roi[tid] == 0) return;

        int dense = (tile_type[tid] == 2);

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int tz0 = tz * Tz;
        int ty0 = ty * Ty;
        int tx0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {

            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = tz0 + lz;
            int y = ty0 + ly;
            int x = tx0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;

            if (!dense) {
                if (!roi_mask[idx]) continue;
            } else {
                // dense: copy all in-domain voxels
            }

            state_dst[idx] = state_src[idx];
        }
    }
    '''
    return _device_cached_rawkernel(
        build_apply_roi_tile_updates_kernel_3d,
        code,
        "apply_roi_tile_updates_3d_flags_packed",
    )





def _compute_pow2_decode_params(tile_size):
    """
    Parameters of the power-of-two bit decoding:
      - used only when Tx is a power of two and (Ty*Tx) is a power of two
      - returns:
          pow2_decode (int32 0/1),
          shift_TyTx, shift_Tx,
          mask_TyTx, mask_Tx
    """
    import math
    Tz, Ty, Tx = [int(v) for v in tile_size]
    TyTx = int(Ty * Tx)

    def _is_pow2(n: int) -> bool:
        return (n > 0) and ((n & (n - 1)) == 0)

    pow2_decode = int(_is_pow2(Tx) and _is_pow2(TyTx))
    if pow2_decode:
        shift_Tx = int(math.log2(Tx))
        shift_TyTx = int(math.log2(TyTx))
        mask_Tx = int(Tx - 1)
        mask_TyTx = int(TyTx - 1)
    else:
        shift_Tx = 0
        shift_TyTx = 0
        mask_Tx = 0
        mask_TyTx = 0

    return pow2_decode, shift_TyTx, shift_Tx, mask_TyTx, mask_Tx


def build_los_kmax_init_kernel_3d():
    """
    Initialise seg_clear (k=0) and kmax for one direction (dz,dy,dx):
      seg[idx] = 1 <=> the first step idx->idx+dir is fluid (mask==1) and inside the domain
      kmax[idx] = 0 if seg[idx]==1 else -1
    """
    import cupy as cp
    code = r'''
    extern "C" __global__
    void init_los_kmax_dir(
        const unsigned char* __restrict__ mask,   // [nvox]
        unsigned char* __restrict__ seg,          // [nvox] 0/1
        signed char* __restrict__ kmax,           // [nvox] -1..K
        const int D, const int H, const int W,
        const int dz, const int dy, const int dx
    )
    {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx]) {
            seg[idx] = 0;
            kmax[idx] = (signed char)(-1);
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int z1 = z + dz;
        int y1 = y + dy;
        int x1 = x + dx;

        if ((unsigned)z1 >= (unsigned)D ||
            (unsigned)y1 >= (unsigned)H ||
            (unsigned)x1 >= (unsigned)W) {
            seg[idx] = 0;
            kmax[idx] = (signed char)(-1);
            return;
        }

        int stride = dz * HW + dy * W + dx;
        int idx1 = idx + stride;

        unsigned char v = mask[idx1];
        if (v) {
            seg[idx] = 1;
            kmax[idx] = (signed char)0;
        } else {
            seg[idx] = 0;
            kmax[idx] = (signed char)(-1);
        }
    }
    '''
    return _device_cached_rawkernel(
        build_los_kmax_init_kernel_3d,
        code,
        "init_los_kmax_dir",
    )


def build_los_kmax_update_kernel_3d():
    """
    Doubling update of seg_clear for one direction:
      seg_next[idx] = seg_prev[idx] & seg_prev[idx + step*dir]
    if seg_next[idx]==1, then kmax[idx]=k
    """
    import cupy as cp
    code = r'''
    extern "C" __global__
    void update_los_kmax_dir(
        const unsigned char* __restrict__ seg_prev,  // [nvox]
        unsigned char* __restrict__ seg_next,        // [nvox]
        signed char* __restrict__ kmax,              // [nvox]
        const int D, const int H, const int W,
        const int dz, const int dy, const int dx,
        const int step,    // 2^(k-1)
        const int k        // current k
    )
    {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!seg_prev[idx]) {
            seg_next[idx] = 0;
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int z2 = z + step * dz;
        int y2 = y + step * dy;
        int x2 = x + step * dx;

        if ((unsigned)z2 >= (unsigned)D ||
            (unsigned)y2 >= (unsigned)H ||
            (unsigned)x2 >= (unsigned)W) {
            seg_next[idx] = 0;
            return;
        }

        int stride = dz * HW + dy * W + dx;
        int idx2 = idx + step * stride;

        unsigned char out = (unsigned char)(seg_prev[idx] & seg_prev[idx2]);
        seg_next[idx] = out;
        if (out) {
            kmax[idx] = (signed char)k;
        }
    }
    '''
    return _device_cached_rawkernel(
        build_los_kmax_update_kernel_3d,
        code,
        "update_los_kmax_dir",
    )


def precompute_los_kmax_27dirs_3d(
    mask_flat,
    D, H, W,
    max_k=None,
    init_kernel=None,
    update_kernel=None,
    verbose=False,
):
    """
    Batched kmax27 (1 + max_k launches):
      - output kmax27: cp.int8 shape (27, nvox), direction-major contiguous
      - no synchronize (the caller synchronises when measuring wall time or copying the result to the CPU)
    """
    import numpy as np
    import cupy as cp

    D = int(D); H = int(H); W = int(W)
    nvox = int(D * H * W)

    if max_k is None:
        maxdim = int(max(D, H, W))
        max_k = int(maxdim.bit_length() - 1)

    # mask_flat: ensure cp.uint8 contiguous 1D
    if not isinstance(mask_flat, cp.ndarray):
        mask_flat = cp.asarray(mask_flat)
    if mask_flat.ndim != 1:
        mask_flat = mask_flat.ravel()
    if mask_flat.dtype != cp.uint8:
        mask_flat = mask_flat.astype(cp.uint8, copy=False)

    # --- choose batched kernels (ignore old per-dir kernels if user passed them) ---
    if (init_kernel is None) or (getattr(init_kernel, "name", "") != "los_kmax27_init_all_dirs_3d"):
        init_kernel = build_los_kmax27_init_all_dirs_3d()
    if (update_kernel is None) or (getattr(update_kernel, "name", "") != "los_kmax27_update_all_dirs_3d"):
        update_kernel = build_los_kmax27_update_all_dirs_3d()

    if verbose:
        print(f"[LOS-kmax][batched] nvox={nvox}, max_k={int(max_k)} (launches={1+int(max_k)})")

    # allocate (dir-major contiguous)
    kmax27 = cp.empty((27, nvox), dtype=cp.int8)

    threads = 256
    blocks = (nvox + threads - 1) // threads

    # init: one launch for all 27 dirs
    init_kernel(
        (int(blocks), 27),
        (int(threads),),
        (
            mask_flat,
            kmax27,
            np.int32(D), np.int32(H), np.int32(W),
            np.int32(nvox),
            np.int32(int(max_k)),
        ),
    )

    # update: k=1..max_k (each one launch for all dirs)
    for k in range(1, int(max_k) + 1):
        update_kernel(
            (int(blocks), 27),
            (int(threads),),
            (
                mask_flat,
                kmax27,
                np.int32(D), np.int32(H), np.int32(W),
                np.int32(nvox),
                np.int32(k),
            ),
        )

    return kmax27

def build_geodesic_roi_jfa_step_active_list_kernels_3d(use_int_offset=True):
    """
    Active-list step kernel:
      - grid = (n_active,)
      - tid = active_ids[blockIdx.x]
      - no in-kernel test if(tile_active[tid]==0) return; (the launch is already compacted)
    """
    import cupy as cp

    if use_int_offset:
        km_macro = r'''
        #define KM_AT(dir, idx) los_kmax27[(dir) * nvox + (idx)]
        '''
    else:
        km_macro = r'''
        #define KM_AT(dir, idx) los_kmax27[(long long)(dir) * (long long)nvox + (long long)(idx)]
        '''

    code = r'''
    #include <cuda_runtime.h>
    ''' + km_macro + r'''

    extern "C" __global__
    void geodesic_roi_jfa3d_step_tiles_active_list_flags_packed(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ tile_type,     // 0=nonROI, 1=mixed, 2=dense
        const int* __restrict__ tile_roi,      // 0/1
        const int* __restrict__ tile_active,   // kept for compatibility (the active-list version does not read it)
        const signed char* __restrict__ los_kmax27, // [27*nvox], dir-major
        const int nvox,

        const unsigned long long* __restrict__ state_in,   // packed
        unsigned long long* __restrict__ state_out,        // packed

        int* __restrict__ tile_changed_out,
        int* __restrict__ tile_unfinished_out,

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,
        const int jump,
        const int jump_k,
        const float eps,
        const int changed_mode,        // 0=frontier, 1=competitive
        const int manhattan_mode,      // 0/1
        const int full_write_mixed,    // 0/1
        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx,

        // ===== NEW =====
        const int* __restrict__ active_ids,  // [n_active]
        const int n_active
    )
    {
        int bid = (int)blockIdx.x;
        if (bid >= n_active) return;

        int tid = active_ids[bid];

        int nTiles = nTilesZ * nTilesY * nTilesX;
        if ((unsigned)tid >= (unsigned)nTiles) return;
        if (tile_roi[tid] == 0) return;   // safety

        int ttype = tile_type[tid];
        int dense = (ttype == 2);

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int tz0 = tz * Tz;
        int ty0 = ty * Ty;
        int tx0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        int local_changed = 0;
        int local_unfinished = 0;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {

            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = tz0 + lz;
            int y = ty0 + ly;
            int x = tx0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;

            // mixed tile: only ROI voxels are "updated"
            if (!dense) {
                if (!roi_mask[idx]) {
                    if (full_write_mixed) {
                        state_out[idx] = state_in[idx]; // fullwrite copy
                    }
                    continue;
                }
            }

            unsigned long long cur = state_in[idx];
            int   old_label = (int)(cur & 0xFFFFFFFFu);
            float old_dist  = __uint_as_float((unsigned int)(cur >> 32));

            int   best_label = old_label;
            float best_dist  = old_dist;

            // 26 neighbors at distance 'jump'
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;

                        int jz = z + jump * dz;
                        int jy = y + jump * dy;
                        int jx = x + jump * dx;

                        if ((unsigned)jz >= (unsigned)D ||
                            (unsigned)jy >= (unsigned)H ||
                            (unsigned)jx >= (unsigned)W) {
                            continue;
                        }

                        int stride = dz * HW + dy * W + dx;
                        int j_idx = idx + jump * stride;

                        if (!mask[j_idx]) continue;

                        unsigned long long nb = state_in[j_idx];
                        int nb_label = (int)(nb & 0xFFFFFFFFu);
                        if (nb_label < 0) continue;
                        float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                        int nnz = (dx != 0) + (dy != 0) + (dz != 0);

                        // A) direct LOS edge
                        int dir_id = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                        signed char km_dir = KM_AT(dir_id, idx);
                        int direct_ok = ((int)km_dir >= jump_k);

                        if (direct_ok) {
                            float step = (nnz == 1) ? (float)jump
                                       : (nnz == 2) ? (float)jump * SQRT2
                                                    : (float)jump * SQRT3;

                            float cand = nb_dist + step;
                            bool improve = (cand + eps < best_dist);
                            bool tie_better_label = (fabsf(cand - best_dist) <= eps) &&
                                                    (nb_label >= 0) &&
                                                    (best_label < 0 || nb_label < best_label);
                            if (improve || tie_better_label) {
                                best_dist  = cand;
                                best_label = nb_label;
                            }
                            continue;
                        }

                        // B) Manhattan fallback
                        if (manhattan_mode == 0) continue;
                        if (nnz < 2) continue;

                        int dir_x = 13 + dx;
                        int dir_y = 13 + 3 * dy;
                        int dir_z = 13 + 9 * dz;

                        int ok = 0;

                        if (nnz == 2) {
                            if (dx != 0 && dy != 0 && dz == 0) {
                                signed char kmx0 = KM_AT(dir_x, idx);
                                if ((int)kmx0 >= jump_k) {
                                    int idx1 = idx + jump * dx;
                                    signed char kmy1 = KM_AT(dir_y, idx1);
                                    if ((int)kmy1 >= jump_k) ok = 1;
                                }
                                if (!ok) {
                                    signed char kmy0 = KM_AT(dir_y, idx);
                                    if ((int)kmy0 >= jump_k) {
                                        int idx1 = idx + jump * dy * W;
                                        signed char kmx1 = KM_AT(dir_x, idx1);
                                        if ((int)kmx1 >= jump_k) ok = 1;
                                    }
                                }
                            } else if (dx != 0 && dz != 0 && dy == 0) {
                                signed char kmx0 = KM_AT(dir_x, idx);
                                if ((int)kmx0 >= jump_k) {
                                    int idx1 = idx + jump * dx;
                                    signed char kmz1 = KM_AT(dir_z, idx1);
                                    if ((int)kmz1 >= jump_k) ok = 1;
                                }
                                if (!ok) {
                                    signed char kmz0 = KM_AT(dir_z, idx);
                                    if ((int)kmz0 >= jump_k) {
                                        int idx1 = idx + jump * dz * HW;
                                        signed char kmx1 = KM_AT(dir_x, idx1);
                                        if ((int)kmx1 >= jump_k) ok = 1;
                                    }
                                }
                            } else if (dy != 0 && dz != 0 && dx == 0) {
                                signed char kmy0 = KM_AT(dir_y, idx);
                                if ((int)kmy0 >= jump_k) {
                                    int idx1 = idx + jump * dy * W;
                                    signed char kmz1 = KM_AT(dir_z, idx1);
                                    if ((int)kmz1 >= jump_k) ok = 1;
                                }
                                if (!ok) {
                                    signed char kmz0 = KM_AT(dir_z, idx);
                                    if ((int)kmz0 >= jump_k) {
                                        int idx1 = idx + jump * dz * HW;
                                        signed char kmy1 = KM_AT(dir_y, idx1);
                                        if ((int)kmy1 >= jump_k) ok = 1;
                                    }
                                }
                            }
                        } else {
                            int offx = jump * dx;
                            int offy = jump * dy * W;
                            int offz = jump * dz * HW;

                            signed char kmx0 = KM_AT(dir_x, idx);
                            signed char kmy0 = KM_AT(dir_y, idx);
                            signed char kmz0 = KM_AT(dir_z, idx);

                            if (!ok && (int)kmx0 >= jump_k) {
                                int i1 = idx + offx;
                                signed char kmy1 = KM_AT(dir_y, i1);
                                if ((int)kmy1 >= jump_k) {
                                    int i2 = i1 + offy;
                                    signed char kmz2 = KM_AT(dir_z, i2);
                                    if ((int)kmz2 >= jump_k) ok = 1;
                                }
                            }
                            if (!ok && (int)kmx0 >= jump_k) {
                                int i1 = idx + offx;
                                signed char kmz1 = KM_AT(dir_z, i1);
                                if ((int)kmz1 >= jump_k) {
                                    int i2 = i1 + offz;
                                    signed char kmy2 = KM_AT(dir_y, i2);
                                    if ((int)kmy2 >= jump_k) ok = 1;
                                }
                            }
                            if (!ok && (int)kmy0 >= jump_k) {
                                int i1 = idx + offy;
                                signed char kmx1 = KM_AT(dir_x, i1);
                                if ((int)kmx1 >= jump_k) {
                                    int i2 = i1 + offx;
                                    signed char kmz2 = KM_AT(dir_z, i2);
                                    if ((int)kmz2 >= jump_k) ok = 1;
                                }
                            }
                            if (!ok && (int)kmy0 >= jump_k) {
                                int i1 = idx + offy;
                                signed char kmz1 = KM_AT(dir_z, i1);
                                if ((int)kmz1 >= jump_k) {
                                    int i2 = i1 + offz;
                                    signed char kmx2 = KM_AT(dir_x, i2);
                                    if ((int)kmx2 >= jump_k) ok = 1;
                                }
                            }
                            if (!ok && (int)kmz0 >= jump_k) {
                                int i1 = idx + offz;
                                signed char kmx1 = KM_AT(dir_x, i1);
                                if ((int)kmx1 >= jump_k) {
                                    int i2 = i1 + offx;
                                    signed char kmy2 = KM_AT(dir_y, i2);
                                    if ((int)kmy2 >= jump_k) ok = 1;
                                }
                            }
                            if (!ok && (int)kmz0 >= jump_k) {
                                int i1 = idx + offz;
                                signed char kmy1 = KM_AT(dir_y, i1);
                                if ((int)kmy1 >= jump_k) {
                                    int i2 = i1 + offy;
                                    signed char kmx2 = KM_AT(dir_x, i2);
                                    if ((int)kmx2 >= jump_k) ok = 1;
                                }
                            }
                        }

                        if (!ok) continue;

                        float step_m = (float)jump * (float)nnz;
                        float cand_m = nb_dist + step_m;

                        bool improve_m = (cand_m + eps < best_dist);
                        bool tie_better_label_m = (fabsf(cand_m - best_dist) <= eps) &&
                                                  (nb_label >= 0) &&
                                                  (best_label < 0 || nb_label < best_label);
                        if (improve_m || tie_better_label_m) {
                            best_dist  = cand_m;
                            best_label = nb_label;
                        }
                    }
                }
            }

            unsigned int out_du = __float_as_uint(best_dist);
            unsigned long long out_pack =
                ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

            state_out[idx] = out_pack;

            if (best_label < 0) local_unfinished = 1;

            if (changed_mode == 0) {
                if (old_label < 0 && best_label >= 0) local_changed = 1;
            } else {
                float diff = old_dist - best_dist;
                if (diff > eps) local_changed = 1;
            }
        }

        int any_changed    = __syncthreads_or(local_changed);
        int any_unfinished = __syncthreads_or(local_unfinished);

        if (threadIdx.x == 0) {
            tile_changed_out[tid]    = any_changed;
            tile_unfinished_out[tid] = any_unfinished;
        }
    }
    '''

    code = code.encode("ascii", "ignore").decode("ascii")
    k_step = _device_cached_rawkernel(
        build_geodesic_roi_jfa_step_active_list_kernels_3d,
        code,
        "geodesic_roi_jfa3d_step_tiles_active_list_flags_packed",
        options=("-std=c++11",),
        cache_key=int(bool(use_int_offset)),
    )
    return k_step, k_step

def build_apply_roi_tile_updates_active_list_kernel_3d():
    """
    Active-list apply kernel:
      - grid = (n_active,)
      - tid = active_ids[blockIdx.x]
    """
    import cupy as cp

    code = r'''
    extern "C" __global__
    void apply_roi_tile_updates_3d_flags_packed_active_list(
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ tile_type,      // 0=nonROI,1=mixed,2=dense
        const int* __restrict__ tile_roi,       // 0/1
        const int* __restrict__ tile_active,    // kept for compatibility (the active-list version does not read it)

        const unsigned long long* __restrict__ state_src,
        unsigned long long* __restrict__ state_dst,

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesY, const int nTilesX,
        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx,

        // ===== NEW =====
        const int* __restrict__ active_ids,
        const int n_active
    )
    {
        int bid = (int)blockIdx.x;
        if (bid >= n_active) return;

        int tid = active_ids[bid];
        if (tile_roi[tid] == 0) return;

        int dense = (tile_type[tid] == 2);

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int tz0 = tz * Tz;
        int ty0 = ty * Ty;
        int tx0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {

            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = tz0 + lz;
            int y = ty0 + ly;
            int x = tx0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;

            if (!dense) {
                if (!roi_mask[idx]) continue;
            }

            state_dst[idx] = state_src[idx];
        }
    }
    '''
    return _device_cached_rawkernel(
        build_apply_roi_tile_updates_active_list_kernel_3d,
        code,
        "apply_roi_tile_updates_3d_flags_packed_active_list",
        options=("-std=c++11",),
    )


def build_geodesic_roi_jfa_step_kernels_3d(use_int_offset=True):
    """
    ROI-JFA step kernel (tile-based, flags), packed-state version

    Manhattan-jump stencil + LOS-safe:
      - when the LOS (kmax27) along the straight direction (dz,dy,dx) does not support the jump,
        Manhattan 2-segment / 3-segment paths (axis-aligned segments) are tried as well:
          nnz=2 (face diagonal):   two axis-aligned segments (both orders tried)
          nnz=3 (space diagonal):  three axis-aligned segments (all 6 orders tried)
      - reachability of each segment is still an O(1) test on kmax27 (axis directions),
        so path existence follows the adjacency of the solver (no crossing of walls).
      - the step cost of a Manhattan path is the axis cost
          step_manhattan = jump * nnz
        a conservative upper bound (never an underestimate); the later jump=1 relaxation / closure
        brings the distance back down.

    Interface:
      - the kernel takes the argument manhattan_mode:
          0 -> Manhattan fallback off
          1 -> Manhattan fallback on

    Returns:
      (k_step, k_step)  (a tuple, for compatibility with the callers)
    """
    import cupy as cp

    # --------- kmax27 index macro: int or long long ----------
    if use_int_offset:
        km_macro = r'''
        #define KM_AT(dir, idx) los_kmax27[(dir) * nvox + (idx)]
        '''
    else:
        km_macro = r'''
        #define KM_AT(dir, idx) los_kmax27[(long long)(dir) * (long long)nvox + (long long)(idx)]
        '''

    code = r'''
    #include <cuda_runtime.h>
    ''' + km_macro + r'''

    extern "C" __global__
    void geodesic_roi_jfa3d_step_tiles_flags_packed(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ tile_type,     // 0=nonROI, 1=mixed, 2=dense
        const int* __restrict__ tile_roi,      // 0/1
        const int* __restrict__ tile_active,   // 0/1
        const signed char* __restrict__ los_kmax27, // [27*nvox], dir-major
        const int nvox,

        const unsigned long long* __restrict__ state_in,   // packed
        unsigned long long* __restrict__ state_out,        // packed

        int* __restrict__ tile_changed_out,
        int* __restrict__ tile_unfinished_out,

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,
        const int jump,
        const int jump_k,
        const float eps,
        const int changed_mode,        // 0=frontier, 1=competitive
        const int manhattan_mode,      // 0/1
        const int full_write_mixed,    // 0/1
        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx
    )
    {
        int tid = (int)blockIdx.x;
        int nTiles = nTilesZ * nTilesY * nTilesX;
        if (tid >= nTiles) return;
        if (tile_roi[tid] == 0) return;
        if (tile_active[tid] == 0) return;

        int ttype = tile_type[tid];
        int dense = (ttype == 2);

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int tz0 = tz * Tz;
        int ty0 = ty * Ty;
        int tx0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        int local_changed = 0;
        int local_unfinished = 0;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {

            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = tz0 + lz;
            int y = ty0 + ly;
            int x = tx0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;

            // mixed tile: only ROI voxels are "updated"
            if (!dense) {
                if (!roi_mask[idx]) {
                    if (full_write_mixed) {
                        state_out[idx] = state_in[idx]; // fullwrite copy
                    }
                    continue;
                }
                // roi_mask==1 implies mask==1 (normally so, but mask is still the reference)
            }

            unsigned long long cur = state_in[idx];
            int   old_label = (int)(cur & 0xFFFFFFFFu);
            float old_dist  = __uint_as_float((unsigned int)(cur >> 32));

            int   best_label = old_label;
            float best_dist  = old_dist;

            // 26 neighbors at distance 'jump'
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;

                        int jz = z + jump * dz;
                        int jy = y + jump * dy;
                        int jx = x + jump * dx;

                        if ((unsigned)jz >= (unsigned)D ||
                            (unsigned)jy >= (unsigned)H ||
                            (unsigned)jx >= (unsigned)W) {
                            continue;
                        }

                        int stride = dz * HW + dy * W + dx;
                        int j_idx = idx + jump * stride;

                        // neighbor must be fluid
                        if (!mask[j_idx]) continue;

                        // neighbor must have label
                        unsigned long long nb = state_in[j_idx];
                        int nb_label = (int)(nb & 0xFFFFFFFFu);
                        if (nb_label < 0) continue;
                        float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                        // nnz for direction
                        int nnz = (dx != 0) + (dy != 0) + (dz != 0);

                        // ------------------------------------------------------------
                        // A) direct LOS edge (standard JFA edge)
                        // ------------------------------------------------------------
                        int dir_id = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1); // 0..26
                        signed char km_dir = KM_AT(dir_id, idx);

                        int direct_ok = ((int)km_dir >= jump_k);

                        if (direct_ok) {
                            float step = (nnz == 1) ? (float)jump
                                       : (nnz == 2) ? (float)jump * SQRT2
                                                    : (float)jump * SQRT3;

                            float cand = nb_dist + step;
                            if (cand + eps < best_dist) {
                                best_dist  = cand;
                                best_label = nb_label;
                            }
                            continue;
                        }

                        // ------------------------------------------------------------
                        // B) Manhattan-jump fallback (only when direct blocked)
                        //    Uses 2/3 axis segments of length 'jump', checked via kmax27.
                        // ------------------------------------------------------------
                        if (manhattan_mode == 0) continue;
                        if (nnz < 2) continue;  // axis direction doesn't need Manhattan fallback

                        // Axis dir ids can be derived from center(13):
                        //   dir_x = 13 + dx
                        //   dir_y = 13 + 3*dy
                        //   dir_z = 13 + 9*dz
                        int dir_x = 13 + dx;      // valid if dx!=0
                        int dir_y = 13 + 3 * dy;  // valid if dy!=0
                        int dir_z = 13 + 9 * dz;  // valid if dz!=0

                        int ok = 0;

                        if (nnz == 2) {
                            // face diagonal: try both L-orders
                            if (dx != 0 && dy != 0 && dz == 0) {
                                // x -> y
                                signed char kmx0 = KM_AT(dir_x, idx);
                                if ((int)kmx0 >= jump_k) {
                                    int idx1 = idx + jump * dx;
                                    signed char kmy1 = KM_AT(dir_y, idx1);
                                    if ((int)kmy1 >= jump_k) ok = 1;
                                }
                                // y -> x
                                if (!ok) {
                                    signed char kmy0 = KM_AT(dir_y, idx);
                                    if ((int)kmy0 >= jump_k) {
                                        int idx1 = idx + jump * dy * W;
                                        signed char kmx1 = KM_AT(dir_x, idx1);
                                        if ((int)kmx1 >= jump_k) ok = 1;
                                    }
                                }
                            } else if (dx != 0 && dz != 0 && dy == 0) {
                                // x -> z
                                signed char kmx0 = KM_AT(dir_x, idx);
                                if ((int)kmx0 >= jump_k) {
                                    int idx1 = idx + jump * dx;
                                    signed char kmz1 = KM_AT(dir_z, idx1);
                                    if ((int)kmz1 >= jump_k) ok = 1;
                                }
                                // z -> x
                                if (!ok) {
                                    signed char kmz0 = KM_AT(dir_z, idx);
                                    if ((int)kmz0 >= jump_k) {
                                        int idx1 = idx + jump * dz * HW;
                                        signed char kmx1 = KM_AT(dir_x, idx1);
                                        if ((int)kmx1 >= jump_k) ok = 1;
                                    }
                                }
                            } else if (dy != 0 && dz != 0 && dx == 0) {
                                // y -> z
                                signed char kmy0 = KM_AT(dir_y, idx);
                                if ((int)kmy0 >= jump_k) {
                                    int idx1 = idx + jump * dy * W;
                                    signed char kmz1 = KM_AT(dir_z, idx1);
                                    if ((int)kmz1 >= jump_k) ok = 1;
                                }
                                // z -> y
                                if (!ok) {
                                    signed char kmz0 = KM_AT(dir_z, idx);
                                    if ((int)kmz0 >= jump_k) {
                                        int idx1 = idx + jump * dz * HW;
                                        signed char kmy1 = KM_AT(dir_y, idx1);
                                        if ((int)kmy1 >= jump_k) ok = 1;
                                    }
                                }
                            }
                        } else {
                            // nnz == 3 : space diagonal, try 6 permutations
                            int offx = jump * dx;
                            int offy = jump * dy * W;
                            int offz = jump * dz * HW;

                            signed char kmx0 = KM_AT(dir_x, idx);
                            signed char kmy0 = KM_AT(dir_y, idx);
                            signed char kmz0 = KM_AT(dir_z, idx);

                            // x -> y -> z
                            if (!ok && (int)kmx0 >= jump_k) {
                                int i1 = idx + offx;
                                signed char kmy1 = KM_AT(dir_y, i1);
                                if ((int)kmy1 >= jump_k) {
                                    int i2 = i1 + offy;
                                    signed char kmz2 = KM_AT(dir_z, i2);
                                    if ((int)kmz2 >= jump_k) ok = 1;
                                }
                            }
                            // x -> z -> y
                            if (!ok && (int)kmx0 >= jump_k) {
                                int i1 = idx + offx;
                                signed char kmz1 = KM_AT(dir_z, i1);
                                if ((int)kmz1 >= jump_k) {
                                    int i2 = i1 + offz;
                                    signed char kmy2 = KM_AT(dir_y, i2);
                                    if ((int)kmy2 >= jump_k) ok = 1;
                                }
                            }
                            // y -> x -> z
                            if (!ok && (int)kmy0 >= jump_k) {
                                int i1 = idx + offy;
                                signed char kmx1 = KM_AT(dir_x, i1);
                                if ((int)kmx1 >= jump_k) {
                                    int i2 = i1 + offx;
                                    signed char kmz2 = KM_AT(dir_z, i2);
                                    if ((int)kmz2 >= jump_k) ok = 1;
                                }
                            }
                            // y -> z -> x
                            if (!ok && (int)kmy0 >= jump_k) {
                                int i1 = idx + offy;
                                signed char kmz1 = KM_AT(dir_z, i1);
                                if ((int)kmz1 >= jump_k) {
                                    int i2 = i1 + offz;
                                    signed char kmx2 = KM_AT(dir_x, i2);
                                    if ((int)kmx2 >= jump_k) ok = 1;
                                }
                            }
                            // z -> x -> y
                            if (!ok && (int)kmz0 >= jump_k) {
                                int i1 = idx + offz;
                                signed char kmx1 = KM_AT(dir_x, i1);
                                if ((int)kmx1 >= jump_k) {
                                    int i2 = i1 + offx;
                                    signed char kmy2 = KM_AT(dir_y, i2);
                                    if ((int)kmy2 >= jump_k) ok = 1;
                                }
                            }
                            // z -> y -> x
                            if (!ok && (int)kmz0 >= jump_k) {
                                int i1 = idx + offz;
                                signed char kmy1 = KM_AT(dir_y, i1);
                                if ((int)kmy1 >= jump_k) {
                                    int i2 = i1 + offy;
                                    signed char kmx2 = KM_AT(dir_x, i2);
                                    if ((int)kmx2 >= jump_k) ok = 1;
                                }
                            }
                        }

                        if (!ok) continue;

                        // Manhattan path cost (safe upper bound)
                        float step_m = (float)jump * (float)nnz;  // 2*jump or 3*jump
                        float cand_m = nb_dist + step_m;

                        if (cand_m + eps < best_dist) {
                            best_dist  = cand_m;
                            best_label = nb_label;
                        }
                    }
                }
            }

            // write packed out
            unsigned int out_du = __float_as_uint(best_dist);
            unsigned long long out_pack =
                ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

            state_out[idx] = out_pack;

            if (best_label < 0) local_unfinished = 1;

            if (changed_mode == 0) {
                // frontier
                if ((old_label < 0 && best_label >= 0) ||
                    (old_label >= 0 && best_label >= 0 && best_label != old_label)) {
                    local_changed = 1;
                }
            } else {
                // competitive
                float diff = old_dist - best_dist;
                if (diff > eps || best_label != old_label) local_changed = 1;
            }
        }

        int any_changed    = __syncthreads_or(local_changed);
        int any_unfinished = __syncthreads_or(local_unfinished);

        if (threadIdx.x == 0) {
            tile_changed_out[tid]    = any_changed;
            tile_unfinished_out[tid] = any_unfinished;
        }
    }
    ''';

    code = code.encode("ascii", "ignore").decode("ascii")
    k_step = _device_cached_rawkernel(
        build_geodesic_roi_jfa_step_kernels_3d,
        code,
        "geodesic_roi_jfa3d_step_tiles_flags_packed",
        cache_key=int(bool(use_int_offset)),   # different offset modes are cached separately
    )
    return k_step, k_step












def build_geodesic_roi_jfa_check_kernels_3d():
    """
    Check kernel for the small-jump stage:
      - writes neither label_out nor dist_out
      - outputs only:
          tile_need_update[tid] = 1  <=>  some ROI voxel v of the tile has best_dist + eps < old_dist
          tile_unfinished[tid]  = 1  <=>  some ROI voxel of the tile has label_in[v] < 0
    """
    code_check_mixed = r'''
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

    extern "C" __global__
    void geodesic_roi_jfa3d_check_mixed(
        const unsigned char* __restrict__ mask,
        const unsigned char* __restrict__ roi_mask,
        const int* __restrict__ tile_ids,
        const int n_tiles,
        const int*  __restrict__ label_in,
        const float* __restrict__ dist_in,
        int*  __restrict__ tile_need_update,
        int*  __restrict__ tile_unfinished,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesY, const int nTilesX,
        const int jump,
        const float eps
    )
    {
        int tile_idx = blockIdx.x;
        if (tile_idx >= n_tiles) return;

        int tid = tile_ids[tile_idx];

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int tz0 = tz * Tz;
        int ty0 = ty * Ty;
        int tx0 = tx * Tx;

        const int HW = H * W;
        const int tile_vox = Tz * Ty * Tx;

        __shared__ int s_need;
        __shared__ int s_unfinished;
        if (threadIdx.x == 0) {
            s_need = 0;
            s_unfinished = 0;
        }
        __syncthreads();

        int local_need = 0;
        int local_unfinished = 0;

        for (int p = threadIdx.x; p < tile_vox; p += blockDim.x) {
            int lz = p / (Ty * Tx);
            int rem2 = p - lz * (Ty * Tx);
            int ly = rem2 / Tx;
            int lx = rem2 - ly * Tx;

            int z = tz0 + lz;
            int y = ty0 + ly;
            int x = tx0 + lx;

            if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) continue;

            int idx = z * HW + y * W + x;

            if (!mask[idx]) continue;
            if (!roi_mask[idx]) continue;

            int   old_label = label_in[idx];
            float old_dist  = dist_in[idx];

            if (old_label < 0) {
                local_unfinished = 1;
            }

            int   best_label = old_label;
            float best_dist  = old_dist;

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;

                        int jz = z + jump * dz;
                        int jy = y + jump * dy;
                        int jx = x + jump * dx;

                        if (jz < 0 || jz >= D || jy < 0 || jy >= H || jx < 0 || jx >= W)
                            continue;

                        int j_idx = jz * HW + jy * W + jx;
                        if (!mask[j_idx]) continue;

                        int neigh_label = label_in[j_idx];
                        if (neigh_label < 0) continue;

                        bool blocked = false;
                        for (int tstep = 1; tstep < jump; ++tstep) {
                            int kz = z + tstep * dz;
                            int ky = y + tstep * dy;
                            int kx = x + tstep * dx;

                            if (kz < 0 || kz >= D || ky < 0 || ky >= H || kx < 0 || kx >= W) {
                                blocked = true;
                                break;
                            }

                            int k_idx = kz * HW + ky * W + kx;
                            if (!mask[k_idx]) {
                                blocked = true;
                                break;
                            }
                        }
                        if (blocked) continue;

                        int dz_tot = jz - z;
                        int dy_tot = jy - y;
                        int dx_tot = jx - x;
                        float step = sqrtf((float)(dz_tot*dz_tot + dy_tot*dy_tot + dx_tot*dx_tot));
                        float cand_dist = dist_in[j_idx] + step;

                        if (cand_dist + eps < best_dist) {
                            best_dist  = cand_dist;
                            best_label = neigh_label;
                        }
                    }
                }
            }

            // if any voxel can shorten its dist, the tile "needs an update"
            if (best_dist + eps < old_dist) {
                local_need = 1;
            }
        }

        if (local_need)      atomicExch(&s_need, 1);
        if (local_unfinished)atomicExch(&s_unfinished, 1);
        __syncthreads();

        if (threadIdx.x == 0) {
            tile_need_update[tid] = s_need;
            tile_unfinished[tid]  = s_unfinished;
        }
    }
    '''

    code_check_dense = r'''
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

    extern "C" __global__
    void geodesic_roi_jfa3d_check_dense(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ tile_ids,
        const int n_tiles,
        const int*  __restrict__ label_in,
        const float* __restrict__ dist_in,
        int*  __restrict__ tile_need_update,
        int*  __restrict__ tile_unfinished,
        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesY, const int nTilesX,
        const int jump,
        const float eps
    )
    {
        int tile_idx = blockIdx.x;
        if (tile_idx >= n_tiles) return;

        int tid = tile_ids[tile_idx];

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int tz0 = tz * Tz;
        int ty0 = ty * Ty;
        int tx0 = tx * Tx;

        const int HW = H * W;
        const int tile_vox = Tz * Ty * Tx;

        __shared__ int s_need;
        __shared__ int s_unfinished;
        if (threadIdx.x == 0) {
            s_need = 0;
            s_unfinished = 0;
        }
        __syncthreads();

        int local_need = 0;
        int local_unfinished = 0;

        for (int p = threadIdx.x; p < tile_vox; p += blockDim.x) {
            int lz = p / (Ty * Tx);
            int rem2 = p - lz * (Ty * Tx);
            int ly = rem2 / Tx;
            int lx = rem2 - ly * Tx;

            int z = tz0 + lz;
            int y = ty0 + ly;
            int x = tx0 + lx;

            if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) continue;

            int idx = z * HW + y * W + x;

            if (!mask[idx]) continue;

            int   old_label = label_in[idx];
            float old_dist  = dist_in[idx];

            if (old_label < 0) {
                local_unfinished = 1;
            }

            int   best_label = old_label;
            float best_dist  = old_dist;

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;

                        int jz = z + jump * dz;
                        int jy = y + jump * dy;
                        int jx = x + jump * dx;

                        if (jz < 0 || jz >= D || jy < 0 || jy >= H || jx < 0 || jx >= W)
                            continue;

                        int j_idx = jz * HW + jy * W + jx;
                        if (!mask[j_idx]) continue;

                        int neigh_label = label_in[j_idx];
                        if (neigh_label < 0) continue;

                        bool blocked = false;
                        for (int tstep = 1; tstep < jump; ++tstep) {
                            int kz = z + tstep * dz;
                            int ky = y + tstep * dy;
                            int kx = x + tstep * dx;

                            if (kz < 0 || kz >= D || ky < 0 || ky >= H || kx < 0 || kx >= W) {
                                blocked = true;
                                break;
                            }

                            int k_idx = kz * HW + ky * W + kx;
                            if (!mask[k_idx]) {
                                blocked = true;
                                break;
                            }
                        }
                        if (blocked) continue;

                        int dz_tot = jz - z;
                        int dy_tot = jy - y;
                        int dx_tot = jx - x;
                        float step = sqrtf((float)(dz_tot*dz_tot + dy_tot*dy_tot + dx_tot*dx_tot));
                        float cand_dist = dist_in[j_idx] + step;

                        if (cand_dist + eps < best_dist) {
                            best_dist  = cand_dist;
                            best_label = neigh_label;
                        }
                    }
                }
            }

            if (best_dist + eps < old_dist) {
                local_need = 1;
            }
        }

        if (local_need)       atomicExch(&s_need, 1);
        if (local_unfinished) atomicExch(&s_unfinished, 1);
        __syncthreads();

        if (threadIdx.x == 0) {
            tile_need_update[tid] = s_need;
            tile_unfinished[tid]  = s_unfinished;
        }
    }
    '''

    k_check_mixed = _device_cached_rawkernel(
        build_geodesic_roi_jfa_check_kernels_3d,
        code_check_mixed,
        "geodesic_roi_jfa3d_check_mixed",
        cache_key="mixed",
    )
    k_check_dense = _device_cached_rawkernel(
        build_geodesic_roi_jfa_check_kernels_3d,
        code_check_dense,
        "geodesic_roi_jfa3d_check_dense",
        cache_key="dense",
    )
    return k_check_mixed, k_check_dense


# ============================================================
# 5. Complete ROI-JFA pipeline (Section 2.4.4)
# ============================================================




def geodesic_voronoi_roi_jfa(
    mask,
    seeds,
    tile_size=(8, 8, 8),
    delta_r=1.0,
    eta_max=0.8,
    r_tile=1,
    enable_stamping=True,
    verbose=False,
    dump_active_tiles=False,
    dump_prefix="roi_tiles",
    dump_label_3d=False,
    viz_policy="deferred",
    n_relax_after=1,
    relax_eps=1e-6,
    changed_mode="frontier",
    profile_gpu=False,
    return_records=False,
    stamping_kernel=None,
    tiles_dual_kernel=None,
    roi_step_kernels=None,
    active_tiles_kernel=None,
    mark_roi_tiles_kernel=None,   # kept for interface compatibility, no longer used
    max_refine_iters=460,
    relax_kernel=None,
    apply_kernel=None,
    active_fallback_kernel=None,

    clearance_kmax27=None,
    los_kmax_init_kernel=None,
    los_kmax_update_kernel=None,

    # -------------------------
    # Manhattan-jump + Closure C
    # -------------------------
    enable_manhattan_jump=True,
    enable_closure=True,
    closure_kernel=None,
    use_active_list_step=True,
):
    import numpy as np
    import cupy as cp
    import time
    import os

    global ROI_JFA_LAST_VIZ_TIME, ROI_JFA_LAST_GPU_TIME, ROI_JFA_LAST_RECORD_TIME
    global ROI_JFA_LAST_TSTAMP_WALL, ROI_JFA_LAST_TJFA_WALL, ROI_JFA_LAST_TCLOSE_WALL, ROI_JFA_LAST_TRELAX_WALL, ROI_JFA_LAST_TPRED_WALL
    global ROI_JFA_LAST_TAFINAL_WALL, ROI_JFA_LAST_TAFINAL_GPU_TIME

    ROI_JFA_LAST_VIZ_TIME = 0.0
    ROI_JFA_LAST_GPU_TIME = 0.0
    ROI_JFA_LAST_RECORD_TIME = 0.0
    ROI_JFA_LAST_TAFINAL_WALL = 0.0
    ROI_JFA_LAST_TAFINAL_GPU_TIME = 0.0

    ROI_JFA_LAST_TSTAMP_WALL = 0.0
    ROI_JFA_LAST_TJFA_WALL   = 0.0
    ROI_JFA_LAST_TCLOSE_WALL = 0.0
    ROI_JFA_LAST_TRELAX_WALL = 0.0
    ROI_JFA_LAST_TPRED_WALL  = 0.0

    viz_policy = str(viz_policy).lower().strip()
    if viz_policy not in ("inline", "deferred", "none"):
        raise ValueError("viz_policy must be 'inline', 'deferred', or 'none'")

    changed_mode = str(changed_mode).lower().strip()
    if changed_mode == "frontier":
        changed_mode_id_base = 0
    elif changed_mode in ("frontier+competitive", "frontier_competitive", "competitive"):
        changed_mode_id_base = 1
    else:
        raise ValueError("changed_mode must be 'frontier' or 'frontier+competitive'")

    enable_manhattan_jump = bool(enable_manhattan_jump)
    manhattan_mode_id = 1 if enable_manhattan_jump else 0

    enable_closure = bool(enable_closure)
    use_active_list_step = bool(use_active_list_step)
    active_list_fraction_threshold = float(os.environ.get("PVFV_ROIJFA_ACTIVE_LIST_THRESHOLD", "0.75"))
    active_list_fraction_threshold = min(1.0, max(0.0, active_list_fraction_threshold))

    record_time_total = 0.0
    gpu_ms_total = 0.0

    # ---------------------------
    # 0. data preparation (backend-aware: a cupy mask / seeds are not pulled back to the CPU)
    # ---------------------------
    mask_cp, (D, H, W), mask_flat = _as_cupy_mask_u8(mask)
    nvox = int(D * H * W)

    if isinstance(seeds, cp.ndarray):
        seeds_cp = seeds.astype(cp.int32, copy=False)
    else:
        seeds_cp = cp.asarray(np.asarray(seeds, dtype=np.int32), dtype=cp.int32)

    if seeds_cp.ndim != 2 or seeds_cp.shape[1] != 3:
        raise ValueError("seeds must be (N,3) with (z,y,x).")

    seeds_np = None
    if return_records or dump_label_3d or dump_active_tiles:
        seeds_np = cp.asnumpy(seeds_cp).astype(np.int64, copy=False)

    Tz, Ty, Tx = tile_size
    nTilesZ = (D + Tz - 1) // Tz
    nTilesY = (H + Ty - 1) // Ty
    nTilesX = (W + Tx - 1) // Tx
    nTiles  = int(nTilesZ * nTilesY * nTilesX)

    pow2_decode, shift_TyTx, shift_Tx, mask_TyTx, mask_Tx = _compute_pow2_decode_params(tile_size)

    if verbose:
        print(f"[ROI-JFA] Domain: {D} x {H} x {W}")
        print(f"[ROI-JFA] Tile size: {Tz} x {Ty} x {Tx}")
        print(f"[ROI-JFA] Tiles: {nTilesZ} x {nTilesY} x {nTilesX} = {nTiles}")
        print(f"[ROI-JFA] pow2 tile decode: {'ENABLED' if pow2_decode else 'disabled'}")
        print(f"[ROI-JFA] Manhattan-jump stencil: {'ENABLED' if enable_manhattan_jump else 'disabled'}")
        print(f"[ROI-JFA] Closure C (jump=1 fill-U): {'ENABLED' if enable_closure else 'disabled'}")
        print(f"[ROI-JFA] Launch-level active-list tiles: {'ADAPTIVE' if use_active_list_step else 'disabled'}")

    # ---------------------------
    # 0.5 LOS clearance (kmax27) precomputation (depends only on the mask, reusable)
    # ---------------------------
    if clearance_kmax27 is None:
        maxdim = int(max(D, H, W))
        max_k = int(maxdim.bit_length() - 1)

        if los_kmax_init_kernel is None:
            los_kmax_init_kernel = build_los_kmax_init_kernel_3d()
        if los_kmax_update_kernel is None:
            los_kmax_update_kernel = build_los_kmax_update_kernel_3d()

        clearance_kmax27 = precompute_los_kmax_27dirs_3d(
            mask_flat,
            D, H, W,
            max_k=max_k,
            init_kernel=los_kmax_init_kernel,
            update_kernel=los_kmax_update_kernel,
            verbose=verbose,
        )

    # =====================================================================================
    # prediction timing starts here (exclude LOS/kmax precompute from t_pred decomposition)
    # =====================================================================================
    t_pred_start = time.time()  # keep for reference; TPRED will be sum of components

    # ---------------------------
    # 1. seed stamping / ROI construction (parallel stamping + packed state)
    # ---------------------------
    if enable_stamping:
        if verbose:
            print("[ROI-JFA] Performing seed stamping (parallel, packed) ...")

        t0_stamp = time.time()
        label_stamp_flat, dist_stamp_flat, roi_mask_cp, radii_cp, state_init = perform_seed_stamping(
            mask, seeds, delta_r=delta_r, stamping_kernel=stamping_kernel,
            parallel=True, return_state=True,
            clearance_kmax27=clearance_kmax27,
        )
        cp.cuda.Device().synchronize()
        t_stamp = time.time() - t0_stamp

        roi_mask_flat = roi_mask_cp.astype(cp.uint8).ravel()
        state_state = state_init
    else:
        if verbose:
            print("[ROI-JFA] Stamping disabled, ROI = full fluid domain (packed init).")

        roi_mask_flat = mask_flat.copy().astype(cp.uint8)

        INF_BITS = np.frombuffer(np.float32(1e20).tobytes(), dtype=np.uint32)[0]
        PACK_INF_NEG1 = (int(INF_BITS) << 32) | 0xFFFFFFFF
        state_state = cp.full(nvox, PACK_INF_NEG1, dtype=cp.uint64)

        set_seeds_kernel = build_set_seeds_packed_kernel()
        seeds_flat_cp = seeds_cp.reshape(-1)
        n_seeds = int(seeds_cp.shape[0])
        threads = 256
        blocks = (n_seeds + threads - 1) // threads
        set_seeds_kernel(
            (int(blocks),),
            (int(threads),),
            (
                mask_flat,
                state_state,
                seeds_flat_cp,
                np.int32(n_seeds),
                np.int32(D), np.int32(H), np.int32(W),
            ),
        )

        radii_cp = None
        label_stamp_flat = None
        dist_stamp_flat = None
        t_stamp = 0.0

    ROI_JFA_LAST_TSTAMP_WALL = float(t_stamp)

    # JFA-stage timer begins AFTER stamping
    t_jfa_stage_start = time.time()

    # output scratch
    state_scratch = state_state.copy()

    if verbose:
        n_fluid = int(cp.count_nonzero(mask_flat).get())
        n_roi   = int(cp.count_nonzero(roi_mask_flat).get())
        eta = n_roi / max(n_fluid, 1)
        print(f"[ROI-JFA] Fluid voxels = {n_fluid}, ROI voxels = {n_roi}, eta = {eta:.4f}")
        if enable_stamping:
            if label_stamp_flat is not None:
                stamped_mask = (label_stamp_flat >= 0) & (mask_flat == 1)
                n_stamped = int(cp.count_nonzero(stamped_mask).get())
                stamped_ratio = n_stamped / max(n_fluid, 1)
                print(f"[ROI-JFA] Stamped voxels = {n_stamped} ({stamped_ratio*100:.2f}%)")
            print(f"[ROI-JFA] Seed stamping time = {t_stamp:.3f} s")
        print(f"[ROI-JFA] NOTE: eta_max={eta_max:.4f} is reference only (no OA-JFA fallback).")

    # ---------------------------
    # 2. tiles and kernels
    # ---------------------------
    if tiles_dual_kernel is None:
        tiles_dual_kernel = build_tiles_dual_3d_kernel()

    tiles_mixed_cp, tiles_dense_cp = build_roi_tiles_3d(
        roi_mask_flat, D, H, W, Tz, Ty, Tx, kernel=tiles_dual_kernel
    )

    tile_type = cp.zeros(nTiles, dtype=cp.int32)   # 0/1/2
    tile_roi  = cp.zeros(nTiles, dtype=cp.int32)   # 0/1

    if tiles_mixed_cp.size > 0:
        tz_m = tiles_mixed_cp[:, 0].astype(cp.int64)
        ty_m = tiles_mixed_cp[:, 1].astype(cp.int64)
        tx_m = tiles_mixed_cp[:, 2].astype(cp.int64)
        ids_m = (tz_m * nTilesY + ty_m) * nTilesX + tx_m
        tile_type[ids_m] = 1
        tile_roi[ids_m]  = 1

    if tiles_dense_cp.size > 0:
        tz_d = tiles_dense_cp[:, 0].astype(cp.int64)
        ty_d = tiles_dense_cp[:, 1].astype(cp.int64)
        tx_d = tiles_dense_cp[:, 2].astype(cp.int64)
        ids_d = (tz_d * nTilesY + ty_d) * nTilesX + tx_d
        tile_type[ids_d] = 2
        tile_roi[ids_d]  = 1

    threads_vox = 256
    blocks_vox = (nvox + threads_vox - 1) // threads_vox

    threads_tiles = 256
    blocks_tiles = (nTiles + threads_tiles - 1) // threads_tiles

    step_kernel_uses_active_list = False
    step_kernel_adaptive_active_list = False
    k_step_active = None
    k_step_full = None
    if roi_step_kernels is None:
        use_int_offset = (27 * nvox) <= 2147483647
        if verbose and (not use_int_offset):
            print("[ROI-JFA] WARNING: 27*nvox too large, fallback to long long kmax offset.")
        if use_active_list_step:
            k_step_active, _ = build_geodesic_roi_jfa_step_active_list_kernels_3d(use_int_offset=use_int_offset)
            k_step_full, _ = build_geodesic_roi_jfa_step_kernels_3d(use_int_offset=use_int_offset)
            k_step = k_step_active
            step_kernel_uses_active_list = True
            step_kernel_adaptive_active_list = True
        else:
            k_step, _ = build_geodesic_roi_jfa_step_kernels_3d(use_int_offset=use_int_offset)
    elif isinstance(roi_step_kernels, dict):
        mode = str(roi_step_kernels.get("mode", "")).lower().strip()
        if mode == "adaptive_active_list":
            k_step_active = roi_step_kernels["kernel"]
            k_step_full = roi_step_kernels.get("kernel_full")
            if k_step_full is None:
                use_int_offset = (27 * nvox) <= 2147483647
                k_step_full, _ = build_geodesic_roi_jfa_step_kernels_3d(use_int_offset=use_int_offset)
            k_step = k_step_active
            step_kernel_uses_active_list = True
            step_kernel_adaptive_active_list = True
        else:
            k_step = roi_step_kernels["kernel"]
            step_kernel_uses_active_list = mode == "active_list"
            k_step_active = k_step if step_kernel_uses_active_list else None
    else:
        k_step = roi_step_kernels[0]

    if active_tiles_kernel is None:
        active_tiles_kernel = build_active_tiles_kernel_3d()

    if active_fallback_kernel is None:
        active_fallback_kernel = build_fallback_full_roi_kernel_3d()

    apply_kernel_uses_active_list = False
    apply_kernel_adaptive_active_list = False
    apply_kernel_active = None
    apply_kernel_full = None
    if isinstance(apply_kernel, dict):
        mode = str(apply_kernel.get("mode", "")).lower().strip()
        if mode == "adaptive_active_list":
            apply_kernel_active = apply_kernel["kernel"]
            apply_kernel_full = apply_kernel.get("kernel_full")
            if apply_kernel_full is None:
                apply_kernel_full = build_apply_roi_tile_updates_kernel_3d()
            apply_kernel = apply_kernel_active
            apply_kernel_uses_active_list = True
            apply_kernel_adaptive_active_list = True
        else:
            apply_kernel_uses_active_list = mode == "active_list"
            apply_kernel = apply_kernel["kernel"]
            apply_kernel_active = apply_kernel if apply_kernel_uses_active_list else None
    elif apply_kernel is None:
        if step_kernel_uses_active_list:
            apply_kernel_active = build_apply_roi_tile_updates_active_list_kernel_3d()
            apply_kernel_full = build_apply_roi_tile_updates_kernel_3d() if step_kernel_adaptive_active_list else None
            apply_kernel = apply_kernel_active
            apply_kernel_uses_active_list = True
            apply_kernel_adaptive_active_list = bool(step_kernel_adaptive_active_list)
        else:
            apply_kernel = build_apply_roi_tile_updates_kernel_3d()

    roi_relax_kernel = build_roi_relax_jump1_packed_kernel_3d()

    if closure_kernel is None:
        closure_kernel = build_roi_closure_jump1_packed_kernel_3d()

    if relax_kernel is None:
        relax_kernel = build_local_relax_packed_kernel_3d()

    if verbose:
        roi_tiles_count = int(cp.count_nonzero(tile_roi).get())
        print(f"[ROI-JFA] ROI tiles: {roi_tiles_count} / {nTiles}")

    # ---------------------------
    # 3. tile flags: changed / unfinished / active
    # ---------------------------
    tile_changed_prev    = tile_roi.copy()
    tile_unfinished_prev = tile_roi.copy()

    tile_changed_next    = cp.zeros_like(tile_changed_prev)
    tile_unfinished_next = cp.zeros_like(tile_unfinished_prev)

    tile_active = cp.zeros(nTiles, dtype=cp.int32)
    any_active  = cp.zeros(1, dtype=cp.int32)

    # ---------------------------
    # 4. main loop (largest jump -> ... -> 2)
    # ---------------------------
    maxdim = int(max(D, H, W))
    jump = 1
    while (jump << 1) <= maxdim:
        jump <<= 1

    if max_refine_iters is None:
        max_refine_iters_cap = max(64, 2 * maxdim)
    else:
        max_refine_iters_cap = int(max_refine_iters)
        if max_refine_iters_cap < 0:
            max_refine_iters_cap = 0

    step = 0
    eps = float(relax_eps)

    active_step_records = []
    label_step_records  = []

    def _record_label_dist_from_state(state_u64):
        lbl = (state_u64 & cp.uint64(0xFFFFFFFF)).astype(cp.int32)
        du32 = (state_u64 >> cp.uint64(32)).astype(cp.uint32)
        d = du32.view(cp.float32)
        return lbl.copy(), d.copy()

    if dump_label_3d and viz_policy == "deferred" and return_records:
        lbl0, dist0 = _record_label_dist_from_state(state_state)
        label_step_records.append((0, 0, lbl0.copy(), dist0.copy()))

    SMALL_JUMP_FULL_ROI = 4

    while jump >= 2:
        step += 1

        if jump <= SMALL_JUMP_FULL_ROI:
            tile_active[...] = tile_roi
        else:
            tile_active.fill(0)
            any_active.fill(0)

            active_tiles_kernel(
                (blocks_tiles,),
                (threads_tiles,),
                (
                    tile_roi,
                    tile_changed_prev,
                    tile_unfinished_prev,
                    tile_active,
                    any_active,
                    int(D), int(H), int(W),
                    int(Tz), int(Ty), int(Tx),
                    int(nTilesZ), int(nTilesY), int(nTilesX),
                    int(jump),
                ),
            )

            active_fallback_kernel(
                (blocks_tiles,),
                (threads_tiles,),
                (
                    tile_roi,
                    tile_active,
                    any_active,
                    int(nTiles),
                ),
            )

        active_ids = None
        n_active = None
        if step_kernel_uses_active_list or apply_kernel_uses_active_list:
            active_ids = cp.where(tile_active != 0)[0].astype(cp.int32)
            n_active = int(active_ids.size)

        use_active_list_this_step = False
        if step_kernel_uses_active_list:
            if active_ids is None:
                active_ids = cp.where(tile_active != 0)[0].astype(cp.int32)
                n_active = int(active_ids.size)
            if step_kernel_adaptive_active_list and k_step_full is not None:
                active_fraction = float(n_active) / float(max(nTiles, 1))
                use_active_list_this_step = active_fraction <= active_list_fraction_threshold
            else:
                use_active_list_this_step = True

        if verbose:
            if n_active is None:
                n_active = int(cp.count_nonzero(tile_active).get())
            n_unfinished_prev = int(cp.count_nonzero(tile_unfinished_prev).get())
            n_changed_prev    = int(cp.count_nonzero(tile_changed_prev).get())
            print(f"[ROI-JFA] step {step}, jump={jump}, active tiles={n_active}, "
                  f"unfinished tiles={n_unfinished_prev}, changed_prev tiles={n_changed_prev}")

        if dump_active_tiles and viz_policy == "deferred" and return_records:
            if active_ids is None:
                active_ids = cp.where(tile_active != 0)[0].astype(cp.int32)
            active_step_records.append((int(step), int(jump), active_ids.copy()))

        if profile_gpu:
            s_evt = cp.cuda.Event()
            e_evt = cp.cuda.Event()
            s_evt.record()

        tile_changed_next.fill(0)
        tile_unfinished_next.fill(0)

        jump_k = int(int(jump).bit_length() - 1)
        full_write_mixed = 1 if (jump <= SMALL_JUMP_FULL_ROI) else 0

        if jump <= SMALL_JUMP_FULL_ROI:
            state_scratch[:] = state_state

        if use_active_list_this_step:
            if n_active > 0:
                (k_step_active or k_step)(
                    (int(n_active),),
                    (int(threads_vox),),
                    (
                        mask_flat,
                        roi_mask_flat,
                        tile_type,
                        tile_roi,
                        tile_active,
                        clearance_kmax27,
                        np.int32(nvox),

                        state_state,
                        state_scratch,

                        tile_changed_next,
                        tile_unfinished_next,

                        np.int32(D), np.int32(H), np.int32(W),
                        np.int32(Tz), np.int32(Ty), np.int32(Tx),
                        np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX),
                        np.int32(jump),
                        np.int32(jump_k),
                        np.float32(eps),
                        np.int32(changed_mode_id_base),
                        np.int32(manhattan_mode_id),
                        np.int32(full_write_mixed),

                        np.int32(pow2_decode),
                        np.int32(shift_TyTx), np.int32(shift_Tx),
                        np.int32(mask_TyTx),  np.int32(mask_Tx),

                        active_ids,
                        np.int32(n_active),
                    ),
                )
        else:
            (k_step_full or k_step)(
                (int(nTiles),),
                (int(threads_vox),),
                (
                    mask_flat,
                    roi_mask_flat,
                    tile_type,
                    tile_roi,
                    tile_active,
                    clearance_kmax27,
                    np.int32(nvox),

                    state_state,
                    state_scratch,

                    tile_changed_next,
                    tile_unfinished_next,

                    np.int32(D), np.int32(H), np.int32(W),
                    np.int32(Tz), np.int32(Ty), np.int32(Tx),
                    np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX),
                    np.int32(jump),
                    np.int32(jump_k),
                    np.float32(eps),
                    np.int32(changed_mode_id_base),
                    np.int32(manhattan_mode_id),
                    np.int32(full_write_mixed),

                    np.int32(pow2_decode),
                    np.int32(shift_TyTx), np.int32(shift_Tx),
                    np.int32(mask_TyTx),  np.int32(mask_Tx),
                ),
            )

        if jump <= SMALL_JUMP_FULL_ROI:
            if profile_gpu:
                e_evt.record()
                e_evt.synchronize()
                gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))
            state_state, state_scratch = state_scratch, state_state
        else:
            if use_active_list_this_step and apply_kernel_uses_active_list:
                if n_active > 0:
                    (apply_kernel_active or apply_kernel)(
                        (int(n_active),),
                        (int(threads_vox),),
                        (
                            roi_mask_flat,
                            tile_type,
                            tile_roi,
                            tile_active,
                            state_scratch,
                            state_state,
                            np.int32(D), np.int32(H), np.int32(W),
                            np.int32(Tz), np.int32(Ty), np.int32(Tx),
                            np.int32(nTilesY), np.int32(nTilesX),
                            np.int32(pow2_decode),
                            np.int32(shift_TyTx), np.int32(shift_Tx),
                            np.int32(mask_TyTx),  np.int32(mask_Tx),
                            active_ids,
                            np.int32(n_active),
                        ),
                    )
            else:
                (apply_kernel_full or apply_kernel)(
                    (int(nTiles),),
                    (int(threads_vox),),
                    (
                        roi_mask_flat,
                        tile_type,
                        tile_roi,
                        tile_active,
                        state_scratch,
                        state_state,
                        np.int32(D), np.int32(H), np.int32(W),
                        np.int32(Tz), np.int32(Ty), np.int32(Tx),
                        np.int32(nTilesY), np.int32(nTilesX),
                        np.int32(pow2_decode),
                        np.int32(shift_TyTx), np.int32(shift_Tx),
                        np.int32(mask_TyTx),  np.int32(mask_Tx),
                    ),
                )

            if profile_gpu:
                e_evt.record()
                e_evt.synchronize()
                gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))

        if dump_label_3d and viz_policy == "deferred" and return_records:
            lbl, dist = _record_label_dist_from_state(state_state)
            label_step_records.append((int(step), int(jump), lbl, dist))

        tile_changed_prev, tile_changed_next = tile_changed_next, tile_changed_prev
        tile_unfinished_prev, tile_unfinished_next = tile_unfinished_next, tile_unfinished_prev

        jump >>= 1

    # ===========================
    # [TIMING] end of JFA main stage (exclude closure / relax / unpack)
    # ===========================
    cp.cuda.Device().synchronize()
    t_jfa_end = time.time()
    ROI_JFA_LAST_TJFA_WALL = float(t_jfa_end - t_jfa_stage_start)

    # ---------------------------
    # 4.5 Closure timing starts here
    # ---------------------------
    t_close_start = time.time()
    did_close = False

    if max_refine_iters_cap > 0:
        if enable_closure:
            did_close = True
            if verbose:
                print(f"[ROI-JFA][C] closure start: max_refine_iters={max_refine_iters_cap}")

            any_changed = cp.zeros(1, dtype=cp.int32)
            any_unfinished = cp.zeros(1, dtype=cp.int32)

            for it in range(int(max_refine_iters_cap)):
                any_changed.fill(0)
                any_unfinished.fill(0)

                if profile_gpu:
                    s_evt = cp.cuda.Event()
                    e_evt = cp.cuda.Event()
                    s_evt.record()

                closure_kernel(
                    (int(blocks_vox),),
                    (int(threads_vox),),
                    (
                        mask_flat,
                        roi_mask_flat,
                        state_state,
                        state_scratch,
                        any_changed,
                        any_unfinished,
                        np.int32(D), np.int32(H), np.int32(W),
                        np.float32(eps),
                    ),
                )

                if profile_gpu:
                    e_evt.record()
                    e_evt.synchronize()
                    gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))

                state_state, state_scratch = state_scratch, state_state

                unfin = int(any_unfinished.get()[0])
                if unfin == 0:
                    if verbose:
                        print(f"[ROI-JFA][C] closure done in {it+1} iterations (no U left).")
                    break

                chg = int(any_changed.get()[0])
                if chg == 0:
                    msg = (
                        "[ROI-JFA][C] closure stalled: unfinished ROI voxels remain but no new voxel "
                        "can be assigned. This usually means: (1) some pore connected component has no seed, "
                        "or (2) ROI has a disconnected island that cannot reach any labeled voxel.\n"
                        "Fix: ensure at least one seed per pore CC, or check mask connectivity / seed placement."
                    )
                    raise RuntimeError(msg)
            else:
                msg = (
                    f"[ROI-JFA][C] closure did not finish within max_refine_iters={max_refine_iters_cap}. "
                    "Increase max_refine_iters, or check if some pore CC has no seed."
                )
                raise RuntimeError(msg)

            if dump_label_3d and viz_policy == "deferred" and return_records:
                lbl, dist = _record_label_dist_from_state(state_state)
                label_step_records.append((int(step + 1), int(1), lbl, dist))

        else:
            # closure disabled: keep original behavior (but your metrics-pre run should set max_refine_iters=0)
            if verbose:
                print("[ROI-JFA] closure disabled, fallback to single ROI relax (jump=1).")

            did_close = False

            if profile_gpu:
                s_evt = cp.cuda.Event()
                e_evt = cp.cuda.Event()
                s_evt.record()

            roi_relax_kernel(
                (int(blocks_vox),),
                (int(threads_vox),),
                (
                    mask_flat,
                    roi_mask_flat,
                    state_state,
                    state_scratch,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.float32(eps),
                ),
            )

            if profile_gpu:
                e_evt.record()
                e_evt.synchronize()
                gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))

            state_state, state_scratch = state_scratch, state_state

            if dump_label_3d and viz_policy == "deferred" and return_records:
                lbl, dist = _record_label_dist_from_state(state_state)
                label_step_records.append((int(step + 1), int(1), lbl, dist))
    else:
        if verbose:
            print("[ROI-JFA] max_refine_iters=0 -> skip closure/jump=1 stage (may leave U).")

    cp.cuda.Device().synchronize()
    t_close_end = time.time()
    t_close = (t_close_end - t_close_start) if did_close else 0.0
    ROI_JFA_LAST_TCLOSE_WALL = float(t_close)

    # ---------------------------
    # 5. post-processing: local relaxation (jump=1, whole fluid domain)
    # ---------------------------
    if n_relax_after is not None and int(n_relax_after) > 0:
        cp.cuda.Device().synchronize()
        t_relax_start = time.time()

        for _ in range(int(n_relax_after)):
            if profile_gpu:
                s_evt = cp.cuda.Event()
                e_evt = cp.cuda.Event()
                s_evt.record()

            relax_kernel(
                (int(blocks_vox),),
                (int(threads_vox),),
                (
                    mask_flat,
                    state_state,
                    state_scratch,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.float32(relax_eps),
                ),
            )

            if profile_gpu:
                e_evt.record()
                e_evt.synchronize()
                gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))

            state_state, state_scratch = state_scratch, state_state

        cp.cuda.Device().synchronize()
        t_relax_end = time.time()
        ROI_JFA_LAST_TRELAX_WALL = float(t_relax_end - t_relax_start)
    else:
        ROI_JFA_LAST_TRELAX_WALL = 0.0

    # ===========================
    # [TIMING] define TPRED as strict sum (exclude unpack/postprocess)
    # ===========================
    ROI_JFA_LAST_TPRED_WALL = float(
        ROI_JFA_LAST_TSTAMP_WALL +
        ROI_JFA_LAST_TJFA_WALL +
        ROI_JFA_LAST_TCLOSE_WALL +
        ROI_JFA_LAST_TRELAX_WALL
    )

    # ---------------------------
    # 6. unpack the output
    # ---------------------------
    label_flat = (state_state & cp.uint64(0xFFFFFFFF)).astype(cp.int32)
    dist_u32 = (state_state >> cp.uint64(32)).astype(cp.uint32)
    dist_flat = dist_u32.view(cp.float32)

    label_final = label_flat.reshape((D, H, W))
    dist_final  = dist_flat.reshape((D, H, W))

    solid = (mask_flat == 0).reshape((D, H, W))
    label_final = cp.where(solid, cp.int32(-1), label_final)
    dist_final  = cp.where(solid, cp.float32(1e20), dist_final)

    ROI_JFA_LAST_RECORD_TIME = float(record_time_total)
    ROI_JFA_LAST_GPU_TIME = float(gpu_ms_total) / 1000.0
    ROI_JFA_LAST_VIZ_TIME = 0.0

    if viz_policy == "deferred" and return_records:
        records = {
            "active_steps": active_step_records,
            "label_steps": label_step_records,
            "meta": {
                "D": int(D), "H": int(H), "W": int(W),
                "nTilesZ": int(nTilesZ), "nTilesY": int(nTilesY), "nTilesX": int(nTilesX),
                "tile_size": tuple(tile_size),
                "active_list_step": bool(step_kernel_uses_active_list),
                "active_list_adaptive": bool(step_kernel_adaptive_active_list),
                "active_list_fraction_threshold": float(active_list_fraction_threshold),
                "seeds": seeds_np,
                "dump_prefix": str(dump_prefix),
            }
        }
        return label_final, dist_final, tile_roi, roi_mask_flat, records

    return label_final, dist_final, tile_roi, roi_mask_flat



def geodesic_voronoi_roi_jfa_sparse_voxels(
    mask,
    seeds,
    tile_size=(8, 8, 16),
    delta_r=1.0,
    eta_max=0.8,
    r_tile=1,
    enable_stamping=True,
    verbose=False,
    viz_policy="none",
    n_relax_after=1,
    relax_eps=1e-6,
    profile_gpu=False,
    return_records=False,
    max_refine_iters=460,
    clearance_kmax27=None,
    los_kmax_init_kernel=None,
    los_kmax_update_kernel=None,
    stamping_kernel=None,
    use_active_list_step=True,
    **_ignored_kwargs,
):
    """
    Experimental sparse-voxel ROI-JFA path.

    The mathematical ROI certificate is unchanged; only the execution set is
    compacted from ROI tiles to ROI voxel ids.  This is useful when C2/tight
    certificates remove many voxels but leave most coarse tiles partially active.
    """
    import cupy as cp
    import numpy as np
    import time

    global ROI_JFA_LAST_VIZ_TIME, ROI_JFA_LAST_GPU_TIME, ROI_JFA_LAST_RECORD_TIME
    global ROI_JFA_LAST_TSTAMP_WALL, ROI_JFA_LAST_TJFA_WALL, ROI_JFA_LAST_TCLOSE_WALL, ROI_JFA_LAST_TRELAX_WALL, ROI_JFA_LAST_TPRED_WALL

    ROI_JFA_LAST_VIZ_TIME = 0.0
    ROI_JFA_LAST_GPU_TIME = 0.0
    ROI_JFA_LAST_RECORD_TIME = 0.0
    ROI_JFA_LAST_TSTAMP_WALL = 0.0
    ROI_JFA_LAST_TJFA_WALL = 0.0
    ROI_JFA_LAST_TCLOSE_WALL = 0.0
    ROI_JFA_LAST_TRELAX_WALL = 0.0
    ROI_JFA_LAST_TPRED_WALL = 0.0

    mask_cp, (D, H, W), mask_flat = _as_cupy_mask_u8(mask)
    nvox = int(D * H * W)
    if isinstance(seeds, cp.ndarray):
        seeds_cp = seeds.astype(cp.int32, copy=False)
    else:
        seeds_cp = cp.asarray(np.asarray(seeds, dtype=np.int32), dtype=cp.int32)

    if clearance_kmax27 is None:
        maxdim = int(max(D, H, W))
        max_k = int(maxdim.bit_length() - 1)
        if los_kmax_init_kernel is None:
            los_kmax_init_kernel = build_los_kmax_init_kernel_3d()
        if los_kmax_update_kernel is None:
            los_kmax_update_kernel = build_los_kmax_update_kernel_3d()
        clearance_kmax27 = precompute_los_kmax_27dirs_3d(
            mask_flat,
            D, H, W,
            max_k=max_k,
            init_kernel=los_kmax_init_kernel,
            update_kernel=los_kmax_update_kernel,
            verbose=False,
        )

    t0_stamp = time.time()
    if enable_stamping:
        label_stamp_flat, dist_stamp_flat, roi_mask_cp, radii_cp, state_state = perform_seed_stamping(
            mask_cp,
            seeds_cp,
            delta_r=delta_r,
            stamping_kernel=stamping_kernel,
            parallel=True,
            return_state=True,
            clearance_kmax27=clearance_kmax27,
        )
        roi_mask_flat = roi_mask_cp.astype(cp.uint8).ravel()
    else:
        roi_mask_flat = mask_flat.copy().astype(cp.uint8)
        INF_BITS = np.frombuffer(np.float32(1e20).tobytes(), dtype=np.uint32)[0]
        PACK_INF_NEG1 = (int(INF_BITS) << 32) | 0xFFFFFFFF
        state_state = cp.full(nvox, np.uint64(PACK_INF_NEG1), dtype=cp.uint64)
        set_seeds_kernel = build_set_seeds_packed_kernel()
        seeds_flat_cp = seeds_cp.reshape(-1)
        n_seeds = int(seeds_cp.shape[0])
        threads = 256
        blocks = (n_seeds + threads - 1) // threads
        set_seeds_kernel(
            (int(blocks),),
            (int(threads),),
            (mask_flat, state_state, seeds_flat_cp, np.int32(n_seeds), np.int32(D), np.int32(H), np.int32(W)),
        )
    cp.cuda.Device().synchronize()
    ROI_JFA_LAST_TSTAMP_WALL = float(time.time() - t0_stamp)

    Tz, Ty, Tx = [int(v) for v in tile_size]
    nTilesZ = (D + Tz - 1) // Tz
    nTilesY = (H + Ty - 1) // Ty
    nTilesX = (W + Tx - 1) // Tx
    nTiles = int(nTilesZ * nTilesY * nTilesX)
    tiles_mixed_cp, tiles_dense_cp = build_roi_tiles_3d(roi_mask_flat, D, H, W, Tz, Ty, Tx)
    tile_roi = cp.zeros(nTiles, dtype=cp.int32)
    if tiles_mixed_cp.size > 0:
        ids = (tiles_mixed_cp[:, 0].astype(cp.int64) * nTilesY + tiles_mixed_cp[:, 1].astype(cp.int64)) * nTilesX + tiles_mixed_cp[:, 2].astype(cp.int64)
        tile_roi[ids] = 1
    if tiles_dense_cp.size > 0:
        ids = (tiles_dense_cp[:, 0].astype(cp.int64) * nTilesY + tiles_dense_cp[:, 1].astype(cp.int64)) * nTilesX + tiles_dense_cp[:, 2].astype(cp.int64)
        tile_roi[ids] = 1

    roi_ids = cp.where(roi_mask_flat != 0)[0].astype(cp.int32)
    n_roi = int(roi_ids.size)
    state_scratch = state_state.copy()
    k_sparse = build_sparse_roi_jfa_step_kernel_3d()
    threads = 256
    blocks_roi = (n_roi + threads - 1) // threads

    maxdim = int(max(D, H, W))
    jump = 1
    while (jump << 1) <= maxdim:
        jump <<= 1

    gpu_ms_total = 0.0
    t_jfa0 = time.time()
    while jump >= 2:
        state_scratch[:] = state_state
        jump_k = int(int(jump).bit_length() - 1)
        if profile_gpu:
            s_evt = cp.cuda.Event()
            e_evt = cp.cuda.Event()
            s_evt.record()
        if n_roi > 0:
            k_sparse(
                (int(blocks_roi),),
                (int(threads),),
                (
                    mask_flat,
                    clearance_kmax27,
                    roi_ids,
                    np.int32(n_roi),
                    np.int32(nvox),
                    state_state,
                    state_scratch,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.int32(jump),
                    np.int32(jump_k),
                    np.float32(float(relax_eps)),
                ),
            )
        if profile_gpu:
            e_evt.record()
            e_evt.synchronize()
            gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))
        state_state, state_scratch = state_scratch, state_state
        jump >>= 1
    cp.cuda.Device().synchronize()
    ROI_JFA_LAST_TJFA_WALL = float(time.time() - t_jfa0)

    closure_kernel = build_roi_closure_jump1_packed_kernel_3d()
    blocks_vox = (nvox + threads - 1) // threads
    max_refine_iters_cap = int(max_refine_iters) if max_refine_iters is not None else max(64, 2 * maxdim)
    t_close0 = time.time()
    did_close = False
    if max_refine_iters_cap > 0:
        did_close = True
        any_changed = cp.zeros(1, dtype=cp.int32)
        any_unfinished = cp.zeros(1, dtype=cp.int32)
        for _ in range(max_refine_iters_cap):
            any_changed.fill(0)
            any_unfinished.fill(0)
            closure_kernel(
                (int(blocks_vox),),
                (int(threads),),
                (
                    mask_flat,
                    roi_mask_flat,
                    state_state,
                    state_scratch,
                    any_changed,
                    any_unfinished,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.float32(float(relax_eps)),
                ),
            )
            state_state, state_scratch = state_scratch, state_state
            if int(any_unfinished.get()[0]) == 0:
                break
            if int(any_changed.get()[0]) == 0:
                raise RuntimeError("[sparse ROI-JFA] closure stalled with unfinished ROI voxels.")
        else:
            raise RuntimeError("[sparse ROI-JFA] closure did not finish within max_refine_iters.")
    cp.cuda.Device().synchronize()
    ROI_JFA_LAST_TCLOSE_WALL = float(time.time() - t_close0) if did_close else 0.0

    if n_relax_after is not None and int(n_relax_after) > 0:
        relax_kernel = build_local_relax_packed_kernel_3d()
        t_relax0 = time.time()
        for _ in range(int(n_relax_after)):
            relax_kernel(
                (int(blocks_vox),),
                (int(threads),),
                (
                    mask_flat,
                    state_state,
                    state_scratch,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.float32(float(relax_eps)),
                ),
            )
            state_state, state_scratch = state_scratch, state_state
        cp.cuda.Device().synchronize()
        ROI_JFA_LAST_TRELAX_WALL = float(time.time() - t_relax0)

    ROI_JFA_LAST_TPRED_WALL = float(
        ROI_JFA_LAST_TSTAMP_WALL + ROI_JFA_LAST_TJFA_WALL + ROI_JFA_LAST_TCLOSE_WALL + ROI_JFA_LAST_TRELAX_WALL
    )
    ROI_JFA_LAST_GPU_TIME = float(gpu_ms_total) / 1000.0

    label_flat = (state_state & cp.uint64(0xFFFFFFFF)).astype(cp.int32)
    dist_u32 = (state_state >> cp.uint64(32)).astype(cp.uint32)
    dist_flat = dist_u32.view(cp.float32)
    label_final = label_flat.reshape((D, H, W))
    dist_final = dist_flat.reshape((D, H, W))
    solid = (mask_flat == 0).reshape((D, H, W))
    label_final = cp.where(solid, cp.int32(-1), label_final)
    dist_final = cp.where(solid, cp.float32(1e20), dist_final)
    return label_final, dist_final, tile_roi, roi_mask_flat

























# ============================================================
# 6. full OA-JFA without the ROI optimisation (for comparison)
# ============================================================

def build_oajfa_kernel():
    code = r'''
    extern "C" __global__
    void oajfa_step(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ label_in,
        const float* __restrict__ dist_in,
        int* __restrict__ label_out,
        float* __restrict__ dist_out,
        const int D, const int H, const int W,
        const int jump
    )
    {
        const int nvox = D * H * W;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nvox) return;

        if (!mask[idx]) {
            label_out[idx] = label_in[idx];
            dist_out[idx] = dist_in[idx];
            return;
        }

        const int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int cur_label = label_in[idx];
        float cur_dist = dist_in[idx];

        label_out[idx] = cur_label;
        dist_out[idx] = cur_dist;

        float best_dist = cur_dist;
        int best_label = cur_label;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + jump * dz;
                    int jy = y + jump * dy;
                    int jx = x + jump * dx;

                    if (jz < 0 || jz >= D || jy < 0 || jy >= H || jx < 0 || jx >= W) {
                        continue;
                    }

                    int j_idx = jz * HW + jy * W + jx;
                    if (!mask[j_idx]) continue;

                    int neigh_label = label_in[j_idx];
                    if (neigh_label < 0) continue;

                    bool blocked = false;
                    for (int t = 1; t < jump; ++t) {
                        int kz = z + t * dz;
                        int ky = y + t * dy;
                        int kx = x + t * dx;

                        if (kz < 0 || kz >= D || ky < 0 || ky >= H || kx < 0 || kx >= W) {
                            blocked = true;
                            break;
                        }

                        int k_idx = kz * HW + ky * W + kx;
                        if (!mask[k_idx]) {
                            blocked = true;
                            break;
                        }
                    }
                    if (blocked) continue;

                    int dz_tot = jz - z;
                    int dy_tot = jy - y;
                    int dx_tot = jx - x;
                    float step = sqrtf((float)(dz_tot*dz_tot + dy_tot*dy_tot + dx_tot*dx_tot));
                    float cand_dist = dist_in[j_idx] + step;

                    if (cand_dist < best_dist) {
                        best_dist = cand_dist;
                        best_label = neigh_label;
                    }
                }
            }
        }

        if (best_label != cur_label || best_dist < cur_dist) {
            label_out[idx] = best_label;
            dist_out[idx] = best_dist;
        }
    }
    '''
    return _device_cached_rawkernel(
        build_oajfa_kernel,
        code,
        "oajfa_step",
    )

def build_local_relax_kernel():
    """
    One local relaxation (jump = 1) over the 3x3x3 neighbourhood,
    used for a little refinement after OA-JFA / ROI-JFA to improve accuracy.

    micro-optimisation:
      - no sqrtf() in the inner loop; nnz-based step {1, sqrt2, sqrt3}
    """
    code = r'''
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

    extern "C" __global__
    void geodesic_local_relax_step(
        const unsigned char* __restrict__ mask,
        const int*   __restrict__ label_in,
        const float* __restrict__ dist_in,
        int*   __restrict__ label_out,
        float* __restrict__ dist_out,
        const int D, const int H, const int W,
        const float eps
    )
    {
        const int nvox = D * H * W;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nvox) return;

        // solid: copy through
        if (!mask[idx]) {
            label_out[idx] = label_in[idx];
            dist_out[idx]  = dist_in[idx];
            return;
        }

        const int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        int   cur_label = label_in[idx];
        float cur_dist  = dist_in[idx];

        int   best_label = cur_label;
        float best_dist  = cur_dist;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        // 3x3x3 neighbourhood, jump = 1
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int jz = z + dz;
                    int jy = y + dy;
                    int jx = x + dx;

                    if (jz < 0 || jz >= D ||
                        jy < 0 || jy >= H ||
                        jx < 0 || jx >= W) {
                        continue;
                    }

                    int j_idx = jz * HW + jy * W + jx;
                    if (!mask[j_idx]) continue;

                    int neigh_label = label_in[j_idx];
                    if (neigh_label < 0) continue;

                    // remove sqrtf: step depends only on nnz for (-1,0,1) offsets
                    int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                    float step = (nnz == 1) ? 1.0f
                               : (nnz == 2) ? SQRT2
                                            : SQRT3;

                    float cand_dist = dist_in[j_idx] + step;

                    // use eps as a guard against jitter near floating-point error
                    if (cand_dist + eps < best_dist) {
                        best_dist  = cand_dist;
                        best_label = neigh_label;
                    }
                }
            }
        }

        label_out[idx] = best_label;
        dist_out[idx]  = best_dist;
    }
    ''';
    return _device_cached_rawkernel(
        build_local_relax_kernel,
        code,
        "geodesic_local_relax_step",
    )


def geodesic_voronoi_oajfa_cuda(
    mask,
    seeds,
    kernel=None,
    n_relax=1,
    relax_eps=1e-6,
    dump_per_step=False,
    dump_prefix="oajfa",
    profile_gpu=False,
):
    """
    Geodesic Voronoi by full OA-JFA (obstacle-aware JFA) + a few local relaxations.

    Seed injection:
      - one vectorised scatter on the GPU writes (label=seed_id, dist=0) for all seeds
        (no per-seed Python loop, no .tolist())
      - seeds outside the domain or in the solid are ignored (not written)

    profile_gpu=True: the CUDA event time of the JFA main loop + relaxation goes to OAJFA_LAST_GPU_TIME (s)
    """
    import numpy as np
    import cupy as cp

    global OAJFA_LAST_GPU_TIME
    OAJFA_LAST_GPU_TIME = 0.0
    gpu_ms_total = 0.0

    if kernel is None:
        kernel = build_oajfa_kernel()

    # ------------------------------------------------------------
    # mask: backend-aware (cp.asarray would copy a cupy mask again)
    # ------------------------------------------------------------
    if isinstance(mask, cp.ndarray):
        mask_cp = mask
        if mask_cp.dtype != cp.uint8:
            mask_cp = mask_cp.astype(cp.uint8, copy=False)
    else:
        mask_cp = cp.asarray(mask, dtype=cp.uint8)

    D, H, W = map(int, mask_cp.shape)
    HW = int(H * W)
    nvox = int(D * H * W)
    mask_flat = mask_cp.ravel()

    # ------------------------------------------------------------
    # seeds: backend-aware
    # ------------------------------------------------------------
    if isinstance(seeds, cp.ndarray):
        seeds_cp = seeds.astype(cp.int32, copy=False)
    else:
        seeds_cp = cp.asarray(np.asarray(seeds, dtype=np.int32), dtype=cp.int32)

    if seeds_cp.ndim != 2 or seeds_cp.shape[1] != 3:
        raise ValueError("seeds must be (N,3) with (z,y,x).")

    n_seeds = int(seeds_cp.shape[0])
    if n_seeds <= 0:
        # no seeds: return all -1/INF (same semantics)
        label = cp.full((D, H, W), -1, dtype=cp.int32)
        dist  = cp.full((D, H, W), cp.float32(1e20), dtype=cp.float32)
        return label, dist

    # ------------------------------------------------------------
    # init label/dist
    # ------------------------------------------------------------
    INF = cp.float32(1e20)
    label_in = cp.full(nvox, -1, dtype=cp.int32)
    dist_in  = cp.full(nvox, INF, dtype=cp.float32)

    # ------------------------------------------------------------
    # vectorized seed injection (skip out-of-bounds / solid)
    # ------------------------------------------------------------
    z = seeds_cp[:, 0].astype(cp.int64, copy=False)
    y = seeds_cp[:, 1].astype(cp.int64, copy=False)
    x = seeds_cp[:, 2].astype(cp.int64, copy=False)

    inb = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)

    if bool(cp.any(inb).item()):
        idx_lin = z * cp.int64(HW) + y * cp.int64(W) + x  # int64
        idx_valid = idx_lin[inb]
        seed_ids_valid = cp.arange(n_seeds, dtype=cp.int32)[inb]

        # only keep seeds in fluid voxels
        in_fluid = (mask_flat[idx_valid] != 0)
        idx_final = idx_valid[in_fluid]
        seed_ids_final = seed_ids_valid[in_fluid]

        if int(idx_final.size) > 0:
            label_in[idx_final] = seed_ids_final
            dist_in[idx_final]  = cp.float32(0.0)

    # work buffers
    label_out = label_in.copy()
    dist_out  = dist_in.copy()

    threads_per_block = 256
    blocks = (nvox + threads_per_block - 1) // threads_per_block

    # ------------------------------------------------------------
    # main OA-JFA loop (unchanged)
    # ------------------------------------------------------------
    maxdim = int(max(D, H, W))
    jump = 1
    while (jump << 1) <= maxdim:
        jump <<= 1

    step = 0
    while jump >= 1:
        step += 1
        jump_cur = jump

        if profile_gpu:
            s_evt = cp.cuda.Event()
            e_evt = cp.cuda.Event()
            s_evt.record()

        kernel(
            (int(blocks),),
            (int(threads_per_block),),
            (mask_flat, label_in, dist_in, label_out, dist_out, np.int32(D), np.int32(H), np.int32(W), np.int32(jump_cur)),
        )

        if profile_gpu:
            e_evt.record()
            e_evt.synchronize()
            gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))
        else:
            cp.cuda.Device().synchronize()

        if dump_per_step:
            label_step = label_out.reshape((D, H, W))
            dist_step  = dist_out.reshape((D, H, W))
            html_path = f"{dump_prefix}_step{step}_jump{jump_cur}_3d.html"
            visualize_exact_geodesic_3d(
                mask_cp,  # use mask_cp directly
                label_step,
                dist_step,
                seeds_cp,
                html_path=html_path,
                color_mode="label",
                smoothing_sigma=1.0,
                mesh_simplify_factor=2,
            )

        label_in, label_out = label_out, label_in
        dist_in,  dist_out  = dist_out,  dist_in
        jump >>= 1

    # ------------------------------------------------------------
    # local relax (unchanged)
    # ------------------------------------------------------------
    if n_relax is not None and int(n_relax) > 0:
        relax_kernel = build_local_relax_kernel()
        for _ in range(int(n_relax)):

            if profile_gpu:
                s_evt = cp.cuda.Event()
                e_evt = cp.cuda.Event()
                s_evt.record()

            relax_kernel(
                (int(blocks),),
                (int(threads_per_block),),
                (
                    mask_flat,
                    label_in, dist_in,
                    label_out, dist_out,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.float32(float(relax_eps)),
                ),
            )

            if profile_gpu:
                e_evt.record()
                e_evt.synchronize()
                gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))
            else:
                cp.cuda.Device().synchronize()

            label_in, label_out = label_out, label_in
            dist_in,  dist_out  = dist_out,  dist_in

    if profile_gpu:
        OAJFA_LAST_GPU_TIME = float(gpu_ms_total) / 1000.0

    label = label_in.reshape((D, H, W))
    dist  = dist_in.reshape((D, H, W))
    return label, dist








# ============================================================
# 7. Test cases
# ============================================================
def make_thin_wall_case(
    D=64,
    H=64,
    W=96,
    n_seeds=40,
    seed_mode="two_sides_random",
    seed_random_state=0,
    wall_thickness=2,
    wall_x=None,
    gate_size=(12, 12),          # (gate_size_z, gate_size_y) -- note the order (z, y)
    gate_center=None,            # (zc, yc)
    seed_margin_from_wall=4,     # keep seeds away from the wall so that seeds across a wall do not give a stamping radius near 0
    seed_near_wall_band=4,       # used when seed_mode="two_sides_near_wall"
):
    """
    Case 1: Thin wall (solid slab) + single gate (rectangular opening), 3D voxel domain.

    Coordinates follow your code convention: (z, y, x).
      - mask[z, y, x] = True  -> fluid
      - mask[z, y, x] = False -> solid

    Geometry:
      - Start with a fully fluid box.
      - Insert a solid wall slab normal to x, with thickness=wall_thickness.
      - Carve a single rectangular gate in the wall (only connection between two chambers).

    Seeds:
      - "random": sample uniformly from all fluid voxels.
      - "two_sides_random": sample ~half seeds from left chamber and ~half from right chamber.
      - "two_sides_near_wall": sample from bands near the wall on both sides (stress test; may reduce stamping radii).

    Returns:
      mask: (D,H,W) bool
      seeds: (n_seeds,3) int64
    """
    import numpy as np

    D = int(D); H = int(H)
    if W is None:
        # a stable, moderate default width so that the exact solver stays fast
        W = int(round(1.5 * max(D, H)))
    W = int(W)

    if D <= 2 or H <= 2 or W <= 4:
        raise ValueError("Domain too small for a thin-wall case.")

    wall_thickness = int(wall_thickness)
    if wall_thickness < 1:
        raise ValueError("wall_thickness must be >= 1")
    if wall_thickness >= W - 2:
        raise ValueError("wall_thickness too large: need space on both sides of the wall.")

    # ----------------------------
    # 1) Build mask (fluid everywhere)
    # ----------------------------
    mask = np.ones((D, H, W), dtype=bool)

    # ----------------------------
    # 2) Insert thin wall at x ~ W/2
    # ----------------------------
    if wall_x is None:
        wall_x = W // 2
    wall_x = int(wall_x)

    # Place a slab [x0, x1) and ensure both chambers non-empty: x0>=1 and x1<=W-1
    x0 = wall_x - (wall_thickness // 2)
    x0 = max(1, min(W - wall_thickness - 1, x0))
    x1 = x0 + wall_thickness

    mask[:, :, x0:x1] = False  # solid wall

    # ----------------------------
    # 3) Carve a single gate (rectangular opening)
    # gate_size = (gZ, gY) in (z,y) directions
    # ----------------------------
    gZ, gY = gate_size
    gZ = int(gZ); gY = int(gY)
    if gZ < 1 or gY < 1:
        raise ValueError("gate_size must be positive.")
    gZ = min(gZ, D)
    gY = min(gY, H)

    if gate_center is None:
        gate_center = (D // 2, H // 2)  # (zc, yc)
    zc, yc = [int(v) for v in gate_center]

    z0g = max(0, zc - gZ // 2)
    z1g = min(D, z0g + gZ)
    y0g = max(0, yc - gY // 2)
    y1g = min(H, y0g + gY)

    # Gate exists through the full wall thickness (x0:x1)
    mask[z0g:z1g, y0g:y1g, x0:x1] = True

    # ----------------------------
    # 4) Sample seeds
    # ----------------------------
    rng = np.random.RandomState(int(seed_random_state))
    fluid_idx = np.argwhere(mask)  # (N,3) in (z,y,x)

    if fluid_idx.shape[0] < n_seeds:
        raise ValueError("Not enough fluid voxels to place seeds.")

    n_seeds = int(n_seeds)
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive.")

    seed_mode = str(seed_mode).lower().strip()

    if seed_mode == "random":
        chosen = rng.choice(fluid_idx.shape[0], size=n_seeds, replace=False)
        seeds = fluid_idx[chosen]

    elif seed_mode in ("two_sides_random", "balanced"):
        margin = int(max(seed_margin_from_wall, 0))

        # left chamber: x <= x0-1-margin
        # right chamber: x >= x1+margin
        left_max_x = x0 - 1 - margin
        right_min_x = x1 + margin

        if left_max_x < 0 or right_min_x >= W:
            raise ValueError(
                "seed_margin_from_wall too large: no room to place seeds on one side."
            )

        left_idx = fluid_idx[fluid_idx[:, 2] <= left_max_x]
        right_idx = fluid_idx[fluid_idx[:, 2] >= right_min_x]

        n_left = n_seeds // 2
        n_right = n_seeds - n_left

        if left_idx.shape[0] < n_left or right_idx.shape[0] < n_right:
            raise ValueError(
                f"Not enough candidates for balanced seeding. "
                f"left={left_idx.shape[0]} (need {n_left}), "
                f"right={right_idx.shape[0]} (need {n_right})."
            )

        cL = rng.choice(left_idx.shape[0], size=n_left, replace=False)
        cR = rng.choice(right_idx.shape[0], size=n_right, replace=False)

        seeds = np.vstack([left_idx[cL], right_idx[cR]])
        rng.shuffle(seeds)

    elif seed_mode in ("two_sides_near_wall", "near_wall"):
        # Stress test: seeds close to the wall on both sides
        band = int(max(seed_near_wall_band, 1))

        # left band: x in [max(0,x0-band), x0-1]
        # right band: x in [x1, min(W-1, x1+band-1)]
        left_lo = max(0, x0 - band)
        left_hi = x0 - 1
        right_lo = x1
        right_hi = min(W - 1, x1 + band - 1)

        left_idx = fluid_idx[(fluid_idx[:, 2] >= left_lo) & (fluid_idx[:, 2] <= left_hi)]
        right_idx = fluid_idx[(fluid_idx[:, 2] >= right_lo) & (fluid_idx[:, 2] <= right_hi)]

        n_left = n_seeds // 2
        n_right = n_seeds - n_left

        if left_idx.shape[0] < n_left or right_idx.shape[0] < n_right:
            raise ValueError(
                f"Not enough near-wall candidates. "
                f"left={left_idx.shape[0]} (need {n_left}), "
                f"right={right_idx.shape[0]} (need {n_right})."
            )

        cL = rng.choice(left_idx.shape[0], size=n_left, replace=False)
        cR = rng.choice(right_idx.shape[0], size=n_right, replace=False)

        seeds = np.vstack([left_idx[cL], right_idx[cR]])
        rng.shuffle(seeds)

    else:
        raise ValueError(
            "Unknown seed_mode. Use 'random', 'two_sides_random', or 'two_sides_near_wall'."
        )

    return mask, np.asarray(seeds, dtype=np.int64)



def make_sinusoidal_channel_case(
    D=64,
    H=64,
    W=None,
    n_seeds=8,
    seed_mode="centerline_uniform",
    seed_random_state=0,
    L_phys=3.6,
    H_max_phys=1.0,
    H_min_phys=0.4,
    n_cycles=3,
):
    if W is None:
        base = min(D, H)
        W = int(round(L_phys / H_max_phys * base))

    z_idx = np.arange(D)[:, None, None]
    y_idx = np.arange(H)[None, :, None]
    x_idx = np.arange(W)[None, None, :]

    x_phys = (x_idx + 0.5) / W * L_phys
    y_phys = ((y_idx + 0.5) / H - 0.5) * H_max_phys
    z_phys = ((z_idx + 0.5) / D - 0.5) * H_max_phys

    rho = np.sqrt(y_phys**2 + z_phys**2)

    h_center = 0.5 * (H_max_phys + H_min_phys)
    amp = 0.5 * (H_max_phys - H_min_phys)
    phase = 2.0 * np.pi * n_cycles * x_phys / L_phys
    h_x = h_center + amp * np.sin(phase)

    r_x = 0.5 * h_x

    mask = rho <= r_x

    if seed_mode == "random":
        rng = np.random.RandomState(seed_random_state)
        fluid_indices = np.argwhere(mask)
        if n_seeds > fluid_indices.shape[0]:
            raise ValueError("too few fluid voxels to place that many seeds")
        chosen = rng.choice(fluid_indices.shape[0], size=n_seeds, replace=False)
        seeds = fluid_indices[chosen]

    elif seed_mode == "centerline_uniform":
        zc = D // 2
        yc = H // 2
        xs = np.linspace(0.1, 0.9, n_seeds) * (W - 1)
        seeds = []
        for x_f in xs:
            x = int(round(x_f))
            if mask[zc, yc, x]:
                seeds.append([zc, yc, x])
            else:
                found = False
                for dz in range(-2, 3):
                    for dy in range(-2, 3):
                        zz = np.clip(zc + dz, 0, D - 1)
                        yy = np.clip(yc + dy, 0, H - 1)
                        if mask[zz, yy, x]:
                            seeds.append([zz, yy, x])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise RuntimeError(f"no fluid voxel found near x={x} to place a seed")
        seeds = np.asarray(seeds, dtype=np.int64)

    else:
        raise ValueError(f"unknown seed_mode: {seed_mode}")

    return mask, seeds

def render_roi_jfa_records(
    mask,
    records,
    dump_active_tiles=True,
    dump_label_3d=True,
    tile_size=(8, 8, 8),
    axis_for_2d="y",
    index_for_2d=None,
    smoothing_sigma=1.0,
    mesh_simplify_factor=2,
):
    """
    Render the records to HTML/PNG files:
      - active tiles: 3D boxes + 2D slice
      - label field: one 3D label HTML per step (visualize_exact_geodesic_3d)
    """
    import numpy as np
    import time

    global ROI_JFA_LAST_VIZ_TIME
    t0 = time.time()

    meta = records.get("meta", {})
    D = int(meta.get("D", mask.shape[0]))
    H = int(meta.get("H", mask.shape[1]))
    W = int(meta.get("W", mask.shape[2]))
    nTilesY = int(meta.get("nTilesY"))
    nTilesX = int(meta.get("nTilesX"))
    dump_prefix = str(meta.get("dump_prefix", "roi_tiles"))
    seeds_np = np.asarray(meta.get("seeds", np.empty((0, 3), dtype=np.int64)), dtype=np.int64)

    active_steps = records.get("active_steps", [])
    label_steps  = records.get("label_steps", [])

    print(f"[render] active_steps = {len(active_steps)}, label_steps = {len(label_steps)}")
    if len(label_steps) == 0:
        print("[render] WARNING: label_steps is empty. "
              "Check geodesic_voronoi_roi_jfa: dump_label_3d=True, viz_policy='deferred', return_records=True.")

    # default 2D slice index
    if index_for_2d is None:
        if axis_for_2d.lower() == "y":
            index_for_2d = H // 2
        elif axis_for_2d.lower() == "z":
            index_for_2d = D // 2
        else:
            index_for_2d = W // 2

    # 1) active tiles
    if dump_active_tiles:
        for (st, jp, active_ids_cp) in active_steps:
            active_ids_np = active_ids_cp.get()

            if active_ids_np.size > 0:
                tz_vis = active_ids_np // (nTilesY * nTilesX)
                rem_vis = active_ids_np - tz_vis * (nTilesY * nTilesX)
                ty_vis = rem_vis // nTilesX
                tx_vis = rem_vis % nTilesX
                tiles_coords_vis = np.stack([tz_vis, ty_vis, tx_vis], axis=1).astype(np.int32)
            else:
                tiles_coords_vis = np.empty((0, 3), dtype=np.int32)

            html_path = f"{dump_prefix}_step{st}_jump{jp}_3d.html"
            visualize_active_tiles_3d_boxes(
                mask,
                tiles_coords_vis,
                tile_size=tile_size,
                html_path=html_path,
                title=f"Active tiles (step={st}, jump={jp})"
            )

            png_path = f"{dump_prefix}_step{st}_jump{jp}_2d.png"
            visualize_active_tiles_slice(
                mask,
                tiles_coords_vis,
                tile_size=tile_size,
                axis=axis_for_2d,
                index=index_for_2d,
                figsize=(8, 4),
                save_path=png_path,
                title=f"Active tiles slice (step={st}, jump={jp})",
            )

    # 2) label field per step (3D HTML)
    if dump_label_3d:
        # output sorted by step/jump (the step-0 stamping comes first)
        label_steps_sorted = sorted(label_steps, key=lambda t: (int(t[0]), int(t[1])))

        for (st, jp, label_cp_flat, dist_cp_flat) in label_steps_sorted:
            label_step = label_cp_flat.reshape((D, H, W))
            dist_step  = dist_cp_flat.reshape((D, H, W))

            html_path_lbl = f"{dump_prefix}_labels_step{st}_jump{jp}_3d.html"
            visualize_exact_geodesic_3d(
                mask,
                label_step,
                dist_step,
                seeds_np,
                html_path=html_path_lbl,
                color_mode="label",
                smoothing_sigma=float(smoothing_sigma),
                mesh_simplify_factor=int(mesh_simplify_factor),
            )

    ROI_JFA_LAST_VIZ_TIME = float(time.time() - t0)
    print(f"[render] done, viz_time = {ROI_JFA_LAST_VIZ_TIME:.3f} s")
    return ROI_JFA_LAST_VIZ_TIME


def build_euclidean_voronoi_clipping_kernel():
    """
    Euclidean Voronoi + clipping baseline (M1) on GPU.

    Under Windows + NVRTC, <math.h> must not be included (NVRTC cannot find the host standard headers).
    No header is included here; sqrtf (CUDA device intrinsic) is used directly.
    """
    import cupy as cp

    dev = int(cp.cuda.runtime.getDevice())
    cache = getattr(build_euclidean_voronoi_clipping_kernel, "_cache", {})
    if dev in cache:
        return cache[dev]

    code = r'''
    extern "C" __global__
    void euclidean_voronoi_clipped(
        const unsigned char* __restrict__ mask,   // [nvox], 1=fluid, 0=solid
        const int* __restrict__ seeds,            // [3*n_seeds], (z,y,x) per seed
        const int n_seeds,
        int* __restrict__ label_out,              // [nvox]
        float* __restrict__ dist_out,             // [nvox]
        const int D, const int H, const int W
    )
    {
        int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;

        if (!mask[idx]) {
            label_out[idx] = -1;
            dist_out[idx]  = 1.0e20f;
            return;
        }

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        // best stores squared distance
        float best = 3.402823466e38f; // ~FLT_MAX
        int best_id = -1;

        for (int k = 0; k < n_seeds; ++k) {
            int sz = seeds[3*k + 0];
            int sy = seeds[3*k + 1];
            int sx = seeds[3*k + 2];

            float dz = (float)(z - sz);
            float dy = (float)(y - sy);
            float dx = (float)(x - sx);

            float d2 = dz*dz + dy*dy + dx*dx;

            // tie-break: if equal, keep earlier k
            if (d2 < best) {
                best = d2;
                best_id = k;
            }
        }

        label_out[idx] = best_id;
        dist_out[idx]  = sqrtf(best);
    }
    '''
    ker = cp.RawKernel(code, "euclidean_voronoi_clipped")
    cache[dev] = ker
    build_euclidean_voronoi_clipping_kernel._cache = cache
    return ker


def euclidean_voronoi_clipping_gpu(
    mask,
    seeds,
    kernel=None,
    profile_gpu=False,
    eps_seed_check=True,
):
    """
    M1: Euclidean Voronoi + clipping baseline on GPU (CuPy).

    Returns:
      label_cp: (D,H,W) cp.int32  (solid -> -1)
      dist_cp:  (D,H,W) cp.float32 (solid -> 1e20)
      gpu_time_s: float or None (if profile_gpu=False)

    Notes:
      - the "clipping" is done in the kernel by writing -1/INF for solid voxels;
        for fluid voxels the label equals that of "Euclidean Voronoi on the whole domain, then clip".
    """
    import numpy as np
    import cupy as cp

    # ------------------------------------------------------------------
    # backend-aware: avoid implicit CuPy -> NumPy conversion
    # ------------------------------------------------------------------
    if eps_seed_check:
        backend = "cupy" if isinstance(mask, cp.ndarray) or isinstance(seeds, cp.ndarray) else "numpy"
        assert_seeds_valid_and_in_pore(mask, seeds, backend=backend)

    # ------------------------------------------------------------------
    # mask / seeds -> CuPy (no NumPy round-trip if already CuPy)
    # ------------------------------------------------------------------
    mask_cp, (D, H, W), mask_flat = _as_cupy_mask_u8(mask)

    seeds_cp = _as_cupy_int32(seeds)
    if seeds_cp.ndim != 2 or seeds_cp.shape[1] != 3:
        raise ValueError("seeds must have shape (n_seeds, 3) with (z,y,x).")
    n_seeds = int(seeds_cp.shape[0])
    if n_seeds <= 0:
        raise ValueError("seeds is empty.")

    # flatten seeds: (N,3) -> (3N,)
    seeds_flat_cp = cp.ascontiguousarray(seeds_cp.reshape(-1))

    nvox = int(D * H * W)

    label_flat = cp.empty(nvox, dtype=cp.int32)
    dist_flat  = cp.empty(nvox, dtype=cp.float32)

    if kernel is None:
        kernel = build_euclidean_voronoi_clipping_kernel()

    threads = 256
    blocks = (nvox + threads - 1) // threads

    gpu_time_s = None
    if profile_gpu:
        s_evt = cp.cuda.Event()
        e_evt = cp.cuda.Event()
        s_evt.record()

    kernel(
        (int(blocks),),
        (int(threads),),
        (
            mask_flat,
            seeds_flat_cp,
            np.int32(n_seeds),
            label_flat,
            dist_flat,
            np.int32(D), np.int32(H), np.int32(W),
        ),
    )

    if profile_gpu:
        e_evt.record()
        e_evt.synchronize()
        gpu_time_s = float(cp.cuda.get_elapsed_time(s_evt, e_evt)) / 1000.0

    label_cp = label_flat.reshape((D, H, W))
    dist_cp  = dist_flat.reshape((D, H, W))
    return label_cp, dist_cp, gpu_time_s



def compute_pore_skeleton_3d(mask, method="skimage", verbose=False):
    """
    Compute a 3D skeleton of the pore space (fluid mask).

    Parameters
    ----------
    mask : (D,H,W) bool/uint8
        True/1 means fluid (pore space).
    method : str
        Currently supports 'skimage'.
    verbose : bool

    Returns
    -------
    skel : (D,H,W) bool
        Skeleton voxels inside pore space.
    """
    import numpy as np

    mask_np = np.asarray(mask).astype(bool)

    if method.lower() != "skimage":
        raise ValueError("Only method='skimage' is supported in this baseline implementation.")

    try:
        from skimage.morphology import skeletonize_3d
    except Exception as e:
        raise ImportError(
            "scikit-image is required for skeletonize_3d. "
            "Install via: pip install scikit-image"
        ) from e

    # skeletonize_3d returns uint8 (0/1 or 0/255 depending on version)
    skel = skeletonize_3d(mask_np)
    skel = (skel > 0)

    # safety: keep skeleton strictly inside pore space
    skel &= mask_np

    if verbose:
        n_fluid = int(mask_np.sum())
        n_skel = int(skel.sum())
        print(f"[O1] skeletonize_3d: fluid voxels={n_fluid}, skeleton voxels={n_skel}, ratio={n_skel/max(n_fluid,1):.4f}")

    return skel


def project_seeds_to_skeleton_components(
    mask,
    skeleton,
    seeds,
    cc_connectivity=6,
    max_seed_snap_radius=4,
    verbose=False,
):
    """
    Project each seed to the nearest skeleton voxel *within the same connected component*
    of the pore space. This avoids 'thin-wall cross-projection' artefacts.

    Parameters
    ----------
    mask : (D,H,W) bool
    skeleton : (D,H,W) bool
    seeds : (N,3) array-like, z,y,x
    cc_connectivity : 6 or 26
        Connectivity used for connected-component labelling of pore space.
    max_seed_snap_radius : int
        If a seed falls on solid due to rounding, we try to snap it to a nearby fluid voxel
        within this radius (Manhattan-like search). If failed, raise.
    verbose : bool

    Returns
    -------
    seeds_skel : (N,3) np.int64
        Projected seed positions on skeleton.
    """
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree

    mask_np = np.asarray(mask).astype(bool)
    skel_np = np.asarray(skeleton).astype(bool)
    seeds_np = np.asarray(seeds, dtype=np.int64)
    D, H, W = mask_np.shape

    # --- connected components of pore space ---
    if cc_connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)  # 6-neigh
    elif cc_connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)  # 26-neigh
    else:
        raise ValueError("cc_connectivity must be 6 or 26")

    cc_label, n_cc = ndimage.label(mask_np, structure=struct)
    if verbose:
        print(f"[O1] connected components in pore mask: {n_cc}")

    skel_coords = np.argwhere(skel_np)
    if skel_coords.size == 0:
        raise RuntimeError("[O1] skeleton is empty. Check your mask or skeletonization step.")

    # group skeleton voxels by component id
    skel_cc = cc_label[skel_np]
    comp_to_coords = {}
    for cid in np.unique(skel_cc):
        if cid == 0:
            continue
        comp_to_coords[int(cid)] = skel_coords[skel_cc == cid]

    # KDTree per component (usually only one component, so cheap)
    comp_to_tree = {cid: cKDTree(coords) for cid, coords in comp_to_coords.items()}

    def _snap_seed_to_fluid(z, y, x):
        """If seed is not in fluid, try to find a nearby fluid voxel."""
        if 0 <= z < D and 0 <= y < H and 0 <= x < W and mask_np[z, y, x]:
            return z, y, x

        # small-radius search
        R = int(max_seed_snap_radius)
        for rr in range(1, R + 1):
            for dz in range(-rr, rr + 1):
                zz = z + dz
                if not (0 <= zz < D):
                    continue
                for dy in range(-rr, rr + 1):
                    yy = y + dy
                    if not (0 <= yy < H):
                        continue
                    for dx in range(-rr, rr + 1):
                        xx = x + dx
                        if not (0 <= xx < W):
                            continue
                        if mask_np[zz, yy, xx]:
                            return zz, yy, xx
        raise RuntimeError(f"[O1] Seed ({z},{y},{x}) is not in fluid and cannot be snapped within radius={R}.")

    seeds_skel = np.empty_like(seeds_np, dtype=np.int64)

    for i in range(seeds_np.shape[0]):
        z, y, x = map(int, seeds_np[i])
        z, y, x = _snap_seed_to_fluid(z, y, x)

        cid = int(cc_label[z, y, x])
        if cid == 0:
            raise RuntimeError(f"[O1] Seed {i} is not in any pore connected component after snapping.")

        if cid not in comp_to_tree:
            # no skeleton voxels in this component (rare but possible)
            # fallback: do global projection (still safe for single-component domains)
            tree_global = cKDTree(skel_coords)
            _, idx = tree_global.query([z, y, x], k=1)
            pz, py, px = skel_coords[idx]
            seeds_skel[i] = [int(pz), int(py), int(px)]
            continue

        tree = comp_to_tree[cid]
        coords = comp_to_coords[cid]
        _, idx = tree.query([z, y, x], k=1)
        pz, py, px = coords[idx]
        seeds_skel[i] = [int(pz), int(py), int(px)]

    if verbose:
        n_unique = len({tuple(v.tolist()) for v in seeds_skel})
        print(f"[O1] projected seeds on skeleton: N={seeds_skel.shape[0]}, unique={n_unique}")

    return seeds_skel

def resolve_duplicate_skeleton_seeds(skeleton, seeds_skel, connectivity=26, verbose=False):
    """
    Ensure all projected seeds land on unique skeleton voxels.
    If duplicates occur, we BFS on skeleton voxels to find nearest unoccupied skeleton voxel.

    Parameters
    ----------
    skeleton : (D,H,W) bool
    seeds_skel : (N,3) np.int64
    connectivity : 6 or 26

    Returns
    -------
    seeds_unique : (N,3) np.int64
    """
    import numpy as np
    from collections import deque

    skel = np.asarray(skeleton).astype(bool)
    seeds = np.asarray(seeds_skel, dtype=np.int64).copy()
    D, H, W = skel.shape

    if connectivity == 6:
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    elif connectivity == 26:
        offsets = []
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append((dz,dy,dx))
    else:
        raise ValueError("connectivity must be 6 or 26")

    occupied = set()
    dup_count = 0

    def _in_bounds(z,y,x):
        return (0 <= z < D) and (0 <= y < H) and (0 <= x < W)

    for i in range(seeds.shape[0]):
        z0, y0, x0 = map(int, seeds[i])
        if not _in_bounds(z0,y0,x0) or (not skel[z0,y0,x0]):
            raise RuntimeError(f"[O1] Seed {i} projected point is not on skeleton: ({z0},{y0},{x0})")

        key = (z0,y0,x0)
        if key not in occupied:
            occupied.add(key)
            continue

        # duplicate -> BFS to find nearest free skeleton voxel
        dup_count += 1
        q = deque([key])
        visited = {key}

        found = None
        while q:
            z,y,x = q.popleft()
            for dz,dy,dx in offsets:
                zz, yy, xx = z+dz, y+dy, x+dx
                k2 = (zz,yy,xx)
                if k2 in visited:
                    continue
                visited.add(k2)
                if not _in_bounds(zz,yy,xx):
                    continue
                if not skel[zz,yy,xx]:
                    continue
                if k2 not in occupied:
                    found = k2
                    q.clear()
                    break
                q.append(k2)

        if found is None:
            raise RuntimeError("[O1] Cannot resolve duplicate seeds on skeleton (skeleton too small or too many seeds).")

        seeds[i] = np.array(found, dtype=np.int64)
        occupied.add(found)

    if verbose and dup_count > 0:
        print(f"[O1] resolved duplicates on skeleton: {dup_count}")

    return seeds


def o1_skeleton_dual_voronoi_gpu(
    mask,
    seeds,
    skeleton_method="skimage",
    cc_connectivity=6,
    skeleton_connectivity=26,
    propagate_max_iter=128,
    relax_eps=1e-6,
    verbose=True,
    profile_gpu=False,
):
    """
    O1 baseline: skeleton-dual / network-driven Voronoi surrogate.

    Pipeline:
      (1) skeletonize pore space (CPU)
      (2) project seeds to skeleton within same pore CC (CPU)
      (3) compute geodesic Voronoi *on skeleton mask* (GPU exact relaxation)
      (4) propagate those (label,dist) into full pore space using jump=1 local relax iterations (GPU)

    Returns
    -------
    label_cp : cp.ndarray int32, shape (D,H,W)
    dist_cp  : cp.ndarray float32, shape (D,H,W)
    meta     : dict (timing + projected seeds + skeleton)
    """
    import time
    import numpy as np
    import cupy as cp

    t0 = time.time()

    # --- (1) skeletonize on CPU ---
    t_s0 = time.time()
    skel_np = compute_pore_skeleton_3d(mask, method=skeleton_method, verbose=verbose)
    t_skeleton = time.time() - t_s0

    # --- (2) project seeds to skeleton, avoid cross-wall projection ---
    t_p0 = time.time()
    seeds_skel = project_seeds_to_skeleton_components(
        mask, skel_np, seeds,
        cc_connectivity=cc_connectivity,
        max_seed_snap_radius=4,
        verbose=verbose,
    )
    seeds_skel = resolve_duplicate_skeleton_seeds(
        skel_np, seeds_skel, connectivity=skeleton_connectivity, verbose=verbose
    )
    t_project = time.time() - t_p0

    # --- (3) Voronoi on skeleton mask (GPU) ---
    skel_mask_cp = cp.asarray(skel_np, dtype=cp.bool_)

    # run the exact relaxation on the skeleton domain (few skeleton voxels)
    t_v0 = time.time()
    label_skel_cp, dist_skel_cp = exact_geodesic_voronoi_gpu(
        skel_mask_cp,
        seeds_skel,
        connectivity=skeleton_connectivity,
        max_iter=None,
        eps=float(relax_eps),
        verbose=False,
        raise_on_nonconvergence=False,   # baseline: must not stop the run
        check_optimality=False,
        profile_gpu=profile_gpu,
    )
    cp.cuda.Device().synchronize()
    t_skel_voronoi_wall = time.time() - t_v0

    # --- (4) propagate skeleton-labelled distances to full pore space ---
    # initial state is exactly (label_skel, dist_skel): only skeleton voxels have finite dist & labels
    mask_cp = cp.asarray(mask, dtype=cp.uint8)
    D, H, W = mask_cp.shape
    nvox = int(D * H * W)
    mask_flat = mask_cp.ravel()

    # pack (dist|label)
    INF = cp.float32(1e20)
    dist_init = dist_skel_cp.astype(cp.float32, copy=False)
    label_init = label_skel_cp.astype(cp.int32, copy=False)

    # ensure: non-fluid voxels stay INF/-1
    # (skeleton is subset of mask, so OK)
    solid = (mask_flat == 0).reshape((D, H, W))
    dist_init = cp.where(solid, INF, dist_init)
    label_init = cp.where(solid, cp.int32(-1), label_init)

    dist_u32 = dist_init.view(cp.uint32)
    label_u32 = label_init.astype(cp.uint32, copy=False)
    state_a = (dist_u32.astype(cp.uint64) << cp.uint64(32)) | label_u32.astype(cp.uint64)
    state_a = state_a.reshape((D, H, W)).ravel()
    state_b = state_a.copy()

    relax_kernel = build_local_relax_packed_kernel_3d()
    threads = 256
    blocks = (nvox + threads - 1) // threads

    gpu_ms_total = 0.0
    t_prop0 = time.time()

    # iterative jump=1 relax until convergence or max_iter
    for it in range(int(propagate_max_iter)):
        if profile_gpu:
            s_evt = cp.cuda.Event(); e_evt = cp.cuda.Event()
            s_evt.record()

        relax_kernel(
            (int(blocks),),
            (int(threads),),
            (
                mask_flat,
                state_a,
                state_b,
                np.int32(D), np.int32(H), np.int32(W),
                np.float32(relax_eps),
            ),
        )

        if profile_gpu:
            e_evt.record(); e_evt.synchronize()
            gpu_ms_total += float(cp.cuda.get_elapsed_time(s_evt, e_evt))
        else:
            cp.cuda.Device().synchronize()

        # early stop: if no change
        changed = bool(cp.any(state_b != state_a).item())
        state_a, state_b = state_b, state_a
        if (not changed):
            if verbose:
                print(f"[O1] propagate: converged in {it+1} iterations")
            break

    t_propagate_wall = time.time() - t_prop0
    t_total = time.time() - t0

    # unpack
    label_flat = (state_a & cp.uint64(0xFFFFFFFF)).astype(cp.int32)
    dist_u32 = (state_a >> cp.uint64(32)).astype(cp.uint32)
    dist_flat = dist_u32.view(cp.float32)

    label_cp = label_flat.reshape((D, H, W))
    dist_cp = dist_flat.reshape((D, H, W))

    # enforce solid convention
    solid3 = (mask_cp == 0)
    label_cp = cp.where(solid3, cp.int32(-1), label_cp)
    dist_cp = cp.where(solid3, INF, dist_cp)

    meta = {
        "seeds_skel": np.asarray(seeds_skel, dtype=np.int64),
        "skeleton": skel_np,  # CPU bool, for visualisation / statistics
        "timing": {
            "t_skeleton_cpu": float(t_skeleton),
            "t_project_cpu": float(t_project),
            "t_skel_voronoi_wall": float(t_skel_voronoi_wall),
            "t_propagate_wall": float(t_propagate_wall),
            "t_total_wall": float(t_total),
            "propagate_gpu_time": float(gpu_ms_total) / 1000.0 if profile_gpu else None,
        }
    }

    if verbose:
        tt = meta["timing"]
        print("[O1] timing summary:")
        print(f"  skeletonize (CPU): {tt['t_skeleton_cpu']:.4f} s")
        print(f"  seed projection (CPU): {tt['t_project_cpu']:.4f} s")
        print(f"  skeleton Voronoi (wall): {tt['t_skel_voronoi_wall']:.4f} s")
        print(f"  propagate to pore (wall): {tt['t_propagate_wall']:.4f} s")
        if tt["propagate_gpu_time"] is not None:
            print(f"  propagate GPU events: {tt['propagate_gpu_time']:.4f} s")
        print(f"  total (wall): {tt['t_total_wall']:.4f} s")

    return label_cp, dist_cp, meta



def make_3d_maze_case(
    D=64,
    H=64,
    W=96,
    n_seeds=20,
    seed_mode="cell_centers_poisson",
    seed_random_state=0,
    corridor_width=3,
    wall_thickness=2,
    loop_prob=0.05,
    seed_min_sep_cells=2,
    ensure_connected=True,
    cc_connectivity=6,
    verbose=False,
):
    """
    Case C (Maze): a 2.5D maze - a perfect maze in the (z,x) plane, extruded along y.

    Constraint:
      - the maze must contain no completely closed space without an opening (also not on the 2D slices with axis='y').
    This construction ("2D maze + extrusion in y") satisfies it:
      - every y slice shows the same 2D maze;
      - the 2D maze is a spanning tree (loops optional), so every cell has at least one opening;
      - the whole pore space is one 6-connected component (with ensure_connected=True only the
        largest component is kept, as a safeguard).

    Output:
      mask[z,y,x] = True  -> fluid
      mask[z,y,x] = False -> solid
      seeds: (N,3) int64 in (z,y,x)
    """
    import numpy as np
    from scipy import ndimage

    D = int(D); H = int(H); W = int(W)
    corridor_width = int(corridor_width)
    wall_thickness = int(wall_thickness)
    n_seeds = int(n_seeds)

    if D < 8 or H < 8 or W < 8:
        raise ValueError("Domain too small for a meaningful maze. Increase D/H/W.")
    if corridor_width < 1:
        raise ValueError("corridor_width must be >= 1")
    if wall_thickness < 1:
        raise ValueError("wall_thickness must be >= 1")
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    if (H - 2 * wall_thickness) <= 1:
        raise ValueError("H is too small relative to wall_thickness; no interior y-range to extrude.")

    rng = np.random.RandomState(int(seed_random_state))

    # ============================================================
    # 0) coarse grid centers in z and x only (2D maze on (z,x))
    # ============================================================
    pitch = corridor_width + wall_thickness
    neg = corridor_width // 2
    pos = corridor_width - neg - 1

    def _centers_1d(dim):
        c0 = wall_thickness + neg
        cmax = dim - wall_thickness - 1 - pos
        if cmax < c0:
            return np.array([], dtype=np.int32)
        n = int((cmax - c0) // pitch + 1)
        return (c0 + pitch * np.arange(n, dtype=np.int32)).astype(np.int32)

    cz = _centers_1d(D)
    cx = _centers_1d(W)

    nz, nx = int(cz.size), int(cx.size)
    if nz < 2 or nx < 2:
        raise ValueError(
            f"Too few maze cells in (z,x): (nz,nx)=({nz},{nx}). "
            f"Try larger D/W or reduce corridor_width/wall_thickness."
        )

    # y extrude range (full interior)
    y0_ex = int(wall_thickness)
    y1_ex = int(H - wall_thickness)  # exclusive
    y_seed = (y0_ex + y1_ex) // 2

    if verbose:
        print(f"[maze] 2.5D maze (2D in z-x, extruded in y)")
        print(f"[maze] voxel domain: D,H,W = {D},{H},{W}")
        print(f"[maze] pitch={pitch}, corridor_width={corridor_width}, wall_thickness={wall_thickness}")
        print(f"[maze] coarse cells: nz={nz}, nx={nx} (total={nz*nx}), y_extrude=[{y0_ex},{y1_ex})")

    # ============================================================
    # 1) build perfect 2D maze edges on (nz,nx) via randomized DFS
    # ============================================================
    visited = np.zeros((nz, nx), dtype=bool)

    def _lin_id(iz, ix):
        return iz * nx + ix

    def _unlin_id(a):
        iz = a // nx
        ix = a - iz * nx
        return int(iz), int(ix)

    edges = set()  # undirected edges as (min_id, max_id)
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]

    sz0 = rng.randint(0, nz)
    sx0 = rng.randint(0, nx)

    stack = [(sz0, sx0)]
    visited[sz0, sx0] = True

    while stack:
        iz, ix = stack[-1]

        neigh = []
        for dz, dx in dirs:
            zz = iz + dz
            xx = ix + dx
            if 0 <= zz < nz and 0 <= xx < nx and (not visited[zz, xx]):
                neigh.append((zz, xx))

        if neigh:
            zz, xx = neigh[rng.randint(len(neigh))]
            visited[zz, xx] = True
            stack.append((zz, xx))

            a = _lin_id(iz, ix)
            b = _lin_id(zz, xx)
            edges.add((a, b) if a < b else (b, a))
        else:
            stack.pop()

    # optional loops (adds cycles, still no isolated cells)
    loop_prob = float(loop_prob)
    if loop_prob > 0.0:
        pos_dirs = [(1,0), (0,1)]
        for iz in range(nz):
            for ix in range(nx):
                a = _lin_id(iz, ix)
                for dz, dx in pos_dirs:
                    zz, xx = iz + dz, ix + dx
                    if not (0 <= zz < nz and 0 <= xx < nx):
                        continue
                    b = _lin_id(zz, xx)
                    e = (a, b) if a < b else (b, a)
                    if e in edges:
                        continue
                    if rng.rand() < loop_prob:
                        edges.add(e)

    # ============================================================
    # 2) carve voxel mask: rooms at cell centers + corridors along edges, extruded in y
    # ============================================================
    mask = np.zeros((D, H, W), dtype=bool)

    def _clamp_box_1d(a0, a1_ex, dim):
        # clamp [a0, a1_ex) into interior [wall_thickness, dim-wall_thickness)
        a0c = max(wall_thickness, int(a0))
        a1c = min(int(dim - wall_thickness), int(a1_ex))
        return a0c, a1c

    # carve room at every (iz,ix)
    for iz in range(nz):
        zc = int(cz[iz])
        z0 = zc - neg
        z1 = z0 + corridor_width  # exclusive
        z0, z1 = _clamp_box_1d(z0, z1, D)

        for ix in range(nx):
            xc = int(cx[ix])
            x0 = xc - neg
            x1 = x0 + corridor_width  # exclusive
            x0, x1 = _clamp_box_1d(x0, x1, W)

            if (z1 > z0) and (x1 > x0):
                mask[z0:z1, y0_ex:y1_ex, x0:x1] = True

    # carve corridors for each edge
    for a, b in edges:
        az, ax = _unlin_id(a)
        bz, bx = _unlin_id(b)

        z0c, x0c = int(cz[az]), int(cx[ax])
        z1c, x1c = int(cz[bz]), int(cx[bx])

        if ax != bx:
            # neighbor in x-direction: carve along x, cross-section in z
            xs = min(x0c, x1c)
            xe = max(x0c, x1c)

            z0 = z0c - neg
            z1 = z0 + corridor_width  # exclusive
            z0, z1 = _clamp_box_1d(z0, z1, D)

            # x corridor is inclusive on the end, so use [xs, xe+1)
            x0 = max(wall_thickness, xs)
            x1 = min(W - wall_thickness - 1, xe)
            if (z1 > z0) and (x1 >= x0):
                mask[z0:z1, y0_ex:y1_ex, x0:(x1 + 1)] = True

        elif az != bz:
            # neighbor in z-direction: carve along z, cross-section in x
            zs = min(z0c, z1c)
            ze = max(z0c, z1c)

            x0 = x0c - neg
            x1 = x0 + corridor_width  # exclusive
            x0, x1 = _clamp_box_1d(x0, x1, W)

            z0 = max(wall_thickness, zs)
            z1 = min(D - wall_thickness - 1, ze)
            if (x1 > x0) and (z1 >= z0):
                mask[z0:(z1 + 1), y0_ex:y1_ex, x0:x1] = True
        else:
            # should not happen in 2D maze
            continue

    # enforce solid boundary shell (hard guarantee)
    wt = wall_thickness
    mask[:wt, :, :] = False
    mask[-wt:, :, :] = False
    mask[:, :wt, :] = False
    mask[:, -wt:, :] = False
    mask[:, :, :wt] = False
    mask[:, :, -wt:] = False

    # ============================================================
    # 3) connectivity check (3D and also 2D slice sanity), keep largest if needed
    # ============================================================
    if ensure_connected:
        if cc_connectivity == 6:
            struct3 = ndimage.generate_binary_structure(3, 1)
        elif cc_connectivity == 26:
            struct3 = ndimage.generate_binary_structure(3, 3)
        else:
            raise ValueError("cc_connectivity must be 6 or 26")

        cc3, ncc3 = ndimage.label(mask, structure=struct3)
        if ncc3 <= 0:
            raise RuntimeError("[maze] Empty pore space after carving.")
        if ncc3 != 1:
            sizes = np.bincount(cc3.ravel())
            sizes[0] = 0
            keep = int(sizes.argmax())
            mask = (cc3 == keep)
            if verbose:
                print(f"[maze] WARNING: ncc3={ncc3}. Keeping largest component id={keep}.")

    # 2D slice connectivity check on the plane used for visualisation, axis='y'
    # confirms that the (z, x) slice at y = y_seed has no closed isolated pocket
    m2 = mask[:, y_seed, :]
    struct2 = ndimage.generate_binary_structure(2, 1)
    cc2, ncc2 = ndimage.label(m2, structure=struct2)
    if ncc2 != 1:
        # the case to avoid: a closed or isolated pocket in the 2D slice
        raise RuntimeError(
            f"[maze] 2D slice at y={y_seed} has ncc2={ncc2} connected components. "
            "This violates 'no fully closed space in the analysis slice'. "
            "Check corridor_width/wall_thickness or carving logic."
        )

    # ============================================================
    # 4) place seeds
    # ============================================================
    seed_mode = str(seed_mode).lower().strip()

    if seed_mode in ("cell_centers_poisson", "cell_centers", "centers"):
        cell_coords = [(iz, ix) for iz in range(nz) for ix in range(nx)]
        rng.shuffle(cell_coords)

        min_sep = int(seed_min_sep_cells) if seed_min_sep_cells is not None else 0
        min_sep2 = float(min_sep * min_sep)

        chosen = []
        for (iz, ix) in cell_coords:
            if len(chosen) >= n_seeds:
                break
            if min_sep <= 0:
                chosen.append((iz, ix))
                continue
            ok = True
            for (jz, jx) in chosen:
                dz = iz - jz
                dx = ix - jx
                if (dz * dz + dx * dx) < min_sep2:
                    ok = False
                    break
            if ok:
                chosen.append((iz, ix))

        if len(chosen) < n_seeds:
            raise ValueError(
                f"Cannot place {n_seeds} seeds with seed_min_sep_cells={min_sep}. "
                f"Got only {len(chosen)}. Reduce seed_min_sep_cells or n_seeds."
            )

        seeds = np.zeros((n_seeds, 3), dtype=np.int64)
        for k, (iz, ix) in enumerate(chosen[:n_seeds]):
            zc = int(cz[iz])
            xc = int(cx[ix])
            seeds[k, 0] = zc
            seeds[k, 1] = int(y_seed)
            seeds[k, 2] = xc

        # safety: ensure seeds are in fluid; if rare mismatch, snap to nearest fluid voxel
        bad = ~mask[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
        if np.any(bad):
            fluid_idx = np.argwhere(mask)
            for kk in np.where(bad)[0]:
                z0, y0, x0 = seeds[kk]
                d2 = (fluid_idx[:, 0] - z0) ** 2 + (fluid_idx[:, 1] - y0) ** 2 + (fluid_idx[:, 2] - x0) ** 2
                jj = int(d2.argmin())
                seeds[kk] = fluid_idx[jj].astype(np.int64)

    elif seed_mode == "random":
        fluid_idx = np.argwhere(mask)
        if fluid_idx.shape[0] < n_seeds:
            raise ValueError("Not enough fluid voxels to place seeds.")
        chosen = rng.choice(fluid_idx.shape[0], size=n_seeds, replace=False)
        seeds = fluid_idx[chosen].astype(np.int64)

    else:
        raise ValueError("Unknown seed_mode. Use 'cell_centers_poisson' or 'random'.")

    if verbose:
        n_fluid = int(mask.sum())
        print(f"[maze] fluid voxels={n_fluid}, fluid ratio={n_fluid / float(D * H * W):.4f}")
        print(f"[maze] seeds placed: {seeds.shape[0]}")
        print(f"[maze] 2D slice connectivity at y={y_seed}: ncc2={int(ncc2)} (must be 1)")

    return mask, seeds

import numpy as np
from scipy import ndimage

def compute_reachable_V(mask, seeds, connectivity=6):
    """
    V: the reachable fluid set (fluid voxels in the same connected component as at least one seed).
    Recommended common setting: connectivity=6.
    Returns: Vmask (D,H,W) bool
    """
    mask_np = np.asarray(mask, dtype=bool)
    seeds_np = np.asarray(seeds, dtype=np.int64)

    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be 6 or 26")

    cc, _ = ndimage.label(mask_np, structure=struct)
    seed_cc = cc[seeds_np[:, 0], seeds_np[:, 1], seeds_np[:, 2]]
    keep = np.unique(seed_cc)
    keep = keep[keep != 0]
    Vmask = np.isin(cc, keep)
    return Vmask


def compute_r_unf(label, Vmask):
    """r_unf = |{p in V: L(p)=-1}| / |V|

    Works with numpy and cupy: for GPU input (cupy.ndarray) the statistics are computed on the GPU and
    only a scalar is returned, so the full label/Vmask arrays are not copied to the CPU with .get().
    """
    import numpy as np
    try:
        import cupy as cp
    except Exception:
        cp = None

    def _is_cupy(x):
        return (cp is not None) and isinstance(x, cp.ndarray)

    if _is_cupy(label) or _is_cupy(Vmask):
        L = label if _is_cupy(label) else cp.asarray(label)
        V = Vmask if _is_cupy(Vmask) else cp.asarray(Vmask)
        if V.dtype != cp.bool_:
            V = (V != 0)
        denom = int(cp.count_nonzero(V).get())
        if denom == 0:
            return np.nan
        num = int(cp.count_nonzero(V & (L < 0)).get())
        return float(num) / float(denom)

    # numpy fallback (original)
    L = np.asarray(label, dtype=np.int32)
    V = np.asarray(Vmask, dtype=bool)
    denom = int(V.sum())
    if denom == 0:
        return np.nan
    return float(np.count_nonzero(V & (L < 0))) / float(denom)
def compute_e_vox(label_pred, label_ref, Vmask):
    """e_vox = |{p in V: L_pred != L_ref}| / |V| (-1 counts as a mismatch)"""
    A = np.asarray(label_pred, dtype=np.int32)
    B = np.asarray(label_ref,  dtype=np.int32)
    V = np.asarray(Vmask, dtype=bool)
    denom = int(V.sum())
    if denom == 0:
        return np.nan
    return float(np.count_nonzero(V & (A != B))) / float(denom)


def compute_eta_roi(roi_mask_flat, Vmask):
    """
    eta_ROI = |ROI and V| / |V|
    roi_mask_flat: 1D (nvox,) uint8, 1=ROI, 0=non-ROI

    Works with numpy and cupy: for GPU input (cupy.ndarray) the statistics are computed on the GPU and
    only a scalar is returned, so the full roi_mask/Vmask arrays are not copied to the CPU with .get().
    """
    import numpy as np
    try:
        import cupy as cp
    except Exception:
        cp = None

    def _is_cupy(x):
        return (cp is not None) and isinstance(x, cp.ndarray)

    if _is_cupy(roi_mask_flat) or _is_cupy(Vmask):
        roi = roi_mask_flat if _is_cupy(roi_mask_flat) else cp.asarray(roi_mask_flat)
        V = Vmask if _is_cupy(Vmask) else cp.asarray(Vmask)
        roi = roi.ravel()
        if V.dtype != cp.bool_:
            V = (V != 0)
        V = V.ravel()
        denom = int(cp.count_nonzero(V).get())
        if denom == 0:
            return np.nan
        num = int(cp.count_nonzero((roi != 0) & V).get())
        return float(num) / float(denom)

    # numpy fallback (original)
    V = np.asarray(Vmask, dtype=bool).ravel()
    roi = np.asarray(roi_mask_flat, dtype=np.uint8).ravel()
    denom = int(V.sum())
    if denom == 0:
        return np.nan
    return float(np.count_nonzero((roi != 0) & V)) / float(denom)
import numpy as np
import math
from scipy import ndimage

import numpy as np
import math
from scipy import ndimage

def _fill_unlabeled_cracks_unique(
    L: np.ndarray,
    V: np.ndarray,
    *,
    connectivity: int = 6,
    max_iters=None,   # None = auto
) -> np.ndarray:
    """
    Unambiguous crack fill: fills only unlabelled (-1) voxels inside V, and only where the labelled
    neighbours carry a single label.
    """
    L0 = np.asarray(L, dtype=np.int32)
    V0 = np.asarray(V, dtype=bool)
    D, H, W = L0.shape
    Lf = L0.copy()

    if max_iters is None:
        # with 6-neighbours the worst-case step count between two points is <= D+H+W
        max_iters = int(D + H + W)
    else:
        max_iters = int(max_iters)

    if connectivity == 6:
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    elif connectivity == 26:
        offsets = []
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append((dz,dy,dx))
    else:
        raise ValueError("connectivity must be 6 or 26")

    def _overlap_1d(n, d):
        if d > 0:
            dst = slice(0, n - d); src = slice(d, n)
        elif d < 0:
            dst = slice(-d, n);    src = slice(0, n + d)
        else:
            dst = slice(0, n);     src = slice(0, n)
        return dst, src

    for _ in range(max_iters):
        unl = V0 & (Lf < 0)
        if not np.any(unl):
            break

        first = np.full_like(Lf, -1, dtype=np.int32)
        conflict = np.zeros_like(V0, dtype=bool)

        for dz, dy, dx in offsets:
            z_dst, z_src = _overlap_1d(D, dz)
            y_dst, y_src = _overlap_1d(H, dy)
            x_dst, x_src = _overlap_1d(W, dx)

            dst = (z_dst, y_dst, x_dst)
            src = (z_src, y_src, x_src)

            unl_dst = unl[dst]
            if not np.any(unl_dst):
                continue

            nb_lab = Lf[src]
            nb_valid = V0[src] & (nb_lab >= 0)
            cond = unl_dst & nb_valid
            if not np.any(cond):
                continue

            f_dst = first[dst]
            c_dst = conflict[dst]

            m_new = cond & (f_dst == -1)
            if np.any(m_new):
                f_dst[m_new] = nb_lab[m_new]

            m_conf = cond & (f_dst != -1) & (f_dst != nb_lab)
            if np.any(m_conf):
                c_dst[m_conf] = True

            first[dst] = f_dst
            conflict[dst] = c_dst

        fill = unl & (first >= 0) & (~conflict)
        if int(np.count_nonzero(fill)) == 0:
            break

        Lf[fill] = first[fill]

    return Lf


import numpy as np
import math
from scipy import ndimage

def compute_cell_cut_rate_and_island_ratio(
    label, seeds, Vmask,
    connectivity=6,
    min_component_voxels=16,
    min_component_fraction=0.01,
    seed_indices=None,            # only evaluate these seed indices (None = all)
    *,
    connect_through_unlabeled=False,   # True: -1 counts as passable medium for connectivity
):
    """
    Two metrics computed per cell and then averaged:

    (1) r_cut: fraction of cut cells = mean_k 1[n_comp_kept(k) >= 2]
    (2) rho_isl: island fraction = mean_k ( |R_k minus C_k| / max(|C_k|,1) )
        where C_k is the main component connected to seed s_k (the component that contains the seed);
        only pieces with size >= threshold count as islands (ignores spurious noise)

    domain: inside V (Vmask==True)
    connectivity: 6-neighbour recommended (as in the definition of V)

    connect_through_unlabeled
      - False (default, strict): connectivity only within the voxels with label==k (-1 breaks connections)
      - True (crack-free): connections may pass through unlabelled (-1) voxels,
        but component sizes still count only the voxels with label==k (-1 does not change the volume count)

    seed_indices
      - None: all seeds (very slow for large n_seeds; combine with struct_metrics_sample sampling upstream)
      - iterable: mean estimated over the given subset of seeds

    Performance
    -----------
    A full-domain scan per k, region_k = (V & (L==k)), costs O(|V| * n_seeds) and stalls for
    |V| ~ 1e7 and n_seeds ~ 1e4.

    Here instead:
      1) one bincount gives the voxel count B_k of every cell;
      2) one ndimage.find_objects gives the axis-aligned bounding box of every cell;
      3) for the cells evaluated, ndimage.label runs only inside their bounding box.

    This reduces the cost to O(|V| + sum box_k), with sum box_k over the evaluated seeds only,
    which is much faster in practice.
    """
    L = np.asarray(label, dtype=np.int32)
    V = np.asarray(Vmask, dtype=bool)
    seeds_np = np.asarray(seeds, dtype=np.int64)

    n_seeds = int(seeds_np.shape[0])
    if n_seeds <= 0:
        return np.nan, np.nan

    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be 6 or 26")

    # ------------------------------------------------------------
    # Precompute per-cell voxel counts B_k in ONE pass
    # ------------------------------------------------------------
    V_flat = V.ravel()
    L_flat = L.ravel()

    inV = V_flat
    valid = inV & (L_flat >= 0)

    if int(np.count_nonzero(valid)) == 0:
        return np.nan, np.nan

    labels_inV = L_flat[valid]
    Bk = np.bincount(labels_inV, minlength=n_seeds).astype(np.int64, copy=False)

    # ------------------------------------------------------------
    # Precompute per-cell bounding boxes in ONE pass
    #   find_objects expects labels in [1..max_label], 0 as background.
    #   We map: label k -> k+1, and treat outside-V or negative as 0.
    # ------------------------------------------------------------
    L_find = L.copy()
    L_find[~V] = -1
    L_find[L_find < 0] = -1
    L_find += 1
    L_find[L_find < 1] = 0
    objs = ndimage.find_objects(L_find, max_label=n_seeds)

    if seed_indices is None:
        seed_indices = range(n_seeds)

    cut_flags = []
    isl_ratios = []

    for k in seed_indices:
        k = int(k)
        if k < 0 or k >= n_seeds:
            continue

        B = int(Bk[k]) if k < Bk.shape[0] else 0
        if B <= 0:
            continue

        slc = objs[k]  # because label k is stored as (k+1) in L_find
        if slc is None:
            continue

        z = int(seeds_np[k, 0])
        y = int(seeds_np[k, 1])
        x = int(seeds_np[k, 2])

        thr = int(max(min_component_voxels, math.ceil(float(min_component_fraction) * float(B))))

        if not connect_through_unlabeled:
            subV = V[slc]
            subL = L[slc]
            sub = subV & (subL == k)

            cc, ncc = ndimage.label(sub, structure=struct)

            z0, y0, x0 = int(slc[0].start), int(slc[1].start), int(slc[2].start)
            z1, y1, x1 = int(slc[0].stop),  int(slc[1].stop),  int(slc[2].stop)

            if (z < z0) or (y < y0) or (x < x0) or (z >= z1) or (y >= y1) or (x >= x1):
                cut_flags.append(1.0)
                isl_ratios.append(float("inf"))
                continue

            seed_cid = int(cc[z - z0, y - y0, x - x0])
            if seed_cid <= 0:
                cut_flags.append(1.0)
                isl_ratios.append(float("inf"))
                continue

            sizes = np.bincount(cc.ravel()).astype(np.int64, copy=False)
            if sizes.size == 0:
                cut_flags.append(1.0)
                isl_ratios.append(float("inf"))
                continue

            sizes[0] = 0
            kept = np.flatnonzero(sizes >= thr)

            cut_flags.append(1.0 if int(kept.size) >= 2 else 0.0)

            A = int(sizes[seed_cid]) if seed_cid < sizes.size else 0
            sum_kept = int(sizes[kept].sum()) if kept.size > 0 else 0
            islands = (sum_kept - A) if (A >= thr) else sum_kept
            isl_ratios.append(float(islands) / float(max(A, 1)))

        else:
            # --------------------------------------------------------
            # crack-free (-1 allowed as passable medium): semantics unchanged.
            # note: this branch is slower; for large problems use it on small samples only.
            # --------------------------------------------------------
            region_k = V & (L == k)
            passable = V & (region_k | (L < 0))

            coords = np.argwhere(passable)
            if coords.size == 0:
                continue

            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0)
            slc2 = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))

            sub_pass = passable[slc2]
            cc, ncc = ndimage.label(sub_pass, structure=struct)

            seed_cid = int(cc[z - z0, y - y0, x - x0])
            if seed_cid <= 0:
                cut_flags.append(1.0)
                isl_ratios.append(float("inf"))
                continue

            # component sizes are counted on label==k voxels only
            sub_k = region_k[slc2]
            ids = cc[sub_k]
            if ids.size == 0:
                cut_flags.append(1.0)
                isl_ratios.append(float("inf"))
                continue

            sizes = np.bincount(ids.ravel(), minlength=ncc + 1).astype(np.int64, copy=False)
            sizes[0] = 0
            kept = np.flatnonzero(sizes >= thr)

            cut_flags.append(1.0 if int(kept.size) >= 2 else 0.0)

            A = int(sizes[seed_cid]) if seed_cid < sizes.size else 0
            sum_kept = int(sizes[kept].sum()) if kept.size > 0 else 0
            islands = (sum_kept - A) if (A >= thr) else sum_kept
            isl_ratios.append(float(islands) / float(max(A, 1)))

    r_cut = float(np.mean(cut_flags)) if len(cut_flags) > 0 else np.nan
    rho_isl = float(np.mean(isl_ratios)) if len(isl_ratios) > 0 else np.nan
    return r_cut, rho_isl

def run_demo_maze_case(visualize=True):
    import time
    import numpy as np
    import cupy as cp
    from scipy import ndimage

    D = 64
    H = 64
    W = 96
    n_seeds = 20

    # maze parameters supported by make_3d_maze_case (method unchanged)
    corridor_width = 3
    wall_thickness = 2
    loop_prob = 0.05
    seed_min_sep_cells = 2

    global ROI_JFA_LAST_VIZ_TIME
    global ROI_JFA_LAST_GPU_TIME
    global ROI_JFA_LAST_RECORD_TIME
    global EXACT_LAST_GPU_TIME
    global OAJFA_LAST_GPU_TIME

    # =========================================================
    # Case C: Maze
    # =========================================================
    print("Building 3D maze case (Case C: Maze / labyrinth)...")
    mask, seeds = make_3d_maze_case(
        D=D, H=H, W=W,
        n_seeds=n_seeds,
        seed_mode="cell_centers_poisson",   # only 'cell_centers_poisson' or 'random' allowed
        seed_random_state=0,
        corridor_width=corridor_width,
        wall_thickness=wall_thickness,
        loop_prob=loop_prob,
        seed_min_sep_cells=seed_min_sep_cells,
        ensure_connected=True,
        cc_connectivity=6,
        verbose=True,
    )

    D, H, W = mask.shape
    print(f"Domain voxel size: {D} x {H} x {W}, {n_seeds} seeds")
    # one check for all methods: seeds inside the domain and in the pore space (mask==1)
    assert_seeds_valid_and_in_pore(mask, seeds, backend="numpy")
    print("=" * 60)

    # quick connectivity self-check (print only, mask unchanged) to confirm that the maze is connected
    try:
        struct6 = ndimage.generate_binary_structure(3, 1)
        cc, ncc = ndimage.label(np.asarray(mask, dtype=bool), structure=struct6)
        n_fluid = int(np.asarray(mask, dtype=bool).sum())
        print(f"[maze-check] fluid voxels={n_fluid}, connected components (6-neigh)={int(ncc)}")
        print("=" * 60)
    except Exception:
        pass

    # records the 'same label at both ends' test of each method M
    two_side_checks = {}
    cell_cut_reports = {}

    # ---------------------------------------------------------
    # [helper] find fluid points on the two sides of a solid wall in the maze
    # ---------------------------------------------------------
    def _find_blocked_pair(mask_np, prefer_axis="y", prefer_index=None, min_wall_len=2):
        """
        Returns (axis, index, z_row, x_left, x_right, wall_len)
        Searches the slices with axis='y' (the (z,x) plane) first and
        finds a segment fluid(x_left) | solid-run | fluid(x_right)
        with a solid run of length >= min_wall_len.
        """
        mask_np = np.asarray(mask_np, dtype=bool)
        D_, H_, W_ = mask_np.shape

        # search y slices (as in the debug / visualisation code)
        if prefer_axis.lower() != "y":
            raise ValueError("This helper currently supports prefer_axis='y' only (to match your debug axis).")

        if prefer_index is None:
            prefer_index = H_ // 2

        # search order in y: from the centre outwards
        ys = list(range(H_))
        ys.sort(key=lambda yy: abs(yy - int(prefer_index)))

        best = None  # (wall_len, y, z, x_left, x_right)
        for yy in ys:
            for zz in range(D_):
                line = mask_np[zz, yy, :]  # (x,)
                in_solid = False
                start = None
                for x in range(W_):
                    if (not line[x]) and (not in_solid):
                        in_solid = True
                        start = x
                    elif line[x] and in_solid:
                        end = x - 1
                        in_solid = False
                        if start is None:
                            continue
                        # solid run [start, end]; both sides must be fluid
                        if start <= 0 or end >= (W_ - 1):
                            continue
                        if (not line[start - 1]) or (not line[end + 1]):
                            continue
                        wall_len = end - start + 1
                        if wall_len < int(min_wall_len):
                            continue
                        cand = (wall_len, yy, zz, start - 1, end + 1)
                        if (best is None) or (cand[0] > best[0]):
                            best = cand
                # runs ending in solid at the end of the line are ignored (no fluid point on the right)
        if best is None:
            return None
        wall_len, yy, zz, xL, xR = best
        return ("y", int(yy), int(zz), int(xL), int(xR), int(wall_len))

    # ---------------------------------------------------------
    # [0] M1: Euclidean Voronoi + clipping baseline (GPU)
    # ---------------------------------------------------------
    print("\n[0] Running M1 baseline: Euclidean Voronoi + clipping (GPU)...")
    t0 = time.time()
    label_euc_cp, dist_euc_cp, euc_evt = m1_euclidean_voronoi_clipping_gpu(
        mask, seeds,
        kernel=None,
        profile_gpu=True,
    )

    cp.cuda.Device().synchronize()
    t1 = time.time()
    euc_wall = t1 - t0
    euc_evt = float(euc_evt) if euc_evt is not None else 0.0
    print(f"    M1 (Euclidean+clipping) done in {euc_wall:.3f} s")

    label_euc = label_euc_cp.get()
    # [ANCHOR] Cell-cut check for M1 (label-based)
    s_m1, d_m1 = analyze_cell_cut_by_label_3d(
        mask, label_euc,
        connectivity=26,
        min_component_voxels=16,
        min_component_fraction=0.01,
        report_topk=10,
        return_details=True,
        name="M1 Euclidean+clipping",
    )
    cell_cut_reports["M1"] = (s_m1, d_m1)

    dist_euc = dist_euc_cp.get()

    # ---------------------------------------------------------
    # two-side probe: find points separated by a solid wall automatically (instead of the fixed 39/56 of the thin wall)
    # ---------------------------------------------------------
    probe = _find_blocked_pair(mask, prefer_axis="y", prefer_index=H // 2, min_wall_len=2)
    if probe is None:
        _skip_two_side_check = True
        axis_probe = "y"
        index_probe = H // 2
        z_row = D // 2
        x_left_of_wall = max(0, W // 2 - 2)
        x_right_of_wall = min(W - 1, W // 2 + 2)
        print("[two-side-check] WARNING: cannot find any (fluid|solid-wall|fluid) pattern in maze; "
              "skip two-side checks for all methods.")
    else:
        _skip_two_side_check = False
        axis_probe, index_probe, z_row, x_left_of_wall, x_right_of_wall, wall_len = probe
        print(f"[two-side-check] auto probe: axis={axis_probe}, index={index_probe}, z_row={z_row}, "
              f"x_left={x_left_of_wall}, x_right={x_right_of_wall}, wall_len={wall_len}")

    # ---------------------------------------------------------
    # 'same label at both ends' test for M1 (both ends, no through-layer in between)
    # ---------------------------------------------------------
    if not _skip_two_side_check:
        two_side_checks["M1"] = check_obstacle_two_sides_same_label_blocked(
            mask, label_euc,
            z=z_row, y=index_probe,
            x_left=x_left_of_wall, x_right=x_right_of_wall,
            name="M1 Euclidean+clipping",
            require_blocked_between=True,
            verbose=True,
        )

    # debug output (same structure, coordinates adapted)
    debug_compare_two_slice_points_same_label(
        mask, label_euc,
        axis="y", index=index_probe,
        pA=(z_row, x_left_of_wall),
        pB=(z_row, x_right_of_wall),
        nameA="left_of_wall",
        nameB="right_of_wall",
    )

    debug_probe_slice_point(mask, label_euc, dist_euc, axis="y", index=index_probe,
                            row=z_row, col=x_left_of_wall, r=2, print_dist=False)
    mid_col = int((x_left_of_wall + x_right_of_wall) // 2)
    debug_probe_slice_point(mask, label_euc, dist_euc, axis="y", index=index_probe,
                            row=z_row, col=mid_col, r=2, print_dist=False)
    debug_probe_slice_point(mask, label_euc, dist_euc, axis="y", index=index_probe,
                            row=z_row, col=x_right_of_wall, r=2, print_dist=False)

    debug_dump_slice_patch(
        mask, label_euc, dist_euc,
        axis="y", index=index_probe,
        row_range=(max(0, z_row - 12), min(D, z_row + 12)),
        col_range=(max(0, min(x_left_of_wall, x_right_of_wall) - 12), min(W, max(x_left_of_wall, x_right_of_wall) + 12)),
        print_dist=False,
    )

    debug_probe_slice_point(
        mask, label_euc, dist_euc,
        axis="y", index=index_probe,
        row=z_row, col=x_left_of_wall,
        r=3,
        print_dist=True,
    )

    # ---------------------------------------------------------
    # [1] exact (wall + event)
    # ---------------------------------------------------------
    print("\n[1] Running exact GPU geodesic Voronoi (pure CuPy relaxation)...")
    t2 = time.time()
    label_exact_cp, dist_exact_cp = exact_geodesic_voronoi_gpu(
        mask, seeds,
        connectivity=26,
        max_iter=None,
        eps=1e-6,
        verbose=False,
        profile_gpu=True,
    )
    cp.cuda.Device().synchronize()
    t3 = time.time()
    exact_wall = t3 - t2
    exact_evt = float(EXACT_LAST_GPU_TIME) if EXACT_LAST_GPU_TIME is not None else 0.0
    print(f"    GPU exact solver done in {exact_wall:.3f} s")

    label_exact = label_exact_cp.get()
    s_ex, d_ex = analyze_cell_cut_by_label_3d(
        mask, label_exact,
        connectivity=26,
        min_component_voxels=16,
        min_component_fraction=0.01,
        report_topk=10,
        return_details=True,
        name="Exact Geodesic",
    )
    cell_cut_reports["Exact"] = (s_ex, d_ex)

    dist_exact = dist_exact_cp.get()

    # 'same label at both ends' test for Exact
    if not _skip_two_side_check:
        two_side_checks["Exact"] = check_obstacle_two_sides_same_label_blocked(
            mask, label_exact,
            z=z_row, y=index_probe,
            x_left=x_left_of_wall, x_right=x_right_of_wall,
            name="Exact Geodesic (GPU)",
            require_blocked_between=True,
            verbose=True,
        )

    # ---------------------------------------------------------
    # [2] Full OA-JFA
    # ---------------------------------------------------------
    print("\n[2] Running Full OA-JFA (without ROI optimization)...")
    oajfa_kernel = build_oajfa_kernel()
    t4 = time.time()
    label_full_cp, dist_full_cp = geodesic_voronoi_oajfa_cuda(
        mask, seeds,
        kernel=oajfa_kernel,
        n_relax=1,
        relax_eps=1e-6,
        dump_per_step=False,
        dump_prefix="oajfa_labels_maze",
        profile_gpu=True,
    )
    cp.cuda.Device().synchronize()
    t5 = time.time()
    oajfa_wall = t5 - t4
    oajfa_evt = float(OAJFA_LAST_GPU_TIME) if OAJFA_LAST_GPU_TIME is not None else 0.0
    print(f"    Full OA-JFA done in {oajfa_wall:.3f} s")

    label_full = label_full_cp.get()
    dist_full = dist_full_cp.get()
    s_oa, d_oa = analyze_cell_cut_by_label_3d(
        mask, label_full,
        connectivity=26,
        min_component_voxels=16,
        min_component_fraction=0.01,
        report_topk=10,
        return_details=True,
        name="Full OA-JFA",
    )
    cell_cut_reports["Full OA-JFA"] = (s_oa, d_oa)

    # 'same label at both ends' test for full OA-JFA
    if not _skip_two_side_check:
        two_side_checks["Full OA-JFA"] = check_obstacle_two_sides_same_label_blocked(
            mask, label_full,
            z=z_row, y=index_probe,
            x_left=x_left_of_wall, x_right=x_right_of_wall,
            name="Full OA-JFA",
            require_blocked_between=True,
            verbose=True,
        )

    # ---------------------------------------------------------
    # [3] ROI-JFA
    # ---------------------------------------------------------
    print("\n[3] Running ROI-JFA (full methodology)...")

    stamping_kernel = build_seed_stamping_los_parallel_packed_kernel()
    tiles_dual_kernel = build_tiles_dual_3d_kernel()
    mark_roi_tiles_kernel = build_mark_roi_tiles_kernel_3d()
    roi_step_kernels = build_geodesic_roi_jfa_step_kernels_3d()
    active_tiles_kernel = build_active_tiles_kernel_3d()
    _ = build_local_relax_kernel()

    # warm-up: removes NVRTC compilation and first-call jitter (as in run_demo_case)
    warm_mask = np.ones((16, 16, 16), dtype=bool)
    warm_seeds = np.array([[6, 6, 6], [10, 10, 10]], dtype=np.int64)
    _ = geodesic_voronoi_roi_jfa(
        warm_mask, warm_seeds,
        tile_size=(8, 8, 8),
        delta_r=1.0,
        eta_max=0.8,
        r_tile=1,
        enable_stamping=True,
        verbose=False,
        dump_active_tiles=False,
        dump_label_3d=False,
        viz_policy="none",
        n_relax_after=1,
        relax_eps=1e-6,
        changed_mode="frontier",
        profile_gpu=False,
        return_records=False,
        stamping_kernel=stamping_kernel,
        tiles_dual_kernel=tiles_dual_kernel,
        roi_step_kernels=roi_step_kernels,
        active_tiles_kernel=active_tiles_kernel,
        mark_roi_tiles_kernel=mark_roi_tiles_kernel,
    )
    cp.cuda.Device().synchronize()

    tile_size = (8, 8, 16)

    t6 = time.time()
    out = geodesic_voronoi_roi_jfa(
        mask, seeds,
        tile_size=tile_size,
        delta_r=1.0,
        eta_max=0.8,
        r_tile=1,
        enable_stamping=True,
        verbose=True,
        max_refine_iters=32,
        dump_active_tiles=True,
        dump_prefix="roi_tiles_maze",
        dump_label_3d=True,
        viz_policy="deferred",
        n_relax_after=1,
        relax_eps=1e-6,
        changed_mode="frontier",
        profile_gpu=True,
        return_records=True,
        stamping_kernel=stamping_kernel,
        tiles_dual_kernel=tiles_dual_kernel,
        roi_step_kernels=None,               # as in run_demo_case: the actual run passes None
        active_tiles_kernel=active_tiles_kernel,
        mark_roi_tiles_kernel=mark_roi_tiles_kernel,
    )
    cp.cuda.Device().synchronize()
    t7 = time.time()

    label_roi_cp, dist_roi_cp, tile_roi_cp, roi_mask_flat_cp, records = out
    label_roi = label_roi_cp.get()
    dist_roi = dist_roi_cp.get()
    s_roi, d_roi = analyze_cell_cut_by_label_3d(
        mask, label_roi,
        connectivity=26,
        min_component_voxels=16,
        min_component_fraction=0.01,
        report_topk=10,
        return_details=True,
        name="ROI-JFA",
    )
    cell_cut_reports["ROI-JFA"] = (s_roi, d_roi)

    roi_wall = t7 - t6
    roi_evt = float(ROI_JFA_LAST_GPU_TIME) if ROI_JFA_LAST_GPU_TIME is not None else 0.0
    roi_record = float(ROI_JFA_LAST_RECORD_TIME) if ROI_JFA_LAST_RECORD_TIME is not None else 0.0

    # 'same label at both ends' test for ROI-JFA
    if not _skip_two_side_check:
        two_side_checks["ROI-JFA"] = check_obstacle_two_sides_same_label_blocked(
            mask, label_roi,
            z=z_row, y=index_probe,
            x_left=x_left_of_wall, x_right=x_right_of_wall,
            name="ROI-JFA",
            require_blocked_between=True,
            verbose=True,
        )

    # ---------------------------------------------------------
    # Error statistics
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Error Statistics:")
    print("=" * 60)

    fluid = np.asarray(mask, dtype=bool)
    n_fluid = int(fluid.sum())
    print(f"Total fluid voxels: {n_fluid}")

    print("\n[M1 Euclidean+clipping vs Exact Geodesic]")
    label_diff_euc = (label_euc[fluid] != label_exact[fluid]).sum()
    label_err_euc = label_diff_euc / n_fluid
    print(f"  Label mismatch: {label_diff_euc} ({label_err_euc*100:.4f}%)")

    print("\n[Full OA-JFA vs Exact]")
    label_diff_full = (label_full[fluid] != label_exact[fluid]).sum()
    label_err_full = label_diff_full / n_fluid
    print(f"  Label mismatch: {label_diff_full} ({label_err_full*100:.4f}%)")

    valid_full = fluid & (label_exact >= 0) & (label_full >= 0)
    if valid_full.sum() > 0:
        dist_diff_full = dist_full[valid_full] - dist_exact[valid_full]
        rmse_full = np.sqrt((dist_diff_full**2).mean())
        mae_full = np.abs(dist_diff_full).mean()
        print(f"  Distance RMSE: {rmse_full:.6f}")
        print(f"  Distance MAE:  {mae_full:.6f}")

    print("\n[ROI-JFA vs Exact]")
    label_diff_roi = (label_roi[fluid] != label_exact[fluid]).sum()
    label_err_roi = label_diff_roi / n_fluid
    print(f"  Label mismatch: {label_diff_roi} ({label_err_roi*100:.4f}%)")

    valid_roi = fluid & (label_exact >= 0) & (label_roi >= 0)
    if valid_roi.sum() > 0:
        dist_diff_roi = dist_roi[valid_roi] - dist_exact[valid_roi]
        rmse_roi = np.sqrt((dist_diff_roi**2).mean())
        mae_roi = np.abs(dist_diff_roi).mean()
        print(f"  Distance RMSE: {rmse_roi:.6f}")
        print(f"  Distance MAE:  {mae_roi:.6f}")

    print("\n" + "=" * 60)
    print("Cell-cut summary (label connectivity within pore):")
    print("=" * 60)
    for key in ["M1", "Exact", "Full OA-JFA", "ROI-JFA"]:
        if key not in cell_cut_reports:
            continue
        s, _ = cell_cut_reports[key]
        print(f"  {key}: unassigned_in_fluid={s['n_unassigned_in_fluid']}, "
              f"cut_labels={s['n_cut_labels']}")

    # ---------------------------------------------------------
    # Two-side obstacle check summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Two-side obstacle checks (blocked probe, no pass layer between sides):")
    print("=" * 60)
    if _skip_two_side_check:
        print("  (SKIPPED) Cannot find a blocked probe line at y_probe for maze.")
    else:
        for key in ["M1", "Exact", "Full OA-JFA", "ROI-JFA"]:
            if key not in two_side_checks:
                continue
            r = two_side_checks[key]
            print(f"  {key}: blocked_between={r['blocked_between']}, same_label={r['same_label']} "
                  f"(L={r['left_label']}, R={r['right_label']}, z={r['z']}, y={r['y']})")

    # ---------------------------------------------------------
    # Timing Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Timing Summary:")
    print("=" * 60)

    print(f"  M1 Euclidean wall:  {euc_wall:.6f} s")
    print(f"  M1 Euclidean event: {euc_evt:.6f} s")

    print(f"  Exact wall:         {exact_wall:.6f} s")
    print(f"  Exact event:        {exact_evt:.6f} s")

    print(f"  OA-JFA wall:        {oajfa_wall:.6f} s")
    print(f"  OA-JFA event:       {oajfa_evt:.6f} s")

    print(f"  ROI-JFA wall (solver+record, no viz): {roi_wall:.6f} s")
    print(f"    - ROI-JFA record time (snapshots):  {roi_record:.6f} s")
    print(f"  ROI-JFA event:      {roi_evt:.6f} s")

    # ---------------------------------------------------------
    # Visualization + Saving 4-method final outputs
    # ---------------------------------------------------------
    if visualize:
        # 0) save a plain mask slice to check the maze visually
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.imshow(np.asarray(mask, dtype=bool)[:, index_probe, :], origin="lower", interpolation="nearest")
            plt.title(f"Maze mask slice (axis=y, index={index_probe})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("maze_mask_slice.png", dpi=300)
            plt.close()
            print("\n[0a] Maze mask slice saved to: maze_mask_slice.png")
        except Exception:
            pass

        # 1) 2D results of the 4 methods (PNG)
        print("\n[viz] Saving 2D slices for 4 methods ...")
        visualize_exact_geodesic_slice(
            mask, label_euc_cp, dist_euc_cp,
            seeds=seeds, axis="y", index=index_probe,
            figsize=(10, 4.5),
            save_path="maze_M1_euclidean_clipping_slice.png",
            dpi=300, draw_boundaries=False, draw_seeds=False,
            tessellation_title="M1 Euclidean Voronoi (clipped by mask)",
            distance_title="M1 Euclidean Distance Field",
            distance_cbar_label="Euclidean Distance",
        )
        visualize_exact_geodesic_slice(
            mask, label_exact_cp, dist_exact_cp,
            seeds=seeds, axis="y", index=index_probe,
            figsize=(10, 4.5),
            save_path="maze_M2_exact_geodesic_slice.png",
            dpi=300, draw_boundaries=False, draw_seeds=False,
            tessellation_title="M2 Exact Geodesic Voronoi",
            distance_title="M2 Exact Geodesic Distance",
            distance_cbar_label="Geodesic Distance",
        )
        visualize_exact_geodesic_slice(
            mask, label_full_cp, dist_full_cp,
            seeds=seeds, axis="y", index=index_probe,
            figsize=(10, 4.5),
            save_path="maze_M3_full_oajfa_slice.png",
            dpi=300, draw_boundaries=False, draw_seeds=False,
            tessellation_title="M3 Full OA-JFA",
            distance_title="M3 Geodesic Distance (OA-JFA)",
            distance_cbar_label="Geodesic Distance",
        )
        visualize_exact_geodesic_slice(
            mask, label_roi_cp, dist_roi_cp,
            seeds=seeds, axis="y", index=index_probe,
            figsize=(10, 4.5),
            save_path="maze_M4_roi_jfa_slice.png",
            dpi=300, draw_boundaries=False, draw_seeds=False,
            tessellation_title="M4 ROI-JFA",
            distance_title="M4 Geodesic Distance (ROI-JFA)",
            distance_cbar_label="Geodesic Distance",
        )

        # 2) rendering of the ROI-JFA deferred records (as in run_demo_case)
        print("\n[3b] Rendering deferred ROI-JFA visualizations (after solver)...")
        viz_time = render_roi_jfa_records(
            mask,
            records,
            dump_active_tiles=True,
            dump_label_3d=True,
            tile_size=tile_size,
            axis_for_2d="y",
            index_for_2d=index_probe,
        )
        print(f"  Deferred ROI-JFA viz time: {viz_time:.3f} s")

        # 3) 3D results of the 4 methods (HTML)
        print("\n[viz] Saving 3D HTML for 4 methods ...")
        visualize_exact_geodesic_3d(
            mask, label_euc_cp, dist_euc_cp, seeds,
            html_path="maze_M1_euclidean_clipped_3d.html",
            color_mode="label", smoothing_sigma=1.0, mesh_simplify_factor=2,
        )
        visualize_exact_geodesic_3d(
            mask, label_exact_cp, dist_exact_cp, seeds,
            html_path="maze_M2_exact_geodesic_3d.html",
            color_mode="label", smoothing_sigma=1.0, mesh_simplify_factor=2,
        )
        visualize_exact_geodesic_3d(
            mask, label_full_cp, dist_full_cp, seeds,
            html_path="maze_M3_full_oajfa_3d.html",
            color_mode="label", smoothing_sigma=1.0, mesh_simplify_factor=2,
        )
        visualize_exact_geodesic_3d(
            mask, label_roi_cp, dist_roi_cp, seeds,
            html_path="maze_M4_roi_jfa_3d.html",
            color_mode="label", smoothing_sigma=1.0, mesh_simplify_factor=2,
        )

        print("\n[viz] Saved files:")
        print("  - maze_mask_slice.png")
        print("  - maze_M1_euclidean_clipping_slice.png")
        print("  - maze_M2_exact_geodesic_slice.png")
        print("  - maze_M3_full_oajfa_slice.png")
        print("  - maze_M4_roi_jfa_slice.png")
        print("  - maze_M1_euclidean_clipped_3d.html")
        print("  - maze_M2_exact_geodesic_3d.html")
        print("  - maze_M3_full_oajfa_3d.html")
        print("  - maze_M4_roi_jfa_3d.html")

    print("\nDone.")


def run_demo_maze_case_metrics_only():
    import time
    import numpy as np
    import cupy as cp
    from scipy import ndimage

    # ----------------------------
    # Case C parameters
    # ----------------------------
    D, H, W = 64, 64, 96
    n_seeds = 20
    corridor_width = 3
    wall_thickness = 2
    loop_prob = 0.05
    seed_min_sep_cells = 2

    # ----------------------------
    # Build maze + seeds
    # ----------------------------
    mask, seeds = make_3d_maze_case(
        D=D, H=H, W=W,
        n_seeds=n_seeds,
        seed_mode="cell_centers_poisson",
        seed_random_state=0,
        corridor_width=corridor_width,
        wall_thickness=wall_thickness,
        loop_prob=loop_prob,
        seed_min_sep_cells=seed_min_sep_cells,
        ensure_connected=True,
        cc_connectivity=6,
        verbose=True,
    )
    assert_seeds_valid_and_in_pore(mask, seeds, backend="numpy")

    # ----------------------------
    # M0: Euclidean + clipping
    # ----------------------------
    t0 = time.time()
    label_euc_cp, dist_euc_cp, _ = m1_euclidean_voronoi_clipping_gpu(mask, seeds, profile_gpu=True)
    cp.cuda.Device().synchronize()
    t_euc = time.time() - t0
    label_euc = label_euc_cp.get()

    # ----------------------------
    # M1: Exact (reference)
    # ----------------------------
    t1 = time.time()
    label_exact_cp, dist_exact_cp = exact_geodesic_voronoi_gpu(mask, seeds, connectivity=26, profile_gpu=True)
    cp.cuda.Device().synchronize()
    t_exact = time.time() - t1
    label_exact = label_exact_cp.get()

    # ----------------------------
    # M2: Full OA-JFA
    # ----------------------------
    t2 = time.time()
    label_full_cp, dist_full_cp = geodesic_voronoi_oajfa_cuda(mask, seeds, profile_gpu=True)
    cp.cuda.Device().synchronize()
    t_oajfa = time.time() - t2
    label_full = label_full_cp.get()

    # ----------------------------
    # M3: ROI-JFA (benchmark mode: no dump/records)
    # ----------------------------
    stamping_kernel = build_seed_stamping_los_parallel_packed_kernel()
    tiles_dual_kernel = build_tiles_dual_3d_kernel()
    active_tiles_kernel = build_active_tiles_kernel_3d()

    tile_size = (8, 8, 16)

    label_roi_cp, dist_roi_cp, tile_roi_cp, roi_mask_flat_cp = geodesic_voronoi_roi_jfa(
        mask, seeds,
        tile_size=tile_size,
        delta_r=1.0,
        enable_stamping=True,
        verbose=False,
        max_refine_iters=32,
        dump_active_tiles=False,
        dump_label_3d=False,
        viz_policy="none",
        n_relax_after=1,
        relax_eps=1e-6,
        changed_mode="frontier",
        profile_gpu=True,
        return_records=False,
        stamping_kernel=stamping_kernel,
        tiles_dual_kernel=tiles_dual_kernel,
        roi_step_kernels=None,
        active_tiles_kernel=active_tiles_kernel,
    )
    cp.cuda.Device().synchronize()
    label_roi = label_roi_cp.get()

    # ----------------------------
    # Metrics (single output)
    # ----------------------------
    Vmask = compute_reachable_V(mask, seeds, connectivity=6)

    # statistics parameters (one set for all cases)
    conn_cc = 6
    min_comp_vox = 16
    min_comp_frac = 0.01

    eta_roi = compute_eta_roi(roi_mask_flat_cp.get(), Vmask)

    def pack(method, Lpred, t_pred, t_stamp=0.0, t_jfa=None, eta=np.nan):
        r_unf = compute_r_unf(Lpred, Vmask)
        r_cut, rho_isl = compute_cell_cut_rate_and_island_ratio(
            Lpred, seeds, Vmask,
            connectivity=conn_cc,
            min_component_voxels=min_comp_vox,
            min_component_fraction=min_comp_frac,
        )
        e_vox = compute_e_vox(Lpred, label_exact, Vmask)
        if t_jfa is None:
            t_jfa = t_pred - t_stamp
        return (method, r_cut, r_unf, rho_isl, e_vox, eta, t_pred, t_stamp, t_jfa)

    rows = []
    rows.append(pack("M0 Euclid", label_euc,  t_euc, 0.0, t_euc, np.nan))
    rows.append(pack("M1 Exact",  label_exact, t_exact, 0.0, t_exact, np.nan))
    rows.append(pack("M2 OA-JFA", label_full, t_oajfa, 0.0, t_oajfa, np.nan))
    rows.append(pack(
        "M3 ROI-JFA",
        label_roi,
        ROI_JFA_LAST_TPRED_WALL,
        ROI_JFA_LAST_TSTAMP_WALL,
        ROI_JFA_LAST_TJFA_WALL,
        eta_roi
    ))

    print("\n" + "=" * 110)
    print("[Case C] Summary metrics (single output)")
    print(f"V: reachable from any seed (6-neigh).  Topology connectivity: {conn_cc}-neigh.")
    print(f"Component filter: min_component_voxels={min_comp_vox}, min_component_fraction={min_comp_frac}")
    print("-" * 110)
    print("{:10s} | {:>7s} {:>7s} {:>10s} {:>10s} {:>8s} | {:>9s} {:>9s} {:>9s}".format(
        "Method", "r_cut", "r_unf", "rho_isl", "e_vox", "etaROI", "t_pred", "t_stamp", "t_jfa"
    ))
    print("-" * 110)
    for (m, rcut, runf, rho, evox, eta, tp, ts, tj) in rows:
        print("{:10s} | {:7.4f} {:7.4f} {:10.4f} {:10.4f} {:8.4f} | {:9.4f} {:9.4f} {:9.4f}".format(
            m, rcut, runf, rho, evox, eta, tp, ts, tj
        ))
    print("=" * 110 + "\n")

def run_case_c_maze_metrics_once(
    D=64, H=64, W=96,
    n_seeds=20,
    corridor_width=3,
    wall_thickness=2,
    loop_prob=0.05,
    seed_min_sep_cells=2,
    seed_random_state=0,
    tile_size=(8, 8, 16),
    min_component_voxels=16,
    min_component_fraction=0.01,
):
    import time
    import numpy as np
    import cupy as cp

    # --------------------------
    # Warm-up (avoid first-call NVRTC/alloc jitter in wall-time)
    # --------------------------
    warm_mask = np.ones((16, 16, 16), dtype=bool)
    warm_seeds = np.array([[6, 6, 6], [10, 10, 10]], dtype=np.int64)

    _ = m1_euclidean_voronoi_clipping_gpu(warm_mask, warm_seeds, profile_gpu=False)
    _ = geodesic_voronoi_oajfa_cuda(warm_mask, warm_seeds, profile_gpu=False)
    _ = exact_geodesic_voronoi_gpu(warm_mask, warm_seeds, connectivity=26, profile_gpu=False)

    cp.cuda.Device().synchronize()

    # --------------------------
    # build Case C
    # --------------------------
    mask, seeds = make_3d_maze_case(
        D=D, H=H, W=W,
        n_seeds=n_seeds,
        seed_mode="cell_centers_poisson",
        seed_random_state=seed_random_state,
        corridor_width=corridor_width,
        wall_thickness=wall_thickness,
        loop_prob=loop_prob,
        seed_min_sep_cells=seed_min_sep_cells,
        ensure_connected=True,
        cc_connectivity=6,
        verbose=False,
    )
    assert_seeds_valid_and_in_pore(mask, seeds, backend="numpy")

    Vmask = compute_reachable_V(mask, seeds, connectivity=6)

    # --------------------------
    # M0: Euclidean+clipping (wall time)
    # --------------------------
    t0 = time.time()
    label_euc_cp, dist_euc_cp, _ = m1_euclidean_voronoi_clipping_gpu(mask, seeds, profile_gpu=False)
    cp.cuda.Device().synchronize()
    t_m0 = time.time() - t0
    label_m0 = label_euc_cp.get()

    # --------------------------
    # M1: Exact geodesic reference (wall time)
    # --------------------------
    t1 = time.time()
    label_exact_cp, dist_exact_cp = exact_geodesic_voronoi_gpu(
        mask, seeds, connectivity=26, max_iter=None, eps=1e-6,
        verbose=False, profile_gpu=False,
    )
    cp.cuda.Device().synchronize()
    t_m1 = time.time() - t1
    label_m1 = label_exact_cp.get()

    # --------------------------
    # M2: Full OA-JFA (wall time)
    # --------------------------
    t2 = time.time()
    label_full_cp, dist_full_cp = geodesic_voronoi_oajfa_cuda(
        mask, seeds, kernel=None,
        n_relax=1, relax_eps=1e-6,
        dump_per_step=False,
        profile_gpu=False,
    )
    cp.cuda.Device().synchronize()
    t_m2 = time.time() - t2
    label_m2 = label_full_cp.get()

    # --------------------------
    # M3-pre: ROI-JFA w/o closure (pre-closure metrics)
    # key: max_refine_iters=0 AND enable_closure=False AND n_relax_after=0
    # --------------------------
    stamping_kernel = build_seed_stamping_los_parallel_packed_kernel()
    tiles_dual_kernel = build_tiles_dual_3d_kernel()
    active_tiles_kernel = build_active_tiles_kernel_3d()

    label_pre_cp, dist_pre_cp, _, roi_mask_flat_cp = geodesic_voronoi_roi_jfa(
        mask, seeds,
        tile_size=tile_size,
        delta_r=1.0,
        enable_stamping=True,
        verbose=False,
        dump_active_tiles=False,
        dump_label_3d=False,
        viz_policy="none",
        n_relax_after=0,             # prevent relax mixing into pre
        relax_eps=1e-6,
        changed_mode="frontier",
        profile_gpu=False,
        return_records=False,
        stamping_kernel=stamping_kernel,
        tiles_dual_kernel=tiles_dual_kernel,
        roi_step_kernels=None,
        active_tiles_kernel=active_tiles_kernel,
        max_refine_iters=0,          # skip jump=1 stage
        enable_closure=False,        # disable closure
    )
    cp.cuda.Device().synchronize()

    label_m3_pre = label_pre_cp.get()
    t_m3_pre_pred   = float(ROI_JFA_LAST_TPRED_WALL)
    t_m3_pre_stamp  = float(ROI_JFA_LAST_TSTAMP_WALL)
    t_m3_pre_jfa    = float(ROI_JFA_LAST_TJFA_WALL)
    t_m3_pre_close  = float(ROI_JFA_LAST_TCLOSE_WALL)   # should be 0
    t_m3_pre_relax  = float(ROI_JFA_LAST_TRELAX_WALL)   # should be 0

    # --------------------------
    # etaROI/etaStamp must be computed from PRE roi_mask (strict definition)
    # --------------------------
    eta_roi = compute_eta_roi(roi_mask_flat_cp.get(), Vmask)
    eta_stamp = (np.nan if not np.isfinite(eta_roi) else float(1.0 - eta_roi))

    # --------------------------
    # M3: ROI-JFA with closure (final)
    # --------------------------
    label_roi_cp, dist_roi_cp, _, roi_mask_flat_cp2 = geodesic_voronoi_roi_jfa(
        mask, seeds,
        tile_size=tile_size,
        delta_r=1.0,
        enable_stamping=True,
        verbose=False,
        dump_active_tiles=False,
        dump_label_3d=False,
        viz_policy="none",
        n_relax_after=32,
        relax_eps=1e-6,
        changed_mode="frontier",
        profile_gpu=False,
        return_records=False,
        stamping_kernel=stamping_kernel,
        tiles_dual_kernel=tiles_dual_kernel,
        roi_step_kernels=None,
        active_tiles_kernel=active_tiles_kernel,
        max_refine_iters=32,
        enable_closure=True,
    )
    cp.cuda.Device().synchronize()

    label_m3 = label_roi_cp.get()

    t_m3_pred   = float(ROI_JFA_LAST_TPRED_WALL)
    t_m3_stamp  = float(ROI_JFA_LAST_TSTAMP_WALL)
    t_m3_jfa    = float(ROI_JFA_LAST_TJFA_WALL)
    t_m3_close  = float(ROI_JFA_LAST_TCLOSE_WALL)
    t_m3_relax  = float(ROI_JFA_LAST_TRELAX_WALL)

    import numpy as np
    import math
    from scipy import ndimage
    
    def _per_seed_cut_stats(
        label, seeds, Vmask,
        *,
        connectivity=6,
        min_component_voxels=16,
        min_component_fraction=0.01,
        connect_through_unlabeled=False,
        label_for_threshold=None,
    ):
        """
        Returns dict[k] -> per-seed statistics (only for the k inside V whose seed label is correct)
          {
            'B': number of voxels of this label in V (from label)
            'B_thr': B used for the threshold (from label_for_threshold, or B)
            'thr': threshold
            'kept_sizes': sizes of the connected components kept after filtering (counted as voxels with label==k), descending
            'n_kept': number of components kept
            'cut': n_kept>=2
            'seed_comp_size': voxels with label==k in the component that contains the seed
            'island_ratio': islands / max(seed_comp_size,1)
          }
        """
        L = np.asarray(label, dtype=np.int32)
        V = np.asarray(Vmask, dtype=bool)
        seeds_np = np.asarray(seeds, dtype=np.int64)
    
        Lthr = None
        if label_for_threshold is not None:
            Lthr = np.asarray(label_for_threshold, dtype=np.int32)
            if Lthr.shape != L.shape:
                raise ValueError("label_for_threshold must have the same shape as label")
    
        if connectivity == 6:
            struct = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 26:
            struct = ndimage.generate_binary_structure(3, 3)
        else:
            raise ValueError("connectivity must be 6 or 26")
    
        out = {}
        n_seeds = int(seeds_np.shape[0])
    
        for k in range(n_seeds):
            z, y, x = map(int, seeds_np[k])
            if not V[z, y, x]:
                continue
            if int(L[z, y, x]) != k:
                # wrong seed label: not diagnosed here (the worst case is computed elsewhere)
                continue
    
            region_k = V & (L == k)
            B = int(region_k.sum())
            if B <= 0:
                continue
    
            # threshold definition
            if Lthr is None:
                B_thr = B
            else:
                B_ref = int(np.count_nonzero(V & (Lthr == k)))
                B_thr = B_ref if B_ref > 0 else B
    
            thr = int(max(min_component_voxels, math.ceil(float(min_component_fraction) * float(B_thr))))
    
            if not connect_through_unlabeled:
                # strict: connectivity only within (L==k)
                coords = np.argwhere(region_k)
                z0, y0, x0 = coords.min(axis=0)
                z1, y1, x1 = coords.max(axis=0)
                slc = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    
                cc, ncc = ndimage.label(region_k[slc], structure=struct)
                seed_cid = int(cc[z - z0, y - y0, x - x0])
                if seed_cid <= 0:
                    continue
    
                sizes = np.bincount(cc.ravel()).astype(np.int64)
                sizes[0] = 0
    
                kept_sizes = [int(sizes[cid]) for cid in range(1, ncc + 1) if int(sizes[cid]) >= thr]
                kept_sizes.sort(reverse=True)
    
                seed_comp_size = int(sizes[seed_cid])
                islands = sum(s for s in kept_sizes if s != seed_comp_size) if seed_comp_size in kept_sizes else sum(kept_sizes)
    
            else:
                # crack-free: connectivity may pass through -1, but the voxel count includes only (L==k)
                passable = V & ((L == k) | (L < 0))
                coords = np.argwhere(passable)
                z0, y0, x0 = coords.min(axis=0)
                z1, y1, x1 = coords.max(axis=0)
                slc = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    
                cc, ncc = ndimage.label(passable[slc], structure=struct)
                seed_cid = int(cc[z - z0, y - y0, x - x0])
                if seed_cid <= 0:
                    continue
    
                sub_k = region_k[slc]          # voxels with label == k only
                ids = cc[sub_k]                # the passable connected components these voxels fall in
                sizes = np.bincount(ids.ravel(), minlength=ncc + 1).astype(np.int64)
                sizes[0] = 0
    
                kept_sizes = [int(sizes[cid]) for cid in range(1, ncc + 1) if int(sizes[cid]) >= thr]
                kept_sizes.sort(reverse=True)
    
                seed_comp_size = int(sizes[seed_cid])
                islands = sum(s for s in kept_sizes if s != seed_comp_size) if seed_comp_size in kept_sizes else sum(kept_sizes)
    
            n_kept = int(len(kept_sizes))
            cut = bool(n_kept >= 2)
            island_ratio = float(islands) / float(max(seed_comp_size, 1))
    
            out[int(k)] = {
                "B": int(B),
                "B_thr": int(B_thr),
                "thr": int(thr),
                "kept_sizes": kept_sizes[:10],
                "n_kept": int(n_kept),
                "cut": bool(cut),
                "seed_comp_size": int(seed_comp_size),
                "island_ratio": float(island_ratio),
            }
    
        return out

    def diagnose_cut_breakdown(
        label_pre,
        seeds,
        Vmask,
        *,
        label_ref_for_thr=None,   # pass Exact (label_m1) or the ROI-JFA result (label_m3)
        connectivity=6,
        min_component_voxels=16,
        min_component_fraction=0.01,
        topk_print=5,
        name="M3-pre",
    ):
        """
        Print and return the breakdown:
          - crack_cut: cut under strict, not cut under crack-free
          - thr_cut:   cut under strict and crack-free, not cut with the fixed threshold
          - true_cut:  still cut with crack-free + fixed threshold (a truly mislabelled island)
        """
        # strict
        S = _per_seed_cut_stats(
            label_pre, seeds, Vmask,
            connectivity=connectivity,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
            connect_through_unlabeled=False,
            label_for_threshold=None,
        )
        strict_cut = sorted([k for k,v in S.items() if v["cut"]])
    
        if len(strict_cut) == 0:
            print(f"[cut-diagnosis] {name}: strict_cut=0 (no cut labels).")
            return {"strict_cut": [], "crack_cut": [], "thr_cut": [], "true_cut": []}
    
        # crack-free (allow -1 connectivity)
        CF = _per_seed_cut_stats(
            label_pre, seeds, Vmask,
            connectivity=connectivity,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
            connect_through_unlabeled=True,
            label_for_threshold=None,
        )
    
        # fixed threshold (use ref B_k)
        FX = _per_seed_cut_stats(
            label_pre, seeds, Vmask,
            connectivity=connectivity,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
            connect_through_unlabeled=False,
            label_for_threshold=label_ref_for_thr,
        )
        CF_FX = _per_seed_cut_stats(
            label_pre, seeds, Vmask,
            connectivity=connectivity,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
            connect_through_unlabeled=True,
            label_for_threshold=label_ref_for_thr,
        )
    
        crack_cut, thr_cut, true_cut = [], [], []
    
        for k in strict_cut:
            cut_cf = bool(CF.get(k, {}).get("cut", True))          # missing counts as still cut
            cut_true = bool(CF_FX.get(k, {}).get("cut", True))
    
            if not cut_cf:
                crack_cut.append(k)
            elif not cut_true:
                thr_cut.append(k)
            else:
                true_cut.append(k)
    
        print("=" * 110)
        print(f"[cut-diagnosis] {name}")
        print(f"  strict_cut = {len(strict_cut)} : {strict_cut}")
        print(f"  crack_cut  = {len(crack_cut)} : {crack_cut}   (strict cut, but crack-free NOT cut)")
        print(f"  thr_cut    = {len(thr_cut)} : {thr_cut}   (still cut crack-free, but fixed-thr removes)")
        print(f"  true_cut   = {len(true_cut)} : {true_cut}  (still cut even crack-free + fixed-thr)")
        print("-" * 110)
    
        # detail: print only the top-k (usually 1 here)
        show = strict_cut[:int(topk_print)]
        for k in show:
            a = S.get(k, {})
            b = CF.get(k, {})
            c = FX.get(k, {})
            d = CF_FX.get(k, {})
    
            print(f"  seed={k}:")
            print(f"    strict   : cut={a.get('cut')}, B={a.get('B')}, thr={a.get('thr')}, kept={a.get('kept_sizes')}")
            print(f"    crackfree: cut={b.get('cut')}, B={b.get('B')}, thr={b.get('thr')}, kept={b.get('kept_sizes')}")
            if label_ref_for_thr is not None:
                print(f"    fixedthr : cut={c.get('cut')}, B_thr={c.get('B_thr')}, thr={c.get('thr')}, kept={c.get('kept_sizes')}")
                print(f"    cf+fx    : cut={d.get('cut')}, B_thr={d.get('B_thr')}, thr={d.get('thr')}, kept={d.get('kept_sizes')}")
            print("")
    
        print("=" * 110)
        return {
            "strict_cut": strict_cut,
            "crack_cut": crack_cut,
            "thr_cut": thr_cut,
            "true_cut": true_cut,
        }
    
    
    # --------------------------
    # pack metrics
    # --------------------------
    def metrics_row(
        name,
        label_np,
        t_pred,
        t_stamp=0.0,
        t_jfa=None,
        t_close=0.0,
        t_relax=0.0,
        eta_roi_val=np.nan,
        eta_stamp_val=np.nan,
    ):
        r_unf = compute_r_unf(label_np, Vmask)

        # FIX: correct function name
        r_cut, rho_isl = compute_cell_cut_rate_and_island_ratio(
            label_np, seeds, Vmask,
            connectivity=6,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
            connect_through_unlabeled=(name == "M3-pre"),   # enabled for pre only
        )

        e_vox = compute_e_vox(label_np, label_m1, Vmask)

        if t_jfa is None:
            # consistent fallback: exclude stamp/close/relax
            t_jfa = float(t_pred) - float(t_stamp) - float(t_close) - float(t_relax)

        return (
            name, float(r_cut), float(r_unf), float(rho_isl), float(e_vox),
            eta_roi_val, eta_stamp_val,
            float(t_pred), float(t_stamp), float(t_jfa), float(t_close), float(t_relax)
        )

    rows = []
    rows.append(metrics_row("M0 Euclid",  label_m0, t_m0, 0.0, t_m0, 0.0, 0.0, np.nan, np.nan))
    rows.append(metrics_row("M1 Exact",   label_m1, t_m1, 0.0, t_m1, 0.0, 0.0, np.nan, np.nan))
    rows.append(metrics_row("M2 OA-JFA",  label_m2, t_m2, 0.0, t_m2, 0.0, 0.0, np.nan, np.nan))

    rows.append(metrics_row("M3-pre",     label_m3_pre, t_m3_pre_pred, t_m3_pre_stamp, t_m3_pre_jfa, t_m3_pre_close, t_m3_pre_relax, eta_roi, eta_stamp))
    rows.append(metrics_row("M3 ROI-JFA", label_m3,     t_m3_pred,     t_m3_stamp,     t_m3_jfa,     t_m3_close,     t_m3_relax,     eta_roi, eta_stamp))

    # --------------------------
    # single output
    # --------------------------
    print("\n" + "=" * 130)
    print("[Case C] Summary metrics (single output)")
    print("V: reachable from any seed (6-neigh).  Topology connectivity: 6-neigh.")
    print(f"Component filter: min_component_voxels={min_component_voxels}, min_component_fraction={min_component_fraction}")
    print("-" * 130)
    print("{:10s} | {:>7s} {:>7s} {:>9s} {:>9s} {:>8s} {:>9s} | {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "Method", "r_cut", "r_unf", "rho_isl", "e_vox", "etaROI", "etaStamp",
        "t_pred", "t_stamp", "t_jfa", "t_close", "t_relax"
    ))
    print("-" * 130)
    for (name, r_cut, r_unf, rho_isl, e_vox, eta_roi_val, eta_stamp_val, t_pred, t_stamp, t_jfa, t_close, t_relax) in rows:
        print("{:10s} | {:7.4f} {:7.4f} {:9.4f} {:9.4f} {:8.4f} {:9.4f} | {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f}".format(
            name, r_cut, r_unf, rho_isl, e_vox,
            float(eta_roi_val) if np.isfinite(eta_roi_val) else float("nan"),
            float(eta_stamp_val) if np.isfinite(eta_stamp_val) else float("nan"),
            t_pred, t_stamp, t_jfa, t_close, t_relax
        ))
    print("=" * 130 + "\n")


    _ = diagnose_cut_breakdown(
        label_m3_pre, seeds, Vmask,
        label_ref_for_thr=label_m1,     # fixed threshold from the Exact B_k, so that the pre/final thresholds are not coupled
        connectivity=6,
        min_component_voxels=min_component_voxels,
        min_component_fraction=min_component_fraction,
        topk_print=5,
        name="M3-pre",
    )
    return rows


# =============================================================================
# Scaling Experiment with Accuracy Metrics
# accuracy of M3 (ROI-JFA) against exact Dijkstra
# entry point: run_scaling_experiment()
# =============================================================================

import numpy as np
import cupy as cp
import time
import csv

# -----------------------------------------------------------------------------
# accuracy metrics
# -----------------------------------------------------------------------------

def compute_accuracy_metrics(dist_m3, label_m3, dist_exact, label_exact, Vmask):
    """
    Accuracy metrics of M3 against Exact.

    Parameters
    ----------
    dist_m3 : ndarray
        geodesic distance from M3 (ROI-JFA)
    label_m3 : ndarray
        Voronoi label from M3
    dist_exact : ndarray
        geodesic distance from Exact (Dijkstra/relaxation)
    label_exact : ndarray
        Voronoi label from Exact
    Vmask : ndarray (bool)
        mask of the reachable fluid voxels

    Notes
    -----
    Works with numpy and cupy: if any input is on the GPU (cupy.ndarray), the statistics are computed on
    the GPU and only scalars are returned, so dist/label are never copied whole with .get() (usually the
    main reason why the printed times are much shorter than the total time).
    """
    import numpy as np
    try:
        import cupy as cp
    except Exception:
        cp = None

    def _is_cupy(x):
        return (cp is not None) and isinstance(x, cp.ndarray)

    use_gpu = (cp is not None) and (
        _is_cupy(dist_m3) or _is_cupy(label_m3) or _is_cupy(dist_exact) or _is_cupy(label_exact) or _is_cupy(Vmask)
    )

    if use_gpu:
        d_m3 = dist_m3 if _is_cupy(dist_m3) else cp.asarray(dist_m3)
        l_m3 = label_m3 if _is_cupy(label_m3) else cp.asarray(label_m3)
        d_ex = dist_exact if _is_cupy(dist_exact) else cp.asarray(dist_exact)
        l_ex = label_exact if _is_cupy(label_exact) else cp.asarray(label_exact)
        V = Vmask if _is_cupy(Vmask) else cp.asarray(Vmask)

        if V.dtype != cp.bool_:
            V = (V != 0)

        # keep float32 on GPU
        if d_m3.dtype != cp.float32:
            d_m3 = d_m3.astype(cp.float32, copy=False)
        if d_ex.dtype != cp.float32:
            d_ex = d_ex.astype(cp.float32, copy=False)

        denom = cp.count_nonzero(V)
        denom_i = int(denom.get())
        if denom_i == 0:
            return {
                'mae': np.nan,
                'mre': np.nan,
                'max_err': np.nan,
                'rmse': np.nan,
                'acc_rate': 0.0,
                'mismatch_rate': 1.0,
            }

        abs_err = cp.abs(d_m3 - d_ex)

        mae = float((cp.sum(cp.where(V, abs_err, 0.0)) / denom).get())
        rmse = float(cp.sqrt(cp.sum(cp.where(V, abs_err * abs_err, 0.0)) / denom).get())
        max_err = float(cp.max(cp.where(V, abs_err, 0.0)).get())

        valid = V & (d_ex > cp.float32(1e-9))
        denom2 = cp.count_nonzero(valid)
        denom2_i = int(denom2.get())
        if denom2_i > 0:
            mre = float((cp.sum(cp.where(valid, abs_err / d_ex, 0.0)) / denom2).get())
        else:
            mre = 0.0

        acc_rate = float((cp.count_nonzero(V & (l_m3 == l_ex)) / denom).get())
        mismatch_rate = 1.0 - acc_rate

        return {
            'mae': mae,
            'mre': mre,
            'max_err': max_err,
            'rmse': rmse,
            'acc_rate': acc_rate,
            'mismatch_rate': mismatch_rate,
        }

    # -----------------------------
    # numpy fallback (original logic)
    # -----------------------------
    V_indices = np.asarray(Vmask, dtype=bool).ravel()
    d_m3 = np.asarray(dist_m3).ravel()[V_indices]
    d_ex = np.asarray(dist_exact).ravel()[V_indices]
    l_m3 = np.asarray(label_m3).ravel()[V_indices]
    l_ex = np.asarray(label_exact).ravel()[V_indices]

    n_V = int(V_indices.sum())
    abs_err = np.abs(d_m3 - d_ex)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    max_err = float(np.max(abs_err))

    valid_mask = d_ex > 1e-9
    if valid_mask.sum() > 0:
        rel_err = abs_err[valid_mask] / d_ex[valid_mask]
        mre = float(np.mean(rel_err))
    else:
        mre = 0.0

    label_match = (l_m3 == l_ex)
    acc_rate = float(label_match.sum()) / n_V if n_V > 0 else 0.0
    mismatch_rate = 1.0 - acc_rate

    return {
        'mae': mae,
        'mre': mre,
        'max_err': max_err,
        'rmse': rmse,
        'acc_rate': acc_rate,
        'mismatch_rate': mismatch_rate,
    }
# -----------------------------------------------------------------------------
# scaling experiment, main function
# -----------------------------------------------------------------------------

def run_scaling_experiment(
    scale=5,
    seed_start_exp=0.2,
    seed_end_exp=4.0,
    seed_step=0.2,
    random_state=42,
    output_csv="roi_jfa_scaling_with_accuracy.csv",
    run_exact=True,
    max_seeds_for_exact=1000,

    # --- speed/overhead controls (optional) ---
    fast_mode=False,               # True: skip expensive CPU structural metrics by default
    struct_metrics_sample=None,    # None=full (original); 0=skip; int=sample that many seeds (estimate)
    free_memory_each_iter=True,    # original code freed memory pool each iter; disable if it hurts perf
):
    """
    Variant (C + D + Exact=26):
      - fixed maze mask: generated once
      - clearance_kmax27: precomputed once and reused (C)
      - mask: uploaded to the GPU once and reused (no cp.asarray(mask) per pass, no pinned-memory jitter/failure)
      - Exact: connectivity=26 (the 26-metric of ROI-JFA)
      - the CuPy memory pool is released after each pass (D), avoiding slow-down / fragmentation and a late OOM
    """
    import numpy as np
    import cupy as cp
    import time
    import csv
    import math

    # =========================================================================
    # parameters
    # =========================================================================
    base_D, base_H, base_W = 64, 64, 96
    base_corridor = 3
    base_wall = 2

    D = base_D * scale
    H = base_H * scale
    W = base_W * scale
    corridor_width = base_corridor * scale
    wall_thickness = base_wall * scale

    tile_size = (8, 8, 16)  # fixed
    loop_prob = 0.05
    seed_min_sep_cells = 0
    min_comp_vox = 16 * scale
    min_comp_frac = 0.01
    max_refine_iters = 1000

    print("=" * 100)
    print("M3 (ROI-JFA) Scaling Experiment with Accuracy Metrics  [C+D + Exact=26]")
    print("=" * 100)
    print(f"Scale: {scale}x")
    print(f"Domain: {D} x {H} x {W} = {D*H*W:,} voxels")
    print(f"Corridor: {corridor_width}, Wall: {wall_thickness}")
    print(f"Tile size: {tile_size} (fixed)")
    print(f"Run exact: {run_exact} (max {max_seeds_for_exact} seeds)")
    print("=" * 100)

    # Seed counts
    exponents = np.arange(seed_start_exp, seed_end_exp + 0.01, seed_step)
    seed_counts = np.unique(np.round(10.0 ** exponents).astype(int))
    print(f"Seed counts ({len(seed_counts)} points): {seed_counts.tolist()}")
    print("=" * 100)

    # =========================================================================
    # precompile kernels (warm-up)
    # =========================================================================
    print("Pre-compiling CUDA kernels...")
    stamp_kern = build_seed_stamping_los_parallel_packed_kernel()
    tiles_kern = build_tiles_dual_3d_kernel()
    active_kern = build_active_tiles_kernel_3d()

    _m = np.ones((16, 16, 16), dtype=bool)
    _s = np.array([[6, 6, 6], [10, 10, 10]], dtype=np.int64)
    _ = geodesic_voronoi_roi_jfa(
        _m, _s, tile_size=(8, 8, 8), verbose=False,
        viz_policy="none",
        stamping_kernel=stamp_kern,
        tiles_dual_kernel=tiles_kern,
        active_tiles_kernel=active_kern
    )
    cp.cuda.Device().synchronize()
    print("Ready.\n")

    # =========================================================================
    # (C) fixed maze mask: generated once, clearance_kmax27 precomputed once
    # =========================================================================
    print("[C] Building ONE fixed maze mask and precomputing LOS-kmax27 once...")
    t0_mask = time.time()
    mask_np, _seeds_dummy = make_3d_maze_case(
        D=D, H=H, W=W,
        n_seeds=1,
        seed_mode="random",
        seed_random_state=random_state,
        corridor_width=corridor_width,
        wall_thickness=wall_thickness,
        loop_prob=loop_prob,
        seed_min_sep_cells=seed_min_sep_cells,
        ensure_connected=True,
        cc_connectivity=6,
        verbose=False,
    )
    t_mask = time.time() - t0_mask
    mask_np = np.asarray(mask_np, dtype=bool)

    n_fluid = int(mask_np.sum())
    print(f"  mask built in {t_mask:.2f}s, fluid={n_fluid:,}, fluid_ratio={n_fluid/(D*H*W):.4f}")

    # upload the mask to the GPU once; ROI-JFA reuses it (avoids pinned-memory errors)
    mask_cp = cp.asarray(mask_np, dtype=cp.uint8)
    mask_flat_cp = mask_cp.ravel()

    # Vmask: ensure_connected=True (single component), so V == mask
    Vmask = mask_np
    n_V = int(Vmask.sum())

    # clearance_kmax27 computed once
    maxdim = int(max(D, H, W))
    max_k = int(maxdim.bit_length() - 1)

    t0_kmax = time.time()
    clearance_kmax27 = precompute_los_kmax_27dirs_3d(
        mask_flat_cp,
        D, H, W,
        max_k=max_k,
        init_kernel=None,
        update_kernel=None,
        verbose=False,
    )
    cp.cuda.Device().synchronize()
    t_kmax = time.time() - t0_kmax
    print(f"  kmax27 precomputed in {t_kmax:.2f}s  (stored & reused)\n")

    # =========================================================================
    # helper: random linear indices filtered by the mask, instead of np.argwhere(mask) (too large)
    # =========================================================================
    def _sample_random_seeds_from_mask(mask_bool, n_seeds, rng, oversample=4.0):
        mask_flat = mask_bool.ravel()
        nvox = mask_flat.size
        need = int(n_seeds)

        chosen = np.empty((0,), dtype=np.int64)

        # fluid_ratio ~ 0.5, so oversample=4 usually reaches 100k in one pass
        while chosen.size < need:
            batch_n = int(math.ceil((need - chosen.size) * float(oversample)))
            batch = rng.randint(0, nvox, size=batch_n, dtype=np.int64)
            batch = batch[mask_flat[batch]]
            if batch.size == 0:
                continue
            chosen = np.unique(np.concatenate([chosen, batch], axis=0))

        chosen = chosen[:need]

        HW = H * W
        z = chosen // HW
        rem = chosen - z * HW
        y = rem // W
        x = rem - y * W

        seeds = np.stack([z, y, x], axis=1).astype(np.int64, copy=False)
        return seeds

    # =========================================================================
    # experiment loop
    # =========================================================================
    results = []
    rng = np.random.RandomState(int(random_state))

    for i, n_seeds in enumerate(seed_counts):
        n_seeds = int(n_seeds)
        exp_val = float(np.log10(n_seeds)) if n_seeds > 0 else float("nan")

        print(f"\n[{i+1}/{len(seed_counts)}] n_seeds={n_seeds} (10^{exp_val:.1f})")
        print("-" * 80)

        try:
            # -------------------------------------------------------------
            # seeds sampled from the fixed mask (maze not rebuilt)
            # -------------------------------------------------------------
            t0 = time.time()
            seeds = _sample_random_seeds_from_mask(mask_np, n_seeds, rng, oversample=4.0)
            t_build = time.time() - t0

            # strict seed check
            assert_seeds_valid_and_in_pore(mask_np, seeds, backend="numpy")

            print(f"  seeds_sample={t_build:.2f}s, fluid={n_fluid:,}, V={n_V:,}")

            # -------------------------------------------------------------
            # M3: ROI-JFA (reuses mask_cp and clearance_kmax27)
            # -------------------------------------------------------------
            label_m3_cp, dist_m3_cp, _, roi_mask_cp = geodesic_voronoi_roi_jfa(
                mask_cp, seeds,
                tile_size=tile_size,
                delta_r=1.0,
                enable_stamping=True,
                verbose=False,
                viz_policy="none",
                n_relax_after=256,
                relax_eps=1e-6,
                changed_mode="frontier",
                stamping_kernel=stamp_kern,
                tiles_dual_kernel=tiles_kern,
                active_tiles_kernel=active_kern,
                max_refine_iters=max_refine_iters,
                enable_closure=True,
                clearance_kmax27=clearance_kmax27,   # (C) reuse
            )
            cp.cuda.Device().synchronize()

            t_m3 = float(ROI_JFA_LAST_TPRED_WALL)
            t_stamp = float(ROI_JFA_LAST_TSTAMP_WALL)
            t_jfa = float(ROI_JFA_LAST_TJFA_WALL)
            t_close = float(ROI_JFA_LAST_TCLOSE_WALL)
            t_relax = float(ROI_JFA_LAST_TRELAX_WALL)

            # quality metrics (eta_roi / r_unf: GPU statistics, no .get() of large arrays)
            eta_roi = compute_eta_roi(roi_mask_cp, mask_flat_cp)
            eta_stamp = 1.0 - eta_roi if np.isfinite(eta_roi) else np.nan

            r_unf = compute_r_unf(label_m3_cp, mask_cp)

            # r_cut / rho_isl: CPU per-cell connectivity statistics, very slow for large n_seeds.
            #   - struct_metrics_sample=None : full (slowest)
            #   - struct_metrics_sample=0    : skipped (fastest)
            #   - struct_metrics_sample=int  : mean over a random sample of seeds (recommended for large n_seeds)
            r_cut = np.nan
            rho_isl = np.nan
            label_m3 = None

            _struct_sample = struct_metrics_sample
            if bool(fast_mode) and (_struct_sample is None):
                # option 1: in fast_mode, r_cut / rho_isl are estimated from a small sample of seeds rather than skipped
                #         to skip them completely (fastest), pass struct_metrics_sample=0
                _struct_sample = min(int(n_seeds), 1024)

            if _struct_sample is None:
                # full (original, very slow for large n_seeds)
                label_m3 = label_m3_cp.get()
                r_cut, rho_isl = compute_cell_cut_rate_and_island_ratio(
                    label_m3, seeds, Vmask,
                    connectivity=6,
                    min_component_voxels=min_comp_vox,
                    min_component_fraction=min_comp_frac,
                    connect_through_unlabeled=False,
                )
            else:
                _struct_sample = int(_struct_sample)
                if _struct_sample > 0:
                    label_m3 = label_m3_cp.get()
                    if n_seeds <= _struct_sample:
                        seed_indices = None
                    else:
                        seed_indices = rng.choice(n_seeds, size=_struct_sample, replace=False)
                    r_cut, rho_isl = compute_cell_cut_rate_and_island_ratio(
                        label_m3, seeds, Vmask,
                        connectivity=6,
                        min_component_voxels=min_comp_vox,
                        min_component_fraction=min_comp_frac,
                        connect_through_unlabeled=False,
                        seed_indices=seed_indices,
                    )

            print(f"  M3: {t_m3:.3f}s (stamp={t_stamp:.3f}, jfa={t_jfa:.3f}, close={t_close:.3f}, relax={t_relax:.3f})")
            print(f"  r_cut={r_cut:.4f}, r_unf={r_unf:.4f}, rho_isl={rho_isl:.4f}, eta_roi={eta_roi:.4f}")

            # -------------------------------------------------------------
            # Exact (optional) with 26 neighbours (Exact=26)
            # -------------------------------------------------------------
            t_exact = np.nan
            mae = np.nan
            mre = np.nan
            max_err = np.nan
            rmse = np.nan
            acc_rate = np.nan
            mismatch_rate = np.nan

            if run_exact and (n_seeds <= int(max_seeds_for_exact)):
                print("  Computing exact (GPU relaxation) with connectivity=26 ...")
                t0_exact = time.time()

                label_exact_cp, dist_exact_cp = exact_geodesic_voronoi_gpu(
                    mask_np, seeds,
                    connectivity=26,   # 26 neighbours
                    max_iter=None,
                    eps=1e-6,
                    verbose=False,
                    profile_gpu=False,
                )
                cp.cuda.Device().synchronize()
                t_exact = time.time() - t0_exact

                # Accuracy metrics on GPU (avoid dist/label downloads)
                acc_metrics = compute_accuracy_metrics(
                    dist_m3_cp, label_m3_cp, dist_exact_cp, label_exact_cp, mask_cp
                )
                mae = acc_metrics["mae"]
                mre = acc_metrics["mre"]
                max_err = acc_metrics["max_err"]
                rmse = acc_metrics["rmse"]
                acc_rate = acc_metrics["acc_rate"]
                mismatch_rate = acc_metrics["mismatch_rate"]

                speedup = t_exact / t_m3 if t_m3 > 0 else np.nan
                print(f"  Exact: {t_exact:.3f}s (speedup: {speedup:.1f}x)")
                print(f"  Accuracy: MAE={mae:.4f}, MRE={mre:.4f}, max_err={max_err:.2f}")
                print(f"  Label acc_rate={acc_rate:.4f} ({acc_rate*100:.2f}%)")
            elif run_exact:
                print(f"  [SKIP] n_seeds={n_seeds} > max_seeds_for_exact={max_seeds_for_exact}")

            # -------------------------------------------------------------
            # save one row
            # -------------------------------------------------------------
            row = {
                "n_seeds": n_seeds,
                "exponent": round(exp_val, 1),
                "D": D, "H": H, "W": W,
                "n_voxels": D*H*W,
                "n_fluid": n_fluid,
                "n_V": n_V,
                "r_cut": r_cut,
                "r_unf": r_unf,
                "rho_isl": rho_isl,
                "eta_roi": eta_roi,
                "eta_stamp": eta_stamp,
                "t_m3": t_m3,
                "t_stamp": t_stamp,
                "t_jfa": t_jfa,
                "t_close": t_close,
                "t_relax": t_relax,
                "t_build": t_build,
                "t_exact": t_exact,
                "mae": mae,
                "mre": mre,
                "max_err": max_err,
                "rmse": rmse,
                "acc_rate": acc_rate,
                "mismatch_rate": mismatch_rate,
                "status": "OK",
            }

            results.append(row)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            results.append({
                "n_seeds": n_seeds,
                "exponent": round(exp_val, 1),
                "D": D, "H": H, "W": W,
                "n_voxels": D*H*W,
                "n_fluid": np.nan,
                "n_V": np.nan,
                "r_cut": np.nan,
                "r_unf": np.nan,
                "rho_isl": np.nan,
                "eta_roi": np.nan,
                "eta_stamp": np.nan,
                "t_m3": np.nan,
                "t_stamp": np.nan,
                "t_jfa": np.nan,
                "t_close": np.nan,
                "t_relax": np.nan,
                "t_build": np.nan,
                "t_exact": np.nan,
                "mae": np.nan,
                "mre": np.nan,
                "max_err": np.nan,
                "rmse": np.nan,
                "acc_rate": np.nan,
                "mismatch_rate": np.nan,
                "status": f"ERROR: {str(e)[:60]}",
            })

        finally:
            # ---------------------------------------------------------
            # (D) free the idle blocks of the memory pool in each pass (avoids slow-down and fragmentation)
            #     note: resident large objects still referenced (mask_cp / clearance_kmax27) are not freed
            # ---------------------------------------------------------
            try:
                if free_memory_each_iter:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    # =========================================================================
    # summary output
    # =========================================================================
    print("\n" + "=" * 160)
    print(f"SUMMARY: {D}x{H}x{W} (scale={scale}x)")
    print("=" * 160)

    hdr = "{:>6} {:>4} | {:>8} {:>8} {:>8} {:>8} | {:>6} {:>6} {:>6} | {:>8} {:>8} | {:>7} {:>7} {:>7} {:>8}".format(
        "seeds", "10^x",
        "eta_roi", "r_cut", "r_unf", "rho_isl",
        "t_m3", "t_exct", "speed",
        "MAE", "max_err",
        "acc%", "MRE%", "mism%", "RMSE"
    )
    print(hdr)
    print("-" * 160)

    for r in results:
        if r["status"] == "OK":
            speedup = r["t_exact"] / r["t_m3"] if np.isfinite(r["t_exact"]) and r["t_m3"] > 0 else np.nan
            print("{:>6d} {:>4.1f} | {:8.4f} {:8.4f} {:8.4f} {:8.4f} | {:6.2f} {:6.2f} {:6.1f} | {:8.4f} {:8.2f} | {:6.2f}% {:6.2f}% {:6.2f}% {:8.4f}".format(
                int(r["n_seeds"]), float(r["exponent"]),
                float(r["eta_roi"]), float(r["r_cut"]), float(r["r_unf"]), float(r["rho_isl"]),
                float(r["t_m3"]),
                float(r["t_exact"]) if np.isfinite(r["t_exact"]) else 0.0,
                float(speedup) if np.isfinite(speedup) else 0.0,
                float(r["mae"]) if np.isfinite(r["mae"]) else 0.0,
                float(r["max_err"]) if np.isfinite(r["max_err"]) else 0.0,
                float(r["acc_rate"])*100 if np.isfinite(r["acc_rate"]) else 0.0,
                float(r["mre"])*100 if np.isfinite(r["mre"]) else 0.0,
                float(r["mismatch_rate"])*100 if np.isfinite(r["mismatch_rate"]) else 0.0,
                float(r["rmse"]) if np.isfinite(r["rmse"]) else 0.0,
            ))
        else:
            print("{:>6d} {:>4.1f} | -- {} --".format(
                int(r["n_seeds"]), float(r["exponent"]), r["status"][:80]
            ))

    print("=" * 160)

    # =========================================================================
    # save CSV
    # =========================================================================
    fieldnames = [
        "n_seeds", "exponent", "D", "H", "W", "n_voxels", "n_fluid", "n_V",
        "r_cut", "r_unf", "rho_isl", "eta_roi", "eta_stamp",
        "t_m3", "t_stamp", "t_jfa", "t_close", "t_relax", "t_build",
        "t_exact",
        "mae", "mre", "max_err", "rmse", "acc_rate", "mismatch_rate",
        "status"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {output_csv}")
    print(f"Success: {sum(1 for r in results if r['status']=='OK')}/{len(results)}")

    return results



# =============================================================================
# quick test function
# =============================================================================

def quick_accuracy_test(n_seeds=100, scale=2):
    import numpy as np
    import cupy as cp
    import time
    import math

    print("=" * 80)
    print(f"Quick Accuracy Test: scale={scale}, n_seeds={n_seeds}  [Exact=26]")
    print("=" * 80)

    D = 64 * scale
    H = 64 * scale
    W = 96 * scale

    mask, _ = make_3d_maze_case(
        D=D, H=H, W=W,
        n_seeds=1,
        seed_mode="random",
        seed_random_state=42,
        corridor_width=3*scale,
        wall_thickness=2*scale,
        verbose=False,
        ensure_connected=True,
        cc_connectivity=6,
    )
    mask = np.asarray(mask, dtype=bool)
    Vmask = mask  # ensure_connected=True

    # seeds sample (avoid np.argwhere)
    rng = np.random.RandomState(42)
    mask_flat = mask.ravel()
    nvox = mask_flat.size
    batch = rng.randint(0, nvox, size=int(math.ceil(n_seeds * 4.0)), dtype=np.int64)
    batch = batch[mask_flat[batch]]
    chosen = np.unique(batch)[:n_seeds]
    HW = H*W
    z = chosen // HW
    rem = chosen - z*HW
    y = rem // W
    x = rem - y*W
    seeds = np.stack([z,y,x], axis=1).astype(np.int64)

    print(f"Domain: {D}x{H}x{W}, fluid={mask.sum():,}, V={Vmask.sum():,}")

    # Exact (26)
    print("Running exact (connectivity=26)...")
    t0 = time.time()
    label_exact, dist_exact = exact_geodesic_voronoi_gpu(mask, seeds, connectivity=26)
    cp.cuda.Device().synchronize()
    t_exact = time.time() - t0
    print(f"  Exact: {t_exact:.3f}s")

    # M3
    print("Running M3 (ROI-JFA)...")
    stamp_kern = build_seed_stamping_los_parallel_packed_kernel()
    tiles_kern = build_tiles_dual_3d_kernel()
    active_kern = build_active_tiles_kernel_3d()

    t0 = time.time()
    label_m3, dist_m3, _, roi_mask = geodesic_voronoi_roi_jfa(
        mask, seeds,
        tile_size=(8,8,16),
        verbose=False,
        viz_policy="none",
        stamping_kernel=stamp_kern,
        tiles_dual_kernel=tiles_kern,
        active_tiles_kernel=active_kern,
        enable_closure=True,
        max_refine_iters=256,
    )
    cp.cuda.Device().synchronize()
    t_m3 = time.time() - t0
    print(f"  M3: {t_m3:.3f}s (speedup: {t_exact/t_m3:.1f}x)")

    acc = compute_accuracy_metrics(dist_m3, label_m3, dist_exact, label_exact, Vmask)

    print("\n" + "=" * 40)
    print("ACCURACY METRICS")
    print("=" * 40)
    print(f"Distance errors:")
    print(f"  MAE  = {acc['mae']:.6f}")
    print(f"  MRE  = {acc['mre']:.6f} ({acc['mre']*100:.4f}%)")
    print(f"  RMSE = {acc['rmse']:.6f}")
    print(f"  max  = {acc['max_err']:.4f}")
    print(f"\nLabel accuracy:")
    print(f"  acc_rate = {acc['acc_rate']:.6f} ({acc['acc_rate']*100:.4f}%)")
    print(f"  mismatch = {acc['mismatch_rate']:.6f} ({acc['mismatch_rate']*100:.4f}%)")
    print("=" * 40)

    # (D) clear the pool for repeated quick tests
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return acc

# =====================================================================================
# Active Final Relaxation (tile-based, Definition-A boundary band) + timing
#       adds functions and variables only; no existing function is changed.
#       usage: geodesic_voronoi_roi_jfa_with_active_final(...) in place of geodesic_voronoi_roi_jfa(...)
# =====================================================================================

# ---- new timing globals (single-run last call) ----
ROI_JFA_LAST_TAFINAL_WALL = 0.0       # active-final wall-time (seconds)
ROI_JFA_LAST_TAFINAL_GPU_TIME = 0.0   # active-final CUDA event time (seconds)


# ============================================================
# [AF-1] Mark boundary tiles (Definition A) from packed state (active-list tiles)
# ============================================================
def build_mark_boundary_tiles_from_state_active_list_kernel_3d():
    """
    Definition A: a boundary tile is a tile with at least one fluid voxel that has a fluid 6-neighbour with a different label.
    Input state_in (packed: dist|label); only the tiles in candidate_ids are processed.
    Output tile_boundary[tid]=1 (int32 flags); all others stay 0 (clear them on the Python side first).
    """
    code = r'''
    extern "C" __global__
    void mark_boundary_tiles_from_state_active_list(
        const unsigned char* __restrict__ mask,              // [nvox]
        const unsigned long long* __restrict__ state_in,     // [nvox]
        int* __restrict__ tile_boundary,                     // [nTiles] int32 (0/1), must be zeroed by host

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,

        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx,

        const int* __restrict__ candidate_ids,               // [n_candidate]
        const int n_candidate
    )
    {
        int bid = (int)blockIdx.x;
        if (bid >= n_candidate) return;

        int tid = candidate_ids[bid];
        int nTiles = nTilesZ * nTilesY * nTilesX;
        if ((unsigned)tid >= (unsigned)nTiles) return;

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int z0 = tz * Tz;
        int y0 = ty * Ty;
        int x0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        int local_hit = 0;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {

            if (local_hit) break;

            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = z0 + lz;
            int y = y0 + ly;
            int x = x0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;
            if (!mask[idx]) continue;

            unsigned long long cur = state_in[idx];
            int lab = (int)(cur & 0xFFFFFFFFu);
            if (lab < 0) continue;

            // 6-neigh check (mark both sides because we check +/- directions)
            // x-1
            if (x > 0) {
                int j = idx - 1;
                if (mask[j]) {
                    int lab2 = (int)(state_in[j] & 0xFFFFFFFFu);
                    if (lab2 >= 0 && lab2 != lab) { local_hit = 1; continue; }
                }
            }
            // x+1
            if (x + 1 < W) {
                int j = idx + 1;
                if (mask[j]) {
                    int lab2 = (int)(state_in[j] & 0xFFFFFFFFu);
                    if (lab2 >= 0 && lab2 != lab) { local_hit = 1; continue; }
                }
            }
            // y-1
            if (y > 0) {
                int j = idx - W;
                if (mask[j]) {
                    int lab2 = (int)(state_in[j] & 0xFFFFFFFFu);
                    if (lab2 >= 0 && lab2 != lab) { local_hit = 1; continue; }
                }
            }
            // y+1
            if (y + 1 < H) {
                int j = idx + W;
                if (mask[j]) {
                    int lab2 = (int)(state_in[j] & 0xFFFFFFFFu);
                    if (lab2 >= 0 && lab2 != lab) { local_hit = 1; continue; }
                }
            }
            // z-1
            if (z > 0) {
                int j = idx - HW;
                if (mask[j]) {
                    int lab2 = (int)(state_in[j] & 0xFFFFFFFFu);
                    if (lab2 >= 0 && lab2 != lab) { local_hit = 1; continue; }
                }
            }
            // z+1
            if (z + 1 < D) {
                int j = idx + HW;
                if (mask[j]) {
                    int lab2 = (int)(state_in[j] & 0xFFFFFFFFu);
                    if (lab2 >= 0 && lab2 != lab) { local_hit = 1; continue; }
                }
            }
        }

        int any_hit = __syncthreads_or(local_hit);
        if (threadIdx.x == 0 && any_hit) {
            tile_boundary[tid] = 1;
        }
    }
    '''
    return _device_cached_rawkernel(
        build_mark_boundary_tiles_from_state_active_list_kernel_3d,
        code,
        "mark_boundary_tiles_from_state_active_list",
        options=("-std=c++11",),
    )


# ============================================================
# [AF-2] Tile-flag dilation (26-neigh on tile lattice)
# ============================================================
def build_dilate_tile_flags_26_kernel_3d():
    """
    26-neighbour dilation on the tile lattice:
      out[tid] = in[tid] OR any neighbour in[tid2]
    used to build the band/halo.
    """
    code = r'''
    extern "C" __global__
    void dilate_tile_flags_26(
        const int* __restrict__ in_flag,   // [nTiles] 0/1
        int* __restrict__ out_flag,        // [nTiles] 0/1
        const int nTilesZ, const int nTilesY, const int nTilesX
    )
    {
        int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nTiles = nTilesZ * nTilesY * nTilesX;
        if (tid >= nTiles) return;

        if (in_flag[tid] != 0) {
            out_flag[tid] = 1;
            return;
        }

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int hit = 0;

        for (int dz = -1; dz <= 1 && !hit; ++dz) {
            int zz = tz + dz;
            if ((unsigned)zz >= (unsigned)nTilesZ) continue;
            for (int dy = -1; dy <= 1 && !hit; ++dy) {
                int yy = ty + dy;
                if ((unsigned)yy >= (unsigned)nTilesY) continue;
                int base = (zz * nTilesY + yy) * nTilesX;
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = tx + dx;
                    if ((unsigned)xx >= (unsigned)nTilesX) continue;
                    int t2 = base + xx;
                    if (in_flag[t2] != 0) { hit = 1; break; }
                }
            }
        }

        out_flag[tid] = hit;
    }
    '''
    return _device_cached_rawkernel(
        build_dilate_tile_flags_26_kernel_3d,
        code,
        "dilate_tile_flags_26",
        options=("-std=c++11",),
    )


# ============================================================
# [AF-3] Copy packed state for a tile active-list (patch dst tiles with src)
# ============================================================
def build_copy_tiles_packed_active_list_kernel_3d():
    """
    Copy state_src to state_dst for the tiles listed in active_ids only (all in-domain voxels of each tile).
    Used with ping-pong buffers, so that an inactive tile does not fall back to an earlier state after a swap.
    """
    code = r'''
    extern "C" __global__
    void copy_tiles_packed_active_list(
        const unsigned long long* __restrict__ state_src,
        unsigned long long* __restrict__ state_dst,

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,

        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx,

        const int* __restrict__ active_ids,  // [n_active]
        const int n_active
    )
    {
        int bid = (int)blockIdx.x;
        if (bid >= n_active) return;

        int tid = active_ids[bid];
        int nTiles = nTilesZ * nTilesY * nTilesX;
        if ((unsigned)tid >= (unsigned)nTiles) return;

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int z0 = tz * Tz;
        int y0 = ty * Ty;
        int x0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {
            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = z0 + lz;
            int y = y0 + ly;
            int x = x0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;
            state_dst[idx] = state_src[idx];
        }
    }
    '''
    return _device_cached_rawkernel(
        build_copy_tiles_packed_active_list_kernel_3d,
        code,
        "copy_tiles_packed_active_list",
        options=("-std=c++11",),
    )


# ============================================================
# [AF-4] Jump=1 relaxation on active tiles (writes tile_dirty_out)
# ============================================================
def build_relax1_tiles_active_list_packed_kernel_3d():
    """
    One 26-neighbour jump=1 relaxation (packed) on the tiles listed in active_ids.
    Also sets tile_dirty_out[tid]=1 if anything in the tile changed (smaller dist or a different label).
    Note: clear tile_dirty_out on the host side first.
    """
    code = r'''
    extern "C" __global__
    void relax1_tiles_active_list_packed(
        const unsigned char* __restrict__ mask,      // [nvox]
        const unsigned long long* __restrict__ state_in,
        unsigned long long* __restrict__ state_out,
        int* __restrict__ tile_dirty_out,            // [nTiles] 0/1, must be zeroed

        const int D, const int H, const int W,
        const int Tz, const int Ty, const int Tx,
        const int nTilesZ, const int nTilesY, const int nTilesX,

        const int pow2_decode,
        const int shift_TyTx, const int shift_Tx,
        const int mask_TyTx, const int mask_Tx,

        const int* __restrict__ active_ids,          // [n_active]
        const int n_active,
        const float eps
    )
    {
        int bid = (int)blockIdx.x;
        if (bid >= n_active) return;

        int tid = active_ids[bid];
        int nTiles = nTilesZ * nTilesY * nTilesX;
        if ((unsigned)tid >= (unsigned)nTiles) return;

        int tz = tid / (nTilesY * nTilesX);
        int rem = tid - tz * (nTilesY * nTilesX);
        int ty = rem / nTilesX;
        int tx = rem - ty * nTilesX;

        int z0 = tz * Tz;
        int y0 = ty * Ty;
        int x0 = tx * Tx;

        const int HW = H * W;
        const int TyTx = Ty * Tx;
        const int tile_vox = Tz * Ty * Tx;

        const float SQRT2 = 1.41421356237f;
        const float SQRT3 = 1.73205080757f;

        int local_dirty = 0;

        for (int p = (int)threadIdx.x; p < tile_vox; p += (int)blockDim.x) {

            int lz, ly, lx;
            if (pow2_decode) {
                lz = p >> shift_TyTx;
                int rem2 = p & mask_TyTx;
                ly = rem2 >> shift_Tx;
                lx = rem2 & mask_Tx;
            } else {
                lz = p / TyTx;
                int rem2 = p - lz * TyTx;
                ly = rem2 / Tx;
                lx = rem2 - ly * Tx;
            }

            int z = z0 + lz;
            int y = y0 + ly;
            int x = x0 + lx;

            if ((unsigned)z >= (unsigned)D || (unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) continue;

            int idx = z * HW + y * W + x;

            unsigned long long cur = state_in[idx];

            if (!mask[idx]) {
                state_out[idx] = cur;
                continue;
            }

            int   cur_label = (int)(cur & 0xFFFFFFFFu);
            float cur_dist  = __uint_as_float((unsigned int)(cur >> 32));

            int   best_label = cur_label;
            float best_dist  = cur_dist;

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;

                        int jz = z + dz;
                        int jy = y + dy;
                        int jx = x + dx;

                        if ((unsigned)jz >= (unsigned)D ||
                            (unsigned)jy >= (unsigned)H ||
                            (unsigned)jx >= (unsigned)W) continue;

                        int j_idx = idx + dz * HW + dy * W + dx;
                        if (!mask[j_idx]) continue;

                        unsigned long long nb = state_in[j_idx];
                        int nb_label = (int)(nb & 0xFFFFFFFFu);
                        if (nb_label < 0) continue;

                        float nb_dist = __uint_as_float((unsigned int)(nb >> 32));

                        int nnz = (dx != 0) + (dy != 0) + (dz != 0);
                        float step = (nnz == 1) ? 1.0f
                                   : (nnz == 2) ? SQRT2
                                                : SQRT3;

                        float cand = nb_dist + step;

                        bool improve = (cand + eps < best_dist);
                        bool tie_better_label = (fabsf(cand - best_dist) <= eps) && (nb_label >= 0) && (best_label < 0 || nb_label < best_label);
                        if (improve || tie_better_label) {
                            best_dist  = cand;
                            best_label = nb_label;
                        }
                    }
                }
            }

            unsigned int out_du = __float_as_uint(best_dist);
            unsigned long long out_pack =
                ((unsigned long long)out_du << 32) | (unsigned long long)(unsigned int)best_label;

            state_out[idx] = out_pack;

            // dirty if changed (distance improved or label changed)
            if ((best_label != cur_label) || (best_dist + eps < cur_dist)) {
                local_dirty = 1;
            }
        }

        int any_dirty = __syncthreads_or(local_dirty);
        if (threadIdx.x == 0 && any_dirty) {
            tile_dirty_out[tid] = 1;
        }
    }
    '''
    return _device_cached_rawkernel(
        build_relax1_tiles_active_list_packed_kernel_3d,
        code,
        "relax1_tiles_active_list_packed",
        options=("-std=c++11",),
    )


# ============================================================
# [AF-5] Active-final driver on packed state (tile ROI-aware)
# ============================================================
def active_final_relax_tiles_packed(
    mask_flat,
    state_init_u64,
    D, H, W,
    *,
    tile_size=(8, 8, 16),
    tile_roi=None,                 # cp.int32 [nTiles] or None. If provided, use it as initial candidate region.
    active_final_iters=0,          # number of active-final iterations
    band_iters=1,                  # how many tile-lattice dilations for boundary band
    halo_iters=1,                  # halo dilation around dirty tiles for next candidate
    eps=1e-6,
    profile_gpu=False,
    verbose=False,
):
    """
    Same method (Definition-A boundary band + jump=1 relaxation); implemented with tiles and dirty/halo shrinking.
    - initial candidates: tile_roi (the ROI tiles) where available; otherwise all tiles.
    - each pass: candidates -> boundary tiles -> band dilation -> one relaxation -> dirty -> halo -> next candidates
    Returns: state_u64 (packed).
    Also updates the global timers ROI_JFA_LAST_TAFINAL_WALL / ROI_JFA_LAST_TAFINAL_GPU_TIME.
    """
    import cupy as cp
    import numpy as np
    import time

    global ROI_JFA_LAST_TAFINAL_WALL, ROI_JFA_LAST_TAFINAL_GPU_TIME
    ROI_JFA_LAST_TAFINAL_WALL = 0.0
    ROI_JFA_LAST_TAFINAL_GPU_TIME = 0.0

    iters = int(active_final_iters)
    if iters <= 0:
        return state_init_u64

    Tz, Ty, Tx = [int(v) for v in tile_size]
    nTilesZ = (int(D) + Tz - 1) // Tz
    nTilesY = (int(H) + Ty - 1) // Ty
    nTilesX = (int(W) + Tx - 1) // Tx
    nTiles  = int(nTilesZ * nTilesY * nTilesX)

    pow2_decode, shift_TyTx, shift_Tx, mask_TyTx, mask_Tx = _compute_pow2_decode_params(tile_size)

    # kernels
    k_mark_boundary = build_mark_boundary_tiles_from_state_active_list_kernel_3d()
    k_dilate_tiles  = build_dilate_tile_flags_26_kernel_3d()
    k_copy_tiles    = build_copy_tiles_packed_active_list_kernel_3d()
    k_relax_tiles   = build_relax1_tiles_active_list_packed_kernel_3d()

    threads_tiles = 256
    blocks_tiles = (nTiles + threads_tiles - 1) // threads_tiles

    # flags
    tile_candidate = cp.zeros(nTiles, dtype=cp.int32)
    tile_boundary  = cp.zeros(nTiles, dtype=cp.int32)
    tile_active    = cp.zeros(nTiles, dtype=cp.int32)
    tile_tmp       = cp.zeros(nTiles, dtype=cp.int32)
    tile_dirty     = cp.zeros(nTiles, dtype=cp.int32)

    # init candidate region
    if tile_roi is None:
        tile_candidate.fill(1)   # all tiles
    else:
        # tile_roi is 0/1 for ROI-tiles (int32)
        tile_candidate[...] = tile_roi.astype(cp.int32, copy=False)
        # widen a bit (halo_iters) so boundary/updates can move
        for _ in range(max(int(halo_iters), 0)):
            tile_tmp.fill(0)
            k_dilate_tiles(
                (int(blocks_tiles),),
                (int(threads_tiles),),
                (tile_candidate, tile_tmp, np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX)),
            )
            tile_candidate, tile_tmp = tile_tmp, tile_candidate

    # two ping-pong buffers
    state_a = state_init_u64
    state_b = state_a.copy()  # full copy ONCE

    prev_active_ids = None

    t0 = time.time()

    # GPU event timing for the whole active-final loop (optional)
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event(); evt1 = cp.cuda.Event()
        evt0.record()

    for it in range(iters):
        # candidate list (tile-level)
        cand_ids = cp.where(tile_candidate != 0)[0].astype(cp.int32)
        n_cand = int(cand_ids.size)
        if n_cand == 0:
            if verbose:
                print(f"[AF] iter {it}: empty candidate -> stop.")
            break

        # boundary tiles in candidate
        tile_boundary.fill(0)
        k_mark_boundary(
            (int(n_cand),),
            (int(threads_tiles),),
            (
                mask_flat,
                state_a,
                tile_boundary,
                np.int32(D), np.int32(H), np.int32(W),
                np.int32(Tz), np.int32(Ty), np.int32(Tx),
                np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX),
                np.int32(pow2_decode),
                np.int32(shift_TyTx), np.int32(shift_Tx),
                np.int32(mask_TyTx),  np.int32(mask_Tx),
                cand_ids, np.int32(n_cand),
            ),
        )

        # band dilation on tile lattice
        tile_active[...] = tile_boundary
        for _ in range(max(int(band_iters), 0)):
            tile_tmp.fill(0)
            k_dilate_tiles(
                (int(blocks_tiles),),
                (int(threads_tiles),),
                (tile_active, tile_tmp, np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX)),
            )
            tile_active, tile_tmp = tile_tmp, tile_active

        active_ids = cp.where(tile_active != 0)[0].astype(cp.int32)
        n_active = int(active_ids.size)
        if verbose:
            print(f"[AF] iter {it}: cand_tiles={n_cand}, boundary_tiles={int(cp.count_nonzero(tile_boundary).item())}, active_tiles={n_active}")

        if n_active == 0:
            # no boundary in candidate -> stop
            break

        # patch state_b for prev_active tiles (so inactive tiles won't regress after swap)
        if prev_active_ids is not None and int(prev_active_ids.size) > 0:
            k_copy_tiles(
                (int(prev_active_ids.size),),
                (int(threads_tiles),),
                (
                    state_a,
                    state_b,
                    np.int32(D), np.int32(H), np.int32(W),
                    np.int32(Tz), np.int32(Ty), np.int32(Tx),
                    np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX),
                    np.int32(pow2_decode),
                    np.int32(shift_TyTx), np.int32(shift_Tx),
                    np.int32(mask_TyTx),  np.int32(mask_Tx),
                    prev_active_ids, np.int32(int(prev_active_ids.size)),
                ),
            )

        # relax once on active tiles; compute dirty
        tile_dirty.fill(0)
        k_relax_tiles(
            (int(n_active),),
            (int(threads_tiles),),
            (
                mask_flat,
                state_a,
                state_b,
                tile_dirty,
                np.int32(D), np.int32(H), np.int32(W),
                np.int32(Tz), np.int32(Ty), np.int32(Tx),
                np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX),
                np.int32(pow2_decode),
                np.int32(shift_TyTx), np.int32(shift_Tx),
                np.int32(mask_TyTx),  np.int32(mask_Tx),
                active_ids, np.int32(n_active),
                np.float32(float(eps)),
            ),
        )

        # next candidate = halo(dirty)  (if no dirty, stop early)
        if int(cp.count_nonzero(tile_dirty).item()) == 0:
            if verbose:
                print(f"[AF] iter {it}: dirty=0 -> stop.")
            state_a, state_b = state_b, state_a
            break

        tile_candidate[...] = tile_dirty
        for _ in range(max(int(halo_iters), 0)):
            tile_tmp.fill(0)
            k_dilate_tiles(
                (int(blocks_tiles),),
                (int(threads_tiles),),
                (tile_candidate, tile_tmp, np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX)),
            )
            tile_candidate, tile_tmp = tile_tmp, tile_candidate

        # swap
        state_a, state_b = state_b, state_a
        prev_active_ids = active_ids

    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        ROI_JFA_LAST_TAFINAL_GPU_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0

    cp.cuda.Device().synchronize()
    ROI_JFA_LAST_TAFINAL_WALL = float(time.time() - t0)
    return state_a


def fullgrid_relax_until_converged_packed(
    mask_flat,
    state_init_u64,
    D, H, W,
    *,
    tile_size=(8, 8, 16),
    max_iters=1024,
    eps=1e-6,
    profile_gpu=False,
    verbose=False,
):
    """
    Full-grid closure over the whole domain:
    - every pass treats all tiles as active;
    - one unit-step packed relaxation over the whole domain;
    - until no tile is dirty anywhere, or max_iters is reached.
    Unlike the ROI/band/halo logic of active-final, this gives the Case-C diagnostics
    a full-grid monotone fallback baseline.
    """
    import cupy as cp
    import numpy as np
    import time

    global ROI_JFA_LAST_TAFINAL_WALL, ROI_JFA_LAST_TAFINAL_GPU_TIME
    ROI_JFA_LAST_TAFINAL_WALL = 0.0
    ROI_JFA_LAST_TAFINAL_GPU_TIME = 0.0

    max_iters = int(max_iters)
    if max_iters <= 0:
        return state_init_u64

    Tz, Ty, Tx = [int(v) for v in tile_size]
    nTilesZ = (int(D) + Tz - 1) // Tz
    nTilesY = (int(H) + Ty - 1) // Ty
    nTilesX = (int(W) + Tx - 1) // Tx
    nTiles  = int(nTilesZ * nTilesY * nTilesX)

    pow2_decode, shift_TyTx, shift_Tx, mask_TyTx, mask_Tx = _compute_pow2_decode_params(tile_size)

    k_relax_tiles = build_relax1_tiles_active_list_packed_kernel_3d()

    threads_tiles = 256
    active_ids = cp.arange(nTiles, dtype=cp.int32)
    n_active = int(active_ids.size)
    tile_dirty = cp.zeros(nTiles, dtype=cp.int32)

    state_a = state_init_u64
    state_b = state_a.copy()

    t0 = time.time()
    evt0 = evt1 = None
    if profile_gpu:
        evt0 = cp.cuda.Event(); evt1 = cp.cuda.Event()
        evt0.record()

    it_used = 0
    for it in range(max_iters):
        # start from previous state everywhere
        state_b[...] = state_a
        tile_dirty.fill(0)

        k_relax_tiles(
            (int(n_active),),
            (int(threads_tiles),),
            (
                mask_flat,
                state_a,
                state_b,
                tile_dirty,
                np.int32(D), np.int32(H), np.int32(W),
                np.int32(Tz), np.int32(Ty), np.int32(Tx),
                np.int32(nTilesZ), np.int32(nTilesY), np.int32(nTilesX),
                np.int32(pow2_decode),
                np.int32(shift_TyTx), np.int32(shift_Tx),
                np.int32(mask_TyTx),  np.int32(mask_Tx),
                active_ids, np.int32(n_active),
                np.float32(float(eps)),
            ),
        )

        dirty_count = int(cp.count_nonzero(tile_dirty).item())
        it_used = it + 1
        state_a, state_b = state_b, state_a

        if verbose:
            print(f"[Full-grid] iter {it}: dirty_tiles={dirty_count}")
        if dirty_count == 0:
            break

    if profile_gpu:
        evt1.record()
        evt1.synchronize()
        ROI_JFA_LAST_TAFINAL_GPU_TIME = float(cp.cuda.get_elapsed_time(evt0, evt1)) / 1000.0

    cp.cuda.Device().synchronize()
    ROI_JFA_LAST_TAFINAL_WALL = float(time.time() - t0)
    if verbose:
        print(f"[Full-grid] used_iters={it_used}")
    return state_a


# ============================================================
# [AF-6] Wrapper: ROI-JFA with n_relax_after=1 + optional active-final refine
# ============================================================
def geodesic_voronoi_roi_jfa_with_active_final(
    mask,
    seeds,
    *,
    tile_size=(8, 8, 16),
    delta_r=1.0,
    eta_max=0.8,
    r_tile=1,
    enable_stamping=True,
    verbose=False,
    viz_policy="none",
    relax_eps=1e-6,
    changed_mode="frontier",
    profile_gpu=False,
    return_records=False,
    stamping_kernel=None,
    tiles_dual_kernel=None,
    roi_step_kernels=None,
    active_tiles_kernel=None,
    mark_roi_tiles_kernel=None,
    max_refine_iters=460,
    relax_kernel=None,
    apply_kernel=None,
    active_fallback_kernel=None,
    enable_closure=True,
    closure_kernel=None,
    clearance_kmax27=None,
    los_kmax_init_kernel=None,
    los_kmax_update_kernel=None,
    enable_manhattan_jump=True,
    use_active_list_step=True,

    n_relax_after=1,

    active_final_iters=0,
    active_final_band_iters=1,
    active_final_halo_iters=1,
):
    """
    Wrapper around geodesic_voronoi_roi_jfa (which is unchanged):
    - forces n_relax_after=1
    - with active_final_iters>0, adds the tile-based active-final refinement (timed separately)
    Same return value as geodesic_voronoi_roi_jfa (with return_records=True the records behave the same).
    """
    import cupy as cp
    import numpy as np

    global ROI_JFA_LAST_TAFINAL_WALL, ROI_JFA_LAST_TAFINAL_GPU_TIME
    global ROI_JFA_LAST_TPRED_WALL, ROI_JFA_LAST_GPU_TIME

    ROI_JFA_LAST_TAFINAL_WALL = 0.0
    ROI_JFA_LAST_TAFINAL_GPU_TIME = 0.0

    # ---- call your original solver (unchanged), but force n_relax_after=1 ----
    out = geodesic_voronoi_roi_jfa(
        mask, seeds,
        tile_size=tile_size,
        delta_r=delta_r,
        eta_max=eta_max,
        r_tile=r_tile,
        enable_stamping=enable_stamping,
        verbose=verbose,
        viz_policy=viz_policy,
        n_relax_after=int(n_relax_after),               # <- forced
        relax_eps=relax_eps,
        changed_mode=changed_mode,
        profile_gpu=profile_gpu,
        return_records=return_records,
        stamping_kernel=stamping_kernel,
        tiles_dual_kernel=tiles_dual_kernel,
        roi_step_kernels=roi_step_kernels,
        active_tiles_kernel=active_tiles_kernel,
        mark_roi_tiles_kernel=mark_roi_tiles_kernel,
        max_refine_iters=max_refine_iters,
        relax_kernel=relax_kernel,
        apply_kernel=apply_kernel,
        active_fallback_kernel=active_fallback_kernel,
        enable_closure=enable_closure,
        closure_kernel=closure_kernel,
        clearance_kmax27=clearance_kmax27,
        los_kmax_init_kernel=los_kmax_init_kernel,
        los_kmax_update_kernel=los_kmax_update_kernel,
        enable_manhattan_jump=enable_manhattan_jump,
        use_active_list_step=use_active_list_step,
    )

    if return_records:
        label_cp, dist_cp, tile_roi, roi_mask_flat, records = out
    else:
        label_cp, dist_cp, tile_roi, roi_mask_flat = out

    iters = int(active_final_iters)
    if iters <= 0:
        return out

    # ---- pack to state ----
    mask_cp, (D, H, W), mask_flat = _as_cupy_mask_u8(mask)
    nvox = int(D * H * W)

    label_flat = label_cp.astype(cp.int32, copy=False).ravel()
    dist_flat  = dist_cp.astype(cp.float32, copy=False).ravel()
    dist_u32 = dist_flat.view(cp.uint32)
    label_u32 = label_flat.astype(cp.uint32, copy=False)

    state_u64 = (dist_u32.astype(cp.uint64) << cp.uint64(32)) | label_u32.astype(cp.uint64)

    # ---- run active-final refine (ROI-aware: use tile_roi as candidate seed) ----
    state_u64 = active_final_relax_tiles_packed(
        mask_flat,
        state_u64,
        int(D), int(H), int(W),
        tile_size=tile_size,
        tile_roi=tile_roi,  # ROI tiles from solver
        active_final_iters=iters,
        band_iters=int(active_final_band_iters),
        halo_iters=int(active_final_halo_iters),
        eps=float(relax_eps),
        profile_gpu=profile_gpu,
        verbose=verbose,
    )

    # ---- unpack to label/dist ----
    label_out_flat = (state_u64 & cp.uint64(0xFFFFFFFF)).astype(cp.int32)
    dist_out_u32   = (state_u64 >> cp.uint64(32)).astype(cp.uint32)
    dist_out_flat  = dist_out_u32.view(cp.float32)

    label_out = label_out_flat.reshape((D, H, W))
    dist_out  = dist_out_flat.reshape((D, H, W))

    # enforce solid convention
    solid = (mask_flat == 0).reshape((D, H, W))
    label_out = cp.where(solid, cp.int32(-1), label_out)
    dist_out  = cp.where(solid, cp.float32(1e20), dist_out)

    # ---- timing accumulation: add active-final into total pred/gpu, but keep it separated ----
    ROI_JFA_LAST_TPRED_WALL = float(ROI_JFA_LAST_TPRED_WALL + ROI_JFA_LAST_TAFINAL_WALL)
    if profile_gpu:
        ROI_JFA_LAST_GPU_TIME = float(ROI_JFA_LAST_GPU_TIME + ROI_JFA_LAST_TAFINAL_GPU_TIME)

    if return_records:
        return label_out, dist_out, tile_roi, roi_mask_flat, records
    return label_out, dist_out, tile_roi, roi_mask_flat


# ============================================================
# [AF-7] New main: scaling experiment with active-final (others unchanged)
# ============================================================
def run_scaling_experiment_active_final(
    scale=5,
    seed_start_exp=0.2,
    seed_end_exp=4.0,
    seed_step=0.2,
    random_state=42,
    output_csv="all_methods_scaling_with_metrics_caseA.csv",

    # --- which methods to run ---
    run_m0=True,      # M0: Euclidean + clipping
    run_m2=True,      # M2: Full OA-JFA
    run_m3=True,      # M3: ROI-JFA + Active-Final
    run_exact=True,   # M1: Exact (reference)

    max_seeds_for_exact=100000,

    # --- M3 controls ---
    active_final_iters=32,
    active_final_band_iters=1,
    active_final_halo_iters=1,

    # --- speed/overhead controls (optional) ---
    fast_mode=False,               # True: default to sampling structural metrics instead of full
    struct_metrics_sample=None,    # None=full; 0=skip; int=sample that many seeds (estimate)
    free_memory_each_iter=True,    # free CuPy memory pool blocks each iter (can reduce fragmentation)
    
    # --- printing ---
    print_compact_table=True,     # Print a TSV-style line per (n_seeds, method)
    print_all_params=False,       # Also print the full row dict (all parameters) per (n_seeds, method)
):
    """
    Multi-method scaling experiment over varying seed counts.

    For each n_seeds (log-spaced):
      - M0: Euclidean Voronoi + clipping (GPU)
      - M1: Exact geodesic Voronoi (reference, optional)
      - M2: Full OA-JFA geodesic Voronoi (GPU)
      - M3: ROI-JFA + Active-Final geodesic Voronoi (GPU)

    Metrics:
      - r_cut, r_unf, rho_isl   : structural metrics on V (CPU for r_cut/rho_isl; GPU for r_unf)
      - e_vox                   : label mismatch rate vs Exact on V (end-to-end; equals mismatch_rate)
      - eta_roi                 : ROI fraction (only for M3; others NaN)
      - t_pred, t_stamp, t_jfa  : timing (M3 decomposed; others t_stamp=0, t_jfa=t_pred)

    Extra (kept for convenience):
      - MAE/MRE/max_err/RMSE, acc_rate, mismatch_rate (distance+label accuracy vs Exact on V)
      - Exact time decomposition (solve/metrics/total)

    Output:
      - CSV in long format: one row per (n_seeds, method).
    """
    import numpy as np
    import cupy as cp
    import time
    import csv
    import math
    import gc

    # -----------------------------
    # Domain / Case-A (Thin Wall + Gate) settings
    # -----------------------------
    # Case A: two chambers separated by a thin wall with a moderate gate.
    #   - This is the "baseline obstacle" used to test whether methods respect barrier constraints
    #     without being dominated by extreme bottlenecks (Case B) or labyrinth topology (Case C).
    base_D, base_H, base_W = 64, 64, 96

    # Base geometry parameters (defined at base resolution, then scaled)
    base_wall_thickness = 2
    base_gate_size = (12, 12)          # (gate_size_y, gate_size_z) in voxels at base scale
    base_seed_min_sep_cells = 2
    base_seed_margin_from_wall = 4     # keep seeds away from the barrier/walls

    D = int(base_D * scale)
    H = int(base_H * scale)
    W = int(base_W * scale)

    wall_thickness = int(base_wall_thickness * scale)
    gate_size = (int(base_gate_size[0] * scale), int(base_gate_size[1] * scale))
    seed_min_sep_cells = int(base_seed_min_sep_cells)
    seed_margin_from_wall = int(base_seed_margin_from_wall * scale)

    tile_size = (8, 8, 16)
    min_comp_vox = int(16 * scale)
    min_comp_frac = 0.01
    max_refine_iters = 1000

    # -----------------------------
    # Seed schedule
    # -----------------------------
    exponents = np.arange(seed_start_exp, seed_end_exp + 0.01, seed_step)
    seed_counts = np.unique(np.round(10.0 ** exponents).astype(int))

    print("=" * 120)
    print("Scaling Experiment: M0/M1/M2/M3 with metrics (incl. end-to-end vs Exact when enabled)")
    print("=" * 120)
    print(f"Scale: {scale}x")
    print(f"Domain: {D} x {H} x {W} = {D*H*W:,} voxels")
    print(f"Tile size (M3): {tile_size}")
    print(f"Active-final (M3): iters={int(active_final_iters)}, band={int(active_final_band_iters)}, halo={int(active_final_halo_iters)}")
    print(f"Seed counts ({len(seed_counts)} points): {seed_counts.tolist()}")
    print("=" * 120)

    # -----------------------------
    # Compile / warm-up kernels once
    # -----------------------------
    print("Pre-compiling / warm-up CUDA kernels...")
    stamp_kern = build_seed_stamping_los_parallel_packed_kernel()
    tiles_kern = build_tiles_dual_3d_kernel()
    active_kern = build_active_tiles_kernel_3d()

    warm_mask_np = np.ones((16, 16, 16), dtype=bool)
    warm_seeds = np.array([[6, 6, 6], [10, 10, 10]], dtype=np.int64)

    if run_m0:
        _ = m1_euclidean_voronoi_clipping_gpu(warm_mask_np, warm_seeds, profile_gpu=False)
    if run_m2:
        _ = geodesic_voronoi_oajfa_cuda(warm_mask_np, warm_seeds, profile_gpu=False)
    if run_m3:
        _ = geodesic_voronoi_roi_jfa_with_active_final(
            warm_mask_np, warm_seeds,
            tile_size=(8, 8, 8),
            verbose=False,
            viz_policy="none",
            stamping_kernel=stamp_kern,
            tiles_dual_kernel=tiles_kern,
            active_tiles_kernel=active_kern,
            enable_closure=True,
            max_refine_iters=8,
            active_final_iters=1,
        )
    if run_exact:
        warm_mask_cp = cp.asarray(warm_mask_np, dtype=cp.uint8)
        _ = exact_geodesic_voronoi_gpu(
            warm_mask_cp,
            warm_seeds,
            connectivity=26,
            max_iter=None,
            eps=1e-6,
            verbose=False,
            profile_gpu=False,
            check_optimality=False,
        )

    cp.cuda.Device().synchronize()
    print("Warm-up done.\n")

    # -----------------------------
    # Build ONE fixed thin-wall mask (Case A) & precompute LOS-kmax27 once
    # -----------------------------
    print("[A] Building ONE fixed thin-wall mask (Case A) and precomputing LOS-kmax27 once...")
    t0_mask = time.time()
    mask_np, _ = make_thin_wall_case(
        D=D, H=H, W=W,
        n_seeds=1,
        wall_thickness=wall_thickness,
        gate_size=gate_size,
        gate_center=None,
        seed_mode="random",
        seed_random_state=int(random_state),
        seed_margin_from_wall=seed_margin_from_wall,
    )
    t_mask = time.time() - t0_mask
    mask_np = np.asarray(mask_np, dtype=bool)
    n_fluid = int(mask_np.sum())
    print(f"  mask built in {t_mask:.2f}s, fluid={n_fluid:,}, wall={wall_thickness}, gate={gate_size}")

    mask_cp = cp.asarray(mask_np, dtype=cp.uint8)
    mask_flat_cp = mask_cp.ravel()

    # For this case, mask is connected; thus V == mask.
    Vmask = mask_np
    n_V = int(Vmask.sum())

    maxdim = int(max(D, H, W))
    max_k = int(maxdim.bit_length() - 1)

    t0_kmax = time.time()
    clearance_kmax27 = precompute_los_kmax_27dirs_3d(
        mask_flat_cp,
        D, H, W,
        max_k=max_k,
        init_kernel=None,
        update_kernel=None,
        verbose=False,
    )
    cp.cuda.Device().synchronize()
    print(f"  kmax27 precomputed in {time.time()-t0_kmax:.2f}s\n")

    # -----------------------------
    # Efficient random seeding without np.argwhere(mask)
    # -----------------------------
    def _sample_random_seeds_from_mask(mask_bool, n_seeds, rng, oversample=4.0):
        mask_flat = mask_bool.ravel()
        nvox = mask_flat.size
        need = int(n_seeds)

        chosen = np.empty((0,), dtype=np.int64)
        while chosen.size < need:
            batch_n = int(math.ceil((need - chosen.size) * float(oversample)))
            batch = rng.randint(0, nvox, size=batch_n, dtype=np.int64)
            batch = batch[mask_flat[batch]]
            if batch.size == 0:
                continue
            chosen = np.unique(np.concatenate([chosen, batch], axis=0))
        chosen = chosen[:need]

        HW = H * W
        z = chosen // HW
        rem = chosen - z * HW
        y = rem // W
        x = rem - y * W
        return np.stack([z, y, x], axis=1).astype(np.int64, copy=False)

    # -----------------------------
    # Helper: structural metrics (r_cut / rho_isl) with optional sampling
    # -----------------------------
    def _resolve_struct_sample(n_seeds, rng):
        _struct_sample = struct_metrics_sample
        if bool(fast_mode) and (_struct_sample is None):
            _struct_sample = min(int(n_seeds), 1024)

        if _struct_sample is None:
            return None  # full
        _struct_sample = int(_struct_sample)
        if _struct_sample <= 0:
            return []  # skip
        if n_seeds <= _struct_sample:
            return None  # full
        return rng.choice(int(n_seeds), size=int(_struct_sample), replace=False)

    def _compute_struct_metrics_for_label(label_cp, seeds_np, seed_indices):
        # r_unf on GPU (scalar download only)
        r_unf_val = compute_r_unf(label_cp, mask_cp)

        if seed_indices == []:
            return np.nan, float(r_unf_val), np.nan

        # r_cut / rho_isl on CPU (needs full label volume on CPU)
        label_np = label_cp.get()
        try:
            r_cut_val, rho_isl_val = compute_cell_cut_rate_and_island_ratio(
                label_np, seeds_np, Vmask,
                connectivity=6,
                min_component_voxels=min_comp_vox,
                min_component_fraction=min_comp_frac,
                connect_through_unlabeled=False,
                seed_indices=None if seed_indices is None else seed_indices,
            )
        finally:
            # release large CPU array ASAP
            del label_np
            gc.collect()
        return float(r_cut_val), float(r_unf_val), float(rho_isl_val)

    # -----------------------------
    # Main loop
    # -----------------------------
    results = []
    rng = np.random.RandomState(int(random_state))

        # ------------------------------------------------------------------
    # Print columns (NOW includes ALL metrics used in this script, incl. accuracy metrics)
    # ------------------------------------------------------------------
    print_cols = [
        "n_seeds", "exponent", "method",
        "D", "H", "W", "n_voxels", "n_fluid", "n_V",
        "r_cut", "r_unf", "rho_isl", "e_vox", "eta_roi", "eta_stamp",
        "t_pred", "t_stamp", "t_jfa", "t_close", "t_relax", "t_active_final", "t_build",
        "t_exact_solve", "t_exact_download", "t_exact_metrics", "t_exact_total",
        "mae", "mre", "max_err", "rmse", "acc_rate", "mismatch_rate",
        "status",
    ]

    def _fmt_val(key, v):
        """Pretty/robust scalar formatting for console printing."""
        if v is None:
            return ""
        # ints
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        # floats
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if not np.isfinite(fv):
                return "nan"
            # small sets of consistent precisions
            if key in ("exponent",):
                return f"{fv:.1f}"
            if key in ("r_cut", "r_unf", "rho_isl", "e_vox", "eta_roi", "eta_stamp",
                       "mae", "mre", "rmse", "acc_rate", "mismatch_rate"):
                return f"{fv:.6f}"
            if key in ("max_err",):
                return f"{fv:.6f}"
            if key.startswith("t_"):
                return f"{fv:.6f}"
            return f"{fv:.6g}"
        # strings / others
        return str(v)

    if print_compact_table:
        print("\n" + "=" * 180)
        print("\t".join(print_cols))
        print("=" * 180)
    for i, n_seeds in enumerate(seed_counts):
        n_seeds = int(n_seeds)
        exp_val = float(np.log10(n_seeds)) if n_seeds > 0 else float("nan")
        print(f"\n[{i+1}/{len(seed_counts)}] n_seeds={n_seeds} (10^{exp_val:.1f})")
        print("-" * 120)

        # ---------------------------------------
        # seeds
        # ---------------------------------------
        t0_build = time.perf_counter()
        seeds = _sample_random_seeds_from_mask(mask_np, n_seeds, rng, oversample=4.0)
        t_build = time.perf_counter() - t0_build
        assert_seeds_valid_and_in_pore(mask_np, seeds, backend="numpy")

        seed_indices = _resolve_struct_sample(n_seeds, rng)

        # Containers for method outputs
        out = {}

        # ---------------------------------------
        # M0: Euclidean + clipping
        # ---------------------------------------
        if run_m0:
            t0 = time.perf_counter()
            label_m0_cp, dist_m0_cp, _ = m1_euclidean_voronoi_clipping_gpu(mask_cp, seeds, profile_gpu=False)
            cp.cuda.Device().synchronize()
            t_m0 = time.perf_counter() - t0
            out["M0 Euclid"] = {
                "label": label_m0_cp,
                "dist": dist_m0_cp,
                "t_pred": float(t_m0),
                "t_stamp": 0.0,
                "t_jfa": float(t_m0),
                "t_close": np.nan,
                "t_relax": np.nan,
                "t_active_final": np.nan,
                "eta_roi": np.nan,
                "eta_stamp": np.nan,
            }

        # ---------------------------------------
        # M2: Full OA-JFA
        # ---------------------------------------
        if run_m2:
            t0 = time.perf_counter()
            label_m2_cp, dist_m2_cp = geodesic_voronoi_oajfa_cuda(mask_cp, seeds, profile_gpu=False)
            cp.cuda.Device().synchronize()
            t_m2 = time.perf_counter() - t0
            out["M2 OA-JFA"] = {
                "label": label_m2_cp,
                "dist": dist_m2_cp,
                "t_pred": float(t_m2),
                "t_stamp": 0.0,
                "t_jfa": float(t_m2),
                "t_close": np.nan,
                "t_relax": np.nan,
                "t_active_final": np.nan,
                "eta_roi": np.nan,
                "eta_stamp": np.nan,
            }

        # ---------------------------------------
        # M3: ROI-JFA + Active-Final
        # ---------------------------------------
        if run_m3:
            label_m3_cp, dist_m3_cp, tile_roi_cp, roi_mask_cp = geodesic_voronoi_roi_jfa_with_active_final(
                mask_cp, seeds,
                tile_size=tile_size,
                delta_r=1.0,
                enable_stamping=True,
                verbose=False,
                viz_policy="none",
                relax_eps=1e-6,
                changed_mode="frontier",
                profile_gpu=False,
                return_records=False,
                stamping_kernel=stamp_kern,
                tiles_dual_kernel=tiles_kern,
                active_tiles_kernel=active_kern,
                max_refine_iters=max_refine_iters,
                enable_closure=True,
                clearance_kmax27=clearance_kmax27,
                active_final_iters=int(active_final_iters),
                active_final_band_iters=int(active_final_band_iters),
                active_final_halo_iters=int(active_final_halo_iters),
            )
            cp.cuda.Device().synchronize()

            t_pred  = float(ROI_JFA_LAST_TPRED_WALL)
            t_stamp = float(ROI_JFA_LAST_TSTAMP_WALL)
            t_jfa   = float(ROI_JFA_LAST_TJFA_WALL)
            t_close = float(ROI_JFA_LAST_TCLOSE_WALL)
            t_relax = float(ROI_JFA_LAST_TRELAX_WALL)
            t_af    = float(ROI_JFA_LAST_TAFINAL_WALL)

            eta_roi = compute_eta_roi(roi_mask_cp, mask_flat_cp)
            eta_stamp = (1.0 - float(eta_roi)) if np.isfinite(eta_roi) else np.nan
            # free large helper buffers (not needed after eta computation)
            try:
                del tile_roi_cp
                del roi_mask_cp
            except Exception:
                pass


            # r_unf for M3 is cheap on GPU, but keep consistent via the generic path below
            out["M3 ROI-JFA"] = {
                "label": label_m3_cp,
                "dist": dist_m3_cp,
                "t_pred": float(t_pred),
                "t_stamp": float(t_stamp),
                "t_jfa": float(t_jfa),
                "t_close": float(t_close),
                "t_relax": float(t_relax),
                "t_active_final": float(t_af),
                "eta_roi": float(eta_roi),
                "eta_stamp": float(eta_stamp),
            }

        # ---------------------------------------
        # M1: Exact (reference)  (computed once; reused for e_vox + accuracy metrics)
        # ---------------------------------------
        label_exact_cp = None
        dist_exact_cp = None
        t_exact_solve = np.nan
        t_exact_download = np.nan
        t_exact_metrics = np.nan
        t_exact_total = np.nan
        exact_status = "SKIP"

        if run_exact and (n_seeds <= int(max_seeds_for_exact)):
            try:
                cp.cuda.Device().synchronize()
                print("  Computing M1 Exact ...")
                t0p = time.perf_counter()
                label_exact_cp, dist_exact_cp = exact_geodesic_voronoi_gpu(
                    mask_cp,
                    seeds,
                    connectivity=26,
                    max_iter=None,
                    eps=1e-6,
                    verbose=False,
                    profile_gpu=True,
                    check_optimality=False,
                )
                cp.cuda.Device().synchronize()
                t_exact_solve = time.perf_counter() - t0p

                # Keep on GPU (avoid full download); metrics are GPU scalar ops
                t_exact_download = 0.0
                t_exact_metrics = 0.0
                t_exact_total = float(t_exact_solve) + float(t_exact_download) + float(t_exact_metrics)
                exact_status = "OK"
            except Exception as e:
                import traceback
                traceback.print_exc()
                exact_status = f"ERROR: {str(e)[:60]}"
        elif run_exact:
            exact_status = f"SKIP(max_seeds_for_exact={int(max_seeds_for_exact)})"

        if label_exact_cp is not None and dist_exact_cp is not None:
            out["M1 Exact"] = {
                "label": label_exact_cp,
                "dist": dist_exact_cp,
                "t_pred": float(t_exact_solve),
                "t_stamp": 0.0,
                "t_jfa": float(t_exact_solve),
                "t_close": np.nan,
                "t_relax": np.nan,
                "t_active_final": np.nan,
                "eta_roi": np.nan,
                "eta_stamp": np.nan,
            }

        # ---------------------------------------
        # Per-method metrics & save rows
        # ---------------------------------------
        method_order = []
        if run_m0: method_order.append("M0 Euclid")
        # Always place Exact in-between if available or requested
        if run_exact: method_order.append("M1 Exact")
        if run_m2: method_order.append("M2 OA-JFA")
        if run_m3: method_order.append("M3 ROI-JFA")

        for method in method_order:
            # handle Exact skipped case
            if method == "M1 Exact" and ("M1 Exact" not in out):

                row_skip = {
                    "n_seeds": n_seeds,
                    "exponent": round(exp_val, 1),
                    "method": "M1 Exact",
                    "D": D, "H": H, "W": W,
                    "n_voxels": D*H*W,
                    "n_fluid": n_fluid,
                    "n_V": n_V,

                    "r_cut": np.nan,
                    "r_unf": np.nan,
                    "rho_isl": np.nan,
                    "e_vox": np.nan,
                    "eta_roi": np.nan,
                    "eta_stamp": np.nan,

                    "t_pred": np.nan,
                    "t_stamp": 0.0,
                    "t_jfa": np.nan,
                    "t_close": np.nan,
                    "t_relax": np.nan,
                    "t_active_final": np.nan,
                    "t_build": float(t_build),

                    "t_exact_solve": float(t_exact_solve),
                    "t_exact_download": float(t_exact_download),
                    "t_exact_metrics": float(t_exact_metrics),
                    "t_exact_total": float(t_exact_total),

                    "mae": np.nan,
                    "mre": np.nan,
                    "max_err": np.nan,
                    "rmse": np.nan,
                    "acc_rate": np.nan,
                    "mismatch_rate": np.nan,

                    "status": exact_status,
                }

                results.append(row_skip)

                if print_compact_table:

                    print("\t".join(_fmt_val(k, row_skip.get(k, None)) for k in print_cols))

                if print_all_params:

                    import json

                    print("ALL_PARAMS=" + json.dumps(row_skip, ensure_ascii=False, allow_nan=True))

                continue

            if method not in out:
                continue

            lab = out[method]["label"]
            dist = out[method]["dist"]

            # structural metrics
            try:
                r_cut, r_unf, rho_isl = _compute_struct_metrics_for_label(lab, seeds, seed_indices)
            except Exception as e:
                import traceback
                traceback.print_exc()
                r_cut = r_unf = rho_isl = np.nan

            # end-to-end / accuracy vs exact (if available)
            mae = mre = max_err = rmse = acc_rate = mismatch_rate = e_vox = np.nan
            if (label_exact_cp is not None) and (dist_exact_cp is not None):
                if method == "M1 Exact":
                    mae = 0.0
                    mre = 0.0
                    max_err = 0.0
                    rmse = 0.0
                    acc_rate = 1.0
                    mismatch_rate = 0.0
                    e_vox = 0.0
                else:
                    acc = compute_accuracy_metrics(dist, lab, dist_exact_cp, label_exact_cp, mask_cp)
                    mae = acc["mae"]
                    mre = acc["mre"]
                    max_err = acc["max_err"]
                    rmse = acc["rmse"]
                    acc_rate = acc["acc_rate"]
                    mismatch_rate = acc["mismatch_rate"]
                    e_vox = mismatch_rate  # same definition on V

            row = {
                "n_seeds": n_seeds,
                "exponent": round(exp_val, 1),
                "method": method,
                "D": D, "H": H, "W": W,
                "n_voxels": D*H*W,
                "n_fluid": n_fluid,
                "n_V": n_V,

                "r_cut": r_cut,
                "r_unf": r_unf,
                "rho_isl": rho_isl,
                "e_vox": e_vox,
                "eta_roi": out[method]["eta_roi"],
                "eta_stamp": out[method]["eta_stamp"],

                "t_pred": out[method]["t_pred"],
                "t_stamp": out[method]["t_stamp"],
                "t_jfa": out[method]["t_jfa"],
                "t_close": out[method]["t_close"],
                "t_relax": out[method]["t_relax"],
                "t_active_final": out[method]["t_active_final"],
                "t_build": float(t_build),

                "t_exact_solve": float(t_exact_solve),
                "t_exact_download": float(t_exact_download),
                "t_exact_metrics": float(t_exact_metrics),
                "t_exact_total": float(t_exact_total),

                "mae": mae,
                "mre": mre,
                "max_err": max_err,
                "rmse": rmse,
                "acc_rate": acc_rate,
                "mismatch_rate": mismatch_rate,

                "status": "OK",
            }
            results.append(row)            # print (now includes ALL parameters/metrics)
            if print_compact_table:
                print("	".join(_fmt_val(k, row.get(k, None)) for k in print_cols))
            if print_all_params:
                import json
                print("ALL_PARAMS=" + json.dumps(row, ensure_ascii=False, allow_nan=True))
        # ---------------------------------------
        # cleanup GPU references for this iter
        # ---------------------------------------
        try:
            for v in out.values():
                try:
                    del v["label"]
                    del v["dist"]
                except Exception:
                    pass
            del out
            if label_exact_cp is not None:
                del label_exact_cp
            if dist_exact_cp is not None:
                del dist_exact_cp
        except Exception:
            pass

        try:
            if free_memory_each_iter:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    # -----------------------------
    # Write CSV (long format)
    # -----------------------------
    fieldnames = [
        "n_seeds", "exponent", "method",
        "D", "H", "W", "n_voxels", "n_fluid", "n_V",
        "r_cut", "r_unf", "rho_isl", "e_vox", "eta_roi", "eta_stamp",
        "t_pred", "t_stamp", "t_jfa", "t_close", "t_relax", "t_active_final", "t_build",
        "t_exact_solve", "t_exact_download", "t_exact_metrics", "t_exact_total",
        "mae", "mre", "max_err", "rmse", "acc_rate", "mismatch_rate",
        "status",
    ]

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 120)
    print(f"Saved to: {output_csv}")
    ok_cnt = sum(1 for r in results if r.get("status") == "OK")
    print(f"Rows: {len(results)}, OK: {ok_cnt}")
    print("=" * 120)
    return results


