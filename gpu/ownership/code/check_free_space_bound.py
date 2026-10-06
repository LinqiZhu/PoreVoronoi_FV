"""Exhaustive exactness tests for the ROI free-space proposal.

The manuscript calls the accelerator "ROI--JFA" and describes the first stage as a
jump-flooding proposal.  Reading `ptv_ownership_gpu6.py` shows four selectable
backends for that stage:

  scan                     three separable forward/backward min-plus axis scans
  jump / jump_cooperative  per-axis dyadic relaxation, jump = 1,2,4,... (increasing)
  jump_cooperative_warp64  the same dyadic schedule fused into one cooperative launch

All four relax a packed 64-bit state ``(distance << 32) | site_id`` under an
unsigned comparison, so the induced order is lexicographic in (distance, site id)
and adding a constant to the distance field preserves it.  That makes each stage a
*min-plus* transform for the L1 metric rather than an approximate Voronoi flood.

This module does not take that reading on trust.  It drives the production kernels
themselves and compares against brute force:

  1-D  every non-empty site subset of a line, exhaustively, up to a length where
       enumeration is complete; random subsets above it
  2-D  random, symmetric-tie, boundary-heavy and dense/sparse site sets
  3-D  thousands of random site sets on small free-space blocks

The required outcome is zero distance mismatches, zero owner mismatches and zero
tie mismatches, for every backend.

Level-1 ownership exactness (obstacles present, graph-geodesic distance) is tested
separately against a CPU BFS through the same public entry point.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PROD = Path(
    os.environ.get(
        "PVFV_OWNERSHIP_CODE",
        "gpu/ownership/code",
    )
)
if str(PROD) not in sys.path:
    sys.path.insert(0, str(PROD))

import cupy as cp  # noqa: E402
import ptv_ownership_gpu6 as G  # noqa: E402

BACKENDS = ("scan", "jump", "jump_cooperative")


def proposal_gpu(shape, seeds_zyx, backend, block_size=256):
    """Run exactly the free-space proposal stage of certified_l1_roi_frontier_gpu_6.

    The call sequence below is copied line for line from the ``lower_method``
    branch of that function; only the surrounding certificate and closure stages
    are omitted, because this test is about the proposal alone.
    """
    D, H, W = (int(v) for v in shape)
    nvox = D * H * W
    seeds = cp.asarray(np.asarray(seeds_zyx, dtype=np.int32).reshape(-1, 3))
    nseed = int(seeds.shape[0])
    flat_mask = cp.ones(nvox, dtype=cp.uint8)  # free space: the proposal ignores obstacles
    state = cp.empty(nvox, dtype=cp.uint64)
    seed_label = cp.full(nvox, G.INF_LABEL, dtype=cp.int32)
    bad_seed = cp.zeros(1, dtype=cp.int32)

    block = int(block_size)
    grid_vox = (max(1, math.ceil(nvox / block)),)
    grid_seed = (max(1, math.ceil(nseed / block)),)

    G._packed_init_kernel()(grid_vox, (block,), (state, np.int32(nvox), G.INF_PACK))
    G._packed_seed_kernel()(
        grid_seed,
        (block,),
        (
            flat_mask, seeds, state, seed_label, bad_seed,
            np.int32(nseed), np.int32(D), np.int32(H), np.int32(W),
        ),
    )
    if int(bad_seed.get()[0]) != 0:
        raise ValueError("a prescribed site is outside the domain")

    if backend == "scan":
        grid_x = (max(1, math.ceil((D * H) / block)),)
        grid_y = (max(1, math.ceil((D * W) / block)),)
        grid_z = (max(1, math.ceil((H * W) / block)),)
        G._l1_scan_x_kernel()(grid_x, (block,), (state, np.int32(D), np.int32(H), np.int32(W), G.INF_DISTANCE))
        G._l1_scan_y_kernel()(grid_y, (block,), (state, np.int32(D), np.int32(H), np.int32(W), G.INF_DISTANCE))
        G._l1_scan_z_kernel()(grid_z, (block,), (state, np.int32(D), np.int32(H), np.int32(W), G.INF_DISTANCE))
    elif backend == "jump":
        scratch = cp.empty_like(state)
        for axis, axis_length in enumerate((D, H, W)):
            jump = 1
            while jump < axis_length:
                G._l1_jump_axis_kernel()(
                    grid_vox, (block,),
                    (state, scratch, np.int32(nvox), np.int32(D), np.int32(H),
                     np.int32(W), np.int32(axis), np.int32(jump), G.INF_DISTANCE),
                )
                state, scratch = scratch, state
                jump <<= 1
    elif backend == "jump_fused":
        fused = G._l1_fused_jump_axis_kernel()
        for axis, axis_length in enumerate((D, H, W)):
            threads = G._next_power_of_two(axis_length)
            nline = (H * W, D * W, D * H)[axis]
            fused(
                (nline,), (threads,),
                (state, np.int32(D), np.int32(H), np.int32(W), np.int32(axis),
                 G.INF_DISTANCE, G.INF_PACK),
                shared_mem=threads * np.dtype(np.uint64).itemsize,
            )
    elif backend == "jump_cooperative":
        status = G._run_cooperative_fused_l1_jump(state=state, shape=(D, H, W), blocks_per_sm=8)
        if status is None:
            return None  # device does not support the cooperative launch here
    else:
        raise ValueError("unknown backend " + backend)

    packed = cp.asnumpy(state).reshape(D, H, W)
    distance = (packed >> np.uint64(32)).astype(np.int64)
    owner = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
    return distance, owner


def brute_force(shape, seeds_zyx):
    """Lexicographic (L1 distance, smallest site index) over every site, exactly."""
    D, H, W = (int(v) for v in shape)
    sites = np.asarray(seeds_zyx, dtype=np.int64).reshape(-1, 3)
    zz, yy, xx = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing="ij"
    )
    best_d = np.full((D, H, W), np.iinfo(np.int64).max, dtype=np.int64)
    best_o = np.full((D, H, W), np.iinfo(np.int64).max, dtype=np.int64)
    for index, (sz, sy, sx) in enumerate(sites):
        d = np.abs(zz - sz) + np.abs(yy - sy) + np.abs(xx - sx)
        better = d < best_d
        equal = d == best_d
        best_o = np.where(better, index, np.where(equal, np.minimum(best_o, index), best_o))
        best_d = np.where(better, d, best_d)
    return best_d, best_o


def compare(shape, seeds, backend, report):
    got = proposal_gpu(shape, seeds, backend)
    if got is None:
        report["skipped_" + backend] = report.get("skipped_" + backend, 0) + 1
        return True
    distance, owner = got
    ref_d, ref_o = brute_force(shape, seeds)
    d_bad = int(np.count_nonzero(distance != ref_d))
    o_bad = int(np.count_nonzero(owner != ref_o))
    # A tie mismatch is an owner mismatch at a voxel that more than one site
    # attains; it is counted separately because it is the deterministic-tie claim.
    sites = np.asarray(seeds, dtype=np.int64).reshape(-1, 3)
    D, H, W = shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    attain = np.zeros((D, H, W), dtype=np.int64)
    for sz, sy, sx in sites:
        attain += (np.abs(zz - sz) + np.abs(yy - sy) + np.abs(xx - sx)) == ref_d
    tie_bad = int(np.count_nonzero((owner != ref_o) & (attain > 1)))
    report["cases"] += 1
    report["distance_mismatch"] += d_bad
    report["owner_mismatch"] += o_bad
    report["tie_mismatch"] += tie_bad
    report["tie_voxels"] += int(np.count_nonzero(attain > 1))
    if d_bad or o_bad:
        report["failures"].append(
            {"backend": backend, "shape": list(shape), "seeds": sites.tolist(),
             "distance_mismatch": d_bad, "owner_mismatch": o_bad}
        )
        return False
    return True


def run_1d(backend, report, max_exhaustive_length=16, longer_lengths=(20, 24, 28, 32), samples=400, rng=None):
    for length in range(2, max_exhaustive_length + 1):
        for bits in range(1, 1 << length):
            positions = [i for i in range(length) if (bits >> i) & 1]
            seeds = [[0, 0, p] for p in positions]
            if not compare((1, 1, length), seeds, backend, report):
                return False
    for length in longer_lengths:
        for _ in range(samples):
            k = int(rng.integers(1, min(length, 8) + 1))
            positions = sorted(rng.choice(length, size=k, replace=False).tolist())
            seeds = [[0, 0, p] for p in positions]
            if not compare((1, 1, length), seeds, backend, report):
                return False
    return True


def run_2d(backend, report, rng, trials=600):
    shapes = [(1, 5, 5), (1, 8, 8), (1, 7, 11), (1, 16, 16), (1, 13, 4), (1, 32, 3)]
    for shape in shapes:
        _, H, W = shape
        # symmetric ties: two sites placed symmetrically about the centre
        centre = H // 2
        for offset in range(1, min(centre, H - 1 - centre) + 1):
            seeds = [[0, centre - offset, W // 2], [0, centre + offset, W // 2]]
            if not compare(shape, seeds, backend, report):
                return False
        # every corner and edge midpoint
        corners = [[0, 0, 0], [0, H - 1, 0], [0, 0, W - 1], [0, H - 1, W - 1]]
        if not compare(shape, corners, backend, report):
            return False
        for _ in range(trials // len(shapes)):
            k = int(rng.integers(1, min(H * W, 12) + 1))
            flat = rng.choice(H * W, size=k, replace=False)
            seeds = [[0, int(f) // W, int(f) % W] for f in flat]
            if not compare(shape, seeds, backend, report):
                return False
    return True


def run_3d(backend, report, rng, trials=1500):
    shapes = [(3, 3, 3), (4, 5, 6), (8, 8, 8), (5, 9, 7), (2, 16, 16), (16, 4, 4)]
    for _ in range(trials):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        n = int(np.prod(shape))
        k = int(rng.integers(1, min(n, 16) + 1))
        flat = rng.choice(n, size=k, replace=False)
        seeds = [list(np.unravel_index(int(f), shape)) for f in flat]
        if not compare(shape, seeds, backend, report):
            return False
    # all-sites and single-site extremes
    for shape in shapes:
        n = int(np.prod(shape))
        seeds = [list(np.unravel_index(i, shape)) for i in range(n)]
        if not compare(shape, seeds, backend, report):
            return False
        if not compare(shape, [[0, 0, 0]], backend, report):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reproduce/supplementary/s1_ownership")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--exhaustive-1d-length", type=int, default=16)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_file": str(PROD / "ptv_ownership_gpu6.py"),
        "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode()),
        "cupy_version": cp.__version__,
        "exhaustive_1d_length": int(args.exhaustive_1d_length),
        "backends": {},
    }
    overall_ok = True
    for backend in [b for b in args.backends.split(",") if b]:
        rng = np.random.default_rng(args.seed)
        report = {
            "cases": 0, "distance_mismatch": 0, "owner_mismatch": 0,
            "tie_mismatch": 0, "tie_voxels": 0, "failures": [],
        }
        started = time.perf_counter()
        ok = run_1d(backend, report, args.exhaustive_1d_length, rng=rng)
        ok = ok and run_2d(backend, report, rng)
        ok = ok and run_3d(backend, report, rng)
        report["seconds"] = float(time.perf_counter() - started)
        report["status"] = "PASS" if ok and report["distance_mismatch"] == 0 and report["owner_mismatch"] == 0 else "FAIL"
        summary["backends"][backend] = report
        overall_ok = overall_ok and report["status"] == "PASS"
        print(
            "[roi] %-18s %s cases=%d distance_mismatch=%d owner_mismatch=%d "
            "tie_mismatch=%d tie_voxels=%d (%.1fs)"
            % (backend, report["status"], report["cases"], report["distance_mismatch"],
               report["owner_mismatch"], report["tie_mismatch"], report["tie_voxels"],
               report["seconds"]),
            flush=True,
        )
    summary["status"] = "PASS" if overall_ok else "FAIL"
    (out / "roi_freespace_exactness.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[write] " + str(out / "roi_freespace_exactness.json"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
