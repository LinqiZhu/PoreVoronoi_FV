from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE / "code"
SHARED_CODE_DIR = PACKAGE.parent / "code"
for path in (CODE_DIR, SHARED_CODE_DIR):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import ptv_ownership_gpu6 as backend  # noqa: E402
from hybrid_site_sources import (  # noqa: E402
    load_particle_window_seed_flat,
    parse_integer_selection,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "mask" not in data.files:
            raise KeyError(f"{path} does not contain a 'mask' array")
        mask = np.asarray(data["mask"], dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    return mask


def _timing_record(samples_s: list[float]) -> dict[str, Any]:
    ordered = sorted(samples_s)
    return {
        "repeats": len(samples_s),
        "times_s": samples_s,
        "min_s": float(ordered[0]),
        "q1_s": float(np.quantile(ordered, 0.25)),
        "median_s": float(statistics.median(ordered)),
        "q3_s": float(np.quantile(ordered, 0.75)),
        "max_s": float(ordered[-1]),
    }


def _timed_interleaved(
    calls: dict[str, Callable[[], tuple[Any, ...]]],
    *,
    repeats: int,
    cp: Any,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, list[float]]]:
    names = list(calls)
    outputs: dict[str, tuple[Any, ...]] = {}
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if repeat % 2 == 1:
            order = list(reversed(order))
        for name in order:
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            outputs[name] = calls[name]()
            cp.cuda.Stream.null.synchronize()
            samples[name].append(float(time.perf_counter() - start))
    return outputs, samples


def _paired_speedup_record(
    exact_samples_s: list[float],
    candidate_samples_s: list[float],
    *,
    seed: int = 20260717,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    exact = np.asarray(exact_samples_s, dtype=np.float64)
    candidate = np.asarray(candidate_samples_s, dtype=np.float64)
    if exact.shape != candidate.shape:
        raise ValueError("paired timing vectors must have the same shape")
    ratios = exact / candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, ratios.size, size=(bootstrap_samples, ratios.size))
    bootstrap_medians = np.median(ratios[indices], axis=1)
    low, high = np.quantile(bootstrap_medians, [0.025, 0.975])
    return {
        "definition": "per-repeat exact wall time divided by candidate wall time",
        "paired_median": float(np.median(ratios)),
        "bootstrap_95_percent_interval": [float(low), float(high)],
        "candidate_faster_fraction": float(np.mean(exact > candidate)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
    }


def _comparison(reference: tuple[Any, ...], candidate: tuple[Any, ...], mask_cp: Any, cp: Any):
    reference_label, reference_distance = reference[:2]
    candidate_label, candidate_distance = candidate[:2]
    pore = mask_cp != 0
    label_mismatch = int(cp.count_nonzero(pore & (candidate_label != reference_label)).get())
    distance_difference = cp.abs(
        candidate_distance.astype(cp.float64, copy=False)
        - reference_distance.astype(cp.float64, copy=False)
    )
    distance_mismatch = int(cp.count_nonzero(pore & (distance_difference != 0.0)).get())
    return {
        "pore_voxels": int(cp.count_nonzero(pore).get()),
        "label_mismatch_voxels": label_mismatch,
        "distance_mismatch_voxels": distance_mismatch,
        "mean_abs_distance_difference": float(cp.mean(distance_difference[pore]).get()),
        "max_abs_distance_difference": float(cp.max(distance_difference[pore]).get()),
        "exact_label_and_distance_match": bool(label_mismatch == 0 and distance_mismatch == 0),
    }


def _nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact and certified arbitrary-site ownership backends on the same "
            "unit-weight six-neighbour pore graph."
        )
    )
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--particle-window", type=Path, required=True)
    parser.add_argument("--particle-frames", default="")
    parser.add_argument("--particle-ids", default="")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-stem", default="ptv_ownership_backend_audit_gpu6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmups < 1:
        raise ValueError("--warmups must be positive")
    if args.repeats < 3:
        raise ValueError("--repeats must be at least three")

    mask_np = _load_mask(args.mask_npz)
    seed_flat, site_metadata = load_particle_window_seed_flat(
        mask_np,
        args.particle_window,
        frame_selection=str(args.particle_frames),
        particle_id_selection=str(args.particle_ids),
    )
    seeds_zyx = np.column_stack(np.unravel_index(seed_flat, mask_np.shape)).astype(
        np.int32, copy=False
    )

    import cupy as cp

    mask_cp = cp.asarray(mask_np, dtype=cp.uint8)
    seeds_cp = cp.asarray(seeds_zyx, dtype=cp.int32)

    selected_particle_ids = parse_integer_selection(
        str(args.particle_ids), label="particle id"
    )
    particle_count = (
        len(selected_particle_ids) if selected_particle_ids is not None else None
    )
    pore_voxel_count = int(np.count_nonzero(mask_np))
    certificate_line_bits = backend.prepare_mask_line_bits64(mask_cp)

    def exact_gpu_propagation_call() -> tuple[Any, ...]:
        return backend.exact_frontier_dijkstra_gpu_6(
            mask_cp,
            seeds_cp,
            validate=False,
            return_float64=False,
        )

    def roi_jfa_call() -> tuple[Any, ...]:
        return backend.certified_roi_jfa_gpu_6(
            mask_cp,
            seeds_cp,
            block_size=256,
            closure_block_size=64,
            validate=False,
            return_float64=False,
            certificate_line_bits=certificate_line_bits,
            pore_voxel_count=pore_voxel_count,
        )

    calls = {
        "exact_gpu_propagation_6": exact_gpu_propagation_call,
        "roi_jfa_gpu_6": roi_jfa_call,
    }
    for _ in range(args.warmups):
        for call in calls.values():
            call()
    cp.cuda.Stream.null.synchronize()

    outputs, samples = _timed_interleaved(calls, repeats=args.repeats, cp=cp)
    timing = {name: _timing_record(values) for name, values in samples.items()}
    reference_output = backend.exact_frontier_dijkstra_gpu_6(
        mask_cp, seeds_cp, validate=True, return_float64=False
    )
    roi_jfa_output = backend.certified_roi_jfa_gpu_6(
        mask_cp,
        seeds_cp,
        block_size=256,
        closure_block_size=64,
        validate=True,
        return_float64=False,
        return_stats=True,
        certificate_line_bits=certificate_line_bits,
        pore_voxel_count=pore_voxel_count,
        profile_stages=True,
    )
    roi_jfa_comparison = _comparison(
        reference_output, roi_jfa_output, mask_cp, cp
    )

    exact_median = float(timing["exact_gpu_propagation_6"]["median_s"])
    roi_jfa_median = float(timing["roi_jfa_gpu_6"]["median_s"])
    roi_jfa_stats = dict(roi_jfa_output[2])
    paired_speedup = _paired_speedup_record(
        samples["exact_gpu_propagation_6"],
        samples["roi_jfa_gpu_6"],
    )

    line_bit_samples: list[float] = []
    for _ in range(args.warmups):
        backend.prepare_mask_line_bits64(mask_cp)
    cp.cuda.Stream.null.synchronize()
    for _ in range(args.repeats):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        backend.prepare_mask_line_bits64(mask_cp)
        cp.cuda.Stream.null.synchronize()
        line_bit_samples.append(float(time.perf_counter() - start))
    line_bit_timing = _timing_record(line_bit_samples)

    module_path = CODE_DIR / "ptv_ownership_gpu6.py"
    loader_path = SHARED_CODE_DIR / "hybrid_site_sources.py"
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if hasattr(device_name, "decode"):
        device_name = device_name.decode()

    payload = {
        "protocol": {
            "site_source": "flow_or_measurement_derived_particle_trajectory",
            "regular_site_generation": False,
            "auxiliary_site_count": 0,
            "graph_connectivity": 6,
            "edge_weight": 1,
            "diagonal_moves": False,
            "tie_rule": "minimum_site_id_from_sorted_unique_flat_voxel_ids",
            "candidate_parameters": "none",
            "cuda_launch_configuration": {
                "roi_jfa_voxel_block_size": 256,
                "roi_jfa_lower_bound_warp_block_size": 256,
                "roi_jfa_closure_block_size": 64,
                "roi_jfa_closure_requested_blocks_per_sm": 8,
                "roi_jfa_lower_bound_requested_blocks_per_sm": 8,
                "selection_rule": "one fixed configuration chosen across all five particle prefixes",
            },
            "jump_schedule": (
                "powers of two on each 64-voxel coordinate line, with one warp "
                "holding two packed positions per lane and grid synchronization "
                "between coordinate axes"
            ),
            "candidate_certificate": (
                "cached 64-bit pore-line masks plus a pore-valid monotone "
                "face-step path"
            ),
            "unresolved_closure": (
                "single-writer cooperative unit-edge six-neighbour relaxation to a "
                "zero-residual fixed point"
            ),
            "timing_order": "interleaved cyclic order with odd-repeat reversal",
            "warmup_calls": int(args.warmups),
            "timed_repeats": int(args.repeats),
            "timing_reference": "exact six-neighbour GPU propagation",
            "timing_boundary": (
                "GPU-resident ownership call; synchronized host wall time; warm CUDA JIT and "
                "memory pool; per-call lower-bound propagation, path certification, fused "
                "closure-state initialization, ROI closure, and device output materialization "
                "included"
            ),
            "excluded_from_timing": [
                "particle file loading",
                "trajectory-to-unique-site reduction",
                "host-to-device input transfer",
                "first-time CUDA compilation",
                "device-to-host output transfer",
                "mask-only 64-bit pore-line cache construction, timed separately",
                "optional post-run fixed-point and pointwise reference audits",
            ],
        },
        "input": {
            "mask_npz": str(args.mask_npz.resolve()),
            "particle_window": str(args.particle_window.resolve()),
            "particle_frames": site_metadata["particle_frames"],
            "particle_ids": site_metadata["particle_ids"],
            "particle_count": particle_count,
            "particle_rows_selected": int(site_metadata["particle_rows_selected"]),
            "particle_site_count": int(seed_flat.size),
            "pore_voxel_count": pore_voxel_count,
            "mask_sha256": _sha256(args.mask_npz),
            "particle_window_sha256": _sha256(args.particle_window),
        },
        "code_sha256": {
            "audit_ptv_ownership_backends.py": _sha256(Path(__file__)),
            "ptv_ownership_gpu6.py": _sha256(module_path),
            "hybrid_site_sources.py": _sha256(loader_path),
        },
        "comparison": {"roi_jfa_gpu_6": roi_jfa_comparison},
        "candidate_structure": {
            "roi_jfa": {
                key: value
                for key, value in roi_jfa_stats.items()
                if not key.endswith("_s") and key != "total_s"
            },
        },
        "profiled_validation_run": {"roi_jfa": roi_jfa_stats},
        "timing": {
            **timing,
            "mask_line_bit_cache_build_gpu_6": line_bit_timing,
        },
        "derived": {
            "exact_gpu_propagation_over_roi_jfa": exact_median / roi_jfa_median,
            "paired_speedup": paired_speedup,
        },
        "decision": {
            "roi_jfa_backend_accepted": bool(
                roi_jfa_comparison["exact_label_and_distance_match"]
            ),
            "speedup_observed": bool(
                roi_jfa_comparison["exact_label_and_distance_match"]
                and paired_speedup["bootstrap_95_percent_interval"][0] > 1.0
            ),
            "scope": "this mask, trajectory window, particle selection, hardware, and timing boundary",
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "gpu": str(device_name),
            "nvidia_smi": _nvidia_smi(),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.out_stem}.json"
    csv_path = args.out_dir / f"{args.out_stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    row = {
        "particle_ids": site_metadata["particle_ids"],
        "particle_count": particle_count,
        "particle_rows_selected": int(site_metadata["particle_rows_selected"]),
        "particle_site_count": int(seed_flat.size),
        "pore_voxels": pore_voxel_count,
        "pore_voxels_per_site": float(pore_voxel_count / seed_flat.size),
        "certified_voxels": int(roi_jfa_stats["certified_voxels"]),
        "roi_voxels": int(roi_jfa_stats["roi_voxels"]),
        "roi_fraction": float(roi_jfa_stats["roi_fraction"]),
        "closure_iterations": int(roi_jfa_stats["closure_iterations"]),
        "roi_jfa_label_mismatch_voxels": int(
            roi_jfa_comparison["label_mismatch_voxels"]
        ),
        "roi_jfa_distance_mismatch_voxels": int(
            roi_jfa_comparison["distance_mismatch_voxels"]
        ),
        "exact_gpu_propagation_median_ms": 1000.0 * exact_median,
        "roi_jfa_median_ms": 1000.0 * roi_jfa_median,
        "mask_line_bit_cache_build_median_ms": 1000.0
        * float(line_bit_timing["median_s"]),
        "exact_gpu_propagation_over_roi_jfa": exact_median / roi_jfa_median,
        "paired_speedup_median": float(paired_speedup["paired_median"]),
        "paired_speedup_ci_low": float(
            paired_speedup["bootstrap_95_percent_interval"][0]
        ),
        "paired_speedup_ci_high": float(
            paired_speedup["bootstrap_95_percent_interval"][1]
        ),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2, sort_keys=True))
    print(json_path)


if __name__ == "__main__":
    main()
