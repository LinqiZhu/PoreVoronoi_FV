from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / ".." / "ownership" / "../../reproduce/table_02/records"
CODE_DIR = ROOT / ".." / "ownership" / "code"
SHARED_CODE_DIR = ROOT / ".." / "code"
for directory in (CODE_DIR, SHARED_CODE_DIR):
    sys.path.insert(0, str(directory))

import ptv_ownership_gpu6 as backend  # noqa: E402
from hybrid_site_sources import load_particle_window_seed_flat  # noqa: E402


COUNTS = (10, 50, 100, 200, 500)
SWEEP_ROUNDS = 3
WARMUPS = 5
REPEATS = 101
ROBUST_SET_TOLERANCE = 1.03

PRODUCTION_CONFIG = {
    "block_size": 256,
    "closure_block_size": 64,
    "cooperative_blocks_per_sm": 8,
    "closure_coordinate_method": "power2_warp_vote",
    "lower_bound_method": "jump_cooperative_warp64",
    "lower_warp_block_size": 256,
    "lower_cooperative_blocks_per_sm": 8,
    "certificate_method": "path_bitset_compact64",
}

STAGES: dict[str, list[dict[str, Any]]] = {
    "voxel_block_size": [
        {"candidate": f"voxel_b{value}", "block_size": value}
        for value in (32, 64, 128, 256, 512)
    ],
    "voxel_block_confirmation": [
        {"candidate": f"voxel_b{value}", "block_size": value}
        for value in (64, 256, 512)
    ],
    "closure_block_size": [
        {"candidate": f"closure_b{value}", "closure_block_size": value}
        for value in (32, 64, 128, 256, 512)
    ],
    "closure_residency": [
        {
            "candidate": f"closure_r{value}",
            "cooperative_blocks_per_sm": value,
        }
        for value in (1, 2, 4, 8, 12, 16, 24)
    ],
    "lower_bound_launch": [
        {
            "candidate": f"lower_warp_b{block}_r{residency}",
            "lower_warp_block_size": block,
            "lower_cooperative_blocks_per_sm": residency,
        }
        for block, residency in (
            (128, 12),
            (128, 16),
            (256, 8),
            (256, 12),
            (256, 16),
            (512, 4),
        )
    ],
}

PRODUCTION_CANDIDATE = {
    "voxel_block_size": "voxel_b256",
    "voxel_block_confirmation": "voxel_b256",
    "closure_block_size": "closure_b64",
    "closure_residency": "closure_r8",
    "lower_bound_launch": "lower_warp_b256_r8",
}

STAGE_PROTOCOL = {
    stage: {"rounds": SWEEP_ROUNDS, "warmups": WARMUPS, "repeats": REPEATS}
    for stage in STAGES
}
STAGE_PROTOCOL["voxel_block_confirmation"] = {
    "rounds": 5,
    "warmups": 20,
    "repeats": 501,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["mask"], dtype=bool)


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(array <= 0):
        raise ValueError("geometric mean requires positive observations")
    return float(np.exp(np.mean(np.log(array))))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare(
    reference: tuple[Any, ...],
    candidate: tuple[Any, ...],
    mask_cp: Any,
    cp: Any,
) -> tuple[int, int]:
    pore = mask_cp != 0
    label_bad = cp.count_nonzero(pore & (reference[0] != candidate[0]))
    distance_bad = cp.count_nonzero(pore & (reference[1] != candidate[1]))
    return int(label_bad.get()), int(distance_bad.get())


def timed_stage(
    calls: dict[str, Callable[[], tuple[Any, ...]]],
    *,
    cp: Any,
    seed: int,
    warmups: int,
    repeats: int,
) -> dict[str, list[float]]:
    names = list(calls)
    samples = {name: [] for name in names}
    for _ in range(warmups):
        for name in names:
            calls[name]()
    cp.cuda.Stream.null.synchronize()

    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        order = [names[int(index)] for index in rng.permutation(len(names))]
        for name in order:
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            calls[name]()
            cp.cuda.Stream.null.synchronize()
            samples[name].append(float(1000.0 * (time.perf_counter() - start)))
    return samples


def merged_config(overrides: dict[str, Any]) -> dict[str, Any]:
    config = dict(PRODUCTION_CONFIG)
    config.update({key: value for key, value in overrides.items() if key != "candidate"})
    return config


def candidate_call(
    mask_cp: Any,
    seeds_cp: Any,
    line_bits: tuple[Any, Any, Any],
    config: dict[str, Any],
) -> tuple[Any, ...]:
    return backend.certified_l1_roi_frontier_gpu_6(
        mask_cp,
        seeds_cp,
        block_size=int(config["block_size"]),
        closure_block_size=int(config["closure_block_size"]),
        validate=False,
        return_float64=False,
        return_stats=False,
        lower_bound_method=str(config["lower_bound_method"]),
        lower_warp_block_size=int(config["lower_warp_block_size"]),
        lower_cooperative_blocks_per_sm=int(
            config["lower_cooperative_blocks_per_sm"]
        ),
        certificate_method=str(config["certificate_method"]),
        certificate_line_bits=line_bits,
        closure_method="cooperative",
        cooperative_blocks_per_sm=int(config["cooperative_blocks_per_sm"]),
        closure_kernel_method="roi_single_writer",
        closure_coordinate_method=str(config["closure_coordinate_method"]),
        output_method="fused",
        pore_count_method="skip",
        defer_closure_status=True,
        status_transfer_method="packed",
        fuse_state_init=True,
        reuse_status_buffer=True,
    )


def main() -> None:
    import cupy as cp

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    loaded_cases: dict[int, dict[str, Any]] = {}
    input_hashes: set[tuple[str, str]] = set()

    for count in COUNTS:
        record_path = AUDIT_DIR / f"ptv{count}_ownership_backend_audit_gpu6.json"
        record = load_json(record_path)
        input_record = record["input"]
        mask_path = Path(input_record["mask_npz"])
        window_path = Path(input_record["particle_window"])
        mask_np = load_mask(mask_path)
        seed_flat, _ = load_particle_window_seed_flat(
            mask_np,
            window_path,
            frame_selection="",
            particle_id_selection=str(input_record["particle_ids"]),
        )
        seeds_np = np.column_stack(np.unravel_index(seed_flat, mask_np.shape)).astype(
            np.int32, copy=False
        )
        mask_cp = cp.asarray(mask_np, dtype=cp.uint8)
        seeds_cp = cp.asarray(seeds_np, dtype=cp.int32)
        line_bits = backend.prepare_mask_line_bits64(mask_cp)
        reference = backend.exact_frontier_dijkstra_gpu_6(
            mask_cp,
            seeds_cp,
            block_size=256,
            validate=True,
        )
        loaded_cases[count] = {
            "input": input_record,
            "mask_cp": mask_cp,
            "seeds_cp": seeds_cp,
            "line_bits": line_bits,
            "reference": reference,
        }
        input_hashes.add(
            (
                str(input_record["mask_sha256"]),
                str(input_record["particle_window_sha256"]),
            )
        )

    if len(input_hashes) != 1:
        raise RuntimeError("launch sweep did not use one mask and trajectory window")

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    configuration_checks: dict[str, dict[str, Any]] = {}

    for stage_index, (stage, candidate_overrides) in enumerate(STAGES.items()):
        stage_protocol = STAGE_PROTOCOL[stage]
        configs = {
            str(overrides["candidate"]): merged_config(overrides)
            for overrides in candidate_overrides
        }
        for count in COUNTS:
            case = loaded_cases[count]
            calls: dict[str, Callable[[], tuple[Any, ...]]] = {}
            for candidate, config in configs.items():

                def call(
                    config: dict[str, Any] = config,
                    case: dict[str, Any] = case,
                ) -> tuple[Any, ...]:
                    return candidate_call(
                        case["mask_cp"],
                        case["seeds_cp"],
                        case["line_bits"],
                        config,
                    )

                calls[candidate] = call

            for sweep_round in range(1, int(stage_protocol["rounds"]) + 1):
                samples = timed_stage(
                    calls,
                    cp=cp,
                    seed=20260717
                    + 100_000 * stage_index
                    + 1_000 * count
                    + sweep_round,
                    warmups=int(stage_protocol["warmups"]),
                    repeats=int(stage_protocol["repeats"]),
                )
                for candidate, values in samples.items():
                    config = configs[candidate]
                    summary_row = {
                        "stage": stage,
                        "candidate": candidate,
                        "sweep_round": sweep_round,
                        "repeat_count": int(stage_protocol["repeats"]),
                        "particle_count": count,
                        "site_count": int(case["input"]["particle_site_count"]),
                        "block_size": int(config["block_size"]),
                        "closure_block_size": int(config["closure_block_size"]),
                        "cooperative_blocks_per_sm": int(
                            config["cooperative_blocks_per_sm"]
                        ),
                        "closure_coordinate_method": str(
                            config["closure_coordinate_method"]
                        ),
                        "lower_bound_method": str(config["lower_bound_method"]),
                        "lower_warp_block_size": int(
                            config["lower_warp_block_size"]
                        ),
                        "lower_cooperative_blocks_per_sm": int(
                            config["lower_cooperative_blocks_per_sm"]
                        ),
                        "certificate_method": str(config["certificate_method"]),
                        "median_ms": float(statistics.median(values)),
                        "p05_ms": percentile(values, 5),
                        "p95_ms": percentile(values, 95),
                    }
                    rows.append(summary_row)
                    raw_rows.append({**summary_row, "samples_ms": values})

        for candidate, config in configs.items():
            label_mismatch = 0
            distance_mismatch = 0
            observed_backends: set[str] = set()
            observed_closure_launch: dict[str, dict[str, Any]] = {}
            for count in COUNTS:
                case = loaded_cases[count]
                output = backend.certified_l1_roi_frontier_gpu_6(
                    case["mask_cp"],
                    case["seeds_cp"],
                    block_size=int(config["block_size"]),
                    closure_block_size=int(config["closure_block_size"]),
                    validate=True,
                    return_stats=True,
                    lower_bound_method=str(config["lower_bound_method"]),
                    lower_warp_block_size=int(config["lower_warp_block_size"]),
                    lower_cooperative_blocks_per_sm=int(
                        config["lower_cooperative_blocks_per_sm"]
                    ),
                    certificate_method=str(config["certificate_method"]),
                    certificate_line_bits=case["line_bits"],
                    closure_method="cooperative",
                    cooperative_blocks_per_sm=int(
                        config["cooperative_blocks_per_sm"]
                    ),
                    closure_kernel_method="roi_single_writer",
                    closure_coordinate_method=str(
                        config["closure_coordinate_method"]
                    ),
                    output_method="fused",
                    pore_count_method="skip",
                    defer_closure_status=False,
                    status_transfer_method="packed",
                    fuse_state_init=True,
                    reuse_status_buffer=True,
                )
                label_bad, distance_bad = compare(
                    case["reference"], output, case["mask_cp"], cp
                )
                label_mismatch += label_bad
                distance_mismatch += distance_bad
                observed_backends.add(str(output[2]["lower_bound_backend"]))
                observed_closure_launch[f"ptv{count}"] = dict(
                    output[2]["closure_launch_status"]
                )
                if int(output[2]["fixed_point_residuals"]) != 0:
                    raise RuntimeError(
                        f"{stage}/{candidate}/ptv{count} has fixed-point residuals"
                    )
            configuration_checks[f"{stage}/{candidate}"] = {
                "label_mismatch_voxels": label_mismatch,
                "distance_mismatch_voxels": distance_mismatch,
                "pointwise_exact": label_mismatch == 0 and distance_mismatch == 0,
                "observed_lower_bound_backends": sorted(observed_backends),
                "observed_closure_launch": observed_closure_launch,
            }

    if not all(
        check["pointwise_exact"] for check in configuration_checks.values()
    ):
        raise RuntimeError("at least one ROI-JFA launch configuration was not exact")

    selections: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage]
        ranking = []
        for candidate in sorted({str(row["candidate"]) for row in stage_rows}):
            values = [
                float(row["median_ms"])
                for row in stage_rows
                if row["candidate"] == candidate
            ]
            ranking.append(
                {
                    "candidate": candidate,
                    "geometric_mean_case_round_median_ms": geometric_mean(values),
                    "worst_case_round_median_ms": max(values),
                    "case_round_medians_ms": values,
                }
            )
        ranking.sort(
            key=lambda item: (
                item["geometric_mean_case_round_median_ms"],
                item["worst_case_round_median_ms"],
            )
        )
        best_score = float(ranking[0]["geometric_mean_case_round_median_ms"])
        production_name = PRODUCTION_CANDIDATE[stage]
        production_record = next(
            item for item in ranking if item["candidate"] == production_name
        )
        production_score = float(
            production_record["geometric_mean_case_round_median_ms"]
        )
        robust_set = [
            item["candidate"]
            for item in ranking
            if float(item["geometric_mean_case_round_median_ms"])
            <= ROBUST_SET_TOLERANCE * best_score
        ]
        selections[stage] = {
            "best_observed": ranking[0],
            "production_candidate": production_name,
            "production_rank": next(
                index + 1
                for index, item in enumerate(ranking)
                if item["candidate"] == production_name
            ),
            "production_over_best": production_score / best_score,
            "robust_set_tolerance": ROBUST_SET_TOLERANCE,
            "production_in_robust_set": production_name in robust_set,
            "robust_set": robust_set,
            "ranking": ranking,
        }

    all_production_choices_robust = all(
        record["production_in_robust_set"] for record in selections.values()
    )
    csv_path = AUDIT_DIR / "ptv_ownership_launch_configuration_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    device_id = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")

    report = {
        "schema_version": "ptv-ownership-roi-kernel-sweep-v2",
        "status": (
            "PASS_POINTWISE_EXACT_PRODUCTION_IN_SCREENED_ROBUST_SET"
            if all_production_choices_robust
            else "CHECK_POINTWISE_EXACT_PRODUCTION_OUTSIDE_SCREENED_ROBUST_SET"
        ),
        "scope": (
            "ROI-JFA-only finite candidate screening on the five audited 64^3 "
            "Berea trajectory-density workloads; no global hardware-independent "
            "optimality claim"
        ),
        "selection_metric": (
            "geometric mean of per-case per-round medians under each stage's "
            "declared interleaved timing protocol"
        ),
        "stage_protocol": STAGE_PROTOCOL,
        "particle_counts": list(COUNTS),
        "same_mask_and_trajectory_window": True,
        "correctness_reference": "exact six-neighbour GPU propagation",
        "all_configurations_pointwise_exact": True,
        "exact_gpu_propagation_launch_configuration_fixed": True,
        "production_configuration": PRODUCTION_CONFIG,
        "production_candidate_by_stage": PRODUCTION_CANDIDATE,
        "all_production_choices_in_screened_3pct_robust_set": (
            all_production_choices_robust
        ),
        "selection": selections,
        "configuration_checks": configuration_checks,
        "raw_rows": raw_rows,
        "device": {
            "id": device_id,
            "name": str(device_name),
            "compute_capability": [
                int(properties["major"]),
                int(properties["minor"]),
            ],
            "multiprocessor_count": int(properties["multiProcessorCount"]),
        },
        "code_sha256": {
            "ptv_ownership_gpu6.py": sha256(CODE_DIR / "ptv_ownership_gpu6.py"),
            "sweep_launch_configurations.py": sha256(Path(__file__)),
        },
        "csv": str(csv_path.relative_to(ROOT)),
    }
    report_path = AUDIT_DIR / "ptv_ownership_launch_configuration_sweep.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": report["device"],
                "production_configuration": PRODUCTION_CONFIG,
                "selection": {
                    stage: {
                        "best_observed": record["best_observed"]["candidate"],
                        "production_candidate": record["production_candidate"],
                        "production_rank": record["production_rank"],
                        "production_over_best": record["production_over_best"],
                        "production_in_robust_set": record[
                            "production_in_robust_set"
                        ],
                    }
                    for stage, record in selections.items()
                },
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not all_production_choices_robust:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
