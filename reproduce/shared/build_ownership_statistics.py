from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "table_02" / "records" / "."
ROUND_DIR = AUDIT_DIR / "timing_repeats"
FIGURE_SOURCE = (
    ROOT
    / "../outputs"
    / "figure_source_data"
    / "table_02_timing_source.csv"
)
GENERATED_TEX = ROOT / "../outputs" / "generated" / "ownership_statistics.tex"
CANONICAL_JSON = AUDIT_DIR / "ptv_ownership_statistics_canonical.json"
COMBINED_CSV = AUDIT_DIR / "ptv_ownership_backend_audit_combined.csv"
COMBINED_JSON = AUDIT_DIR / "ptv_ownership_backend_audit_combined.json"
STABILITY_ROW_CSV = AUDIT_DIR / "ptv_ownership_timing_stability_4runs.csv"
STABILITY_SUMMARY_CSV = (
    AUDIT_DIR / "ptv_ownership_timing_stability_4runs_summary.csv"
)
STABILITY_JSON = AUDIT_DIR / "ptv_ownership_timing_stability_4runs.json"
CPU_REFERENCE_JSON = AUDIT_DIR / "ptv_ownership_gpu6_cpu_reference_verification.json"
LAUNCH_SWEEP_JSON = AUDIT_DIR / "ptv_ownership_launch_configuration_sweep.json"

COUNTS = (10, 50, 100, 200, 500)
RERUN_ROUNDS = 3
PRIMARY_WARMUPS = 10
PRIMARY_REPEATS = 101
BOOTSTRAP_SAMPLES = 20_000
PRIMARY_BOOTSTRAP_SEED = 20260717

EXPECTED_PRODUCTION_CONFIG = {
    "block_size": 256,
    "closure_block_size": 64,
    "cooperative_blocks_per_sm": 8,
    "closure_coordinate_method": "power2_warp_vote",
    "lower_bound_method": "jump_cooperative_warp64",
    "lower_warp_block_size": 256,
    "lower_cooperative_blocks_per_sm": 8,
    "certificate_method": "path_bitset_compact64",
}


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _classification(low: float, high: float) -> str:
    if low > 1.0:
        return "faster"
    if high < 1.0:
        return "slower"
    return "indistinguishable"


def _bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap input must be a nonempty one-dimensional array")
    rng = np.random.default_rng(seed)
    medians = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        stop = min(start + 500, samples)
        draw = rng.integers(0, values.size, size=(stop - start, values.size))
        medians[start:stop] = np.median(values[draw], axis=1)
    return [float(value) for value in np.quantile(medians, (0.025, 0.975))]


def _paired_statistics(
    reference_times: np.ndarray,
    candidate_times: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    if reference_times.shape != candidate_times.shape:
        raise ValueError("paired timing vectors must have the same shape")
    if reference_times.ndim != 1 or reference_times.size == 0:
        raise ValueError("paired timing vectors must be nonempty and one-dimensional")
    ratios = reference_times / candidate_times
    interval = _bootstrap_interval(ratios, seed=seed)
    return {
        "definition": "per-repeat reference wall time divided by ROI-JFA wall time",
        "paired_median": float(np.median(ratios)),
        "bootstrap_95_percent_interval": interval,
        "candidate_faster_fraction": float(
            np.mean(reference_times > candidate_times)
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": seed,
    }


def _require_close(actual: float, expected: float, message: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-14):
        raise RuntimeError(f"{message}: {actual!r} != {expected!r}")


def _validate_stored_pair(
    stored: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    source: Path,
    label: str,
) -> None:
    _require_close(
        float(stored["paired_median"]),
        float(recomputed["paired_median"]),
        f"{source}: stored {label} paired median drifted",
    )
    for index, endpoint in enumerate(("low", "high")):
        _require_close(
            float(stored["bootstrap_95_percent_interval"][index]),
            float(recomputed["bootstrap_95_percent_interval"][index]),
            f"{source}: stored {label} interval {endpoint} drifted",
        )
    _require_close(
        float(stored["candidate_faster_fraction"]),
        float(recomputed["candidate_faster_fraction"]),
        f"{source}: stored {label} faster fraction drifted",
    )


def _source_paths() -> list[tuple[str, int, Path]]:
    paths: list[tuple[str, int, Path]] = []
    for count in COUNTS:
        paths.append(
            (
                "paper_record",
                count,
                AUDIT_DIR / f"ptv{count}_ownership_backend_audit_gpu6.json",
            )
        )
    for round_index in range(1, RERUN_ROUNDS + 1):
        for count in COUNTS:
            paths.append(
                (
                    f"repeat_{round_index}",
                    count,
                    ROUND_DIR
                    / f"repeat{round_index}"
                    / f"ptv{count}_repeat{round_index}.json",
                )
            )
    return paths


def _times(payload: dict[str, Any], name: str, path: Path) -> np.ndarray:
    values = np.asarray(payload["timing"][name]["times_s"], dtype=np.float64)
    if values.shape != (PRIMARY_REPEATS,):
        raise RuntimeError(
            f"{path}: {name} has {values.size} repeats, expected {PRIMARY_REPEATS}"
        )
    stored_median = float(payload["timing"][name]["median_s"])
    _require_close(
        stored_median,
        float(np.median(values)),
        f"{path}: {name} median drifted",
    )
    return values


def _extract_record(
    round_name: str,
    particle_count: int,
    path: Path,
) -> dict[str, Any]:
    payload = _load_json(path)
    input_record = payload["input"]
    protocol = payload["protocol"]
    comparison = payload["comparison"]
    structure = payload["candidate_structure"]["roi_jfa"]

    if int(input_record["particle_count"]) != particle_count:
        raise RuntimeError(f"{path}: particle-count metadata does not match its prefix")
    if int(input_record["particle_site_count"]) <= 0:
        raise RuntimeError(f"{path}: no trajectory-derived sites")
    if int(protocol["graph_connectivity"]) != 6 or bool(protocol["diagonal_moves"]):
        raise RuntimeError(f"{path}: ownership graph is not strictly six-neighbour")
    if bool(protocol["regular_site_generation"]):
        raise RuntimeError(f"{path}: regular sites are not admissible")
    if int(protocol["auxiliary_site_count"]) != 0:
        raise RuntimeError(f"{path}: auxiliary sites are not admissible")
    if protocol["site_source"] != "flow_or_measurement_derived_particle_trajectory":
        raise RuntimeError(f"{path}: site source is not trajectory derived")
    if int(protocol["warmup_calls"]) != PRIMARY_WARMUPS:
        raise RuntimeError(f"{path}: warm-up count drifted")
    if int(protocol["timed_repeats"]) != PRIMARY_REPEATS:
        raise RuntimeError(f"{path}: repeat count drifted")
    if protocol["cuda_launch_configuration"] != {
        "roi_jfa_voxel_block_size": 256,
        "roi_jfa_lower_bound_warp_block_size": 256,
        "roi_jfa_closure_block_size": 64,
        "roi_jfa_closure_requested_blocks_per_sm": 8,
        "roi_jfa_lower_bound_requested_blocks_per_sm": 8,
        "selection_rule": "one fixed configuration chosen across all five particle prefixes",
    }:
        raise RuntimeError(f"{path}: production launch configuration drifted")

    result = comparison["roi_jfa_gpu_6"]
    if (
        not bool(result["exact_label_and_distance_match"])
        or int(result["label_mismatch_voxels"]) != 0
        or int(result["distance_mismatch_voxels"]) != 0
    ):
        raise RuntimeError(f"{path}: ROI-JFA is not pointwise exact")
    if int(structure["fixed_point_residuals"]) != 0:
        raise RuntimeError(f"{path}: ROI-JFA has fixed-point residuals")
    if structure["lower_bound_backend"] != "jump_cooperative_warp64_init":
        raise RuntimeError(f"{path}: warp64 lower-bound kernel was not executed")
    if structure["certificate_method"] != "path_bitset_compact64":
        raise RuntimeError(f"{path}: cached bitset certificate was not executed")
    if structure["closure_kernel_method"] != "roi_single_writer":
        raise RuntimeError(f"{path}: single-writer ROI closure was not executed")
    if structure["closure_coordinate_method"] != "power2_warp_vote":
        raise RuntimeError(f"{path}: production warp-vote closure was not executed")
    if not bool(structure["certificate_bits_cached"]):
        raise RuntimeError(f"{path}: mask line-bit cache was not reused")
    if int(structure["pore_voxels"]) != int(input_record["pore_voxel_count"]):
        raise RuntimeError(f"{path}: pore count drifted")

    exact = _times(payload, "exact_gpu_propagation_6", path)
    roi_jfa = _times(payload, "roi_jfa_gpu_6", path)
    cache_build = _times(payload, "mask_line_bit_cache_build_gpu_6", path)

    speedup_pair = _paired_statistics(
        exact, roi_jfa, seed=PRIMARY_BOOTSTRAP_SEED
    )
    _validate_stored_pair(
        payload["derived"]["paired_speedup"],
        speedup_pair,
        source=path,
        label="ROI-JFA",
    )

    pore_voxels = int(input_record["pore_voxel_count"])
    roi_voxels = int(structure["roi_voxels"])
    certified_voxels = int(structure["certified_voxels"])
    if certified_voxels + roi_voxels != pore_voxels:
        raise RuntimeError(f"{path}: certified and ROI voxel counts do not close")

    speedup_low, speedup_high = speedup_pair["bootstrap_95_percent_interval"]
    return {
        "round": round_name,
        "particle_count": particle_count,
        "particle_rows_selected": int(input_record["particle_rows_selected"]),
        "site_count": int(input_record["particle_site_count"]),
        "pore_voxels": pore_voxels,
        "pore_voxels_per_site": float(
            pore_voxels / int(input_record["particle_site_count"])
        ),
        "certified_voxels": certified_voxels,
        "roi_voxels": roi_voxels,
        "certified_percent": 100.0 * certified_voxels / pore_voxels,
        "roi_percent": 100.0 * roi_voxels / pore_voxels,
        "closure_iterations": int(structure["closure_iterations"]),
        "fixed_point_residuals": int(structure["fixed_point_residuals"]),
        "label_mismatch_voxels": int(
            comparison["roi_jfa_gpu_6"]["label_mismatch_voxels"]
        ),
        "distance_mismatch_voxels": int(
            comparison["roi_jfa_gpu_6"]["distance_mismatch_voxels"]
        ),
        "exact_gpu_propagation_median_ms": 1000.0 * float(np.median(exact)),
        "roi_jfa_median_ms": 1000.0 * float(np.median(roi_jfa)),
        "mask_line_bit_cache_build_median_ms": 1000.0
        * float(np.median(cache_build)),
        "paired_speedup_median": float(speedup_pair["paired_median"]),
        "paired_speedup_ci_low": float(speedup_low),
        "paired_speedup_ci_high": float(speedup_high),
        "timing_class": _classification(speedup_low, speedup_high),
        "mask_sha256": str(input_record["mask_sha256"]),
        "particle_window_sha256": str(input_record["particle_window_sha256"]),
        "source_json": _relative(path),
        "code_sha256": dict(payload["code_sha256"]),
        "_exact_times": exact,
        "_roi_jfa_times": roi_jfa,
    }


def _public_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_") and key not in {"round", "code_sha256"}
    }


def _stability_run_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": record["round"],
        "particle_count": record["particle_count"],
        "site_count": record["site_count"],
        "exact_gpu_propagation_median_ms": record[
            "exact_gpu_propagation_median_ms"
        ],
        "roi_jfa_median_ms": record["roi_jfa_median_ms"],
        "paired_speedup_median": record["paired_speedup_median"],
        "paired_speedup_ci_low": record["paired_speedup_ci_low"],
        "paired_speedup_ci_high": record["paired_speedup_ci_high"],
        "classification": record["timing_class"],
        "certified_percent": record["certified_percent"],
        "roi_percent": record["roi_percent"],
        "label_mismatch_voxels": record["label_mismatch_voxels"],
        "distance_mismatch_voxels": record["distance_mismatch_voxels"],
        "fixed_point_residuals": record["fixed_point_residuals"],
        "source_json": record["source_json"],
    }


def _stability_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in COUNTS:
        selected = [row for row in records if row["particle_count"] == count]
        if len(selected) != RERUN_ROUNDS + 1:
            raise RuntimeError(f"particle count {count} does not have four runs")
        pooled_reference = np.concatenate([row["_exact_times"] for row in selected])
        pooled_candidate = np.concatenate([row["_roi_jfa_times"] for row in selected])
        pooled_pair = _paired_statistics(
            pooled_reference,
            pooled_candidate,
            seed=PRIMARY_BOOTSTRAP_SEED + count,
        )
        pooled_low, pooled_high = pooled_pair["bootstrap_95_percent_interval"]
        classifications = Counter(str(row["timing_class"]) for row in selected)
        speedup_ratios = np.asarray(
            [row["paired_speedup_median"] for row in selected],
            dtype=np.float64,
        )
        exact_medians = np.asarray(
            [row["exact_gpu_propagation_median_ms"] for row in selected],
            dtype=np.float64,
        )
        roi_jfa_medians = np.asarray(
            [row["roi_jfa_median_ms"] for row in selected],
            dtype=np.float64,
        )
        all_pointwise_exact = all(
            row["label_mismatch_voxels"] == 0
            and row["distance_mismatch_voxels"] == 0
            and row["fixed_point_residuals"] == 0
            for row in selected
        )
        rows.append(
            {
                "particle_count": count,
                "site_count": selected[0]["site_count"],
                "run_count": len(selected),
                "paired_sample_count": int(pooled_reference.size),
                "all_pointwise_exact": all_pointwise_exact,
                "exact_gpu_propagation_median_ms_across_runs": float(
                    np.median(exact_medians)
                ),
                "exact_gpu_propagation_median_ms_min": float(
                    np.min(exact_medians)
                ),
                "exact_gpu_propagation_median_ms_max": float(
                    np.max(exact_medians)
                ),
                "roi_jfa_median_ms_across_runs": float(
                    np.median(roi_jfa_medians)
                ),
                "roi_jfa_median_ms_min": float(np.min(roi_jfa_medians)),
                "roi_jfa_median_ms_max": float(np.max(roi_jfa_medians)),
                "round_ratio_median": float(np.median(speedup_ratios)),
                "round_ratio_min": float(np.min(speedup_ratios)),
                "round_ratio_max": float(np.max(speedup_ratios)),
                "pooled_ratio_median": float(pooled_pair["paired_median"]),
                "pooled_ratio_ci_low": float(pooled_low),
                "pooled_ratio_ci_high": float(pooled_high),
                "pooled_classification": _classification(
                    float(pooled_low), float(pooled_high)
                ),
                "round_classifications": dict(sorted(classifications.items())),
            }
        )
    return rows


def _validate_auxiliary_evidence(
    stable_code_hashes: dict[str, str],
) -> dict[str, Any]:
    cpu = _load_json(CPU_REFERENCE_JSON)
    if cpu["code_sha256"]["ptv_ownership_gpu6.py"] != stable_code_hashes[
        "ptv_ownership_gpu6.py"
    ]:
        raise RuntimeError("CPU reference used a different ownership implementation")
    if cpu["code_sha256"]["hybrid_site_sources.py"] != stable_code_hashes[
        "hybrid_site_sources.py"
    ]:
        raise RuntimeError("CPU reference used a different site loader")
    result = cpu["result"]
    if not bool(result["passed"]):
        raise RuntimeError("CPU ownership reference did not pass")
    if int(cpu["protocol"]["random_cases"]) != 100:
        raise RuntimeError("CPU ownership reference coverage drifted")
    if len(result["ptv_cases"]) != len(COUNTS):
        raise RuntimeError("CPU ownership reference does not cover all PTV rows")
    for key, value in result.items():
        if key.endswith("_mismatches") and int(value) != 0:
            raise RuntimeError(f"CPU ownership reference reports {key}={value}")
    for case in result["ptv_cases"]:
        for key, value in case.items():
            if (
                key.endswith("_mismatches")
                or key.endswith("_fixed_point_residuals")
            ) and int(value) != 0:
                raise RuntimeError(f"CPU PTV reference reports {key}={value}")

    sweep = _load_json(LAUNCH_SWEEP_JSON)
    if sweep["code_sha256"]["ptv_ownership_gpu6.py"] != stable_code_hashes[
        "ptv_ownership_gpu6.py"
    ]:
        raise RuntimeError("launch sweep used a different ownership implementation")
    if sweep["status"] != (
        "PASS_POINTWISE_EXACT_PRODUCTION_IN_SCREENED_ROBUST_SET"
    ):
        raise RuntimeError("ROI-JFA launch sweep did not pass")
    if not bool(sweep["all_configurations_pointwise_exact"]):
        raise RuntimeError("at least one launch-sweep configuration was not exact")
    if not bool(sweep["all_production_choices_in_screened_3pct_robust_set"]):
        raise RuntimeError("production launch is outside a screened robust set")
    if not bool(sweep["exact_gpu_propagation_launch_configuration_fixed"]):
        raise RuntimeError("exact GPU propagation configuration was not fixed")
    if sweep["production_configuration"] != EXPECTED_PRODUCTION_CONFIG:
        raise RuntimeError("launch sweep production configuration drifted")

    stage_protocol = sweep["stage_protocol"]
    screening = stage_protocol["voxel_block_size"]
    confirmation = stage_protocol["voxel_block_confirmation"]
    configuration_count = sum(
        len(record["ranking"]) for record in sweep["selection"].values()
    )
    return {
        "cpu_reference": {
            "status": "passed",
            "random_case_count": int(cpu["protocol"]["random_cases"]),
            "ptv_case_count": len(result["ptv_cases"]),
            "source_json": _relative(CPU_REFERENCE_JSON),
            "source_json_sha256": _sha256(CPU_REFERENCE_JSON),
        },
        "launch_sweep": {
            "status": sweep["status"],
            "scope": sweep["scope"],
            "stage_count": len(sweep["selection"]),
            "configuration_count": configuration_count,
            "screening_round_count": int(screening["rounds"]),
            "screening_repeats_per_case_round": int(screening["repeats"]),
            "confirmation_round_count": int(confirmation["rounds"]),
            "confirmation_repeats_per_case_round": int(
                confirmation["repeats"]
            ),
            "all_configurations_pointwise_exact": True,
            "exact_gpu_propagation_configuration_fixed": True,
            "production_configuration": sweep["production_configuration"],
            "production_in_screened_3pct_robust_set": True,
            "best_observed_by_stage": {
                stage: record["best_observed"]["candidate"]
                for stage, record in sweep["selection"].items()
            },
            "source_json": _relative(LAUNCH_SWEEP_JSON),
            "source_json_sha256": _sha256(LAUNCH_SWEEP_JSON),
        },
    }


def build_canonical_payload() -> dict[str, Any]:
    records = [
        _extract_record(round_name, count, path)
        for round_name, count, path in _source_paths()
    ]
    primary = [row for row in records if row["round"] == "paper_record"]
    if len(primary) != len(COUNTS):
        raise RuntimeError("primary ownership record count drifted")

    mask_hashes = {row["mask_sha256"] for row in records}
    window_hashes = {row["particle_window_sha256"] for row in records}
    if len(mask_hashes) != 1 or len(window_hashes) != 1:
        raise RuntimeError("ownership records do not share one mask and trajectory window")

    code_names = (
        "ptv_ownership_gpu6.py",
        "audit_ptv_ownership_backends.py",
        "hybrid_site_sources.py",
    )
    stable_code_hashes: dict[str, str] = {}
    for name in code_names:
        values = {row["code_sha256"][name] for row in records}
        if len(values) != 1:
            raise RuntimeError(f"ownership records do not share one {name} hash")
        stable_code_hashes[name] = values.pop()

    current_paths = {
        "ptv_ownership_gpu6.py": ROOT
        / "../gpu"
        / "ownership"
        / "code"
        / "ptv_ownership_gpu6.py",
        "audit_ptv_ownership_backends.py": ROOT
        / "../gpu"
        / "ownership"
        / "code"
        / "time_roi_vs_bfs.py",
        "hybrid_site_sources.py": ROOT
        / "../gpu"
        / "code"
        / "hybrid_site_sources.py",
    }
    for name, path in current_paths.items():
        if _sha256(path) != stable_code_hashes[name]:
            raise RuntimeError(f"shipped {name} differs from the timed code")

    auxiliary = _validate_auxiliary_evidence(stable_code_hashes)
    summary = _stability_summary(records)
    primary_rows = [_public_row(row) for row in primary]
    stability_run_rows = [_stability_run_row(row) for row in records]

    pore_counts = {int(row["pore_voxels"]) for row in primary}
    if len(pore_counts) != 1:
        raise RuntimeError("primary ownership rows do not share one pore count")
    speedup_ratios = [float(row["paired_speedup_median"]) for row in primary]
    all_primary_exact = all(
        row["label_mismatch_voxels"] == 0
        and row["distance_mismatch_voxels"] == 0
        and row["fixed_point_residuals"] == 0
        for row in primary
    )
    if not all_primary_exact:
        raise RuntimeError("primary ownership rows are not pointwise exact")

    generated_projections = [
        COMBINED_CSV,
        COMBINED_JSON,
        STABILITY_ROW_CSV,
        STABILITY_SUMMARY_CSV,
        STABILITY_JSON,
        FIGURE_SOURCE,
        GENERATED_TEX,
    ]
    return {
        "schema_version": 4,
        "status": "PASS_CANONICAL_ROI_JFA_STATISTICS",
        "authority": (
            "All paper-facing ROI-JFA ownership numbers are projected from this "
            "record; raw audit JSON files are the only numerical inputs."
        ),
        "generated_by": _relative(Path(__file__)),
        "statistics_builder_sha256": _sha256(Path(__file__)),
        "input_manifest": {
            "raw_records": [
                {
                    "round": round_name,
                    "particle_count": count,
                    "path": _relative(path),
                    "sha256": _sha256(path),
                }
                for round_name, count, path in _source_paths()
            ],
            "same_mask": True,
            "same_trajectory_window": True,
            "mask_sha256": next(iter(mask_hashes)),
            "particle_window_sha256": next(iter(window_hashes)),
            "stable_code_sha256": stable_code_hashes,
        },
        "statistical_definition": {
            "timing_scope": (
                "synchronized GPU-resident ownership call with warm CUDA JIT and "
                "memory pool"
            ),
            "reference": "exact six-neighbour GPU propagation",
            "point_estimator": "median of synchronized host wall times",
            "paired_ratio": (
                "per-repeat exact GPU propagation time divided by ROI-JFA time"
            ),
            "interval": "percentile bootstrap interval for the paired median",
            "confidence_level": 0.95,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": PRIMARY_BOOTSTRAP_SEED,
            "pooled_bootstrap_seed_rule": (
                "PRIMARY_BOOTSTRAP_SEED plus particle count"
            ),
            "mask_line_bit_cache_boundary": (
                "mask-only cache construction is timed separately and excluded "
                "from repeated ownership calls"
            ),
            "end_to_end_particle_pipeline": False,
        },
        "primary_rows": primary_rows,
        "stability_run_rows": stability_run_rows,
        "stability_summary_rows": summary,
        "paper_values": {
            "prefix_count": len(primary),
            "particle_counts": [int(row["particle_count"]) for row in primary],
            "particle_rows_selected": [
                int(row["particle_rows_selected"]) for row in primary
            ],
            "site_counts": [int(row["site_count"]) for row in primary],
            "pore_voxel_count": pore_counts.pop(),
            "certified_voxel_counts": [
                int(row["certified_voxels"]) for row in primary
            ],
            "roi_voxel_counts": [int(row["roi_voxels"]) for row in primary],
            "certified_percent_range": [
                min(float(row["certified_percent"]) for row in primary),
                max(float(row["certified_percent"]) for row in primary),
            ],
            "roi_percent_range": [
                max(float(row["roi_percent"]) for row in primary),
                min(float(row["roi_percent"]) for row in primary),
            ],
            "speedup_range": [min(speedup_ratios), max(speedup_ratios)],
            "mask_line_bit_cache_build_median_ms_range": [
                min(
                    float(row["mask_line_bit_cache_build_median_ms"])
                    for row in primary
                ),
                max(
                    float(row["mask_line_bit_cache_build_median_ms"])
                    for row in primary
                ),
            ],
            "exact_case_count": sum(
                row["label_mismatch_voxels"] == 0
                and row["distance_mismatch_voxels"] == 0
                and row["fixed_point_residuals"] == 0
                for row in primary
            ),
            "faster_case_count": sum(
                row["timing_class"] == "faster" for row in primary
            ),
            "warmup_count": PRIMARY_WARMUPS,
            "repeat_count": PRIMARY_REPEATS,
            "rerun_round_count": RERUN_ROUNDS,
            "total_run_count": RERUN_ROUNDS + 1,
            "additional_verified_field_count": RERUN_ROUNDS * len(COUNTS),
            "pooled_sample_count_per_prefix": (RERUN_ROUNDS + 1)
            * PRIMARY_REPEATS,
        },
        "auxiliary_evidence": auxiliary,
        "generated_projections": [_relative(path) for path in generated_projections],
    }


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _figure_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in payload["primary_rows"]:
        rows.append(
            {
                "particle_count": source["particle_count"],
                "site_count": source["site_count"],
                "pore_voxels": source["pore_voxels"],
                "certified_voxels": source["certified_voxels"],
                "roi_voxels": source["roi_voxels"],
                "certified_percent": source["certified_percent"],
                "roi_percent": source["roi_percent"],
                "exact_gpu_propagation_median_ms": source[
                    "exact_gpu_propagation_median_ms"
                ],
                "roi_jfa_median_ms": source["roi_jfa_median_ms"],
                "paired_speedup": source["paired_speedup_median"],
                "paired_ci_low": source["paired_speedup_ci_low"],
                "paired_ci_high": source["paired_speedup_ci_high"],
            }
        )
    return rows


def _fmt_ratio(value: float, low: float, high: float) -> str:
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def _join_ints(values: list[int]) -> str:
    if len(values) == 1:
        return str(values[0])
    return ", ".join(str(value) for value in values[:-1]) + f", and {values[-1]}"


def _classification_sentence(
    rows: list[dict[str, Any]],
    *,
    pooled: bool,
) -> str:
    field = "pooled_classification" if pooled else "timing_class"
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(int(row["particle_count"]))
    prefix = "Across the pooled four-run evidence, " if pooled else ""
    if set(groups) == {"faster"}:
        subject = "all" if pooled else "All"
        count_words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        count = count_words.get(len(rows), str(len(rows)))
        return (
            f"{prefix}{subject} {count} audited particle prefixes are faster than "
            "exact six-neighbour GPU propagation."
        )
    fragments = []
    labels = {
        "faster": "faster",
        "slower": "slower",
        "indistinguishable": "not resolved from parity",
    }
    for classification in ("faster", "indistinguishable", "slower"):
        if classification in groups:
            counts = _join_ints(groups[classification])
            fragments.append(f"the {counts}-particle prefixes are {labels[classification]}")
    return prefix + ", while ".join(fragments) + "."


def _tex_text(payload: dict[str, Any]) -> str:
    primary = payload["primary_rows"]
    summary = payload["stability_summary_rows"]
    values = payload["paper_values"]
    auxiliary = payload["auxiliary_evidence"]
    densest = primary[-1]

    speedup_ratios = ", ".join(
        _fmt_ratio(
            float(row["paired_speedup_median"]),
            float(row["paired_speedup_ci_low"]),
            float(row["paired_speedup_ci_high"]),
        )
        for row in primary
    )
    pooled_ratios = ", ".join(
        _fmt_ratio(
            float(row["pooled_ratio_median"]),
            float(row["pooled_ratio_ci_low"]),
            float(row["pooled_ratio_ci_high"]),
        )
        for row in summary
    )
    table_rows = "\n".join(
        (
            f"{int(row['particle_count'])} & "
            f"{int(row['particle_rows_selected'])} & "
            f"{int(row['site_count'])} & "
            f"{float(row['certified_percent']):.2f} & "
            f"{float(row['roi_percent']):.2f} & "
            f"{int(row['label_mismatch_voxels'])} & "
            f"{int(row['distance_mismatch_voxels'])} & "
            f"{float(row['exact_gpu_propagation_median_ms']):.3f} & "
            f"{float(row['roi_jfa_median_ms']):.3f} & "
            f"{_fmt_ratio(float(row['paired_speedup_median']), float(row['paired_speedup_ci_low']), float(row['paired_speedup_ci_high']))} \\\\"
        )
        for row in primary
    )
    main_table_rows = "\n".join(
        (
            f"{int(row['particle_count'])} & "
            f"{int(row['site_count'])} & "
            f"{float(row['certified_percent']):.2f} & "
            f"{float(row['roi_percent']):.2f} & "
            f"{float(row['exact_gpu_propagation_median_ms']):.3f} & "
            f"{float(row['roi_jfa_median_ms']):.3f} & "
            f"{float(row['paired_speedup_median']):.2f} \\\\"
        )
        for row in primary
    )
    production = auxiliary["launch_sweep"]["production_configuration"]
    speedup_range = values["speedup_range"]
    cache_range = values["mask_line_bit_cache_build_median_ms_range"]
    certified_range = values["certified_percent_range"]
    roi_range = values["roi_percent_range"]

    lines = [
        "% Generated by reproduce/shared/build_ownership_statistics.py.",
        "% Source: reproduce/table_02/records/",
        "% ptv_ownership_statistics_canonical.json. Do not edit manually.",
        rf"\providecommand{{\OwnershipPrefixCount}}{{{values['prefix_count']}}}",
        rf"\providecommand{{\OwnershipParticleList}}{{{_join_ints(values['particle_counts'])}}}",
        rf"\providecommand{{\OwnershipSiteList}}{{{_join_ints(values['site_counts'])}}}",
        rf"\providecommand{{\OwnershipPoreVoxelCount}}{{{values['pore_voxel_count']}}}",
        rf"\providecommand{{\OwnershipCertifiedVoxelList}}{{{_join_ints(values['certified_voxel_counts'])}}}",
        rf"\providecommand{{\OwnershipROIVoxelList}}{{{_join_ints(values['roi_voxel_counts'])}}}",
        rf"\providecommand{{\OwnershipCertifiedRange}}{{{certified_range[0]:.2f}\%--{certified_range[1]:.2f}\%}}",
        rf"\providecommand{{\OwnershipUnresolvedRange}}{{{roi_range[0]:.2f}\% to {roi_range[1]:.2f}\%}}",
        rf"\providecommand{{\OwnershipSpeedupRange}}{{{speedup_range[0]:.2f}--{speedup_range[1]:.2f}}}",
        rf"\providecommand{{\OwnershipMaskCacheRange}}{{{cache_range[0]:.3f}--{cache_range[1]:.3f}}}",
        rf"\providecommand{{\OwnershipDensestRatio}}{{{float(densest['paired_speedup_median']):.2f}}}",
        (
            rf"\providecommand{{\OwnershipDensestInterval}}"
            rf"{{{float(densest['paired_speedup_ci_low']):.2f}--"
            rf"{float(densest['paired_speedup_ci_high']):.2f}}}"
        ),
        rf"\providecommand{{\OwnershipSpeedupSequence}}{{{speedup_ratios}}}",
        rf"\providecommand{{\OwnershipPooledRatioSequence}}{{{pooled_ratios}}}",
        (
            rf"\providecommand{{\OwnershipClassificationSentence}}"
            rf"{{{_classification_sentence(primary, pooled=False)}}}"
        ),
        (
            rf"\providecommand{{\OwnershipPooledClassificationSentence}}"
            rf"{{{_classification_sentence(summary, pooled=True)}}}"
        ),
        rf"\providecommand{{\OwnershipRepeatCount}}{{{values['repeat_count']}}}",
        rf"\providecommand{{\OwnershipWarmupCount}}{{{values['warmup_count']}}}",
        rf"\providecommand{{\OwnershipPooledSampleCount}}{{{values['pooled_sample_count_per_prefix']}}}",
        rf"\providecommand{{\OwnershipRerunRoundCount}}{{{values['rerun_round_count']}}}",
        rf"\providecommand{{\OwnershipTotalRunCount}}{{{values['total_run_count']}}}",
        rf"\providecommand{{\OwnershipAdditionalFieldCount}}{{{values['additional_verified_field_count']}}}",
        rf"\providecommand{{\OwnershipDensestSiteCount}}{{{values['site_counts'][-1]}}}",
        rf"\providecommand{{\OwnershipCPURandomCaseCount}}{{{auxiliary['cpu_reference']['random_case_count']}}}",
        rf"\providecommand{{\OwnershipLaunchStageCount}}{{{auxiliary['launch_sweep']['stage_count']}}}",
        rf"\providecommand{{\OwnershipLaunchConfigurationCount}}{{{auxiliary['launch_sweep']['configuration_count']}}}",
        rf"\providecommand{{\OwnershipLaunchSweepRoundCount}}{{{auxiliary['launch_sweep']['screening_round_count']}}}",
        rf"\providecommand{{\OwnershipLaunchSweepRepeatCount}}{{{auxiliary['launch_sweep']['screening_repeats_per_case_round']}}}",
        rf"\providecommand{{\OwnershipLaunchConfirmationRoundCount}}{{{auxiliary['launch_sweep']['confirmation_round_count']}}}",
        rf"\providecommand{{\OwnershipLaunchConfirmationRepeatCount}}{{{auxiliary['launch_sweep']['confirmation_repeats_per_case_round']}}}",
        rf"\providecommand{{\OwnershipLaunchCaseRoundMedianCount}}{{{values['prefix_count'] * auxiliary['launch_sweep']['confirmation_round_count']}}}",
        rf"\providecommand{{\OwnershipCandidateBlockSize}}{{{production['block_size']}}}",
        rf"\providecommand{{\OwnershipCandidateClosureBlockSize}}{{{production['closure_block_size']}}}",
        rf"\providecommand{{\OwnershipCandidateBlocksPerSM}}{{{production['cooperative_blocks_per_sm']}}}",
        rf"\providecommand{{\OwnershipCandidateLowerBlocksPerSM}}{{{production['lower_cooperative_blocks_per_sm']}}}",
        r"\providecommand{\OwnershipExactBlockSize}{256}",
        rf"\providecommand{{\OwnershipExactCaseCount}}{{{values['exact_case_count']}}}",
        rf"\providecommand{{\OwnershipFasterCaseCount}}{{{values['faster_case_count']}}}",
        r"\providecommand{\OwnershipMainTableRows}{%",
        main_table_rows,
        "}",
        r"\providecommand{\OwnershipSITableRows}{%",
        table_rows,
        "}",
        "",
    ]
    return "\n".join(lines)


def expected_artifacts() -> dict[Path, str]:
    payload = build_canonical_payload()
    primary_rows = payload["primary_rows"]
    stability_rows = payload["stability_run_rows"]
    summary_rows = payload["stability_summary_rows"]
    combined = {
        "status": "PASS_POINTWISE_EXACT_PRIMARY_ROI_JFA_AUDIT",
        "canonical_source": _relative(CANONICAL_JSON),
        "row_count": len(primary_rows),
        "same_mask": True,
        "paper_facing_scope": payload["statistical_definition"]["timing_scope"],
        "paper_facing_reference": payload["statistical_definition"]["reference"],
        "timing_class_counts": dict(
            Counter(row["timing_class"] for row in primary_rows)
        ),
        "figure_source_checked": _relative(FIGURE_SOURCE),
        "rows": primary_rows,
    }
    stability = {
        "status": "PASS_FOUR_RUN_ROI_JFA_STABILITY_AUDIT",
        "canonical_source": _relative(CANONICAL_JSON),
        "same_mask": True,
        "same_trajectory_window": True,
        "run_definition": (
            "one paper record plus three fresh 101-repeat processes per prefix"
        ),
        "all_pointwise_exact": all(
            row["all_pointwise_exact"] for row in summary_rows
        ),
        "stable_code_hashes": payload["input_manifest"]["stable_code_sha256"],
        "row_csv": _relative(STABILITY_ROW_CSV),
        "summary_csv": _relative(STABILITY_SUMMARY_CSV),
        "summary": summary_rows,
    }
    return {
        CANONICAL_JSON: _json_text(payload),
        COMBINED_CSV: _csv_text(primary_rows),
        COMBINED_JSON: _json_text(combined),
        STABILITY_ROW_CSV: _csv_text(stability_rows),
        STABILITY_SUMMARY_CSV: _csv_text(summary_rows),
        STABILITY_JSON: _json_text(stability),
        FIGURE_SOURCE: _csv_text(_figure_rows(payload)),
        GENERATED_TEX: _tex_text(payload),
    }


def write_artifacts() -> dict[str, Any]:
    artifacts = expected_artifacts()
    for path, text in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    return {
        "status": "WROTE_CANONICAL_ROI_JFA_STATISTICS",
        "artifact_count": len(artifacts),
        "artifacts": [_relative(path) for path in artifacts],
    }


def validate_artifacts() -> dict[str, Any]:
    artifacts = expected_artifacts()
    mismatches = []
    for path, expected in artifacts.items():
        if not path.is_file():
            mismatches.append(f"missing:{_relative(path)}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            actual = handle.read()
        if actual != expected:
            mismatches.append(f"drifted:{_relative(path)}")
    if mismatches:
        raise RuntimeError("canonical ownership projections drifted: " + ", ".join(mismatches))
    return {
        "status": "PASS_CANONICAL_ROI_JFA_PROJECTIONS",
        "artifact_count": len(artifacts),
        "artifacts": [_relative(path) for path in artifacts],
    }


def load_canonical_payload() -> dict[str, Any]:
    validate_artifacts()
    return _load_json(CANONICAL_JSON)


def main() -> None:
    result = write_artifacts()
    validation = validate_artifacts()
    print(
        json.dumps(
            {
                **result,
                "validation_status": validation["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
