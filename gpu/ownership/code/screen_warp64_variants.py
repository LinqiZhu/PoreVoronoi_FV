from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
WARP_BLOCKS = (32, 64, 128, 256, 512)
RESIDENCIES = (1, 2, 4, 8, 12, 16)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["mask"], dtype=bool)


def candidate_call(case: dict[str, Any], config: dict[str, Any]):
    common = {
        "mask": case["mask"],
        "seeds_zyx": case["seeds"],
        "block_size": int(config.get("voxel_block_size", 256)),
        "closure_block_size": int(config.get("closure_block_size", 64)),
        "validate": False,
        "return_float64": False,
        "return_stats": False,
        "closure_method": "cooperative",
        "cooperative_blocks_per_sm": int(
            config.get("closure_residency", 8)
        ),
        "closure_kernel_method": "roi_single_writer",
        "closure_coordinate_method": str(
            config.get("closure_coordinate_method", "power2_warp_vote")
        ),
        "output_method": "fused",
        "pore_count_method": "skip",
        "defer_closure_status": True,
        "status_transfer_method": "packed",
        "fuse_state_init": True,
        "reuse_status_buffer": True,
    }
    if config["kind"] == "selected":
        return backend.certified_l1_roi_frontier_gpu_6(
            **common,
            lower_bound_method="jump_cooperative_warp64",
            lower_warp_block_size=256,
            lower_cooperative_blocks_per_sm=8,
            certificate_method="path_bitset_compact64",
            certificate_line_bits=case["line_bits"],
            pore_voxel_count=int(case["pore_ids"].size),
        )
    certificate_backend = {
        "warp64_full": "path_bitset",
        "warp64_compact_generic": "path_bitset_compact",
        "warp64_compact": "path_bitset_compact64",
    }.get(str(config["kind"]), "path_bitset")
    return backend.certified_l1_roi_frontier_gpu_6(
        **common,
        lower_bound_method="jump_cooperative_warp64",
        lower_warp_block_size=int(config.get("block_size", 256)),
        lower_cooperative_blocks_per_sm=int(config.get("residency", 8)),
        certificate_method=certificate_backend,
        certificate_line_bits=case["line_bits"],
        pore_voxel_count=int(case["pore_ids"].size),
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def time_calls(
    calls: dict[str, Callable[[], tuple[Any, ...]]],
    *,
    cp: Any,
    seed: int,
    warmups: int,
    repeats: int,
) -> dict[str, list[float]]:
    names = list(calls)
    for _ in range(warmups):
        for call in calls.values():
            call()
    cp.cuda.Stream.null.synchronize()
    samples = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        for index in rng.permutation(len(names)):
            name = names[int(index)]
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            calls[name]()
            cp.cuda.Stream.null.synchronize()
            samples[name].append(1000.0 * (time.perf_counter() - start))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--finalists-only", action="store_true")
    parser.add_argument("--launch-screen", action="store_true")
    parser.add_argument("--launch-finalists", action="store_true")
    args = parser.parse_args()

    import cupy as cp

    cases: dict[int, dict[str, Any]] = {}
    for count in COUNTS:
        record = load_json(AUDIT_DIR / f"ptv{count}_ownership_backend_audit_gpu6.json")
        input_record = record["input"]
        mask_np = load_mask(Path(input_record["mask_npz"]))
        seed_flat, _ = load_particle_window_seed_flat(
            mask_np,
            Path(input_record["particle_window"]),
            frame_selection="",
            particle_id_selection=str(input_record["particle_ids"]),
        )
        seeds_np = np.column_stack(np.unravel_index(seed_flat, mask_np.shape)).astype(
            np.int32, copy=False
        )
        mask = cp.asarray(mask_np, dtype=cp.uint8)
        seeds = cp.asarray(seeds_np, dtype=cp.int32)
        cases[count] = {
            "mask": mask,
            "seeds": seeds,
            "line_bits": backend.prepare_mask_line_bits64(mask),
            "pore_ids": backend.prepare_mask_pore_ids(mask),
            "reference": backend.exact_frontier_dijkstra_gpu_6(
                mask, seeds, block_size=256, validate=True
            ),
            "site_count": int(seeds.shape[0]),
        }

    configs: dict[str, dict[str, Any]] = {"selected": {"kind": "selected"}}
    if args.launch_screen:
        configs = {
            "compact_reference": {"kind": "warp64_compact"},
            **{
                f"voxel_b{value}": {
                    "kind": "warp64_compact",
                    "voxel_block_size": value,
                }
                for value in (32, 64, 128, 256, 512)
            },
            **{
                f"closure_b{value}": {
                    "kind": "warp64_compact",
                    "closure_block_size": value,
                }
                for value in (32, 64, 128, 256, 512)
            },
            **{
                f"closure_r{value}": {
                    "kind": "warp64_compact",
                    "closure_residency": value,
                }
                for value in (1, 2, 4, 8, 12, 16)
            },
        }
    if args.launch_finalists:
        configs = {
            "launch_b256_r8": {"kind": "warp64_compact"},
            "launch_b512_r4": {
                "kind": "warp64_compact",
                "voxel_block_size": 512,
                "closure_residency": 4,
            },
            "launch_b256_r4": {
                "kind": "warp64_compact",
                "closure_residency": 4,
            },
            "launch_b512_r8": {
                "kind": "warp64_compact",
                "voxel_block_size": 512,
                "closure_residency": 8,
            },
            "launch_b256_r12": {
                "kind": "warp64_compact",
                "closure_residency": 12,
            },
        }
    candidate_pairs = (
        ()
        if args.finalists_only
        else tuple(
            (block_size, residency)
            for block_size in WARP_BLOCKS
            for residency in RESIDENCIES
        )
    )
    if not args.launch_screen and not args.launch_finalists:
        for block_size, residency in candidate_pairs:
                configs[f"warp_b{block_size}_r{residency}"] = {
                    "kind": "warp64",
                    "block_size": block_size,
                    "residency": residency,
                }
    if args.finalists_only and not args.launch_screen and not args.launch_finalists:
        configs = {
            "power2_compact_r4": {
                "kind": "warp64_compact_generic",
                "block_size": 256,
                "residency": 8,
                "closure_residency": 4,
                "closure_coordinate_method": "power2",
            },
            "warp_vote_compact_r8": {
                "kind": "warp64_compact_generic",
                "block_size": 256,
                "residency": 8,
                "closure_residency": 8,
                "closure_coordinate_method": "power2_warp_vote",
            },
            "production_compact64_r8": {
                "kind": "warp64_compact",
                "block_size": 256,
                "residency": 8,
                "closure_residency": 8,
                "closure_coordinate_method": "power2_warp_vote",
            },
        }

    checks: dict[str, dict[str, Any]] = {}
    pore = cases[10]["mask"] != 0
    for name, config in configs.items():
        output = candidate_call(cases[10], config)
        checks[name] = {
            "ptv10_label_mismatch_voxels": int(
                cp.count_nonzero(pore & (output[0] != cases[10]["reference"][0])).get()
            ),
            "ptv10_distance_mismatch_voxels": int(
                cp.count_nonzero(pore & (output[1] != cases[10]["reference"][1])).get()
            ),
        }
    if any(
        item["ptv10_label_mismatch_voxels"]
        or item["ptv10_distance_mismatch_voxels"]
        for item in checks.values()
    ):
        raise RuntimeError("at least one candidate failed the ptv10 pointwise check")

    rows: list[dict[str, Any]] = []
    for count, case in cases.items():
        calls = {
            name: (
                lambda case=case, config=config: candidate_call(case, config)
            )
            for name, config in configs.items()
        }
        for round_index in range(1, int(args.rounds) + 1):
            samples = time_calls(
                calls,
                cp=cp,
                seed=20260717 + count * 100 + round_index,
                warmups=int(args.warmups),
                repeats=int(args.repeats),
            )
            for name, values in samples.items():
                rows.append(
                    {
                        "candidate": name,
                        "particle_count": count,
                        "site_count": int(case["site_count"]),
                        "round": round_index,
                        "median_ms": float(statistics.median(values)),
                        "p05_ms": percentile(values, 5),
                        "p95_ms": percentile(values, 95),
                    }
                )

    ranking = []
    for name in configs:
        values = [float(row["median_ms"]) for row in rows if row["candidate"] == name]
        ranking.append(
            {
                "candidate": name,
                "geometric_mean_case_round_median_ms": geometric_mean(values),
                "worst_case_round_median_ms": max(values),
            }
        )
    ranking.sort(
        key=lambda item: (
            item["geometric_mean_case_round_median_ms"],
            item["worst_case_round_median_ms"],
        )
    )
    repeat_stress: dict[str, Any] | None = None
    if args.finalists_only:
        stress_case = cases[10]
        stress_pore = stress_case["mask"] != 0
        stress_checks: dict[str, dict[str, int]] = {}
        for name, config in configs.items():
            failing_calls = 0
            maximum_label_mismatches = 0
            maximum_distance_mismatches = 0
            for _ in range(500):
                output = candidate_call(stress_case, config)
                label_mismatches = int(
                    cp.count_nonzero(
                        stress_pore
                        & (output[0] != stress_case["reference"][0])
                    ).get()
                )
                distance_mismatches = int(
                    cp.count_nonzero(
                        stress_pore
                        & (output[1] != stress_case["reference"][1])
                    ).get()
                )
                failing_calls += int(
                    label_mismatches != 0 or distance_mismatches != 0
                )
                maximum_label_mismatches = max(
                    maximum_label_mismatches, label_mismatches
                )
                maximum_distance_mismatches = max(
                    maximum_distance_mismatches, distance_mismatches
                )
            stress_checks[name] = {
                "failing_calls": failing_calls,
                "maximum_label_mismatch_voxels": maximum_label_mismatches,
                "maximum_distance_mismatch_voxels": maximum_distance_mismatches,
            }
        repeat_stress = {
            "particle_count": 10,
            "repeats_per_candidate": 500,
            "checks": stress_checks,
        }
        if any(
            item["failing_calls"]
            or item["maximum_label_mismatch_voxels"]
            or item["maximum_distance_mismatch_voxels"]
            for item in stress_checks.values()
        ):
            raise RuntimeError("at least one finalist failed repeated pointwise stress")
    report = {
        "schema_version": "ptv-roi-jfa-warp64-candidate-screen-v2",
        "scope": (
            "Finite ROI-JFA kernel screen on five fixed 64^3 Berea workloads; "
            "not a hardware-independent global optimality claim."
        ),
        "protocol": {
            "rounds": int(args.rounds),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "interleaved_random_order": True,
        },
        "configurations": configs,
        "checks": checks,
        "repeat_stress": repeat_stress,
        "ranking": ranking,
        "rows": rows,
    }
    output_path = AUDIT_DIR / "ptv_ownership_warp64_variant_screen.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "top": ranking[:10],
                "reference": next(
                    item
                    for item in ranking
                    if item["candidate"]
                    == (
                        "compact_reference"
                        if args.launch_screen
                        else (
                            "launch_b256_r8"
                            if args.launch_finalists
                            else (
                                "production_compact64_r8"
                                if args.finalists_only
                                else "selected"
                            )
                        )
                    )
                ),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
