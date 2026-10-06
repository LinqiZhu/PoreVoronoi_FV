"""Gauge-fixed stability and conditioning audit for every study row.

Each row is rebuilt from its own frozen site set or partition through the same
production path, so the audited ``A`` and ``D`` are the blocks the solver
actually assembled - never a surrogate inferred from output fields.

Usage:
    python audit_gauge_fixed_stability.py [--study A|B] [--case CASE] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import study_common as ec
import stability_core as sc

STABILITY_COLUMNS = [
    "protocol_id", "study_source", "case", "level_or_budget", "method_id",
    "pressure_dimension_before_gauge", "pressure_dimension_after_gauge",
    "rank_defect_estimate_after_gauge", "schur_lambda_min_positive", "schur_lambda_max",
    "beta_h", "schur_condition_proxy", "spectral_method", "spectral_tolerance",
    "spectral_iterations", "spectral_residual_max", "gauge_choice",
    "gauge_sensitivity_relative_solution_difference", "kkt_symmetry_defect",
    "linear_solver_iterations", "linear_solver_relative_residual", "mass_inf_per_volume",
    "momentum_residual_inf", "spectral_status", "run_directory",
]


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def enumerate_rows_from_run_directories(delivery_root: Path) -> list[dict[str, Any]]:
    """Fall back to the run manifests when a final table has not been written yet."""
    rows: list[dict[str, Any]] = []
    base = delivery_root / "runs" / "controlled_refinement"
    if base.is_dir():
        for case_dir in sorted(base.iterdir()):
            for level_dir in sorted(case_dir.iterdir()):
                cached = level_dir / "run_manifest.json"
                if not cached.is_file():
                    continue
                record = json.loads(cached.read_text(encoding="utf-8"))["row"]
                rows.append(
                    {
                        "study_source": "A_controlled_refinement",
                        "case": str(record["case"]),
                        "level_or_budget": str(record["level_label"]),
                        "method_id": "proposed",
                        "run_directory": str(level_dir),
                        "N_sites": int(record["N_sites"]),
                        "N_cv": int(record["N_cv"]),
                        "source_row": record,
                    }
                )
    base = delivery_root / "runs" / "site_rule_comparison"
    if base.is_dir():
        for case_dir in sorted(base.iterdir()):
            for budget_dir in sorted(case_dir.iterdir()):
                for method_dir in sorted(budget_dir.iterdir()):
                    cached = method_dir / "run_manifest.json"
                    if not cached.is_file():
                        continue
                    record = json.loads(cached.read_text(encoding="utf-8"))["row"]
                    rows.append(
                        {
                            "study_source": "B_site_rule_comparison",
                            "case": str(record["case"]),
                            "level_or_budget": budget_dir.name,
                            "method_id": str(record["method_id"]),
                            "run_directory": str(method_dir),
                            "N_sites": int(record["N_sites"]),
                            "N_cv": int(record["N_cv"]),
                            "source_row": record,
                        }
                    )
    return rows


def enumerate_rows(delivery_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_table(delivery_root / "tables" / "controlled_refinement.csv"):
        rows.append(
            {
                "study_source": "A_controlled_refinement",
                "case": record["case"],
                "level_or_budget": record["level_label"],
                "method_id": "proposed",
                "run_directory": record["run_directory"],
                "N_sites": int(record["N_sites"]),
                "N_cv": int(record["N_cv"]),
                "source_row": record,
            }
        )
    for record in read_table(delivery_root / "tables" / "site_rule_comparison.csv"):
        if record["method_id"] == "proposed":
            # identical system to the study-A row it re-uses; audited once under study A
            continue
        rows.append(
            {
                "study_source": "B_site_rule_comparison",
                "case": record["case"],
                "level_or_budget": record["budget_label"],
                "method_id": record["method_id"],
                "run_directory": record["run_directory"],
                "N_sites": int(record["N_sites"]),
                "N_cv": int(record["N_cv"]),
                "source_row": record,
            }
        )
    return rows


def rebuild_system(run_directory: Path) -> tuple[Any, dict[str, Any]]:
    manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    stored = np.load(run_directory / "sites_and_partition.npz")
    mask_path = Path(manifest["input_paths"]["mask_npz"])
    boot = ec.bootstrap()
    seed_flat = stored["seed_flat"].astype(np.int64)
    if seed_flat.size:
        geom, _meta, _time = ec.build_geometry_from_seed_flat(
            seed_flat, mask_path=mask_path, seed_spec=f"stability_{run_directory.name}"
        )
    else:
        geom, _meta, _time, _labels = ec.build_geometry_from_partition(
            stored["labels"], mask_path=mask_path, seed_spec=f"stability_{run_directory.name}"
        )
    trace = boot["build_hybrid_trace_geometry"](
        boot["ns"], geom, boot["cfg"], trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
    )
    system = boot["assemble_moment_constrained_hybrid_stokes"](
        trace,
        viscosity=float(boot["cfg"].nu),
        body_force=np.asarray(ec.FORWARD_ARGS["body_force"], dtype=np.float64),
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]),
    )
    del geom
    ec.free_gpu()
    return system, manifest


def audit_row(item: dict[str, Any], delivery_root: Path, with_gauge: bool, resume: bool) -> dict[str, Any]:
    tag = f"{item['study_source']}__{item['case']}__{item['level_or_budget']}__{item['method_id']}"
    out_dir = delivery_root / "runs" / "stability_audit" / tag
    cached = out_dir / "stability_record.json"
    if resume and cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))["row"]
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    system, manifest = rebuild_system(Path(item["run_directory"]))
    A = system.velocity_matrix
    D = system.divergence_matrix
    n_cells = int(system.trace_geometry.n_cells)

    solver = sc.VelocityBlockSolver(A)
    spectrum = sc.schur_spectrum(
        A, D, system.trace_geometry.cell_volume, gauge_index=n_cells - 1, solver=solver
    )
    kkt = sc.build_reduced_kkt(A, sc.gauge_reduced_divergence(D, n_cells - 1))
    symmetry = sc.kkt_symmetry_defect(kkt)
    del kkt

    gauge_record: dict[str, Any] | None = None
    if with_gauge:
        gauge_record = sc.gauge_sensitivity(
            system,
            rtol=float(ec.FORWARD_ARGS["linear_rtol"]),
            maxiter=int(ec.FORWARD_ARGS["linear_maxiter"]),
            refinement_steps=int(ec.FORWARD_ARGS["linear_refinement_steps"]),
        )

    source = item["source_row"]
    row = {
        "protocol_id": ec.PROTOCOL_ID,
        "study_source": item["study_source"],
        "case": item["case"],
        "level_or_budget": item["level_or_budget"],
        "method_id": item["method_id"],
        "pressure_dimension_before_gauge": spectrum["pressure_dimension_before_gauge"],
        "pressure_dimension_after_gauge": spectrum["pressure_dimension_after_gauge"],
        "rank_defect_estimate_after_gauge": spectrum["rank_defect_estimate_after_gauge"],
        "schur_lambda_min_positive": spectrum["schur_lambda_min_positive"],
        "schur_lambda_max": spectrum["schur_lambda_max"],
        "beta_h": spectrum["beta_h"],
        "schur_condition_proxy": spectrum["schur_condition_proxy"],
        "spectral_method": spectrum["spectral_method"],
        "spectral_tolerance": spectrum["spectral_tolerance"],
        "spectral_iterations": spectrum["spectral_iterations"],
        "spectral_residual_max": spectrum["spectral_residual_max"],
        "gauge_choice": spectrum["gauge_choice"],
        "gauge_sensitivity_relative_solution_difference": (
            gauge_record["gauge_sensitivity_relative_solution_difference"] if gauge_record else ""
        ),
        "kkt_symmetry_defect": symmetry,
        "linear_solver_iterations": int(float(source["linear_solver_iterations"])),
        "linear_solver_relative_residual": float(source["linear_solver_relative_residual"]),
        "mass_inf_per_volume": float(source["mass_inf_per_volume"]),
        "momentum_residual_inf": float(source["momentum_residual_inf"]),
        "spectral_status": spectrum["spectral_status"],
        "run_directory": str(out_dir),
    }
    ec.write_json(
        cached,
        {
            "row": row,
            "spectrum": spectrum,
            "gauge_sensitivity": gauge_record,
            "kkt_symmetry_defect": symmetry,
            "forward_run_directory": item["run_directory"],
            "forward_run_manifest_sha256": ec.sha256_file(
                Path(item["run_directory"]) / "run_manifest.json"
            ),
            "elapsed_s": float(time.perf_counter() - start),
            "code_sha256": {
                name: ec.sha256_file(ec.CODE_DIR / name) for name in ec.REQUIRED_CODE_FILES
            },
        },
    )
    del system, solver
    gc.collect()
    ec.free_gpu()
    return row


def gauge_sensitivity_targets(rows: list[dict[str, Any]]) -> set[str]:
    """One coarse, one intermediate and one fine study-A level per case."""
    targets: set[str] = set()
    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        if not str(item["study_source"]).startswith("A"):
            continue
        by_case.setdefault(item["case"], []).append(item)
    for case, items in by_case.items():
        ordered = sorted(items, key=lambda entry: entry["N_cv"])
        if not ordered:
            continue
        picks = {0, len(ordered) // 2, len(ordered) - 1}
        for index in sorted(picks):
            entry = ordered[index]
            targets.add(
                f"{entry['study_source']}__{entry['case']}__{entry['level_or_budget']}__{entry['method_id']}"
            )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery-root", type=Path, default=ec.DELIVERY_ROOT)
    parser.add_argument("--study", default="")
    parser.add_argument("--case", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-gauge-sensitivity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    delivery_root = Path(args.delivery_root)
    rows = enumerate_rows(delivery_root) or enumerate_rows_from_run_directories(delivery_root)
    if args.study:
        wanted = tuple(item.strip().upper() for item in args.study.split(",") if item.strip())
        rows = [item for item in rows if str(item["study_source"]).upper().startswith(wanted)]
    if args.case:
        wanted_cases = {item.strip() for item in args.case.split(",") if item.strip()}
        rows = [item for item in rows if item["case"] in wanted_cases]
    targets = (
        set()
        if args.no_gauge_sensitivity
        else gauge_sensitivity_targets(
            enumerate_rows(delivery_root) or enumerate_rows_from_run_directories(delivery_root)
        )
    )

    if args.dry_run:
        for item in rows:
            tag = f"{item['study_source']}__{item['case']}__{item['level_or_budget']}__{item['method_id']}"
            print(tag, "N_cv", item["N_cv"], "gauge_sensitivity", tag in targets, flush=True)
        return 0

    for item in sorted(rows, key=lambda entry: entry["N_cv"]):
        tag = f"{item['study_source']}__{item['case']}__{item['level_or_budget']}__{item['method_id']}"
        start = time.perf_counter()
        row = audit_row(item, delivery_root, tag in targets, args.resume)
        print(
            f"[phaseC] {tag}: n_p={row['pressure_dimension_after_gauge']} "
            f"lambda_min={row['schur_lambda_min_positive']:.6e} lambda_max={row['schur_lambda_max']:.6e} "
            f"beta_h={row['beta_h']:.6e} cond={row['schur_condition_proxy']:.6e} "
            f"defect={row['rank_defect_estimate_after_gauge']} status={row['spectral_status']} "
            f"({time.perf_counter() - start:.1f}s)",
            flush=True,
        )

    directory = delivery_root / "runs" / "stability_audit"
    merged: dict[str, dict[str, Any]] = {}
    if directory.is_dir():
        for item_dir in sorted(directory.iterdir()):
            cached = item_dir / "stability_record.json"
            if cached.is_file():
                row = json.loads(cached.read_text(encoding="utf-8"))["row"]
                merged[item_dir.name] = row
    ordered = [merged[key] for key in sorted(merged)]
    ec.write_csv(delivery_root / "tables" / "stability_audit.csv", ordered, STABILITY_COLUMNS)
    print(f"[phaseC] wrote {len(ordered)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
