from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


NUMERICAL_FIELDS = (
    "K_eff_parallel",
    "K_ref_parallel",
    "e_K_percent",
    "e_phi_percent",
    "e_u_percent",
    "e_u_parallel_percent",
    "mass_inf_per_volume",
    "linear_solver_relative_residual",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 1.0e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one frozen-site cross-flow prediction for paper use."
    )
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--reference-continuation-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    final = load_json(args.final_manifest)
    pilot = load_json(args.pilot_manifest)
    window = load_json(args.window_manifest)
    reference = load_json(args.reference_continuation_manifest)
    row = final["row"]
    pilot_row = pilot["row"]
    input_record = final["input"]
    reference_row = next(
        item for item in reference["rows"] if item["axis"] == row["evaluation_flow_axis"]
    )

    require(row["site_flow_axis"] == "x", "Site flow axis is not frozen x-flow")
    require(row["evaluation_flow_axis"] == "y", "Evaluation flow axis is not y")
    require(row["site_evaluation_axes_distinct"] is True, "Flow axes are not distinct")
    require(
        row["validation_role"]
        == "frozen_site_cross_flow_prediction_independent_of_evaluation_flow_sites",
        "Cross-flow validation role is missing",
    )
    require(input_record["label_backend"] == "exact_frontier_gpu", "Wrong label backend")
    require(int(row["particle_site_count"]) == 7779, "Unexpected frozen site count")
    require(int(row["auxiliary_site_count"]) == 0, "Auxiliary sites are present")
    require(
        int(input_record["particle_snap_nonzero_unique_rounded"]) == 0,
        "Transformed particle sites were snapped",
    )
    require(
        window["status"] == "PASS_FROZEN_CROSS_FLOW_WINDOW_TRANSFORM",
        "Window transformation did not pass",
    )
    require(window["same_physical_mask"] is True, "Physical masks differ")
    require(window["site_selection_changed"] is False, "Site selection changed")
    require(
        int(window["source_unique_rounded_sites"])
        == int(window["target_unique_rounded_sites"])
        == int(row["particle_site_count"]),
        "Source/target site counts differ",
    )
    require(
        final["input_sha256"]["particle_window"] == window["output_window_sha256"],
        "Final run does not use the audited transformed window",
    )
    require(
        final["input_sha256"]["particle_window_provenance"]
        == file_sha256(args.window_manifest),
        "Final run provenance hash mismatch",
    )
    require(
        final["input_sha256"]["reference_npz"] == reference_row["reference_sha256"],
        "Final run reference hash mismatch",
    )
    require(int(row["repeat_count"]) == 3, "Final run does not contain three repeats")
    require(len(final["cached_call_times_s"]) == 3, "Timing repeat count mismatch")
    require(int(row["linear_solver_info"]) == 0, "Linear solver did not converge")
    require(float(row["linear_solver_relative_residual"]) <= 1.0e-12, "Linear residual too large")
    require(float(row["mass_inf_per_volume"]) <= 1.0e-12, "Mass residual too large")
    require(row["reference_steady_converged"] is False, "Reference status drifted")
    require(
        int(row["reference_steps_completed_cumulative"])
        == int(reference_row["steps_completed_cumulative"]),
        "Reference step count mismatch",
    )
    require(
        close(
            row["reference_parent_to_child_K_rel_change_percent"],
            reference_row["comparison_parent_to_child"]["K_rel_change_percent"],
        ),
        "Reference continuation change mismatch",
    )
    for field in NUMERICAL_FIELDS:
        pilot_value = (
            pilot_row["e_u_x_percent"]
            if field == "e_u_parallel_percent" and field not in pilot_row
            else pilot_row[field]
        )
        require(close(row[field], pilot_value), f"Pilot/final mismatch in {field}")

    evidence = {
        "status": "PASS_FROZEN_SITE_CROSS_FLOW_PREDICTION",
        "claim_boundary": (
            "Independent of the evaluation-flow site layout; not experimental or "
            "cross-rock validation."
        ),
        "site_flow_axis": "x",
        "evaluation_flow_axis": "y",
        "same_physical_mask": True,
        "site_selection_changed": False,
        "particle_site_count": int(row["particle_site_count"]),
        "auxiliary_site_count": int(row["auxiliary_site_count"]),
        "reference_steady_converged": bool(row["reference_steady_converged"]),
        "reference_steps_completed_cumulative": int(
            row["reference_steps_completed_cumulative"]
        ),
        "reference_parent_to_child_K_rel_change_percent": float(
            row["reference_parent_to_child_K_rel_change_percent"]
        ),
        "metrics": {field: row[field] for field in NUMERICAL_FIELDS},
        "e_u_physical_percent": {
            axis: row[f"e_u_physical_{axis}_percent"] for axis in ("x", "y", "z")
        },
        "repeat_count": int(row["repeat_count"]),
        "cached_call_times_s": final["cached_call_times_s"],
        "cached_call_median_s": float(row["t_cached_call_median_s"]),
        "linear_solver_iterations": int(row["linear_solver_iterations"]),
        "linear_solver_info": int(row["linear_solver_info"]),
        "input_sha256": final["input_sha256"],
        "code_sha256": final["code_sha256"],
        "source_manifests": {
            "final": str(args.final_manifest.resolve()),
            "pilot": str(args.pilot_manifest.resolve()),
            "window": str(args.window_manifest.resolve()),
            "reference_continuation": str(
                args.reference_continuation_manifest.resolve()
            ),
        },
        "source_manifest_sha256": {
            "final": file_sha256(args.final_manifest),
            "pilot": file_sha256(args.pilot_manifest),
            "window": file_sha256(args.window_manifest),
            "reference_continuation": file_sha256(
                args.reference_continuation_manifest
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "frozen_x_sites_to_y_flow_evidence.json"
    csv_path = args.out_dir / "frozen_x_sites_to_y_flow_evidence.csv"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    flat_row = {
        "site_flow_axis": evidence["site_flow_axis"],
        "evaluation_flow_axis": evidence["evaluation_flow_axis"],
        "particle_site_count": evidence["particle_site_count"],
        "N_cv": row["N_cv"],
        "compression_Nfl_over_Ncv": input_record["C_comp"],
        "K_eff_parallel": row["K_eff_parallel"],
        "K_ref_parallel": row["K_ref_parallel"],
        "e_K_percent": row["e_K_percent"],
        "e_phi_percent": row["e_phi_percent"],
        "e_u_percent": row["e_u_percent"],
        "e_u_parallel_percent": row["e_u_parallel_percent"],
        "mass_inf_per_volume": row["mass_inf_per_volume"],
        "linear_solver_relative_residual": row["linear_solver_relative_residual"],
        "cached_call_median_s": row["t_cached_call_median_s"],
        "repeat_count": row["repeat_count"],
        "reference_steady_converged": row["reference_steady_converged"],
        "validation_role": row["validation_role"],
    }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_row))
        writer.writeheader()
        writer.writerow(flat_row)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
