"""Run the locked baselines under the operator of the paper and export the table (study B, step 2).

Accuracy is opened only after ``protocols/baseline_selection_locked.json`` exists
and its SHA-256 is recorded.  The proposed rows re-use the study-A runs whose
site sets and geometries are already frozen and hashed.

Usage:
    python run_site_rule_comparison.py [--case CASE] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import build_site_rule_partitions as bcb
import study_common as ec
import forward_runs as fr

METHOD_DISPLAY = {
    "proposed": "PoreVoronoi-FV (trajectory sites)",
    "mask_graph_fps": "Mask-only graph FPS",
    "connected_block_agglomeration": "Connected block agglomeration",
    "euclidean_partition_with_deterministic_connectivity_repair": "Euclidean partition (repaired)",
}

BASELINE_COLUMNS = [
    "protocol_id", "case", "case_display", "budget_label", "method_id", "method_display",
    "site_or_partition_rule", "selection_used_reference_errors", "admissibility_status",
    "N_f", "N_sites", "N_cv", "N_edges", "N_connected_patches", "N_trace_modes",
    "N_trace_vector_dofs", "N_pressure_dofs_after_gauge", "N_system",
    "proposed_N_system", "relative_N_system_difference_from_proposed",
    "e_K_percent", "e_phi_percent", "e_u_percent", "e_p_percent",
    "mass_inf_per_volume", "momentum_residual_inf", "linear_solver_iterations",
    "linear_solver_relative_residual", "beta_h", "schur_condition_proxy",
    "t_initial_build_s", "t_cached_call_median_s", "factor_memory_MiB",
    "solver_formulation_id", "source_code_sha256", "input_bundle_sha256", "run_directory",
    # extra provenance columns beyond the required schema
    "level_label", "h_S_over_h", "q_S_over_h", "mesh_ratio_hS_over_qS",
    "fragmented_edge_fraction", "patches_per_edge_mean", "orientation_anisotropy",
    "cell_volume_cv", "K_eff_parallel", "K_ref_parallel", "graph_metric_periodic_x",
    # Pressure-metric companion columns.  e_p_percent is exported in the physical
    # convention; the multiplier-convention value, the cosine that
    # established the sign relation, and the reference-degeneracy flags travel with it so
    # that regenerating this table never silently drops them.
    "e_p_percent_multiplier_convention",
    "pressure_multiplier_vs_reference_cosine",
    "e_p_sign_reconciliation_applied",
    "p_h_Mp_norm",
    "p_ref_Mp_norm",
    "pressure_scale_fL_sqrtV",
    "e_p_reference_degenerate",
    "e_p_interpretable",
    # Interpretation columns: the signed permeability error, the
    # reference-convergence state, and the descriptor-comparability note. They must survive
    # any regeneration of this table, so they belong in the authoritative column list.
    "e_K_signed_percent",
    "reference_steady_converged",
    "reference_steady_momentum_inf",
    "reference_steps_completed",
    "geometry_descriptor_status",
]


def locked_record(delivery_root: Path) -> tuple[dict[str, Any], str]:
    path = delivery_root / "protocols" / "baseline_selection_locked.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Run build_site_rule_partitions.py first; the result-blind selection "
            "must be locked before any baseline accuracy result is opened"
        )
    return json.loads(path.read_text(encoding="utf-8")), ec.sha256_file(path)


def phase_a_row(delivery_root: Path, run_directory: str) -> dict[str, Any]:
    manifest = Path(run_directory) / "run_manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["row"]


def run_baseline(
    case: str,
    budget: str,
    method: str,
    selection: dict[str, Any],
    delivery_root: Path,
    resume: bool,
) -> dict[str, Any]:
    run_dir = delivery_root / "runs" / "site_rule_comparison" / case / budget / method
    cached = run_dir / "run_manifest.json"
    if resume and cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))["row"]

    paths = ec.case_paths(case)
    mask = ec.load_mask(paths["mask"])
    if method == "mask_graph_fps":
        count = int(selection["candidate_site_count"])
        sites = bcb.mask_fps_order(case, count, delivery_root)[:count]
        outcome = fr.run_point(
            case=case,
            seed_flat=np.asarray(sites, dtype=np.int64),
            mask_path=paths["mask"],
            reference_path=paths["reference"],
            run_dir=run_dir,
            seed_spec=f"mask_graph_fps_{case}_{budget}_n{count}",
            graph_periodic_x=False,
            method_id=method,
            site_rule=selection["rule"],
        )
    elif method == "connected_block_agglomeration":
        offset = tuple(int(value) for value in selection["block_offset"])
        if str(selection.get("family", "fixed_side")) == "graded":
            block_count = int(selection["requested_block_count"])
            labels, divisions = bcb.graded_block_labels(mask, block_count)
            if list(divisions) != list(selection["axis_divisions"]):
                raise RuntimeError(
                    "The locked graded block selection does not reproduce its axis divisions"
                )
            tag = "graded_%dx%dx%d" % (divisions[0], divisions[1], divisions[2])
        else:
            sides = tuple(int(value) for value in selection["block_sides"])
            labels = bcb.block_labels(mask, sides, offset)
            tag = "%dx%dx%d" % (sides[0], sides[1], sides[2])
        outcome = fr.run_point(
            case=case,
            partition_labels=labels,
            mask_path=paths["mask"],
            reference_path=paths["reference"],
            run_dir=run_dir,
            seed_spec=f"connected_block_{case}_{budget}_{tag}",
            graph_periodic_x=True,
            method_id=method,
            site_rule=selection["rule"],
        )
    elif method == "euclidean_partition_with_deterministic_connectivity_repair":
        family = json.loads(
            (
                delivery_root / "source_data" / "site_families" / case / "site_family_manifest.json"
            ).read_text(encoding="utf-8")
        )
        level = next(
            item for item in family["levels"] if item["level_label"] == selection["level_label"]
        )
        sites = np.load(level["sites_npz"])["seed_flat"].astype(np.int64)
        labels, _report = bcb.euclidean_partition_with_repair(mask, sites)
        outcome = fr.run_point(
            case=case,
            partition_labels=labels,
            mask_path=paths["mask"],
            reference_path=paths["reference"],
            run_dir=run_dir,
            seed_spec=f"euclidean_repaired_{case}_{budget}_{selection['level_label']}",
            graph_periodic_x=True,
            method_id=method,
            site_rule=selection["rule"],
        )
    else:
        raise ValueError(f"Unknown baseline method {method!r}")
    return outcome["row"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery-root", type=Path, default=ec.DELIVERY_ROOT)
    parser.add_argument("--case", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    delivery_root = Path(args.delivery_root)
    record, record_sha = locked_record(delivery_root)
    cases = [item.strip() for item in args.case.split(",") if item.strip()] or list(record["cases"])

    rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for case in cases:
        entry = record["cases"][case]
        for budget, proposed in entry["proposed"].items():
            proposed_system = int(proposed["N_system"])
            base_row = phase_a_row(delivery_root, proposed["run_directory"])
            rows.append(
                {
                    **base_row,
                    "budget_label": budget,
                    "method_id": "proposed",
                    "method_display": METHOD_DISPLAY["proposed"],
                    "site_or_partition_rule": (
                        "frozen trajectory-derived sites selected by deterministic nested "
                        "graph-farthest-point sampling; production graph-geodesic ownership"
                    ),
                    "selection_used_reference_errors": False,
                    "proposed_N_system": proposed_system,
                    "relative_N_system_difference_from_proposed": 0.0,
                    "admissibility_status": base_row.get("admissibility_status", "PASS"),
                    "level_label": proposed["level_label"],
                }
            )
            for method in (
                "mask_graph_fps",
                "connected_block_agglomeration",
                "euclidean_partition_with_deterministic_connectivity_repair",
            ):
                block = entry.get(method)
                if not isinstance(block, dict) or "selections" not in block:
                    continue
                selection = block["selections"].get(budget)
                if selection is None:
                    continue
                if not bool(selection.get("within_maximum_10pct", True)):
                    unmatched.append(
                        {
                            "case": case,
                            "budget_label": budget,
                            "method_id": method,
                            "N_cv": selection.get("N_cv"),
                            "N_system": selection.get("N_system"),
                            "proposed_N_system": proposed_system,
                            "relative_N_system_difference_from_proposed": selection.get(
                                "relative_N_system_difference_from_proposed"
                            ),
                            "status": "UNMATCHED",
                            "reason": (
                                "the nearest admissible candidate in this family is outside the "
                                "10% maximum of the matching rule; the family has no free size "
                                "parameter because it reuses the proposed method's own frozen "
                                "site set, so no better candidate exists. The row is reported "
                                "here and excluded from tables/site_rule_comparison.csv "
                                "rather than filed as a matched comparison."
                            ),
                        }
                    )
                    print(
                        f"[phaseB] {case}/{budget}/{method}: UNMATCHED "
                        f"({float(selection['relative_N_system_difference_from_proposed']):+.2%}); "
                        "reported, not run",
                        flush=True,
                    )
                    continue
                if args.dry_run:
                    print(case, budget, method, selection, flush=True)
                    continue
                start = time.perf_counter()
                row = run_baseline(case, budget, method, selection, delivery_root, args.resume)
                relative = (int(row["N_system"]) - proposed_system) / max(proposed_system, 1)
                row.update(
                    {
                        "budget_label": budget,
                        "level_label": "",
                        "method_display": METHOD_DISPLAY[method],
                        "selection_used_reference_errors": False,
                        "proposed_N_system": proposed_system,
                        "relative_N_system_difference_from_proposed": float(relative),
                    }
                )
                rows.append(row)
                print(
                    f"[phaseB] {case}/{budget}/{method}: N_system={row['N_system']} "
                    f"({relative:+.3%}) e_K={row['e_K_percent']:.4f} e_u={row['e_u_percent']:.4f} "
                    f"admissible={row['admissibility_status']} ({time.perf_counter() - start:.1f}s)",
                    flush=True,
                )
    if args.dry_run:
        return 0

    for row in rows:
        row.setdefault("beta_h", "")
        row.setdefault("schur_condition_proxy", "")
    ec.write_csv(
        delivery_root / "tables" / "site_rule_comparison.csv", rows, BASELINE_COLUMNS
    )
    audit = {
        "protocol_id": ec.PROTOCOL_ID,
        "locked_selection_record": str(delivery_root / "protocols" / "baseline_selection_locked.json"),
        "locked_selection_sha256": record_sha,
        "selection_used_reference_errors": False,
        "evidence_selection_was_result_blind": (
            "build_site_rule_partitions.py never loads a reference NPZ, never calls "
            "solve_forward, and never computes an error metric; it evaluates only ownership, "
            "trace geometry, admissibility and N_system.  The locked record was written and "
            "hashed before this module opened any accuracy value."
        ),
        "rows": [
            {
                "case": row["case"],
                "budget_label": row["budget_label"],
                "method_id": row["method_id"],
                "N_system": row["N_system"],
                "proposed_N_system": row["proposed_N_system"],
                "relative_N_system_difference_from_proposed": row[
                    "relative_N_system_difference_from_proposed"
                ],
                "within_preferred_5pct": abs(float(row["relative_N_system_difference_from_proposed"]))
                <= bcb.MATCH_PREFERRED,
                "within_maximum_10pct": abs(float(row["relative_N_system_difference_from_proposed"]))
                <= bcb.MATCH_MAXIMUM,
                "admissibility_status": row["admissibility_status"],
                "matched": abs(float(row["relative_N_system_difference_from_proposed"]))
                <= bcb.MATCH_MAXIMUM,
            }
            for row in rows
        ],
    }
    audit["unmatched_candidates_reported_not_filed"] = unmatched
    audit["status"] = "PASS" if all(item["matched"] for item in audit["rows"]) else "FAIL"
    ec.write_json(delivery_root / "audits" / "baseline_matching_audit.json", audit)
    print(f"[phaseB] wrote {len(rows)} rows; matching audit {audit['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
