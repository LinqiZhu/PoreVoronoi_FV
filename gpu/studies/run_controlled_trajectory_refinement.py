"""Controlled trajectory-derived refinement runs and empirical slopes (study A, step 2).

Runs the retained solver on every deterministic nested level built by
``build_nested_trajectory_site_family.py`` and exports the full geometry,
accuracy, conservation, cost and solver record.

Usage:
    python run_controlled_trajectory_refinement.py [--case CASE] [--level N] [--resume]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import study_common as ec
import forward_runs as fr

SLOPE_METRICS = ("e_K_percent", "e_phi_percent", "e_u_percent", "e_p_percent")
SLOPE_RULE = (
    "A log-log fit is reported only for a fixed deterministic refinement family "
    "with at least three finest levels of strictly decreasing h_S and positive "
    "finite errors.  The result is an empirical slope, not a theoretical or "
    "verified convergence order."
)

CONTROLLED_REFINEMENT_COLUMNS = [
    "protocol_id", "case", "case_display", "level_index", "level_label", "selection_rule",
    "full_candidate_site_count", "target_site_count", "N_sites", "N_f", "N_cv", "N_edges",
    "N_interface_facelets", "N_connected_patches", "N_trace_modes", "N_trace_vector_dofs",
    "N_pressure_dofs_after_gauge", "N_system", "h_S_over_h", "q_S_over_h", "mesh_ratio_hS_over_qS",
    "cell_graph_radius_mean", "cell_graph_radius_p95", "cell_graph_radius_max",
    "cell_volume_mean", "cell_volume_cv", "cell_volume_min", "cell_volume_max",
    "patches_per_edge_mean", "fragmented_edge_fraction", "area_weighted_chi_p05",
    "area_weighted_chi_median", "chi_min", "orientation_anisotropy",
    "K_eff_parallel", "K_ref_parallel", "e_K_percent", "e_phi_percent", "e_u_percent", "e_p_percent",
    "mass_inf_per_volume", "momentum_residual_inf", "linear_solver_iterations",
    "linear_solver_relative_residual", "t_geometry_build_s", "t_trace_geometry_s",
    "t_matrix_assembly_s", "t_factorization_s", "t_cached_call_median_s", "factor_memory_MiB",
    "solver_formulation_id", "source_code_sha256", "input_bundle_sha256", "run_directory",
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


def family_manifest(case: str, delivery_root: Path) -> dict[str, Any]:
    path = delivery_root / "source_data" / "site_families" / case / "site_family_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing site family for {case}; run build_nested_trajectory_site_family.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_level(case: str, level: dict[str, Any], manifest: dict[str, Any], delivery_root: Path, resume: bool) -> dict[str, Any]:
    run_dir = (
        delivery_root
        / "runs"
        / "controlled_refinement"
        / case
        / f"level_{level['level_index']:02d}_{level['level_label']}"
    )
    cached = run_dir / "run_manifest.json"
    if resume and cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))["row"]

    sites = np.load(level["sites_npz"])["seed_flat"].astype(np.int64)
    if ec.sha256_array(np.sort(sites)) != level["site_set_sha256"]:
        raise RuntimeError(f"{case}/{level['level_label']}: frozen site set hash mismatch")
    paths = ec.case_paths(case)
    outcome = fr.run_point(
        case=case,
        seed_flat=sites,
        mask_path=paths["mask"],
        reference_path=paths["reference"],
        run_dir=run_dir,
        seed_spec=f"nested_graph_fps_{case}_{level['level_label']}_n{sites.size}",
        repeats=int(ec.FORWARD_ARGS["repeats"]),
        extra_inputs={"particle_window": paths["particle_window"]},
        save_state=True,
    )
    row = outcome["row"]
    row.update(
        {
            "level_index": int(level["level_index"]),
            "level_label": str(level["level_label"]),
            "selection_rule": manifest["selection_rule"],
            "full_candidate_site_count": int(manifest["candidate_pool_size"]),
            "target_site_count": int(level["target_site_count"]),
            "site_set_sha256": level["site_set_sha256"],
            "candidate_pool_sha256": manifest["candidate_pool_sha256"],
        }
    )
    run_manifest = outcome["manifest"]
    run_manifest["row"] = row
    run_manifest["level"] = level
    run_manifest["selection_rule"] = manifest["selection_rule"]
    run_manifest["graph_metric_periodic_x"] = manifest["graph_metric_periodic_x"]
    ec.write_json(cached, run_manifest)
    return row


def degenerate_pressure_cases(delivery_root: Path) -> set[str]:
    """Cases whose reference pressure sits at the reference solver's zero floor."""
    path = delivery_root / "source_data" / "pressure_metric_diagnostics.csv"
    if not path.is_file():
        return set()
    import csv as _csv

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            row["case"]
            for row in _csv.DictReader(handle)
            if str(row.get("e_p_reference_degenerate", "")).casefold() in {"true", "1"}
        }


def fit_empirical_slopes(
    rows: list[dict[str, Any]], degenerate_pressure: set[str] | None = None
) -> list[dict[str, Any]]:
    degenerate_pressure = degenerate_pressure or set()
    output: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    for case, case_rows in sorted(by_case.items()):
        ordered = sorted(case_rows, key=lambda item: int(item["N_sites"]))
        h_values = [float(item["h_S_over_h"]) for item in ordered]
        # longest strictly decreasing suffix in h_S over the finest levels
        start = len(ordered) - 1
        while start - 1 >= 0 and h_values[start - 1] > h_values[start]:
            start -= 1
        window = ordered[start:]
        for metric in SLOPE_METRICS:
            errors = [float(item[metric]) for item in window]
            if metric == "e_p_percent" and case in degenerate_pressure:
                output.append(
                    {
                        "protocol_id": ec.PROTOCOL_ID,
                        "metric": metric,
                        "case": case,
                        "levels_used": "",
                        "n_points": len(window),
                        "slope": "",
                        "intercept": "",
                        "r_squared": "",
                        "eligible": False,
                        "reason": (
                            "the reference pressure for this case is identically constant to the "
                            "reference solver's round-off, so e_p_percent has no interpretable "
                            "denominator; see source_data/pressure_metric_diagnostics.csv"
                        ),
                        "interpretation": SLOPE_RULE,
                    }
                )
                continue
            eligible = (
                len(window) >= 3
                and all(np.isfinite(value) and value > 0.0 for value in errors)
                and all(h_values[start + index] > h_values[start + index + 1] for index in range(len(window) - 1))
            )
            if not eligible:
                output.append(
                    {
                        "protocol_id": ec.PROTOCOL_ID,
                        "metric": metric,
                        "case": case,
                        "levels_used": "",
                        "n_points": len(window),
                        "slope": "",
                        "intercept": "",
                        "r_squared": "",
                        "eligible": False,
                        "reason": (
                            "fewer than three finest levels with strictly decreasing h_S"
                            if len(window) < 3
                            else "a compared error is not positive and finite"
                        ),
                        "interpretation": SLOPE_RULE,
                    }
                )
                continue
            x = np.log(np.asarray([float(item["h_S_over_h"]) for item in window], dtype=np.float64))
            y = np.log(np.asarray(errors, dtype=np.float64))
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            residual = float(np.sum((y - predicted) ** 2))
            total = float(np.sum((y - float(np.mean(y))) ** 2))
            r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
            output.append(
                {
                    "protocol_id": ec.PROTOCOL_ID,
                    "metric": metric,
                    "case": case,
                    "levels_used": ";".join(str(item["level_label"]) for item in window),
                    "n_points": len(window),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_squared),
                    "eligible": True,
                    "reason": "",
                    "interpretation": SLOPE_RULE,
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery-root", type=Path, default=ec.DELIVERY_ROOT)
    parser.add_argument("--case", default="")
    parser.add_argument("--level", default="", help="Comma separated level indices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    delivery_root = Path(args.delivery_root)
    cases = [item.strip() for item in args.case.split(",") if item.strip()] or [
        "orthogonal_duct",
        "skewed_duct",
        "bentheimer_crop",
    ]
    wanted_levels = {int(item) for item in args.level.split(",") if item.strip()} or None

    rows: list[dict[str, Any]] = []
    for case in cases:
        manifest = family_manifest(case, delivery_root)
        for level in manifest["levels"]:
            if wanted_levels is not None and int(level["level_index"]) not in wanted_levels:
                continue
            if args.dry_run:
                print(case, level["level_index"], level["level_label"], level["N_sites"], flush=True)
                continue
            start = time.perf_counter()
            row = run_level(case, level, manifest, delivery_root, args.resume)
            rows.append(row)
            print(
                f"[phaseA] {case}/{level['level_label']} N_sites={row['N_sites']} "
                f"N_system={row['N_system']} h_S={row['h_S_over_h']:.3f} q_S={row['q_S_over_h']:.3f} "
                f"e_K={row['e_K_percent']:.4f} e_u={row['e_u_percent']:.4f} e_p={row['e_p_percent']:.4f} "
                f"({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    if args.dry_run:
        return 0

    # Merge with any level already on disk so a partial invocation still writes a full table.
    all_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for case in ("orthogonal_duct", "skewed_duct", "bentheimer_crop"):
        directory = delivery_root / "runs" / "controlled_refinement" / case
        if not directory.is_dir():
            continue
        for level_dir in sorted(directory.iterdir()):
            cached = level_dir / "run_manifest.json"
            if cached.is_file():
                row = json.loads(cached.read_text(encoding="utf-8"))["row"]
                all_rows[(str(row["case"]), int(row["level_index"]))] = row
    ordered = [all_rows[key] for key in sorted(all_rows)]
    ec.write_csv(
        delivery_root / "tables" / "controlled_refinement.csv", ordered, CONTROLLED_REFINEMENT_COLUMNS
    )
    slopes = fit_empirical_slopes(ordered, degenerate_pressure_cases(delivery_root))
    ec.write_csv(
        delivery_root / "tables" / "empirical_slopes.csv",
        slopes,
        [
            "protocol_id",
            "metric",
            "case",
            "levels_used",
            "n_points",
            "slope",
            "intercept",
            "r_squared",
            "eligible",
            "reason",
            "interpretation",
        ],
    )
    print(f"[phaseA] wrote {len(ordered)} rows and {len(slopes)} slope records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
