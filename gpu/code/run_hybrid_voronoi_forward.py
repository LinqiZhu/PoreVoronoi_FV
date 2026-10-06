from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(r".")
PACKAGE = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE / "code"
if str(CODE_DIR) in sys.path:
    sys.path.remove(str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR))

import run_segmented_selector_geodesic_rows as segmented  # noqa: E402
from hybrid_voronoi_trace import (  # noqa: E402
    assemble_moment_constrained_hybrid_stokes,
    build_hybrid_trace_factorization,
    build_hybrid_trace_geometry,
    solve_moment_constrained_hybrid_stokes,
)
from hybrid_site_sources import load_particle_window_seed_flat  # noqa: E402


DEFAULT_REFERENCE = (
    ROOT
    / "outputs"
    / "reference_data"
    / "reference_data"
    / "bentheimer_sandstone_crop__reference.npz"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "forward_runs"


def _vector(text: str) -> np.ndarray:
    values = np.asarray([float(part.strip()) for part in text.split(",")], dtype=np.float64)
    if values.size != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated numbers")
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_cross_flow_provenance(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.particle_window_provenance is None:
        if args.site_flow_axis or args.evaluation_flow_axis:
            raise ValueError(
                "--particle-window-provenance is required when flow axes are declared"
            )
        return None
    if not args.site_flow_axis or not args.evaluation_flow_axis:
        raise ValueError(
            "--site-flow-axis and --evaluation-flow-axis are required with provenance"
        )
    provenance_path = Path(args.particle_window_provenance)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checks = {
        "status": provenance.get("status") == "PASS_FROZEN_CROSS_FLOW_WINDOW_TRANSFORM",
        "source_axis": provenance.get("source_axis") == args.site_flow_axis,
        "target_axis": provenance.get("target_axis") == args.evaluation_flow_axis,
        "output_window": provenance.get("output_window_sha256")
        == _file_sha256(Path(args.particle_window)),
        "target_mask": provenance.get("target_mask_sha256")
        == _file_sha256(Path(args.mask_npz)),
        "same_physical_mask": provenance.get("same_physical_mask") is True,
        "site_selection_changed": provenance.get("site_selection_changed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Cross-flow provenance failed checks: {failed}")
    return provenance


def _validation_role(args: argparse.Namespace) -> str:
    if not args.site_flow_axis and not args.evaluation_flow_axis:
        return "state_conditioned_site_diagnostic_not_independent_forward_validation"
    if args.site_flow_axis == args.evaluation_flow_axis:
        return "same_flow_site_diagnostic_not_independent_forward_validation"
    return "frozen_site_cross_flow_prediction_independent_of_evaluation_flow_sites"


def _reference_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "metadata" not in data.files:
            return {}
        raw = data["metadata"].item()
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unsupported reference metadata payload in {path}")


def _runner_args(args: argparse.Namespace) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        pressure_gauge_eps=1.0e-8,
        tikhonov=0.01,
        velocity_reconstruction_tikhonov=1.0e-10,
        pressure_gradient_weight="area",
        dt=5.0,
        n_steps=100,
        report_every=100,
        solver_identity_audit=False,
        face_mode="tight_clip",
        wall_ref_ccomp=64.0,
        wall_length_exponent=0.15,
        face_operator="geodesic_weights",
    )


def _build_geometry(
    runner: Any,
    ns: dict[str, Any],
    cfg: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], float]:
    if args.mask_npz is None:
        if args.case != "bentheimer_sandstone_crop":
            raise ValueError("--mask-npz is required for a non-Bentheimer case")
        mask = runner.load_mask_npz(ns, runner.BENTHEIMER_INPUT)
        input_path = Path(runner.BENTHEIMER_INPUT)
    else:
        input_path = Path(args.mask_npz)
        mask = runner.load_mask_npz(ns, input_path)

    cp = ns["cp"]
    mask_np = cp.asnumpy(mask).astype(bool, copy=False)
    if args.site_mode != "particle_window":
        raise ValueError("The paper runner accepts only flow-derived PTV particle windows")
    if args.particle_window is None:
        raise ValueError("--particle-window is required")
    cross_flow_provenance = _load_cross_flow_provenance(args)
    seed_flat_np, particle_metadata = load_particle_window_seed_flat(
        mask_np,
        Path(args.particle_window),
        frame_selection=str(args.particle_frames),
        particle_id_selection=str(args.particle_ids),
        max_snap_displacement_vox=0.0,
    )
    seed_flat = cp.asarray(seed_flat_np, dtype=cp.int64)
    seed_id = f"particle_window_{Path(args.particle_window).stem}_n{seed_flat_np.size}"
    site_metadata: dict[str, Any] = {
        "site_mode": "particle_window",
        "site_origin": "flow_or_measurement_derived_particle_trajectory",
        "validation_role": _validation_role(args),
        "site_flow_axis": str(args.site_flow_axis),
        "evaluation_flow_axis": str(args.evaluation_flow_axis),
        "site_evaluation_axes_distinct": bool(
            args.site_flow_axis
            and args.evaluation_flow_axis
            and args.site_flow_axis != args.evaluation_flow_axis
        ),
        "flow_or_measurement_state_used_for_site_generation": True,
        **particle_metadata,
        "particle_site_count": int(seed_flat_np.size),
        "auxiliary_candidate_count": 0,
        "auxiliary_retained_count": 0,
        "auxiliary_site_rule": "none",
        "total_site_count": int(seed_flat_np.size),
    }
    if cross_flow_provenance is not None:
        site_metadata["cross_flow_coordinate_contract"] = {
            "source_axis": cross_flow_provenance["source_axis"],
            "target_axis": cross_flow_provenance["target_axis"],
            "same_physical_mask": cross_flow_provenance["same_physical_mask"],
            "site_selection_changed": cross_flow_provenance["site_selection_changed"],
            "source_unique_rounded_sites": cross_flow_provenance[
                "source_unique_rounded_sites"
            ],
            "target_unique_rounded_sites": cross_flow_provenance[
                "target_unique_rounded_sites"
            ],
            "target_solver_to_physical_component_map": cross_flow_provenance[
                "target_solver_to_physical_component_map"
            ],
        }

    build_start = time.perf_counter()
    geom, meta = ns["pvfv_build_geometry_from_seed_flat_timed"](
        mask,
        seed_flat,
        cfg,
        split_face_components=True,
        label_mode=str(args.label_backend),
        seed_spec=seed_id,
    )
    build_time = float(time.perf_counter() - build_start)
    input_record = {
        "mask_npz": str(input_path),
        "seed_id": seed_id,
        "seed_count": int(seed_flat.size),
        "label_backend": str(args.label_backend),
        **site_metadata,
    }
    input_record.update({str(key): value for key, value in (meta or {}).items() if np.isscalar(value)})
    return geom, input_record, build_time


def _velocity_errors(U: np.ndarray, U_ref: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    numerator = float(np.sum(volume[:, None] * (U - U_ref) ** 2))
    denominator = max(float(np.sum(volume[:, None] * U_ref**2)), 1.0e-300)
    result = {"e_u_percent": 100.0 * np.sqrt(numerator / denominator)}
    for component, index in (("x", 0), ("y", 1), ("z", 2)):
        component_num = float(np.sum(volume * (U[:, index] - U_ref[:, index]) ** 2))
        component_den = max(float(np.sum(volume * U_ref[:, index] ** 2)), 1.0e-300)
        result[f"e_u_{component}_percent"] = 100.0 * np.sqrt(component_num / component_den)
    return result


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    segmented.configure_roi_environment()
    os.environ["PVFV_LABEL_BACKEND"] = str(args.label_backend)
    runner = segmented.load_runner()
    runner_args = _runner_args(args)
    ns, _fixed_wall_clone, _global_wall_clone = segmented.install_namespace(runner, runner_args)
    cp = ns["cp"]
    cfg = segmented.make_cfg(runner, ns, Path(args.out_dir), runner_args)
    cfg.body_force = tuple(float(value) for value in args.body_force)

    geom, input_record, geometry_build_time = _build_geometry(runner, ns, cfg, args)
    trace_start = time.perf_counter()
    trace = build_hybrid_trace_geometry(ns, geom, cfg, trace_basis=str(args.trace_basis))
    trace_geometry_time = float(time.perf_counter() - trace_start)
    system = assemble_moment_constrained_hybrid_stokes(
        trace,
        viscosity=float(cfg.nu),
        body_force=np.asarray(args.body_force, dtype=np.float64),
        viscous_form=str(args.viscous_form),
    )
    factorization = build_hybrid_trace_factorization(
        system,
        solver=str(args.linear_solver),
        iterative_rtol=float(args.linear_rtol),
        iterative_maxiter=int(args.linear_maxiter),
        iterative_refinement_steps=int(args.linear_refinement_steps),
        velocity_lu_ordering=str(args.velocity_lu_ordering),
    )
    repeats = max(int(args.repeats), 1)
    results = [
        solve_moment_constrained_hybrid_stokes(system, factorization=factorization)
        for _ in range(repeats)
    ]
    result = results[-1]
    cached_times = [float(item["cached_call_time_s"]) for item in results]

    reference_geom, reference_result = segmented.load_reference_npz(ns, Path(args.reference_npz), cfg)
    if reference_geom is None or reference_result is None:
        raise FileNotFoundError(args.reference_npz)
    U_ref_gpu, _p_ref_gpu = ns["coarsen_voxel_reference_to_coarse_gpu"](
        reference_geom, reference_result, geom
    )
    U_ref = cp.asnumpy(U_ref_gpu).astype(np.float64)
    volume = cp.asnumpy(geom.volume).astype(np.float64)
    dvec = cp.asnumpy(geom.dvec).astype(np.float64)
    velocity_metrics = _velocity_errors(np.asarray(result["U"]), U_ref, volume)
    physical_velocity_metrics: dict[str, float] = {}
    component_map: dict[str, str] = {}
    cross_flow_contract = input_record.get("cross_flow_coordinate_contract")
    if isinstance(cross_flow_contract, dict):
        component_map = {
            str(key): str(value)
            for key, value in cross_flow_contract[
                "target_solver_to_physical_component_map"
            ].items()
        }
        physical_velocity_metrics = {
            f"e_u_physical_{physical_axis}_percent": velocity_metrics[
                f"e_u_{solver_axis}_percent"
            ]
            for solver_axis, physical_axis in component_map.items()
        }
    reference_metadata = _reference_metadata(Path(args.reference_npz))
    e_phi = 100.0 * float(
        ns["pvfv_flux_rel_error"](
            reference_geom,
            reference_result,
            geom,
            {"phi": cp.asarray(result["phi"]), "K_eff_x": 0.0},
        )
    )
    mean_flux = np.sum(np.asarray(result["phi"])[:, None] * dvec, axis=0) / np.sum(volume)
    mean_velocity = np.sum(volume[:, None] * np.asarray(result["U"]), axis=0) / np.sum(volume)
    force_x = float(args.body_force[0])
    if force_x == 0.0:
        K_eff_x = float("nan")
        e_K = float("nan")
    else:
        K_eff_x = float(cfg.nu) * float(mean_flux[0]) / force_x
        e_K = 100.0 * abs(K_eff_x - float(reference_result["K_eff_x"])) / abs(
            float(reference_result["K_eff_x"])
        )
    cached_median = float(statistics.median(cached_times))
    initial_build = (
        geometry_build_time
        + trace_geometry_time
        + float(system.assembly_time_s)
        + float(factorization.factorization_time_s)
    )

    row: dict[str, Any] = {
        "case": str(args.case),
        "site_mode": str(input_record["site_mode"]),
        "site_origin": str(input_record["site_origin"]),
        "validation_role": str(input_record["validation_role"]),
        "site_flow_axis": str(input_record["site_flow_axis"]),
        "evaluation_flow_axis": str(input_record["evaluation_flow_axis"]),
        "site_evaluation_axes_distinct": bool(
            input_record["site_evaluation_axes_distinct"]
        ),
        "particle_site_count": int(input_record.get("particle_site_count", 0)),
        "auxiliary_site_count": int(input_record.get("auxiliary_retained_count", 0)),
        "solver_formulation_id": str(result["solver_formulation_id"]),
        "linear_solver": str(result["linear_solver"]),
        "factorization_ordering": str(result["factorization_ordering"]),
        "linear_solver_iterations": int(result["linear_solver_iterations"]),
        "linear_solver_refinement_steps": int(
            result["linear_solver_refinement_steps"]
        ),
        "linear_solver_info": int(result["linear_solver_info"]),
        "linear_solver_relative_residual": float(
            result["linear_solver_relative_residual"]
        ),
        "linear_solver_rtol": float(result["linear_solver_rtol"]),
        "linear_solver_maxiter": int(result["linear_solver_maxiter"]),
        "pressure_gauge": str(result["pressure_gauge"]),
        "trace_basis": str(args.trace_basis),
        "viscous_form": str(args.viscous_form),
        "cell_velocity_definition": "constant_flux_first_moment",
        "interior_velocity_dofs_per_cell": 0,
        "cell_pressure_dofs_per_cell": 1,
        "N_cv": int(geom.n_cells),
        "N_edges": int(geom.owner.size),
        "N_interface_facelets": int(trace.n_facelets),
        "N_connected_patches": int(trace.n_patches),
        "N_trace_modes": int(trace.n_trace_modes),
        "N_trace_vector_dofs": int(result["n_trace_dofs"]),
        "K_eff_x": K_eff_x,
        "K_ref_x": float(reference_result["K_eff_x"]),
        "K_eff_parallel": K_eff_x,
        "K_ref_parallel": float(reference_result["K_eff_x"]),
        "e_K_percent": e_K,
        "e_phi_percent": e_phi,
        **velocity_metrics,
        **physical_velocity_metrics,
        "e_u_parallel_percent": float(velocity_metrics["e_u_x_percent"]),
        "mass_inf_per_volume": float(result["mass_inf_per_volume"]),
        "momentum_residual_inf": float(result["momentum_residual_inf"]),
        "velocity_matrix_symmetry_defect": float(result["velocity_matrix_symmetry_defect"]),
        "mean_velocity_x": float(mean_velocity[0]),
        "mean_flux_x": float(mean_flux[0]),
        "mean_velocity_parallel": float(mean_velocity[0]),
        "mean_flux_parallel": float(mean_flux[0]),
        "mean_velocity_flux_relative_mismatch": float(
            np.linalg.norm(mean_velocity - mean_flux)
            / max(np.linalg.norm(mean_velocity), 1.0e-300)
        ),
        "t_geometry_build_s": geometry_build_time,
        "t_trace_geometry_s": trace_geometry_time,
        "t_matrix_assembly_s": float(system.assembly_time_s),
        "t_factorization_s": float(factorization.factorization_time_s),
        "t_initial_build_s": initial_build,
        "t_cached_call_median_s": cached_median,
        "t_cached_call_min_s": min(cached_times),
        "factor_nnz": int(result["factor_nnz"]),
        "factor_memory_MiB": float(result["factor_bytes"]) / float(2**20),
        "baseline_solve_s": float(args.baseline_solve_s),
        "reference_total_s": float(args.reference_total_s),
        "cached_speedup_vs_baseline": (
            float(args.baseline_solve_s) / cached_median
            if np.isfinite(float(args.baseline_solve_s))
            else float("nan")
        ),
        "cached_speedup_vs_reference": (
            float(args.reference_total_s) / cached_median
            if np.isfinite(float(args.reference_total_s))
            else float("nan")
        ),
        "repeat_count": repeats,
        "reference_steady_converged": reference_metadata.get("steady_converged"),
        "reference_steps_completed_cumulative": reference_metadata.get(
            "steps_completed_cumulative"
        ),
        "reference_final_rel_dU": reference_metadata.get("final_rel_dU"),
        "reference_parent_to_child_K_rel_change_percent": reference_metadata.get(
            "comparison_parent_to_child", {}
        ).get("K_rel_change_percent"),
    }
    code_paths = {
        "run_hybrid_voronoi_forward.py": Path(__file__).resolve(),
        "hybrid_voronoi_trace.py": CODE_DIR / "hybrid_voronoi_trace.py",
        "run_segmented_selector_geodesic_rows.py": (
            CODE_DIR / "run_segmented_selector_geodesic_rows.py"
        ),
        "flow_runner.py": CODE_DIR / "flow_runner.py",
        "geodesic_face_operator.py": CODE_DIR / "geodesic_face_operator.py",
        "roi_jfa_backend.py": CODE_DIR / "roi_jfa_backend.py",
        "hybrid_site_sources.py": CODE_DIR / "hybrid_site_sources.py",
        "flow_notebook": PACKAGE
        / "notebooks"
        / "flow_solver.ipynb",
    }
    input_paths = {
        "mask_npz": Path(str(input_record["mask_npz"])),
        "reference_npz": Path(args.reference_npz),
    }
    input_paths["particle_window"] = Path(args.particle_window)
    if args.particle_window_provenance is not None:
        input_paths["particle_window_provenance"] = Path(
            args.particle_window_provenance
        )
    manifest = {
        "source_root": str(PACKAGE.resolve()),
        "code_sha256": {
            name: _file_sha256(path) for name, path in code_paths.items()
        },
        "input_sha256": {
            name: _file_sha256(path) for name, path in input_paths.items()
        },
        "input": input_record,
        "row": row,
        "cached_call_times_s": cached_times,
        "reference_npz": str(Path(args.reference_npz)),
        "reference_metadata": reference_metadata,
        "solver_to_physical_component_map": component_map,
        "method_invariants": {
            "cell_velocity_is_constant": True,
            "cell_interior_subvoxel_velocity_field": False,
            "interface_trace_is_single_valued": True,
            "solid_wall_trace": "zero",
            "stabilization_length": "voxel_size",
            "stabilization_multiplier": "canonical_viscous_form_coefficient",
            "reference_tuned_coefficients": False,
            "site_source_is_explicit": True,
            "particle_sites_are_never_generated_by_stride": True,
            "particle_sites_are_flow_or_measurement_derived": True,
            "particle_mode_allows_auxiliary_sites": False,
            "production_runner_accepts_only_particle_window": True,
            "zero_snap_displacement_required": True,
            "main_paper_site_mode": "particle_window",
        },
    }
    return row, {"manifest": manifest, "result": result, "U_ref": U_ref, "volume": volume}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the constant-cell-velocity hybrid Voronoi trace formulation."
    )
    parser.add_argument("--case", default="bentheimer_sandstone_crop")
    parser.add_argument("--mask-npz", type=Path)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--site-mode",
        default="particle_window",
        choices=["particle_window"],
        help="The paper runner accepts only flow-derived PTV trajectory windows.",
    )
    parser.add_argument("--particle-window", type=Path, required=True)
    parser.add_argument("--particle-window-provenance", type=Path)
    parser.add_argument("--site-flow-axis", choices=["x", "y", "z"], default="")
    parser.add_argument(
        "--evaluation-flow-axis", choices=["x", "y", "z"], default=""
    )
    parser.add_argument(
        "--particle-frames",
        default="",
        help="Optional comma-separated ids or inclusive ranges such as 0:69; empty means all rows.",
    )
    parser.add_argument(
        "--particle-ids",
        default="",
        help="Optional nested particle-id selection such as 0:499; empty means all particles.",
    )
    parser.add_argument(
        "--label-backend",
        default="exact_frontier_gpu",
        choices=["exact_frontier_gpu", "roi_jfa"],
    )
    parser.add_argument(
        "--trace-basis",
        default="connected_p1",
        choices=["connected_p0", "connected_p1", "facelet_p0"],
    )
    parser.add_argument(
        "--viscous-form",
        default="symmetric_gradient",
        choices=["symmetric_gradient", "full_gradient"],
    )
    parser.add_argument("--body-force", type=_vector, default=np.asarray([0.002, 0.0, 0.0]))
    parser.add_argument(
        "--linear-solver",
        default="direct_lu",
        choices=["direct_lu", "minres", "schur_cg"],
    )
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-maxiter", type=int, default=20000)
    parser.add_argument("--linear-refinement-steps", type=int, default=3)
    parser.add_argument(
        "--velocity-lu-ordering",
        default="COLAMD",
        choices=["COLAMD", "MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-solve-s", type=float, default=float("nan"))
    parser.add_argument("--reference-total-s", type=float, default=float("nan"))
    parser.add_argument("--export-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row, artifacts = run(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case}__{args.trace_basis}__{args.viscous_form}"
    _write_csv(out_dir / "hybrid_voronoi_forward.csv", row)
    _write_json(out_dir / "hybrid_voronoi_forward_manifest.json", artifacts["manifest"])
    if args.export_state:
        result = artifacts["result"]
        np.savez_compressed(
            out_dir / f"{stem}__state.npz",
            U=np.asarray(result["U"], dtype=np.float64),
            U_ref=np.asarray(artifacts["U_ref"], dtype=np.float64),
            p=np.asarray(result["p"], dtype=np.float64),
            phi=np.asarray(result["phi"], dtype=np.float64),
            volume=np.asarray(artifacts["volume"], dtype=np.float64),
            face_velocity=np.asarray(result["face_velocity"], dtype=np.float64),
            face_flux_sorted=np.asarray(result["face_flux_sorted"], dtype=np.float64),
            trace_coefficients=np.asarray(result["trace_coefficients"], dtype=np.float64),
            solver_formulation_id=np.asarray(result["solver_formulation_id"]),
            linear_solver=np.asarray(result["linear_solver"]),
            linear_solver_relative_residual=np.asarray(
                result["linear_solver_relative_residual"], dtype=np.float64
            ),
        )
    print(json.dumps(row, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
