from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
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
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_segmented_selector_geodesic_rows as segmented  # noqa: E402
from bentheimer_reference_adapter import (  # noqa: E402
    aggregate_dns_cell_state_to_ownership,
    aggregate_dns_face_flux_to_ownership,
    load_dns_metrics,
)
from hybrid_site_sources import (  # noqa: E402
    harmonic_complete_constant_cell_states,
    load_particle_window_cell_states,
    load_particle_window_seed_flat,
)
from hybrid_voronoi_trace import (  # noqa: E402
    build_hybrid_trace_geometry,
    build_hybrid_trace_projection_factorization,
    build_hybrid_trace_projection_operators,
    project_constant_cell_states_to_conservative_trace,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "particle_transfer"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(text: str) -> np.ndarray:
    values = np.asarray([float(part.strip()) for part in text.split(",")], dtype=np.float64)
    if values.size != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated numbers")
    return values


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


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


def _velocity_errors(velocity: np.ndarray, reference: np.ndarray, volume: np.ndarray, prefix: str) -> dict[str, float]:
    numerator = float(np.sum(volume[:, None] * (velocity - reference) ** 2))
    denominator = max(float(np.sum(volume[:, None] * reference**2)), 1.0e-300)
    result = {f"{prefix}_percent": 100.0 * np.sqrt(numerator / denominator)}
    for component, index in (("x", 0), ("y", 1), ("z", 2)):
        component_num = float(np.sum(volume * (velocity[:, index] - reference[:, index]) ** 2))
        component_den = max(float(np.sum(volume * reference[:, index] ** 2)), 1.0e-300)
        result[f"{prefix}_{component}_percent"] = 100.0 * np.sqrt(component_num / component_den)
    return result


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    total_start = time.perf_counter()
    segmented.configure_roi_environment()
    os.environ["PVFV_LABEL_BACKEND"] = str(args.label_backend)
    runner = segmented.load_runner()
    runner_args = _runner_args(args)
    ns, _fixed_wall_clone, _global_wall_clone = segmented.install_namespace(runner, runner_args)
    cp = ns["cp"]
    cfg = segmented.make_cfg(runner, ns, Path(args.out_dir), runner_args)
    cfg.body_force = tuple(float(value) for value in args.body_force)

    mask_source = np.load(Path(args.mask_npz), allow_pickle=True)
    mask_np_loaded = (
        np.asarray(mask_source, dtype=bool)
        if isinstance(mask_source, np.ndarray)
        else np.asarray(mask_source["mask"], dtype=bool)
    )
    if not isinstance(mask_source, np.ndarray):
        mask_source.close()
    mask = cp.asarray(mask_np_loaded)
    mask_np = cp.asnumpy(mask).astype(bool, copy=False)
    site_load_start = time.perf_counter()
    particle_seed_flat, site_metadata = load_particle_window_seed_flat(
        mask_np,
        Path(args.particle_window),
        frame_selection=str(args.particle_frames),
        particle_id_selection=str(args.particle_ids),
        max_snap_displacement_vox=float(args.max_site_snap_vox),
        x_boundary_mode=str(args.x_boundary_mode),
        coordinate_center_offset_vox=float(args.coordinate_center_offset_vox),
    )
    site_load_time = float(time.perf_counter() - site_load_start)
    print(
        f"[TRACE-TRANSFER] sites={particle_seed_flat.size} load={site_load_time:.3f}s",
        flush=True,
    )
    seed_flat = cp.asarray(particle_seed_flat, dtype=cp.int64)
    geometry_start = time.perf_counter()
    geom, geometry_metadata = ns["pvfv_build_geometry_from_seed_flat_timed"](
        mask,
        seed_flat,
        cfg,
        split_face_components=True,
        label_mode=str(args.label_backend),
        seed_spec={
            "stage": "particle_trace_transfer",
            "family": "flow_derived_particle_trajectory",
            "seed_id": f"particle_window_{Path(args.particle_window).stem}",
        },
    )
    geometry_time = float(time.perf_counter() - geometry_start)
    print(f"[TRACE-TRANSFER] geometry={geometry_time:.3f}s cells={int(geom.n_cells)}", flush=True)
    labels_np = cp.asnumpy(geom.labels).astype(np.int32, copy=False)
    state_window = (
        Path(args.state_window) if args.state_window is not None else Path(args.particle_window)
    )
    state_frames = str(args.state_frames).strip() or str(args.particle_frames)
    state_load_start = time.perf_counter()
    measured, state_velocity, state_counts, state_metadata = load_particle_window_cell_states(
        mask_np,
        labels_np,
        state_window,
        frame_selection=state_frames,
        particle_id_selection=str(args.state_particle_ids),
        chunk_rows=int(args.state_chunk_rows),
        max_snap_displacement_vox=float(args.max_state_snap_vox),
        x_boundary_mode=str(args.x_boundary_mode),
        coordinate_center_offset_vox=float(args.coordinate_center_offset_vox),
    )
    state_load_time = float(time.perf_counter() - state_load_start)
    print(
        "[TRACE-TRANSFER] "
        f"state_load={state_load_time:.3f}s measured={int(np.count_nonzero(measured))}/{measured.size}",
        flush=True,
    )
    trace_start = time.perf_counter()
    trace_geometry_start = time.perf_counter()
    trace = build_hybrid_trace_geometry(ns, geom, cfg, trace_basis=str(args.trace_basis))
    trace_geometry_time = float(time.perf_counter() - trace_geometry_start)
    print(
        "[TRACE-TRANSFER] "
        f"trace_geometry={trace_geometry_time:.3f}s facelets={trace.n_facelets} patches={trace.n_patches}",
        flush=True,
    )
    face_conductance = (trace.voxel_size * trace.voxel_size) / np.maximum(
        trace.face_r_owner_g + trace.face_r_neigh_g,
        1.0e-300,
    )
    edge_conductance = np.bincount(
        trace.face_parent_edge,
        weights=face_conductance,
        minlength=int(geom.owner.size),
    )
    harmonic_start = time.perf_counter()
    state_velocity, completion_metadata = harmonic_complete_constant_cell_states(
        cp.asnumpy(geom.owner),
        cp.asnumpy(geom.neigh),
        edge_conductance,
        measured,
        state_velocity,
    )
    harmonic_time = float(time.perf_counter() - harmonic_start)
    print(f"[TRACE-TRANSFER] harmonic_completion={harmonic_time:.3f}s", flush=True)
    operators = build_hybrid_trace_projection_operators(trace)
    factorization = build_hybrid_trace_projection_factorization(
        operators,
        preserve_mean_components=(0,) if args.throughflow_constraint != "none" else (),
        solver=str(args.projection_solver),
        direct_cell_limit=int(args.direct_cell_limit),
        iterative_rtol=float(args.projection_rtol),
        iterative_maxiter=int(args.projection_maxiter),
    )
    trace_build_time = float(time.perf_counter() - trace_start)
    print(
        "[TRACE-TRANSFER] "
        f"operator={operators.assembly_time_s:.3f}s preparation={factorization.factorization_time_s:.3f}s",
        flush=True,
    )
    repeats = max(int(args.repeats), 1)
    results = [
        project_constant_cell_states_to_conservative_trace(
            operators,
            state_velocity,
            factorization=factorization,
            preserved_mean_target=str(args.throughflow_constraint),
        )
        for _ in range(repeats)
    ]
    result = results[-1]
    cached_times = [float(item["cached_call_time_s"]) for item in results]
    print(
        f"[TRACE-TRANSFER] projections={repeats} median={statistics.median(cached_times):.3f}s",
        flush=True,
    )

    reference_start = time.perf_counter()
    volume = cp.asnumpy(geom.volume).astype(np.float64)
    dvec = cp.asnumpy(geom.dvec).astype(np.float64)
    reference_metadata: dict[str, Any]
    if str(args.reference_mode) == "npz":
        if args.reference_npz is None:
            raise ValueError("--reference-npz is required when --reference-mode=npz")
        reference_geom, reference_result = segmented.load_reference_npz(
            ns,
            Path(args.reference_npz),
            cfg,
        )
        if reference_geom is None or reference_result is None:
            raise FileNotFoundError(args.reference_npz)
        reference_velocity_gpu, _reference_pressure_gpu = ns["coarsen_voxel_reference_to_coarse_gpu"](
            reference_geom,
            reference_result,
            geom,
        )
        reference_velocity = cp.asnumpy(reference_velocity_gpu).astype(np.float64)
        e_phi = 100.0 * float(
            ns["pvfv_flux_rel_error"](
                reference_geom,
                reference_result,
                geom,
                {"phi": cp.asarray(result["phi"]), "K_eff_x": 0.0},
            )
        )
        e_phi_target_raw = 100.0 * float(
            ns["pvfv_flux_rel_error"](
                reference_geom,
                reference_result,
                geom,
                {"phi": cp.asarray(result["phi_target_raw"]), "K_eff_x": 0.0},
            )
        )
        e_phi_unconstrained_trace = 100.0 * float(
            ns["pvfv_flux_rel_error"](
                reference_geom,
                reference_result,
                geom,
                {"phi": cp.asarray(result["phi_unconstrained_trace"]), "K_eff_x": 0.0},
            )
        )
        reference_permeability = float(reference_result["K_eff_x"])
        reference_metadata = {
            "reference_mode": "voxel_reference_npz",
            "reference_npz": str(Path(args.reference_npz)),
        }
    elif str(args.reference_mode) == "bentheimer_dns":
        required_paths = {
            "dns_cell_csv": args.dns_cell_csv,
            "dns_face_csv": args.dns_face_csv,
            "dns_metrics_csv": args.dns_metrics_csv,
        }
        missing_paths = [name for name, value in required_paths.items() if value is None]
        if missing_paths:
            raise ValueError(
                "Bentheimer DNS reference mode requires paths for: " + ", ".join(missing_paths)
            )
        dns_metrics = load_dns_metrics(Path(args.dns_metrics_csv))
        n_dns_cells = int(dns_metrics["dns_cells"])
        reference_velocity, _reference_pressure, dns_cell_to_owner, cell_metadata = (
            aggregate_dns_cell_state_to_ownership(
                labels_np,
                Path(args.dns_cell_csv),
                n_cells=int(geom.n_cells),
                n_dns_cells=n_dns_cells,
                chunk_rows=int(args.dns_cell_chunk_rows),
            )
        )
        reference_flux, flux_metadata = aggregate_dns_face_flux_to_ownership(
            dns_cell_to_owner,
            cp.asnumpy(geom.owner),
            cp.asnumpy(geom.neigh),
            Path(args.dns_face_csv),
            n_cells=int(geom.n_cells),
            chunk_rows=int(args.dns_face_chunk_rows),
        )
        flux_difference = np.asarray(result["phi"], dtype=np.float64) - reference_flux
        e_phi = 100.0 * float(
            np.sqrt(
                np.sum(flux_difference * flux_difference)
                / max(np.sum(reference_flux * reference_flux), 1.0e-300)
            )
        )
        target_flux_difference = (
            np.asarray(result["phi_target_raw"], dtype=np.float64) - reference_flux
        )
        e_phi_target_raw = 100.0 * float(
            np.sqrt(
                np.sum(target_flux_difference * target_flux_difference)
                / max(np.sum(reference_flux * reference_flux), 1.0e-300)
            )
        )
        unconstrained_flux_difference = (
            np.asarray(result["phi_unconstrained_trace"], dtype=np.float64) - reference_flux
        )
        e_phi_unconstrained_trace = 100.0 * float(
            np.sqrt(
                np.sum(unconstrained_flux_difference * unconstrained_flux_difference)
                / max(np.sum(reference_flux * reference_flux), 1.0e-300)
            )
        )
        reference_permeability = float(dns_metrics["K_dns_pore_mean"])
        reference_result = {"K_eff_x": reference_permeability}
        reference_metadata = {
            "reference_mode": "bentheimer_raw_dns_csv",
            "dns_metrics_csv": str(Path(args.dns_metrics_csv)),
            "dns_reference_K_definition": "K_dns_pore_mean",
            **cell_metadata,
            **flux_metadata,
        }
    else:
        raise ValueError(f"Unknown reference mode: {args.reference_mode!r}")
    reference_time = float(time.perf_counter() - reference_start)
    print(f"[TRACE-TRANSFER] reference_aggregation={reference_time:.3f}s", flush=True)

    velocity_metrics = _velocity_errors(np.asarray(result["U"]), reference_velocity, volume, "e_u")
    state_metrics = _velocity_errors(state_velocity, reference_velocity, volume, "e_u_state")
    trace_moment_metrics = _velocity_errors(
        np.asarray(result["U_trace_moment"]),
        reference_velocity,
        volume,
        "e_u_trace_moment",
    )
    mean_flux = np.sum(np.asarray(result["phi"])[:, None] * dvec, axis=0) / np.sum(volume)
    mean_target_flux = (
        np.sum(np.asarray(result["phi_target_raw"])[:, None] * dvec, axis=0) / np.sum(volume)
    )
    mean_unconstrained_trace_flux = (
        np.sum(np.asarray(result["phi_unconstrained_trace"])[:, None] * dvec, axis=0)
        / np.sum(volume)
    )
    mean_velocity = np.sum(volume[:, None] * np.asarray(result["U"]), axis=0) / np.sum(volume)
    mean_trace_moment_velocity = np.sum(
        volume[:, None] * np.asarray(result["U_trace_moment"]),
        axis=0,
    ) / np.sum(volume)
    force_x = float(args.body_force[0])
    if force_x == 0.0:
        permeability = float("nan")
        permeability_error = float("nan")
        target_permeability = float("nan")
        target_permeability_error = float("nan")
        unconstrained_trace_permeability = float("nan")
        unconstrained_trace_permeability_error = float("nan")
    else:
        permeability = float(cfg.nu) * float(mean_flux[0]) / force_x
        permeability_error = 100.0 * abs(permeability - reference_permeability) / abs(
            reference_permeability
        )
        target_permeability = float(cfg.nu) * float(mean_target_flux[0]) / force_x
        target_permeability_error = (
            100.0
            * abs(target_permeability - reference_permeability)
            / abs(reference_permeability)
        )
        unconstrained_trace_permeability = (
            float(cfg.nu) * float(mean_unconstrained_trace_flux[0]) / force_x
        )
        unconstrained_trace_permeability_error = (
            100.0
            * abs(unconstrained_trace_permeability - reference_permeability)
            / abs(reference_permeability)
        )

    row: dict[str, Any] = {
        "case": str(args.case),
        "validation_role": "flow_conditioned_sampled_state_transfer_not_independent_forward_solve",
        "reference_mode": str(reference_metadata["reference_mode"]),
        "site_origin": "flow_derived_particle_trajectory",
        "particle_sites_are_regular_lattice": False,
        "auxiliary_site_count": 0,
        "particle_site_count": int(particle_seed_flat.size),
        "N_fl": int(np.count_nonzero(mask_np)),
        "all_cells_directly_measured": bool(np.all(measured)),
        "directly_measured_cell_fraction": float(np.mean(measured)),
        "harmonic_completed_cell_count": int(completion_metadata["state_harmonic_completed_cells"]),
        "all_cells_have_constant_state_after_completion": True,
        "state_representation": "direct_particle_mean_plus_exact_graph_harmonic_constant_completion",
        "state_interior_gradient_reconstruction": "none",
        "solver_formulation_id": str(result["solver_formulation_id"]),
        "trace_basis": str(args.trace_basis),
        "cell_velocity_definition": str(result["cell_velocity_definition"]),
        "interior_velocity_dofs_per_cell": 0,
        "N_cv": int(geom.n_cells),
        "compression_ratio_Nfl_over_Ncv": float(np.count_nonzero(mask_np) / max(int(geom.n_cells), 1)),
        "N_interface_facelets": int(trace.n_facelets),
        "N_connected_patches": int(trace.n_patches),
        "N_trace_modes": int(trace.n_trace_modes),
        "N_trace_vector_dofs": int(result["n_trace_dofs"]),
        "K_eff_x": permeability,
        "K_ref_x": reference_permeability,
        "e_K_percent": permeability_error,
        "e_phi_percent": e_phi,
        "K_eff_x_target_raw": target_permeability,
        "e_K_target_raw_percent": target_permeability_error,
        "K_eff_x_unconstrained_trace": unconstrained_trace_permeability,
        "e_K_unconstrained_trace_percent": unconstrained_trace_permeability_error,
        "e_phi_target_raw_percent": e_phi_target_raw,
        "e_phi_unconstrained_trace_percent": e_phi_unconstrained_trace,
        **velocity_metrics,
        **state_metrics,
        **trace_moment_metrics,
        "mass_inf_per_volume": float(result["mass_inf_per_volume"]),
        "preserved_mean_velocity_residual_inf": float(
            result["preserved_mean_velocity_residual_inf"]
        ),
        "trace_correction_relative_l2": float(result["trace_correction_relative_l2"]),
        "mean_state_velocity_flux_relative_mismatch": float(
            np.linalg.norm(mean_velocity - mean_flux) / max(np.linalg.norm(mean_velocity), 1.0e-300)
        ),
        "mean_trace_moment_velocity_flux_relative_mismatch": float(
            np.linalg.norm(mean_trace_moment_velocity - mean_flux)
            / max(np.linalg.norm(mean_trace_moment_velocity), 1.0e-300)
        ),
        "t_geometry_build_s": geometry_time,
        "t_particle_site_load_s": site_load_time,
        "t_particle_state_load_s": state_load_time,
        "t_trace_geometry_s": trace_geometry_time,
        "t_harmonic_completion_s": harmonic_time,
        "t_trace_operator_and_factorization_s": trace_build_time,
        "t_projection_operator_assembly_s": float(operators.assembly_time_s),
        "t_projection_factorization_s": float(factorization.factorization_time_s),
        "t_cached_call_median_s": float(statistics.median(cached_times)),
        "t_cached_call_min_s": float(min(cached_times)),
        "factor_nnz": int(result["factor_nnz"]),
        "factor_memory_MiB": float(result["factor_bytes"]) / float(2**20),
        "projection_solver": str(result["projection_solver"]),
        "throughflow_constraint": str(args.throughflow_constraint),
        "throughflow_constraint_active": bool(result["throughflow_constraint_active"]),
        "preserved_mean_target": str(result["preserved_mean_target"]),
        "projection_solver_rtol": float(result["projection_solver_rtol"]),
        "projection_solver_maxiter": int(result["projection_solver_maxiter"]),
        "projection_solver_iteration_count_total": int(
            result["projection_solver_iteration_count_total"]
        ),
        "projection_solver_iteration_count_max": int(
            result["projection_solver_iteration_count_max"]
        ),
        "projection_solver_internal_relative_residual_max": float(
            result["projection_solver_internal_relative_residual_max"]
        ),
        "projection_solver_setup_iteration_count_total": int(
            result["projection_solver_setup_iteration_count_total"]
        ),
        "projection_solver_setup_iteration_count_max": int(
            result["projection_solver_setup_iteration_count_max"]
        ),
        "projection_solver_setup_internal_relative_residual_max": float(
            result["projection_solver_setup_internal_relative_residual_max"]
        ),
        "projection_system_relative_residual": float(
            result["projection_system_relative_residual"]
        ),
        "repeat_count": repeats,
        "t_reference_aggregation_s": reference_time,
        "t_total_pipeline_s": float(time.perf_counter() - total_start),
    }
    manifest = {
        "execution": {
            "utc_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "argv": [sys.executable, *sys.argv],
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "code_sha256": {
                path.name: _sha256(path)
                for path in (
                    Path(__file__),
                    CODE_DIR / "hybrid_site_sources.py",
                    CODE_DIR / "hybrid_voronoi_trace.py",
                    CODE_DIR / "geodesic_face_operator.py",
                    CODE_DIR / "bentheimer_reference_adapter.py",
                )
            },
        },
        "input": {
            "mask_npz": str(Path(args.mask_npz)),
            "reference_npz": "" if args.reference_npz is None else str(Path(args.reference_npz)),
            "particle_window": str(Path(args.particle_window)),
            "particle_frames": str(args.particle_frames) or "all",
            "particle_ids": str(args.particle_ids) or "all",
            "state_window": str(state_window),
            "state_frames": state_frames or "all",
            "state_particle_ids": str(args.state_particle_ids) or "all",
            "label_backend": str(args.label_backend),
            **site_metadata,
            **state_metadata,
            **completion_metadata,
            **reference_metadata,
            **{str(key): value for key, value in geometry_metadata.items() if np.isscalar(value)},
        },
        "row": row,
        "method_invariants": {
            "particle_sites_come_from_trajectory_coordinates": True,
            "regular_particle_placement": False,
            "mask_only_auxiliary_sites": False,
            "one_constant_velocity_per_cell": True,
            "reported_cell_velocity_is_particle_state": True,
            "projected_trace_does_not_redefine_cell_velocity": True,
            "cell_interior_linear_or_fine_velocity_field": False,
            "unmeasured_cell_completion": "exact_graph_harmonic_constant_state",
            "shared_interface_trace": True,
            "exact_cell_mass_balance": True,
            "periodic_throughflow_constraint": str(args.throughflow_constraint),
            "projection_constraints": (
                "cell_mass_balance_only"
                if args.throughflow_constraint == "none"
                else "cell_mass_balance_plus_declared_global_mean"
            ),
            "projection_objective": "facelet_area_weighted_trace_L2",
            "reference_values_used_in_projection": False,
            "independent_forward_validation": False,
        },
        "cached_call_times_s": cached_times,
    }
    artifacts = {
        "manifest": manifest,
        "result": result,
        "state_velocity": state_velocity,
        "state_counts": state_counts,
        "reference_velocity": reference_velocity,
        "volume": volume,
    }
    return row, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer flow-derived particle states through a shared conservative interface trace "
            "while retaining exactly one constant velocity per Voronoi cell."
        )
    )
    parser.add_argument("--case", default="particle_trace_transfer")
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument(
        "--reference-mode",
        choices=["npz", "bentheimer_dns"],
        default="npz",
    )
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument("--dns-cell-csv", type=Path)
    parser.add_argument("--dns-face-csv", type=Path)
    parser.add_argument("--dns-metrics-csv", type=Path)
    parser.add_argument("--dns-cell-chunk-rows", type=int, default=300_000)
    parser.add_argument("--dns-face-chunk-rows", type=int, default=750_000)
    parser.add_argument("--particle-window", type=Path, required=True)
    parser.add_argument("--particle-frames", default="")
    parser.add_argument(
        "--particle-ids",
        default="",
        help="Optional inclusive particle-id selection, for example 0:7999.",
    )
    parser.add_argument(
        "--state-window",
        type=Path,
        help="Optional denser particle-state window; defaults to the site-defining particle window.",
    )
    parser.add_argument(
        "--state-frames",
        default="",
        help="Optional state-window frame selection; defaults to --particle-frames.",
    )
    parser.add_argument(
        "--state-particle-ids",
        default="",
        help="Optional inclusive state-particle-id selection, for example 0:39999.",
    )
    parser.add_argument("--state-chunk-rows", type=int, default=250_000)
    parser.add_argument(
        "--x-boundary-mode",
        choices=["periodic_wrap", "clip"],
        default="periodic_wrap",
        help="Coordinate convention for rounded x values at the stored window boundary.",
    )
    parser.add_argument(
        "--coordinate-center-offset-vox",
        type=float,
        default=0.0,
        help="Subtract this source-coordinate center offset before mapping to zero-based voxel indices.",
    )
    parser.add_argument(
        "--max-site-snap-vox",
        type=float,
        default=0.0,
        help="Fail if a site coordinate must move farther than this to reach a pore voxel.",
    )
    parser.add_argument(
        "--max-state-snap-vox",
        type=float,
        default=0.0,
        help="Fail if a state coordinate must move farther than this to reach a pore voxel.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
    parser.add_argument("--body-force", type=_vector, default=np.asarray([0.002, 0.0, 0.0]))
    parser.add_argument(
        "--projection-solver",
        choices=["auto", "direct_lu", "projected_cg"],
        default="auto",
    )
    parser.add_argument(
        "--throughflow-constraint",
        choices=["particle_state_mean", "face_target_mean", "none"],
        default="none",
    )
    parser.add_argument("--direct-cell-limit", type=int, default=30_000)
    parser.add_argument("--projection-rtol", type=float, default=1.0e-11)
    parser.add_argument("--projection-maxiter", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--export-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row, artifacts = run(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "particle_trace_transfer.csv", row)
    (out_dir / "particle_trace_transfer_manifest.json").write_text(
        json.dumps(artifacts["manifest"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.export_state:
        result = artifacts["result"]
        np.savez_compressed(
            out_dir / "particle_trace_transfer_state.npz",
            U=np.asarray(result["U"], dtype=np.float64),
            U_trace_moment=np.asarray(result["U_trace_moment"], dtype=np.float64),
            U_state=np.asarray(artifacts["state_velocity"], dtype=np.float64),
            U_ref=np.asarray(artifacts["reference_velocity"], dtype=np.float64),
            phi=np.asarray(result["phi"], dtype=np.float64),
            phi_target_raw=np.asarray(result["phi_target_raw"], dtype=np.float64),
            phi_unconstrained_trace=np.asarray(
                result["phi_unconstrained_trace"], dtype=np.float64
            ),
            face_velocity=np.asarray(result["face_velocity"], dtype=np.float64),
            face_velocity_target=np.asarray(result["face_velocity_target"], dtype=np.float64),
            state_counts=np.asarray(artifacts["state_counts"], dtype=np.int64),
            volume=np.asarray(artifacts["volume"], dtype=np.float64),
        )
    print(json.dumps(row, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
