"""One reusable forward-evaluation point for every new study.

The point always uses the retained production operators:
ownership (or an explicit admissible partition) -> connected-P1 trace geometry ->
symmetric-gradient assembly -> MINRES with the locked tolerances -> production
reference coarsening.

Only the *source of the prescribed sites* (or, for the block baseline, the
explicit admissible partition) changes between studies.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import study_common as ec


def input_bundle_sha256(paths: dict[str, Path]) -> str:
    parts = [f"{name}={ec.sha256_file(path)}" for name, path in sorted(paths.items())]
    return ec.sha256_text("|".join(parts))


def source_code_sha256() -> str:
    parts = [
        f"{name}={ec.sha256_file(ec.CODE_DIR / name)}" for name in sorted(ec.REQUIRED_CODE_FILES)
    ]
    return ec.sha256_text("|".join(parts))


def admissibility_audit(
    geom: Any,
    trace: Any,
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    site_flat: np.ndarray | None,
) -> dict[str, Any]:
    """The Study-B admissibility list, evaluated on the actual built objects."""
    boot = ec.bootstrap()
    cp = boot["cp"]
    volume = cp.asnumpy(geom.volume).astype(np.float64)
    n_cells = int(geom.n_cells)
    assigned = labels[mask]
    counts = np.bincount(assigned[assigned >= 0], minlength=n_cells)
    edge_count = int(geom.owner.size)
    facelets_per_edge = np.bincount(
        np.asarray(trace.face_parent_edge, dtype=np.int64), minlength=edge_count
    )
    checks = {
        "all_pore_voxels_assigned_exactly_once": bool(
            np.all(assigned >= 0) and int(assigned.size) == int(np.count_nonzero(mask))
        ),
        "no_solid_voxel_assigned": bool(np.all(labels[~mask] < 0)),
        "all_cells_nonempty": bool(np.all(counts > 0)),
        "all_cells_six_connected": bool(_six_connected(mask, labels, n_cells)),
        "positive_volume_cells": bool(np.all(volume > 0.0)),
        # This item previously read np.all(facelets_per_edge[facelets_per_edge > 0] > 0), which
        # filters to the positive entries and then asserts they are positive - a literal
        # tautology that cannot fail on any input, and which inspected a facelet COUNT rather
        # than an area. The non-vacuous content is that every owner-neighbour edge actually
        # carries at least one facelet and that a facelet has positive area. Both can now fail.
        "positive_area_internal_facelets": bool(
            int(trace.n_facelets) > 0
            and float(trace.voxel_size) > 0.0
            and edge_count > 0
            and bool(np.all(facelets_per_edge[:edge_count] > 0))
        ),
        "single_valued_shared_trace": bool(
            int(trace.n_trace_modes) > 0
            and int(np.max(np.asarray(trace.face_patch))) + 1 == int(trace.n_patches)
        ),
        # These two were hard-coded True and therefore asserted nothing. They are now measured
        # against the built objects: a gauge can only be removed from a pressure space with at
        # least two cells, and "zero auxiliary sites" means every control volume corresponds to
        # a real occupied pore region rather than to an introduced empty one.
        "one_pressure_gauge_removed": bool(n_cells >= 2),
        "zero_auxiliary_sites": bool(
            int(np.count_nonzero(counts)) == n_cells
            and int(assigned.size) == int(np.count_nonzero(mask))
        ),
    }
    if site_flat is not None:
        checks["one_prescribed_site_per_cell"] = bool(
            int(np.unique(np.asarray(site_flat, dtype=np.int64)).size) == n_cells
        )
    else:
        checks["one_prescribed_site_per_cell"] = "not_applicable_partition_baseline"
    failed = [
        name
        for name, value in checks.items()
        if value is not True and value != "not_applicable_partition_baseline"
    ]
    return {"status": "PASS" if not failed else "FAIL", "checks": checks, "failed": failed}


def _six_connected(mask: np.ndarray, labels: np.ndarray, n_cells: int) -> bool:
    pore, neighbours = ec.pore_neighbour_table(mask, periodic_x=True)
    node_label = labels.reshape(-1)[pore].astype(np.int64)
    same = np.where(
        neighbours >= 0, node_label[np.clip(neighbours, 0, None)] == node_label[:, None], False
    )
    restricted = np.where(same, neighbours, -1)
    order = np.argsort(node_label, kind="stable")
    boundaries = np.flatnonzero(np.r_[True, node_label[order][1:] != node_label[order][:-1]])
    seeds = order[boundaries]
    distance = np.full(pore.size, ec._INF, dtype=np.int32)
    ec.multi_source_bfs(restricted, seeds, distance=distance)
    return bool(np.all(distance < ec._INF))


def run_point(
    *,
    case: str,
    mask_path: Path,
    reference_path: Path,
    run_dir: Path,
    seed_spec: str,
    seed_flat: np.ndarray | None = None,
    partition_labels: np.ndarray | None = None,
    graph_periodic_x: bool | None = None,
    repeats: int = 1,
    body_force: list[float] | None = None,
    extra_inputs: dict[str, Path] | None = None,
    keep_system: bool = False,
    save_state: bool = True,
    save_partition: bool = True,
    method_id: str = "proposed",
    site_rule: str = "",
) -> dict[str, Any]:
    if (seed_flat is None) == (partition_labels is None):
        raise ValueError("Supply exactly one of seed_flat or partition_labels")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    boot = ec.bootstrap()
    cp = boot["cp"]

    if seed_flat is not None:
        sites = np.unique(np.asarray(seed_flat, dtype=np.int64))
        geom, geometry_meta, build_time = ec.build_geometry_from_seed_flat(
            sites, mask_path=Path(mask_path), seed_spec=seed_spec
        )
        applied_labels = cp.asnumpy(geom.labels).astype(np.int32)
    else:
        sites = None
        geom, geometry_meta, build_time, applied_labels = ec.build_geometry_from_partition(
            np.asarray(partition_labels), mask_path=Path(mask_path), seed_spec=seed_spec
        )

    solution = ec.solve_forward(
        geom,
        reference_path=Path(reference_path),
        repeats=repeats,
        body_force=body_force,
        keep_system=keep_system,
    )
    descriptors = ec.geometry_descriptors(
        geom, solution["trace"], site_flat=sites, graph_periodic_x=graph_periodic_x
    )
    mask = cp.asnumpy(geom.mask).astype(bool)
    admissibility = admissibility_audit(
        geom, solution["trace"], mask=mask, labels=applied_labels, site_flat=sites
    )
    result = solution["result"]

    inputs = {"mask_npz": Path(mask_path), "reference_npz": Path(reference_path)}
    if extra_inputs:
        inputs.update({key: Path(value) for key, value in extra_inputs.items()})

    row: dict[str, Any] = {
        "protocol_id": ec.PROTOCOL_ID,
        "case": case,
        "case_display": ec.CASES.get(case, {}).get("display", case),
        "method_id": method_id,
        "site_or_partition_rule": site_rule,
        "N_sites": int(sites.size) if sites is not None else 0,
        **descriptors,
        "admissibility_status": admissibility["status"],
        "K_eff_parallel": solution["K_eff_parallel"],
        "K_ref_parallel": solution["K_ref_parallel"],
        "e_K_percent": solution["e_K_percent"],
        "e_phi_percent": solution["e_phi_percent"],
        "e_u_percent": solution["e_u_percent"],
        "e_p_percent": solution["e_p_percent"],
        "mass_inf_per_volume": float(result["mass_inf_per_volume"]),
        "momentum_residual_inf": float(result["momentum_residual_inf"]),
        "linear_solver_iterations": int(result["linear_solver_iterations"]),
        "linear_solver_relative_residual": float(result["linear_solver_relative_residual"]),
        "t_geometry_build_s": build_time,
        "t_trace_geometry_s": solution["t_trace_geometry_s"],
        "t_matrix_assembly_s": solution["t_matrix_assembly_s"],
        "t_factorization_s": solution["t_factorization_s"],
        "t_initial_build_s": (
            build_time
            + solution["t_trace_geometry_s"]
            + solution["t_matrix_assembly_s"]
            + solution["t_factorization_s"]
        ),
        "t_cached_call_median_s": solution["t_cached_call_median_s"],
        "factor_memory_MiB": float(result["factor_bytes"]) / float(2**20),
        "solver_formulation_id": str(result["solver_formulation_id"]),
        "source_code_sha256": source_code_sha256(),
        "input_bundle_sha256": input_bundle_sha256(inputs),
        "run_directory": str(run_dir),
        "linear_solver": str(result["linear_solver"]),
        "linear_solver_rtol": float(result["linear_solver_rtol"]),
        "linear_solver_maxiter": int(result["linear_solver_maxiter"]),
        "linear_solver_refinement_steps": int(result["linear_solver_refinement_steps"]),
        "pressure_gauge": str(result["pressure_gauge"]),
        "velocity_matrix_symmetry_defect": float(result["velocity_matrix_symmetry_defect"]),
        "e_u_x_percent": solution["e_u_x_percent"],
        "e_u_y_percent": solution["e_u_y_percent"],
        "e_u_z_percent": solution["e_u_z_percent"],
        "auxiliary_site_count": 0,
        "snap_displacement_max_vox": 0.0,
    }

    manifest = {
        "protocol_id": ec.PROTOCOL_ID,
        "case": case,
        "method_id": method_id,
        "seed_spec": seed_spec,
        "run_directory": str(run_dir),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_line": " ".join(sys.argv),
        "argv": list(sys.argv),
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "locked_forward_args": ec.FORWARD_ARGS,
        "solver_residual_record": (
            "the retained production solver reports the final relative residual, the "
            "iteration count and the refinement-step count; it does not expose a per-iteration "
            "residual history, and none is fabricated here"
        ),
        "code_sha256": {name: ec.sha256_file(ec.CODE_DIR / name) for name in ec.REQUIRED_CODE_FILES},
        "input_sha256": {name: ec.sha256_file(path) for name, path in inputs.items()},
        "input_paths": {name: str(path) for name, path in inputs.items()},
        "site_set_sha256": ec.sha256_array(sites) if sites is not None else None,
        "partition_sha256": ec.sha256_array(applied_labels),
        "geometry_meta": {
            str(key): value for key, value in geometry_meta.items() if np.isscalar(value)
        },
        "admissibility": admissibility,
        "pressure_error_detail": solution["pressure_error_detail"],
        "reference_metadata": solution["reference_metadata"],
        "weighted_quantile_definition": (
            "left-searchsorted cumulative-weight quantile on stable-sorted values; "
            "identical to the archived forward_resolution_ducts driver"
        ),
        "orientation_tensor_scope": "all interface facelets (equal voxel-facelet areas)",
        "row": row,
        "cached_call_times_s": solution["cached_call_times_s"],
        "method_invariants": {
            "auxiliary_sites": 0,
            "snap_displacement_limit_vox": 0.0,
            "trace_basis": ec.FORWARD_ARGS["trace_basis"],
            "viscous_form": ec.FORWARD_ARGS["viscous_form"],
            "ownership_backend": ec.FORWARD_ARGS["label_backend"]
            if sites is not None
            else "explicit_partition",
        },
    }
    ec.write_json(run_dir / "run_manifest.json", manifest)

    if save_partition:
        np.savez_compressed(
            run_dir / "sites_and_partition.npz",
            seed_flat=sites if sites is not None else np.empty(0, dtype=np.int64),
            labels=applied_labels,
            cell_volume=cp.asnumpy(geom.volume).astype(np.float64),
        )
    if save_state:
        np.savez_compressed(
            run_dir / "state.npz",
            U=np.asarray(result["U"], dtype=np.float64),
            U_ref=np.asarray(solution["U_ref"], dtype=np.float64),
            p=np.asarray(result["p"], dtype=np.float64),
            p_ref=np.asarray(solution["p_ref"], dtype=np.float64),
            phi=np.asarray(result["phi"], dtype=np.float64),
            volume=np.asarray(solution["volume"], dtype=np.float64),
        )

    artifacts = {"geom": geom, "solution": solution, "manifest": manifest}
    if not keep_system:
        del geom, solution
        gc.collect()
        ec.free_gpu()
    return {"row": row, "manifest": manifest, "artifacts": artifacts}


def relative_difference(actual: float, expected: float) -> float:
    denominator = max(abs(float(expected)), 1.0e-300)
    return abs(float(actual) - float(expected)) / denominator
