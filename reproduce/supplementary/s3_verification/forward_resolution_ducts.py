from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPUTE_ROOT = (
    PAPER_ROOT.parent / "../gpu"
)
DEFAULT_EVIDENCE_DIR = (
    PAPER_ROOT / "s3_verification" / "."
)
DEFAULT_RUN_ROOT = (
    DEFAULT_COMPUTE_ROOT / "../outputs" / "forward_resolution_ducts"
)

CASES: dict[str, dict[str, Any]] = {
    "orthogonal_duct": {
        "display": "Orthogonal duct",
        "run_case": "orthogonal_duct",
        "mask": "orthogonal_duct/reference_flow.npz",
        "particle": (
            "orthogonal_duct/particle_tracks.csv.gz"
        ),
        "prefixes": [90, 180, 360, 721],
        "full_expected": {
            "N_cv": 4937,
            "e_K_percent": 4.185415605160604,
            "e_phi_percent": 4.331951032597175,
            "e_u_percent": 3.8930379557346697,
        },
    },
    "skewed_duct": {
        "display": "Skewed duct",
        "run_case": "skewed_duct",
        "mask": "skewed_duct/reference_flow.npz",
        "particle": (
            "skewed_duct/particle_tracks.csv.gz"
        ),
        "prefixes": [89, 178, 356, 712],
        "full_expected": {
            "N_cv": 5268,
            "e_K_percent": 5.400464228051188,
            "e_phi_percent": 6.293405269321623,
            "e_u_percent": 5.907948363524675,
        },
    },
}

FORWARD_ARGS = {
    "label_backend": "exact_frontier_gpu",
    "trace_basis": "connected_p1",
    "viscous_form": "symmetric_gradient",
    "body_force": [0.002, 0.0, 0.0],
    "linear_solver": "minres",
    "linear_rtol": 1.0e-14,
    "linear_maxiter": 50000,
    "linear_refinement_steps": 1,
    "repeats": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one data row in {path}, found {len(rows)}")
    return rows[0]


def _number(row: dict[str, str], key: str, *, integer: bool = False) -> int | float:
    value = row[key]
    return int(float(value)) if integer else float(value)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = probability * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _run_point(
    *,
    compute_root: Path,
    run_root: Path,
    case_name: str,
    spec: dict[str, Any],
    trajectory_count: int,
    rerun: bool,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    point_dir = run_root / case_name / f"n{trajectory_count:04d}"
    csv_path = point_dir / "hybrid_voronoi_forward.csv"
    manifest_path = point_dir / "hybrid_voronoi_forward_manifest.json"
    expected_ids = f"0:{trajectory_count - 1}"

    if not rerun and csv_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest["input"]["particle_ids"]) == expected_ids:
            return point_dir, manifest, _read_csv_row(csv_path)

    point_dir.mkdir(parents=True, exist_ok=True)
    runner = compute_root / "code" / "run_hybrid_voronoi_forward.py"
    mask = compute_root / "../data/controlled_cases" / "." / spec["mask"]
    particle = (
        compute_root / "../data/controlled_cases" / spec["particle"]
    )
    command = [
        sys.executable,
        str(runner),
        "--case",
        f"{spec['run_case']}_resolution_n{trajectory_count:04d}",
        "--mask-npz",
        str(mask),
        "--reference-npz",
        str(mask),
        "--out-dir",
        str(point_dir),
        "--particle-window",
        str(particle),
        "--particle-ids",
        expected_ids,
        "--label-backend",
        str(FORWARD_ARGS["label_backend"]),
        "--trace-basis",
        str(FORWARD_ARGS["trace_basis"]),
        "--viscous-form",
        str(FORWARD_ARGS["viscous_form"]),
        "--body-force",
        ",".join(str(value) for value in FORWARD_ARGS["body_force"]),
        "--linear-solver",
        str(FORWARD_ARGS["linear_solver"]),
        "--linear-rtol",
        str(FORWARD_ARGS["linear_rtol"]),
        "--linear-maxiter",
        str(FORWARD_ARGS["linear_maxiter"]),
        "--linear-refinement-steps",
        str(FORWARD_ARGS["linear_refinement_steps"]),
        "--repeats",
        str(FORWARD_ARGS["repeats"]),
    ]
    stdout_path = point_dir / "run_stdout.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            command,
            cwd=compute_root,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Forward run failed for {case_name}/{trajectory_count}; "
            f"see {stdout_path}"
        )
    return (
        point_dir,
        json.loads(manifest_path.read_text(encoding="utf-8")),
        _read_csv_row(csv_path),
    )


def _geometry_metrics(
    *,
    compute_root: Path,
    case_name: str,
    spec: dict[str, Any],
    trajectory_count: int,
) -> dict[str, float]:
    code_dir = compute_root / "code"
    sys.path.insert(0, str(code_dir))
    try:
        import run_hybrid_voronoi_forward as forward
        from hybrid_voronoi_trace import build_hybrid_trace_geometry
    finally:
        sys.path.pop(0)

    mask = compute_root / "../data/controlled_cases" / "." / spec["mask"]
    particle = (
        compute_root / "../data/controlled_cases" / spec["particle"]
    )
    run_args = types.SimpleNamespace(
        case=f"{case_name}_resolution_geometry_n{trajectory_count:04d}",
        mask_npz=mask,
        reference_npz=mask,
        out_dir=DEFAULT_RUN_ROOT / "_geometry_scratch",
        site_mode="particle_window",
        particle_window=particle,
        particle_window_provenance=None,
        site_flow_axis="",
        evaluation_flow_axis="",
        particle_frames="",
        particle_ids=f"0:{trajectory_count - 1}",
        label_backend=FORWARD_ARGS["label_backend"],
        trace_basis=FORWARD_ARGS["trace_basis"],
        viscous_form=FORWARD_ARGS["viscous_form"],
        body_force=np.asarray(FORWARD_ARGS["body_force"], dtype=np.float64),
        linear_solver=FORWARD_ARGS["linear_solver"],
        linear_rtol=FORWARD_ARGS["linear_rtol"],
        linear_maxiter=FORWARD_ARGS["linear_maxiter"],
        linear_refinement_steps=FORWARD_ARGS["linear_refinement_steps"],
        velocity_lu_ordering="COLAMD",
        repeats=1,
        baseline_solve_s=float("nan"),
        reference_total_s=float("nan"),
        export_state=False,
    )
    forward.segmented.configure_roi_environment()
    os.environ["PVFV_LABEL_BACKEND"] = str(FORWARD_ARGS["label_backend"])
    runner = forward.segmented.load_runner()
    runner_args = forward._runner_args(run_args)
    namespace, _fixed_wall_clone, _global_wall_clone = (
        forward.segmented.install_namespace(runner, runner_args)
    )
    cfg = forward.segmented.make_cfg(
        runner, namespace, run_args.out_dir, runner_args
    )
    cfg.body_force = tuple(float(value) for value in FORWARD_ARGS["body_force"])
    geom, _input_record, _build_time = forward._build_geometry(
        runner, namespace, cfg, run_args
    )
    trace = build_hybrid_trace_geometry(
        namespace, geom, cfg, trace_basis=str(FORWARD_ARGS["trace_basis"])
    )
    cp = namespace["cp"]
    labels = cp.asnumpy(geom.labels).astype(np.int64, copy=False)
    distance = cp.asnumpy(geom.dist).astype(np.float64, copy=False)
    valid = labels >= 0
    valid_labels = labels[valid]
    valid_distance = distance[valid]
    n_cells = int(geom.n_cells)

    radii = np.zeros(n_cells, dtype=np.float64)
    np.maximum.at(radii, valid_labels, valid_distance)

    parent = np.asarray(trace.face_parent_edge, dtype=np.int64)
    normals = np.asarray(trace.face_normal, dtype=np.float64)
    n_edges = int(geom.owner.size)
    area_count = np.bincount(parent, minlength=n_edges).astype(np.float64)
    vector_sum = np.zeros((n_edges, 3), dtype=np.float64)
    np.add.at(vector_sum, parent, normals)
    edge_valid = area_count > 0
    chi = np.linalg.norm(vector_sum[edge_valid], axis=1) / area_count[edge_valid]
    chi_weight = area_count[edge_valid]

    metrics = {
        "h_S_over_h": float(np.max(valid_distance)),
        "h_S_hat": float(np.max(valid_distance) / 24.0),
        "cell_graph_radius_mean": float(np.mean(radii)),
        "cell_graph_radius_p95": float(np.quantile(radii, 0.95)),
        "cell_graph_radius_max": float(np.max(radii)),
        "area_weighted_chi_p05": _weighted_quantile(chi, chi_weight, 0.05),
        "area_weighted_chi_median": _weighted_quantile(
            chi, chi_weight, 0.50
        ),
        "chi_min": float(np.min(chi)),
    }
    del geom, trace, labels, distance
    cp.get_default_memory_pool().free_all_blocks()
    return metrics


def _record(
    *,
    case_name: str,
    spec: dict[str, Any],
    trajectory_count: int,
    point_dir: Path,
    manifest: dict[str, Any],
    csv_row: dict[str, str],
    geometry: dict[str, float],
) -> dict[str, Any]:
    input_record = manifest["input"]
    reference = manifest["reference_metadata"]
    n_f = int(reference["N_ref_cells"])
    n_cv = _number(csv_row, "N_cv", integer=True)
    n_z = _number(csv_row, "N_trace_vector_dofs", integer=True)
    n_p = int(n_cv) - 1
    n_system = int(n_z) + n_p
    return {
        "case": case_name,
        "case_display": spec["display"],
        "initial_trajectory_count": trajectory_count,
        "particle_id_range": f"0:{trajectory_count - 1}",
        "frames_per_trajectory": 100,
        "trajectory_record_count": int(input_record["particle_rows_selected"]),
        "unique_trajectory_site_count": int(input_record["particle_site_count"]),
        "N_f": n_f,
        "N_cv": int(n_cv),
        "N_edges": _number(csv_row, "N_edges", integer=True),
        "N_interface_facelets": _number(
            csv_row, "N_interface_facelets", integer=True
        ),
        "N_connected_patches": _number(
            csv_row, "N_connected_patches", integer=True
        ),
        "N_trace_modes": _number(csv_row, "N_trace_modes", integer=True),
        "N_z": int(n_z),
        "N_p": n_p,
        "N_system": n_system,
        "pressure_space_compression": n_f / n_p,
        "nominal_voxel_state_ratio": (4 * n_f - 1) / n_system,
        **geometry,
        "K_eff_x": _number(csv_row, "K_eff_x"),
        "K_ref_x": _number(csv_row, "K_ref_x"),
        "e_K_percent": _number(csv_row, "e_K_percent"),
        "e_phi_percent": _number(csv_row, "e_phi_percent"),
        "e_u_percent": _number(csv_row, "e_u_percent"),
        "mass_inf_per_volume": _number(csv_row, "mass_inf_per_volume"),
        "momentum_residual_inf": _number(csv_row, "momentum_residual_inf"),
        "linear_solver_iterations": _number(
            csv_row, "linear_solver_iterations", integer=True
        ),
        "linear_solver_relative_residual": _number(
            csv_row, "linear_solver_relative_residual"
        ),
        "t_geometry_build_s": _number(csv_row, "t_geometry_build_s"),
        "t_trace_geometry_s": _number(csv_row, "t_trace_geometry_s"),
        "t_matrix_assembly_s": _number(csv_row, "t_matrix_assembly_s"),
        "t_factorization_s": _number(csv_row, "t_factorization_s"),
        "t_cached_call_median_s": _number(
            csv_row, "t_cached_call_median_s"
        ),
        "auxiliary_site_count": _number(
            csv_row, "auxiliary_site_count", integer=True
        ),
        "snap_displacement_max_vox": float(
            input_record["particle_snap_displacement_max_vox"]
        ),
        "solver_formulation_id": csv_row["solver_formulation_id"],
        "source_hashes": manifest["code_sha256"],
        "input_hashes": manifest["input_sha256"],
        "run_directory": str(point_dir),
    }


def _audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    if len(records) != 8:
        failures.append(f"expected 8 rows, found {len(records)}")
    for case_name, spec in CASES.items():
        case_rows = [row for row in records if row["case"] == case_name]
        case_rows.sort(key=lambda row: row["initial_trajectory_count"])
        if [row["initial_trajectory_count"] for row in case_rows] != spec["prefixes"]:
            failures.append(f"{case_name}: trajectory prefixes changed")
        site_counts = [row["unique_trajectory_site_count"] for row in case_rows]
        if any(right < left for left, right in zip(site_counts, site_counts[1:])):
            failures.append(f"{case_name}: site counts are not nested-monotone")
        for row in case_rows:
            expected_records = 100 * row["initial_trajectory_count"]
            if row["trajectory_record_count"] != expected_records:
                failures.append(
                    f"{case_name}/{row['initial_trajectory_count']}: "
                    "record count mismatch"
                )
            if row["unique_trajectory_site_count"] != row["N_cv"]:
                failures.append(
                    f"{case_name}/{row['initial_trajectory_count']}: "
                    "site/cell count mismatch"
                )
            if row["auxiliary_site_count"] != 0:
                failures.append(
                    f"{case_name}/{row['initial_trajectory_count']}: "
                    "auxiliary site present"
                )
            if row["snap_displacement_max_vox"] != 0.0:
                failures.append(
                    f"{case_name}/{row['initial_trajectory_count']}: "
                    "nonzero particle snapping"
                )
            if row["mass_inf_per_volume"] > 1.0e-12:
                failures.append(
                    f"{case_name}/{row['initial_trajectory_count']}: "
                    "mass residual exceeds 1e-12"
                )
            if row["momentum_residual_inf"] > 1.0e-12:
                failures.append(
                    f"{case_name}/{row['initial_trajectory_count']}: "
                    "momentum residual exceeds 1e-12"
                )
        full = case_rows[-1]
        for key, expected in spec["full_expected"].items():
            actual = full[key]
            tolerance = 0 if isinstance(expected, int) else 1.0e-12
            if abs(actual - expected) > tolerance:
                failures.append(
                    f"{case_name}: full-density {key}={actual} "
                    f"does not reproduce {expected}"
                )

    formulation_ids = {row["solver_formulation_id"] for row in records}
    if formulation_ids != {
        "moment_constrained_hybrid_voronoi_connected_p1_symmetric_gradient_v1"
    }:
        failures.append(f"unexpected formulation IDs: {sorted(formulation_ids)}")
    core_hashes = {
        row["source_hashes"]["hybrid_voronoi_trace.py"] for row in records
    }
    if core_hashes != {
        "d43f1af57352aa53b6598329819c663c1be248690082758b13c0501a46a5f823"
    }:
        failures.append(f"unexpected core hashes: {sorted(core_hashes)}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": {
            "row_count": len(records),
            "nested_particle_prefix_protocol": not any(
                "prefix" in failure for failure in failures
            ),
            "site_counts_nondecreasing": not any(
                "nested-monotone" in failure for failure in failures
            ),
            "one_cell_per_unique_site": not any(
                "site/cell" in failure for failure in failures
            ),
            "zero_auxiliary_sites": not any(
                "auxiliary" in failure for failure in failures
            ),
            "zero_particle_snapping": not any(
                "snapping" in failure for failure in failures
            ),
            "residual_gate_1e-12": not any(
                "residual" in failure for failure in failures
            ),
            "full_density_locked_baselines_reproduced": not any(
                "full-density" in failure for failure in failures
            ),
            "single_locked_formulation_and_core_hash": not any(
                "formulation" in failure or "core hash" in failure
                for failure in failures
            ),
        },
        "non_gate": (
            "Error monotonicity is reported but is not an acceptance condition; "
            "the study is a nested support-response audit, not a convergence proof."
        ),
        "failures": failures,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    scalar_records = [
        {
            key: value
            for key, value in record.items()
            if key not in {"source_hashes", "input_hashes"}
        }
        for record in records
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_records[0]))
        writer.writeheader()
        writer.writerows(scalar_records)


def _plot(path: Path, records: list[dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    colors = {"orthogonal_duct": "#2f6f9f", "skewed_duct": "#c66b2b"}
    markers = {
        "e_K_percent": "o",
        "e_phi_percent": "s",
        "e_u_percent": "^",
    }
    labels = {
        "e_K_percent": r"$e_K$",
        "e_phi_percent": r"$e_\phi$",
        "e_u_percent": r"$e_u$",
    }
    for case_name, spec in CASES.items():
        rows = sorted(
            (row for row in records if row["case"] == case_name),
            key=lambda row: row["initial_trajectory_count"],
        )
        for metric in markers:
            line_label = f"{spec['display']}, {labels[metric]}"
            axes[0].plot(
                [row["unique_trajectory_site_count"] for row in rows],
                [row[metric] for row in rows],
                marker=markers[metric],
                color=colors[case_name],
                linewidth=1.15,
                markersize=4.2,
                label=line_label,
            )
            axes[1].plot(
                [row["h_S_hat"] for row in rows],
                [row[metric] for row in rows],
                marker=markers[metric],
                color=colors[case_name],
                linewidth=1.15,
                markersize=4.2,
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("unique trajectory-derived sites, $N_c$")
    axes[0].set_ylabel("relative error (%)")
    axes[0].set_title("a   Nested trajectory prefixes", loc="left")
    axes[1].set_xlabel(r"graph fill distance, $h_S/24h$")
    axes[1].set_ylabel("relative error (%)")
    axes[1].set_title("b   Geometric support scale", loc="left")
    axes[1].invert_xaxis()
    for axis in axes:
        axis.grid(True, color="#d7dee3", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(
        loc="upper right",
        frameon=False,
        fontsize=6.5,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.6,
    )
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the eight-point nested trajectory-support study for the "
            "orthogonal and skewed ducts."
        )
    )
    parser.add_argument("--compute-root", type=Path, default=DEFAULT_COMPUTE_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    compute_root = args.compute_root.resolve()
    evidence_dir = args.evidence_dir.resolve()
    run_root = args.run_root.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case_name, spec in CASES.items():
        for trajectory_count in spec["prefixes"]:
            print(
                f"[resolution] {case_name}: {trajectory_count} trajectories",
                flush=True,
            )
            point_dir, manifest, csv_row = _run_point(
                compute_root=compute_root,
                run_root=run_root,
                case_name=case_name,
                spec=spec,
                trajectory_count=trajectory_count,
                rerun=args.rerun,
            )
            geometry = _geometry_metrics(
                compute_root=compute_root,
                case_name=case_name,
                spec=spec,
                trajectory_count=trajectory_count,
            )
            records.append(
                _record(
                    case_name=case_name,
                    spec=spec,
                    trajectory_count=trajectory_count,
                    point_dir=point_dir,
                    manifest=manifest,
                    csv_row=csv_row,
                    geometry=geometry,
                )
            )

    records.sort(key=lambda row: (row["case"], row["initial_trajectory_count"]))
    protocol = {
        "protocol_id": "nested_trajectory_support_forward_resolution_ducts_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Controlled nested trajectory-support response study; not a formal "
            "mesh-refinement convergence proof and not an asymptotic order estimate."
        ),
        "site_rule": (
            "Retain all 100 frames from stable particle-id prefixes of the "
            "wall-safe flow-generated trajectories; do not reseed, "
            "relocate, regularize, or add mask-derived sites."
        ),
        "cases": {
            case_name: {
                "trajectory_prefix_counts": spec["prefixes"],
                "particle_id_ranges": [
                    f"0:{count - 1}" for count in spec["prefixes"]
                ],
                "mask": spec["mask"],
                "particle_window": spec["particle"],
                "mask_sha256": _sha256(
                    compute_root / "../data/controlled_cases" / "." / spec["mask"]
                ),
                "particle_window_sha256": _sha256(
                    compute_root
                    / "../data/controlled_cases"
                    / spec["particle"]
                ),
            }
            for case_name, spec in CASES.items()
        },
        "locked_forward": FORWARD_ARGS,
        "source_sha256": {
            name: _sha256(compute_root / "code" / name)
            for name in (
                "run_hybrid_voronoi_forward.py",
                "hybrid_voronoi_trace.py",
                "hybrid_site_sources.py",
                "run_segmented_selector_geodesic_rows.py",
            )
        },
        "reported_metrics": [
            key
            for key in records[0]
            if key not in {"source_hashes", "input_hashes", "run_directory"}
        ],
    }
    audit = _audit(records)
    if audit["status"] != "PASS":
        raise RuntimeError(json.dumps(audit, indent=2))

    _write_csv(evidence_dir / "forward_resolution_ducts.csv", records)
    (evidence_dir / "forward_resolution_ducts.json").write_text(
        json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
    )
    (evidence_dir / "forward_resolution_protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8"
    )
    (evidence_dir / "forward_resolution_consistency_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    figure_stem = (
        PAPER_ROOT
        / "../../outputs"
        / "figures"
        / "forward_resolution_ducts"
    )
    _plot(figure_stem, records)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
