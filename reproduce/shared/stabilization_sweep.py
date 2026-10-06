"""Residual-stabilization sensitivity, and the fibrous case closed on it.

The assembly of the GPU pipeline hard-codes the residual stabilization weight

    w = 2 nu A_facelet / h        (symmetric-gradient form)

with no exposed multiplier.  This module introduces a dimensionless multiplier
tau, with tau = 1 the default rule, and sweeps it.

The pipeline's module is NOT modified.  Instead the assembly loop is reproduced
here with tau exposed, and the reproduction is checked: at tau = 1 the velocity
matrix, the divergence matrix and the load matrix must agree with the pipeline's
assembly *bitwise*.  If they do not, the sweep refuses to run.

Because the stabilization enters additively,

    A(tau) = A_consistent + tau * A_stabilization ,

only two assemblies are needed: tau = 1 (which is also the bitwise check) and
tau = 0.  Every other tau is formed exactly as A(0) + tau * (A(1) - A(0)), so no
value in the sweep carries a second discretization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402
import stability_core as sc  # noqa: E402

PROTOCOL_ID = "pvfv_stabilization_sweep"

FIBROUS = {
    "case": "fibrous_filter_proxy",
    "mask": "examples/segmented_masks/fibrous_filter_proxy_16x64x64.npz",
    "mask_sha256": "b22cee71aa4622d1480cf83385c32f7f666e4f74407124d13832ba96bf431eeb",
    "reference": "outputs/reference_data/fibrous_filter_proxy__reference.npz",
    "reference_sha256": "1a07bede72e147814b81ac8e8fa81d000e0a3d52f34acb164e95b4358103090a",
    "sites": "outputs/fibrous_wall_biased_1256_sites.npz",
    "sites_sha256": "1826a654ec09ddf1854399487f44035218717216ca6db6feb3702c52b582be77",
}

DEFAULT_TAUS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


# --------------------------------------------------------------------------
# Assembly with the stabilization multiplier exposed.
# The body below is the retained `assemble_moment_constrained_hybrid_stokes`
# loop with one change: `stabilization_weight` is multiplied by `tau`.
# --------------------------------------------------------------------------
def assemble_with_tau(trace, *, viscosity, body_force, viscous_form, tau):
    import hybrid_voronoi_trace as hvt

    area = trace.voxel_size * trace.voxel_size
    n_modes = trace.n_trace_modes
    n_trace_dofs = 3 * n_modes
    body_force = np.asarray(body_force, dtype=np.float64).reshape(3)

    owner_r = hvt._minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_owner],
        trace.domain_length_x, trace.periodic_x,
    )
    neigh_r = hvt._minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_neigh],
        trace.domain_length_x, trace.periodic_x,
    )
    wall_r = hvt._minimum_image_x(
        trace.wall_centroid - trace.cell_centroid[trace.wall_cell],
        trace.domain_length_x, trace.periodic_x,
    )

    n_facelets = trace.n_facelets
    incident_cell = np.concatenate([trace.face_owner, trace.face_neigh, trace.wall_cell])
    incident_face = np.concatenate([
        np.arange(n_facelets, dtype=np.int32),
        np.arange(n_facelets, dtype=np.int32),
        np.full(trace.wall_cell.size, -1, dtype=np.int32),
    ])
    incident_normal = np.concatenate(
        [trace.face_normal, -trace.face_normal, trace.wall_normal], axis=0
    )
    incident_r = np.concatenate([owner_r, neigh_r, wall_r], axis=0)
    order = np.argsort(incident_cell, kind="stable")
    incident_cell = incident_cell[order]
    incident_face = incident_face[order]
    incident_normal = incident_normal[order]
    incident_r = incident_r[order]
    cell_start = np.searchsorted(incident_cell, np.arange(trace.n_cells + 1))

    matrix_rows, matrix_cols, matrix_values = [], [], []
    divergence_rows, divergence_cols, divergence_values = [], [], []
    body_force_matrix = np.zeros((n_trace_dofs, 3), dtype=np.float64)
    cell_recovery = []
    cell_gradient = []

    for cell in range(trace.n_cells):
        first, last = int(cell_start[cell]), int(cell_start[cell + 1])
        cell_faces = incident_face[first:last]
        cell_normals = incident_normal[first:last]
        cell_r = incident_r[first:last]

        mode_set = set()
        for facelet in cell_faces[cell_faces >= 0]:
            for mode in trace.face_mode_ids[int(facelet)]:
                if mode >= 0:
                    mode_set.add(int(mode))
        local_modes = np.asarray(sorted(mode_set), dtype=np.int32)
        if local_modes.size == 0:
            cell_recovery.append((np.empty(0, dtype=np.int64), np.zeros((3, 0))))
            cell_gradient.append(np.zeros((0, 3, 3)))
            continue
        local_index = {int(mode): index for index, mode in enumerate(local_modes)}
        n_local_dofs = 3 * int(local_modes.size)
        recovery = np.zeros((3, n_local_dofs), dtype=np.float64)
        gradient_basis = np.zeros((n_local_dofs, 3, 3), dtype=np.float64)
        divergence = np.zeros(n_local_dofs, dtype=np.float64)
        volume = float(trace.cell_volume[cell])

        for facelet, normal, position in zip(cell_faces, cell_normals, cell_r):
            if facelet < 0:
                continue
            ids = trace.face_mode_ids[int(facelet)]
            values = trace.face_mode_values[int(facelet)]
            for mode, value in zip(ids, values):
                if mode < 0:
                    continue
                local_mode = local_index[int(mode)]
                block = slice(3 * local_mode, 3 * local_mode + 3)
                recovery[:, block] += (area * value / volume) * np.outer(position, normal)
                divergence[block] += area * value * normal
                for component in range(3):
                    gradient_basis[3 * local_mode + component, component, :] += (
                        area * value / volume
                    ) * normal

        if viscous_form == "full_gradient":
            local_matrix = viscosity * volume * np.einsum(
                "aij,bij->ab", gradient_basis, gradient_basis
            )
            stabilization_weight = viscosity * area / trace.voxel_size
        elif viscous_form == "symmetric_gradient":
            symmetric_basis = 0.5 * (gradient_basis + np.transpose(gradient_basis, (0, 2, 1)))
            local_matrix = 2.0 * viscosity * volume * np.einsum(
                "aij,bij->ab", symmetric_basis, symmetric_basis
            )
            stabilization_weight = 2.0 * viscosity * area / trace.voxel_size
        else:
            raise ValueError("Unsupported viscous form: " + str(viscous_form))
        stabilization_weight = stabilization_weight * float(tau)  # <-- the only change

        for facelet, position in zip(cell_faces, cell_r):
            trace_map = np.zeros((3, n_local_dofs), dtype=np.float64)
            if facelet >= 0:
                ids = trace.face_mode_ids[int(facelet)]
                values = trace.face_mode_values[int(facelet)]
                for mode, value in zip(ids, values):
                    if mode < 0:
                        continue
                    local_mode = local_index[int(mode)]
                    block = slice(3 * local_mode, 3 * local_mode + 3)
                    trace_map[:, block] += value * np.eye(3)
            gradient_at_face = np.einsum("aij,j->ia", gradient_basis, position)
            residual_map = trace_map - recovery - gradient_at_face
            local_matrix += stabilization_weight * (residual_map.T @ residual_map)

        global_dofs = (
            3 * local_modes[:, None] + np.arange(3, dtype=np.int64)[None, :]
        ).reshape(-1)
        matrix_rows.append(np.repeat(global_dofs, n_local_dofs))
        matrix_cols.append(np.tile(global_dofs, n_local_dofs))
        matrix_values.append(local_matrix.reshape(-1))
        body_force_matrix[global_dofs] += volume * recovery.T
        nonzero = np.flatnonzero(divergence)
        divergence_rows.extend([cell] * int(nonzero.size))
        divergence_cols.extend(global_dofs[nonzero].tolist())
        divergence_values.extend(divergence[nonzero].tolist())
        cell_recovery.append((global_dofs, recovery))
        cell_gradient.append(gradient_basis)

    rows = np.concatenate(matrix_rows)
    cols = np.concatenate(matrix_cols)
    values = np.concatenate(matrix_values)
    velocity_matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(n_trace_dofs, n_trace_dofs)
    ).tocsr()
    velocity_matrix = (0.5 * (velocity_matrix + velocity_matrix.T)).tocsr()
    divergence_matrix = sparse.coo_matrix(
        (divergence_values, (divergence_rows, divergence_cols)),
        shape=(trace.n_cells, n_trace_dofs),
    ).tocsr()
    return {
        "A": velocity_matrix,
        "D": divergence_matrix,
        "B": body_force_matrix,
        "cell_recovery": cell_recovery,
        "cell_gradient": cell_gradient,
        "rhs": body_force_matrix @ body_force,
    }


def bitwise_gate(trace, production_system, viscosity, body_force, viscous_form):
    reproduced = assemble_with_tau(
        trace, viscosity=viscosity, body_force=body_force,
        viscous_form=viscous_form, tau=1.0,
    )
    checks = {}
    for name, mine, theirs in (
        ("A", reproduced["A"], production_system.velocity_matrix),
        ("D", reproduced["D"], production_system.divergence_matrix),
    ):
        same_pattern = (
            np.array_equal(mine.indptr, theirs.indptr)
            and np.array_equal(mine.indices, theirs.indices)
        )
        checks[name + "_pattern_identical"] = bool(same_pattern)
        checks[name + "_values_bitwise_identical"] = bool(
            same_pattern and np.array_equal(mine.data, theirs.data)
        )
        checks[name + "_max_absolute_difference"] = float(
            abs(mine - theirs).max() if mine.nnz or theirs.nnz else 0.0
        )
    checks["B_bitwise_identical"] = bool(
        np.array_equal(reproduced["B"], production_system.body_force_matrix)
    )
    checks["B_max_absolute_difference"] = float(
        np.max(np.abs(reproduced["B"] - production_system.body_force_matrix))
    )
    checks["status"] = "PASS" if all(
        checks[key] for key in checks if key.endswith("identical")
    ) else "FAIL"
    return checks, reproduced


def solve_at_tau(A, D, rhs, *, rtol, maxiter, refinement_steps=1):
    """The retained MINRES path: drop the last mass row, block-diagonal
    preconditioner from diag(A) and diag(D diag(A)^-1 D^T), then centre."""
    from scipy.sparse.linalg import LinearOperator, minres

    n_trace = A.shape[0]
    n_cells = D.shape[0]
    D_r = D[:-1].tocsr()
    diagonal = np.asarray(A.diagonal(), dtype=np.float64)
    if np.any(diagonal <= 0.0):
        return {"status": "non_positive_velocity_diagonal", "A_diag_min": float(diagonal.min())}
    inverse_diagonal = 1.0 / diagonal
    schur_diagonal = np.asarray(D_r.multiply(D_r) @ inverse_diagonal).reshape(-1)
    if np.any(schur_diagonal <= 0.0):
        return {"status": "non_positive_schur_diagonal",
                "schur_diag_min": float(schur_diagonal.min())}
    kkt = sparse.bmat(
        [[A, D_r.T], [D_r, sparse.csr_matrix((n_cells - 1, n_cells - 1))]], format="csr"
    )
    inverse_preconditioner = np.concatenate([inverse_diagonal, 1.0 / schur_diagonal])
    preconditioner = LinearOperator(
        kkt.shape,
        matvec=lambda v: inverse_preconditioner * v,
        rmatvec=lambda v: inverse_preconditioner * v,
        dtype=np.float64,
    )
    full_rhs = np.concatenate([rhs, np.zeros(n_cells - 1)])
    iterations = 0

    def count(_x):
        nonlocal iterations
        iterations += 1

    started = time.perf_counter()
    solution, info = minres(
        kkt, full_rhs, rtol=rtol, maxiter=maxiter, M=preconditioner,
        callback=count, check=False,
    )
    # The retained path applies `iterative_refinement_steps` residual corrections
    # (FORWARD_ARGS sets one); reproduced here so the sweep and the pipeline's
    # solve stop on the same criterion.
    residual_limit = max(20.0 * rtol, 1.0e-12)
    rhs_norm = max(float(np.linalg.norm(full_rhs)), 1.0e-300)
    refinements = 0
    while info == 0 and refinements < refinement_steps:
        correction_rhs = np.asarray(full_rhs - kkt @ solution).reshape(-1)
        if float(np.linalg.norm(correction_rhs)) / rhs_norm <= residual_limit:
            break
        correction, info = minres(
            kkt, correction_rhs, rtol=rtol, maxiter=maxiter, M=preconditioner,
            callback=count, check=False,
        )
        solution = solution + correction
        refinements += 1
    elapsed = float(time.perf_counter() - started)
    residual = float(
        np.linalg.norm(kkt @ solution - full_rhs) / max(np.linalg.norm(full_rhs), 1e-300)
    )
    pressure = np.concatenate([solution[n_trace:], np.zeros(1)])
    pressure -= float(np.mean(pressure))
    return {
        "status": "converged" if info == 0 and residual <= max(20.0 * rtol, 1e-12)
        else "not_converged",
        "info": int(info),
        "iterations": int(iterations),
        "relative_residual": residual,
        "seconds": elapsed,
        "refinement_steps": int(refinements),
        "z": solution[:n_trace],
        "p": pressure,
        "A_diag_min": float(diagonal.min()),
        "schur_diag_min": float(schur_diagonal.min()),
        "kkt_nnz": int(kkt.nnz),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/supplementary/s3_verification")
    parser.add_argument("--taus", default=",".join(str(t) for t in DEFAULT_TAUS))
    parser.add_argument("--case", default="fibrous", choices=["fibrous", "orthogonal_duct", "bentheimer_crop", "skewed_duct"])
    parser.add_argument("--control-sites", type=int, default=1600,
                        help="mask-graph-FPS site count for a non-fibrous control case")
    parser.add_argument("--spectra", action="store_true", help="also estimate beta_h per tau")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    taus = [float(t) for t in args.taus.split(",") if t]

    boot = ec.bootstrap(root / "bootstrap")
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]

    if args.case == "fibrous":
        spec = FIBROUS
        mask_path = Path(spec["mask"])
        reference_path = Path(spec["reference"])
        for key, path in (("mask", mask_path), ("reference", reference_path),
                          ("sites", Path(spec["sites"]))):
            digest = ec.sha256_file(path)
            if digest != spec[key + "_sha256"]:
                raise RuntimeError(
                    "%s hash mismatch: %s != %s" % (key, digest, spec[key + "_sha256"])
                )
        with np.load(spec["sites"], allow_pickle=True) as data:
            key = "seed_flat" if "seed_flat" in data.files else data.files[0]
            sites = np.unique(np.asarray(data[key], dtype=np.int64).reshape(-1))
        case_name = spec["case"]
        seed_spec = "fibrous_filter_proxy:archived_wall_biased_1256"
    else:
        paths = ec.case_paths(args.case)
        mask_path, reference_path = paths["mask"], paths["reference"]
        mask = ec.load_mask(mask_path)
        order = ec.graph_farthest_point_order(
            mask, np.flatnonzero(mask.reshape(-1)).astype(np.int64),
            periodic_x=False, count=int(args.control_sites),
        )
        sites = np.unique(order)
        case_name = args.case
        seed_spec = "%s:mask_graph_fps:n%d" % (args.case, args.control_sites)

    geom, geometry_meta, build_time = ec.build_geometry_from_seed_flat(
        sites, mask_path=mask_path, seed_spec=seed_spec
    )
    trace = boot["build_hybrid_trace_geometry"](
        ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
    )
    body_force = np.asarray(ec.FORWARD_ARGS["body_force"], dtype=np.float64)
    production = boot["assemble_moment_constrained_hybrid_stokes"](
        trace, viscosity=float(cfg.nu), body_force=body_force,
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]),
    )

    gate, reproduced_one = bitwise_gate(
        trace, production, float(cfg.nu), body_force, str(ec.FORWARD_ARGS["viscous_form"])
    )
    print("[gate] tau=1 reproduction vs production assembly: " + gate["status"])
    for key, value in gate.items():
        print("        %-34s %s" % (key, value))
    ec.write_json(root / ("assembly_bitwise_gate__%s.json" % case_name), gate)
    if gate["status"] != "PASS":
        print("[gate] refusing to sweep: the reproduction is not bitwise identical")
        return 1

    zero = assemble_with_tau(
        trace, viscosity=float(cfg.nu), body_force=body_force,
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]), tau=0.0,
    )
    A0, A1 = zero["A"], reproduced_one["A"]
    A_stab = (A1 - A0).tocsr()
    D = reproduced_one["D"]
    rhs = reproduced_one["rhs"]

    # Reference fields, aggregated with the pipeline's coarsening operator.
    reference_geom, reference_result = boot["forward"].segmented.load_reference_npz(
        ns, reference_path, cfg
    )
    U_ref_gpu, p_ref_gpu = ns["coarsen_voxel_reference_to_coarse_gpu"](
        reference_geom, reference_result, geom
    )
    U_ref = cp.asnumpy(U_ref_gpu).astype(np.float64)
    p_ref = cp.asnumpy(p_ref_gpu).astype(np.float64)
    volume = cp.asnumpy(geom.volume).astype(np.float64)
    dvec = cp.asnumpy(geom.dvec).astype(np.float64)
    K_ref = float(reference_result["K_eff_x"])

    rows = []
    for tau in taus:
        A = (A0 + tau * A_stab).tocsr()
        A = (0.5 * (A + A.T)).tocsr()
        out = solve_at_tau(
            A, D, rhs,
            rtol=float(ec.FORWARD_ARGS["linear_rtol"]),
            maxiter=int(ec.FORWARD_ARGS["linear_maxiter"]),
            refinement_steps=int(ec.FORWARD_ARGS["linear_refinement_steps"]),
        )
        row = {
            "protocol_id": PROTOCOL_ID,
            "case": case_name,
            "tau": tau,
            "N_cv": int(geom.n_cells),
            "N_sites": int(sites.size),
            "N_z": int(A.shape[0]),
            "N_system": int(A.shape[0] + geom.n_cells - 1),
            "nnz_A": int(A.nnz),
            "nnz_D": int(D.nnz),
            "solver_status": out["status"],
            "A_diag_min": out.get("A_diag_min"),
            "schur_diag_min": out.get("schur_diag_min"),
        }
        if "z" not in out:
            rows.append(row)
            print("[tau] %s tau=%.4g  %s" % (case_name, tau, out["status"]), flush=True)
            continue
        z, p = out["z"], out["p"]
        U = np.zeros((geom.n_cells, 3), dtype=np.float64)
        for cell, (global_dofs, recovery) in enumerate(reproduced_one["cell_recovery"]):
            if global_dofs.size:
                U[cell] = recovery @ z[global_dofs]
        face_velocity = np.zeros((trace.n_facelets, 3), dtype=np.float64)
        coefficients = z.reshape(trace.n_trace_modes, 3)
        for column in range(trace.face_mode_ids.shape[1]):
            mode = trace.face_mode_ids[:, column]
            valid = mode >= 0
            if np.any(valid):
                face_velocity[valid] += (
                    trace.face_mode_values[valid, column, None] * coefficients[mode[valid]]
                )
        face_flux = (trace.voxel_size ** 2) * np.sum(face_velocity * trace.face_normal, axis=1)
        phi = np.bincount(
            trace.face_parent_edge, weights=face_flux,
            minlength=trace.stored_edge_sign_from_sorted.size,
        ) * trace.stored_edge_sign_from_sorted

        total_volume = float(volume.sum())
        e_u = 100.0 * float(
            np.sqrt(np.sum(volume * np.sum((U - U_ref) ** 2, axis=1)))
            / max(np.sqrt(np.sum(volume * np.sum(U_ref ** 2, axis=1))), 1e-300)
        )
        e_p, pressure_detail = ec.gauge_invariant_pressure_error(p, p_ref, volume)
        mean_flux = np.sum(phi[:, None] * dvec, axis=0) / total_volume
        K_eff = float(cfg.nu) * float(mean_flux[0]) / float(body_force[0])
        Dz = np.asarray(D @ z).reshape(-1)
        abs_Dz = np.asarray(abs(D) @ np.abs(z)).reshape(-1)
        eps = np.finfo(np.float64).eps
        row.update({
            "e_K_signed_percent": 100.0 * (K_eff - K_ref) / abs(K_ref),
            "e_K_percent": 100.0 * abs(K_eff - K_ref) / abs(K_ref),
            "e_u_percent": e_u,
            "e_p_percent": e_p,
            "K_eff_parallel": K_eff,
            "K_ref_parallel": K_ref,
            "mass_inf_per_volume": float(np.max(np.abs(Dz) / np.maximum(volume, 1e-300))),
            "mass_backward_error_dimensionless": float(
                np.max(np.abs(Dz)) / (np.max(abs_Dz) + eps * max(1.0, float(np.max(abs_Dz))))
            ),
            "momentum_residual_inf": float(np.max(np.abs(A @ z + D.T @ p - rhs))),
            "linear_solver_iterations": out["iterations"],
            "linear_solver_relative_residual": out["relative_residual"],
            "solve_seconds": out["seconds"],
            "pressure_multiplier_vs_reference_cosine":
                pressure_detail["multiplier_vs_reference_cosine"],
        })
        if args.spectra:
            # schur_spectrum performs the gauge reduction itself: it must be given
            # the FULL divergence operator and the FULL cell-volume vector, exactly
            # as audit_gauge_fixed_stability.py calls it.  Passing an already
            # reduced D_r removes a second row and shifts beta_h.
            spectrum = sc.schur_spectrum(A, D, volume, gauge_index=D.shape[0] - 1)
            row.update({
                "beta_h": spectrum.get("beta_h"),
                "schur_lambda_min_positive": spectrum.get("schur_lambda_min_positive"),
                "schur_lambda_max": spectrum.get("schur_lambda_max"),
                "schur_condition_proxy": spectrum.get("schur_condition_proxy"),
                "rank_defect_estimate_after_gauge": spectrum.get("rank_defect_estimate_after_gauge"),
            })
        rows.append(row)
        print(
            "[tau] %s tau=%-6.4g %s  e_K=%8.4f%% e_u=%8.4f%% e_p=%8.4f%% "
            "iters=%6d rel=%.2e eta_m=%.2e"
            % (case_name, tau, out["status"], row["e_K_percent"], row["e_u_percent"],
               row["e_p_percent"], row["linear_solver_iterations"],
               row["linear_solver_relative_residual"],
               row["mass_backward_error_dimensionless"]),
            flush=True,
        )

    columns = sorted({k for r in rows for k in r})
    lead = ["protocol_id", "case", "tau", "N_cv", "N_system", "solver_status",
            "e_K_percent", "e_u_percent", "e_p_percent"]
    columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
    ec.write_csv(root / ("stabilization_sweep__%s.csv" % case_name), rows, columns)
    print("[write] " + str(root / ("stabilization_sweep__%s.csv" % case_name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
