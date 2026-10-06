"""Manufactured Stokes benchmark for the shared-trace mixed formulation.

The controlled-case study measures every error against an image-resolved
*reference solver* on the same voxel mask, so it can show that one
discretization approaches another but never that either approaches the
continuum.  This module supplies the continuum limb: a smooth
divergence-free Stokes solution with a known pressure, on a fully walled cube,
refined in the voxel size and in the prescribed-site spacing together.

Manufactured solution, on the cube [0,L]^3 with L = N h:

    psi(x,y,z) = sin^2(pi x/L) sin^2(pi y/L) sin^2(pi z/L)
    u          = curl(0, 0, psi) = ( d psi/dy , -d psi/dx , 0 )
    p          = sin(2 pi x/L) sin(2 pi y/L) sin(2 pi z/L)
    f          = -div(2 nu eps(u)) + grad p

`u` is divergence-free by construction and vanishes on all six faces of the cube,
which is exactly the homogeneous no-slip condition the method imposes weakly on
wall facelets; `p` has zero mean.  With div u = 0 and constant nu,
div(2 nu eps(u)) = nu lap(u), and f is obtained by symbolic differentiation - it
is never transcribed by hand.

Nothing in the GPU pipeline is modified.  The load vector is the only object
this module supplies, and it is assembled with the pipeline's *own* per-cell
recovery operators: the assembly forms

    rhs = B g ,      B[dofs_K] += |K| R_K^T      (one constant vector g)

so writing  B[:,0] <- sum_K |K| R_K^T f_K  and taking g = (1,0,0) reproduces the
consistent cell-averaged load functional  sum_K |K| f_K . (R_K z)  for a
cell-varying f while leaving A, D, the factorization, MINRES, the pressure gauge,
the cell recovery and every residual exactly as the pipeline computes
them.

All reference quantities - cell means of u and p, and interface fluxes - are
integrals, evaluated by tensor Gauss-Legendre quadrature whose order is verified
in-run rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stabilization_sweep as ss  # noqa: E402

PROTOCOL_ID = "pvfv_manufactured_stokes"


# --------------------------------------------------------------------------
# Symbolic manufactured solution
# --------------------------------------------------------------------------
class Manufactured:
    def __init__(self, length: float, viscosity: float):
        x, y, z = sp.symbols("x y z", real=True)
        L = sp.Float(length)
        nu = sp.Float(viscosity)
        psi = (
            sp.sin(sp.pi * x / L) ** 2
            * sp.sin(sp.pi * y / L) ** 2
            * sp.sin(sp.pi * z / L) ** 2
        )
        u = sp.Matrix([sp.diff(psi, y), -sp.diff(psi, x), sp.Integer(0)])
        p = sp.sin(2 * sp.pi * x / L) * sp.sin(2 * sp.pi * y / L) * sp.sin(2 * sp.pi * z / L)

        # div(2 nu eps(u)) = nu lap(u) + nu grad(div u); div u = 0 identically here,
        # and that identity is checked symbolically below rather than assumed.
        divergence = sp.simplify(sp.diff(u[0], x) + sp.diff(u[1], y) + sp.diff(u[2], z))
        if divergence != 0:
            raise AssertionError("the manufactured velocity is not divergence free")
        eps = sp.Matrix(3, 3, lambda i, j: sp.Rational(1, 2) * (
            sp.diff(u[i], (x, y, z)[j]) + sp.diff(u[j], (x, y, z)[i])
        ))
        stress = 2 * nu * eps
        div_stress = sp.Matrix(
            [sum(sp.diff(stress[i, j], (x, y, z)[j]) for j in range(3)) for i in range(3)]
        )
        grad_p = sp.Matrix([sp.diff(p, x), sp.diff(p, y), sp.diff(p, z)])
        f = sp.simplify(-div_stress + grad_p)

        # Independent check: with div u = 0 the same forcing must equal
        # -nu lap(u) + grad p.  Any disagreement means the derivation is wrong.
        lap_u = sp.Matrix([sum(sp.diff(u[i], v, 2) for v in (x, y, z)) for i in range(3)])
        residual = sp.simplify(f - (-nu * lap_u + grad_p))
        if any(component != 0 for component in residual):
            raise AssertionError("the two derivations of the manufactured forcing disagree")

        self.symbols = (x, y, z)
        self.length = float(length)
        self.viscosity = float(viscosity)
        self.psi = psi
        self.u_sym = u
        self.p_sym = p
        self.f_sym = f
        self._u = [sp.lambdify((x, y, z), component, "numpy") for component in u]
        self._p = sp.lambdify((x, y, z), p, "numpy")
        self._f = [sp.lambdify((x, y, z), component, "numpy") for component in f]

    def velocity(self, X, Y, Z):
        return np.stack([np.broadcast_to(np.asarray(fn(X, Y, Z), dtype=np.float64), X.shape)
                         for fn in self._u], axis=-1)

    def pressure(self, X, Y, Z):
        return np.broadcast_to(np.asarray(self._p(X, Y, Z), dtype=np.float64), X.shape)

    def forcing(self, X, Y, Z):
        return np.stack([np.broadcast_to(np.asarray(fn(X, Y, Z), dtype=np.float64), X.shape)
                         for fn in self._f], axis=-1)


def gauss(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * nodes, 0.5 * weights  # mapped to the unit interval [-1/2, 1/2] offsets


def voxel_integrals(field, centres, h, order):
    """Exact-to-quadrature integral of a vector or scalar field over each voxel.

    `centres` has shape (n, 3) in physical units; the return is the integral over
    the voxel, i.e. the mean times h^3.
    """
    offset, weight = gauss(order)
    total = None
    for i, wi in zip(offset, weight):
        for j, wj in zip(offset, weight):
            for k, wk in zip(offset, weight):
                X = centres[:, 0] + i * h
                Y = centres[:, 1] + j * h
                Z = centres[:, 2] + k * h
                value = field(X, Y, Z)
                contribution = (wi * wj * wk) * value
                total = contribution if total is None else total + contribution
    return total * (h ** 3)


def face_integrals(field_component, centroids, normals, h, order):
    """Integral of u.n over each facelet (a voxel face of side h)."""
    offset, weight = gauss(order)
    axis = np.argmax(np.abs(normals), axis=1)
    tangents = np.zeros((centroids.shape[0], 2, 3), dtype=np.float64)
    for a in range(3):
        rows = np.flatnonzero(axis == a)
        others = [t for t in range(3) if t != a]
        tangents[rows, 0, others[0]] = 1.0
        tangents[rows, 1, others[1]] = 1.0
    total = np.zeros(centroids.shape[0], dtype=np.float64)
    for i, wi in zip(offset, weight):
        for j, wj in zip(offset, weight):
            point = centroids + (i * h) * tangents[:, 0, :] + (j * h) * tangents[:, 1, :]
            velocity = field_component(point[:, 0], point[:, 1], point[:, 2])
            total += (wi * wj) * np.sum(velocity * normals, axis=1)
    return total * (h * h)


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------
def cube_mask_path(root: Path, n: int) -> Path:
    path = root / "masks" / ("cube_n%d.npz" % n)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mask=np.ones((n, n, n), dtype=bool))
    return path


def lattice_sites(n: int, m: int) -> np.ndarray:
    """Structured nested lattice of m^3 sites inside an n^3 voxel cube."""
    if m > n:
        raise ValueError("more sites per side than voxels per side")
    index = np.rint((np.arange(m) + 0.5) * n / m - 0.5).astype(np.int64)
    index = np.clip(index, 0, n - 1)
    if np.unique(index).size != m:
        raise ValueError("the requested lattice is degenerate at this voxel resolution")
    zz, yy, xx = np.meshgrid(index, index, index, indexing="ij")
    return np.unique((zz * n + yy) * n + xx)


def run_level(n, m, order, root, boot, resume=True, load_functional="linear"):
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    label = "N%03d_m%02d_%s" % (n, m, load_functional)
    run_dir = root / "runs" / label
    row_path = run_dir / "row.json"
    if resume and row_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    h = float(cfg.voxel_size)
    length = float(n) * h
    manufactured = Manufactured(length, float(cfg.nu))
    mask_path = cube_mask_path(root, n)
    sites = lattice_sites(n, m)

    started = time.perf_counter()
    geom, geometry_meta, build_time = ec.build_geometry_from_seed_flat(
        sites, mask_path=mask_path, seed_spec="manufactured_cube:lattice:n%d_m%d" % (n, m)
    )
    trace_start = time.perf_counter()
    trace = boot["build_hybrid_trace_geometry"](
        ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
    )
    trace_time = float(time.perf_counter() - trace_start)

    system = boot["assemble_moment_constrained_hybrid_stokes"](
        trace,
        viscosity=float(cfg.nu),
        body_force=np.array([1.0, 0.0, 0.0]),
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]),
    )

    # ---- reference quantities, as integrals -------------------------------
    labels = cp.asnumpy(geom.labels).astype(np.int64)
    mask = cp.asnumpy(geom.mask).astype(bool)
    pore = np.flatnonzero(mask.reshape(-1))
    voxel_owner = labels.reshape(-1)[pore]
    zz, yy, xx = np.unravel_index(pore, mask.shape)
    voxel_centres = np.stack(
        [(xx + 0.5) * h, (yy + 0.5) * h, (zz + 0.5) * h], axis=1
    )
    n_cells = int(geom.n_cells)
    cell_volume = cp.asnumpy(geom.volume).astype(np.float64)

    def forcing_first_moment(X, Y, Z):
        f = manufactured.forcing(X, Y, Z)
        point = np.stack([X, Y, Z], axis=-1)
        return (f[..., :, None] * point[..., None, :]).reshape(f.shape[:-1] + (9,))

    u_voxel_integral = voxel_integrals(manufactured.velocity, voxel_centres, h, order)
    p_voxel_integral = voxel_integrals(manufactured.pressure, voxel_centres, h, order)
    f_voxel_integral = voxel_integrals(manufactured.forcing, voxel_centres, h, order)
    fx_voxel_integral = voxel_integrals(forcing_first_moment, voxel_centres, h, order)

    u_cell = np.zeros((n_cells, 3), dtype=np.float64)
    f_cell = np.zeros((n_cells, 3), dtype=np.float64)
    for component in range(3):
        np.add.at(u_cell[:, component], voxel_owner, u_voxel_integral[:, component])
        np.add.at(f_cell[:, component], voxel_owner, f_voxel_integral[:, component])
    p_cell = np.zeros(n_cells, dtype=np.float64)
    np.add.at(p_cell, voxel_owner, p_voxel_integral)
    fx_cell = np.zeros((n_cells, 9), dtype=np.float64)
    for component in range(9):
        np.add.at(fx_cell[:, component], voxel_owner, fx_voxel_integral[:, component])
    u_cell /= cell_volume[:, None]
    f_cell /= cell_volume[:, None]
    p_cell /= cell_volume

    # ---- load vector -------------------------------------------------------
    # The assembly builds  rhs = sum_K |K| R_K^T g  for one constant g,
    # i.e. it pairs the force with the cell-MEAN part of the reconstruction only.
    # For a uniform force that is exact, because the first moment of (x - x_K)
    # over the cell vanishes.  For a varying force it drops
    #     ( int_K f (x)(x - x_K)^T ) : G_K ,
    # which is O(H) and does not vanish under refinement.  Both functionals are
    # assembled and reported so the method's own order is not confounded with the
    # order of the load.
    #
    # `cell_gradient` comes from the assembly reproduction whose A, D and B are
    # checked bitwise-identical to the pipeline's (see stabilization_sweep.bitwise_gate).
    reproduced = ss.assemble_with_tau(
        trace, viscosity=float(cfg.nu), body_force=np.array([1.0, 0.0, 0.0]),
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]), tau=1.0,
    )
    if not (
        np.array_equal(reproduced["A"].indptr, system.velocity_matrix.indptr)
        and np.array_equal(reproduced["A"].data, system.velocity_matrix.data)
        and np.array_equal(reproduced["D"].data, system.divergence_matrix.data)
        and np.array_equal(reproduced["B"], system.body_force_matrix)
    ):
        raise RuntimeError(
            "the assembly reproduction used for the gradient basis is not bitwise "
            "identical to the production assembly on this case"
        )
    centroid = np.asarray(trace.cell_centroid, dtype=np.float64)
    moment = fx_cell.reshape(n_cells, 3, 3) - (
        f_cell[:, :, None] * cell_volume[:, None, None] * centroid[:, None, :]
    )
    load_mean = np.zeros(system.body_force_matrix.shape[0], dtype=np.float64)
    load_linear = np.zeros_like(load_mean)
    for cell, (global_dofs, recovery) in enumerate(system.cell_recovery):
        if not global_dofs.size:
            continue
        mean_part = float(cell_volume[cell]) * (recovery.T @ f_cell[cell])
        load_mean[global_dofs] += mean_part
        gradient_basis = reproduced["cell_gradient"][cell]
        moment_part = np.einsum("aij,ij->a", gradient_basis, moment[cell])
        load_linear[global_dofs] += mean_part + moment_part
    load = load_linear if load_functional == "linear" else load_mean
    system.body_force_matrix[:, 0] = load
    system.body_force_matrix[:, 1:] = 0.0

    factorization = boot["build_hybrid_trace_factorization"](
        system,
        solver=str(ec.FORWARD_ARGS["linear_solver"]),
        iterative_rtol=float(ec.FORWARD_ARGS["linear_rtol"]),
        iterative_maxiter=int(ec.FORWARD_ARGS["linear_maxiter"]),
        iterative_refinement_steps=int(ec.FORWARD_ARGS["linear_refinement_steps"]),
        velocity_lu_ordering="COLAMD",
    )
    result = boot["solve_moment_constrained_hybrid_stokes"](system, factorization=factorization)

    # ---- errors -----------------------------------------------------------
    U = np.asarray(result["U"], dtype=np.float64)
    total_volume = float(cell_volume.sum())

    def weighted_norm(field):
        if field.ndim == 1:
            return float(np.sqrt(np.sum(cell_volume * field ** 2)))
        return float(np.sqrt(np.sum(cell_volume * np.sum(field ** 2, axis=1))))

    e_u = 100.0 * weighted_norm(U - u_cell) / max(weighted_norm(u_cell), 1e-300)
    e_p, pressure_detail = ec.gauge_invariant_pressure_error(
        np.asarray(result["p"], dtype=np.float64), p_cell, cell_volume
    )

    face_exact = face_integrals(
        manufactured.velocity, trace.face_centroid, trace.face_normal, h, order
    )
    aggregate_exact = np.bincount(
        trace.face_parent_edge,
        weights=face_exact,
        minlength=trace.stored_edge_sign_from_sorted.size,
    ) * trace.stored_edge_sign_from_sorted
    phi = np.asarray(result["phi"], dtype=np.float64)
    e_phi = 100.0 * float(
        np.linalg.norm(phi - aggregate_exact) / max(np.linalg.norm(aggregate_exact), 1e-300)
    )

    # ---- dimensionless algebraic mass balance -----------------------------
    z = result["trace_coefficients"].reshape(-1)
    D = system.divergence_matrix
    Dz = np.asarray(D @ z).reshape(-1)
    abs_Dz = np.asarray(abs(D) @ np.abs(z)).reshape(-1)
    eps = np.finfo(np.float64).eps
    eta_m = float(
        np.max(np.abs(Dz)) / (np.max(abs_Dz) + eps * max(1.0, float(np.max(abs_Dz))))
    )

    row = {
        "protocol_id": PROTOCOL_ID,
        "case": "manufactured_cube",
        "level_label": label,
        "voxels_per_side": int(n),
        "sites_per_side": int(m),
        "domain_length": length,
        "voxel_size": h,
        "h_over_L": 1.0 / float(n),
        "H_over_L": 1.0 / float(m),
        "H_over_h": float(n) / float(m),
        "quadrature_order": int(order),
        "load_functional": load_functional,
        "periodic_x": bool(getattr(cfg, "periodic_x", False)),
        "viscosity": float(cfg.nu),
        "N_f": int(mask.sum()),
        "N_sites": int(sites.size),
        "N_cv": n_cells,
        "N_facelets": int(trace.n_facelets),
        "N_patches": int(trace.n_patches),
        "N_trace_modes": int(trace.n_trace_modes),
        "N_z": int(3 * trace.n_trace_modes),
        "N_p_after_gauge": n_cells - 1,
        "N_system": int(3 * trace.n_trace_modes + n_cells - 1),
        "nnz_A": int(system.velocity_matrix.nnz),
        "nnz_D": int(system.divergence_matrix.nnz),
        "e_u_percent": e_u,
        "e_p_percent": e_p,
        "e_phi_percent": e_phi,
        "mass_inf_per_volume": float(result["mass_inf_per_volume"]),
        "mass_backward_error_dimensionless": eta_m,
        "momentum_residual_inf": float(result["momentum_residual_inf"]),
        "linear_solver_iterations": int(result["linear_solver_iterations"]),
        "linear_solver_relative_residual": float(result["linear_solver_relative_residual"]),
        "pressure_gauge": str(result["pressure_gauge"]),
        "velocity_matrix_symmetry_defect": float(result["velocity_matrix_symmetry_defect"]),
        "t_geometry_build_s": build_time,
        "t_trace_geometry_s": trace_time,
        "t_matrix_assembly_s": float(system.assembly_time_s),
        "t_factorization_s": float(factorization.factorization_time_s),
        "t_solve_s": float(result["solve_time_s"]),
        "factor_memory_MiB": float(result["factor_bytes"]) / float(2 ** 20),
        "wall_seconds": float(time.perf_counter() - started),
        "solver_formulation_id": str(result["solver_formulation_id"]),
        "pressure_multiplier_vs_reference_cosine": pressure_detail["multiplier_vs_reference_cosine"],
        "e_p_percent_multiplier_convention": pressure_detail["e_p_percent_multiplier_convention"],
        "run_directory": str(run_dir),
    }
    ec.write_json(row_path, row)
    np.savez_compressed(
        run_dir / "state.npz",
        U=U, U_exact_cell_mean=u_cell,
        p=np.asarray(result["p"], dtype=np.float64), p_exact_cell_mean=p_cell,
        phi=phi, phi_exact=aggregate_exact,
        volume=cell_volume, f_cell=f_cell, sites=sites,
    )
    del system, factorization, geom
    ec.free_gpu()
    print(
        "[mms] %s N_cv=%d N_system=%d H/h=%.2f e_u=%.4f%% e_p=%.4f%% e_phi=%.4f%% "
        "eta_m=%.2e (%.1fs)"
        % (label, row["N_cv"], row["N_system"], row["H_over_h"], e_u, e_p, e_phi,
           eta_m, row["wall_seconds"]),
        flush=True,
    )
    return row


def quadrature_check(length, viscosity, n, orders=(3, 4, 5, 6)):
    """Verify the quadrature order in-run instead of assuming it."""
    manufactured = Manufactured(length, viscosity)
    h = length / n
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    centres = np.stack(
        [(xx.ravel() + 0.5) * h, (yy.ravel() + 0.5) * h, (zz.ravel() + 0.5) * h], axis=1
    )
    record = {}
    previous = None
    for order in orders:
        integral = voxel_integrals(manufactured.pressure, centres, h, order)
        total = float(integral.sum())
        divergence_probe = float(
            np.abs(voxel_integrals(manufactured.velocity, centres, h, order)).sum()
        )
        record[str(order)] = {
            "pressure_integral_over_domain": total,
            "pressure_mean_relative_to_amplitude": total / (length ** 3),
            "velocity_absolute_integral": divergence_probe,
        }
        if previous is not None:
            record[str(order)]["change_from_previous_order"] = abs(total - previous)
        previous = total
    return record


def boundary_check(length, viscosity, samples=4096, seed=20260827):
    """The manufactured velocity must vanish on every face of the cube."""
    manufactured = Manufactured(length, viscosity)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for axis in range(3):
        for value in (0.0, length):
            point = rng.uniform(0.0, length, size=(samples, 3))
            point[:, axis] = value
            velocity = manufactured.velocity(point[:, 0], point[:, 1], point[:, 2])
            worst = max(worst, float(np.max(np.abs(velocity))))
    return worst


def divergence_check(length, viscosity, samples=4096, seed=20260827):
    """Numerical confirmation that div u = 0 away from the symbolic identity."""
    x, y, z = sp.symbols("x y z", real=True)
    manufactured = Manufactured(length, viscosity)
    div = sum(sp.diff(manufactured.u_sym[i], (x, y, z)[i]) for i in range(3))
    fn = sp.lambdify((x, y, z), sp.simplify(div), "numpy")
    rng = np.random.default_rng(seed)
    point = rng.uniform(0.0, length, size=(samples, 3))
    value = np.asarray(fn(point[:, 0], point[:, 1], point[:, 2]), dtype=np.float64)
    return float(np.max(np.abs(np.broadcast_to(value, (samples,)))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/table_03")
    parser.add_argument(
        "--levels", default="16:4,24:6,32:8,48:12,64:16",
        help="comma separated voxels_per_side:sites_per_side pairs",
    )
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--load-functionals", default="linear,mean",
                        help="comma separated: linear (cell-mean + first moment) and/or mean")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    boot = ec.bootstrap(root / "bootstrap")
    cfg = boot["cfg"]
    # A manufactured cube is walled on all six faces; the default physics
    # config is x-periodic, so the flag is switched here and recorded per row.
    cfg.periodic_x = False

    checks = {
        "boundary_velocity_max": boundary_check(16.0, float(cfg.nu)),
        "divergence_max": divergence_check(16.0, float(cfg.nu)),
        "quadrature": quadrature_check(16.0, float(cfg.nu), 16),
        "forcing_derivation": "two independent symbolic derivations agreed exactly",
    }
    ec.write_json(root / "manufactured_checks.json", checks)
    print("[check] max |u| on the cube boundary = %.3e" % checks["boundary_velocity_max"])
    print("[check] max |div u| in the interior   = %.3e" % checks["divergence_max"])
    for order, item in checks["quadrature"].items():
        print("[check] order %s: integral of p over the cube = %+.6e%s"
              % (order, item["pressure_integral_over_domain"],
                 ("  (change %.2e)" % item["change_from_previous_order"])
                 if "change_from_previous_order" in item else ""))
    if args.checks_only:
        return 0

    rows = []
    for item in args.levels.split(","):
        n, m = (int(v) for v in item.split(":"))
        for functional in args.load_functionals.split(","):
            rows.append(run_level(n, m, args.order, root, boot, resume=args.resume,
                                  load_functional=functional))
    columns = sorted({k for r in rows for k in r})
    lead = ["protocol_id", "case", "level_label", "load_functional", "voxels_per_side", "sites_per_side",
            "h_over_L", "H_over_L", "H_over_h", "N_cv", "N_system",
            "e_u_percent", "e_p_percent", "e_phi_percent"]
    columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
    ec.write_csv(root / "manufactured_stokes.csv", rows, columns)
    print("[write] " + str(root / "manufactured_stokes.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
