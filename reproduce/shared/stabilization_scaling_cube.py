"""Does the residual-stabilization weight carry the right length scale?

The retained weight is  w = 2 nu A_facelet / h  with h the VOXEL size.  A
dimensional estimate of the two terms of the local matrix gives

    consistent    ~ 2 nu V_K |sym G|^2          ~ nu H
    stabilization ~ sum_facelets w |residual|^2 ~ 2 nu h . 6 (H/h)^2 = 12 nu H^2/h

so their ratio grows like H/h: the coarser a control volume is *in voxels*, the
more the stabilization dominates, at fixed cell size in physical units.  If that
estimate is right, then on the manufactured cube two levels with the same H/L
but different H/h must differ in accuracy, and the optimal global multiplier tau
must scale like h/H.

This module measures exactly that.  It sweeps tau on cube levels chosen so that
H/L is held fixed while H/h varies, using the bitwise-checked assembly
reproduction, and reports where each level's error is minimised.

A confirmed h/H scaling would say the stabilization length should be the CELL
scale, not the voxel scale.  A refuted one would say the H/h dependence has some
other origin.  Both outcomes are reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402
import manufactured_stokes as ms  # noqa: E402
import stabilization_sweep as ss  # noqa: E402

PROTOCOL_ID = "pvfv_stabilization_scaling"


def run(n, m, taus, order, root, boot):
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    h = float(cfg.voxel_size)
    length = float(n) * h
    manufactured = ms.Manufactured(length, float(cfg.nu))
    mask_path = ms.cube_mask_path(root, n)
    sites = ms.lattice_sites(n, m)

    geom, _meta, _t = ec.build_geometry_from_seed_flat(
        sites, mask_path=mask_path, seed_spec="stab_scaling:n%d_m%d" % (n, m)
    )
    trace = boot["build_hybrid_trace_geometry"](
        ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
    )
    one = ss.assemble_with_tau(
        trace, viscosity=float(cfg.nu), body_force=np.array([1.0, 0.0, 0.0]),
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]), tau=1.0,
    )
    zero = ss.assemble_with_tau(
        trace, viscosity=float(cfg.nu), body_force=np.array([1.0, 0.0, 0.0]),
        viscous_form=str(ec.FORWARD_ARGS["viscous_form"]), tau=0.0,
    )
    A0, A_stab, D = zero["A"], (one["A"] - zero["A"]).tocsr(), one["D"]

    labels = cp.asnumpy(geom.labels).astype(np.int64)
    mask = cp.asnumpy(geom.mask).astype(bool)
    pore = np.flatnonzero(mask.reshape(-1))
    voxel_owner = labels.reshape(-1)[pore]
    zz, yy, xx = np.unravel_index(pore, mask.shape)
    centres = np.stack([(xx + 0.5) * h, (yy + 0.5) * h, (zz + 0.5) * h], axis=1)
    n_cells = int(geom.n_cells)
    cell_volume = cp.asnumpy(geom.volume).astype(np.float64)

    def forcing_first_moment(X, Y, Z):
        f = manufactured.forcing(X, Y, Z)
        point = np.stack([X, Y, Z], axis=-1)
        return (f[..., :, None] * point[..., None, :]).reshape(f.shape[:-1] + (9,))

    u_int = ms.voxel_integrals(manufactured.velocity, centres, h, order)
    p_int = ms.voxel_integrals(manufactured.pressure, centres, h, order)
    f_int = ms.voxel_integrals(manufactured.forcing, centres, h, order)
    fx_int = ms.voxel_integrals(forcing_first_moment, centres, h, order)

    u_cell = np.zeros((n_cells, 3)); f_cell = np.zeros((n_cells, 3))
    fx_cell = np.zeros((n_cells, 9)); p_cell = np.zeros(n_cells)
    for c in range(3):
        np.add.at(u_cell[:, c], voxel_owner, u_int[:, c])
        np.add.at(f_cell[:, c], voxel_owner, f_int[:, c])
    for c in range(9):
        np.add.at(fx_cell[:, c], voxel_owner, fx_int[:, c])
    np.add.at(p_cell, voxel_owner, p_int)
    u_cell /= cell_volume[:, None]; f_cell /= cell_volume[:, None]; p_cell /= cell_volume

    centroid = np.asarray(trace.cell_centroid, dtype=np.float64)
    moment = fx_cell.reshape(n_cells, 3, 3) - (
        f_cell[:, :, None] * cell_volume[:, None, None] * centroid[:, None, :]
    )
    load = np.zeros(A0.shape[0], dtype=np.float64)
    for cell, (dofs, recovery) in enumerate(one["cell_recovery"]):
        if not dofs.size:
            continue
        load[dofs] += float(cell_volume[cell]) * (recovery.T @ f_cell[cell])
        load[dofs] += np.einsum("aij,ij->a", one["cell_gradient"][cell], moment[cell])

    face_exact = ms.face_integrals(
        manufactured.velocity, trace.face_centroid, trace.face_normal, h, order
    )
    phi_exact = np.bincount(
        trace.face_parent_edge, weights=face_exact,
        minlength=trace.stored_edge_sign_from_sorted.size,
    ) * trace.stored_edge_sign_from_sorted

    rows = []
    for tau in taus:
        A = (A0 + tau * A_stab).tocsr()
        A = (0.5 * (A + A.T)).tocsr()
        out = ss.solve_at_tau(
            A, D, load,
            rtol=float(ec.FORWARD_ARGS["linear_rtol"]),
            maxiter=int(ec.FORWARD_ARGS["linear_maxiter"]),
            refinement_steps=int(ec.FORWARD_ARGS["linear_refinement_steps"]),
        )
        row = {
            "protocol_id": PROTOCOL_ID, "voxels_per_side": n, "sites_per_side": m,
            "H_over_L": 1.0 / m, "H_over_h": n / m, "tau": tau,
            "tau_times_H_over_h": tau * n / m,
            "N_cv": n_cells, "N_system": int(A.shape[0] + n_cells - 1),
            "solver_status": out["status"],
        }
        if "z" in out:
            z = out["z"]
            U = np.zeros((n_cells, 3))
            for cell, (dofs, recovery) in enumerate(one["cell_recovery"]):
                if dofs.size:
                    U[cell] = recovery @ z[dofs]
            coefficients = z.reshape(trace.n_trace_modes, 3)
            face_velocity = np.zeros((trace.n_facelets, 3))
            for column in range(trace.face_mode_ids.shape[1]):
                mode = trace.face_mode_ids[:, column]
                valid = mode >= 0
                if np.any(valid):
                    face_velocity[valid] += (
                        trace.face_mode_values[valid, column, None] * coefficients[mode[valid]]
                    )
            flux = (trace.voxel_size ** 2) * np.sum(face_velocity * trace.face_normal, axis=1)
            phi = np.bincount(
                trace.face_parent_edge, weights=flux,
                minlength=trace.stored_edge_sign_from_sorted.size,
            ) * trace.stored_edge_sign_from_sorted
            e_u = 100.0 * float(
                np.sqrt(np.sum(cell_volume * np.sum((U - u_cell) ** 2, axis=1)))
                / np.sqrt(np.sum(cell_volume * np.sum(u_cell ** 2, axis=1)))
            )
            e_p, _ = ec.gauge_invariant_pressure_error(out["p"], p_cell, cell_volume)
            e_phi = 100.0 * float(
                np.linalg.norm(phi - phi_exact) / np.linalg.norm(phi_exact)
            )
            row.update({
                "e_u_percent": e_u, "e_p_percent": e_p, "e_phi_percent": e_phi,
                "linear_solver_iterations": out["iterations"],
                "linear_solver_relative_residual": out["relative_residual"],
            })
        rows.append(row)
        print(
            "[stab] N=%3d m=%2d H/h=%5.2f tau=%-8.4g tau*H/h=%-8.4g %s "
            "e_u=%8.3f%% e_p=%8.3f%% e_phi=%8.3f%%"
            % (n, m, n / m, tau, tau * n / m, out["status"],
               row.get("e_u_percent", float("nan")), row.get("e_p_percent", float("nan")),
               row.get("e_phi_percent", float("nan"))),
            flush=True,
        )
    del geom
    ec.free_gpu()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/figure_06/voxel_refinement")
    parser.add_argument("--levels", default="16:4,32:8,48:12,32:4,48:4,64:4",
                        help="voxels:sites pairs; pairs with equal H/L and different H/h are the test")
    parser.add_argument("--taus", default="0.0625,0.125,0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--order", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    boot = ec.bootstrap(root / "bootstrap")
    boot["cfg"].periodic_x = False
    taus = [float(t) for t in args.taus.split(",") if t]

    rows = []
    for item in args.levels.split(","):
        n, m = (int(v) for v in item.split(":"))
        rows.extend(run(n, m, taus, args.order, root, boot))
    columns = sorted({k for r in rows for k in r})
    lead = ["protocol_id", "voxels_per_side", "sites_per_side", "H_over_L", "H_over_h",
            "tau", "tau_times_H_over_h", "e_u_percent", "e_p_percent", "e_phi_percent"]
    columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
    ec.write_csv(root / "stabilization_scaling.csv", rows, columns)
    print("[write] " + str(root / "stabilization_scaling.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
