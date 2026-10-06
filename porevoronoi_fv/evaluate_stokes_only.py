"""E0 analysis: gates E0-a on the pure solve of the published partition, one JSON per case plus a summary.

Stages (requeue-safe, each with its own checkpoint):
  protocol  write <out_dir>/protocol.json (the declared gates) once; refuse to run if an existing one differs
  eig       spectral quantities for the declared velocity tolerance (SuperLU shift-invert; may be timed out)
  a         apply the gates to e0.npz and write <out_dir>/<case>/a.json
  summary   collect <out_dir>/*/a.json into <out_dir>/summary.json
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
import os
import time

import config_io
config_io.thread_env(1)
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh, splu, svds

import case_loader
import readouts
import stokes_solve
import run_stokes_only
import velocity_recovery as rp

PROTOCOL = {
    "id": "stokes-only-gates-v1",
    "E0_solver": "paper CPU MINRES: hybrid_voronoi_trace factorization(minres) + solve, rtol 1e-14, maxiter 50000, "
                 "refinement 1; hybrid_voronoi_trace's own gate (info==0 and ||Kx-rhs||2/||rhs||2 <= max(20 rtol, "
                 "1e-12)) raises on failure; forward operator = vectorised assembly (fast_assembly), whose equality "
                 "with the production loop is gate G1",
    "G1": "fast_assembly.gate(production loop assembly, vectorised assembly) on the published complex with periodic_x: "
          "A and D max relative entry difference < 1e-12 with pattern differences only below 1e-12, body-force "
          "matrix and cell-recovery blocks < 1e-12, identical recovery dofs",
    "E0a_table": "for N_c, e_K,arch^s, e_K,flux^s, e_phi, e_u,cell: the E0 value formatted to the printed number "
                 "of decimals equals the printed Table literal (string equality, sign included)",
    "E0a_mass": "r_inf^m of a new solve is a round-off quantity whose printed digits are not reproducible; gate: "
                "eta_m(z_E0) <= 1e-14 (mass backward error of the paper); r_inf^m and its printed-literal "
                "comparison are reported, not gated",
    "E0a_cellU": "||V^(1/2)(U_E0 - U_arch)||_2 <= tau, tau = s_R (||r_E0||_2 + ||r_arch||_2) / lam_lo + "
                 "||V^(1/2)(U_arch - R z_arch)||_2, where K is the reduced MINRES KKT of the driver's forward "
                 "assembly, r = K x - rhs for the E0 state and for the archived state (both in this K, so assembly "
                 "differences are inside r_arch), lam_lo = |lambda|_min(K) - ||K v - lambda v||_2 from ARPACK "
                 "shift-invert at sigma = 0 (SuperLU), s_R = sigma_max(V^(1/2) R) from ARPACK svds. Derivation: "
                 "x_E0 - x_arch = K^-1 (r_E0 - r_arch) and U = R z. NOT_RUN if the eig stage did not finish",
    "stated_before_running": True,
}


def stage_protocol(cfg):
    path = os.path.join(cfg["out_dir"], "protocol.json")
    if os.path.exists(path):
        old = config_io.load_json(path)
        if {k: old.get(k) for k in PROTOCOL} != PROTOCOL:
            raise SystemExit("protocol.json exists and differs from the declared gates; refusing to continue")
        print(dict(stage="protocol", status="unchanged", sha256=config_io.sha(path)), flush=True)
        return
    config_io.save_json(path, dict(PROTOCOL, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   code=config_io.code_hashes()))
    print(dict(stage="protocol", status="written", sha256=config_io.sha(path)), flush=True)


def spectral(K, RV, label="eig"):
    """|lambda|_min of symmetric K by ARPACK shift-invert at 0 (SuperLU) and sigma_max of RV by ARPACK svds."""
    K = K.tocsc()
    t = time.perf_counter()
    print(dict(stage=label, status="factorize", n=K.shape[0], nnz=K.nnz), flush=True)
    lu = splu(K, permc_spec="MMD_AT_PLUS_A")
    t_lu = time.perf_counter() - t
    print(dict(stage=label, status="factorized", seconds=t_lu, fill=int(lu.L.nnz + lu.U.nnz)), flush=True)
    beat = config_io.Beat(label, every=60.0)

    def op(v):
        beat()
        return lu.solve(np.asarray(v, dtype=np.float64))
    OPinv = LinearOperator(K.shape, matvec=op, dtype=np.float64)
    t = time.perf_counter()
    v0 = np.random.default_rng(11).normal(size=K.shape[0])
    vals, vecs = eigsh(K, k=1, sigma=0.0, which="LM", OPinv=OPinv, tol=1e-8, maxiter=10000, v0=v0)
    lam = float(vals[0])
    v = vecs[:, 0]
    eig_res = float(np.linalg.norm(K @ v - lam * v) / np.linalg.norm(v))
    t_eig = time.perf_counter() - t
    u, s, vt = svds(RV, k=1, which="LM", tol=1e-10, random_state=np.random.default_rng(12))
    s_R = float(s[0])
    sv_res = float(np.linalg.norm(RV @ vt[0] - s_R * u[:, 0]))
    return dict(n=int(K.shape[0]), kkt_nnz=int(K.nnz), lu_fill_nnz=int(lu.L.nnz + lu.U.nnz), t_factorize_s=t_lu,
                t_eigsh_s=t_eig, lambda_min_abs=abs(lam), lambda_signed=lam, eig_residual=eig_res,
                lambda_lower=max(abs(lam) - eig_res, 0.0), sigma_max_RV=s_R, sigma_residual=sv_res,
                sigma_upper=s_R + sv_res)


def stage_eig(cfg, code, out):
    path = os.path.join(out, "eig.json")
    if config_io.done(path, config_io.load_json):
        print(dict(case=code, stage="eig", status="checkpoint"), flush=True)
        return
    case, part, _ = run_stokes_only.build_published(cfg, code)
    system, _ = stokes_solve.assemble(case, part["trace"], "fast", cfg["viscous_form"])
    K = stokes_solve.reduced_kkt(system.velocity_matrix, system.divergence_matrix)
    R, _, _ = rp.recovery_operators(part["trace"], system)
    sq = np.sqrt(np.repeat(np.asarray(part["geom"].volume, np.float64), 3))
    res = spectral(K, (sparse.diags(sq) @ R).tocsr(), label=f"{code}-eig")
    res.update(case=code, environment=config_io.environment())
    config_io.save_json(path, res)
    print(dict(case=code, stage="eig", status="done", lam=res["lambda_signed"], eig_res=res["eig_residual"],
               s_R=res["sigma_max_RV"]), flush=True)


def stage_a(cfg, code, out):
    path = os.path.join(out, "a.json")
    e0 = config_io.load_json(os.path.join(out, "e0.json"))
    npz_path = os.path.join(out, "e0.npz")
    if config_io.sha(npz_path) != e0["npz_sha256"]:
        raise RuntimeError("e0.npz hash mismatch")
    new = np.load(npz_path)
    lits = config_io.load_json(os.path.join(cfg["in_dir"], cfg["table_literals"]))["rows"][code]
    case, part, _ = run_stokes_only.build_published(cfg, code)
    case.state = {k: np.asarray(v) for k, v in np.load(case.paths["state"]).items()}
    if not np.array_equal(new["seeds"], part["seeds"]) or not np.array_equal(new["volume"], part["geom"].volume):
        raise RuntimeError("E0 state belongs to a different partition")
    system, _ = stokes_solve.assemble(case, part["trace"], "fast", cfg["viscous_form"])
    D = system.divergence_matrix
    z_new, p_new, U_new, phi_new = new["z"], new["p"], new["U"], new["phi"]
    row, U_ref, _ = readouts.table_row(case, part, phi_new, U_new, z=z_new, D=D)
    printed = dict(N_c=dict(value=row["N_c"], printed=lits["N_c"], formatted=str(row["N_c"]),
                            match=str(row["N_c"]) == lits["N_c"]),
                   eK_arch_s=readouts.literal_fixed(row["eK_arch_s"], lits["eK_arch_s"]),
                   eK_flux_s=readouts.literal_fixed(row["eK_flux_s"], lits["eK_flux_s"]),
                   e_phi=readouts.literal_fixed(row["e_phi"], lits["e_phi"]),
                   e_u_cell=readouts.literal_fixed(row["e_u_cell"], lits["e_u_cell"]))
    r_inf_literal = readouts.literal_sci(row["r_inf_m"], lits["r_inf_mantissa"], lits["r_inf_exponent"])
    gates = {}
    g1p = os.path.join(out, "g1.json")
    gates["G1"] = ("PASS" if config_io.load_json(g1p)["pass_"] else "FAIL") if os.path.exists(g1p) else "NOT_RUN"
    gates["E0a_table"] = "PASS" if all(v["match"] for v in printed.values()) else "FAIL"
    gates["E0a_mass"] = "PASS" if row["eta_m"] <= 1e-14 else "FAIL"
    # archived-state comparison
    st = case.state
    K = stokes_solve.reduced_kkt(system.velocity_matrix, D)
    rhs = np.concatenate([system.rhs_trace, np.zeros(D.shape[0] - 1)])
    x_new = stokes_solve.state_vector(z_new, p_new)
    x_arch = stokes_solve.state_vector(st["trace_coefficients"], st["p"])
    r_new = K @ x_new - rhs
    r_arch = K @ x_arch - rhs
    R, G, gdef = rp.recovery_operators(part["trace"], system)
    sqV = np.sqrt(np.asarray(part["geom"].volume, np.float64))[:, None]
    U_arch = np.asarray(st["U"], np.float64)
    U_arch_rec = (R @ np.asarray(st["trace_coefficients"], np.float64).ravel()).reshape(-1, 3)
    delta_R = float(np.linalg.norm(sqV * (U_arch - U_arch_rec)))
    obs = float(np.linalg.norm(sqV * (U_new - U_arch)))
    norm_arch = float(np.linalg.norm(sqV * U_arch))
    cellU = dict(observed_VL2_diff=obs, observed_relative=obs / norm_arch, r_E0_2=float(np.linalg.norm(r_new)),
                 r_arch_2=float(np.linalg.norm(r_arch)), rhs_2=float(np.linalg.norm(rhs)),
                 eta_E0=float(np.linalg.norm(r_new) / np.linalg.norm(rhs)),
                 eta_arch_in_driver_system=float(np.linalg.norm(r_arch) / np.linalg.norm(rhs)),
                 eta_arch_recorded=float(st["linear_solver_relative_residual"]), delta_R=delta_R,
                 z_relative_diff=float(np.linalg.norm(x_new[:z_new.size] - x_arch[:z_new.size]) / np.linalg.norm(x_arch[:z_new.size])))
    # the solver gate on the produced numbers: recorded info/residual and the residual recomputed from e0.npz
    limit = float(e0["residual_gate_limit"])
    gates["E0_solver"] = ("PASS" if (int(e0["linear_solver_info"]) == 0 and float(e0["linear_solver_relative_residual"])
                                    <= limit and cellU["eta_E0"] <= limit) else "FAIL")
    eig_path = os.path.join(out, "eig.json")
    if os.path.exists(eig_path):
        eig = config_io.load_json(eig_path)
        if eig["lambda_lower"] > 0:
            tau = eig["sigma_upper"] * (cellU["r_E0_2"] + cellU["r_arch_2"]) / eig["lambda_lower"] + delta_R
            cellU.update(tau=tau, tau_relative=tau / norm_arch, eig=eig)
            gates["E0a_cellU"] = "PASS" if obs <= tau else "FAIL"
        else:
            cellU.update(eig=eig, note="eigen lower bound not positive")
            gates["E0a_cellU"] = "NOT_RUN"
    else:
        gates["E0a_cellU"] = "NOT_RUN"
    Hfield = rp.point_operator(part["xyz"], part["geom"].labels.ravel()[part["pore"]], part["trace"], R, G)
    res = dict(case=code, tag=case.tag, gates=gates, protocol_id=PROTOCOL["id"], table_row_E0=row, printed=printed,
               r_inf_m_literal_comparison_not_gated=r_inf_literal, cell_velocity=cellU,
               full_field_voxel_error_E0=readouts.full_field_error(z_new, Hfield, case),
               full_field_voxel_error_archived=readouts.full_field_error(np.asarray(st["trace_coefficients"]).ravel(), Hfield, case),
               recovery_gradient_divergence_defect=gdef, e0_receipt=e0, environment=config_io.environment(),
               code=config_io.code_hashes())
    config_io.save_json(path, res)
    print(dict(case=code, stage="a", gates=gates, printed={k: (v["formatted"], v["printed"]) for k, v in printed.items()},
               eta_m=row["eta_m"], obs=obs, tau=cellU.get("tau")), flush=True)


def stage_summary(cfg):
    rows = {}
    for code in sorted(cfg["cases"]):
        p = os.path.join(cfg["out_dir"], code, "a.json")
        if not os.path.exists(p):
            rows[code] = dict(status="missing")
            continue
        a = config_io.load_json(p)
        r = a["table_row_E0"]
        rows[code] = dict(gates=a["gates"], N_c=r["N_c"], eK_arch_s=r["eK_arch_s"], eK_flux_s=r["eK_flux_s"],
                          e_phi=r["e_phi"], e_u_cell=r["e_u_cell"], r_inf_m=r["r_inf_m"], eta_m=r["eta_m"],
                          full_field_voxel_error_E0=a["full_field_voxel_error_E0"],
                          iterations=a["e0_receipt"]["linear_solver_iterations"],
                          relative_residual=a["e0_receipt"]["linear_solver_relative_residual"],
                          solve_s=a["e0_receipt"]["solve_call_s"], cellU_observed=a["cell_velocity"]["observed_VL2_diff"],
                          cellU_tau=a["cell_velocity"].get("tau"))
    config_io.save_json(os.path.join(cfg["out_dir"], "summary.json"), dict(protocol=PROTOCOL["id"], cases=rows,
                                                                          environment=config_io.environment()))
    print(json.dumps({c: v.get("gates", v) for c, v in rows.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--case")
    ap.add_argument("--stage", choices=["protocol", "eig", "a", "summary"], required=True)
    a = ap.parse_args()
    cfg = config_io.load_cfg(a.cfg)
    config_io.ensure_dir(cfg["out_dir"])
    if a.stage == "protocol":
        return stage_protocol(cfg)
    if a.stage == "summary":
        return stage_summary(cfg)
    out = config_io.ensure_dir(os.path.join(cfg["out_dir"], a.case))
    {"eig": stage_eig, "a": stage_a}[a.stage](cfg, a.case, out)


if __name__ == "__main__":
    main()
