"""No-solve self-tests of the E0 analysis wiring.

1. spectral(): ARPACK shift-invert |lambda|_min and svds sigma_max on the tiny periodic instance equal dense
   numpy eigvalsh / svd (the tolerance instrument measures what it claims).
2. analysis dry run: an e0.npz/e0.json is fabricated FROM THE ARCHIVED STATE (flagged 'dry_run_from_archived_state')
   in a separate output tree; evaluate_stokes_only.stage_a must then give E0a_table PASS, E0a_mass PASS, E0_solver PASS and an
   observed cell-velocity difference of exactly 0. A second fabricated state with one trace coefficient perturbed
   (and its stored cell velocity) perturbed must FAIL E0a_cellU against a synthetic tau = delta_R, and FAIL E0a_mass and E0_solver (instrument bites). No flow solve is performed.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import copy
import os

import config_io
config_io.thread_env(1)
import numpy as np
from scipy import sparse

import evaluate_stokes_only
import run_stokes_only
import readouts
import stokes_solve
import velocity_recovery as rp
import check_selectors_and_assembly


def spectral_tiny():
    case, seeds = check_selectors_and_assembly.tiny_case()
    part = check_selectors_and_assembly.cell_complex.build(case, seeds, order="paper")
    system, _ = stokes_solve.assemble(case, part["trace"], "fast")
    K = stokes_solve.reduced_kkt(system.velocity_matrix, system.divergence_matrix)
    R, _, _ = rp.recovery_operators(part["trace"], system)
    RV = (sparse.diags(np.sqrt(np.repeat(part["geom"].volume, 3))) @ R).tocsr()
    got = evaluate_stokes_only.spectral(K, RV, label="tiny")
    dense_lam = float(np.min(np.abs(np.linalg.eigvalsh(K.toarray()))))
    dense_sig = float(np.linalg.svd(RV.toarray(), compute_uv=False)[0])
    return dict(arpack_lambda=got["lambda_min_abs"], dense_lambda=dense_lam, arpack_sigma=got["sigma_max_RV"],
                dense_sigma=dense_sig, n=got["n"],
                pass_=bool(abs(got["lambda_min_abs"] - dense_lam) <= 1e-6 * dense_lam
                           and abs(got["sigma_max_RV"] - dense_sig) <= 1e-8 * dense_sig))


def fabricate(cfg, code, out, perturb=False):
    case, part, _ = run_stokes_only.build_published(cfg, code)
    st = np.load(case.paths["state"])
    z = np.asarray(st["trace_coefficients"], np.float64).ravel().copy()
    U = np.asarray(st["U"]).copy()
    if perturb:
        z[0] += 1e-6
        U[0, 0] += 1e-6
    config_io.ensure_dir(out)
    npz = os.path.join(out, "e0.npz")
    config_io.save_npz(npz, z=z, p=st["p"], U=U, phi=st["phi"], face_flux_sorted=st["face_flux_sorted"],
                       volume=part["geom"].volume, seeds=part["seeds"])
    config_io.save_json(os.path.join(out, "e0.json"), dict(dry_run_from_archived_state=True,
                                                           npz_sha256=config_io.sha(npz),
                                                           linear_solver_info=0, linear_solver_relative_residual=float(
                                                               st["linear_solver_relative_residual"]),
                                                           residual_gate_limit=1e-12, linear_solver_iterations=-1,
                                                           solve_call_s=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--case", default="c1")
    a = ap.parse_args()
    cfg = config_io.load_cfg(a.cfg)
    report = dict(spectral_tiny=spectral_tiny())
    print(report["spectral_tiny"], flush=True)
    dry = copy.deepcopy(cfg)
    dry["out_dir"] = os.path.join(cfg["out_dir"], "dry")
    out = os.path.join(dry["out_dir"], a.case)
    fabricate(dry, a.case, out)
    for name in ("a.json", "eig.json", "g1.json"):
        if os.path.exists(os.path.join(out, name)):
            os.replace(os.path.join(out, name), os.path.join(out, name + ".old"))
    evaluate_stokes_only.stage_a(dry, a.case, out)
    res = config_io.load_json(os.path.join(out, "a.json"))
    report["dry_archived"] = dict(gates=res["gates"], observed=res["cell_velocity"]["observed_VL2_diff"],
                                  eta_arch=res["cell_velocity"]["eta_arch_in_driver_system"],
                                  printed={k: (v["formatted"], v["printed"]) for k, v in res["printed"].items()},
                                  eta_m=res["table_row_E0"]["eta_m"])
    out2 = os.path.join(dry["out_dir"], a.case + "p")
    fabricate(dry, a.case, out2, perturb=True)
    config_io.save_json(os.path.join(out2, "eig.json"), dict(lambda_lower=1e300, sigma_upper=0.0))  # tau = delta_R only
    evaluate_stokes_only.stage_a(dry, a.case, out2)
    res2 = config_io.load_json(os.path.join(out2, "a.json"))
    report["dry_perturbed"] = dict(gates=res2["gates"], observed=res2["cell_velocity"]["observed_VL2_diff"],
                                   tau=res2["cell_velocity"].get("tau"))
    ok = (report["spectral_tiny"]["pass_"] and res["gates"]["E0a_table"] == "PASS" and res["gates"]["E0a_mass"] == "PASS"
          and res["gates"]["E0_solver"] == "PASS" and report["dry_archived"]["observed"] == 0.0
          and res2["gates"]["E0a_cellU"] == "FAIL" and res2["gates"]["E0a_mass"] == "FAIL"
          and res2["gates"]["E0_solver"] == "FAIL")
    report["pass_"] = bool(ok)
    config_io.save_json(os.path.join(dry["out_dir"], "check_gate_logic.json"), report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
