#!/usr/bin/env python3
"""Quickstart: one controlled case of the paper, end to end, through the CPU package porevoronoi_fv.

The script makes the same calls as porevoronoi_fv/run_stokes_only.py (stage e0) and
porevoronoi_fv/evaluate_stokes_only.py (stage a), without their checkpoint files, and adds no numerics of its own:

  1. load porevoronoi_fv/config.json (config_io.load_cfg) and one case c1..c6 (case_loader.load_case, which checks
     the SHA-256 of the reference field and of the particle tracks against the case's run record
     data/controlled_cases/<case>/run_record.json);
  2. map every record of the particle tracks to its voxel (the site rule), build the cells by exact
     six-neighbour graph ownership, then the facelets and the connected-P1 trace, and stop unless the counts equal
     those of the paper's complex (run_stokes_only.build_published -> case_loader.select_records,
     cell_complex.sites_of, cell_complex.build);
  3. assemble the Stokes system with the vectorised assembler (stokes_solve.assemble, kind 'fast');
  4. solve the Stokes-only calculation (alpha = 0) with the paper's MINRES settings from config.json
     (stokes_solve.solve_entry; hybrid_voronoi_trace.solve_moment_constrained_hybrid_stokes raises if MINRES misses
     its residual gate);
  5. compute the readouts of Table 5 of the paper (readouts.table_row) and compare each with the printed value stored
     in data/controlled_cases/expected/printed_table_values.json, using the literal comparison of the package
     (readouts.literal_fixed / readouts.literal_sci, as in evaluate_stokes_only.stage_a).

With --assisted it also runs the one-record variant of the assisted calculation (in each cell the record nearest
the cell centroid, observations.nearest_centroid_selection; weight theta = config "alpha" = 1000) through
porevoronoi_fv/assisted/assisted_arms.py solve_arm(arm="AN"), and prints for both calculations the velocity error on
the pore voxels that hold no record, e_u,vox(U) (assisted_arms.leave_out_e_ff).

Inputs are read from data/controlled_cases/ and never written. Output: a summary on stdout and one JSON file under
outputs/quickstart/ (or --out). Exit status 0 when every check passes, 1 otherwise.

  python examples/quickstart.py                  # orthogonal duct (c1)
  python examples/quickstart.py --case c6        # Bentheimer crop
  python examples/quickstart.py --assisted       # c1, plus the one-record assisted calculation
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True  # keep the repository free of __pycache__ (the package scripts run with python -B)
import argparse
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.normpath(os.path.join(HERE, os.pardir, "porevoronoi_fv"))
sys.path.insert(0, PACKAGE)  # the package modules import each other as flat modules

import config_io  # noqa: E402
config_io.thread_env(1)  # one BLAS thread, as in every package script; must run before NumPy is imported
import numpy as np  # noqa: E402

import run_stokes_only  # noqa: E402  build_published
import case_loader  # noqa: E402  case loader and record partitions (load_case, select_records)
import readouts  # noqa: E402  readouts and literal comparison (table_row, literal_fixed, literal_sci)
import stokes_solve  # noqa: E402  assembly and the Stokes-only / assisted solve entry (assemble, solve_entry)
import velocity_recovery as rp  # noqa: E402  recovery operators R, G (recovery_operators)

CASES = ("c1", "c2", "c3", "c4", "c5", "c6")

# Checks, as declared by the package (not chosen here):
ETA_M_LIMIT = 1e-14         # check E0a_mass of evaluate_stokes_only.py (PROTOCOL["E0a_mass"], applied in stage_a)
ASSISTED_KKT_LIMIT = 1e-12  # assisted/assisted_protocol.json "solver.residual_gate"
ASSISTED_ETA_LIMIT = 1e-12  # assisted/assisted_protocol.json "solver.mass_gate"

# Printed values used only by --assisted (printed_table_values.json holds the Stokes-only Table 5 values only).
# Velocity error (%) on the pore voxels without a record, e_u,vox(U), at the default weight tau = 1:
#   Stokes-only: Table 6 of the paper, column "At tau = 1, Stokes-only";
#   one-record assisted (arm AN): extended data table ED5.7 of the evidence archive supplied with the manuscript,
#   columns U_PN and U_AN (see docs/figures_and_tables.csv, row Table 6).
PRINTED_U = {"c1": ("4.53", "2.76"), "c2": ("9.00", "7.77"), "c3": ("8.05", "5.65"),
             "c4": ("17.65", "15.11"), "c5": ("15.21", "12.39"), "c6": ("23.77", "21.11")}
# Share (%) of pore voxels that hold a record, |S_g|/N_f: ED5.7 column S_g_over_N_f_pct.
PRINTED_SG = {"c1": "13.39", "c2": "14.48", "c3": "13.84", "c4": "14.08", "c5": "16.95", "c6": "15.76"}


def say(text=""):
    print(text, flush=True)


def show(label, computed, printed="", status="", note=""):
    say(f"  {label:<36s} {computed:>14s} {printed:>14s}  {status:<4s} {note}".rstrip())


def verdict(ok):
    return "PASS" if ok else "FAIL"


class Checks:
    """Collects the PASS/FAIL results; the exit status is 0 only if all pass."""

    def __init__(self):
        self.items = {}

    def add(self, name, ok):
        self.items[name] = bool(ok)
        return verdict(ok)

    @property
    def all_pass(self):
        return all(self.items.values())


# ------------------------------------------------------------------------------------------- Stokes-only calculation
def stokes_only(cfg, code, checks):
    timing = {}

    # Step 2: sites -> cells -> facelets -> connected-P1 trace. run_stokes_only.build_published selects every record
    # of the particle tracks (case_loader.select_records(case, "pub")), maps them to sites (cell_complex.sites_of),
    # builds the complex in the paper's cell order (cell_complex.build) and raises unless N_cv, N_edges,
    # N_trace_modes, N_connected_patches and N_interface_facelets equal the run record and the partition checks pass.
    say("[1/4] Building the cells from the particle records (run_stokes_only.build_published) ...")
    t = time.perf_counter()
    case, part, site_s = run_stokes_only.build_published(cfg, code)
    timing["build_s"] = time.perf_counter() - t
    geom, trace = part["geom"], part["trace"]
    rec = case.records
    D, H, W = case.shape
    say(f"      case {code} ({case.tag}): {D} x {H} x {W} voxels (z, y, x), {case.pore.size} pore voxels, "
        f"periodic in x")
    say(f"      {rec['row'].size} records of {np.unique(rec['particle']).size} particles in "
        f"{np.unique(rec['frame']).size} frames -> {part['seeds'].size} sites")
    say(f"      {geom.n_cells} cells, {geom.owner.size} neighbour pairs, {trace.n_facelets} interface facelets, "
        f"{trace.n_patches} connected patches, {trace.n_trace_modes} trace modes")
    say(f"      all counts equal the run record ({timing['build_s']:.1f} s)")

    # Step 3: vectorised assembly, as in run_stokes_only.stage_e0. Its equality with the production loop assembly
    # is check G1 (run_stokes_only.py --stage g1); it is not repeated here.
    say("[2/4] Assembling the Stokes system (stokes_solve.assemble, vectorised) ...")
    t = time.perf_counter()
    system, _ = stokes_solve.assemble(case, trace, "fast", cfg["viscous_form"])
    timing["assembly_s"] = time.perf_counter() - t
    n_z = int(system.velocity_matrix.shape[0])
    say(f"      {n_z} trace unknowns (3 x {trace.n_trace_modes} trace modes), {geom.n_cells - 1} pressure unknowns, "
        f"nnz(A) = {system.velocity_matrix.nnz} ({timing['assembly_s']:.1f} s)")

    # Step 4: the Stokes-only solve, exactly as run_stokes_only.stage_e0. With alpha = 0 stokes_solve.solve_entry
    # passes the forward A and b unchanged to the paper's MINRES (stokes_solve.paper_solve) and asserts that the KKT
    # matrix MINRES iterates on equals the one built from the forward assembly (stokes_solve.assert_pure_identity).
    s = cfg["solver"]
    say(f"[3/4] Solving the Stokes-only calculation: MINRES, rtol {s['rtol']:g}, maxiter {s['maxiter']}, "
        f"{s['refinement_steps']} refinement step; a progress line every 500 iterations or 60 s ...")
    result, receipt, fac, solved = stokes_solve.solve_entry(case, part, system, alpha=0.0, solver_cfg=cfg["solver"],
                                                            label=f"{code}-stokes-only",
                                                            trace_basis=cfg["trace_basis"])
    del fac, solved
    timing["factorization_s"] = receipt["factorization_s"]
    timing["solve_s"] = receipt["solve_call_s"]
    say(f"      done in {timing['solve_s']:.1f} s")

    # Step 5: Table 5 readouts from (phi, U, z), as evaluate_stokes_only.stage_a does from e0.npz.
    say("[4/4] Computing the readouts (readouts.table_row) ...")
    z = np.asarray(result["trace_coefficients"], np.float64).ravel()
    row, _, _ = readouts.table_row(case, part, result["phi"], result["U"], z=z, D=system.divergence_matrix)

    # Literal comparison with the printed Table 5 values, as in evaluate_stokes_only.stage_a.
    lits = config_io.load_json(os.path.join(cfg["in_dir"], cfg["table_literals"]))["rows"][code]
    printed = dict(N_c=dict(value=row["N_c"], printed=lits["N_c"], formatted=str(row["N_c"]),
                            match=str(row["N_c"]) == lits["N_c"]),
                   eK_arch_s=readouts.literal_fixed(row["eK_arch_s"], lits["eK_arch_s"]),
                   eK_flux_s=readouts.literal_fixed(row["eK_flux_s"], lits["eK_flux_s"]),
                   e_phi=readouts.literal_fixed(row["e_phi"], lits["e_phi"]),
                   e_u_cell=readouts.literal_fixed(row["e_u_cell"], lits["e_u_cell"]))
    # r_inf^m of a new solve is a round-off quantity; the package reports its literal comparison but does not check
    # it. Mass balance is checked through the backward error eta_m instead (check E0a_mass).
    r_inf_literal = readouts.literal_sci(row["r_inf_m"], lits["r_inf_mantissa"], lits["r_inf_exponent"])
    man = case.manifest["row"]
    limit = float(receipt["residual_gate_limit"])
    solver_ok = (int(receipt["linear_solver_info"]) == 0
                 and float(receipt["linear_solver_relative_residual"]) <= limit)  # check E0_solver

    say("")
    say(f"Stokes-only calculation, case {code}: computed values against Table 5 of the paper")
    say("(printed values from data/controlled_cases/expected/printed_table_values.json unless noted;")
    say(" PASS = the computed value, rounded to the printed number of decimals, equals the printed string)")
    show("quantity", "computed", "printed", "check")
    show("cells N_c", str(row["N_c"]), lits["N_c"], checks.add("N_c", printed["N_c"]["match"]))
    show("trace unknowns N_Z", str(n_z), str(man["N_trace_vector_dofs"]),
         checks.add("N_Z", n_z == int(man["N_trace_vector_dofs"])), "printed: run record")
    show("MINRES iterations", str(receipt["linear_solver_iterations"]), str(man["linear_solver_iterations"]), "",
         "printed: archived run (run record); not checked")
    show("MINRES relative KKT residual", f"{receipt['linear_solver_relative_residual']:.2e}", f"<= {limit:.1e}",
         checks.add("solver", solver_ok), "info == 0 and residual <= max(20 rtol, 1e-12)")
    show("r_inf^mass = max_i |(D z)_i|/V_i", f"{row['r_inf_m']:.2e}", r_inf_literal["printed"], "",
         "round-off: reported, not checked")
    show("eta_m (mass backward error)", f"{row['eta_m']:.2e}", f"<= {ETA_M_LIMIT:.0e}",
         checks.add("eta_m", row["eta_m"] <= ETA_M_LIMIT))
    for key, label in (("e_u_cell", "e_u,cell (%)"), ("e_phi", "e_phi (%)"),
                       ("eK_flux_s", "e_K,flux^s (%), signed"), ("eK_arch_s", "e_K,stored^s (%), signed")):
        p = printed[key]
        show(label, f"{p['value']:.4f}", p["printed"], checks.add(key, p["match"]))

    out = dict(counts=dict(records=int(rec["row"].size), sites=int(part["seeds"].size), cells=int(geom.n_cells),
                           neighbour_pairs=int(geom.owner.size), facelets=int(trace.n_facelets),
                           patches=int(trace.n_patches), trace_modes=int(trace.n_trace_modes), trace_unknowns=n_z,
                           A_nnz=int(system.velocity_matrix.nnz)),
               table_row=row, printed=printed, r_inf_m_literal_comparison_not_gated=r_inf_literal,
               solver=receipt, timing=dict(timing, site_input_s=site_s, **part["timing"]),
               partition_checks=part["checks"])
    return case, part, system, z, out


# ------------------------------------------------------------------------------------------- assisted calculation
def assisted(cfg, code, case, part, system, z_pure, checks):
    """One-record assisted calculation (arm AN of porevoronoi_fv/assisted/assisted_arms.py) and e_u,vox(U) for both
    calculations."""
    sys.path.insert(0, os.path.join(PACKAGE, "assisted"))
    import assisted_arms  # porevoronoi_fv/assisted/assisted_arms.py: library of the assisted calculations

    # Every record of every frame: the published partition (assisted_arms.published_records).
    rec = case_loader.select_records(case, "pub")

    # Voxel field of the Stokes-only trace: affine reconstruction in each cell (R, G) evaluated at every pore voxel,
    # as assisted_arms.solve_arm builds it. The voxels with a record are the sites S_g, i.e. the voxels of all
    # published records; assisted_arms.leave_out_e_ff zero-weights them in the relative L2 error.
    say("")
    say("[assisted 1/2] Velocity error of the Stokes-only field on voxels without a record "
        "(assisted_arms.leave_out_e_ff) ...")
    R, G, _ = rp.recovery_operators(part["trace"], system)
    Hfield = assisted_arms.field_operator(part, R, G)
    lo_pure = assisted_arms.leave_out_e_ff(case, part, Hfield, z_pure, case.records["flat"][rec])
    del R, G, Hfield

    # The one-record assisted calculation, exactly as the package runs it: assisted_arms.solve_arm(arm="AN")
    # rebuilds the same complex from the same sites, selects one record per cell
    # (observations.nearest_centroid_selection), forms the observation operator and gamma (observations.observations,
    # observations.gamma_of), solves with stokes_solve.solve_entry(alpha = config alpha) and evaluates e_u,vox(U)
    # with the selected records' voxels left out.
    say(f"[assisted 2/2] Solving the assisted calculation, one record per cell, theta = {cfg['alpha']:g} "
        "(assisted_arms.solve_arm, arm AN) ...")
    try:
        info, _ = assisted_arms.solve_arm(cfg, case, part["seeds"], rec, arm="AN", label=f"{code}-assisted")
    except assisted_arms.Stop as stop:
        say(f"      FAIL: {stop.outcome}: {stop.message}")
        checks.add("assisted_solve", False)
        return dict(status="failed", outcome=stop.outcome, message=stop.message)
    say(f"      done in {info['solver']['solve_call_s']:.1f} s")

    pr_pure, pr_an = PRINTED_U[code]
    e_pure = 100.0 * lo_pure["e_ff_leave_out"]
    e_an = 100.0 * info["e_ff_leave_out"]
    lit_pure = readouts.literal_fixed(e_pure, pr_pure)
    lit_an = readouts.literal_fixed(e_an, pr_an)
    lit_sg = readouts.literal_fixed(100.0 * lo_pure["observed_voxel_fraction"], PRINTED_SG[code])
    solver_an = (int(info["solver"]["linear_solver_info"]) == 0
                 and info["kkt_relative_residual_recomputed"] <= ASSISTED_KKT_LIMIT)

    say("")
    say(f"Assisted calculation, one record per cell, case {code}")
    show("quantity", "computed", "printed", "check")
    show("observations (one record per cell)", str(info["n_obs"]), str(info["n_cells"]),
         checks.add("AN_one_record_per_cell", info["n_obs"] == info["n_cells"] == part["geom"].n_cells),
         "printed: number of cells")
    show("MINRES iterations", str(info["solver"]["linear_solver_iterations"]), "", "", "not checked")
    show("recomputed KKT relative residual", f"{info['kkt_relative_residual_recomputed']:.2e}",
         f"<= {ASSISTED_KKT_LIMIT:.0e}", checks.add("AN_solver", solver_an))
    show("eta_m (mass backward error)", f"{info['eta_m']:.2e}", f"<= {ASSISTED_ETA_LIMIT:.0e}",
         checks.add("AN_eta_m", info["eta_m"] <= ASSISTED_ETA_LIMIT))

    say("")
    say(f"Velocity error on the {lo_pure['pore_voxels'] - lo_pure['observed_voxels']} pore voxels without a record, "
        f"e_u,vox(U) (%)")
    show("quantity", "computed", "printed", "check")
    show("pore voxels holding a record (%)", f"{lit_sg['value']:.4f}", lit_sg["printed"],
         checks.add("S_g_share", lit_sg["match"] and lo_pure["observed_voxels"] == info["observed_voxels"]),
         "printed: ED5.7")
    show("Stokes-only", f"{e_pure:.4f}", pr_pure, checks.add("U_stokes_only", lit_pure["match"]),
         "printed: Table 6, tau = 1")
    show("assisted, one record per cell", f"{e_an:.4f}", pr_an, checks.add("U_assisted_AN", lit_an["match"]),
         "printed: ED5.7 (U_AN)")
    show("ratio assisted / Stokes-only", f"{e_an / e_pure:.3f}", "", "",
         "paper: 0.61-0.89 over the six cases (text with Table 6)")

    return dict(status="ok", e_u_vox_U_stokes_only=lo_pure, e_u_vox_U_assisted=dict(
                    e_ff_leave_out=info["e_ff_leave_out"], observed_voxels=info["observed_voxels"],
                    pore_voxels=info["pore_voxels"], observed_voxel_fraction=info["observed_voxel_fraction"]),
                printed=dict(U_stokes_only=lit_pure, U_assisted=lit_an, S_g_share=lit_sg),
                arm_AN={k: v for k, v in info.items() if k not in ("partition_checks",)})


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Rebuild one controlled case of the paper with the CPU package "
                                             "porevoronoi_fv and compare its readouts with the printed values.")
    ap.add_argument("--case", choices=CASES, default="c1",
                    help="c1 orthogonal duct, c2 skewed duct, c3 thin-wall, c4 narrow throat, c5 maze, "
                         "c6 Bentheimer crop (default c1)")
    ap.add_argument("--assisted", action="store_true",
                    help="also solve the one-record assisted calculation and print e_u,vox(U) for both")
    ap.add_argument("--cfg", default=os.path.join(PACKAGE, "config.json"),
                    help="package configuration (default porevoronoi_fv/config.json)")
    ap.add_argument("--out", default=None, help="folder for the JSON summary (default outputs/quickstart)")
    a = ap.parse_args()

    cfg = config_io.load_cfg(a.cfg)
    out_dir = os.path.abspath(a.out or os.path.join(cfg["out_dir"], "quickstart"))
    checks = Checks()
    t0 = time.perf_counter()
    say(f"PoreVoronoi-FV quickstart: case {a.case} ({cfg['cases'][a.case]['tag']}), config {cfg['_path']}")

    try:
        case, part, system, z, summary = stokes_only(cfg, a.case, checks)
    except RuntimeError as exc:  # raised by build_published, load_case or the hybrid_voronoi_trace residual gate
        say(f"FAIL: {exc}")
        return 1
    report = dict(case=a.case, tag=case.tag, cfg=cfg["_path"], stokes_only=summary)
    if a.assisted:
        report["assisted"] = assisted(cfg, a.case, case, part, system, z, checks)

    report.update(checks=checks.items, all_checks_pass=checks.all_pass, total_s=time.perf_counter() - t0,
                  environment=config_io.environment(), code=config_io.code_hashes())
    path = os.path.join(out_dir, f"{a.case}.json")
    config_io.save_json(path, report)
    say("")
    say(f"{sum(checks.items.values())} of {len(checks.items)} checks pass; total {report['total_s']:.1f} s; "
        f"summary written to {path}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
