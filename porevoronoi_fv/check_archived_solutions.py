"""Gates L1, L2, L3 from ARCHIVED states only (no flow solve; one vectorised assembly per case for D and R).

L1  every value of Table 5 (Stokes-only errors on the controlled cases) recomputed from the archived state equals
    the printed value to its printed digits, and expected/table_source_values.csv (and
    expected/permeability_readouts.json for e_K,flux) to >= 10 significant digits.
L2  the complex built by the driver equals the archived one: counts vs run record; per-cell volumes; per-cell U_ref;
    per-facelet velocity and flux and per-edge flux recomputed from the archived trace coefficients through the
    driver's facelet/mode tables; cell velocities recovered by the driver's recovery operator.
L3  every stored record velocity equals the reference U of its paper-rule voxel; the floor(+0.5) convention gives the
    same voxel; clipped/wrapped row counts equal the run record's.
Negative controls (the instrument must bite): site-order labels must fail the volume comparison; a one-edge roll of
the archived phi must fail the e_phi literal.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import csv
import os
import time

import config_io
config_io.thread_env(1)
import numpy as np

import case_loader
import cell_complex
import readouts
import stokes_solve
import velocity_recovery as rp

SIG = 10.0


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def run_case(cfg, code, lit, csvrow, kflux):
    t0 = time.perf_counter()
    case = case_loader.load_case(cfg, code)
    man = case.manifest
    mi, mr = man["input"], man["row"]
    st = case.state
    r = case.records
    out = dict(case=code, tag=case.tag, state_sha256=case.state_sha256, reference_sha256=man["input_sha256"]["reference_npz"],
               window_sha256=man["input_sha256"]["particle_window"])

    # ---------------- L3
    lab = case.reference["labels"].ravel()[r["flat"]]
    diff = np.abs(r["vel"] - case.reference["U"][lab])
    L3 = dict(records=int(r["row"].size), particles=int(np.unique(r["particle"]).size),
              max_abs_velocity_diff=float(diff.max()), records_with_nonzero_diff=int(np.sum(np.any(diff != 0, axis=1))),
              floor_convention_mismatch=int(r["convention_mismatch"]), clipped_rows=int(r["clipped_rows"]),
              wrapped_rows=int(r["wrapped_rows"]), manifest_clipped_rows=int(mi["particle_clipped_coordinate_rows"]),
              manifest_wrapped_rows=int(mi["particle_periodic_x_wrapped_rows"]),
              manifest_rows_selected=int(mi["particle_rows_selected"]))
    L3["pass"] = bool(L3["max_abs_velocity_diff"] == 0.0 and L3["floor_convention_mismatch"] == 0
                      and L3["clipped_rows"] == L3["manifest_clipped_rows"]
                      and L3["wrapped_rows"] == L3["manifest_wrapped_rows"]
                      and L3["records"] == L3["manifest_rows_selected"])
    out["L3"] = L3

    # ---------------- L2
    seeds = cell_complex.sites_of(case, np.arange(r["row"].size))
    part = cell_complex.build(case, seeds, order="paper")
    geom, trace = part["geom"], part["trace"]
    z_arch = np.asarray(st["trace_coefficients"], np.float64).ravel()
    counts = dict(N_sites=(int(seeds.size), int(mi["particle_unique_snapped_sites"])),
                  N_cv=(int(geom.n_cells), int(mr["N_cv"])), N_edges=(int(geom.owner.size), int(mr["N_edges"])),
                  N_interface_facelets=(int(trace.n_facelets), int(mr["N_interface_facelets"])),
                  N_connected_patches=(int(trace.n_patches), int(mr["N_connected_patches"])),
                  N_trace_modes=(int(trace.n_trace_modes), int(mr["N_trace_modes"])),
                  state_modes=(int(trace.n_trace_modes), int(st["trace_coefficients"].shape[0])),
                  state_facelets=(int(trace.n_facelets), int(st["face_flux_sorted"].shape[0])),
                  state_edges=(int(geom.owner.size), int(st["phi"].shape[0])))
    U_ref = readouts.u_ref_cells(case, geom, part["pore"])
    fv, ff, phi_rec = readouts.face_fields(z_arch, trace)
    t = time.perf_counter()
    system, asm = stokes_solve.assemble(case, trace, "fast")
    t_asm = time.perf_counter() - t
    U_rec = readouts.cell_velocity(z_arch, system)
    L2 = dict(counts={k: dict(driver=v[0], archived=v[1], equal=v[0] == v[1]) for k, v in counts.items()},
              volume_bitwise=bool(np.array_equal(geom.volume, st["volume"])),
              U_ref_max_rel_diff=rel(U_ref, st["U_ref"]), U_ref_bitwise=bool(np.array_equal(U_ref, st["U_ref"])),
              face_velocity_bitwise=bool(np.array_equal(fv, st["face_velocity"])),
              face_velocity_max_rel_diff=rel(fv, st["face_velocity"]),
              face_flux_bitwise=bool(np.array_equal(ff, st["face_flux_sorted"])),
              edge_phi_bitwise=bool(np.array_equal(phi_rec, st["phi"])), edge_phi_max_rel_diff=rel(phi_rec, st["phi"]),
              U_recovered_max_rel_diff=rel(U_rec, st["U"]), U_recovered_bitwise=bool(np.array_equal(U_rec, st["U"])),
              partition_checks=part["checks"], assembly=asm)
    # negative control: cells numbered by site id (not the paper's order) must NOT match the archived arrays
    part_site = cell_complex.build(case, seeds, order="site")
    L2["negative_control_site_order_volume_bitwise"] = bool(np.array_equal(part_site["geom"].volume, st["volume"]))
    L2["pass"] = bool(all(v["equal"] for v in L2["counts"].values()) and L2["volume_bitwise"]
                      and L2["U_ref_max_rel_diff"] < 1e-13 and L2["face_velocity_bitwise"] and L2["face_flux_bitwise"]
                      and L2["edge_phi_bitwise"] and L2["U_recovered_max_rel_diff"] < 1e-12
                      and part["checks"]["pass_"] and not L2["negative_control_site_order_volume_bitwise"])
    out["L2"] = L2

    # ---------------- L1 (from the archived state: phi, U, trace coefficients)
    row, _, phi_ref = readouts.table_row(case, part, st["phi"], st["U"], z=z_arch, D=system.divergence_matrix)
    # the archived U_ref is what the paper used for e_u; recompute with it too (it differs from the driver's by roundoff)
    row["e_u_cell_with_archived_U_ref"] = readouts.e_u_cell(st["U"], st["U_ref"], st["volume"])
    printed = dict(N_c=dict(value=row["N_c"], printed=lit["N_c"], formatted=str(row["N_c"]), match=str(row["N_c"]) == lit["N_c"]),
                   eK_arch_s=readouts.literal_fixed(row["eK_arch_s"], lit["eK_arch_s"]),
                   eK_flux_s=readouts.literal_fixed(row["eK_flux_s"], lit["eK_flux_s"]),
                   e_phi=readouts.literal_fixed(row["e_phi"], lit["e_phi"]),
                   e_u_cell=readouts.literal_fixed(row["e_u_cell"], lit["e_u_cell"]),
                   r_inf_m=readouts.literal_sci(row["r_inf_m"], lit["r_inf_mantissa"], lit["r_inf_exponent"]))
    source = dict(N_c=(row["N_c"], int(csvrow["N_c"])),
                  e_u_percent=(row["e_u_cell"], float(csvrow["e_u_percent"])),
                  e_phi_percent=(row["e_phi"], float(csvrow["e_phi_percent"])),
                  e_K_percent=(row["eK_arch_abs"], float(csvrow["e_K_percent"])),
                  mass_inf_per_volume=(row["r_inf_m"], float(csvrow["mass_inf_per_volume"])),
                  e_K_common_flux_signed_percent=(row["eK_flux_s"], float(kflux["e_K_common_flux_signed_percent"])),
                  K_method=(row["K_method"], float(kflux["K_method"])),
                  K_reference_flux=(row["K_ref_flux"], float(kflux["K_reference_flux"])),
                  K_reference_archived=(row["K_ref_arch"], float(kflux["K_reference_archived"])),
                  manifest_K_eff_x=(row["K_method"], float(mr["K_eff_x"])))
    sig = {k: dict(driver=v[0], source=v[1], significant_digits=readouts.sig_digits(float(v[0]), float(v[1])))
           for k, v in source.items()}
    table_keys = ("N_c", "e_u_percent", "e_phi_percent", "e_K_percent", "mass_inf_per_volume",
                  "e_K_common_flux_signed_percent")
    sign_ok = bool(np.sign(row["eK_arch_s"]) == np.sign(float(lit["eK_arch_s"])))
    neg_phi = np.roll(np.asarray(st["phi"]), 1)
    neg = readouts.literal_fixed(readouts.e_phi(neg_phi, phi_ref), lit["e_phi"])
    L1 = dict(row=row, printed=printed, source=sig, sign_matches_printed=sign_ok,
              negative_control_rolled_phi_e_phi=neg,
              r_inf_m_note="machine-precision zero: achieved r_inf_m and eta_m (backward error) reported in row")
    L1["printed_all_match"] = bool(all(v["match"] for v in printed.values()))
    L1["source_min_significant_digits_table_columns"] = float(min(sig[k]["significant_digits"] for k in table_keys))
    L1["pass"] = bool(L1["printed_all_match"] and sign_ok and L1["source_min_significant_digits_table_columns"] >= SIG
                      and not neg["match"])
    # full-field voxel velocity error of the archived solution (not a printed value)
    R, G, defect = rp.recovery_operators(trace, system)
    Hfield = rp.point_operator(part["xyz"], geom.labels.ravel()[part["pore"]], trace, R, G)
    L1["full_field_voxel_error_archived"] = readouts.full_field_error(z_arch, Hfield, case)
    L1["recovery_gradient_divergence_defect"] = defect
    out["L1"] = L1
    out["seconds"] = dict(total=time.perf_counter() - t0, fast_assembly=t_asm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--cases", default="c1,c2,c3,c4,c5,c6")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = config_io.load_cfg(a.cfg)
    out_dir = config_io.ensure_dir(a.out or os.path.join(cfg["out_dir"], "check_archived_solutions"))
    lits = config_io.load_json(os.path.join(cfg["in_dir"], cfg["table_literals"]))["rows"]
    with open(os.path.join(cfg["in_dir"], cfg["table_csv"]), newline="", encoding="utf-8") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    kf = {r["case"]: r for r in config_io.load_json(os.path.join(cfg["in_dir"], cfg["kflux_receipt"]))["cases"]}
    summary = dict(environment=config_io.environment(), code=config_io.code_hashes(), cases={})
    for code in a.cases.split(","):
        tag = cfg["cases"][code]["tag"]
        res = run_case(cfg, code, lits[code], rows[tag], kf[tag])
        config_io.save_json(os.path.join(out_dir, f"{code}.json"), res)
        summary["cases"][code] = dict(L1=res["L1"]["pass"], L2=res["L2"]["pass"], L3=res["L3"]["pass"],
                                      L1_min_sig=res["L1"]["source_min_significant_digits_table_columns"],
                                      printed={k: (v["formatted"], v["printed"]) for k, v in res["L1"]["printed"].items()},
                                      seconds=res["seconds"]["total"])
        print(code, summary["cases"][code], flush=True)
    config_io.save_json(os.path.join(out_dir, "summary.json"), summary)


if __name__ == "__main__":
    main()
