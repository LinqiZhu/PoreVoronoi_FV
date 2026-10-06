"""The three arms PN, AN, MN of one case on the trajectory partition (every record of every frame).

Every arm is a JSON+NPZ checkpoint, so a requeue replays. Writes <out>/<case>/{pn,an,mn}.json and
<out>/<case>/arms_summary.json. Run from porevoronoi_fv/:
  python -B assisted/run_assisted_arms.py --case c1
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import gc
import os

import assisted_arms
import config_io
import case_loader
import numpy as np

E0_TOL_REL = 1e-8          # strict agreement reported; the gate itself is the protocol's printed precision
ARM_FILE = {"PN": "pn.json", "AN": "an.json", "MN": "mn.json"}


def printed(value, decimals=2):
    text = f"{float(value):.{decimals}f}"
    return text[1:] if text == "-" + "0." + "0" * decimals else text


def e0_anchor(case, pn):
    """D_pure_matches_E0: PN must reproduce the archived published row of this case (manifest['row'])."""
    row = case.manifest["row"]
    got = dict(eK_abs=abs(float(pn["table_row"]["eK_arch_s"])), e_phi=float(pn["table_row"]["e_phi"]),
               e_u_cell=float(pn["table_row"]["e_u_cell"]), N_c=int(pn["n_cells"]))
    exp = dict(eK_abs=float(row["e_K_percent"]), e_phi=float(row["e_phi_percent"]),
               e_u_cell=float(row["e_u_percent"]), N_c=int(row["N_cv"]))
    rel = {k: abs(got[k] - exp[k]) / max(1.0, abs(exp[k])) for k in ("eK_abs", "e_phi", "e_u_cell")}
    same_print = all(printed(got[k]) == printed(exp[k]) for k in ("eK_abs", "e_phi", "e_u_cell"))
    return dict(got=got, published=exp, relative_difference=rel, max_relative_difference=max(rel.values()),
                agrees_to_printed_precision=bool(same_print and got["N_c"] == exp["N_c"]),
                agrees_within_1e8=bool(max(rel.values()) <= E0_TOL_REL and got["N_c"] == exp["N_c"]))


def gate_verdicts(chk, arms, anchor, prot):
    sol = prot["solver"]
    res_lim, eta_lim = 1e-12, 1e-12
    ok = {a: arms[a].get("status") == "ok" for a in assisted_arms.ARMS}
    g = {}
    g["D_part"] = "PASS" if chk["ok"] else "FAIL"
    g["D_basis"] = "PASS" if all(ok[a] and arms[a].get("trace_basis") == "connected_p1"
                                 for a in assisted_arms.ARMS) else "FAIL"
    if not (ok["AN"] and ok["MN"]):
        g["D_obs"] = "FAIL"
    else:
        an, mn = arms["AN"], arms["MN"]
        g["D_obs"] = "PASS" if (int(an["n_obs"]) == int(an["n_cells"]) and int(an["records_per_cell_min"]) >= 1
                                and int(mn["n_obs"]) == int(chk["n_records"])
                                and float(an["point_operator_defect"]) <= 1e-10
                                and float(mn["point_operator_defect"]) <= 1e-10) else "FAIL"
    g["D_solver"] = "PASS" if all(ok[a] and float(arms[a]["kkt_relative_residual_recomputed"]) <= res_lim
                                  and int((arms[a].get("solver") or {}).get("linear_solver_info", -1)) == 0
                                  for a in assisted_arms.ARMS) else "FAIL"
    g["D_mass"] = "PASS" if all(ok[a] and float(arms[a]["eta_m"]) <= eta_lim for a in assisted_arms.ARMS) else "FAIL"
    g["D_pure_matches_E0"] = "PASS" if anchor["agrees_to_printed_precision"] else "FAIL"
    del sol
    return g


def summarise(arms, chk, anchor):
    rows = {}
    for a in assisted_arms.ARMS:
        j = arms[a]
        if j.get("status") != "ok":
            rows[a] = dict(status=j.get("status"), outcome=j.get("outcome"), message=j.get("message"))
            continue
        r = j["table_row"]
        rows[a] = dict(status="ok", n_cells=int(j["n_cells"]), n_dofs=int(j["n_dofs"]), A_nnz=int(j["A_nnz"]),
                       e_ff=float(j["e_ff"]), e_u_cell=float(r["e_u_cell"]), e_phi=float(r["e_phi"]),
                       eK_arch_s=float(r["eK_arch_s"]), eK_flux_s=float(r["eK_flux_s"]), K_method=float(r["K_method"]),
                       eta_m=float(j["eta_m"]), r_inf_m=float(j["r_inf_m"]),
                       kkt_relative_residual=float(j["kkt_relative_residual_recomputed"]),
                       iterations=int((j.get("solver") or {}).get("linear_solver_iterations", -1)))
        if a != "PN":
            rows[a].update(n_obs=int(j["n_obs"]), gamma=float(j["gamma"]), weight=float(j["weight"]),
                           mu_times_w=float(j["mu_times_w"]),
                           observation_misfit=float(j["observation_misfit"]),
                           plain_stokes_momentum_relative_residual=float(j["plain_stokes_momentum_relative_residual"]),
                           records_per_cell_min=int(j["records_per_cell_min"]),
                           records_per_cell_median=float(j["records_per_cell_median"]),
                           records_per_cell_max=int(j["records_per_cell_max"]),
                           cells_with_zero_records=int(j["cells_with_zero_records"]),
                           e_ff_leave_out=float(j["e_ff_leave_out"]),
                           observed_voxel_fraction=float(j["observed_voxel_fraction"]),
                           naive_eK_arch_s=float(j["naive_eK_arch_s"]))
    out = dict(arms=rows, partition=chk, e0_anchor=anchor)
    if all(rows[a].get("status") == "ok" for a in assisted_arms.ARMS):
        pn = rows["PN"]
        for a in ("AN", "MN"):
            rows[a]["vs_PN"] = dict(
                e_ff_ratio=rows[a]["e_ff"] / pn["e_ff"],
                e_u_cell_ratio=rows[a]["e_u_cell"] / pn["e_u_cell"],
                e_phi_ratio=rows[a]["e_phi"] / pn["e_phi"],
                abs_eK_change=abs(rows[a]["eK_arch_s"]) - abs(pn["eK_arch_s"]),
                A_nnz_ratio=rows[a]["A_nnz"] / pn["A_nnz"],
                iterations_ratio=rows[a]["iterations"] / max(pn["iterations"], 1))
    return out


def run_case(cfg, code):
    prot = assisted_arms.protocol()
    out = config_io.ensure_dir(os.path.join(cfg["out_dir"], code))
    case = case_loader.load_case(cfg, code, with_state=False)
    rec, seeds, part = assisted_arms.published_partition(cfg, case)
    chk = assisted_arms.check_partition(case, rec, seeds, part)
    exp = prot["partition"]["expected_N_c"].get(code)
    chk["matches_declared_N_c"] = bool(exp is None or int(chk["n_cells"]) == int(exp))
    chk["ok"] = bool(chk["ok"] and chk["matches_declared_N_c"])
    print(dict(case=code, stage="partition", **{k: v for k, v in chk.items() if k != "partition_checks"}), flush=True)
    d1_path = os.path.join(out, "arms_summary.json")
    if not chk["ok"]:
        config_io.save_json(d1_path, assisted_arms.stamp(dict(case=code, outcome="partition_invalid", partition=chk)))
        return
    del part
    gc.collect()

    arms = {}
    for arm in assisted_arms.ARMS:
        arms[arm] = assisted_arms.load_or_run(
            os.path.join(out, ARM_FILE[arm]),
            lambda arm=arm: assisted_arms.solve_arm(cfg, case, seeds, rec, arm=arm, label=f"{code}-{arm.lower()}"))
        j = arms[arm]
        print(dict(case=code, stage=arm, status=j.get("status"), e_ff=j.get("e_ff"),
                   n_obs=j.get("n_obs"), misfit=j.get("observation_misfit"),
                   iterations=(j.get("solver") or {}).get("linear_solver_iterations"),
                   checkpoint=j.get("_from_checkpoint")), flush=True)
        gc.collect()

    anchor = e0_anchor(case, arms["PN"]) if arms["PN"].get("status") == "ok" else dict(
        agrees_to_printed_precision=False, note="PN did not solve")
    gates = gate_verdicts(chk, arms, anchor, prot)
    summary = summarise(arms, chk, anchor)
    config_io.save_json(d1_path, assisted_arms.stamp(dict(case=code, gates=gates,
                                                          inputs=assisted_arms.input_hashes(cfg, code), **summary)))
    print(dict(case=code, stage="arms_summary", gates=gates,
               e0_anchor_max_rel=anchor.get("max_relative_difference")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    cfg = assisted_arms.load_cfg()
    if a.case == "sum":
        rows = {}
        for code in ("c1", "c2", "c3", "c4", "c5", "c6"):
            p = os.path.join(cfg["out_dir"], code, "arms_summary.json")
            if os.path.exists(p):
                js = config_io.load_json(p)
                rows[code] = {k: js[k] for k in ("gates", "arms", "partition", "e0_anchor") if k in js}
        config_io.save_json(os.path.join(cfg["out_dir"], "arms_summary_all_cases.json"),
                            assisted_arms.stamp(dict(cases=rows)))
        print(dict(stage="arms_summary_all_cases", cases=sorted(rows)), flush=True)
        return
    run_case(cfg, a.case)


if __name__ == "__main__":
    main()
