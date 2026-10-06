"""check_records: independent verification of the cross-validation outputs.  Recomputes, it does
not re-solve.  Run from this folder, after run_cross_validation.py and ../sweep/run_sweep.py.

  1. Re-reads every stored z from the part-A npz files and recomputes e_ff / e_ff_U / e_ff_O and m_rec
     from scratch (rebuilding the operators), comparing against the JSON the run wrote.
  2. Recomputes every pooled e_cv from the per-fold residual norms.
  3. Cross-checks the PN sweep against the grid sweep's outputs/stabilization_weight/sweep/sweep_summary.json
     and the tau = 1 rows of both arms against the stored solves outputs/assisted/<case>/{pn,mn}.json,
     including the printed MN e_u,vox on U.
  4. Prints the derived bands quoted in the text.

  python -B check_records.py [case ...]      (no argument: cheap checks 2-4 only; a case name adds check 1)
"""
import glob
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_OUT = os.path.join(os.path.dirname(HERE), "../../outputs/stabilization_weight/sweep", ".")
PACKAGE_DIR = "../../../porevoronoi_fv"
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sweep"))
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, os.path.join(PACKAGE_DIR, "assisted"))

import config_io  # noqa: E402
config_io.thread_env(1)
import numpy as np  # noqa: E402

OUT = os.path.join(HERE, "../../../outputs/stabilization_weight/cross_validation")
CASES = ["c1", "c2", "c3", "c4", "c5", "c6"]
PUB = ("../../../outputs/assisted")


def rows(case, pat):
    return [config_io.load_json(p) for p in sorted(glob.glob(os.path.join(OUT, case, pat)))]


def check_fields(case):
    import assisted_arms, case_loader  # noqa: E402
    import velocity_recovery as rp  # noqa: E402
    import fast_assembly_weighted  # noqa: E402
    cfg = assisted_arms.load_cfg()
    c = case_loader.load_case(cfg, case, with_state=False)
    rec, seeds, part = assisted_arms.published_partition(cfg, c)
    tr = part["trace"]
    nc = int(tr.n_cells)
    sysm, _ = fast_assembly_weighted.assemble_vectorised(tr, viscosity=c.nu, body_force=c.force,
                                        viscous_form=cfg["viscous_form"], backend="numpy",
                                        tau_cell=np.full(nc, 1.0))
    R, G, _ = rp.recovery_operators(tr, sysm)
    Hf = assisted_arms.field_operator(part, R, G)
    pore = np.asarray(c.pore, np.int64)
    O = np.unique(np.asarray(c.records["flat"], np.int64)[rec])
    pos = np.searchsorted(pore, O)
    wU = np.ones(pore.size)
    wU[pos] = 0.0
    sel = assisted_arms.selection_all(c, part, rec)
    H = rp.point_operator(c.records["xyz"][rec], np.asarray(sel["cell"], np.int64), tr, R, G)
    y = np.asarray(c.records["vel"][rec], np.float64)
    ny = float(np.linalg.norm(y.ravel()))
    worst = (0.0, None)
    n = 0
    for p in sorted(glob.glob(os.path.join(OUT, case, "a_*.npz"))):
        z = np.load(p)["z"]
        j = config_io.load_json(p[:-4] + ".json")
        v = (Hf @ z).reshape(-1, 3)
        got = dict(e_ff=rp.normerr(v, c.u_ref_vox), e_ff_U=rp.normerr(v, c.u_ref_vox, weight=wU),
                   e_ff_O=rp.normerr(v, c.u_ref_vox, weight=1.0 - wU),
                   m_rec=float(np.linalg.norm(H @ z - y.ravel()) / ny))
        for k, val in got.items():
            d = abs(val - j[k]) / max(abs(j[k]), 1e-300)
            n += 1
            if d > worst[0]:
                worst = (d, (os.path.basename(p), k, val, j[k]))
    print("CHECK 1 %s: %d recomputed field quantities, worst relative difference %.3e %s"
          % (case, n, worst[0], worst[1]))


def check_cv():
    S = config_io.load_json(os.path.join(OUT, "cv_summary.json"))
    worst = (0.0, None)
    n = 0
    for case in CASES:
        per = {}
        for r in rows(case, "b_CV_*.json"):
            per.setdefault(round(r["tau"], 6), []).append(r)
        for tau, rs in per.items():
            assert len(rs) == 3, (case, tau, len(rs))
            e = math.sqrt(sum(r["resid_test"] ** 2 for r in rs) / sum(r["norm_test"] ** 2 for r in rs))
            got = S["partB"][case]["per_tau"]["%.4f" % tau]["e_cv"]
            d = abs(e - got) / got
            n += 1
            if d > worst[0]:
                worst = (d, (case, tau, e, got))
            # the folds must partition the records exactly once
            assert sum(r["n_test"] for r in rs) == S["meta"][case]["n_records"], (case, tau)
            for r in rs:
                assert r["n_train"] + r["n_test"] == S["meta"][case]["n_records"]
    print("CHECK 2: %d pooled e_cv recomputed, worst relative difference %.3e %s" % (n, worst[0], worst[1]))


def check_sweep_and_stored():
    sweep_sum = config_io.load_json(os.path.join(SWEEP_OUT, "sweep_summary.json"))
    idx = {(r["case"], round(r["tau_uniform"], 6)): r for r in sweep_sum["rows"] if r["kind"] == "uniform"}
    worst, n = (0.0, None), 0
    for case in CASES:
        for r in rows(case, "a_PN_*.json"):
            a = idx.get((case, round(r["tau"], 6)))
            if not a:
                continue
            for key, mine in (("e_u_cell", r["table_row"]["e_u_cell"]), ("e_phi", r["table_row"]["e_phi"]),
                              ("eK_flux_s", r["table_row"]["eK_flux_s"]), ("e_ff", r["e_ff"])):
                b = a[key]
                d = abs(mine - b) / max(abs(b), 1e-300)
                n += 1
                if d > worst[0]:
                    worst = (d, (case, r["tau"], key, mine, b))
    print("CHECK 3a: %d PN values against the grid sweep, worst relative difference %.3e %s" % (n, worst[0], worst[1]))
    worst, n = (0.0, None), 0
    for case in CASES:
        for arm in ("PN", "MN"):
            j = config_io.load_json(os.path.join(OUT, case, "a_%s_u001.0000.json" % arm))
            p = config_io.load_json(os.path.join(PUB, case, "%s.json" % arm.lower()))
            pairs = [(k, j["table_row"][k], p["table_row"][k]) for k in
                     ("K_method", "eK_arch_s", "eK_flux_s", "e_phi", "e_u_cell", "mean_flux_x")]
            pairs.append(("e_ff", j["e_ff"], p["e_ff"]))
            if arm == "MN":
                pairs.append(("e_ff_U vs e_ff_leave_out", j["e_ff_U"], p["e_ff_leave_out"]))
            for k, a, b in pairs:
                d = abs(a - b) / max(abs(b), 1e-300)
                n += 1
                if d > worst[0]:
                    worst = (d, (case, arm, k, a, b))
    print("CHECK 3b: %d tau=1 values against the stored solves, worst relative difference %.3e %s"
          % (n, worst[0], worst[1]))
    # the printed MN column on U
    printed = [2.777, 7.333, 5.664, 14.329, 12.208, 21.836]
    print("CHECK 3c: MN e_ff_U at tau=1 vs the printed MN-on-U column")
    for case, want in zip(CASES, printed):
        got = 100.0 * config_io.load_json(os.path.join(OUT, case, "a_MN_u001.0000.json"))["e_ff_U"]
        print("    %s  here %.4f   printed %.3f   %s" % (case, got, want, "OK" if abs(got - want) < 5e-4 else "MISMATCH"))


def bands():
    S = config_io.load_json(os.path.join(OUT, "cv_summary.json"))
    print("\nDERIVED BANDS AND GAPS")
    print("case | MN: tau*  e_u(tau*)  e_u(1)  gap(1)%  | tau_CV  e_u(CV)  gap(CV)%  | PN: tau*  gap(1)%")
    for c in CASES:
        mn = S["compareA"][c]["MN"]
        pn = S["compareA"][c]["PN"]
        b = S["partB"][c]
        g1 = 100.0 * (mn["tau1"]["e_u_cell"] - mn["best"]["e_u_cell"]) / mn["best"]["e_u_cell"]
        gp = 100.0 * (pn["tau1"]["e_u_cell"] - pn["best"]["e_u_cell"]) / pn["best"]["e_u_cell"]
        ecv = mn["by_tau"]["%.4f" % b["tau_cv"]]["e_u_cell"]
        gcv = 100.0 * (ecv - mn["best"]["e_u_cell"]) / mn["best"]["e_u_cell"]
        print("%-4s | %6.4f %9.4f %8.4f %8.2f | %6.4f %8.4f %9.2f | %6.4f %8.2f"
              % (c, mn["best_tau"], mn["best"]["e_u_cell"], mn["tau1"]["e_u_cell"], g1,
                 b["tau_cv"], ecv, gcv, pn["best_tau"], gp))

    def band(vals):
        return "%.2f-%.2f" % (min(vals), max(vals))
    for arm, which in (("PN", "tau1"), ("PN", "best"), ("MN", "tau1"), ("MN", "best")):
        v = [abs(S["compareA"][c][arm][which]["eK_flux_s"]) for c in CASES]
        print("|eK_flux| band, %s at %s: %s %%" % (arm, "tau=1" if which == "tau1" else "its best tau", band(v)))
    v = [abs(S["compareA"][c]["MN"]["by_tau"]["%.4f" % S["partB"][c]["tau_cv"]]["eK_flux_abs"]) for c in CASES]
    print("|eK_flux| band, MN at the CV-selected tau: %s %%" % band(v))
    v = [abs(S["compareA"][c]["PN"]["by_tau"]["%.4f" % S["mrec_selector"][c]["PN"]["tau_mrec"]]["eK_flux_abs"])
         for c in CASES]
    print("|eK_flux| band, PN at the record-misfit-selected tau: %s %%" % band(v))
    for arm, which in (("PN", "tau1"), ("PN", "best"), ("MN", "tau1"), ("MN", "best")):
        v = [S["compareA"][c][arm][which]["e_ff_U_pct"] for c in CASES]
        print("e_u,vox on U band, %s at %s: %s %%" % (arm, "tau=1" if which == "tau1" else "its best tau", band(v)))
    # c2 margin
    p = S["partB"]["c2"]["per_tau"]
    print("\nc2 CV margin: e_cv(1.0) = %.8f vs e_cv(1.4142) = %.8f, relative gap %.4f %%"
          % (p["1.0000"]["e_cv"], p["1.4142"]["e_cv"],
             100.0 * (p["1.4142"]["e_cv"] - p["1.0000"]["e_cv"]) / p["1.0000"]["e_cv"]))
    p = S["partB"]["c1"]["per_tau"]
    print("c1 CV margin: e_cv(0.7071) = %.8f vs e_cv(0.5) = %.8f vs e_cv(1.0) = %.8f"
          % (p["0.7071"]["e_cv"], p["0.5000"]["e_cv"], p["1.0000"]["e_cv"]))
    # how often PN@best beats MN@1
    for m in ("e_u_cell", "e_ff_pct", "e_ff_U_pct", "e_phi", "eK_flux_abs"):
        w = [c for c in CASES if S["compareA"][c]["PN"]["best"][m] < S["compareA"][c]["MN"]["tau1"][m]]
        print("PN at its best tau beats MN at tau=1 on %-11s in %d/6: %s" % (m, len(w), ",".join(w)))
    for m in ("e_u_cell", "e_ff_pct", "e_ff_U_pct", "e_phi", "eK_flux_abs"):
        w = [c for c in CASES if S["compareA"][c]["PN"]["best"][m] < S["compareA"][c]["MN"]["best"][m]]
        print("PN at its best tau beats MN at ITS best tau on %-11s in %d/6: %s" % (m, len(w), ",".join(w)))
    # total cost
    n = sum(1 for c in CASES for r in rows(c, "a_*.json")) + sum(1 for c in CASES for r in rows(c, "b_CV_*.json"))
    t = (sum(r.get("solve_s", 0) for c in CASES for r in rows(c, "a_*.json"))
         + sum(r.get("solve_s", 0) for c in CASES for r in rows(c, "b_CV_*.json")))
    print("\nsolves: %d, total solver wall time %.1f core-hours" % (n, t / 3600.0))


if __name__ == "__main__":
    check_cv()
    check_sweep_and_stored()
    bands()
    for a in sys.argv[1:]:
        check_fields(a)
