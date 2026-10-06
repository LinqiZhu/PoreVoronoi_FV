"""cv_tables: formatting only.  Reads <out>/cv_summary.json (written by run_cross_validation.py --sum;
<out> = outputs/stabilization_weight/cross_validation) and prints the tables.  It introduces NO
decision rule: every verdict it prints was computed by run_cross_validation.summarise() under the
rules fixed in run_cross_validation.py's header.  The labels CV table 1 ... CV table 6 number the
tables of this printout only; they are not table numbers of the paper.

  python -B cv_tables.py > ../../../outputs/stabilization_weight/cross_validation/cv_tables.txt
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "../../../outputs/stabilization_weight/cross_validation", "cv_summary.json"), encoding="utf-8"))
CASES = ["c1", "c2", "c3", "c4", "c5", "c6"]
TAG = {c: (S["meta"].get(c, {}) or {}).get("tag", "") for c in CASES}


def f(x, n=3):
    return "-" if x is None else ("%%.%df" % n) % x


def section(t):
    print("\n" + "=" * 110)
    print(t)
    print("=" * 110)


# ---------------------------------------------------------------- CV table 1 full sweep
section("CV table 1  PART A - full eleven-point sweep, both arms, trajectory partition of c1..c6")
print("e_ff / e_ff_U / e_ff_O are voxel velocity errors in %, all pore voxels / unobserved U / observed O.")
print("%-4s %-3s %8s %8s %8s %8s %9s %9s %9s %9s %7s %4s %10s %10s"
      % ("case", "arm", "tau", "e_ff%", "e_ffU%", "e_ffO%", "e_u_cell", "e_phi", "eK_flux", "m_rec", "iters",
         "info", "kkt", "eta_m"))
for c in CASES:
    for arm in ("PN", "MN"):
        for r in [r for r in S["rowsA"] if r["case"] == c and r["arm"] == arm]:
            print("%-4s %-3s %8.4f %8s %8s %8s %9s %9s %9s %9s %7s %4s %10.2e %10.2e"
                  % (c, arm, r["tau"], f(100 * r["e_ff"], 3) if r["e_ff"] else "-",
                     f(100 * r["e_ff_U"], 3) if r["e_ff_U"] else "-",
                     f(100 * r["e_ff_O"], 3) if r["e_ff_O"] else "-",
                     f(r["e_u_cell"]), f(r["e_phi"]), f(r["eK_flux_s"]), f(r["m_rec"], 5),
                     r["iters"], r["info"], r["kkt"] or 0.0, r["eta_m"] or 0.0))
    print()

# ---------------------------------------------------------------- CV table 2 checks
section("CV table 2  CHECKS")
for c in CASES:
    g = S["gates"].get(c, {})
    for arm in ("PN", "MN"):
        v = g.get(arm)
        if v:
            print("%-4s %-3s G_repro ok=%s  max_rel=%.3e on %s" % (c, arm, v["ok"], v["max_rel"], v["worst_key"]))
bad = [r for r in S["rowsA"] if r["status"] != "ok" or r["info"] != 0 or (r["kkt"] or 1) > 1e-12
       or (r["eta_m"] or 1) > 1e-12]
print("G_solver / G_mass failures among part-A solves: %d" % len(bad))
for r in bad:
    print("   ", r["case"], r["arm"], r["tau"], r["status"], r["info"], r["kkt"], r["eta_m"])

# ---------------------------------------------------------------- CV table 3 the answer to A
section("CV table 3  PART A - the answer.  MN and PN each at its own best tau, and MN at tau = 1.")
MET = [("e_u_cell", "e_u_cell%"), ("e_phi", "e_phi%"), ("eK_flux_abs", "|eK|%"),
       ("e_ff_pct", "e_ff%"), ("e_ff_U_pct", "e_ffU%")]
for key, name in MET:
    print("\n--- %s ---" % name)
    print("%-4s %-18s %8s %9s %8s %9s %9s %9s %9s %9s %9s"
          % ("case", "tag", "tauPN*", "PN@tau*", "tauMN*", "MN@tau*", "PN@1", "MN@1",
             "D_pub", "D_tuned", "share%"))
    for c in CASES:
        a = S["compareA"].get(c, {})
        if "PN" not in a or "MN" not in a:
            continue
        pn, mn = a["PN"], a["MN"]
        v = lambda d: d.get(key)  # noqa: E731
        pn_b, mn_b, pn_1, mn_1 = v(pn["best"]), v(mn["best"]), v(pn["tau1"]), v(mn["tau1"])
        d_pub = None if (pn_1 is None or mn_1 is None) else pn_1 - mn_1
        d_tun = None if (pn_b is None or mn_b is None) else pn_b - mn_b
        share = None if (not d_pub or d_tun is None) else 100.0 * d_tun / d_pub
        print("%-4s %-18s %8.4f %9s %8.4f %9s %9s %9s %9s %9s %9s"
              % (c, TAG[c], pn["best_tau"], f(pn_b), mn["best_tau"], f(mn_b), f(pn_1), f(mn_1),
                 f(d_pub), f(d_tun), f(share, 1)))

section("CV table 3b PART A - MN at tau = 1 against PN at its own best tau (the fair comparison "
        "when only the pure arm is retuned)")
print("%-4s %-18s %-10s %9s %9s %9s %9s %9s" % ("case", "tag", "metric", "PN@tau*", "MN@1", "MN@1-PN*",
                                                "D_pub", "share%"))
for c in CASES:
    a = S["compareA"].get(c, {})
    if "PN" not in a or "MN" not in a:
        continue
    pn, mn = a["PN"], a["MN"]
    for key, name in MET:
        pn_b, pn_1, mn_1 = pn["best"].get(key), pn["tau1"].get(key), mn["tau1"].get(key)
        d_pub = None if (pn_1 is None or mn_1 is None) else pn_1 - mn_1
        d_half = None if (pn_b is None or mn_1 is None) else pn_b - mn_1
        share = None if (not d_pub or d_half is None) else 100.0 * d_half / d_pub
        print("%-4s %-18s %-10s %9s %9s %9s %9s %9s"
              % (c, TAG[c], name, f(pn_b), f(mn_1), f(d_half), f(d_pub), f(share, 1)))
    print()

section("CV table 3c PART A - where each metric's minimiser sits (tau of argmin, per arm)")
print("%-4s %-3s %9s %9s %9s %9s %9s %9s" % ("case", "arm", "e_u_cell", "e_phi", "|eK|", "e_ff", "e_ff_U", "m_rec"))
for c in CASES:
    for arm in ("PN", "MN"):
        a = S["compareA"].get(c, {}).get(arm)
        if not a:
            continue
        am = a["argmin_tau"]
        print("%-4s %-3s %9s %9s %9s %9s %9s %9s"
              % (c, arm, f(am.get("e_u_cell"), 4), f(am.get("e_phi"), 4), f(am.get("eK_flux_abs"), 4),
                 f(am.get("e_ff"), 4), f(am.get("e_ff_U"), 4), f(am.get("m_rec"), 4)))

# ---------------------------------------------------------------- CV table 4 part B
section("CV table 4  PART B - particle-grouped 3-fold CV of the assisted arm, seed 20260919. "
        "e_cv = pooled held-out record misfit.")
for c in CASES:
    b = S["partB"].get(c)
    if not b:
        continue
    print("\n%s (%s)  tau_CV = %s   tau_ref = %s   verdict %s   tau_ref inside CV grid: %s"
          % (c, TAG[c], f(b.get("tau_cv"), 4), f(b.get("tau_ref"), 4), b.get("verdict"),
             b.get("tau_ref_inside_cv_grid")))
    print("   %8s %10s %12s %12s %12s" % ("tau", "e_cv", "fold0", "fold1", "fold2"))
    for k, v in sorted(b["per_tau"].items(), key=lambda kv: float(kv[0])):
        if v.get("e_cv") is None:
            print("   %8s   INCOMPLETE (%s folds)" % (k, v.get("n_folds")))
            continue
        star = " <-- CV pick" if abs(v["tau"] - (b.get("tau_cv") or -1)) < 1e-12 else ""
        print("   %8s %10.6f %12.6f %12.6f %12.6f%s"
              % (k, v["e_cv"], *v["per_fold_rel"], star))

section("CV table 4b PART B - penalty of the CV-selected tau against the reference-best tau, "
        "evaluated on the FULL-RECORD part-A MN solves")
print("%-4s %-18s %8s %8s %8s %9s %9s %9s %9s %9s %-6s"
      % ("case", "tag", "tau_CV", "tau_ref", "d_grid", "e_u_cell", "e_phi", "|eK|", "e_ff", "e_ffU", "verdict"))
for c in CASES:
    b = S["partB"].get(c)
    if not b or b.get("penalty") is None:
        continue
    p = b["penalty"]
    print("%-4s %-18s %8.4f %8.4f %8s %9s %9s %9s %9s %9s %-6s"
          % (c, TAG[c], b["tau_cv"], b["tau_ref"], "in" if b["tau_ref_inside_cv_grid"] else "OUT",
             f(p.get("e_u_cell"), 2), f(p.get("e_phi"), 2), f(p.get("eK_flux_abs"), 2),
             f(p.get("e_ff_pct"), 2), f(p.get("e_ff_U_pct"), 2), b["verdict"]))
print("\n(penalties are 100*(metric(tau_CV) - metric(tau_ref))/metric(tau_ref), in per cent of the "
      "reference-best value; 0.00 means the CV rule picked the reference-best tau)")

section("CV table 5  The record-misfit selector applied to the pure arm (and, for contrast, in-sample to MN)")
print("%-4s %-3s %9s %9s %9s %9s %9s %9s %9s %-6s %s"
      % ("case", "arm", "tau_mrec", "tau_ref", "e_u_cell", "e_phi", "|eK|", "e_ff", "e_ffU", "verdict", "note"))
for c in CASES:
    for arm in ("PN", "MN"):
        d = S["mrec_selector"].get(c, {}).get(arm)
        if not d:
            continue
        p = d["penalty"]
        print("%-4s %-3s %9s %9s %9s %9s %9s %9s %9s %-6s %s"
              % (c, arm, f(d["tau_mrec"], 4), f(d["tau_ref"], 4), f(p.get("e_u_cell"), 2),
                 f(p.get("e_phi"), 2), f(p.get("eK_flux_abs"), 2), f(p.get("e_ff_pct"), 2),
                 f(p.get("e_ff_U_pct"), 2), d["verdict"],
                 "IN-SAMPLE (not a hold-out)" if d["in_sample"] else "out of sample by construction"))

section("CV table 6  DECLARED USABILITY RULE for the CV selector on the assisted arm")
vs = [(c, S["partB"][c].get("verdict"), (S["partB"][c].get("penalty") or {}).get("e_u_cell"))
      for c in CASES if c in S["partB"] and S["partB"][c].get("verdict")]
ok = sum(1 for _, v, _ in vs if v in ("EXACT", "CLOSE"))
worst = max([p for _, _, p in vs if p is not None], default=None)
print("EXACT or CLOSE on %d of %d cases; worst primary penalty %s %%" % (ok, len(vs), f(worst, 2)))
print("declared rule: USABLE iff (EXACT or CLOSE) on >= 5 of 6 AND worst primary penalty <= 25 %%")
print("VERDICT: %s" % ("USABLE" if (ok >= 5 and worst is not None and worst <= 25.0) else "NOT USABLE"))
print("per case:", ", ".join("%s %s (%s%%)" % (c, v, f(p, 1)) for c, v, p in vs))
