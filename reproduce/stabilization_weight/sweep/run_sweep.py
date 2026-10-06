"""run_sweep: hold-out test of tau = 1 and of a FIXED LOCAL stabilization rule, on c1..c6.

  python -B run_sweep.py --case c1            one case (checkpointed, writes <out>/<case>/*.json)
  python -B run_sweep.py --sum                collect every finished solve into <out>/sweep_summary.json

Run from this folder.  <out> is outputs/stabilization_weight/sweep/; the stored PN solves at tau = 1
are read from outputs/assisted/<case>/pn.json, written by porevoronoi_fv/assisted/run_assisted_arms.py.

Every rule below was fixed before the script was run on any of c1..c6.  The only numbers used to fix a
free parameter are stated in "CALIBRATION" and come from a sweep of the manufactured cube, not from
c1..c6.

WHAT IS FIXED IN THE SCHEME
  The assembly weights the residual stabilization of every cell by
      w = 2 nu A_facelet / h      (symmetric_gradient; hybrid_voronoi_trace.py, fast_assembly.py)
  with no multiplier.  Writing w_i = tau_i * 2 nu A / h, the paper is tau_i = 1 everywhere.

CODE
  The multiplier is exposed in COPIES of the package assemblers, hybrid_voronoi_trace_weighted.py and
  fast_assembly_weighted.py, which differ from porevoronoi_fv/hybrid_voronoi_trace.py and fast_assembly.py
  only by the optional argument tau_cell.  check_equal_to_package.py proves on a real complex that with
  tau_cell = None and with tau_cell = 1 both copies are BITWISE equal to the package modules, that the
  two copies agree with each other at a non-uniform tau (fast_assembly.gate), and that
  A(c tau) - A(0) = c (A(tau) - A(0)), i.e. the multiplier scales the cell stabilization block and
  nothing else.  <out>/gate.json carries the result.  Solves use fast_assembly_weighted (the vectorised
  path that stokes_solve.assemble calls); everything downstream - recovery operators, MINRES settings,
  readouts - is the unmodified package code.

CASES AND ARM
  c1..c6 of porevoronoi_fv/config.json, on the trajectory partition of the paper
  (assisted_arms.published_partition: every window record of every frame).  Arm PN, the pure
  finite-volume solve (alpha = 0), which is the Table 6 configuration and the one whose accuracy the
  stabilization controls.  The velocity-assisted arms are NOT swept here: their theta = 1e3 observation
  term is a second weight and would confound the stabilization question.

SOLVER
  cfg["solver"] unchanged: hybrid_voronoi_trace MINRES, rtol 1e-14, maxiter 50000, 1 refinement step,
  1 BLAS thread.  A solve that raises is recorded as status "solve_failed" with its message; it is not
  retried and no other setting is tried.

METRIC
  PRIMARY   e_u_cell (%), readouts.table_row: the volume-weighted relative error of the cell velocities
            against the reference cell means.  This is the quantity the manufactured-cube sweep calls
            e_u, so the two are directly comparable.
  SECONDARY e_phi (%), eK_flux_s (signed %), e_ff (full voxel-field error, fraction), all from the
            same readout call assisted_arms.readouts.
  "Best tau" of a case = the grid value minimising the PRIMARY metric (ties -> the smaller tau).
  "gap of X" = 100 * (e_u_cell(X) - e_u_cell(best)) / e_u_cell(best).

GRID (part A, uniform tau)
  TAUS = 2^(j/2), j = -4..6 = 0.25, 0.3536, 0.5, 0.7071, 1, 1.4142, 2, 2.8284, 4, 5.6569, 8.
  If a case's minimiser falls on an endpoint that is reported as an endpoint and the grid is NOT
  extended for that case.

THE FIXED LOCAL RULE (part B)
  Geometry only, no reference solution, no per-case tuning:
      RULE_V  (primary)    tau_i = ALPHA * h / H_i ,   H_i = V_i^(1/3)   (V_i = cell volume)
      RULE_F  (declared secondary)  tau_i = ALPHA * h / H_i ,
                           H_i = h * sqrt(n_facelet,i / 6)   (a cube of side H has 6 (H/h)^2 facelets;
                           n_facelet,i counts interior facelets and wall facelets of cell i, the same
                           incidence list the stabilization sums over)
  Both are reported whatever they give.  Neither coefficient is re-fitted per case.

CALIBRATION (ONE case, one number, fixed before c1..c6 were run)
  ALPHA = 4.756828460010884.
  It is tau_best * (H/h) on a SINGLE level of the manufactured cube: the fixed-physics (L = 16 h)
  lattice cube at N = 16, m = 4, i.e. H/h = 4, whose best tau on that sweep's grid is 2^(1/4) =
  1.189207115002721 (the manufactured-cube rows are in the evidence archive supplied with the
  manuscript).  On that cube every cell is 4x4x4 voxels, so V^(1/3) = 4 h and
  sqrt(n_facelet/6) = sqrt(96/6) = 4: RULE_V and RULE_F give the same tau there, which is why one
  coefficient serves both.  The value is not re-derived from any c1..c6 run.

CHECKS
  G_repro   at tau = 1 every entry of table_row and e_ff reproduces the stored PN solve
            (outputs/assisted/<case>/pn.json) to <= 1e-6 relative.  A failure is reported; the sweep
            still runs, and every number from that case is labelled unreproduced.
  G_solver  linear_solver_info == 0 and the recomputed KKT relative residual <= 1e-12 on every solve.
  G_mass    eta_m <= 1e-12 on every solve.

ALSO RECORDED (a-priori diagnostics a user CAN measure without a reference solution)
  per case: N_c, h, the distribution of H_i/h for both readings of H_i, cell aspect (the ratio of the
  largest to the smallest eigenvalue of the cell's voxel second-moment tensor), facelets per cell;
  at tau = 1: the stabilization share of the discrete energy, z^T A_stab z / z^T A z, and the
  pressure-to-viscous load ratio rho_p = ||D^T p|| / ||A z||.  Both need only the method's own solve.

WHAT THIS CANNOT SHOW
  The manufactured-cube sweep shows that the best tau*H/h is load-dependent (about 5 for the L = 16
  cube, 64-136 for the pressure-dominated L = 64 cube).  c1..c6 are all driven by the same uniform body force
  cfg["force"], so this sweep fixes the rule's transferability ACROSS GEOMETRIES at one forcing, not
  across forcings.  That limit is reported with the result.
"""
import sys, os, time, json, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = "../../../porevoronoi_fv"
sys.path.insert(0, HERE)
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, os.path.join(PACKAGE_DIR, "assisted"))

import config_io
config_io.thread_env(1)
import numpy as np

import assisted_arms, case_loader, cell_complex, readouts, stokes_solve
import velocity_recovery as rp
import fast_assembly_weighted

OUT = os.path.join(HERE, "../../../outputs/stabilization_weight/sweep")
CASES = ["c1", "c2", "c3", "c4", "c5", "c6"]
TAUS = [float(2.0 ** (j / 2.0)) for j in range(-4, 7)]
ALPHA = 4.756828460010884
PUB = ("../../../outputs/assisted")


def say(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def tag_of(kind, tau=None):
    if kind == "uniform":
        return "u%+08.4f" % tau
    return kind


def cell_geometry(case, part):
    """Per-cell quantities a user can measure from the partition alone."""
    tr = part["trace"]
    geom = part["geom"]
    h = float(tr.voxel_size)
    V = np.asarray(tr.cell_volume, np.float64)
    inc = np.concatenate([tr.face_owner, tr.face_neigh, tr.wall_cell])
    nfac = np.bincount(inc, minlength=int(tr.n_cells)).astype(np.float64)
    Hv = V ** (1.0 / 3.0)
    Hf = h * np.sqrt(np.maximum(nfac, 1.0) / 6.0)
    # cell aspect from the voxel second-moment tensor
    lab = np.asarray(geom.labels).ravel()
    pore = np.asarray(case.pore, np.int64)
    own = lab[pore]
    zyx = np.column_stack(np.unravel_index(pore, case.shape)).astype(np.float64)
    xyz = (zyx[:, ::-1] + 0.5) * h
    nc = int(tr.n_cells)
    cnt = np.bincount(own, minlength=nc).astype(np.float64)
    mean = np.column_stack([np.bincount(own, weights=xyz[:, a], minlength=nc) / np.maximum(cnt, 1)
                            for a in range(3)])
    d = xyz - mean[own]
    M = np.zeros((nc, 3, 3))
    for a in range(3):
        for b in range(3):
            M[:, a, b] = np.bincount(own, weights=d[:, a] * d[:, b], minlength=nc) / np.maximum(cnt, 1)
    ev = np.linalg.eigvalsh(M + 1e-30 * np.eye(3))
    aspect = np.sqrt(np.maximum(ev[:, 2], 0) / np.maximum(ev[:, 0], 1e-30))
    return dict(h=h, V=V, nfac=nfac, Hv=Hv, Hf=Hf, aspect=aspect)


def pct(a, name):
    a = np.asarray(a, np.float64)
    return {name + "_min": float(a.min()), name + "_p10": float(np.percentile(a, 10)),
            name + "_med": float(np.median(a)), name + "_p90": float(np.percentile(a, 90)),
            name + "_max": float(a.max()), name + "_mean": float(a.mean())}


def tau_vector(kind, cg, nc):
    if kind == "RULE_V":
        return ALPHA * cg["h"] / cg["Hv"]
    if kind == "RULE_F":
        return ALPHA * cg["h"] / cg["Hf"]
    raise ValueError(kind)


def one_solve(cfg, case, part, R, G, Hfield, tau_cell, label):
    tr = part["trace"]
    system, _ = fast_assembly_weighted.assemble_vectorised(tr, viscosity=case.nu, body_force=case.force,
                                          viscous_form=cfg["viscous_form"], backend="numpy",
                                          tau_cell=tau_cell)
    t0 = time.perf_counter()
    result, receipt, fac, solved = stokes_solve.solve_entry(case, part, system, alpha=0.0,
                                                  solver_cfg=cfg["solver"], label=label,
                                                  trace_basis=cfg["trace_basis"], check_identity=True)
    wall = time.perf_counter() - t0
    A_used, b_used = solved.velocity_matrix, solved.body_force_matrix @ solved.body_force
    diag = assisted_arms.readouts(case, part, system, A_used, b_used, result, Hfield)
    z = np.asarray(result["trace_coefficients"], np.float64).ravel()
    p = np.asarray(result["p"], np.float64)
    out = dict(diag)
    out.update(solver=receipt, solve_s=wall, n_dofs=int(system.velocity_matrix.shape[0]))
    out["rho_p"] = float(np.linalg.norm(system.divergence_matrix.T @ p)
                         / max(float(np.linalg.norm(system.velocity_matrix @ z)), 1e-300))
    del fac
    return out, system, z


def run_case(code):
    cfg = assisted_arms.load_cfg()
    odir = os.path.join(OUT, code)
    os.makedirs(odir, exist_ok=True)
    case = case_loader.load_case(cfg, code, with_state=False)
    rec, seeds, part = assisted_arms.published_partition(cfg, case)
    chk = assisted_arms.check_partition(case, rec, seeds, part)
    tr = part["trace"]
    nc = int(tr.n_cells)
    cg = cell_geometry(case, part)
    meta = dict(case=code, tag=case.tag, ALPHA=ALPHA, taus=TAUS, n_cells=nc,
                n_dofs=int(3 * tr.n_trace_modes), n_facelets=int(tr.n_facelets),
                partition_check=chk, h=cg["h"])
    for arr, nm in ((cg["Hv"] / cg["h"], "Hv_over_h"), (cg["Hf"] / cg["h"], "Hf_over_h"),
                    (cg["nfac"], "nfacelet"), (cg["aspect"], "aspect"), (cg["V"], "volume")):
        meta.update(pct(arr, nm))
    for kind in ("RULE_V", "RULE_F"):
        meta.update(pct(tau_vector(kind, cg, nc), "tau_" + kind))
    config_io.save_json(os.path.join(odir, "meta.json"), meta)
    say(code, "cells", nc, "Hv/h med", round(meta["Hv_over_h_med"], 3),
        "tau_RULE_V med", round(meta["tau_RULE_V_med"], 3))

    system0, _ = fast_assembly_weighted.assemble_vectorised(tr, viscosity=case.nu, body_force=case.force,
                                           viscous_form=cfg["viscous_form"], backend="numpy",
                                           tau_cell=np.zeros(nc))
    A0 = system0.velocity_matrix
    R, G, gdef = rp.recovery_operators(tr, system0)
    Hfield = assisted_arms.field_operator(part, R, G)

    jobs = [("uniform", t) for t in TAUS] + [("RULE_V", None), ("RULE_F", None)]
    for kind, tau in jobs:
        tg = tag_of(kind, tau)
        path = os.path.join(odir, "s_%s.json" % tg)
        if os.path.exists(path) and config_io.load_json(path).get("status") in ("ok", "solve_failed"):
            say(code, tg, "skip (done)")
            continue
        tvec = np.full(nc, float(tau)) if kind == "uniform" else tau_vector(kind, cg, nc)
        rec_out = dict(case=code, kind=kind, tau_uniform=(float(tau) if kind == "uniform" else None),
                       tag=tg, ALPHA=ALPHA)
        rec_out.update(pct(tvec, "tau"))
        try:
            got, system, z = one_solve(cfg, case, part, R, G, Hfield, tvec, "%s-%s" % (code, tg))
            rec_out.update(got)
            Az = float(z @ (system.velocity_matrix @ z))
            A0z = float(z @ (A0 @ z))
            rec_out["stab_energy_share"] = (Az - A0z) / Az if Az else None
            rec_out["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec_out.update(status="solve_failed", error=repr(e)[:400])
        config_io.save_json(path, rec_out)
        tr_row = rec_out.get("table_row", {})
        say(code, tg, rec_out["status"], "e_u_cell", tr_row.get("e_u_cell"),
            "e_phi", tr_row.get("e_phi"), "e_ff", rec_out.get("e_ff"),
            "%.0fs" % rec_out.get("solve_s", 0))

    # G_repro against the stored PN solve
    pub = os.path.join(PUB, code, "pn.json")
    g = dict(gate="G_repro", published=pub, ok=None)
    p1 = os.path.join(odir, "s_%s.json" % tag_of("uniform", 1.0))
    if os.path.exists(pub) and os.path.exists(p1):
        a = config_io.load_json(pub)
        b = config_io.load_json(p1)
        worst, where = 0.0, None
        for k, v in a["table_row"].items():
            if isinstance(v, (int, float)) and k in b.get("table_row", {}):
                w = b["table_row"][k]
                d = abs(w - v) / max(abs(v), 1e-300)
                if d > worst:
                    worst, where = d, k
        d = abs(b.get("e_ff", 0) - a["e_ff"]) / max(abs(a["e_ff"]), 1e-300)
        if d > worst:
            worst, where = d, "e_ff"
        g.update(ok=bool(worst <= 1e-6), max_rel=worst, worst_key=where)
    config_io.save_json(os.path.join(odir, "gate_repro.json"), g)
    say(code, "G_repro", json.dumps(g))


def summarise():
    rows, meta, gates = [], {}, {}
    for c in CASES:
        odir = os.path.join(OUT, c)
        if not os.path.isdir(odir):
            continue
        if os.path.exists(os.path.join(odir, "meta.json")):
            meta[c] = config_io.load_json(os.path.join(odir, "meta.json"))
        if os.path.exists(os.path.join(odir, "gate_repro.json")):
            gates[c] = config_io.load_json(os.path.join(odir, "gate_repro.json"))
        for f in sorted(os.listdir(odir)):
            if f.startswith("s_") and f.endswith(".json"):
                r = config_io.load_json(os.path.join(odir, f))
                t = r.get("table_row", {})
                rows.append(dict(case=c, kind=r["kind"], tag=r["tag"], status=r["status"],
                                 tau_uniform=r.get("tau_uniform"), tau_med=r.get("tau_med"),
                                 tau_min=r.get("tau_min"), tau_max=r.get("tau_max"),
                                 e_u_cell=t.get("e_u_cell"), e_phi=t.get("e_phi"),
                                 eK_flux_s=t.get("eK_flux_s"), e_ff=r.get("e_ff"),
                                 eta_m=r.get("eta_m"),
                                 kkt=r.get("kkt_relative_residual_recomputed"),
                                 iters=(r.get("solver") or {}).get("linear_solver_iterations"),
                                 info=(r.get("solver") or {}).get("linear_solver_info"),
                                 stab_share=r.get("stab_energy_share"), rho_p=r.get("rho_p"),
                                 solve_s=r.get("solve_s")))
    summary = dict(rows=rows, meta=meta, gates=gates, ALPHA=ALPHA, taus=TAUS,
                   script_sha256=config_io.sha(os.path.abspath(__file__)))
    # per-case comparison table
    comp = {}
    for c in CASES:
        u = [r for r in rows if r["case"] == c and r["kind"] == "uniform"
             and r["status"] == "ok" and r["e_u_cell"] is not None]
        if not u:
            continue
        u.sort(key=lambda r: (r["e_u_cell"], r["tau_uniform"]))
        best = u[0]
        one = next((r for r in u if abs(r["tau_uniform"] - 1.0) < 1e-12), None)
        e = {}
        for kind in ("RULE_V", "RULE_F"):
            rr = next((r for r in rows if r["case"] == c and r["kind"] == kind
                       and r["status"] == "ok"), None)
            e[kind] = rr
        def gap(x):
            return None if (x is None or best["e_u_cell"] in (None, 0)) else \
                100.0 * (x - best["e_u_cell"]) / best["e_u_cell"]
        comp[c] = dict(best_tau=best["tau_uniform"], best_e_u=best["e_u_cell"],
                       best_is_endpoint=bool(best["tau_uniform"] in (min(TAUS), max(TAUS))),
                       e_u_tau1=(one or {}).get("e_u_cell"), gap_tau1=gap((one or {}).get("e_u_cell")),
                       e_u_RULE_V=(e["RULE_V"] or {}).get("e_u_cell"),
                       gap_RULE_V=gap((e["RULE_V"] or {}).get("e_u_cell")),
                       e_u_RULE_F=(e["RULE_F"] or {}).get("e_u_cell"),
                       gap_RULE_F=gap((e["RULE_F"] or {}).get("e_u_cell")),
                       tau_RULE_V_med=(e["RULE_V"] or {}).get("tau_med"),
                       tau_RULE_F_med=(e["RULE_F"] or {}).get("tau_med"),
                       n_grid=len(u))
    summary["compare"] = comp
    config_io.save_json(os.path.join(OUT, "sweep_summary.json"), summary)
    hdr = ("case  best_tau  e_u(best)   e_u(1)  gap(1)%   e_u(RV)  gap(RV)%  tauRV_med   "
           "e_u(RF)  gap(RF)%")
    print(hdr)
    for c, v in comp.items():
        def f(x, n=3):
            return "-" if x is None else ("%%.%df" % n) % x
        print("%-5s %8s %10s %8s %8s %9s %9s %10s %9s %9s"
              % (c, f(v["best_tau"], 4), f(v["best_e_u"]), f(v["e_u_tau1"]), f(v["gap_tau1"], 2),
                 f(v["e_u_RULE_V"]), f(v["gap_RULE_V"], 2), f(v["tau_RULE_V_med"]),
                 f(v["e_u_RULE_F"]), f(v["gap_RULE_F"], 2)))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--sum", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.sum:
        summarise()
        return
    if not a.case:
        raise SystemExit("--case or --sum")
    run_case(a.case)


if __name__ == "__main__":
    main()
