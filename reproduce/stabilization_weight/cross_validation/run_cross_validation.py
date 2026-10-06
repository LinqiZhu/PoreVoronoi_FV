"""run_cross_validation: (A) does the velocity assist survive a properly chosen stabilisation weight,
and (B) can the records choose that weight with no reference field?

  python -B run_cross_validation.py --case c1 --kind MN   part A, assisted sweep, 11 taus   (checkpointed)
  python -B run_cross_validation.py --case c1 --kind PN   part A, pure sweep, 11 taus       (checkpointed)
  python -B run_cross_validation.py --case c1 --kind CV   part B, 8 taus x 3 folds          (checkpointed)
  python -B run_cross_validation.py --sum                 collate everything into <out>/cv_summary.json (cv_tables.py prints its tables)

Run from this folder.  <out> is outputs/stabilization_weight/cross_validation/; the stored PN and MN
solves at tau = 1 are read from outputs/assisted/<case>/{pn,mn}.json, written by
porevoronoi_fv/assisted/run_assisted_arms.py.

Every rule below was fixed before this script was run on any case.  No free parameter of this study is
fitted to c1..c6.

WHAT WAS RUN BEFORE THIS STUDY, AND WHY IT CANNOT BIAS IT
  A probe on c1 did exactly two things: it confirmed that a local MN solve at tau = 1 through the
  weighted assembler of ../sweep reproduces the stored MN solve, and it measured the wall time of one
  full-record solve (75.9 s) and one 2/3-record refit (74.7 s).  Those two times fix the fold count and
  the CV grid size below against the stated budget.  The probe computed no sweep, no error on U and no
  selection criterion, so no decision rule here depends on any accuracy result.

THE WEIGHT
  The assembly weights the residual stabilisation of cell i by w = 2 nu A_facelet / h
  (hybrid_voronoi_trace.py, fast_assembly.py).  Writing w_i = tau_i * 2 nu A / h, the default scheme is
  tau_i = 1 everywhere; tau = 1 is a dimensional choice, fitted to nothing, so c1..c6 are hold-outs.

CODE
  Solves use fast_assembly_weighted.assemble_vectorised (../sweep/fast_assembly_weighted.py), the copy of
  the package's vectorised assembler whose only difference is the optional per-cell multiplier tau_cell,
  shown by ../sweep/check_equal_to_package.py to be BITWISE equal to porevoronoi_fv/fast_assembly.py at
  tau_cell = None and at tau_cell = 1.  Everything downstream - recovery operators (velocity_recovery),
  observation operator (observations / velocity_recovery.point_operator), solver entry
  (stokes_solve.solve_entry), readouts (assisted_arms.readouts, readouts.table_row) - is the unmodified
  package code, which is only read here.
  The recovery operators R, G and the voxel field operator are built ONCE per case from the tau = 1
  assembly; they are functions of the trace geometry and of system.cell_recovery / divergence_matrix only,
  none of which the stabilisation block touches, so they are identical at every tau
  (velocity_recovery.recovery_operators asserts its own divergence consistency defect < 1e-11 on every
  build).

CASES, PARTITION, ARMS
  c1..c6 of porevoronoi_fv/config.json on the trajectory partition of the paper
  (assisted_arms.published_partition: every window record of every frame; 69,100-83,700 records,
  691-837 particles, 100 frames).
    PN   pure finite-volume solve, alpha = 0 (Table 6 configuration).
    MN   velocity-assisted, selector S-M (EVERY record is an observation), theta = alpha = cfg["alpha"]
         = 1e3 as in the paper, per-record weight 1/N_obs (stokes_solve.solve_entry),
         gamma = observations.gamma_of.
  The AN arm (one record per cell) is not swept.

SOLVER
  cfg["solver"] unchanged: hybrid_voronoi_trace MINRES, rtol 1e-14, maxiter 50000, 1 refinement step,
  1 BLAS thread.
  A solve that raises is recorded as status "solve_failed" with its message, is not retried, and no other
  setting is tried.

GRIDS
  PART A (both arms):  TAUS = 2^(j/2), j = -4..6 = 0.25, 0.3536, 0.5, 0.7071, 1, 1.4142, 2, 2.8284, 4,
  5.6569, 8 - the same eleven points as ../sweep/run_sweep.py.  A minimiser on an endpoint is reported
  as an endpoint and
  the grid is NOT extended.
  PART B (CV):  TAUS_CV = 2^(j/2), j = -4..3 = 0.25 ... 2.8284, the first EIGHT points of the same grid.
  DECLARED BUDGET REDUCTION: the CV part costs 6 cases x 8 taus x 3 folds = 144 refits on top of 132
  part-A solves; at the probed ~75 s solo (~150 s at the permitted three concurrent processes) the full
  eleven-point CV grid would add about 1.5 h for the three largest taus, which the grid sweep puts
  far above every observed optimum (PN's best tau is 0.5-1.41 on all six cases).  Part A keeps all eleven
  points.  If a case's reference-best tau from part A falls outside TAUS_CV, that case's CV comparison is
  reported as TRUNCATED and tau_CV is compared both to the global reference-best tau and to the best tau
  within TAUS_CV.

METRICS (identical definitions for every arm, tau and fold; the same operators as the grid sweep and
the assisted-arm readouts of the paper)
  O   = the pore voxels holding at least one window record, np.unique(records.flat[rec]) - the
        partition sites.  U = every other pore voxel.  O is built from the FULL record set and
        is therefore the same set for PN, for MN and for every fold; it is the O of the leave-out readout.
  e_ff      full-field voxel velocity error, velocity_recovery.normerr(Hfield z, u_ref_vox) over all pore
            voxels (fraction).  This is the grid sweep's e_ff and the paper's e_u,vox on all pore voxels.
  e_ff_U    the same with O zero-weighted (equals assisted_arms.leave_out_e_ff for MN).  This is the
            paper's "e_u,vox on the unobserved voxels U".
  e_ff_O    the same with U zero-weighted.
  e_u_cell, e_phi, eK_flux_s, eK_arch_s  (%): readouts.table_row through assisted_arms.readouts, unmodified.
  m_rec(tau) = ||H_all z - y_all||_2 / ||y_all||_2, the misfit of the arm's own field against ALL
            records, with the SAME H_all for both arms.  For MN this is an IN-SAMPLE misfit and
            is labelled as such; for PN no record was used, so it is out of sample.
  PRIMARY metric for "reference-best tau": e_u_cell, the metric of the grid sweep, so PN's sweep here and
  the grid sweep are directly comparable.  Ties -> the smaller tau.  The argmin under e_ff, e_ff_U, e_phi and |eK_flux_s| is
  reported alongside, and any disagreement is reported.
  "gap of X over Y" = 100 * (metric(X) - metric(Y)) / metric(Y).

PART B: THE FOLD RULE AND THE CRITERION  (fixed before any fold was built)
  Particle-grouped K = 3 folds.  p = np.unique(records.particle[rec]) ascending; perm =
  np.random.default_rng(20260919).permutation(p.size); fold[perm[t]] = t % 3; a record's fold is its
  particle's fold.  SEED 20260919.  Every record of a particle is in one fold, so a held-out record's
  trajectory never appears in training.
  For each case, each tau in TAUS_CV and each fold f: an MN solve whose observation set is the records
  with fold != f, weight 1/N_train (the paper's rule 1/N_obs applied to the training set), gamma
  unchanged (it depends only on alpha, nu and the cell volumes, not on the records).  The partition,
  cells, seeds and operators are NOT rebuilt - they come from the record POSITIONS, which a user has for
  every record whether or not its velocity is used; only the velocity data are held out.  Nothing in the
  criterion touches the reference field.
  CRITERION (the number the rule minimises), pooled over folds:
      e_cv(tau) = sqrt( sum_f ||H_te,f z_f - y_te,f||^2 / sum_f ||y_te,f||^2 )
  The folds partition the records, so the denominator is ||y_all||^2.
  tau_CV = argmin_{TAUS_CV} e_cv, ties -> the smaller tau.
  SUCCESS RULE, fixed in advance and reported whatever it gives:
      EXACT     tau_CV == tau_ref (the part-A reference-best tau under the primary metric).
      CLOSE     not exact, but the primary-metric penalty of tau_CV against tau_ref, evaluated on the
                FULL-RECORD part-A solves, is <= 10 %.
      FAIL      otherwise.
  The rule is declared USABLE as a reference-free selector for the data-assisted branch only if it is
  EXACT or CLOSE on at least 5 of the 6 cases AND its worst-case primary penalty over the six is <= 25 %.
  Otherwise it is declared NOT usable, and that is reported as the answer.
  Penalties of tau_CV against tau_ref are reported in EVERY metric, not only the primary one.

PART B FOR THE PURE ARM
  PN fits nothing to the records, so there is nothing to hold out: every record is already out of sample
  and a K-fold split of PN would produce K identical solves.  The available analogue is the same misfit
  on all records, m_rec(tau), computed from the part-A PN sweep.  Declared test: does argmin_tau m_rec
  (over the full eleven-point grid) equal PN's reference-best tau?  Reported with the same EXACT/CLOSE/
  FAIL rule.  The same quantity is reported for MN with the explicit warning that for MN it is in-sample
  and therefore not a hold-out criterion.

CHECKS
  G_repro   at tau = 1 the DECLARED ACCURACY KEYS of table_row - K_method, eK_arch_s, eK_flux_s, e_phi,
            e_u_cell, mean_flux_x - together with e_ff, and for MN additionally e_ff_U against the
            stored solve's e_ff_leave_out, reproduce the stored solves outputs/assisted/<case>/{pn,mn}.json
            to <= 1e-6 relative.  eta_m and r_inf_m are EXCLUDED by declaration: they are numerically
            zero (1e-17..1e-15 absolute), so their relative difference between two MINRES runs on
            different hardware is meaningless (the G_repro of ../sweep/run_sweep.py does compare them,
            which is why that check can fail on other hardware).  Mass is checked separately by G_mass.
            A failure is reported; the sweep still runs and every number from that case is labelled
            unreproduced.
  G_solver  linear_solver_info == 0 and the recomputed KKT relative residual <= 1e-12 on every solve.
  G_mass    eta_m <= 1e-12 on every solve.
  G_operators  velocity_recovery.recovery_operators' divergence defect < 1e-11 (its own assert) and
            assisted_arms.check_partition ok on every case.

WHAT THIS CANNOT SHOW
  One forcing (the uniform cfg["force"] of every controlled case), so the transferability of any selected
  tau across FORCINGS is untested; the manufactured-cube sweep shows the optimum is load-dependent.  One
  observation weight (theta = 1e3), so a tau-theta interaction beyond that value is untested.  Six
  controlled cases and no rock partition other than the Bentheimer crop.  The reference-best tau is located
  with the reference field and is never an achievable accuracy - it bounds the size of the stabilisation
  artefact, nothing more.  The CV criterion scores against MEASURED record velocities, so it inherits
  whatever error those carry; on c1..c6 they are noise-free DNS samples, which favours it.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(os.path.dirname(HERE), "sweep")
PACKAGE_DIR = "../../../porevoronoi_fv"
sys.path.insert(0, SWEEP_DIR)
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, os.path.join(PACKAGE_DIR, "assisted"))

import config_io  # noqa: E402
config_io.thread_env(1)
import numpy as np  # noqa: E402

import assisted_arms, case_loader, readouts, stokes_solve  # noqa: E402
import velocity_recovery as rp  # noqa: E402
import fast_assembly_weighted  # noqa: E402

OUT = os.path.join(HERE, "../../../outputs/stabilization_weight/cross_validation")
CASES = ["c1", "c2", "c3", "c4", "c5", "c6"]
TAUS = [float(2.0 ** (j / 2.0)) for j in range(-4, 7)]
TAUS_CV = [float(2.0 ** (j / 2.0)) for j in range(-4, 4)]
KFOLD = 3
SEED = 20260919
ACC_KEYS = ("K_method", "eK_arch_s", "eK_flux_s", "e_phi", "e_u_cell", "mean_flux_x")
PUB = ("../../../outputs/assisted")


def say(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def tau_tag(tau):
    return "u%08.4f" % tau


def build(code):
    """Everything that does not depend on tau: case, partition, operators, O/U weights, H_all."""
    cfg = assisted_arms.load_cfg()
    case = case_loader.load_case(cfg, code, with_state=False)
    rec, seeds, part = assisted_arms.published_partition(cfg, case)
    chk = assisted_arms.check_partition(case, rec, seeds, part)
    tr = part["trace"]
    nc = int(tr.n_cells)
    system1, _ = fast_assembly_weighted.assemble_vectorised(tr, viscosity=case.nu, body_force=case.force,
                                           viscous_form=cfg["viscous_form"], backend="numpy",
                                           tau_cell=np.full(nc, 1.0))
    R, G, gdef = rp.recovery_operators(tr, system1)
    Hfield = assisted_arms.field_operator(part, R, G)
    pore = np.asarray(case.pore, np.int64)
    O = np.unique(np.asarray(case.records["flat"], np.int64)[rec])
    pos = np.searchsorted(pore, O)
    assert np.all(pore[pos] == O), "observed voxel not in pore list"
    wU = np.ones(pore.size, np.float64)
    wU[pos] = 0.0
    wO = 1.0 - wU
    sel_all = assisted_arms.selection_all(case, part, rec)
    H_all = rp.point_operator(case.records["xyz"][rec], np.asarray(sel_all["cell"], np.int64), tr, R, G)
    y_all = np.asarray(case.records["vel"][rec], np.float64)
    ny = float(np.linalg.norm(y_all.ravel()))
    # particle-grouped folds (declared rule, seed 20260919)
    pid = np.asarray(case.records["particle"], np.int64)[rec]
    up = np.unique(pid)
    perm = np.random.default_rng(SEED).permutation(up.size)
    gp = np.empty(up.size, np.int64)
    gp[perm] = np.arange(up.size) % KFOLD
    fold = gp[np.searchsorted(up, pid)]
    return dict(cfg=cfg, case=case, rec=rec, seeds=seeds, part=part, tr=tr, nc=nc, chk=chk,
                R=R, G=G, gdef=gdef, Hfield=Hfield, pore=pore, O=O, wU=wU, wO=wO,
                sel_all=sel_all, H_all=H_all, y_all=y_all, ny=ny, fold=fold,
                n_particles=int(up.size))


def field_errors(B, z):
    val = (B["Hfield"] @ z).reshape(-1, 3)
    ref = B["case"].u_ref_vox
    return dict(e_ff=rp.normerr(val, ref), e_ff_U=rp.normerr(val, ref, weight=B["wU"]),
                e_ff_O=rp.normerr(val, ref, weight=B["wO"]))


def rec_misfit(B, z, rows=None):
    H, y = B["H_all"], B["y_all"]
    if rows is None:
        return float(np.linalg.norm(H @ z - y.ravel()) / max(B["ny"], 1e-300))
    Hs = H[rows]
    ys = y.ravel()[rows]
    return float(np.linalg.norm(Hs @ z - ys)), float(np.linalg.norm(ys))


def solve(B, tau, *, arm, records=None, label="s"):
    """One solve at a uniform tau.  records=None -> the full record set (part A / MN full fit)."""
    cfg, case, part, tr, nc = B["cfg"], B["case"], B["part"], B["tr"], B["nc"]
    system, _ = fast_assembly_weighted.assemble_vectorised(tr, viscosity=case.nu, body_force=case.force,
                                          viscous_form=cfg["viscous_form"], backend="numpy",
                                          tau_cell=np.full(nc, float(tau)))
    if arm == "PN":
        sel, alpha = None, 0.0
    else:
        sel = B["sel_all"] if records is None else assisted_arms.selection_all(case, part, records)
        alpha = float(cfg["alpha"])
    t0 = time.perf_counter()
    result, receipt, fac, solved = stokes_solve.solve_entry(case, part, system, alpha=alpha, selection=sel,
                                                  R=B["R"], G=B["G"], solver_cfg=cfg["solver"],
                                                  label=label, trace_basis=cfg["trace_basis"],
                                                  check_identity=True)
    wall = time.perf_counter() - t0
    del fac
    z = np.asarray(result["trace_coefficients"], np.float64).ravel()
    A_used, b_used = solved.velocity_matrix, solved.body_force_matrix @ solved.body_force
    return dict(result=result, receipt=receipt, system=system, z=z, A_used=A_used, b_used=b_used,
                solve_s=wall, n_obs=(0 if sel is None else int(np.asarray(sel["record"]).size)))


def run_part_a(code, arm):
    B = build(code)
    odir = os.path.join(OUT, code)
    os.makedirs(odir, exist_ok=True)
    meta = dict(case=code, tag=B["case"].tag, arm=arm, n_cells=B["nc"], n_dofs=int(3 * B["tr"].n_trace_modes),
                n_facelets=int(B["tr"].n_facelets), n_records=int(B["rec"].size), n_particles=B["n_particles"],
                pore_voxels=int(B["pore"].size), observed_voxels=int(B["O"].size),
                partition_check=B["chk"], recovery_defect=B["gdef"], taus=TAUS, taus_cv=TAUS_CV,
                kfold=KFOLD, seed=SEED, fold_sizes=[int(np.sum(B["fold"] == f)) for f in range(KFOLD)])
    config_io.save_json(os.path.join(odir, "meta_%s.json" % arm), meta)
    say(code, arm, "cells", B["nc"], "records", int(B["rec"].size), "particles", B["n_particles"],
        "pore", int(B["pore"].size), "O", int(B["O"].size))
    for tau in TAUS:
        tag = "a_%s_%s" % (arm, tau_tag(tau))
        path = os.path.join(odir, "%s.json" % tag)
        if os.path.exists(path) and config_io.load_json(path).get("status") in ("ok", "solve_failed"):
            say(code, tag, "skip (done)")
            continue
        row = dict(case=code, arm=arm, part="A", tau=float(tau), tag=tag)
        try:
            S = solve(B, tau, arm=arm, label="%s-%s" % (code, tag))
            H = y = None
            if arm != "PN":
                # exactly what observations.observations(case, part, R, G, sel_all) returns: same xyz,
                # same owners, same R, G, same row order (the records are the sorted full set).
                H, y = B["H_all"], B["y_all"]
            diag = assisted_arms.readouts(B["case"], B["part"], S["system"], S["A_used"], S["b_used"], S["result"],
                               B["Hfield"], H=H, y=y, assisted=(arm != "PN"))
            row.update(diag)
            row.update(field_errors(B, S["z"]))
            row["m_rec"] = rec_misfit(B, S["z"])
            row.update(solver=S["receipt"], solve_s=S["solve_s"], n_obs=S["n_obs"], status="ok")
            np.savez_compressed(os.path.join(odir, "%s.npz" % tag), z=S["z"])
        except Exception as e:  # noqa: BLE001
            row.update(status="solve_failed", error=repr(e)[:500])
        config_io.save_json(path, row)
        t = row.get("table_row", {})
        say(code, tag, row["status"], "e_u_cell", t.get("e_u_cell"), "e_ff", row.get("e_ff"),
            "e_ff_U", row.get("e_ff_U"), "m_rec", row.get("m_rec"), "%.0fs" % row.get("solve_s", 0))
    gate_repro(code, arm, B)


def gate_repro(code, arm, B):
    pub = os.path.join(PUB, code, "%s.json" % arm.lower())
    p1 = os.path.join(OUT, code, "a_%s_%s.json" % (arm, tau_tag(1.0)))
    g = dict(gate="G_repro", arm=arm, published=pub, keys=list(ACC_KEYS) + ["e_ff"], ok=None)
    if os.path.exists(pub) and os.path.exists(p1):
        a, b = config_io.load_json(pub), config_io.load_json(p1)
        worst, where = 0.0, None
        for k in ACC_KEYS:
            if k in a["table_row"] and k in b.get("table_row", {}):
                d = abs(b["table_row"][k] - a["table_row"][k]) / max(abs(a["table_row"][k]), 1e-300)
                if d > worst:
                    worst, where = d, k
        d = abs(b.get("e_ff", 0.0) - a["e_ff"]) / max(abs(a["e_ff"]), 1e-300)
        if d > worst:
            worst, where = d, "e_ff"
        if arm == "MN" and "e_ff_leave_out" in a:
            d = abs(b.get("e_ff_U", 0.0) - a["e_ff_leave_out"]) / max(abs(a["e_ff_leave_out"]), 1e-300)
            if d > worst:
                worst, where = d, "e_ff_U(vs e_ff_leave_out)"
        g.update(ok=bool(worst <= 1e-6), max_rel=worst, worst_key=where)
    config_io.save_json(os.path.join(OUT, code, "gate_repro_%s.json" % arm), g)
    say(code, arm, "G_repro", json.dumps(g))


def run_cv(code):
    B = build(code)
    odir = os.path.join(OUT, code)
    os.makedirs(odir, exist_ok=True)
    rec, fold = B["rec"], B["fold"]
    rows3 = np.arange(rec.size)[:, None] * 3 + np.arange(3)
    say(code, "CV folds", [int(np.sum(fold == f)) for f in range(KFOLD)], "particles", B["n_particles"])
    for tau in TAUS_CV:
        for f in range(KFOLD):
            tag = "b_CV_%s_f%d" % (tau_tag(tau), f)
            path = os.path.join(odir, "%s.json" % tag)
            if os.path.exists(path) and config_io.load_json(path).get("status") in ("ok", "solve_failed"):
                say(code, tag, "skip (done)")
                continue
            te = fold == f
            row = dict(case=code, arm="MN", part="B", tau=float(tau), fold=f, tag=tag,
                       n_train=int(np.sum(~te)), n_test=int(np.sum(te)))
            try:
                S = solve(B, tau, arm="MN", records=rec[~te], label="%s-%s" % (code, tag))
                z = S["z"]
                r_te, n_te = rec_misfit(B, z, rows3[te].ravel())
                r_tr, n_tr = rec_misfit(B, z, rows3[~te].ravel())
                row.update(resid_test=r_te, norm_test=n_te, resid_train=r_tr, norm_train=n_tr,
                           rel_test=r_te / max(n_te, 1e-300), rel_train=r_tr / max(n_tr, 1e-300))
                row.update(field_errors(B, z))
                row["kkt_relative_residual_recomputed"] = assisted_arms.kkt_residual(
                    S["A_used"], S["system"].divergence_matrix, S["b_used"], z,
                    np.asarray(S["result"]["p"], np.float64))
                mm = readouts.mass(S["system"].divergence_matrix, z,
                             np.asarray(B["part"]["geom"].volume, np.float64))
                row.update(eta_m=mm["eta_m"], r_inf_m=mm["r_inf_m"])
                row.update(solver=S["receipt"], solve_s=S["solve_s"], n_obs=S["n_obs"], status="ok")
            except Exception as e:  # noqa: BLE001
                row.update(status="solve_failed", error=repr(e)[:500])
            config_io.save_json(path, row)
            say(code, tag, row["status"], "rel_test", row.get("rel_test"), "e_ff_U", row.get("e_ff_U"),
                "%.0fs" % row.get("solve_s", 0))


# --------------------------------------------------------------------------------------- collation
def _metrics(r):
    t = r.get("table_row", {})
    return dict(e_u_cell=t.get("e_u_cell"), e_phi=t.get("e_phi"), eK_flux_s=t.get("eK_flux_s"),
                eK_flux_abs=(None if t.get("eK_flux_s") is None else abs(t["eK_flux_s"])),
                e_ff_pct=(None if r.get("e_ff") is None else 100.0 * r["e_ff"]),
                e_ff_U_pct=(None if r.get("e_ff_U") is None else 100.0 * r["e_ff_U"]),
                e_ff_O_pct=(None if r.get("e_ff_O") is None else 100.0 * r["e_ff_O"]),
                m_rec=r.get("m_rec"), tau=r.get("tau"))


def summarise():
    A = {c: {"PN": [], "MN": []} for c in CASES}
    CV = {c: {} for c in CASES}
    gates, meta = {}, {}
    for c in CASES:
        odir = os.path.join(OUT, c)
        if not os.path.isdir(odir):
            continue
        for f in sorted(os.listdir(odir)):
            p = os.path.join(odir, f)
            if f.startswith("meta_"):
                meta[c] = config_io.load_json(p)
            elif f.startswith("gate_repro_"):
                gates.setdefault(c, {})[f[11:-5]] = config_io.load_json(p)
            elif f.startswith("a_") and f.endswith(".json"):
                r = config_io.load_json(p)
                A[c][r["arm"]].append(r)
            elif f.startswith("b_CV_") and f.endswith(".json"):
                r = config_io.load_json(p)
                CV[c].setdefault(r["tau"], []).append(r)
    out = dict(taus=TAUS, taus_cv=TAUS_CV, kfold=KFOLD, seed=SEED, meta=meta, gates=gates,
               script_sha256=config_io.sha(os.path.abspath(__file__)))
    # ---- part A rows
    rowsA = []
    for c in CASES:
        for arm in ("PN", "MN"):
            for r in sorted(A[c][arm], key=lambda r: r["tau"]):
                t = r.get("table_row", {})
                rowsA.append(dict(case=c, arm=arm, tau=r["tau"], status=r["status"],
                                  e_u_cell=t.get("e_u_cell"), e_phi=t.get("e_phi"),
                                  eK_flux_s=t.get("eK_flux_s"), eK_arch_s=t.get("eK_arch_s"),
                                  e_ff=r.get("e_ff"), e_ff_U=r.get("e_ff_U"), e_ff_O=r.get("e_ff_O"),
                                  m_rec=r.get("m_rec"), eta_m=r.get("eta_m"),
                                  kkt=r.get("kkt_relative_residual_recomputed"),
                                  iters=(r.get("solver") or {}).get("linear_solver_iterations"),
                                  info=(r.get("solver") or {}).get("linear_solver_info"),
                                  obs_misfit=r.get("observation_misfit"), solve_s=r.get("solve_s")))
    out["rowsA"] = rowsA
    # ---- part A comparison
    comp = {}
    for c in CASES:
        d = {}
        for arm in ("PN", "MN"):
            rs = A[c][arm]
            if not rs:
                continue
            ok = [r for r in rs if r.get("status") == "ok" and r.get("table_row", {}).get("e_u_cell") is not None]
            if not ok:
                continue
            ok.sort(key=lambda r: (r["table_row"]["e_u_cell"], r["tau"]))
            best = ok[0]
            one = next((r for r in ok if abs(r["tau"] - 1.0) < 1e-12), None)
            argmins = {}
            for key, sign, get in (("e_u_cell", 1.0, lambda r: r["table_row"].get("e_u_cell")),
                                   ("e_phi", 1.0, lambda r: r["table_row"].get("e_phi")),
                                   ("eK_flux_abs", 1.0, lambda r: abs(r["table_row"].get("eK_flux_s", 1e9))),
                                   ("e_ff", 1.0, lambda r: r.get("e_ff")),
                                   ("e_ff_U", 1.0, lambda r: r.get("e_ff_U")),
                                   ("m_rec", 1.0, lambda r: r.get("m_rec"))):
                cand = [r for r in ok if get(r) is not None]
                cand.sort(key=lambda r: (get(r), r["tau"]))
                argmins[key] = cand[0]["tau"] if cand else None
            d[arm] = dict(best_tau=best["tau"], best=_metrics(best),
                          tau1=(_metrics(one) if one else None),
                          best_is_endpoint=bool(best["tau"] in (min(TAUS), max(TAUS))),
                          argmin_tau=argmins, n_grid=len(ok),
                          by_tau={("%.4f" % r["tau"]): _metrics(r) for r in ok})
        comp[c] = d
    out["compareA"] = comp
    # ---- part B
    cvsum = {}
    for c in CASES:
        if not CV[c]:
            continue
        per = {}
        for tau, rs in sorted(CV[c].items()):
            ok = [r for r in rs if r.get("status") == "ok"]
            if len(ok) != KFOLD:
                per["%.4f" % tau] = dict(n_folds=len(ok), incomplete=True)
                continue
            num = sum(r["resid_test"] ** 2 for r in ok)
            den = sum(r["norm_test"] ** 2 for r in ok)
            per["%.4f" % tau] = dict(tau=tau, n_folds=len(ok), e_cv=float(np.sqrt(num / max(den, 1e-300))),
                                     per_fold_rel=[r["rel_test"] for r in sorted(ok, key=lambda r: r["fold"])],
                                     per_fold_train_rel=[r["rel_train"] for r in sorted(ok, key=lambda r: r["fold"])],
                                     per_fold_e_ff_U=[r["e_ff_U"] for r in sorted(ok, key=lambda r: r["fold"])],
                                     n_train=[r["n_train"] for r in sorted(ok, key=lambda r: r["fold"])],
                                     n_test=[r["n_test"] for r in sorted(ok, key=lambda r: r["fold"])])
        good = [v for v in per.values() if v.get("e_cv") is not None]
        sel = min(good, key=lambda v: (v["e_cv"], v["tau"])) if good else None
        cvsum[c] = dict(per_tau=per, tau_cv=(sel["tau"] if sel else None), e_cv_min=(sel["e_cv"] if sel else None),
                        complete=bool(len(good) == len(TAUS_CV)))
        mn = comp.get(c, {}).get("MN")
        if sel and mn:
            tref = mn["best_tau"]
            inside = any(abs(tref - t) < 1e-12 for t in TAUS_CV)
            bt = mn["by_tau"]
            kcv, kref = "%.4f" % sel["tau"], "%.4f" % tref
            pen = {}
            if kcv in bt and kref in bt:
                for m in ("e_u_cell", "e_phi", "eK_flux_abs", "e_ff_pct", "e_ff_U_pct", "e_ff_O_pct"):
                    a, b = bt[kcv].get(m), bt[kref].get(m)
                    pen[m] = None if (a is None or not b) else 100.0 * (a - b) / b
            # best inside the CV grid, for the truncated case
            ins = [(t, bt["%.4f" % t]) for t in TAUS_CV if "%.4f" % t in bt]
            tref_in = min(ins, key=lambda kv: (kv[1]["e_u_cell"], kv[0]))[0] if ins else None
            verdict = ("EXACT" if abs(sel["tau"] - tref) < 1e-12
                       else ("CLOSE" if (pen.get("e_u_cell") is not None and pen["e_u_cell"] <= 10.0) else "FAIL"))
            cvsum[c].update(tau_ref=tref, tau_ref_inside_cv_grid=inside, tau_ref_within_cv_grid=tref_in,
                            penalty=pen, verdict=verdict)
    out["partB"] = cvsum
    # ---- pure-arm / in-sample record-misfit selector
    msel = {}
    for c in CASES:
        d = {}
        for arm in ("PN", "MN"):
            a = comp.get(c, {}).get(arm)
            if not a:
                continue
            tm = a["argmin_tau"].get("m_rec")
            tref = a["best_tau"]
            bt = a["by_tau"]
            pen = {}
            if tm is not None and ("%.4f" % tm) in bt and ("%.4f" % tref) in bt:
                for m in ("e_u_cell", "e_phi", "eK_flux_abs", "e_ff_pct", "e_ff_U_pct"):
                    x, y = bt["%.4f" % tm].get(m), bt["%.4f" % tref].get(m)
                    pen[m] = None if (x is None or not y) else 100.0 * (x - y) / y
            d[arm] = dict(tau_mrec=tm, tau_ref=tref, penalty=pen,
                          verdict=("EXACT" if (tm is not None and abs(tm - tref) < 1e-12)
                                   else ("CLOSE" if (pen.get("e_u_cell") is not None and pen["e_u_cell"] <= 10.0)
                                         else "FAIL")),
                          in_sample=(arm != "PN"))
        msel[c] = d
    out["mrec_selector"] = msel
    config_io.save_json(os.path.join(OUT, "cv_summary.json"), out)
    print(json.dumps({c: {a: (v.get("best_tau"), v.get("argmin_tau")) for a, v in comp.get(c, {}).items()}
                      for c in CASES}, indent=1))
    print(json.dumps({c: dict(tau_cv=v.get("tau_cv"), tau_ref=v.get("tau_ref"), verdict=v.get("verdict"))
                      for c, v in cvsum.items()}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--kind", choices=["MN", "PN", "CV"])
    ap.add_argument("--sum", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.sum:
        summarise()
        return
    if not a.case or not a.kind:
        raise SystemExit("--case and --kind, or --sum")
    if a.kind == "CV":
        run_cv(a.case)
    else:
        run_part_a(a.case, a.kind)


if __name__ == "__main__":
    main()
