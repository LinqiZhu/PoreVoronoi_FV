"""Unit tests without any flow solve.

S  selectors on the loaded paper data, all six cases:
   A, all records of the window (pub): vectorised rule == independent brute-force loop; exactly one record per cell;
   every selected record lies in its cell; partition checks (wrap_same == 0, cell x-extent < L/2); particle
   statistics incl. the maximum-matching bound; point-operator defect on the selected observations; gamma.
   B, one frame (config key single_frame): same, plus every selected record from a distinct particle and
   N_cells == occupied voxels at that frame.
G  G1 routine on a tiny synthetic periodic instance: production loop vs vectorised assembly passes
   fast_assembly.gate; the gate fails when one A entry or one D entry is perturbed (instrument bites).
P  pure-path operator identity on the tiny instance: gamma = 0 returns the forward A and b objects and the MINRES
   KKT built by hybrid_voronoi_trace (factorization only, no iteration) equals the independent reduced KKT
   entrywise; gamma > 0 with random observations changes A (instrument bites).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import os
import time
from types import SimpleNamespace

import config_io
config_io.thread_env(1)
import numpy as np

import case_loader
import cell_complex
import observations
import stokes_solve
import hybrid_voronoi_trace
import velocity_recovery as rp


def selector_tests(cfg, code):
    case = case_loader.load_case(cfg, code, with_state=False)
    out = dict(case=code)
    for design, spec in (("A", "pub"), ("B", f"frm:{int(cfg['single_frame'])}")):
        t = time.perf_counter()
        rec = case_loader.select_records(case, spec)
        seeds = cell_complex.sites_of(case, rec)
        part = cell_complex.build(case, seeds, order="paper")
        sel = observations.nearest_centroid_selection(case, part, rec)
        brute = observations.brute_force_selection(case, part, rec)
        geom = part["geom"]
        owners_of_selected = geom.labels.ravel()[case.records["flat"][sel["record"]]]
        one_per_cell = bool(np.all(np.bincount(owners_of_selected, minlength=geom.n_cells) == 1))
        inside = bool(np.array_equal(owners_of_selected, np.arange(geom.n_cells)))
        stats = observations.particle_stats(case, part, rec, sel)
        system, _ = stokes_solve.assemble(case, part["trace"], "fast")
        R, G, gd = rp.recovery_operators(part["trace"], system)
        H, y, xyz = observations.observations(case, part, R, G, sel)
        defect = observations.point_operator_defect(H, xyz, sel["cell"], part["trace"], R, G)
        # a second, reversed-order call must give the same records (determinism, order independence)
        sel_rev = observations.nearest_centroid_selection(case, part, rec[::-1].copy())
        res = dict(spec=spec, n_records=int(rec.size), n_cells=int(geom.n_cells),
                   equals_brute_force=bool(np.array_equal(sel["record"], brute)),
                   order_independent=bool(np.array_equal(sel["record"], sel_rev["record"])),
                   one_record_per_cell=one_per_cell, selected_inside_own_cell=inside,
                   partition_checks=part["checks"], particle_stats=stats, point_operator_defect=defect,
                   gradient_divergence_defect=gd, H_shape=list(H.shape),
                   gamma_alpha1000=observations.gamma_of(case, part["trace"], cfg["alpha"]), weight=1.0 / y.shape[0],
                   seconds=time.perf_counter() - t)
        ok = (res["equals_brute_force"] and res["order_independent"] and one_per_cell and inside
              and part["checks"]["pass_"] and defect < 1e-10)
        if design == "B":
            pid = case.records["particle"][sel["record"]]
            res["distinct_particles"] = int(np.unique(pid).size)
            res["occupied_voxels_at_frame"] = int(np.unique(case.records["flat"][rec]).size)
            res["particles_at_frame"] = int(np.unique(case.records["particle"][rec]).size)
            ok = ok and res["distinct_particles"] == geom.n_cells == res["occupied_voxels_at_frame"]
        res["pass"] = bool(ok)
        out[design] = res
    return out


def tiny_case(seed=5):
    rng = np.random.default_rng(seed)
    shape = (5, 6, 12)
    mask = np.ones(shape, dtype=bool)
    mask[1:4, 2:4, 4:7] = False       # interior obstacle
    mask[0, 0, 9:12] = False           # a wall notch touching the periodic seam
    case = SimpleNamespace(mask=mask, shape=shape, h=1.0, periodic_x=True, length_x=float(shape[2]), nu=1.0,
                           force=np.array([0.002, 0.0, 0.0]))
    pore = np.flatnonzero(mask.ravel())
    seeds = np.sort(rng.choice(pore, size=14, replace=False))
    return case, seeds


def g1_tiny():
    case, seeds = tiny_case()
    part = cell_complex.build(case, seeds, order="paper")
    res, prod, fast = stokes_solve.g1(case, part["trace"])
    import fast_assembly
    # instrument bites: perturb one A value and one D value of the vectorised system
    A = fast.velocity_matrix.copy()
    A.data[A.data.size // 2] += 1e-9 * float(np.abs(A.data).max())  # 1e-9 of the gate's scale (tol 1e-12)
    bad_A = fast_assembly.gate(prod, SimpleNamespace(**{**fast.__dict__, "velocity_matrix": A}))
    D = fast.divergence_matrix.copy()
    D.data[0] += 1e-9 * float(np.abs(D.data).max())
    bad_D = fast_assembly.gate(prod, SimpleNamespace(**{**fast.__dict__, "divergence_matrix": D}))
    return dict(n_cells=int(part["geom"].n_cells), n_modes=int(part["trace"].n_trace_modes),
                wrap_same=part["checks"]["wrap_same"], g1_pass=res["pass_"], D_bitwise=res["D_bitwise"],
                A_bitwise=res["A_bitwise"], A_max_rel_diff=res["gate"]["A"]["max_rel_diff"],
                D_max_rel_diff=res["gate"]["D"]["max_rel_diff"], perturbed_A_gate_pass=bool(bad_A["pass"]),
                perturbed_D_gate_pass=bool(bad_D["pass"]),
                pass_=bool(res["pass_"] and not bad_A["pass"] and not bad_D["pass"])), (case, part, fast)


def pure_identity_tiny(case, part, system):
    A, b, info = observations.assisted_operator(system, None, None, 0.0, 0.0)
    same_objects = A is system.velocity_matrix and b is system.rhs_trace
    fac = hybrid_voronoi_trace.build_hybrid_trace_factorization(
        system, solver="minres", iterative_rtol=1e-14, iterative_maxiter=50000,
        iterative_refinement_steps=1)  # builds the KKT; no iteration
    kkt_equal = stokes_solve.csr_identical(fac.kkt_matrix,
                                           stokes_solve.reduced_kkt(system.velocity_matrix, system.divergence_matrix))
    rhs_equal = bool(np.array_equal(system.body_force_matrix @ system.body_force, system.rhs_trace))
    R, G, _ = rp.recovery_operators(part["trace"], system)
    rng = np.random.default_rng(3)
    cells = np.arange(part["geom"].n_cells)
    xyz = part["geom"].centroid + rng.uniform(-0.2, 0.2, (cells.size, 3))
    H = rp.point_operator(xyz, cells, part["trace"], R, G)
    A2, b2, info2 = observations.assisted_operator(system, H, rng.normal(size=(cells.size, 3)), 1.0, 1.0 / cells.size)
    assisted_differs = not stokes_solve.csr_identical(A2, system.velocity_matrix)
    sym = float(abs(A2 - A2.T).max() / abs(A2).max())
    return dict(gamma0_returns_forward_objects=bool(same_objects), kkt_bitwise_equal=kkt_equal, rhs_bitwise_equal=rhs_equal,
                assisted_operator_differs=bool(assisted_differs), assisted_symmetry_defect=sym,
                pass_=bool(same_objects and kkt_equal and rhs_equal and assisted_differs and sym < 1e-14))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--cases", default="c1,c2,c3,c4,c5,c6")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = config_io.load_cfg(a.cfg)
    out_dir = config_io.ensure_dir(a.out or os.path.join(cfg["out_dir"], "check_selectors_and_assembly"))
    report = dict(environment=config_io.environment(), code=config_io.code_hashes())
    g, (case, part, fast) = g1_tiny()
    report["G1_tiny"] = g
    print("G1_tiny", g, flush=True)
    report["pure_identity_tiny"] = pure_identity_tiny(case, part, fast)
    print("pure_identity_tiny", report["pure_identity_tiny"], flush=True)
    report["selectors"] = {}
    for code in a.cases.split(","):
        r = selector_tests(cfg, code)
        report["selectors"][code] = r
        config_io.save_json(os.path.join(out_dir, f"s_{code}.json"), r)
        print(code, {d: dict(pass_=r[d]["pass"], cells=r[d]["n_cells"],
                             distinct=r[d]["particle_stats"]["distinct_particles_among_selected"],
                             unique_cells=r[d]["particle_stats"]["cells_whose_selected_particle_is_unique"],
                             matching=r[d]["particle_stats"]["max_cells_with_distinct_particle_matching"],
                             defect=r[d]["point_operator_defect"]) for d in ("A", "B")}, flush=True)
    report["pass_"] = bool(g["pass_"] and report["pure_identity_tiny"]["pass_"]
                           and all(r[d]["pass"] for r in report["selectors"].values() for d in ("A", "B")))
    config_io.save_json(os.path.join(out_dir, "summary.json"), report)
    print("check_selectors_and_assembly pass", report["pass_"])


if __name__ == "__main__":
    main()
