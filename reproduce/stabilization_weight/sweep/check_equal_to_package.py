"""check_equal_to_package: the weighted copies hybrid_voronoi_trace_weighted.py and
fast_assembly_weighted.py reproduce porevoronoi_fv/hybrid_voronoi_trace.py and fast_assembly.py.

The copies differ from the package modules only by an optional per-cell stabilization
multiplier tau_cell.  This script proves, on real complexes, that (m2 = hybrid_voronoi_trace,
m3 = fast_assembly, *_w = the weighted copy)

  G_m2_none   m2_w.assemble(..., tau_cell=None)   is BITWISE equal to m2.assemble(...)
  G_m2_ones   m2_w.assemble(..., tau_cell=1)      is BITWISE equal to m2.assemble(...)
  G_m3_none   m3_w.assemble_vectorised(..., tau_cell=None) is BITWISE equal to m3.assemble_vectorised(...)
  G_m3_ones   m3_w.assemble_vectorised(..., tau_cell=1)    is BITWISE equal to m3.assemble_vectorised(...)
  G_cross     for a NON-uniform random tau_cell, m2_w and m3_w agree to round-off
              (fast_assembly.gate, the vectorised-vs-loop check, applied to the tau systems)
  G_affine    A(tau) assembled directly equals A_cons + sum_i tau_i A_stab,i formed from
              A(tau_cell = e_i-style probes)?  -- tested in the cheap form
              A(c*tau) - A(0) == c * (A(tau) - A(0)) for c = 3.7, which is what "the multiplier
              multiplies the cell's stabilization block and nothing else" means.

  python -B check_equal_to_package.py --cases c1 [--m2]      (--m2 also runs the slow loop assembler)
"""
import sys, os, time, json, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = "../../../porevoronoi_fv"
sys.path.insert(0, HERE)
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, os.path.join(PACKAGE_DIR, "assisted"))

import config_io
config_io.thread_env(2)
import numpy as np

import assisted_arms, case_loader, cell_complex, stokes_solve
import hybrid_voronoi_trace, fast_assembly
import hybrid_voronoi_trace_weighted, fast_assembly_weighted

OUT = os.path.join(HERE, "../../../outputs/stabilization_weight/sweep")
os.makedirs(OUT, exist_ok=True)


def say(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def bitwise(a, b):
    a, b = a.tocsr(), b.tocsr()
    return bool(a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
                and np.array_equal(a.indices, b.indices) and np.array_equal(a.data, b.data))


def sys_bitwise(x, y):
    return dict(A=bitwise(x.velocity_matrix, y.velocity_matrix),
                D=bitwise(x.divergence_matrix, y.divergence_matrix),
                rhs=bool(np.array_equal(x.rhs_trace, y.rhs_trace)),
                bfm=bool(np.array_equal(x.body_force_matrix, y.body_force_matrix)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="c1")
    ap.add_argument("--m2", action="store_true")
    a = ap.parse_args()
    cfg = assisted_arms.load_cfg()
    vf = cfg["viscous_form"]
    report = {}
    for c in a.cases.split(","):
        case = case_loader.load_case(cfg, c, with_state=False)
        rec, seeds, part = assisted_arms.published_partition(cfg, case)
        tr = part["trace"]
        nc = int(tr.n_cells)
        kw = dict(viscosity=case.nu, body_force=case.force, viscous_form=vf)
        r = {}
        say(c, "n_cells", nc)

        f3 = fast_assembly.assemble_vectorised(tr, backend="numpy", **kw)[0]
        f3n = fast_assembly_weighted.assemble_vectorised(tr, backend="numpy", tau_cell=None, **kw)[0]
        f3o = fast_assembly_weighted.assemble_vectorised(tr, backend="numpy",
                                        tau_cell=np.ones(nc), **kw)[0]
        r["G_m3_none"] = sys_bitwise(f3, f3n)
        r["G_m3_ones"] = sys_bitwise(f3, f3o)
        say(c, "m3 gates", json.dumps(r["G_m3_none"]), json.dumps(r["G_m3_ones"]))

        rng = np.random.default_rng(20260918)
        tau = np.exp(rng.normal(0.0, 0.8, nc))          # non-uniform, strictly positive
        f3t = fast_assembly_weighted.assemble_vectorised(tr, backend="numpy", tau_cell=tau, **kw)[0]
        r["tau_stats"] = dict(min=float(tau.min()), med=float(np.median(tau)), max=float(tau.max()))

        # G_affine: the multiplier scales the cell stabilization block linearly and nothing else.
        f3z = fast_assembly_weighted.assemble_vectorised(tr, backend="numpy", tau_cell=np.zeros(nc), **kw)[0]
        c_scale = 3.7
        f3s = fast_assembly_weighted.assemble_vectorised(tr, backend="numpy", tau_cell=c_scale * tau, **kw)[0]
        L = (f3t.velocity_matrix - f3z.velocity_matrix).tocsr()
        Rm = (f3s.velocity_matrix - f3z.velocity_matrix).tocsr()
        num = float(abs(Rm - c_scale * L).max()) if Rm.nnz else 0.0
        den = float(abs(Rm).max())
        r["G_affine_rel"] = num / den if den else 0.0
        r["G_zero_D_bitwise"] = bitwise(f3z.divergence_matrix, f3.divergence_matrix)
        say(c, "affine rel", r["G_affine_rel"])

        if a.m2:
            t0 = time.perf_counter()
            p2 = hybrid_voronoi_trace.assemble_moment_constrained_hybrid_stokes(tr, **kw)
            p2n = hybrid_voronoi_trace_weighted.assemble_moment_constrained_hybrid_stokes(tr, tau_cell=None, **kw)
            p2o = hybrid_voronoi_trace_weighted.assemble_moment_constrained_hybrid_stokes(tr, tau_cell=np.ones(nc), **kw)
            r["G_m2_none"] = sys_bitwise(p2, p2n)
            r["G_m2_ones"] = sys_bitwise(p2, p2o)
            p2t = hybrid_voronoi_trace_weighted.assemble_moment_constrained_hybrid_stokes(tr, tau_cell=tau, **kw)
            r["G_cross_tau1"] = fast_assembly.gate(p2, f3n)
            r["G_cross_tau"] = fast_assembly.gate(p2t, f3t)
            r["m2_seconds"] = time.perf_counter() - t0
            say(c, "m2 gates", json.dumps(r["G_m2_none"]), json.dumps(r["G_m2_ones"]),
                "cross_tau pass", r["G_cross_tau"].get("pass"))
        report[c] = r
    json.dump(report, open(os.path.join(OUT, "gate.json"), "w"), indent=1, default=str)
    say("written", os.path.join(OUT, "gate.json"))


if __name__ == "__main__":
    main()
