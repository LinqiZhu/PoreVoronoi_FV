"""Assembly, the G1 fast-vs-production gate, and the pure / velocity-assisted solve entry point.

Solver: the paper's CPU MINRES (hybrid_voronoi_trace.build_hybrid_trace_factorization(solver='minres') +
solve_moment_constrained_hybrid_stokes), with the paper row's settings rtol 1e-14, maxiter 50000, refinement 1
(run record row), and the paper's residual gate inside hybrid_voronoi_trace (hybrid_voronoi_trace.py:1595-1602:
info == 0 and ||K x - rhs||_2 / ||rhs||_2 <= max(20 rtol, 1e-12), else RuntimeError). The assisted system is passed
through the same code with A, b replaced. A heartbeat wraps the MINRES callback without editing hybrid_voronoi_trace.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
from scipy import sparse

import config_io
import observations
import hybrid_voronoi_trace
import fast_assembly


def assemble(case, trace, kind, viscous_form="symmetric_gradient"):
    t = time.perf_counter()
    if kind == "fast":
        system, sizes = fast_assembly.assemble_vectorised(trace, viscosity=case.nu, body_force=case.force,
                                                          viscous_form=viscous_form, backend="numpy")
    elif kind == "prod":
        system, sizes = hybrid_voronoi_trace.assemble_moment_constrained_hybrid_stokes(
            trace, viscosity=case.nu, body_force=case.force, viscous_form=viscous_form), {}
    else:
        raise ValueError(kind)
    return system, dict(kind=kind, seconds=time.perf_counter() - t, **sizes)


def csr_identical(a, b):
    a, b = a.tocsr(), b.tocsr()
    return bool(a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
                and np.array_equal(a.data, b.data))


def g1(case, trace, viscous_form="symmetric_gradient"):
    """Gate G1: vectorised (fast) assembly equals the production loop assembly on this periodic complex."""
    prod, tp = assemble(case, trace, "prod", viscous_form)
    fast, tf = assemble(case, trace, "fast", viscous_form)
    gate = fast_assembly.gate(prod, fast)
    out = dict(gate=gate, pass_=bool(gate["pass"]), t_prod_s=tp["seconds"], t_fast_s=tf["seconds"],
               D_bitwise=csr_identical(prod.divergence_matrix, fast.divergence_matrix),
               A_bitwise=csr_identical(prod.velocity_matrix, fast.velocity_matrix),
               rhs_bitwise=bool(np.array_equal(prod.rhs_trace, fast.rhs_trace)),
               periodic_x=bool(trace.periodic_x), n_cells=int(trace.n_cells), n_dofs=int(3 * trace.n_trace_modes))
    return out, prod, fast


@contextmanager
def minres_heartbeat(label, each=500, every=60.0):
    original = hybrid_voronoi_trace.minres

    def wrapped(A, b, *args, callback=None, **kwargs):
        beat = config_io.Beat(label, every=every, each=each)

        def chained(xk):
            if callback is not None:
                callback(xk)
            beat()
        return original(A, b, *args, callback=chained, **kwargs)

    hybrid_voronoi_trace.minres = wrapped
    try:
        yield
    finally:
        hybrid_voronoi_trace.minres = original


def reduced_kkt(A, D):
    """The MINRES saddle point exactly as hybrid_voronoi_trace.build_hybrid_trace_factorization forms it
    (hybrid_voronoi_trace.py:1341, 1363-1369)."""
    Dr = D[:-1].tocsr()
    zero = sparse.csr_matrix((D.shape[0] - 1, D.shape[0] - 1), dtype=np.float64)
    return sparse.bmat([[A, Dr.T], [Dr, zero]], format="csr")


def state_vector(z, p):
    """Solution vector of the reduced KKT from (z, centred p): hybrid_voronoi_trace appends p_last = 0 then centres."""
    return np.concatenate([np.asarray(z, np.float64).ravel(), np.asarray(p[:-1], np.float64) - float(p[-1])])


def paper_solve(system, A=None, b=None, *, rtol, maxiter, refinement_steps, label="minres"):
    if (A is None) != (b is None):
        raise ValueError("pass both A and b, or neither")
    if A is None:
        solved = system
    else:
        nz = A.shape[0]
        solved = replace(system, velocity_matrix=A, rhs_trace=b,
                         body_force_matrix=np.column_stack([b, np.zeros(nz), np.zeros(nz)]),
                         body_force=np.array([1.0, 0.0, 0.0]))
    t0 = time.perf_counter()
    fac = hybrid_voronoi_trace.build_hybrid_trace_factorization(
        solved, solver="minres", iterative_rtol=rtol, iterative_maxiter=maxiter,
        iterative_refinement_steps=refinement_steps)
    t1 = time.perf_counter()
    with minres_heartbeat(label):
        result = hybrid_voronoi_trace.solve_moment_constrained_hybrid_stokes(solved, factorization=fac)
    t2 = time.perf_counter()
    receipt = {k: result[k] for k in ("linear_solver", "linear_solver_iterations", "linear_solver_refinement_steps",
                                      "linear_solver_info", "linear_solver_relative_residual", "linear_solver_rtol",
                                      "linear_solver_maxiter", "mass_inf_per_volume", "momentum_residual_inf",
                                      "velocity_matrix_symmetry_defect", "pressure_gauge", "solver_formulation_id",
                                      "n_trace_dofs", "factor_nnz")}
    receipt.update(factorization_s=t1 - t0, solve_call_s=t2 - t1, residual_gate_limit=max(20.0 * rtol, 1e-12),
                   residual_gate="hybrid_voronoi_trace.py:1595-1602 info==0 and relative 2-norm KKT residual <= limit")
    return result, receipt, fac, solved


def assert_pure_identity(system, fac, solved):
    """Bit-identity of the pure operator with the forward assembly, entrywise on the KKT MINRES iterated on."""
    independent = reduced_kkt(system.velocity_matrix, system.divergence_matrix)
    assert csr_identical(fac.kkt_matrix, independent), "pure KKT differs from the forward assembly"
    assert np.array_equal(solved.body_force_matrix @ solved.body_force, system.rhs_trace), "pure rhs differs"
    return True


def solve_entry(case, part, system, *, alpha, selection=None, R=None, G=None, solver_cfg, label="solve",
                trace_basis="connected_p1", check_identity=True):
    """Pure (alpha = 0) or velocity-assisted solve on a built partition.

    check_identity=False lets a timing driver run assert_pure_identity after its timed section."""
    trace = part["trace"]
    assert trace.trace_basis == "connected_p1" == trace_basis, trace.trace_basis
    t = time.perf_counter()
    if float(alpha) == 0.0:
        gamma, weight, H, y = 0.0, 0.0, None, None
    else:
        if selection is None or R is None or G is None:
            raise ValueError("assisted solve needs a selection and the recovery operators")
        H, y, _ = observations.observations(case, part, R, G, selection)
        gamma = observations.gamma_of(case, trace, alpha)
        weight = 1.0 / float(y.shape[0])  # one weight per observation: 1 / N_obs
    A, b, info = observations.assisted_operator(system, H, y, gamma, weight)
    t_obs = time.perf_counter() - t
    kw = dict(rtol=float(solver_cfg["rtol"]), maxiter=int(solver_cfg["maxiter"]),
              refinement_steps=int(solver_cfg["refinement_steps"]), label=label)
    if info["pure"]:
        assert A is system.velocity_matrix and b is system.rhs_trace
        result, receipt, fac, solved = paper_solve(system, **kw)
        if check_identity:
            info["kkt_bitwise_equals_forward_assembly"] = assert_pure_identity(system, fac, solved)
    else:
        result, receipt, fac, solved = paper_solve(system, A, b, **kw)
    receipt.update(alpha=float(alpha), observation_assembly_s=t_obs, **info)
    return result, receipt, fac, solved
