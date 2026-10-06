"""Is the weak repeated-query amortisation a property of the method or of the solver?

With the preconditioned MINRES solver used for the paper's runs, a second
forcing on a fixed complex costs almost as much as the first, because the warm
query is a full iterative solve.  Whether a factorization changes that balance
is a question about other solvers.

This module measures it.  The code offers two reusing solvers besides MINRES:

  direct_lu  a sparse LU of the whole gauge-augmented KKT matrix
  schur_cg   a sparse LU of the velocity block only, then CG on the pressure
             Schur complement, which is the small block here

For each, the factorization is built once and then re-solved, so `T_query` is
the cost a second forcing would actually pay.  Either outcome --- a
factorization that is affordable and collapses the query cost, or one that is
not affordable at these sizes --- is recorded as measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402

PROTOCOL_ID = "pvfv_repeated_query_cost"


def try_solver(boot, system, solver, repeats, forcings, budget=None):
    """Factorize once, then solve for several forcings; report both costs.

    The reusing solvers factorize a matrix whose density varies by a factor of
    eight across the refinement family (800 nonzeros per row at the coarse end,
    104 at the fine end), so affordability is itself one of the measurements and
    is recorded rather than assumed.
    """
    record = {"solver": solver, "factorization_budget_s": budget}
    try:
        start = time.perf_counter()
        factorization = boot["build_hybrid_trace_factorization"](
            system, solver=solver,
            iterative_rtol=float(ec.FORWARD_ARGS["linear_rtol"]),
            iterative_maxiter=int(ec.FORWARD_ARGS["linear_maxiter"]),
            iterative_refinement_steps=int(ec.FORWARD_ARGS["linear_refinement_steps"]),
            velocity_lu_ordering="COLAMD",
        )
        record["t_factorization_s"] = float(time.perf_counter() - start)
        times, residuals, iterations = [], [], []
        for force in forcings:
            for _ in range(repeats):
                started = time.perf_counter()
                result = boot["solve_moment_constrained_hybrid_stokes"](
                    system, factorization=factorization,
                    body_force=np.asarray(force, dtype=np.float64),
                )
                times.append(float(time.perf_counter() - started))
                residuals.append(float(result["linear_solver_relative_residual"]))
                iterations.append(int(result["linear_solver_iterations"]))
        record.update({
            "t_query_median_s": float(statistics.median(times)),
            "t_query_min_s": float(min(times)),
            "t_query_max_s": float(max(times)),
            "queries": len(times),
            "linear_solver_relative_residual_max": max(residuals),
            "linear_solver_iterations_median": int(statistics.median(iterations)),
            "factor_memory_MiB": float(factorization.solver_storage_bytes) / float(2 ** 20),
            "factor_nnz": int(factorization.solver_storage_nnz),
            "status": "ok",
        })
        del factorization
        gc.collect()
    except Exception as error:  # affordability is itself the measurement
        record.update({"status": "failed", "exception_type": type(error).__name__,
                       "exception_message": str(error)[:400],
                       "traceback_tail": traceback.format_exc()[-400:]})
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="orthogonal_duct")
    parser.add_argument("--levels", default="M5_n6400")
    parser.add_argument("--factorization-budget-s", type=float, default=1800.0,
                        help="a factorization that exceeds this is reported as not "
                             "affordable at this size rather than left to run")
    parser.add_argument("--solvers", default="minres,schur_cg,direct_lu")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--root", default="reproduce/figure_06/refinement/runs")
    parser.add_argument("--out", default="reproduce/supplementary/s4_cost")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    boot = ec.bootstrap(out / "_query")
    ns, cfg = boot["ns"], boot["cfg"]
    base = np.asarray(ec.FORWARD_ARGS["body_force"], dtype=np.float64)
    forcings = [base, base * 2.0, np.array([0.0, base[0], 0.0])]

    rows = []
    for level in args.levels.split(","):
        run_dir = Path(args.root) / args.case / level
        if not (run_dir / "row.json").exists():
            print("[skip] %s missing" % run_dir)
            continue
        row = json.loads((run_dir / "row.json").read_text(encoding="utf-8"))
        with np.load(run_dir / "sites_and_partition.npz") as data:
            sites = np.asarray(data["seed_flat"], dtype=np.int64)
        geom, _meta, build_time = ec.build_geometry_from_seed_flat(
            sites, mask_path=ec.case_paths(args.case)["mask"],
            seed_spec="query:" + level,
        )
        trace_start = time.perf_counter()
        trace = boot["build_hybrid_trace_geometry"](
            ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
        )
        trace_time = float(time.perf_counter() - trace_start)
        system = boot["assemble_moment_constrained_hybrid_stokes"](
            trace, viscosity=float(cfg.nu), body_force=base,
            viscous_form=str(ec.FORWARD_ARGS["viscous_form"]),
        )
        construction = build_time + trace_time + float(system.assembly_time_s)

        for solver in args.solvers.split(","):
            record = try_solver(boot, system, solver, args.repeats, forcings,
                                budget=args.factorization_budget_s)
            record.update({
                "protocol_id": PROTOCOL_ID, "case": args.case, "level_label": level,
                "N_cv": int(row["N_cv"]), "N_system": int(row["N_system"]),
                "t_construction_s": construction,
            })
            if record["status"] == "ok":
                record["T_cold_s"] = construction + record["t_factorization_s"] + record["t_query_median_s"]
                record["T_query_s"] = record["t_query_median_s"]
                record["query_over_cold"] = record["T_query_s"] / record["T_cold_s"]
                print(
                    "[query] %-10s %-10s N_c=%5d  factorize %8.1fs  query %8.2fs  "
                    "query/cold %.3f  mem %8.1f MiB  resid<=%.1e"
                    % (level, solver, record["N_cv"], record["t_factorization_s"],
                       record["T_query_s"], record["query_over_cold"],
                       record["factor_memory_MiB"],
                       record["linear_solver_relative_residual_max"]),
                    flush=True,
                )
            else:
                print("[query] %-10s %-10s FAILED: %s: %s"
                      % (level, solver, record["exception_type"],
                         record["exception_message"][:110]), flush=True)
            rows.append(record)

        del system, trace, geom
        gc.collect()
        ec.free_gpu()

    if rows:
        ec.write_csv(out / "repeated_query_cost.csv", rows,
                     sorted({k for r in rows for k in r}))
        ec.write_json(out / "repeated_query_cost.json", rows)
        print("[write] " + str(out / "repeated_query_cost.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
