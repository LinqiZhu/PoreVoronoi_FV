"""Does the discrete pressure coupling degrade as the complex is refined?

The admissibility check tests Stokes solvability one complex at a time, which
answers "is this system solvable" but not "does solvability survive
refinement".  The second question matters because a mesh-uniform inf-sup
constant is exactly what the analysis in the paper does not supply.

This module measures the substitute: the discrete pressure-coupling proxy
beta_h and the Schur condition proxy along the controlled refinement families,
so their trend under refinement is visible even though no bound is proved.

Every spectral quantity comes from `stability_core.schur_spectrum`, called with
the FULL divergence operator and the FULL cell-volume vector exactly as
`gpu/studies/audit_gauge_fixed_stability.py` calls it; that convention was
verified to reproduce that script's fibrous-proxy value to 8.5e-13 relative.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
import stability_core as sc  # noqa: E402

PROTOCOL_ID = "pvfv_stability_along_refinement"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/figure_06/refinement/runs")
    parser.add_argument("--out", default="reproduce/supplementary/s2_discretization")
    parser.add_argument("--levels", default="M0,M1,M2,M3",
                        help="level prefixes to audit; the finest levels are "
                             "excluded by default because the dense Schur "
                             "decomposition is quadratic in the pressure dimension")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = tuple(args.levels.split(","))
    boot = ec.bootstrap(out / "_stability")
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]

    rows = []
    for row_path in sorted(Path(args.root).glob("*/*/row.json")):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if not row["level_label"].startswith(wanted):
            continue
        cache = out / "per_level" / (row["case"] + "__" + row["level_label"] + ".json")
        if args.resume and cache.exists():
            rows.append(json.loads(cache.read_text(encoding="utf-8")))
            print("[resume] %s %s" % (row["case"], row["level_label"]), flush=True)
            continue

        with np.load(row_path.parent / "sites_and_partition.npz") as data:
            sites = np.asarray(data["seed_flat"], dtype=np.int64)
        started = time.perf_counter()
        geom, _meta, _t = ec.build_geometry_from_seed_flat(
            sites, mask_path=ec.case_paths(row["case"])["mask"],
            seed_spec="stability:" + row["level_label"],
        )
        trace = boot["build_hybrid_trace_geometry"](
            ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
        )
        system = boot["assemble_moment_constrained_hybrid_stokes"](
            trace, viscosity=float(cfg.nu),
            body_force=np.asarray(ec.FORWARD_ARGS["body_force"], dtype=np.float64),
            viscous_form=str(ec.FORWARD_ARGS["viscous_form"]),
        )
        A = system.velocity_matrix.tocsr()
        D = system.divergence_matrix.tocsr()
        volume = np.asarray(trace.cell_volume, dtype=np.float64)
        n_cells = int(trace.n_cells)

        spectrum = sc.schur_spectrum(A, D, volume, gauge_index=n_cells - 1)
        kkt = sc.build_reduced_kkt(A, sc.gauge_reduced_divergence(D, n_cells - 1))
        record = {
            "protocol_id": PROTOCOL_ID,
            "case": row["case"],
            "level_label": row["level_label"],
            "N_cv": int(row["N_cv"]),
            "N_system": int(row["N_system"]),
            "h_S_over_h": row.get("h_S_over_h"),
            "q_S_over_h": row.get("q_S_over_h"),
            "e_u_percent": row["e_u_percent"],
            "kkt_symmetry_defect": sc.kkt_symmetry_defect(kkt),
            "A_diag_min": float(np.min(A.diagonal())),
            "wall_seconds": float(time.perf_counter() - started),
        }
        for key in ("pressure_dimension_before_gauge", "pressure_dimension_after_gauge",
                    "rank_defect_estimate_after_gauge", "schur_lambda_min_positive",
                    "schur_lambda_max", "beta_h", "schur_condition_proxy",
                    "spectral_method", "spectral_status", "spectral_residual_max",
                    "spectral_tolerance", "gauge_choice"):
            if key in spectrum:
                record[key] = spectrum[key]
        cache.parent.mkdir(parents=True, exist_ok=True)
        ec.write_json(cache, record)
        rows.append(record)
        print(
            "[stab] %-16s %-10s N_c=%5d  H_g/h=%s  rank defect=%s  "
            "beta_h=%.4e  cond=%.4e  %s (%.0fs)"
            % (record["case"], record["level_label"], record["N_cv"],
               record.get("h_S_over_h"), record.get("rank_defect_estimate_after_gauge"),
               record.get("beta_h", float("nan")),
               record.get("schur_condition_proxy", float("nan")),
               record.get("spectral_status"), record["wall_seconds"]),
            flush=True,
        )
        del system, A, D, kkt, geom, trace
        ec.free_gpu()

    if rows:
        ec.write_csv(out / "stability_along_refinement.csv", rows,
                     sorted({k for r in rows for k in r}))
        print("[write] " + str(out / "stability_along_refinement.csv"))
        for case in sorted({r["case"] for r in rows}):
            block = sorted((r for r in rows if r["case"] == case),
                           key=lambda r: r["N_cv"])
            betas = [r.get("beta_h") for r in block if r.get("beta_h") is not None]
            if len(betas) > 1:
                print("  %-16s beta_h: %s   (finest/coarsest = %.3f)"
                      % (case, ", ".join("%.3e" % b for b in betas),
                         betas[-1] / betas[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
