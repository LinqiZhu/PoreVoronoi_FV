"""A computable upper bound on the dimensionless mass-balance backward error.

The exact quantity of Eq. (mass-backward-error) is

    eta_m = ||D z||_inf / ( || |D| |z| ||_inf + eps max(1, || |D| |z| ||_inf) ),

whose denominator sums the unsigned throughput of every *facelet* of a cell.
The archived states store the aggregate flux per stored edge, not per facelet.
Replacing the facelet sum by the aggregate-edge sum can only make the
denominator smaller, because aggregation can cancel within an edge but never
grow the total.  The ratio computed here is therefore an **upper bound** on
eta_m, and a bound is exactly what a statement of the form "below 1e-12"
requires.

The numerator is exact: (D z)_i is the signed sum of the aggregate fluxes of the
edges incident on cell i, which is what the stored `phi` holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

PROTOCOL_ID = "pvfv_mass_backward_error"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/figure_06/refinement/runs")
    parser.add_argument("--out", default="reproduce/supplementary/s3_verification")
    args = parser.parse_args()

    boot = ec.bootstrap(Path(args.out) / "_eta")
    cp = boot["cp"]
    rows = []
    for row_path in sorted(Path(args.root).glob("*/*/row.json")):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        case = row["case"]
        with np.load(row_path.parent / "sites_and_partition.npz") as data:
            sites = np.asarray(data["seed_flat"], dtype=np.int64)
        with np.load(row_path.parent / "state.npz") as data:
            phi = np.asarray(data["phi"], dtype=np.float64)
            volume = np.asarray(data["volume"], dtype=np.float64)
        geom, _meta, _t = ec.build_geometry_from_seed_flat(
            sites, mask_path=ec.case_paths(case)["mask"],
            seed_spec="eta:" + row["level_label"],
        )
        owner = cp.asnumpy(geom.owner).astype(np.int64)
        neigh = cp.asnumpy(geom.neigh).astype(np.int64)
        n_cells = int(geom.n_cells)
        signed = np.zeros(n_cells, dtype=np.float64)
        unsigned = np.zeros(n_cells, dtype=np.float64)
        np.add.at(signed, owner, phi)
        np.add.at(signed, neigh, -phi)
        np.add.at(unsigned, owner, np.abs(phi))
        np.add.at(unsigned, neigh, np.abs(phi))
        eps = np.finfo(np.float64).eps
        denominator = float(np.max(unsigned))
        eta = float(np.max(np.abs(signed)) /
                    (denominator + eps * max(1.0, denominator)))
        record = {
            "protocol_id": PROTOCOL_ID,
            "case": case,
            "level_label": row["level_label"],
            "N_cv": int(row["N_cv"]),
            "mass_inf_per_volume_archived": float(row["mass_inf_per_volume"]),
            "mass_inf_absolute": float(np.max(np.abs(signed))),
            "aggregate_unsigned_throughput_max": denominator,
            "eta_m_aggregate_upper_bound": eta,
        }
        rows.append(record)
        print(
            "[eta] %-16s %-10s N_c=%5d  |Dz|_inf=%.3e  denom=%.3e  "
            "eta_m <= %.3e" % (case, row["level_label"], record["N_cv"],
                               record["mass_inf_absolute"], denominator, eta),
            flush=True,
        )
        del geom
        ec.free_gpu()

    if rows:
        ec.write_csv(Path(args.out) / "mass_backward_error.csv", rows,
                     sorted({k for r in rows for k in r}))
        worst = max(r["eta_m_aggregate_upper_bound"] for r in rows)
        print("[write] %s" % (Path(args.out) / "mass_backward_error.csv"))
        print("worst upper bound over %d rows: %.3e" % (len(rows), worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
