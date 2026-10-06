"""The two sides of `e_K` are computed by different functionals.

Permeability is the certificate this method's accuracy claim rests on, so the
functional used to form it has to be the same on both sides of the comparison.
It is not.

  method side     K = nu * (sum_f phi_f d_f)_x / sum_i V_i / f_x        (flux)
  reference side  the value stored in the reference `.npz`, which the loader
                  passes through verbatim, and which reproduces
                  nu * (sum_i V_i U_i)_x / sum_i V_i / f_x              (velocity)

Both appear in `study_common.solve_forward` and in
`run_particle_trace_transfer`; the reference loader
(`run_segmented_selector_geodesic_rows.load_reference_npz`) sets
`"K_eff_x": float(data["K_eff_x"])` with no recomputation.

On a continuum incompressible field the two functionals agree.  On a discrete
field they differ by the reconstruction inconsistency of whichever field they are
applied to, and that inconsistency is not symmetric here:

  * the method's own two values agree to machine precision after the
    divergence-constrained projection --- the archived runs record
    `mean_trace_moment_velocity_flux_relative_mismatch` between 7.7e-16 and
    5.6e-14 over fourteen runs;
  * the reference's do not, by 0.000 % on an axis-aligned duct up to 3.874 % on
    a real-rock crop.

So the offset is a property of the reference field alone, it is case-dependent,
and it enters every reported `e_K` as a systematic term.  This module measures it
on every reference available locally and recomputes `e_K` under the consistent
flux-to-flux definition wherever the run records store the method's own value.

Which functional is "right" is a separate question and is not decided here.  The
flux form is chosen for the recomputed number because it is the conservative
quantity, it is what the method already computes, and a Darcy velocity is a flux.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

PROTOCOL_ID = "pvfv_permeability_readout"
COMPUTE_ROOT = Path(os.environ.get(
    "PVFV_GPU_ROOT",
    "gpu",
))
FORCE_X = 0.002
VISCOSITY = 1.0

REFERENCES = {
    "orthogonal_duct": "../data/controlled_cases/orthogonal_duct/reference_flow.npz",
    "skewed_duct": "../data/controlled_cases/skewed_duct/reference_flow.npz",
    "bentheimer_crop": "../data/controlled_cases/bentheimer_crop/reference_flow.npz",
    "berea_heldout_64": "../data/berea64/reference_flow_x.npz",
    "A_thin_wall": "../data/controlled_cases/thin_wall/reference_flow.npz",
    "B_narrow_throat": "../data/controlled_cases/narrow_throat/reference_flow.npz",
    "C_maze": "../data/controlled_cases/maze/reference_flow.npz",
}
# References the manuscript relies on that are not present in this working tree.
MISSING_REFERENCES = ("public_berea_128", "bentheimer_225")


def functionals(path):
    """The two permeability functionals evaluated on one reference field."""
    data = np.load(path, allow_pickle=True)
    volume = np.asarray(data["volume"], dtype=np.float64)
    total = float(volume.sum())
    flux = float(np.sum(np.asarray(data["phi"])[:, None]
                        * np.asarray(data["dvec"]), axis=0)[0] / total)
    velocity = float(np.sum(volume[:, None] * np.asarray(data["U"]), axis=0)[0] / total)
    return {
        "K_stored": float(data["K_eff_x"]),
        "K_flux": VISCOSITY * flux / FORCE_X,
        "K_velocity": VISCOSITY * velocity / FORCE_X,
        "cells": int(volume.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="reproduce/figure_06/refinement/runs")
    parser.add_argument("--out", default="reproduce/supplementary/s2_discretization")
    args = parser.parse_args()

    offsets, rows = {}, []
    for case, relative in REFERENCES.items():
        path = COMPUTE_ROOT / relative
        if not path.exists():
            print("[skip] %s missing" % relative)
            continue
        f = functionals(path)
        stored_is_velocity = abs(f["K_stored"] - f["K_velocity"]) <= 1e-9 * abs(f["K_stored"])
        offset = 100.0 * abs(f["K_flux"] - f["K_stored"]) / abs(f["K_stored"])
        offsets[case] = f["K_flux"]
        rows.append({
            "protocol_id": PROTOCOL_ID, "kind": "reference", "case": case,
            "cells": f["cells"], "K_stored": f["K_stored"], "K_flux": f["K_flux"],
            "K_velocity": f["K_velocity"],
            "stored_equals_velocity_functional": stored_is_velocity,
            "definition_offset_percent": offset,
        })
        print("[ref ] %-18s stored %.8g  flux %.8g  velocity %.8g  offset %6.3f%%  "
              "stored==velocity: %s"
              % (case, f["K_stored"], f["K_flux"], f["K_velocity"], offset,
                 stored_is_velocity), flush=True)

    for record in sorted(Path(args.runs).glob("*/*/row.json")):
        parts = os.path.normpath(str(record)).split(os.sep)
        case, level = parts[-3], parts[-2]
        if case not in offsets:
            continue
        row = json.loads(record.read_text(encoding="utf-8"))
        method = row.get("K_eff_parallel")
        stored = row.get("K_ref_parallel")
        if method is None or stored is None:
            continue
        as_is = 100.0 * abs(method - stored) / abs(stored)
        fixed = 100.0 * abs(method - offsets[case]) / abs(offsets[case])
        rows.append({
            "protocol_id": PROTOCOL_ID, "kind": "run", "case": case,
            "level_label": level, "N_cv": int(row["N_cv"]),
            "K_method_flux": method, "K_reference_stored": stored,
            "K_reference_flux": offsets[case],
            "e_K_as_reported_percent": as_is,
            "e_K_consistent_percent": fixed,
            "shift_percentage_points": fixed - as_is,
        })

    report = {
        "protocol_id": PROTOCOL_ID,
        "references": {r["case"]: r for r in rows if r["kind"] == "reference"},
        "offset_min_percent": min((r["definition_offset_percent"]
                                   for r in rows if r["kind"] == "reference"), default=None),
        "offset_max_percent": max((r["definition_offset_percent"]
                                   for r in rows if r["kind"] == "reference"), default=None),
        "references_where_stored_is_velocity_functional": sum(
            1 for r in rows if r["kind"] == "reference"
            and r["stored_equals_velocity_functional"]),
        "references_checked": sum(1 for r in rows if r["kind"] == "reference"),
        "runs_rescored": sum(1 for r in rows if r["kind"] == "run"),
        "max_abs_shift_percentage_points": max(
            (abs(r["shift_percentage_points"]) for r in rows if r["kind"] == "run"),
            default=None),
        "references_not_available_locally": list(MISSING_REFERENCES),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in rows for k in r})
    with open(out / "permeability_readouts.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (out / "permeability_readouts.json").write_text(
        json.dumps({"summary": report, "rows": rows}, indent=2, default=float),
        encoding="utf-8")

    print("\n%-16s %-6s %12s %12s %10s" % ("case", "level", "e_K as-is", "e_K face-flux", "shift"))
    for r in rows:
        if r["kind"] != "run":
            continue
        print("%-16s %-6s %11.3f%% %11.3f%% %+9.3f"
              % (r["case"], r["level_label"].split("_")[0],
                 r["e_K_as_reported_percent"], r["e_K_consistent_percent"],
                 r["shift_percentage_points"]))
    print("\n[write] " + str(out / "permeability_readouts.csv"))
    print("references whose stored K equals the velocity functional: %d of %d"
          % (report["references_where_stored_is_velocity_functional"],
             report["references_checked"]))
    print("definition offset spans %.3f%% to %.3f%%"
          % (report["offset_min_percent"], report["offset_max_percent"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
