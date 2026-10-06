"""Self-convergence of the refinement ladder, independent of the reference.

The error against a fixed image-resolved reference can stall for two different
reasons, and they call for different conclusions:

  (a) the discrete solutions stop changing but sit at a fixed distance from the
      reference - the method converges, to something the reference is not; or
  (b) the discrete solutions keep changing - the method is not converging.

The reference cannot distinguish them.  This module can: it paints each level's
cell velocities back onto the voxel grid through that level's own owner label
field, and measures the successive difference between levels on the common voxel
grid.  Nothing is interpolated and no reference is used.

It also records, per case, whether the archived reference reached its own steady
criterion, because that is the leading candidate explanation for a floor and it
must be visible next to the numbers rather than buried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402

PROTOCOL_ID = "pvfv_self_convergence"


def level_arrays(run_dir):
    """Owner labels, cell velocities, coarsened reference and cell volumes."""
    with np.load(run_dir / "sites_and_partition.npz") as data:
        labels = np.asarray(data["labels"]).astype(np.int64)
        volume = np.asarray(data["cell_volume"], dtype=np.float64)
    with np.load(run_dir / "state.npz") as data:
        U = np.asarray(data["U"], dtype=np.float64)
        U_ref = np.asarray(data["U_ref"], dtype=np.float64)
    flat = labels.reshape(-1)
    pore = np.flatnonzero(flat >= 0)
    return {
        "labels_flat": flat, "pore": pore, "owner": flat[pore],
        "U": U, "U_ref": U_ref, "volume": volume,
    }


def coarsen(pore, owner, n_cells, voxel_field):
    """Volume mean of a per-voxel field over the cells of one partition.

    Every voxel has the same volume, so the volume mean is the arithmetic mean;
    this is the same aggregation `coarsen_voxel_reference_to_coarse_gpu` applies
    to the image-resolved reference, so the self-convergence estimator and the
    reference estimator are the same functional of their two targets.
    """
    total = np.zeros((n_cells, voxel_field.shape[1]), dtype=np.float64)
    count = np.bincount(owner, minlength=n_cells).astype(np.float64)
    for component in range(voxel_field.shape[1]):
        np.add.at(total[:, component], owner, voxel_field[:, component])
    return total / np.maximum(count, 1.0)[:, None]


def weighted_relative(a, b, volume):
    numerator = float(np.sqrt(np.sum(volume * np.sum((a - b) ** 2, axis=1))))
    denominator = float(np.sqrt(np.sum(volume * np.sum(b ** 2, axis=1))))
    return 100.0 * numerator / max(denominator, 1e-300)


def reference_metadata(case):
    paths = ec.case_paths(case)
    with np.load(paths["reference"], allow_pickle=True) as data:
        if "metadata" not in data.files:
            return {}
        raw = data["metadata"].item()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {"raw": raw[:400]}
    return raw if isinstance(raw, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/figure_06/refinement")
    parser.add_argument("--out", default="reproduce/supplementary/s3_verification")
    args = parser.parse_args()

    root = Path(args.root) / "runs"
    out = Path(args.out)
    rows = []
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        levels = sorted(
            (d for d in case_dir.iterdir() if (d / "state.npz").exists()),
            key=lambda d: int(d.name.split("_")[0][1:]),
        )
        if len(levels) < 2:
            continue
        meta = reference_metadata(case)
        data = [level_arrays(level) for level in levels]
        finest = data[-1]
        # The finest level's solution painted onto the pore voxels; this is the
        # self-convergence target, and it is aggregated onto each coarser
        # partition by exactly the aggregation used for the reference.
        finest_voxels = finest["U"][finest["owner"]]
        reference_voxels = finest["U_ref"][finest["owner"]]
        for level, arrays in zip(levels, data):
            n_cells = arrays["U"].shape[0]
            if not np.array_equal(arrays["pore"], finest["pore"]):
                raise RuntimeError("levels do not share a pore voxel set")
            target_self = coarsen(arrays["pore"], arrays["owner"], n_cells, finest_voxels)
            target_reference_voxel = coarsen(
                arrays["pore"], arrays["owner"], n_cells, reference_voxels
            )
            volume = arrays["volume"]
            row = {
                "protocol_id": PROTOCOL_ID,
                "case": case,
                "level_label": level.name,
                "N_cv": int(n_cells),
                "pore_voxels": int(arrays["pore"].size),
                "e_u_vs_reference_percent": weighted_relative(
                    arrays["U"], arrays["U_ref"], volume
                ),
                "e_u_vs_finest_level_percent": weighted_relative(
                    arrays["U"], target_self, volume
                ),
                "finest_level_vs_reference_percent": weighted_relative(
                    target_self, target_reference_voxel, volume
                ),
                "reference_steady_converged": meta.get("steady_converged"),
                "reference_steady_momentum_inf": meta.get("steady_momentum_inf"),
                "reference_steps_completed": meta.get("steps_completed"),
            }
            rows.append(row)
            print(
                "[self] %-16s %-10s N_c=%5d  e_u(ref)=%8.4f%%  e_u(finest)=%8.4f%%  "
                "finest-vs-ref=%8.4f%%  ref_steady=%s"
                % (case, level.name, n_cells, row["e_u_vs_reference_percent"],
                   row["e_u_vs_finest_level_percent"],
                   row["finest_level_vs_reference_percent"],
                   row["reference_steady_converged"]),
                flush=True,
            )
    if rows:
        columns = sorted({k for r in rows for k in r})
        lead = ["protocol_id", "case", "level_label", "N_cv", "pore_voxels",
                "e_u_vs_reference_percent", "e_u_vs_finest_level_percent",
                "finest_level_vs_reference_percent",
                "reference_steady_converged", "reference_steady_momentum_inf"]
        columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
        ec.write_csv(out / "self_convergence.csv", rows, columns)
        print("[write] " + str(out / "self_convergence.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
