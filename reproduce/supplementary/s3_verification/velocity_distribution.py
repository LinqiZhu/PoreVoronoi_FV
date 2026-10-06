"""Does the reconstructed field have the right velocity distribution?

Permeability is the wrong thing to score this method on.  In the Stokes regime
the permeability of a representative volume is a property of the geometry: take
an REV, drive it periodically, read the ratio.  Nothing in that calculation needs
a measured trajectory, so an error measure built on it cannot show what the
Lagrangian input buys.

What the method actually delivers is a locally conservative velocity *field*, and
the downstream uses of a pore-scale field --- dispersion, mixing, residence-time
distributions, reactive fronts, particle filtration --- are controlled by the
velocity distribution and especially by its tails, not by the mean and not by an
L2 norm.  Two fields with the same `e_u` can transport very differently.

This module measures the distribution directly from the archived states, which
already store the method's cell velocities and the reference coarsened onto the
same cells.  Three statistics are reported per level:

  W1/mean    the volume-weighted Wasserstein-1 distance between the two
             distributions of the axial cell velocity, divided by the reference
             mean, so it is dimensionless and comparable across cases;
  back-flow  the volume fraction carrying negative axial velocity, in the method
             and in the reference.  Recirculation is a transport-relevant
             feature that an L2 norm barely sees;
  quantiles  absolute volume-weighted quantiles of the axial velocity.

Quantile *ratios* were tried first and discarded: on a real pore space the tenth
percentile of the axial velocity sits at essentially zero, so the ratio is a
quotient of two near-zero numbers and produced values such as -40.8 that carry
no information.  The absolute quantiles and the Wasserstein distance are
well-conditioned and are what is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

PROTOCOL_ID = "pvfv_velocity_distribution"
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
CASE_ORDER = {"orthogonal_duct": 0, "skewed_duct": 1, "bentheimer_crop": 2}


def weighted_quantile(values, weights, probabilities):
    """Volume-weighted quantiles, midpoint convention."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(probabilities, cumulative, values)


def wasserstein1(a, b, weights, samples=999):
    """Volume-weighted W1 between two cell-value distributions on the same cells.

    Computed as the mean absolute difference of the two weighted quantile
    functions, which is the standard identity for the one-dimensional case.
    """
    probabilities = np.linspace(0.5 / samples, 1.0 - 0.5 / samples, samples)
    return float(np.mean(np.abs(
        weighted_quantile(a, weights, probabilities)
        - weighted_quantile(b, weights, probabilities)
    )))


def analyse(state_path, row_path):
    data = np.load(state_path)
    velocity = np.asarray(data["U"])[:, 0]
    reference = np.asarray(data["U_ref"])[:, 0]
    volume = np.asarray(data["volume"])
    total = volume.sum()
    reference_mean = float(np.sum(volume * reference) / total)
    row = json.loads(Path(row_path).read_text(encoding="utf-8"))
    record = {
        "protocol_id": PROTOCOL_ID,
        "N_cv": int(row["N_cv"]),
        "h_S_over_h": row.get("h_S_over_h"),
        "e_u_percent": row.get("e_u_percent"),
        "reference_mean_u_x": reference_mean,
        "method_mean_u_x": float(np.sum(volume * velocity) / total),
        "w1_over_mean": wasserstein1(velocity, reference, volume) / abs(reference_mean),
        "backflow_volume_fraction_percent":
            100.0 * float(volume[velocity < 0.0].sum() / total),
        "backflow_volume_fraction_reference_percent":
            100.0 * float(volume[reference < 0.0].sum() / total),
    }
    for probability in QUANTILES:
        tag = "q%02d" % int(round(100 * probability))
        record[tag + "_method"] = float(weighted_quantile(velocity, volume, probability))
        record[tag + "_reference"] = float(weighted_quantile(reference, volume, probability))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/figure_06/refinement/runs")
    parser.add_argument("--out", default="reproduce/supplementary/s3_verification")
    args = parser.parse_args()

    rows = []
    for state in sorted(Path(args.root).glob("*/*/state.npz")):
        parts = os.path.normpath(str(state)).split(os.sep)
        case, level = parts[-3], parts[-2]
        row_path = state.parent / "row.json"
        if not row_path.exists():
            continue
        record = analyse(state, row_path)
        record["case"] = case
        record["level_label"] = level
        rows.append(record)

    rows.sort(key=lambda r: (CASE_ORDER.get(r["case"], 9),
                             int(re.search(r"M(\d+)", r["level_label"]).group(1))))
    for record in rows:
        print("[dist] %-16s %-10s W1/mean %7.4f   back-flow %5.2f%% vs %5.2f%%   "
              "e_u %6.2f%%"
              % (record["case"], record["level_label"], record["w1_over_mean"],
                 record["backflow_volume_fraction_percent"],
                 record["backflow_volume_fraction_reference_percent"],
                 record["e_u_percent"]), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in rows for k in r})
    import csv
    with open(out / "velocity_distribution.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (out / "velocity_distribution.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("[write] " + str(out / "velocity_distribution.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
