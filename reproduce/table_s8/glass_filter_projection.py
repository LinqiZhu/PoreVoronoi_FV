"""The sampled-velocity branch driven by MEASURED X-ray particle velocimetry.

Every velocity in the manuscript so far is simulated: the reference fields come
from an image-resolved Stokes solver, and the trajectories are advected in those
fields.  The sampled-data branch is motivated by velocimetry but has never been
given a measured record.

This module supplies one, from the released dataset of

  T. Bultreys, S. Van Offenwert, W. Goethals, M. N. Boone, J. Aelterman and
  V. Cnudde, "X-ray Tomographic Micro-Particle Velocimetry in Porous Media",
  2022, Zenodo DOI 10.5281/zenodo.6010490,

which contains a segmented sintered-glass filter and 165 435 tracked velocity
records from 4416 particles over 59 time frames.

Registration, measured rather than assumed
------------------------------------------
The tracking output is expressed in the coordinates of the crop used for the
simulation, `cropD 175 160 60`, not of the released mask header.  Mapping a
tracking point (z, y, x) to the full image by (z+60, y+160, x+175) with no axis
permutation puts **100.00 %** of the records inside the released pore mask,
against a chance level equal to the porosity, 6.48 %.  Every other offset and
permutation tested falls to chance.  That check runs here before anything else
and the driver refuses to continue if it fails.

Three defects in the released package are recorded by the same checks and are
reported rather than worked around silently:

  * the mask's own `.mhd` declares `DimSize 657 657 490`, but the raw file is
    283 161 744 bytes, which is 657 x 657 x **656**;
  * the resliced simulated velocity fields are 499 x 338 x 323, while the
    simulation's own summary reports a 486 x 330 x 322 grid, and no z shift
    aligns their fluid voxels with the pore mask better than 1.4 times chance,
    against 15.4 for the tracking records.  They are therefore not used;
  * the README states the flow is along x, while the records give a mean
    velocity of -0.321 voxel/frame along z against -0.018 along x.

Validation without a reference field
------------------------------------
Because the released simulated field cannot be registered, there is no
image-resolved reference on this crop.  The measured data can still validate the
method, by held-out prediction: the particles are split into two disjoint sets,
the complex and the projection are built from the first, and the second is
predicted at its own recorded positions.  Two scales are reported with it.  The
trivial predictor, the global mean velocity, bounds the result from above.  The
*within-cell spread* of the measured records bounds it from below: a cell-constant
state cannot reproduce the variation between two records that share a cell, so no
method of this form can beat that floor, and a held-out error at the floor is the
best attainable rather than a poor one.

Measuring that floor shows the record-level question to be the wrong one here.
Real pore-scale velocity varies over decades inside a single control volume, and
the floor is 74 to 93 per cent of the record RMS at every support size tried, so
no cell-constant Eulerian state can predict an individual Lagrangian record.  The
Eulerian question is asked instead: on the cells that *both* halves sample, does
the state reconstructed from one half agree with the independent cell mean of the
other?  That target carries no within-cell variability, and its own sampling
uncertainty is quantified by splitting the fit half again.

The prescribed-site count is swept rather than fixed.  Taking every visited voxel
as a site is the worst available choice for this dataset: the particles move
about 0.7 voxel per frame, so consecutive frames land in different voxels and each
cell receives about two records.  Coarser supports aggregate more records per cell
at the cost of resolution, and the sweep exposes that trade-off directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

PROTOCOL_ID = "pvfv_glass_filter_projection"

DATASET = {
    "name": "porous_glass",
    "root": Path("data/glass_filter"),
    "mask_raw": "2_segmentedImage/000_6_HQ_sample2_poreMask.raw",
    "mask_shape": (656, 657, 657),
    "tracking": "4_trackingOutput/sinteredGlass_unsmoothed_velocityPoints.csv",
    # from 3_simulatedVelocityFields/inputFile.mhd: cropD 175 160 60 -> 497 497 546
    "crop_origin_zyx": (60, 160, 175),
    "voxel_size_m": 11.79e-6,
    "frame_interval_s": 35.0,
    "citation": "Bultreys et al. 2022, Zenodo 10.5281/zenodo.6010490",
}


def load_dataset(spec):
    root = Path(spec["root"])
    raw = root / spec["mask_raw"]
    expected = int(np.prod(spec["mask_shape"]))
    actual = raw.stat().st_size
    if actual != expected:
        raise RuntimeError(
            "released mask is %d bytes, which is not %s = %d"
            % (actual, "x".join(str(v) for v in spec["mask_shape"]), expected)
        )
    mask = np.fromfile(raw, dtype=np.uint8).reshape(spec["mask_shape"]) == 255
    records = pd.read_csv(root / spec["tracking"])
    records = records[records["velMags"] > 0].reset_index(drop=True)
    return mask, records


def registration_check(mask, records, origin):
    """Measured, not assumed: tracked points must sit inside the pore mask."""
    points = np.rint(records[["z", "y", "x"]].to_numpy()).astype(np.int64) + np.asarray(origin)
    inside_box = np.all((points >= 0) & (points < np.asarray(mask.shape)), axis=1)
    hit = mask[points[inside_box, 0], points[inside_box, 1], points[inside_box, 2]]
    chance = float(mask.mean())
    fraction = float(hit.mean())
    return {
        "records": int(len(records)),
        "records_inside_image": int(inside_box.sum()),
        "fraction_in_pore": fraction,
        "chance_level_porosity": chance,
        "ratio_over_chance": fraction / max(chance, 1e-300),
        "passes": bool(fraction > 0.95),
    }, points


def build_window(records, points, origin_local, size, mask_crop):
    """Records that fall inside the analysis block, in local voxel coordinates."""
    local = points - np.asarray(origin_local)
    inside = np.all((local >= 0) & (local < size), axis=1)
    block = records[inside].copy()
    coordinates = local[inside]
    keep = mask_crop[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
    block = block[keep].reset_index(drop=True)
    coordinates = coordinates[keep]
    block["z_voxel"] = coordinates[:, 0].astype(float)
    block["y_voxel"] = coordinates[:, 1].astype(float)
    block["x_mod_voxel"] = coordinates[:, 2].astype(float)
    # the tracking file stores (vz, vy, vx); the solver's cell velocity is
    # physical (x, y, z), so the columns are renamed rather than reordered
    block["Ux"] = block["vx"].to_numpy()
    block["Uy"] = block["vy"].to_numpy()
    block["Uz"] = block["vz"].to_numpy()
    block["particle_id"] = block["particle"].astype(int)
    block["frame_id"] = block["frame"].astype(int)
    return block, coordinates


def cell_states(owner_of_record, velocity, n_cells):
    """Arithmetic mean of the measured velocities in each cell, and its mask."""
    total = np.zeros((n_cells, 3), dtype=np.float64)
    count = np.bincount(owner_of_record, minlength=n_cells).astype(np.float64)
    for component in range(3):
        np.add.at(total[:, component], owner_of_record, velocity[:, component])
    measured = count > 0
    state = np.zeros((n_cells, 3), dtype=np.float64)
    state[measured] = total[measured] / count[measured, None]
    return state, measured, count


def relative_error(prediction, truth):
    numerator = float(np.sqrt(np.sum((prediction - truth) ** 2)))
    denominator = float(np.sqrt(np.sum(truth ** 2)))
    return 100.0 * numerator / max(denominator, 1e-300)


def within_cell_floor(owner_of_record, velocity, n_cells):
    """The error a cell-constant state cannot avoid.

    For every cell holding at least two records, the deviation of each record
    from the mean of the *other* records in its cell is the error an ideal
    cell-constant predictor makes on it.  The leave-one-out form is used so the
    floor is not optimistically biased by including the record in its own target.
    """
    total = np.zeros((n_cells, 3), dtype=np.float64)
    count = np.bincount(owner_of_record, minlength=n_cells).astype(np.float64)
    for component in range(3):
        np.add.at(total[:, component], owner_of_record, velocity[:, component])
    usable = count[owner_of_record] >= 2
    if not usable.any():
        return float("nan"), 0
    others = (total[owner_of_record[usable]] - velocity[usable]) / (
        count[owner_of_record[usable]][:, None] - 1.0
    )
    return relative_error(others, velocity[usable]), int(usable.sum())


def choose_sites(visited_flat, mask_crop, target, periodic_x=False):
    """A nested prescribed support of about `target` sites drawn from the visited voxels.

    Deterministic graph farthest-point sampling on the pore graph, restricted to
    the voxels the particles actually visited, so the support stays inside the
    measured region.  With `target` at or above the number of visited voxels the
    whole set is returned, which is the every-visited-voxel choice.
    """
    visited = np.unique(np.asarray(visited_flat, dtype=np.int64))
    if int(target) >= visited.size:
        return visited, "all visited voxels"
    order = ec.graph_farthest_point_order(
        mask_crop, visited, periodic_x=periodic_x, count=int(target)
    )
    return np.unique(order[: int(target)]), "graph farthest-point over the visited voxels"



def retain_observed_components(crop, coordinates, in_fit, periodic_x=False):
    """Keep only the six-connected pore components holding a fit observation.

    The harmonic completion of an unobserved component is not unique, and the
    production code refuses it rather than filling it silently.  On this measured
    dataset that refusal fires: 294 tracked particles in a 96-voxel block do not
    reach every pore component.  The retention rule is therefore declared here --
    a component survives if at least one *fit* record lies in it -- and the volume
    it discards is reported rather than absorbed.
    """
    from scipy import sparse
    from scipy.sparse import csgraph

    pore, neighbours = ec.pore_neighbour_table(crop, periodic_x=periodic_x)
    lookup = np.full(crop.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(pore.size, dtype=np.int64)
    rows, cols = [], []
    for column in range(neighbours.shape[1]):
        other = neighbours[:, column]
        ok = other >= 0
        rows.append(np.flatnonzero(ok))
        cols.append(other[ok])
    graph = sparse.coo_matrix(
        (np.ones(sum(r.size for r in rows), dtype=np.int8),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(pore.size, pore.size),
    ).tocsr()
    n_comp, component = csgraph.connected_components(graph, directed=False)

    size = crop.shape[0]
    flat = (coordinates[:, 0] * size + coordinates[:, 1]) * size + coordinates[:, 2]
    node = lookup[flat]
    observed = np.unique(component[node[in_fit]])
    keep_node = np.isin(component, observed)
    retained = np.zeros(crop.shape, dtype=bool)
    retained.reshape(-1)[pore[keep_node]] = True
    record_kept = np.isin(component[node], observed)
    info = {
        "pore_components": int(n_comp),
        "components_with_a_fit_observation": int(observed.size),
        "pore_voxels_before": int(crop.sum()),
        "pore_voxels_retained": int(retained.sum()),
        "pore_volume_discarded_fraction": 1.0 - float(retained.sum()) / float(crop.sum()),
        "records_before": int(record_kept.size),
        "records_retained": int(record_kept.sum()),
        "retention_rule": "six-connected pore components containing at least one fit record",
    }
    return retained, record_kept, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--origin", default="252,328,319",
                        help="analysis block origin in FULL-image (z,y,x) voxels")
    parser.add_argument("--site-counts", default="100,200,400,800,1600,0",
                        help="prescribed-site counts to sweep; 0 means every visited voxel")
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--out", default="reproduce/table_s8")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = DATASET
    size = int(args.size)
    origin = tuple(int(v) for v in args.origin.split(","))

    print("[xptv] loading %s" % spec["citation"], flush=True)
    mask, records = load_dataset(spec)
    check, points = registration_check(mask, records, spec["crop_origin_zyx"])
    print("[xptv] registration: %.4f of %d records in pore, chance %.4f, ratio %.2f -> %s"
          % (check["fraction_in_pore"], check["records"], check["chance_level_porosity"],
             check["ratio_over_chance"], "PASS" if check["passes"] else "FAIL"), flush=True)
    if not check["passes"]:
        raise RuntimeError("the tracking records do not register to the released mask")

    crop = mask[origin[0]:origin[0] + size,
                origin[1]:origin[1] + size,
                origin[2]:origin[2] + size]
    if crop.shape != (size, size, size):
        raise RuntimeError("analysis block runs past the image: %s" % (crop.shape,))
    window, coordinates = build_window(records, points, origin, size, crop)
    visited_flat = (coordinates[:, 0] * size + coordinates[:, 1]) * size + coordinates[:, 2]
    print("[xptv] block %d^3 at %s: %d pore voxels (porosity %.3f), %d measured records, "
          "%d particles, %d visited voxels"
          % (size, origin, int(crop.sum()), crop.mean(), len(window),
             window["particle_id"].nunique(), int(np.unique(visited_flat).size)), flush=True)

    boot = ec.bootstrap(out / "_boot")
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    cfg.periodic_x = False

    import hybrid_site_sources as hss
    from hybrid_voronoi_trace import (
        build_hybrid_trace_projection_factorization,
        build_hybrid_trace_projection_operators,
        project_constant_cell_states_to_conservative_trace,
    )

    velocity = window[["Ux", "Uy", "Uz"]].to_numpy()
    particles = window["particle_id"].to_numpy()
    unique_particles = np.unique(particles)  # noqa: F841 - split is over particles
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_particles)
    cut = int(round(len(shuffled) * (1.0 - args.holdout_fraction)))
    fit_particles = set(shuffled[:cut].tolist())
    in_fit = np.array([p in fit_particles for p in particles])
    print("[xptv] hold-out split: %d fit particles / %d test particles, %d / %d records"
          % (len(fit_particles), len(shuffled) - len(fit_particles),
             int(in_fit.sum()), int((~in_fit).sum())), flush=True)

    crop, record_kept, components = retain_observed_components(crop, coordinates, in_fit)
    window = window[record_kept].reset_index(drop=True)
    coordinates = coordinates[record_kept]
    velocity = velocity[record_kept]
    particles = particles[record_kept]
    in_fit = in_fit[record_kept]
    visited_flat = (coordinates[:, 0] * size + coordinates[:, 1]) * size + coordinates[:, 2]
    print("[xptv] retained %d of %d pore components (%.2f%% of the pore volume discarded), "
          "%d of %d records kept"
          % (components["components_with_a_fit_observation"], components["pore_components"],
             100.0 * components["pore_volume_discarded_fraction"],
             components["records_retained"], components["records_before"]), flush=True)

    mask_path = out / ("xptv_%s_block%d_%d_%d_%d.npz" % ((spec["name"], size) + origin))
    np.savez_compressed(mask_path, mask=crop)
    window.to_csv(out / "xptv_measured_window.csv", index=False)

    rows = []
    for target in [int(v) for v in args.site_counts.split(",")]:
        sites, rule = choose_sites(visited_flat, crop, target if target > 0 else 1 << 30)
        geom, _meta, build_time = ec.build_geometry_from_seed_flat(
            sites, mask_path=mask_path,
            seed_spec="xptv:%s:block%d:n%d" % (spec["name"], size, sites.size),
        )
        trace = boot["build_hybrid_trace_geometry"](
            ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
        )
        n_cells = int(geom.n_cells)
        labels = cp.asnumpy(geom.labels).astype(np.int64)
        owner_of_record = labels[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
        if np.any(owner_of_record < 0):
            raise RuntimeError("a measured record fell outside the ownership partition")
        volume = cp.asnumpy(geom.volume).astype(np.float64)

        face_conductance = (trace.voxel_size ** 2) / np.maximum(
            trace.face_r_owner_g + trace.face_r_neigh_g, 1e-300
        )
        edge_conductance = np.bincount(
            trace.face_parent_edge, weights=face_conductance,
            minlength=int(geom.owner.size),
        )
        operators = build_hybrid_trace_projection_operators(trace)
        factorization = build_hybrid_trace_projection_factorization(operators)

        floor_all, floor_records = within_cell_floor(owner_of_record, velocity, n_cells)

        state, measured, count = cell_states(
            owner_of_record[in_fit], velocity[in_fit], n_cells
        )
        completed, completion = hss.harmonic_complete_constant_cell_states(
            cp.asnumpy(geom.owner), cp.asnumpy(geom.neigh), edge_conductance, measured, state
        )
        started = time.perf_counter()
        result = project_constant_cell_states_to_conservative_trace(
            operators, completed, factorization=factorization
        )
        elapsed = float(time.perf_counter() - started)

        pre = np.asarray(result["U"], dtype=np.float64)
        post = np.asarray(result["U_trace_moment"], dtype=np.float64)
        z = np.asarray(result["trace_coefficients"]).reshape(-1)
        D = operators.divergence_matrix
        Dz = np.asarray(D @ z).reshape(-1)
        abs_Dz = np.asarray(abs(D) @ np.abs(z)).reshape(-1)
        eps = np.finfo(np.float64).eps
        denominator = float(np.max(abs_Dz))

        test = ~in_fit
        truth = velocity[test]
        cells = owner_of_record[test]
        trivial = np.repeat(velocity[in_fit].mean(axis=0)[None, :], truth.shape[0], axis=0)

        # Eulerian observable: the independent cell means of the held-out half,
        # on the cells both halves sample.  This target has no within-cell
        # variability in it, so it is the question a cell-constant state can
        # actually answer.
        test_state, test_measured, test_count = cell_states(cells, truth, n_cells)
        both = measured & test_measured
        weight = np.minimum(count, test_count)[both][:, None]
        global_mean = velocity[in_fit].mean(axis=0)[None, :]

        def weighted_error(prediction):
            difference = (prediction[both] - test_state[both]) * np.sqrt(weight)
            target = test_state[both] * np.sqrt(weight)
            return 100.0 * float(np.linalg.norm(difference)) / max(
                float(np.linalg.norm(target)), 1e-300)

        # sampling uncertainty of the target itself: split the fit half again and
        # compare its two independent cell means on the same cells
        halves = np.zeros(len(particles), dtype=bool)
        fit_ids = np.unique(particles[in_fit])
        halves[np.isin(particles, rng.permutation(fit_ids)[: len(fit_ids) // 2])] = True
        a_state, a_measured, a_count = cell_states(
            owner_of_record[in_fit & halves], velocity[in_fit & halves], n_cells)
        b_state, b_measured, b_count = cell_states(
            owner_of_record[in_fit & ~halves], velocity[in_fit & ~halves], n_cells)
        shared = a_measured & b_measured
        target_noise = (
            100.0 * float(np.linalg.norm(a_state[shared] - b_state[shared]))
            / max(float(np.linalg.norm(b_state[shared])), 1e-300)
            if shared.any() else float("nan")
        )
        row = {
            "protocol_id": PROTOCOL_ID, "dataset": spec["name"],
            "citation": spec["citation"], "site_rule": rule,
            "block_size": size, "block_origin_zyx": str(list(origin)),
            "N_f": int(crop.sum()), "N_sites": int(sites.size), "N_c": n_cells,
            "N_facelets": int(trace.n_facelets), "N_trace_modes": int(trace.n_trace_modes),
            "records": int(len(window)),
            "pore_components": components["pore_components"],
            "components_retained": components["components_with_a_fit_observation"],
            "pore_volume_discarded_fraction": components["pore_volume_discarded_fraction"],
            "records_per_cell_mean": float(len(window)) / n_cells,
            "fit_records": int(in_fit.sum()), "test_records": int(test.sum()),
            "cells_measured_in_fit": int(measured.sum()),
            "coverage_fraction": float(measured.mean()),
            "harmonic_completed_cells": int(completion["state_harmonic_completed_cells"]),
            "within_cell_floor_percent": floor_all,
            "within_cell_floor_records": floor_records,
            "heldout_error_trivial_mean_percent": relative_error(trivial, truth),
            "heldout_error_pre_projection_percent": relative_error(pre[cells], truth),
            "heldout_error_post_projection_percent": relative_error(post[cells], truth),
            "eulerian_cells_sampled_by_both_halves": int(both.sum()),
            "eulerian_error_trivial_mean_percent": weighted_error(
                np.repeat(global_mean, n_cells, axis=0)),
            "eulerian_error_pre_projection_percent": weighted_error(pre),
            "eulerian_error_post_projection_percent": weighted_error(post),
            "eulerian_target_sampling_uncertainty_percent": target_noise,
            "mass_inf_per_volume": float(np.max(np.abs(Dz) / np.maximum(volume, 1e-300))),
            "mass_backward_error_dimensionless": float(
                np.max(np.abs(Dz)) / (denominator + eps * max(1.0, denominator))
            ),
            "projection_correction_relative": float(
                np.linalg.norm(post - completed) / max(np.linalg.norm(completed), 1e-300)
            ),
            "projection_seconds": elapsed,
            "t_geometry_build_s": build_time,
            "voxel_size_m": spec["voxel_size_m"],
            "frame_interval_s": spec["frame_interval_s"],
        }
        rows.append(row)
        print(
            "[xptv] N_c=%5d rec/cell %5.2f cov %.3f | record-level: floor %6.2f trivial %6.2f "
            "pre %6.2f post %6.2f | Eulerian on %4d shared cells: noise %6.2f trivial %6.2f "
            "pre %6.2f post %6.2f | eta_m %.1e"
            % (n_cells, row["records_per_cell_mean"], row["coverage_fraction"],
               row["within_cell_floor_percent"], row["heldout_error_trivial_mean_percent"],
               row["heldout_error_pre_projection_percent"],
               row["heldout_error_post_projection_percent"],
               row["eulerian_cells_sampled_by_both_halves"],
               row["eulerian_target_sampling_uncertainty_percent"],
               row["eulerian_error_trivial_mean_percent"],
               row["eulerian_error_pre_projection_percent"],
               row["eulerian_error_post_projection_percent"],
               row["mass_backward_error_dimensionless"]),
            flush=True,
        )
        del geom, trace, operators, factorization
        ec.free_gpu()

    ec.write_json(out / "xptv_registration_check.json", check)
    ec.write_json(out / "xptv_component_retention.json", components)
    ec.write_csv(out / "xptv_real_data.csv", rows, sorted({k for r in rows for k in r}))
    ec.write_json(out / "xptv_real_data.json", rows)
    print("[write] " + str(out / "xptv_real_data.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
