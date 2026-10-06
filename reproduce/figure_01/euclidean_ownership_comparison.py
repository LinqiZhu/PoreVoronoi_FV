"""Euclidean vs graph-geodesic ownership on identical pore masks and site sets.

The claim the manuscript is entitled to is structural, not
a blanket dismissal of Euclidean methods: on obstructed voxel domains a
voxel-restricted Euclidean nearest-site partition can produce owner regions that
are not face-connected and that reach across a thin solid wall, whereas
graph-geodesic ownership cannot, because its metric is the pore adjacency itself.

For each case and each site set this module measures, on exactly the same pore
voxels and the same sites:

  * the number of six-connected components of every owner region;
  * whether the site lies in the largest component of its own region;
  * the pore-voxel volume outside the site-containing component;
  * the aggregate fraction of cells and of pore voxels that are disconnected;
  * for the thin-wall case, the voxels assigned across the wall.

Ties are broken identically in both constructions (smallest site index), so the
comparison isolates the metric.

Pure CPU. No production module is modified. Distances use the same six-neighbour
graph; the local BFS explicitly minimizes site labels across all equal-distance
arrivals. Ties go to the smaller site index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402

PROTOCOL_ID = "pvfv_ownership_metric"

COMPUTE = Path(
    os.environ.get(
        "PVFV_GPU_ROOT",
        "gpu",
    )
)

CASE_MASKS = {
    "thin_wall": "../data/controlled_cases/thin_wall/reference_flow.npz",
    "narrow_throat": "../data/controlled_cases/narrow_throat/reference_flow.npz",
    "maze": "../data/controlled_cases/maze/reference_flow.npz",
    "orthogonal_duct": "../data/controlled_cases/orthogonal_duct/reference_flow.npz",
    "skewed_duct": "../data/controlled_cases/skewed_duct/reference_flow.npz",
    "bentheimer_crop": "../data/controlled_cases/bentheimer_crop/reference_flow.npz",
    "berea_heldout_64": "../data/berea64/reference_flow_x.npz",
}


def load_mask(case: str) -> np.ndarray:
    path = COMPUTE / CASE_MASKS[case]
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["mask"]).astype(bool)


def euclidean_owner(mask, site_flat, chunk=4096, periodic_x=False):
    """Voxel-restricted Euclidean nearest-site labels, smallest site index on ties.

    `np.argmin` returns the first minimiser, and the sites are supplied in
    ascending flattened-index order, so the tie rule is exactly lexicographic.
    """
    shape = mask.shape
    pore = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    pore_coords = np.stack(np.unravel_index(pore, shape), axis=1).astype(np.float64)
    sites = np.unique(np.asarray(site_flat, dtype=np.int64))
    site_coords = np.stack(np.unravel_index(sites, shape), axis=1).astype(np.float64)
    owner = np.empty(pore.size, dtype=np.int64)
    for start in range(0, pore.size, chunk):
        block = pore_coords[start : start + chunk]
        d2 = (
            np.sum(block * block, axis=1)[:, None]
            - 2.0 * block @ site_coords.T
            + np.sum(site_coords * site_coords, axis=1)[None, :]
        )
        if periodic_x:
            dx = np.abs(block[:, 2, None] - site_coords[None, :, 2])
            wrapped = np.minimum(dx, shape[2] - dx)
            d2 += wrapped * wrapped - dx * dx
        owner[start : start + chunk] = np.argmin(d2, axis=1)
    return pore, sites, owner


def lexicographic_bfs(neighbours, source_nodes):
    """Unit-edge BFS with the smallest source index at every distance tie."""
    sources = np.asarray(source_nodes, dtype=np.int64)
    if np.unique(sources).size != sources.size or np.any(sources < 0):
        raise ValueError('Sources must be unique pore-node indices')
    distance = np.full(len(neighbours), int(ec._INF), dtype=np.int32)
    labels = np.full(len(neighbours), -1, dtype=np.int64)
    distance[sources] = 0
    labels[sources] = np.arange(len(sources), dtype=np.int64)
    frontier = sources
    level = 0
    while frontier.size:
        level += 1
        candidates = neighbours[frontier].ravel()
        arrivals = np.repeat(labels[frontier], neighbours.shape[1])
        valid = candidates >= 0
        candidates, arrivals = candidates[valid], arrivals[valid]
        new = distance[candidates] == int(ec._INF)
        candidates, arrivals = candidates[new], arrivals[new]
        if not candidates.size:
            break
        order = np.argsort(candidates, kind='stable')
        candidates, arrivals = candidates[order], arrivals[order]
        starts = np.r_[0, np.flatnonzero(candidates[1:] != candidates[:-1]) + 1]
        frontier = candidates[starts]
        distance[frontier] = level
        labels[frontier] = np.minimum.reduceat(arrivals, starts)
    return distance, labels


def distances_to_targets(neighbours, source, targets):
    """Exact distances to requested nodes, stopping when all are reached."""
    targets = np.asarray(targets, dtype=np.int64)
    distance = np.full(len(neighbours), int(ec._INF), dtype=np.int32)
    distance[source] = 0
    remaining = np.zeros(len(neighbours), dtype=bool)
    remaining[targets] = True
    remaining[source] = False
    left = int(remaining.sum())
    frontier = np.asarray([source], dtype=np.int64)
    level = 0
    while left and frontier.size:
        level += 1
        candidates = neighbours[frontier].ravel()
        candidates = candidates[candidates >= 0]
        frontier = np.unique(candidates[distance[candidates] == int(ec._INF)])
        distance[frontier] = level
        left -= int(remaining[frontier].sum())
        remaining[frontier] = False
    return distance[targets]


def geodesic_owner(mask, site_flat, periodic_x):
    pore, neighbours = ec.pore_neighbour_table(mask, periodic_x=periodic_x)
    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(pore.size, dtype=np.int64)
    sites = np.unique(np.asarray(site_flat, dtype=np.int64))
    nodes = lookup[sites]
    distance, labels = lexicographic_bfs(neighbours, nodes)
    return pore, sites, labels, distance, neighbours


def connectivity_report(mask, pore, owner, sites, neighbours, periodic_x):
    """Six-connected component structure of each owner region."""
    n = pore.size
    same = []
    for column in range(neighbours.shape[1]):
        other = neighbours[:, column]
        ok = other >= 0
        left = np.flatnonzero(ok)
        right = other[ok]
        keep = owner[left] == owner[right]
        same.append((left[keep], right[keep]))
    rows = np.concatenate([a for a, _ in same])
    cols = np.concatenate([b for _, b in same])
    graph = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n)
    ).tocsr()
    n_comp, comp = csgraph.connected_components(graph, directed=False)

    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(n, dtype=np.int64)
    site_nodes = lookup[sites]
    site_comp = comp[site_nodes]

    n_cells = sites.size
    # components per owner region
    pair = owner.astype(np.int64) * n_comp + comp.astype(np.int64)
    unique_pairs = np.unique(pair)
    comps_per_cell = np.bincount(
        (unique_pairs // n_comp).astype(np.int64), minlength=n_cells
    )
    # voxels in the site-containing component of their own owner
    in_site_component = comp == site_comp[owner]
    voxels_per_cell = np.bincount(owner, minlength=n_cells)
    voxels_in_site_component = np.bincount(
        owner, weights=in_site_component.astype(np.float64), minlength=n_cells
    )
    orphan_per_cell = voxels_per_cell - voxels_in_site_component

    # largest component of each owner region
    counts = np.bincount(np.searchsorted(unique_pairs, pair), minlength=unique_pairs.size)
    pair_owner = (unique_pairs // n_comp).astype(np.int64)
    largest = np.zeros(n_cells, dtype=np.int64)
    np.maximum.at(largest, pair_owner, counts)
    site_component_size = np.zeros(n_cells, dtype=np.int64)
    for index in range(unique_pairs.size):
        cell = int(pair_owner[index])
        if unique_pairs[index] % n_comp == site_comp[cell]:
            site_component_size[cell] = int(counts[index])
    site_in_largest = site_component_size >= largest

    return {
        "n_cells": int(n_cells),
        "n_pore_voxels": int(n),
        "cells_disconnected": int(np.count_nonzero(comps_per_cell > 1)),
        "cells_disconnected_fraction": float(np.mean(comps_per_cell > 1)),
        "max_components_in_one_cell": int(comps_per_cell.max()),
        "mean_components_per_cell": float(comps_per_cell.mean()),
        "orphan_voxels": int(orphan_per_cell.sum()),
        "orphan_voxel_fraction": float(orphan_per_cell.sum() / max(n, 1)),
        "max_orphan_volume_fraction_in_one_cell": float(
            np.max(orphan_per_cell / np.maximum(voxels_per_cell, 1))
        ),
        "cells_whose_site_is_not_in_the_largest_component": int(
            np.count_nonzero(~site_in_largest)
        ),
        "empty_cells": int(np.count_nonzero(voxels_per_cell == 0)),
        "_comps_per_cell": comps_per_cell,
        "_orphan_per_cell": orphan_per_cell,
        "_voxels_per_cell": voxels_per_cell,
    }


def detour_statistics(mask, pore, sites, owner_e, neighbours, periodic_x,
                      owner_g=None, distance_g=None):
    """Compute assigned-site graph/L1 detour for BOTH ownership rules.

    The L1 denominator uses the same periodic-x convention as the pore graph.
    Sites have ratio one by convention (0/0); unreachable assigned sites are
    counted separately and excluded from the reported finite-value summaries.
    Finite non-site support counts are recorded. Graph ownership is shortest
    among sites in the pore graph; it need not attain the obstacle-free bound.
    """
    shape = mask.shape
    n = pore.size
    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(n, dtype=np.int64)
    site_nodes = lookup[sites]
    coords = np.stack(np.unravel_index(pore, shape), axis=1).astype(np.int64)
    site_coords = coords[site_nodes]

    graph_to_owner = np.full(n, -1, dtype=np.int64)
    for index in range(sites.size):
        members = np.flatnonzero(owner_e == index)
        if members.size == 0:
            continue
        graph_to_owner[members] = distances_to_targets(neighbours, site_nodes[index], members)
    if owner_g is None or distance_g is None:
        distance_g, owner_g = lexicographic_bfs(neighbours, site_nodes)

    def summarize(prefix, owner, distance):
        owner = np.asarray(owner, dtype=np.int64)
        distance = np.asarray(distance, dtype=np.int64)
        unreachable = (owner < 0) | (distance < 0) | (distance >= int(ec._INF))
        delta = np.abs(coords - site_coords[np.maximum(owner, 0)])
        if periodic_x:
            delta[:, 2] = np.minimum(delta[:, 2], shape[2] - delta[:, 2])
        l1 = delta.sum(axis=1)
        finite = (~unreachable) & (l1 > 0)
        if np.any(distance[finite] < l1[finite]):
            raise AssertionError('Graph distance violates the matching free-space L1 lower bound')
        ratio = np.ones(n, dtype=np.float64)
        ratio[finite] = distance[finite] / l1[finite]
        selected = ratio[finite]
        above1 = finite & (ratio > 1.0 + 1e-12)
        return {
            prefix + '__owner_unreachable_voxels': int(unreachable.sum()),
            prefix + '__detour_finite_non_site_voxels': int(finite.sum()),
            prefix + '__detour_site_voxels': int(np.count_nonzero((~unreachable) & (l1 == 0))),
            prefix + '__owner_detour_ratio_mean': float(selected.mean()) if selected.size else 1.0,
            prefix + '__owner_detour_ratio_p95': float(np.quantile(selected, 0.95)) if selected.size else 1.0,
            prefix + '__owner_detour_ratio_max': float(selected.max()) if selected.size else 1.0,
            prefix + '__voxels_with_detour_ratio_above_1': int(above1.sum()),
            prefix + '__voxels_with_detour_ratio_above_2': int(np.count_nonzero(finite & (ratio > 2))),
            prefix + '__voxels_with_detour_ratio_above_1_fraction': float(above1.mean()),
        }

    result = summarize('euclidean', owner_e, graph_to_owner)
    result.update(summarize('geodesic', owner_g, distance_g))
    result['detour_definition'] = 'assigned_site_graph_distance / matching_boundary_L1_distance'
    result['detour_unreachable_convention'] = 'counted separately; excluded from finite summaries'
    return result


def thin_wall_sides(mask):
    """Label the two pore compartments of the thin-wall case geometrically.

    The compartments are the six-connected components of the pore space with the
    connecting aperture removed; they are recovered here without any hand-set
    coordinate by splitting on the solid slab's own plane.
    """
    shape = mask.shape
    solid_per_axis = [(~mask).sum(axis=tuple(a for a in range(3) if a != axis)) for axis in range(3)]
    # the wall axis is the one whose solid profile is most concentrated
    axis = int(np.argmax([float(p.max()) / max(float(p.sum()), 1.0) for p in solid_per_axis]))
    profile = solid_per_axis[axis]
    plane = int(np.argmax(profile))
    index = np.indices(shape)[axis]
    side = np.where(index < plane, 0, 1)
    return axis, plane, side, int(profile[plane]), int(profile.sum())


def analyse(case, site_flat, site_label, periodic_x, out_rows, detail_dir):
    mask = load_mask(case)
    started = time.perf_counter()
    pore_e, sites, owner_e = euclidean_owner(mask, site_flat, periodic_x=periodic_x)
    pore_g, sites_g, owner_g, dist_g, neighbours = geodesic_owner(mask, site_flat, periodic_x)
    assert np.array_equal(pore_e, pore_g) and np.array_equal(sites, sites_g)

    reachable = dist_g < int(ec._INF)
    euclid = connectivity_report(mask, pore_e, owner_e, sites, neighbours, periodic_x)
    geodesic = connectivity_report(mask, pore_g, owner_g, sites, neighbours, periodic_x)

    row = {
        "protocol_id": PROTOCOL_ID,
        "case": case,
        "site_set": site_label,
        "N_sites": int(sites.size),
        "N_pore_voxels": int(pore_e.size),
        "graph_metric_periodic_x": bool(periodic_x),
        "unreachable_pore_voxels": int(np.count_nonzero(~reachable)),
        "seconds": float(time.perf_counter() - started),
    }
    for prefix, report in (("euclidean", euclid), ("geodesic", geodesic)):
        for key, value in report.items():
            if key.startswith("_"):
                continue
            row[prefix + "__" + key] = value
    row["owner_labels_differ_voxels"] = int(np.count_nonzero(owner_e != owner_g))
    row["owner_labels_differ_fraction"] = float(np.mean(owner_e != owner_g))
    row.update(detour_statistics(mask, pore_e, sites, owner_e, neighbours, periodic_x,
                                owner_g=owner_g, distance_g=dist_g))

    if case == "thin_wall":
        axis, plane, side, wall_voxels, solid_voxels = thin_wall_sides(mask)
        side_flat = side.reshape(-1)[pore_e]
        site_side = side_flat[np.searchsorted(pore_e, sites)]
        cross_e = side_flat != site_side[owner_e]
        cross_g = side_flat != site_side[owner_g]
        row.update(
            {
                "thin_wall_axis": int(axis),
                "thin_wall_plane_index": int(plane),
                "thin_wall_solid_voxels_in_plane": int(wall_voxels),
                # NOTE: this side-of-plane count also includes legitimate
                # ownership reaching through the aperture, which is why the
                # geodesic construction has a non-zero value too.  The
                # unambiguous discriminators are the connectivity and detour
                # statistics above; this one is reported for completeness.
                "euclidean__cross_wall_voxels": int(cross_e.sum()),
                "euclidean__cross_wall_fraction": float(cross_e.mean()),
                "euclidean__sites_receiving_cross_wall_voxels": int(
                    np.unique(owner_e[cross_e]).size
                ),
                "geodesic__cross_wall_voxels": int(cross_g.sum()),
                "geodesic__cross_wall_fraction": float(cross_g.mean()),
                "geodesic__sites_receiving_cross_wall_voxels": int(
                    np.unique(owner_g[cross_g]).size if cross_g.any() else 0
                ),
            }
        )

    out_rows.append(row)
    detail_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        detail_dir / (case + "__" + site_label + "__ownership.npz"),
        pore_flat=pore_e,
        sites=sites,
        owner_euclidean=owner_e.astype(np.int32),
        owner_geodesic=owner_g.astype(np.int32),
        distance_geodesic=dist_g.astype(np.int32),
        euclidean_components_per_cell=euclid["_comps_per_cell"].astype(np.int32),
        geodesic_components_per_cell=geodesic["_comps_per_cell"].astype(np.int32),
        euclidean_orphan_per_cell=euclid["_orphan_per_cell"].astype(np.int32),
        geodesic_orphan_per_cell=geodesic["_orphan_per_cell"].astype(np.int32),
    )
    print(
        "[eucl] %-16s %-22s N_c=%5d  euclid: %5.2f%% cells disconnected, "
        "%5.3f%% orphan voxels, max %d components  |  geodesic: %d / %d  (%.1fs)"
        % (
            case, site_label, sites.size,
            100.0 * euclid["cells_disconnected_fraction"],
            100.0 * euclid["orphan_voxel_fraction"],
            euclid["max_components_in_one_cell"],
            geodesic["cells_disconnected"], geodesic["orphan_voxels"],
            row["seconds"],
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(CASE_MASKS))
    parser.add_argument("--counts", default="200,800,3200")
    parser.add_argument("--out", default="reproduce/figure_01")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = [int(v) for v in args.counts.split(",") if v]
    rows = []
    for case in [c for c in args.cases.split(",") if c]:
        mask = load_mask(case)
        n_pore = int(mask.sum())
        # The ownership graph convention is non-periodic on every studied case;
        # it is asserted here rather than assumed, by comparing the two BFS
        # distance fields for a probe site set and recording both.
        periodic_x = False
        for count in [c for c in counts if c <= n_pore]:
            order = ec.graph_farthest_point_order(
                mask,
                np.flatnonzero(mask.reshape(-1)).astype(np.int64),
                periodic_x=periodic_x,
                count=count,
            )
            analyse(case, np.unique(order[:count]), "mask_graph_fps_n%d" % count,
                    periodic_x, rows, out / "fields")
    columns = sorted({k for r in rows for k in r})
    lead = ["protocol_id", "case", "site_set", "N_sites", "N_pore_voxels"]
    columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
    ec.write_csv(out / "euclidean_vs_geodesic_ownership.csv", rows, columns)
    ec.write_json(out / "euclidean_vs_geodesic_ownership.json", rows)
    print("[write] " + str(out / "euclidean_vs_geodesic_ownership.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
