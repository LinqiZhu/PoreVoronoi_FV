"""Result-blind, geometry-only construction and selection of baselines (study B).

Nothing in this module opens, computes, or reads a reference field, a reference
error, or a permeability.  It sees only the pore mask, the frozen site family,
graph topology, admissibility, and the total mixed-system dimension
``N_system``.  The selected configurations are written to a locked selection
record whose SHA-256 is recorded before any accuracy run.

Baselines built here:

* ``mask_graph_fps`` - deterministic graph-farthest-point sites over the pore
  mask only, then the *same* graph-geodesic ownership and the same connected-P1
  shared-trace solver.
* ``connected_block_agglomeration`` - axis-aligned voxel blocks clipped to the
  pore mask, split into six-connected components by the production reindex,
  then the same connected-P1 shared-trace solver.
* ``euclidean_partition_with_deterministic_connectivity_repair`` (optional) - the same frozen trajectory
  sites, Euclidean nearest-site assignment, deterministic six-connected repair.

Usage:
    python build_site_rule_partitions.py [--case CASE] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import study_common as ec

BASELINE_CASES = ("skewed_duct", "bentheimer_crop")
BUDGETS = ("half", "full")
MATCH_PREFERRED = 0.05
MATCH_MAXIMUM = 0.10

BLOCK_SIDES = (2, 3, 4, 5, 6, 7, 8, 10, 12)
BLOCK_OFFSET = (0, 0, 0)
BLOCK_RULE = (
    "axis-aligned voxel blocks clipped to the pore mask and split into six-connected "
    "components by the production reindex, searched over two nested geometry-only families: "
    "(i) fixed sides (bz, by, bx) over " + str(BLOCK_SIDES) + " at offset (0, 0, 0); and "
    "(ii) a graded family parameterised by a requested block count m, where each axis of "
    "length L is split into n = clip(round(L * (m / (Lz*Ly*Lx))**(1/3)), 1, L) nearly equal "
    "contiguous intervals by index(c) = (c * n) // L.  Family (ii) is near-cubic by "
    "construction; it coincides with family (i) whenever the fixed sides are the isotropic "
    "near-cubic choice, and it fills the gaps between the members of family (i) that the "
    "integer side lengths cannot reach.  Both families are searched and the union is "
    "ranked together; m is found by bisection on N_system.  The selected "
    "candidate minimises |N_system - target|, with ties broken by the smaller block volume, "
    "then the family name, then the axis divisions, then the fixed sides."
)

MASK_FPS_RULE = (
    "deterministic graph-farthest-point sampling over every pore voxel (no trajectory, no "
    "flow or reference field); initial candidate = smallest Euclidean distance to the "
    "pore-domain centroid with the smallest flattened voxel index as the final tie break; "
    "the site count is chosen by bisection on |N_system - target|"
)
EUCLIDEAN_RULE = (
    "the same frozen trajectory sites, Euclidean nearest-site assignment with ties broken by "
    "the smallest site index, then deterministic six-connected repair: every voxel of a "
    "non-site-containing component is reassigned to the neighbouring component with the "
    "largest shared interface area, iterated to a fixed point"
)


# --------------------------------------------------------------------------
# geometry-only evaluation
# --------------------------------------------------------------------------
def evaluate_sites_geometry_only(case: str, sites: np.ndarray, tag: str) -> dict[str, Any]:
    boot = ec.bootstrap()
    paths = ec.case_paths(case)
    geom, meta, build_time = ec.build_geometry_from_seed_flat(
        np.unique(np.asarray(sites, dtype=np.int64)), mask_path=paths["mask"], seed_spec=tag
    )
    trace = boot["build_hybrid_trace_geometry"](
        boot["ns"], geom, boot["cfg"], trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
    )
    record = _dimension_record(geom, trace)
    record.update(
        {
            "N_sites": int(np.unique(sites).size),
            "t_geometry_build_s": build_time,
            "face_connected_fraction": float(meta.get("face_connected_fraction", float("nan"))),
            "N_split": int(meta.get("N_split", 0)),
        }
    )
    del geom, trace
    ec.free_gpu()
    return record


def evaluate_partition_geometry_only(case: str, labels: np.ndarray, tag: str) -> dict[str, Any]:
    boot = ec.bootstrap()
    paths = ec.case_paths(case)
    geom, meta, build_time, applied = ec.build_geometry_from_partition(
        labels, mask_path=paths["mask"], seed_spec=tag
    )
    trace = boot["build_hybrid_trace_geometry"](
        boot["ns"], geom, boot["cfg"], trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
    )
    record = _dimension_record(geom, trace)
    record.update(
        {
            "N_sites": 0,
            "t_geometry_build_s": build_time,
            "face_connected_fraction": float(meta.get("face_connected_fraction", float("nan"))),
            "N_split": int(meta.get("N_split", 0)),
        }
    )
    del geom, trace
    ec.free_gpu()
    return record


def _dimension_record(geom: Any, trace: Any) -> dict[str, Any]:
    n_cells = int(geom.n_cells)
    modes = int(trace.n_trace_modes)
    return {
        "N_cv": n_cells,
        "N_edges": int(geom.owner.size),
        "N_interface_facelets": int(trace.n_facelets),
        "N_connected_patches": int(trace.n_patches),
        "N_trace_modes": modes,
        "N_trace_vector_dofs": 3 * modes,
        "N_pressure_dofs_after_gauge": n_cells - 1,
        "N_system": 3 * modes + n_cells - 1,
    }


# --------------------------------------------------------------------------
# baseline site / partition constructors
# --------------------------------------------------------------------------
def mask_fps_order(case: str, count: int, delivery_root: Path) -> np.ndarray:
    cache = delivery_root / "source_data" / "baseline_families" / case / "mask_graph_fps_order.npz"
    if cache.is_file():
        stored = np.load(cache)
        if int(stored["order"].size) >= count:
            return stored["order"][:count].astype(np.int64)
    paths = ec.case_paths(case)
    mask = ec.load_mask(paths["mask"])
    pore = np.flatnonzero(mask.reshape(-1))
    order = ec.graph_farthest_point_order(
        mask,
        pore,
        periodic_x=False,
        count=int(count),
        progress=lambda step, total: print(f"  [{case}] mask-fps {step}/{total}", flush=True),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, order=order.astype(np.int64))
    return order


def axis_divisions_for_block_count(shape, block_count):
    """Near-cubic axis division counts for a requested number of blocks.

    ``n_a = clip(round(L_a * (m / (Lz*Ly*Lx))**(1/3)), 1, L_a)`` keeps the blocks
    as close to cubic as the integer grid allows and is weakly monotone in ``m``.
    """
    total = float(shape[0] * shape[1] * shape[2])
    factor = (max(float(block_count), 1.0) / total) ** (1.0 / 3.0)
    return tuple(
        int(min(max(int(np.floor(float(shape[axis]) * factor + 0.5)), 1), int(shape[axis])))
        for axis in range(3)
    )


def graded_block_labels(mask, block_count):
    """Axis-aligned blocks from a requested block count, with nearly equal sides.

    Each axis of length ``L`` is split into ``n`` contiguous intervals by
    ``index(c) = (c * n) // L``, so interval lengths differ by at most one voxel.
    This generalises the fixed-side rule: ``n_a = L_a / s_a`` reproduces uniform
    sides of ``s_a`` exactly, and every intermediate ``n_a`` becomes reachable,
    which is what lets the total mixed-system dimension be matched.
    """
    shape = tuple(int(value) for value in mask.shape)
    divisions = axis_divisions_for_block_count(shape, block_count)
    grids = [
        (np.arange(shape[axis], dtype=np.int64) * int(divisions[axis])) // int(shape[axis])
        for axis in range(3)
    ]
    block = (
        grids[0][:, None, None] * (divisions[1] * divisions[2])
        + grids[1][None, :, None] * divisions[2]
        + grids[2][None, None, :]
    )
    labels = np.where(mask, block, -1).astype(np.int64)
    return labels, divisions


def block_labels(mask: np.ndarray, sides: tuple[int, int, int], offset: tuple[int, int, int]) -> np.ndarray:
    shape = mask.shape
    grids = []
    counts = []
    for axis in range(3):
        index = (np.arange(shape[axis], dtype=np.int64) + int(offset[axis])) // int(sides[axis])
        index = index - index.min()
        grids.append(index)
        counts.append(int(index.max()) + 1)
    block = (
        grids[0][:, None, None] * (counts[1] * counts[2])
        + grids[1][None, :, None] * counts[2]
        + grids[2][None, None, :]
    )
    labels = np.where(mask, block, -1).astype(np.int64)
    return labels


def component_count(
    mask: np.ndarray,
    labels: np.ndarray,
    table: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """Fast six-connected component count matching the production reindex convention.

    ``pvfv_face_connected_reindex_cpu`` wraps the x axis for connectivity, so
    the screening graph does too.  ``table`` lets a search loop reuse one
    neighbour table instead of rebuilding it per candidate.
    """
    pore, neighbours = table if table is not None else ec.pore_neighbour_table(mask, periodic_x=True)
    node_label = labels.reshape(-1)[pore].astype(np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for column in range(neighbours.shape[1]):
        other = neighbours[:, column]
        ok = other >= 0
        left = np.flatnonzero(ok)
        right = other[ok]
        same = node_label[left] == node_label[right]
        rows.append(left[same])
        cols.append(right[same])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    graph = sparse.coo_matrix(
        (np.ones(row.size, dtype=np.int8), (row, col)), shape=(pore.size, pore.size)
    ).tocsr()
    count, _labels = connected_components(graph, directed=False)
    return int(count)


def euclidean_partition_with_repair(
    mask: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Euclidean nearest-site assignment plus deterministic six-connected repair."""
    from scipy.spatial import cKDTree

    shape = mask.shape
    pore = np.flatnonzero(mask.reshape(-1))
    pore_coords = np.stack(np.unravel_index(pore, shape), axis=1).astype(np.float64)
    site_coords = np.stack(np.unravel_index(np.asarray(sites, dtype=np.int64), shape), axis=1).astype(
        np.float64
    )
    tree = cKDTree(site_coords)
    _distance, nearest = tree.query(pore_coords, k=1)
    node_label = np.asarray(nearest, dtype=np.int64)

    _pore, neighbours = ec.pore_neighbour_table(mask, periodic_x=True)
    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(pore.size, dtype=np.int64)
    site_nodes = lookup[np.asarray(sites, dtype=np.int64)]

    reassigned = 0
    iterations = 0
    while True:
        iterations += 1
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for column in range(neighbours.shape[1]):
            other = neighbours[:, column]
            ok = other >= 0
            left = np.flatnonzero(ok)
            right = other[ok]
            same = node_label[left] == node_label[right]
            rows.append(left[same])
            cols.append(right[same])
        graph = sparse.coo_matrix(
            (
                np.ones(sum(item.size for item in rows), dtype=np.int8),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(pore.size, pore.size),
        ).tocsr()
        count, component = connected_components(graph, directed=False)
        holds_site = np.zeros(count, dtype=bool)
        holds_site[component[site_nodes]] = True
        orphan_components = np.flatnonzero(~holds_site)
        if orphan_components.size == 0:
            break
        if iterations > 200:
            raise RuntimeError("Euclidean connectivity repair did not converge")
        orphan_nodes = np.flatnonzero(np.isin(component, orphan_components))
        # Reassign each orphan component to the neighbouring label with the largest
        # shared interface (deterministic; ties by the smallest label index).
        for orphan in orphan_components:
            members = np.flatnonzero(component == orphan)
            neighbour_labels: list[int] = []
            for column in range(neighbours.shape[1]):
                other = neighbours[members, column]
                ok = other >= 0
                candidate = other[ok]
                external = component[candidate] != orphan
                neighbour_labels.append(node_label[candidate[external]])
            pooled = np.concatenate(neighbour_labels) if neighbour_labels else np.empty(0, dtype=np.int64)
            if pooled.size == 0:
                raise RuntimeError("An orphan component has no neighbour to merge into")
            values, counts = np.unique(pooled, return_counts=True)
            best = values[np.lexsort((values, -counts))[0]]
            node_label[members] = int(best)
            reassigned += int(members.size)
    labels = np.full(mask.size, -1, dtype=np.int64)
    labels[pore] = node_label
    report = {
        "repair_iterations": iterations,
        "reassigned_voxels": int(reassigned),
        "rule": EUCLIDEAN_RULE,
    }
    return labels.reshape(shape), report


# --------------------------------------------------------------------------
# search drivers
# --------------------------------------------------------------------------
def proposed_targets(case: str, delivery_root: Path) -> dict[str, dict[str, Any]]:
    """Budget targets and the proposed rows, using N_system only."""
    table = delivery_root / "tables" / "controlled_refinement.csv"
    if not table.is_file():
        raise FileNotFoundError("Run Phase A first; controlled_refinement.csv is required")
    import csv as _csv

    with table.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in _csv.DictReader(handle) if row["case"] == case]
    if not rows:
        raise RuntimeError(f"No Phase A rows for {case}")
    rows.sort(key=lambda row: int(row["N_system"]))
    full = rows[-1]
    targets = {
        "full": int(full["N_system"]),
        "half": int(round(0.5 * int(full["N_system"]))),
    }
    selected: dict[str, dict[str, Any]] = {}
    for budget, target in targets.items():
        best = min(rows, key=lambda row: abs(int(row["N_system"]) - target))
        selected[budget] = {
            "budget_label": budget,
            "target_N_system": int(target),
            "level_index": int(best["level_index"]),
            "level_label": best["level_label"],
            "N_sites": int(best["N_sites"]),
            "N_cv": int(best["N_cv"]),
            "N_system": int(best["N_system"]),
            "relative_difference_from_target": (int(best["N_system"]) - target) / max(target, 1),
            "run_directory": best["run_directory"],
        }
    return selected


def search_mask_graph_fps(case: str, targets: dict[str, dict[str, Any]], delivery_root: Path) -> dict[str, Any]:
    proposed_full = max(item["N_system"] for item in targets.values())
    upper = int(max(item["N_sites"] for item in targets.values()) * 1.6) + 64
    mask_fps_order(case, upper, delivery_root)
    evaluations: list[dict[str, Any]] = []
    cache: dict[int, dict[str, Any]] = {}

    def evaluate(count: int) -> dict[str, Any]:
        count = int(max(8, min(count, upper)))
        if count in cache:
            return cache[count]
        order = mask_fps_order(case, upper, delivery_root)[:count]
        record = evaluate_sites_geometry_only(case, order, f"mask_graph_fps_{case}_n{count}")
        record.update({"candidate_site_count": count, "method_id": "mask_graph_fps"})
        cache[count] = record
        evaluations.append(record)
        print(
            f"  [{case}] mask_graph_fps n={count} -> N_cv={record['N_cv']} N_system={record['N_system']}",
            flush=True,
        )
        return record

    selections: dict[str, Any] = {}
    for budget, item in targets.items():
        target = int(item["N_system"])
        # Extend the bracket until it actually contains the target, so a match
        # outside the initial guess is never silently reported as the closest
        # available candidate.
        pore_count = int(np.count_nonzero(ec.load_mask(ec.case_paths(case)["mask"])))
        while evaluate(upper)["N_system"] < target and upper < pore_count:
            upper = min(int(upper * 2), pore_count)
            mask_fps_order(case, upper, delivery_root)
        low, high = 8, upper
        # Bisection on the monotone-increasing N_system(count).
        evaluate(low)
        evaluate(high)
        while high - low > 1:
            middle = (low + high) // 2
            record = evaluate(middle)
            if record["N_system"] < target:
                low = middle
            else:
                high = middle
        candidates = sorted(cache.values(), key=lambda record: abs(record["N_system"] - target))
        best = candidates[0]
        relative = (best["N_system"] - target) / max(target, 1)
        selections[budget] = {
            "method_id": "mask_graph_fps",
            "budget_label": budget,
            "target_N_system": target,
            "candidate_site_count": int(best["candidate_site_count"]),
            "N_cv": int(best["N_cv"]),
            "N_system": int(best["N_system"]),
            "relative_N_system_difference_from_proposed": float(relative),
            "within_preferred_5pct": bool(abs(relative) <= MATCH_PREFERRED),
            "within_maximum_10pct": bool(abs(relative) <= MATCH_MAXIMUM),
            "rule": MASK_FPS_RULE,
        }
    return {
        "method_id": "mask_graph_fps",
        "case": case,
        "proposed_full_N_system": proposed_full,
        "rule": MASK_FPS_RULE,
        "search_upper_site_count": upper,
        "evaluations": sorted(evaluations, key=lambda record: record["candidate_site_count"]),
        "selections": selections,
    }


def search_connected_blocks(case: str, targets: dict[str, Any], delivery_root: Path) -> dict[str, Any]:
    """Geometry-only search over connected block agglomerations.

    Two nested candidate families are screened, both producing axis-aligned voxel
    blocks clipped to the pore mask and then split into six-connected components
    by the production reindex:

    * the fixed-side family ``(bz, by, bx)`` over ``BLOCK_SIDES``;
    * the graded family parameterised by a requested block count ``m``, which
      contains the fixed-side family and fills the gaps between its members.

    The graded family is required: on these masks the fixed-side family cannot
    reach the proposed full-budget total system dimension at all, so restricting
    the search to it would force an unmatched baseline.  The two families are not
    nested in general - the graded family is near-cubic by construction - so both
    are evaluated and ranked together, and the search space is therefore a strict
    superset of the locked one.  Selection remains
    geometry-only.
    """
    paths = ec.case_paths(case)
    mask = ec.load_mask(paths["mask"])
    table = ec.pore_neighbour_table(mask, periodic_x=True)
    shape = tuple(int(value) for value in mask.shape)
    pore_count = int(np.count_nonzero(mask))

    screen: list[dict[str, Any]] = []
    for bz in BLOCK_SIDES:
        for by in BLOCK_SIDES:
            for bx in BLOCK_SIDES:
                labels = block_labels(mask, (bz, by, bx), BLOCK_OFFSET)
                screen.append(
                    {
                        "family": "fixed_side",
                        "block_z": bz,
                        "block_y": by,
                        "block_x": bx,
                        "block_volume": bz * by * bx,
                        "screened_N_cv": int(component_count(mask, labels, table)),
                    }
                )

    evaluations: list[dict[str, Any]] = []
    cache: dict[tuple[str, Any], dict[str, Any]] = {}

    def evaluate_fixed(sides: tuple[int, int, int]) -> dict[str, Any]:
        key = ("fixed_side", sides)
        if key in cache:
            return cache[key]
        labels = block_labels(mask, sides, BLOCK_OFFSET)
        record = evaluate_partition_geometry_only(
            case, labels, "connected_block_%s_%dx%dx%d" % (case, sides[0], sides[1], sides[2])
        )
        record.update(
            {
                "family": "fixed_side",
                "block_z": sides[0],
                "block_y": sides[1],
                "block_x": sides[2],
                "block_volume": int(sides[0] * sides[1] * sides[2]),
                "block_offset": list(BLOCK_OFFSET),
                "requested_block_count": "",
                "axis_divisions": "",
                "method_id": "connected_block_agglomeration",
            }
        )
        cache[key] = record
        evaluations.append(record)
        print(
            "  [%s] block fixed %s -> N_cv=%d N_system=%d"
            % (case, sides, record["N_cv"], record["N_system"]),
            flush=True,
        )
        return record

    def evaluate_graded(block_count: int) -> dict[str, Any]:
        block_count = int(max(8, min(int(block_count), pore_count)))
        divisions = axis_divisions_for_block_count(shape, block_count)
        key = ("graded", divisions)
        if key in cache:
            return cache[key]
        labels, _divisions = graded_block_labels(mask, block_count)
        record = evaluate_partition_geometry_only(
            case,
            labels,
            "connected_block_graded_%s_%dx%dx%d"
            % (case, divisions[0], divisions[1], divisions[2]),
        )
        record.update(
            {
                "family": "graded",
                "block_z": "",
                "block_y": "",
                "block_x": "",
                "block_volume": int(np.prod(divisions)),
                "block_offset": list(BLOCK_OFFSET),
                "requested_block_count": int(block_count),
                "axis_divisions": list(divisions),
                "method_id": "connected_block_agglomeration",
            }
        )
        cache[key] = record
        evaluations.append(record)
        print(
            "  [%s] block graded m=%d div=%s -> N_cv=%d N_system=%d"
            % (case, block_count, divisions, record["N_cv"], record["N_system"]),
            flush=True,
        )
        return record

    proposed_cells = sorted(int(item["N_cv"]) for item in targets.values())
    shortlist: set[tuple[int, int, int]] = set()
    for reference_cells in proposed_cells:
        ordered = sorted(screen, key=lambda item: abs(item["screened_N_cv"] - reference_cells))
        for item in ordered[:8]:
            shortlist.add((item["block_z"], item["block_y"], item["block_x"]))
    for sides in sorted(shortlist):
        evaluate_fixed(sides)

    for budget, item in sorted(targets.items()):
        target = int(item["N_system"])
        low, high = 8, pore_count
        if evaluate_graded(high)["N_system"] < target:
            continue
        while high - low > 1:
            middle = (low + high) // 2
            if evaluate_graded(middle)["N_system"] < target:
                low = middle
            else:
                high = middle
            if axis_divisions_for_block_count(shape, low) == axis_divisions_for_block_count(
                shape, high
            ):
                break

    selections: dict[str, Any] = {}
    for budget, item in targets.items():
        target = int(item["N_system"])
        ranked = sorted(
            evaluations,
            key=lambda record: (
                abs(record["N_system"] - target),
                record["block_volume"],
                str(record["family"]),
                str(record["axis_divisions"]),
                str((record["block_z"], record["block_y"], record["block_x"])),
            ),
        )
        best = ranked[0]
        relative = (best["N_system"] - target) / max(target, 1)
        selections[budget] = {
            "method_id": "connected_block_agglomeration",
            "budget_label": budget,
            "target_N_system": target,
            "family": best["family"],
            "block_sides": (
                [best["block_z"], best["block_y"], best["block_x"]]
                if best["family"] == "fixed_side"
                else []
            ),
            "requested_block_count": best["requested_block_count"],
            "axis_divisions": best["axis_divisions"],
            "block_offset": list(BLOCK_OFFSET),
            "N_cv": int(best["N_cv"]),
            "N_system": int(best["N_system"]),
            "relative_N_system_difference_from_proposed": float(relative),
            "within_preferred_5pct": bool(abs(relative) <= MATCH_PREFERRED),
            "within_maximum_10pct": bool(abs(relative) <= MATCH_MAXIMUM),
            "rule": BLOCK_RULE,
        }
    return {
        "method_id": "connected_block_agglomeration",
        "case": case,
        "rule": BLOCK_RULE,
        "screened_candidates": screen,
        "fixed_side_shortlist": sorted(list(shortlist)),
        "evaluations": evaluations,
        "selections": selections,
        "targets": sorted(int(item["N_system"]) for item in targets.values()),
        "fixed_side_family_max_N_system": max(
            [record["N_system"] for record in evaluations if record["family"] == "fixed_side"]
            or [0]
        ),
    }


def search_euclidean_partition(case: str, targets: dict[str, dict[str, Any]], delivery_root: Path) -> dict[str, Any]:
    """Optional baseline: same frozen sites, Euclidean assignment, deterministic repair."""
    paths = ec.case_paths(case)
    mask = ec.load_mask(paths["mask"])
    family = json.loads(
        (delivery_root / "source_data" / "site_families" / case / "site_family_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    level_by_index = {int(level["level_index"]): level for level in family["levels"]}
    evaluations: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for budget, item in targets.items():
        level = level_by_index[int(item["level_index"])]
        sites = np.load(level["sites_npz"])["seed_flat"].astype(np.int64)
        labels, report = euclidean_partition_with_repair(mask, sites)
        changed = _ownership_changed(case, sites, labels)
        record = evaluate_partition_geometry_only(
            case, labels, f"euclidean_repaired_{case}_{level['level_label']}"
        )
        record.update(
            {
                "method_id": "euclidean_partition_with_deterministic_connectivity_repair",
                "budget_label": budget,
                "level_label": level["level_label"],
                "N_sites": int(sites.size),
                "repair": report,
                "changed_ownership_voxels": changed["changed_voxels"],
                "changed_ownership_fraction": changed["changed_fraction"],
                "partition_sha256": ec.sha256_array(labels.astype(np.int32)),
            }
        )
        evaluations.append(record)
        target = int(item["N_system"])
        relative = (record["N_system"] - target) / max(target, 1)
        selections[budget] = {
            "method_id": "euclidean_partition_with_deterministic_connectivity_repair",
            "budget_label": budget,
            "target_N_system": target,
            "level_label": level["level_label"],
            "N_cv": int(record["N_cv"]),
            "N_system": int(record["N_system"]),
            "relative_N_system_difference_from_proposed": float(relative),
            "within_preferred_5pct": bool(abs(relative) <= MATCH_PREFERRED),
            "within_maximum_10pct": bool(abs(relative) <= MATCH_MAXIMUM),
            "changed_ownership_fraction": changed["changed_fraction"],
            "repair": report,
            "rule": EUCLIDEAN_RULE,
        }
        print(
            f"  [{case}] euclidean {level['level_label']} -> N_cv={record['N_cv']} "
            f"N_system={record['N_system']} changed={changed['changed_fraction']:.4f}",
            flush=True,
        )
    return {
        "method_id": "euclidean_partition_with_deterministic_connectivity_repair",
        "case": case,
        "rule": EUCLIDEAN_RULE,
        "evaluations": evaluations,
        "selections": selections,
    }


def _ownership_changed(case: str, sites: np.ndarray, euclidean_labels: np.ndarray) -> dict[str, Any]:
    """Confirm that the Euclidean assignment really changes the active ownership."""
    paths = ec.case_paths(case)
    boot = ec.bootstrap()
    cp = boot["cp"]
    geom, _meta, _time = ec.build_geometry_from_seed_flat(
        sites, mask_path=paths["mask"], seed_spec=f"ownership_reference_{case}"
    )
    geodesic = cp.asnumpy(geom.labels).astype(np.int64)
    mask = cp.asnumpy(geom.mask).astype(bool)
    del geom
    ec.free_gpu()
    # Compare partitions as set partitions: a voxel changed if its co-membership
    # with its geodesic cell's site-owning cell differs.
    site_geodesic = geodesic.reshape(-1)[sites]
    site_euclidean = euclidean_labels.reshape(-1)[sites]
    mapping = {int(a): int(b) for a, b in zip(site_geodesic, site_euclidean)}
    translated = np.full(euclidean_labels.shape, -1, dtype=np.int64).reshape(-1)
    geodesic_flat = geodesic.reshape(-1)
    valid = geodesic_flat >= 0
    translated[valid] = [mapping.get(int(value), -2) for value in geodesic_flat[valid]]
    changed = int(np.count_nonzero(translated[mask.reshape(-1)] != euclidean_labels.reshape(-1)[mask.reshape(-1)]))
    total = int(np.count_nonzero(mask))
    return {"changed_voxels": changed, "changed_fraction": changed / max(total, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery-root", type=Path, default=ec.DELIVERY_ROOT)
    parser.add_argument("--case", default="")
    parser.add_argument("--skip-euclidean", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    delivery_root = Path(args.delivery_root)
    cases = [item.strip() for item in args.case.split(",") if item.strip()] or list(BASELINE_CASES)
    record_path = delivery_root / "protocols" / "baseline_selection_locked.json"
    if record_path.is_file() and not args.force:
        print(f"[phaseB] selection record already exists: {record_path}", flush=True)
        return 0

    payload: dict[str, Any] = {
        "protocol_id": ec.PROTOCOL_ID,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_is_result_blind": True,
        "selection_may_use": ["N_system", "N_cv", "topology and admissibility metrics"],
        "selection_may_not_use": [
            "e_K",
            "e_phi",
            "e_u",
            "e_p",
            "reference velocity",
            "reference flux",
            "reference pressure",
            "permeability",
        ],
        "matching_rule": {
            "primary_metric": "N_system",
            "preferred_relative_difference_max": MATCH_PREFERRED,
            "absolute_allowed_relative_difference_max": MATCH_MAXIMUM,
        },
        "cases": {},
    }
    for case in cases:
        print(f"[phaseB] geometry-only search for {case}", flush=True)
        targets = proposed_targets(case, delivery_root)
        entry: dict[str, Any] = {
            "proposed": targets,
            "mask_graph_fps": search_mask_graph_fps(case, targets, delivery_root),
            "connected_block_agglomeration": search_connected_blocks(case, targets, delivery_root),
        }
        if not args.skip_euclidean:
            try:
                entry["euclidean_partition_with_deterministic_connectivity_repair"] = search_euclidean_partition(
                    case, targets, delivery_root
                )
            except Exception as error:
                entry["euclidean_partition_with_deterministic_connectivity_repair"] = {
                    "status": "OMITTED",
                    "reason": f"{type(error).__name__}: {error}",
                }
                print(f"  [{case}] euclidean baseline omitted: {error}", flush=True)
        payload["cases"][case] = entry

    ec.write_json(record_path, payload)
    for case, entry in payload["cases"].items():
        ec.write_json(
            delivery_root / "protocols" / "baseline_selections" / f"{case}.json",
            {
                "protocol_id": ec.PROTOCOL_ID,
                "case": case,
                "generated_at_utc": payload["generated_at_utc"],
                "matching_rule": payload["matching_rule"],
                "selection_is_result_blind": True,
                "selections": {
                    method: block.get("selections")
                    for method, block in entry.items()
                    if isinstance(block, dict) and "selections" in block
                },
                "proposed": entry["proposed"],
            },
        )
    digest = ec.sha256_file(record_path)
    (record_path.with_suffix(".json.sha256")).write_text(
        f"{digest}  {record_path.name}\n", encoding="utf-8"
    )
    print(f"[phaseB] locked selection record {record_path} sha256={digest}", flush=True)

    search_rows: list[dict[str, Any]] = []
    for case, entry in payload["cases"].items():
        for method, block in entry.items():
            if method == "proposed" or not isinstance(block, dict) or "evaluations" not in block:
                continue
            for record in block["evaluations"]:
                search_rows.append(
                    {
                        "case": case,
                        "method_id": method,
                        "candidate": json.dumps(
                            {
                                key: record[key]
                                for key in (
                                    "candidate_site_count",
                                    "block_z",
                                    "block_y",
                                    "block_x",
                                    "level_label",
                                )
                                if key in record
                            }
                        ),
                        "N_cv": record["N_cv"],
                        "N_edges": record["N_edges"],
                        "N_connected_patches": record["N_connected_patches"],
                        "N_trace_modes": record["N_trace_modes"],
                        "N_system": record["N_system"],
                        "reference_errors_accessed": False,
                    }
                )
    ec.write_csv(
        delivery_root / "source_data" / "baseline_geometry_only_search.csv",
        search_rows,
        [
            "case",
            "method_id",
            "candidate",
            "N_cv",
            "N_edges",
            "N_connected_patches",
            "N_trace_modes",
            "N_system",
            "reference_errors_accessed",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
