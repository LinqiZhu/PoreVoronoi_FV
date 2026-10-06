"""Frozen candidate pool and deterministic nested site family (study A, step 1).

The candidate pool is exactly the set of unique pore voxels visited by the full
frozen trajectory window, produced by the production loader
``hybrid_site_sources.load_particle_window_seed_flat`` with
``max_snap_displacement_vox=0.0``.  No mask, reference-field, pressure-field or
error-field information is ever consulted.

The nested ordering is deterministic graph-farthest-point sampling on that
frozen pool, using the same six-neighbour pore-graph metric and the same
boundary contract as the production ownership build.  The boundary contract is
not assumed: it is *measured* by comparing a six-neighbour BFS against the
production ``geom.dist`` field under both the periodic and the non-periodic
convention, and the matching convention is recorded.

Usage:
    python build_nested_trajectory_site_family.py [--case CASE] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import study_common as ec

LEVEL_FRACTIONS = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
MINIMUM_SITES = 128
SELECTION_RULE = (
    "deterministic nested graph-farthest-point sampling on the frozen "
    "trajectory-site pool; initial candidate = smallest Euclidean distance to the "
    "pore-domain centroid with the smallest flattened voxel index as the final tie "
    "break; subsequent candidate = maximum six-neighbour pore-graph distance to the "
    "selected set with the smallest flattened voxel index as the final tie break; "
    "no reseeding, relocation, snapping or reference-field access"
)


def level_counts(pool_size: int) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for index, fraction in enumerate(LEVEL_FRACTIONS):
        target = int(np.floor(fraction * pool_size + 0.5))
        target = max(target, MINIMUM_SITES)
        target = min(target, pool_size)
        levels.append(
            {
                "level_index": index,
                "level_label": f"L{index}_frac{fraction:g}",
                "fraction": fraction,
                "target_site_count": int(target),
            }
        )
    # Deduplicate levels that collapse onto the same count (small pools).
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for level in levels:
        if level["target_site_count"] in seen:
            continue
        seen.add(level["target_site_count"])
        unique.append(level)
    for position, level in enumerate(unique):
        level["level_index"] = position
    return unique


def periodicity_audit(case: str, pool: np.ndarray) -> dict[str, Any]:
    """Measure which six-neighbour boundary convention production ownership uses."""
    boot = ec.bootstrap()
    cp = boot["cp"]
    paths = ec.case_paths(case)
    geom, _meta, _time = ec.build_geometry_from_seed_flat(
        pool, mask_path=paths["mask"], seed_spec=f"periodicity_audit_{case}"
    )
    mask = cp.asnumpy(geom.mask).astype(bool)
    labels = cp.asnumpy(geom.labels).astype(np.int64)
    distance = cp.asnumpy(geom.dist).astype(np.float64)
    findings: dict[str, Any] = {}
    for periodic in (False, True):
        pore, neighbours = ec.pore_neighbour_table(mask, periodic_x=periodic)
        lookup = np.full(mask.size, -1, dtype=np.int64)
        lookup[pore] = np.arange(pore.size, dtype=np.int64)
        nodes = lookup[np.unique(pool)]
        computed, _labels = ec.multi_source_bfs(neighbours, nodes)
        production = distance.reshape(-1)[pore]
        findings[f"periodic_x_{periodic}"] = {
            "identical": bool(np.array_equal(computed.astype(np.float64), production)),
            "max_absolute_difference": float(np.max(np.abs(computed.astype(np.float64) - production))),
        }
    matching = [key for key, value in findings.items() if value["identical"]]
    audit = {
        "case": case,
        "trace_periodic_x_flag": None,
        "production_h_S_over_h": float(np.max(distance[labels >= 0])),
        "findings": findings,
        "matched_convention": matching[0] if matching else None,
        "status": "PASS" if len(matching) >= 1 else "FAIL",
    }
    ec.free_gpu()
    del geom
    return audit


def build_family(case: str, delivery_root: Path, force: bool) -> dict[str, Any]:
    out_dir = delivery_root / "source_data" / "site_families" / case
    manifest_path = out_dir / "site_family_manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pool, window_metadata = ec.full_trajectory_seed_flat(case)
    pool = np.unique(np.asarray(pool, dtype=np.int64))
    paths = ec.case_paths(case)

    audit = periodicity_audit(case, pool)
    if audit["status"] != "PASS":
        raise RuntimeError(f"{case}: could not reproduce production ownership distances")
    periodic_x = audit["matched_convention"] == "periodic_x_True"

    mask = ec.load_mask(paths["mask"])
    shape = mask.shape
    coordinates = np.stack(np.unravel_index(pool, shape), axis=1).astype(np.int64)
    np.savez_compressed(
        out_dir / "candidate_pool.npz", seed_flat=pool, coordinates_zyx=coordinates
    )
    ec.write_csv(
        out_dir / "candidate_pool.csv",
        [
            {"flat_index": int(flat), "z": int(z), "y": int(y), "x": int(x)}
            for flat, (z, y, x) in zip(pool, coordinates)
        ],
        ["flat_index", "z", "y", "x"],
    )

    levels = level_counts(int(pool.size))
    maximum_ordered = max(level["target_site_count"] for level in levels if level["target_site_count"] < pool.size)
    maximum_ordered = min(max(maximum_ordered, MINIMUM_SITES), int(pool.size))
    start = time.perf_counter()
    order = ec.graph_farthest_point_order(
        mask,
        pool,
        periodic_x=periodic_x,
        count=int(maximum_ordered),
        progress=lambda step, total: print(f"  [{case}] fps {step}/{total}", flush=True),
    )
    order_time = float(time.perf_counter() - start)

    level_records: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    for level in levels:
        count = int(level["target_site_count"])
        if count >= int(pool.size):
            sites = pool.copy()
            source = "complete frozen candidate pool"
        else:
            sites = np.asarray(order[:count], dtype=np.int64)
            source = "first %d entries of the frozen graph-FPS ordering" % count
        sorted_sites = np.sort(sites)
        if previous is not None and not np.all(np.isin(previous, sorted_sites)):
            raise RuntimeError(f"{case}: level {level['level_label']} is not nested")
        previous = sorted_sites
        level_coords = np.stack(np.unravel_index(sorted_sites, shape), axis=1).astype(np.int64)
        level_dir = out_dir / f"level_{level['level_index']:02d}_{level['level_label']}"
        level_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            level_dir / "sites.npz",
            seed_flat=sorted_sites,
            selection_order=sites,
            coordinates_zyx=level_coords,
        )
        ec.write_csv(
            level_dir / "sites.csv",
            [
                {
                    "selection_rank": int(rank),
                    "flat_index": int(flat),
                    "z": int(z),
                    "y": int(y),
                    "x": int(x),
                }
                for rank, (flat, (z, y, x)) in enumerate(zip(sites, np.stack(np.unravel_index(sites, shape), axis=1)))
            ],
            ["selection_rank", "flat_index", "z", "y", "x"],
        )
        record = dict(level)
        record.update(
            {
                "case": case,
                "N_sites": int(sorted_sites.size),
                "source": source,
                "site_set_sha256": ec.sha256_array(sorted_sites),
                "selection_order_sha256": ec.sha256_array(np.asarray(sites, dtype=np.int64)),
                "sites_npz": str(level_dir / "sites.npz"),
                "all_sites_trajectory_visited": bool(np.all(np.isin(sorted_sites, pool))),
            }
        )
        level_records.append(record)

    full_level = level_records[-1]
    if full_level["N_sites"] != int(pool.size):
        raise RuntimeError(f"{case}: the full level does not equal the frozen pool")

    manifest = {
        "protocol_id": ec.PROTOCOL_ID,
        "case": case,
        "case_display": ec.CASES[case]["display"],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": SELECTION_RULE,
        "level_fractions": LEVEL_FRACTIONS,
        "level_count_rule": "floor(fraction * pool_size + 0.5), clipped to [minimum_sites, pool_size]",
        "minimum_sites": MINIMUM_SITES,
        "reference_field_used": False,
        "error_metric_used": False,
        "reseed_or_relocate": False,
        "mask_derived_sites_added": 0,
        "candidate_pool_size": int(pool.size),
        "candidate_pool_sha256": ec.sha256_array(pool),
        "candidate_pool_npz": str(out_dir / "candidate_pool.npz"),
        "mask_path": str(paths["mask"]),
        "mask_sha256": ec.sha256_file(paths["mask"]),
        "particle_window_path": str(paths["particle_window"]),
        "particle_window_sha256": ec.sha256_file(paths["particle_window"]),
        "window_metadata": {
            str(key): value for key, value in window_metadata.items() if np.isscalar(value)
        },
        "ownership_boundary_contract_audit": audit,
        "graph_metric_periodic_x": periodic_x,
        "farthest_point_order_seconds": order_time,
        "levels": level_records,
    }
    ec.write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery-root", type=Path, default=ec.DELIVERY_ROOT)
    parser.add_argument("--case", default="", help="Comma separated subset of cases")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a manifest exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = [item.strip() for item in args.case.split(",") if item.strip()] or [
        "orthogonal_duct",
        "skewed_duct",
        "bentheimer_crop",
    ]
    if args.dry_run:
        for case in cases:
            pool, _meta = ec.full_trajectory_seed_flat(case)
            print(case, "pool", np.unique(pool).size, level_counts(int(np.unique(pool).size)))
        return 0

    summary = []
    for case in cases:
        start = time.perf_counter()
        manifest = build_family(case, Path(args.delivery_root), args.force)
        summary.append(
            {
                "case": case,
                "pool": manifest["candidate_pool_size"],
                "levels": [level["N_sites"] for level in manifest["levels"]],
                "periodic_x": manifest["graph_metric_periodic_x"],
                "seconds": round(time.perf_counter() - start, 1),
            }
        )
        print(json.dumps(summary[-1]), flush=True)
    ec.write_json(
        Path(args.delivery_root) / "source_data" / "site_families" / "site_family_summary.json", summary
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
