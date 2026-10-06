"""Controlled prescribed-site refinement on a FIXED voxel mask.

A *trajectory-nested* candidate pool does not give a refinement parameter: its
graph fill distance h_S saturates (13,12,12,12,11,11 on the Bentheimer crop),
because densifying a trajectory support packs sites along paths already visited,
so no convergence order can be stated.

This driver refines a *geometry-only* nested family instead: deterministic
graph-farthest-point sampling over **every pore voxel**.  That family is the one
the site-rule comparison (Table S3) measured at two isolated budgets, where it
already reached h_S/h = 3 on the Bentheimer crop with 2350 sites against 11 for
6739 trajectory sites.  Here it is run as a ladder so that h_S is a genuine
refinement parameter.

Nothing about the discretization changes: mask, reference, boundary condition,
viscosity, forcing, trace basis, stabilization and solver are the fixed values
in study_common.FORWARD_ARGS.  Only the prescribed site set varies.

Every numerical operator is the one of the forward runs in gpu/code, reached
through study_common / forward_runs exactly as the other scripts in gpu/studies
reach it.  No file of gpu/code is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
import forward_runs as fr  # noqa: E402

PROTOCOL_ID = "pvfv_mask_fps_refinement"

# Geometric ladder.  Nested by construction: level k is the length-N_k prefix of
# one deterministic farthest-point ordering, so every coarse site survives into
# every finer level.
DEFAULT_LEVELS = (200, 400, 800, 1600, 3200, 6400)

SITE_RULE = (
    "deterministic graph-farthest-point sampling over every pore voxel; "
    "initial candidate = smallest Euclidean distance to the pore-domain centroid "
    "with the smallest flattened voxel index as the final tie break; subsequent "
    "candidate = maximum six-neighbour pore-graph distance to the selected set, "
    "smallest flattened index breaking ties"
)


def fps_order_path(root: Path, case: str, count: int) -> Path:
    return root / "site_families" / (case + "__mask_graph_fps_order_n%d.npz" % count)


def build_or_load_order(case, mask, periodic_x, count, root):
    """Deterministic graph-FPS ordering over the whole pore space, cached."""
    family_dir = root / "site_families"
    family_dir.mkdir(parents=True, exist_ok=True)
    for cached in sorted(family_dir.glob(case + "__mask_graph_fps_order_n*.npz")):
        with np.load(cached, allow_pickle=True) as data:
            order = data["order"].astype(np.int64)
            meta = json.loads(str(data["meta"]))
        if order.size >= count and bool(meta["periodic_x"]) == bool(periodic_x):
            return order[:count], meta

    pore_flat = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    start = time.perf_counter()
    order = ec.graph_farthest_point_order(
        mask,
        pore_flat,
        periodic_x=periodic_x,
        count=int(count),
        progress=lambda step, total: print(
            "[fps] %s: %d/%d" % (case, step, total), flush=True
        ),
    )
    elapsed = float(time.perf_counter() - start)
    meta = {
        "case": case,
        "rule": SITE_RULE,
        "candidate_pool": "every pore voxel (no trajectory, no flow or reference field)",
        "periodic_x": bool(periodic_x),
        "count": int(count),
        "pore_voxel_count": int(pore_flat.size),
        "build_seconds": elapsed,
        "order_sha256": ec.sha256_array(order),
    }
    path = fps_order_path(root, case, int(count))
    np.savez_compressed(path, order=order, meta=json.dumps(meta))
    return order, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="orthogonal_duct,skewed_duct,bentheimer_crop")
    parser.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS))
    parser.add_argument("--root", default=None, help="Output root for this study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Geometry only: build the family and report its separation descriptors",
    )
    args = parser.parse_args()

    root = Path(
        args.root or "reproduce/figure_06/refinement"
    )
    root.mkdir(parents=True, exist_ok=True)
    cases = [c for c in args.cases.split(",") if c]
    levels = [int(v) for v in args.levels.split(",") if v]

    boot = ec.bootstrap(root / "bootstrap")
    cp = boot["cp"]
    rows = []

    for case in cases:
        paths = ec.case_paths(case)
        mask = ec.load_mask(paths["mask"])
        n_pore = int(mask.sum())
        case_levels = [n for n in levels if n <= n_pore]

        # Measure, never assume, the ownership graph convention for this case:
        # build one small partition and compare the production distance field
        # against BFS under both boundary conventions.
        probe_order, _ = build_or_load_order(case, mask, False, 64, root)
        probe_sites = np.unique(probe_order[:64])
        geom_probe, _meta, _t = ec.build_geometry_from_seed_flat(
            probe_sites, mask_path=paths["mask"], seed_spec=case + ":convention_probe"
        )
        convention = ec.detect_ownership_graph_convention(
            mask, cp.asnumpy(geom_probe.dist).astype(np.float64), probe_sites
        )
        del geom_probe
        ec.free_gpu()
        if not convention["reproduced"]:
            raise RuntimeError(
                case + ": neither six-neighbour boundary convention reproduces the "
                "production ownership distance field; refusing to guess"
            )
        periodic_x = bool(convention["matched_periodic_x"])

        order, family_meta = build_or_load_order(
            case, mask, periodic_x, max(case_levels), root
        )
        family_meta = dict(family_meta)
        family_meta["ownership_graph_convention"] = convention
        family_meta["levels"] = case_levels
        ec.write_json(root / "site_families" / (case + "__family.json"), family_meta)

        if args.dry_run:
            for n in case_levels:
                sites = np.unique(order[:n])
                q = ec.separation_distance(mask, sites, periodic_x=periodic_x)
                print("[dry] %s n=%d q_S/h=%s" % (case, n, q), flush=True)
            continue

        for index, n in enumerate(case_levels):
            label = "M%d_n%d" % (index, n)
            run_dir = root / "runs" / case / label
            row_path = run_dir / "row.json"
            if args.resume and row_path.exists():
                rows.append(json.loads(row_path.read_text(encoding="utf-8")))
                print("[resume] %s %s" % (case, label), flush=True)
                continue
            sites = np.unique(order[:n])
            started = time.perf_counter()
            out = fr.run_point(
                case=case,
                mask_path=paths["mask"],
                reference_path=paths["reference"],
                run_dir=run_dir,
                seed_spec=case + ":mask_graph_fps:n%d" % n,
                seed_flat=sites,
                graph_periodic_x=periodic_x,
                method_id="mask_graph_fps",
                site_rule=SITE_RULE,
            )
            row = out["row"]
            row.update(
                {
                    "protocol_id": PROTOCOL_ID,
                    "level_index": index,
                    "level_label": label,
                    "target_site_count": int(n),
                    "selection_rule": "mask_graph_fps_nested_prefix",
                    "candidate_pool": family_meta["candidate_pool"],
                    "graph_metric_periodic_x": periodic_x,
                    "wall_seconds": float(time.perf_counter() - started),
                }
            )
            ec.write_json(row_path, row)
            rows.append(row)
            print(
                "[run] %s %s N_cv=%s N_system=%s h_S/h=%s e_u=%.4f%% e_p=%.4f%% "
                "e_phi=%.4f%% e_K=%.4f%% (%.1fs)"
                % (
                    case,
                    label,
                    row["N_cv"],
                    row["N_system"],
                    row.get("h_S_over_h"),
                    row["e_u_percent"],
                    row["e_p_percent"],
                    row["e_phi_percent"],
                    row["e_K_percent"],
                    row["wall_seconds"],
                ),
                flush=True,
            )

    if rows:
        columns = sorted({key for row in rows for key in row})
        lead = [
            "protocol_id", "case", "case_display", "level_index", "level_label",
            "method_id", "N_sites", "N_f", "N_cv", "N_system",
            "h_S_over_h", "q_S_over_h", "mesh_ratio_hS_over_qS",
            "e_K_percent", "e_phi_percent", "e_u_percent", "e_p_percent",
        ]
        columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
        ec.write_csv(root / "mask_fps_refinement.csv", rows, columns)
        print("[write] %s  (%d rows)" % (root / "mask_fps_refinement.csv", len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
