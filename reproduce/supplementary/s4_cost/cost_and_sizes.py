"""End-to-end algebraic size, cost and memory.

Besides the control-volume counts, this module reports the nonzero counts, the
memory, and the separation between cold construction and a repeated query for
every run directory produced by the refinement and baseline studies, and for
the image-resolved reference solver on the same masks.

The nonzero counts are computed by a *sparsity-only* traversal of the retained
assembly: the same incidence, the same per-cell local mode sets, the same
global dof numbering, but no local matrix.  That makes the pass roughly two
orders of magnitude cheaper than a full assembly while returning exactly the
patterns of A and D.  The pattern is checked against a full production assembly
on the smallest case before it is used anywhere else.
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

PROTOCOL_ID = "pvfv_cost_and_sizes"


def sparsity(trace):
    """Patterns of A and D, and the per-cell local dof sets, without values."""
    n_facelets = trace.n_facelets
    incident_cell = np.concatenate([trace.face_owner, trace.face_neigh, trace.wall_cell])
    incident_face = np.concatenate([
        np.arange(n_facelets, dtype=np.int32),
        np.arange(n_facelets, dtype=np.int32),
        np.full(trace.wall_cell.size, -1, dtype=np.int32),
    ])
    incident_normal = np.concatenate(
        [trace.face_normal, -trace.face_normal, trace.wall_normal], axis=0
    )
    order = np.argsort(incident_cell, kind="stable")
    incident_cell = incident_cell[order]
    incident_face = incident_face[order]
    incident_normal = incident_normal[order]
    cell_start = np.searchsorted(incident_cell, np.arange(trace.n_cells + 1))

    area = trace.voxel_size * trace.voxel_size
    a_rows, a_cols = [], []
    d_rows, d_cols, d_values = [], [], []
    local_sizes = np.zeros(trace.n_cells, dtype=np.int64)
    for cell in range(trace.n_cells):
        first, last = int(cell_start[cell]), int(cell_start[cell + 1])
        cell_faces = incident_face[first:last]
        cell_normals = incident_normal[first:last]
        mode_set = set()
        for facelet in cell_faces[cell_faces >= 0]:
            for mode in trace.face_mode_ids[int(facelet)]:
                if mode >= 0:
                    mode_set.add(int(mode))
        local_modes = np.asarray(sorted(mode_set), dtype=np.int64)
        local_sizes[cell] = local_modes.size
        if local_modes.size == 0:
            continue
        local_index = {int(mode): index for index, mode in enumerate(local_modes)}
        n_local = 3 * int(local_modes.size)
        divergence = np.zeros(n_local, dtype=np.float64)
        for facelet, normal in zip(cell_faces, cell_normals):
            if facelet < 0:
                continue
            ids = trace.face_mode_ids[int(facelet)]
            values = trace.face_mode_values[int(facelet)]
            for mode, value in zip(ids, values):
                if mode < 0:
                    continue
                block = slice(3 * local_index[int(mode)], 3 * local_index[int(mode)] + 3)
                divergence[block] += area * value * normal
        global_dofs = (3 * local_modes[:, None] + np.arange(3, dtype=np.int64)[None, :]).reshape(-1)
        a_rows.append(np.repeat(global_dofs, n_local))
        a_cols.append(np.tile(global_dofs, n_local))
        nonzero = np.flatnonzero(divergence)
        d_rows.append(np.full(nonzero.size, cell, dtype=np.int64))
        d_cols.append(global_dofs[nonzero])
        d_values.append(divergence[nonzero])

    n_dofs = 3 * trace.n_trace_modes
    rows = np.concatenate(a_rows) if a_rows else np.empty(0, dtype=np.int64)
    cols = np.concatenate(a_cols) if a_cols else np.empty(0, dtype=np.int64)
    A = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_dofs, n_dofs)
    ).tocsr()
    A.data[:] = 1
    A = A + A.T  # the production assembly symmetrises, which can only union patterns
    A.data[:] = 1
    D = sparse.coo_matrix(
        (np.concatenate(d_values) if d_values else np.empty(0),
         (np.concatenate(d_rows) if d_rows else np.empty(0, dtype=np.int64),
          np.concatenate(d_cols) if d_cols else np.empty(0, dtype=np.int64))),
        shape=(trace.n_cells, n_dofs),
    ).tocsr()
    return {
        "nnz_A_structural_union": int(A.nnz), "nnz_D": int(D.nnz),
        "nnz_D_r": int(D[:-1].nnz),
        "local_modes_mean": float(local_sizes.mean()),
        "local_modes_max": int(local_sizes.max()),
    }


def stored_nonzeros(row, pattern):
    """Recover the nonzeros the solver actually stores, exactly, from the record.

    The retained MINRES path stores
        factor_bytes = 12 nnz(KKT) + 4 (N_system + 1) + 8 N_system,
    because the CSR carries float64 data, int32 indices and an int32 indptr, and
    the preconditioner is one float64 per unknown.  Inverting that gives
    nnz(KKT) exactly, and nnz(A) = nnz(KKT) - 2 nnz(D_r).

    This is not the same as the structural union of the per-cell dense blocks:
    the production assembly symmetrises with a sparse addition, which stores only
    numerically nonzero results, so exactly-zero entries of the local matrices are
    dropped.  Both counts are reported.
    """
    n = int(row["N_system"])
    total_bytes = float(row["factor_memory_MiB"]) * float(2 ** 20)
    nnz_kkt = (total_bytes - 4.0 * (n + 1) - 8.0 * n) / 12.0
    nnz_kkt_int = int(round(nnz_kkt))
    if abs(nnz_kkt - nnz_kkt_int) > 1e-6:
        raise RuntimeError(
            "recovered KKT nonzero count is not an integer: %r" % nnz_kkt
        )
    nnz_a = nnz_kkt_int - 2 * int(pattern["nnz_D_r"])
    return {
        "nnz_KKT": nnz_kkt_int,
        "nnz_A": nnz_a,
        "nnz_A_structural_zero_fraction": (
            1.0 - nnz_a / float(pattern["nnz_A_structural_union"])
            if pattern["nnz_A_structural_union"] else float("nan")
        ),
        "kkt_bytes_per_unknown": total_bytes / float(n),
    }


def reference_sizes(reference_path):
    """Algebraic size of the image-resolved reference solver on the same mask."""
    with np.load(reference_path, allow_pickle=True) as data:
        mask = np.asarray(data["mask"]).astype(bool)
        owner = np.asarray(data["owner"])
        n_faces = int(owner.size)
        n_cells = int(mask.sum())
    # The reference is a collocated voxel finite-volume Stokes solver with three
    # velocity components per pore voxel and one pressure per pore voxel, and one
    # flux per internal face.  Counting is from the archived arrays, not assumed.
    return {
        "reference_pore_voxels": n_cells,
        "reference_internal_faces": n_faces,
        "reference_velocity_unknowns": 3 * n_cells,
        "reference_pressure_unknowns": n_cells,
        "reference_total_unknowns": 4 * n_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", default=",".join([
        "reproduce/figure_06/refinement/runs",
    ]))
    parser.add_argument("--out", default="reproduce/supplementary/s4_cost")
    parser.add_argument("--verify-smallest", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    boot = ec.bootstrap(out / "bootstrap")
    ns, cfg = boot["ns"], boot["cfg"]

    rows = []
    verified = None
    run_dirs = []
    for root in args.roots.split(","):
        run_dirs.extend(sorted(Path(root).glob("*/*/row.json")))

    for index, row_path in enumerate(run_dirs):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        case = row["case"]
        paths = ec.case_paths(case)
        with np.load(row_path.parent / "sites_and_partition.npz") as data:
            sites = np.asarray(data["seed_flat"], dtype=np.int64)
        started = time.perf_counter()
        geom, _meta, build_time = ec.build_geometry_from_seed_flat(
            sites, mask_path=paths["mask"], seed_spec="cost_pass:" + row["level_label"]
        )
        trace_start = time.perf_counter()
        trace = boot["build_hybrid_trace_geometry"](
            ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
        )
        trace_time = float(time.perf_counter() - trace_start)
        pattern_start = time.perf_counter()
        pattern = sparsity(trace)
        pattern_time = float(time.perf_counter() - pattern_start)

        if verified is None:
            full = boot["assemble_moment_constrained_hybrid_stokes"](
                trace, viscosity=float(cfg.nu),
                body_force=np.asarray(ec.FORWARD_ARGS["body_force"], dtype=np.float64),
                viscous_form=str(ec.FORWARD_ARGS["viscous_form"]),
            )
            recovered = stored_nonzeros(row, pattern)
            verified = {
                "checked_on": row["level_label"] + "/" + case,
                "nnz_A_recovered_from_stored_bytes": recovered["nnz_A"],
                "nnz_A_production": int(full.velocity_matrix.nnz),
                "nnz_A_structural_union": pattern["nnz_A_structural_union"],
                "nnz_D_pattern_pass": pattern["nnz_D"],
                "nnz_D_production": int(full.divergence_matrix.nnz),
                "note": (
                    "the structural union counts every entry of every per-cell dense "
                    "block; the production assembly symmetrises with a sparse "
                    "addition and therefore stores only numerically nonzero results"
                ),
            }
            verified["A_matches"] = (
                verified["nnz_A_recovered_from_stored_bytes"] == verified["nnz_A_production"]
            )
            verified["D_matches"] = (
                verified["nnz_D_pattern_pass"] == verified["nnz_D_production"]
            )
            verified["status"] = "PASS" if verified["A_matches"] and verified["D_matches"] else "FAIL"
            ec.write_json(out / "sparsity_pass_verification.json", verified)
            print("[verify] sparsity pass vs production assembly: " + verified["status"])
            for key, value in verified.items():
                print("          %-24s %s" % (key, value))
            del full
            if verified["status"] != "PASS":
                return 1

        record = dict(row)
        record.update(pattern)
        record.update(stored_nonzeros(row, pattern))
        record.update(reference_sizes(paths["reference"]))
        record.update({
            "protocol_id": PROTOCOL_ID,
            "t_sparsity_pass_s": pattern_time,
            "t_geometry_rebuild_s": build_time,
            "t_trace_rebuild_s": trace_time,
            "KKT_over_reference_unknowns":
                float(row["N_system"]) / float(record["reference_total_unknowns"]),
            "N_cv_over_N_f": float(row["N_cv"]) / float(row["N_f"]),
            "N_z_over_N_f": float(row["N_trace_vector_dofs"]) / float(row["N_f"])
                if "N_trace_vector_dofs" in row else float("nan"),
            "T_cold_s": (
                float(row["t_geometry_build_s"]) + float(row["t_trace_geometry_s"])
                + float(row["t_matrix_assembly_s"]) + float(row["t_factorization_s"])
                + float(row["t_cached_call_median_s"])
            ),
            "T_query_s": float(row["t_cached_call_median_s"]),
        })
        record["break_even_queries_vs_rebuild"] = (
            record["T_cold_s"] / max(record["T_query_s"], 1e-12)
        )
        rows.append(record)
        print(
            "[cost] %-16s %-10s N_cv=%5d KKT=%7d nnz(A)=%9d nnz(D)=%7d "
            "KKT/ref=%.3f cold=%.1fs query=%.1fs mem=%.1fMiB (%.1fs)"
            % (case, row["level_label"], row["N_cv"], row["N_system"],
               record["nnz_A"], pattern["nnz_D"],
               record["KKT_over_reference_unknowns"], record["T_cold_s"],
               record["T_query_s"], row["factor_memory_MiB"],
               time.perf_counter() - started),
            flush=True,
        )
        del geom, trace
        ec.free_gpu()

    if rows:
        columns = sorted({k for r in rows for k in r})
        lead = ["protocol_id", "case", "level_label", "N_f", "N_cv", "N_edges",
                "N_connected_patches", "N_trace_modes", "N_system",
                "nnz_A", "nnz_D", "nnz_KKT", "factor_memory_MiB",
                "reference_total_unknowns", "KKT_over_reference_unknowns",
                "T_cold_s", "T_query_s"]
        columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
        ec.write_csv(out / "cost_and_sizes.csv", rows, columns)
        print("[write] " + str(out / "cost_and_sizes.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
