"""An exact reduction of ker A to a 6 N_c rigid-motion constraint system.

The assembled viscous form is a sum of squares,

    y^T A y = sum_i [ 2 nu V_i |eps_i(y)|^2 + sum_gamma w |s_{i,gamma}(y)|^2 ],

so A is symmetric positive semi-definite and y^T A y = 0 if and only if A y = 0.
Both terms vanish exactly when, for every cell i,

    s_{i,gamma}(y) = 0  for every facelet of the cell, and  sym(G_i y) = 0.

The first says the trace on the cell boundary is the affine field
x -> R_i y + (G_i y)(x - x_i), and that this field is zero at a wall facelet,
because a wall facelet contributes its own zero trace.  The second says the
affine field is a rigid motion.  Writing a_i = R_i y and G_i y = skew(omega_i),
the kernel is therefore in bijection with the solutions of

    a_i + omega_i x r_{i,gamma}  =  a_j + omega_j x r_{j,gamma}   (shared facelet)
    a_i + omega_i x r_{i,gamma}  =  0                             (wall facelet)

in 6 N_c unknowns.  The map is injective because a rigid motion that produces a
zero trace on every facelet forces every retained mode coefficient to zero, the
modes being an orthonormalised basis of the affine functions the patch actually
supports; and it is surjective because a rigid motion restricted to a planar
patch lies in that same affine space.  Hence

    dim ker A  =  dim null(M),

with M the sparse constraint matrix above.  This module builds M and computes
its nullity, which decides the kernel question exactly at a fraction of the cost
of touching A, and reports the smallest singular values so the decision is not a
single thresholded number.
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
from scipy.sparse.linalg import eigsh, splu

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

PROTOCOL_ID = "pvfv_kernel_exact"


def rigid_blocks(rows0, cells, offsets, sign):
    """Rows of [ I3 | -[r]x ] * sign for one cell block, as COO triplets."""
    n = cells.size
    r = offsets
    rows, cols, vals = [], [], []
    for component in range(3):
        rows.append(rows0 + component)
        cols.append(6 * cells + component)
        vals.append(np.full(n, float(sign)))
    # -[r]x = [[0, r2, -r1], [-r2, 0, r0], [r1, -r0, 0]]
    cross = [
        (0, 1, r[:, 2]), (0, 2, -r[:, 1]),
        (1, 0, -r[:, 2]), (1, 2, r[:, 0]),
        (2, 0, r[:, 1]), (2, 1, -r[:, 0]),
    ]
    for component, axis, value in cross:
        rows.append(rows0 + component)
        cols.append(6 * cells + 3 + axis)
        vals.append(sign * value)
    return rows, cols, vals


def build_constraints(trace):
    """M with 3 rows per shared facelet and 3 per wall facelet, 6 N_c columns."""
    import hybrid_voronoi_trace as hvt

    n_cells = int(trace.n_cells)
    owner = np.asarray(trace.face_owner, dtype=np.int64)
    neigh = np.asarray(trace.face_neigh, dtype=np.int64)
    owner_r = hvt._minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_owner],
        trace.domain_length_x, trace.periodic_x,
    )
    neigh_r = hvt._minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_neigh],
        trace.domain_length_x, trace.periodic_x,
    )
    wall_cell = np.asarray(trace.wall_cell, dtype=np.int64)
    wall_r = hvt._minimum_image_x(
        trace.wall_centroid - trace.cell_centroid[trace.wall_cell],
        trace.domain_length_x, trace.periodic_x,
    )

    n_shared = owner.size
    n_wall = wall_cell.size
    rows, cols, vals = [], [], []
    base = 3 * np.arange(n_shared, dtype=np.int64)
    for pack in (rigid_blocks(base, owner, owner_r, +1.0),
                 rigid_blocks(base, neigh, neigh_r, -1.0)):
        rows += pack[0]; cols += pack[1]; vals += pack[2]
    base_wall = 3 * n_shared + 3 * np.arange(n_wall, dtype=np.int64)
    pack = rigid_blocks(base_wall, wall_cell, wall_r, +1.0)
    rows += pack[0]; cols += pack[1]; vals += pack[2]

    M = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * (n_shared + n_wall), 6 * n_cells),
    ).tocsr()
    return M, n_shared, n_wall


def nullity(M, k=6, shift=1.0e-10):
    """Smallest eigenvalues of M^T M by shift-invert, and the implied nullity."""
    G = (M.T @ M).tocsc()
    n = G.shape[0]
    scale = float(abs(G).max())
    identity = sparse.identity(n, format="csc")
    shifted = (G + shift * scale * identity).tocsc()
    try:
        factor = splu(shifted)
        operator = sparse.linalg.LinearOperator(
            (n, n), matvec=factor.solve, dtype=np.float64
        )
        inverse_values = eigsh(operator, k=min(k, n - 2), which="LM",
                               return_eigenvectors=False, maxiter=20000)
        values = np.sort(1.0 / inverse_values - shift * scale)
        method = "shift_invert_on_MtM"
    except Exception as error:
        values = np.sort(eigsh(G, k=min(k, n - 2), which="SA",
                               return_eigenvectors=False, maxiter=50000, tol=1e-12))
        method = "smallest_algebraic_on_MtM (%s)" % type(error).__name__
    tolerance = max(n, M.shape[0]) * np.finfo(np.float64).eps * scale
    return values, int(np.count_nonzero(values <= tolerance)), tolerance, scale, method


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reproduce/figure_06/refinement/runs")
    parser.add_argument("--levels", default="M0,M1,M2,M3,M4,M5")
    parser.add_argument("--out", default="reproduce/supplementary/s2_discretization")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = tuple(args.levels.split(","))
    boot = ec.bootstrap(out / "_kernel_exact")
    ns, cfg = boot["ns"], boot["cfg"]

    rows = []
    for row_path in sorted(Path(args.root).glob("*/*/row.json")):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if not row["level_label"].startswith(wanted):
            continue
        with np.load(row_path.parent / "sites_and_partition.npz") as data:
            sites = np.asarray(data["seed_flat"], dtype=np.int64)
        started = time.perf_counter()
        geom, _meta, _t = ec.build_geometry_from_seed_flat(
            sites, mask_path=ec.case_paths(row["case"])["mask"],
            seed_spec="kernel_exact:" + row["level_label"],
        )
        trace = boot["build_hybrid_trace_geometry"](
            ns, geom, cfg, trace_basis=str(ec.FORWARD_ARGS["trace_basis"])
        )
        M, n_shared, n_wall = build_constraints(trace)
        values, defect, tolerance, scale, method = nullity(M)
        record = {
            "protocol_id": PROTOCOL_ID, "case": row["case"],
            "level_label": row["level_label"], "N_cv": int(row["N_cv"]),
            "h_S_over_h": row.get("h_S_over_h"),
            "rigid_unknowns": int(6 * trace.n_cells),
            "constraint_rows": int(M.shape[0]),
            "shared_facelets": int(n_shared), "wall_facelets": int(n_wall),
            "MtM_scale": scale, "rank_tolerance": tolerance,
            "smallest_eigenvalues_MtM": [float(v) for v in values],
            "kernel_dimension": int(defect),
            "ker_A_trivial": bool(defect == 0),
            "spectral_method": method,
            "wall_seconds": float(time.perf_counter() - started),
        }
        rows.append(record)
        print(
            "[kerA] %-16s %-10s N_c=%5d  6N_c=%6d rows=%7d  lambda_min(M^T M)=%.4e  "
            "tol=%.2e  dim ker A = %d  (%.0fs)"
            % (record["case"], record["level_label"], record["N_cv"],
               record["rigid_unknowns"], record["constraint_rows"],
               values[0], tolerance, defect, record["wall_seconds"]),
            flush=True,
        )
        del geom, trace, M
        ec.free_gpu()

    if rows:
        flat = []
        for r in rows:
            item = dict(r)
            item["smallest_eigenvalues_MtM"] = json.dumps(r["smallest_eigenvalues_MtM"])
            flat.append(item)
        ec.write_csv(out / "kernel_exact.csv", flat, sorted({k for r in flat for k in r}))
        ec.write_json(out / "kernel_exact.json", rows)
        trivial = sum(1 for r in rows if r["ker_A_trivial"])
        print("[write] " + str(out / "kernel_exact.csv"))
        print("ker A is trivial on %d of %d complexes" % (trivial, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
