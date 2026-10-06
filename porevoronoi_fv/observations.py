"""Observation selectors, observation operator and observation weight.

Declared selection rule (both partitions below): within each cell, among the partition's records whose paper-rule
voxel is owned by the cell, keep the record whose physical position is nearest the cell's volume centroid (Euclidean,
minimum image in x); ties broken by the lower record index (CSV row order). Exactly one record per cell is asserted.
  all records (pub): partition = all records of the window ('pub'); one record per cell.
  single frame (frm:F): partition = one frame ('frm:F'); sites are that frame's occupied voxels (paper site rule),
            one record per cell, and every selected record comes from a distinct particle (asserted).
Weight: gamma = alpha nu sum_i V_i / (mask.size h^3)^(2/3), per-record weight 1/N.
Observation operator: velocity_recovery.point_operator (R row + G (x - c), minimum image in x).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy import sparse

import hybrid_voronoi_trace as hvt
import velocity_recovery as rp


def nearest_centroid_selection(case, part, rec_idx):
    geom, trace = part["geom"], part["trace"]
    rec_idx = np.asarray(rec_idx, dtype=np.int64)
    flat = case.records["flat"][rec_idx]
    owner = geom.labels.ravel()[flat].astype(np.int64)
    if np.any(owner < 0):
        raise RuntimeError("record voxel without owner")
    delta = hvt._minimum_image_x(case.records["xyz"][rec_idx] - geom.centroid[owner], trace.domain_length_x,
                                 trace.periodic_x)
    d2 = np.einsum("ij,ij->i", delta, delta)
    order = np.lexsort((rec_idx, d2, owner))
    so = owner[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = so[1:] != so[:-1]
    chosen = order[first]
    sel_rec, sel_owner = rec_idx[chosen], owner[chosen]
    counts = np.bincount(sel_owner, minlength=geom.n_cells)
    if not np.all(counts == 1):
        raise AssertionError(f"one record per cell violated: min {counts.min()} max {counts.max()}")
    by_cell = np.empty(geom.n_cells, dtype=np.int64)
    by_cell[sel_owner] = sel_rec
    return dict(record=by_cell, cell=np.arange(geom.n_cells), d2=d2[chosen][np.argsort(sel_owner)],
                candidates_per_cell=np.bincount(owner, minlength=geom.n_cells))


def brute_force_selection(case, part, rec_idx):
    """Independent per-cell loop implementation of the declared rule (used only to test the vectorised one)."""
    geom = part["geom"]
    L = part["trace"].domain_length_x
    best = {}
    for r in np.asarray(rec_idx, dtype=np.int64).tolist():
        c = int(geom.labels.ravel()[case.records["flat"][r]])
        d = case.records["xyz"][r] - geom.centroid[c]
        dx = d[0] - L * np.round(d[0] / L)
        d2 = dx * dx + d[1] * d[1] + d[2] * d[2]
        cur = best.get(c)
        if cur is None or d2 < cur[0] or (d2 == cur[0] and r < cur[1]):
            best[c] = (d2, r)
    out = np.full(geom.n_cells, -1, dtype=np.int64)
    for c, (_, r) in best.items():
        out[c] = r
    return out


def particle_stats(case, part, rec_idx, selection):
    geom = part["geom"]
    sel = selection["record"]
    pid = case.records["particle"][sel]
    uniq, cnt = np.unique(pid, return_counts=True)
    cells_with_unique_particle = int(np.sum(cnt[np.searchsorted(uniq, pid)] == 1))
    rec_idx = np.asarray(rec_idx, dtype=np.int64)
    owner = geom.labels.ravel()[case.records["flat"][rec_idx]].astype(np.int64)
    parts_all = case.records["particle"][rec_idx]
    pu, pinv = np.unique(parts_all, return_inverse=True)
    B = sparse.coo_matrix((np.ones(owner.size), (owner, pinv)), shape=(geom.n_cells, pu.size)).tocsr()
    B.data[:] = 1
    match = maximum_bipartite_matching(B, perm_type="column")
    return dict(n_cells=int(geom.n_cells), n_records_in_partition=int(rec_idx.size),
                n_particles_in_partition=int(pu.size), distinct_particles_among_selected=int(uniq.size),
                cells_whose_selected_particle_is_unique=cells_with_unique_particle,
                max_cells_with_distinct_particle_matching=int(np.sum(match >= 0)),
                candidates_per_cell_median=float(np.median(selection["candidates_per_cell"])),
                candidates_per_cell_max=int(selection["candidates_per_cell"].max()))


def gamma_of(case, trace, alpha):
    # gamma = alpha * nu * sum_i V_i / (image volume)^(2/3)
    return float(alpha) * case.nu * float(trace.cell_volume.sum()) / ((case.mask.size * case.h ** 3) ** (2 / 3))


def observations(case, part, R, G, selection):
    sel = selection["record"]
    xyz = case.records["xyz"][sel]
    owners = selection["cell"]
    H = rp.point_operator(xyz, owners, part["trace"], R, G)
    y = case.records["vel"][sel]
    return H, y, xyz


def point_operator_defect(H, xyz, owners, trace, R, G, seed=9):
    """Independent evaluation of the affine point map at a random trace vector."""
    probe = np.random.default_rng(seed).normal(size=H.shape[1])
    U = (R @ probe).reshape(-1, 3)
    grad = (G @ probe).reshape(-1, 3, 3)
    delta = hvt._minimum_image_x(xyz - trace.cell_centroid[owners], trace.domain_length_x, trace.periodic_x)
    direct = U[owners] + np.einsum("nij,nj->ni", grad[owners], delta)
    return float(np.max(np.abs(H @ probe - direct.ravel())))


def assisted_operator(system, H, y, gamma, weight):
    """(A, b) of min 0.5 z'Az - b'z + (gamma w / 2)||Hz - y||^2 s.t. Dz = 0. gamma = 0 returns the forward objects."""
    if gamma == 0.0:
        return system.velocity_matrix, system.rhs_trace, dict(pure=True, gamma=0.0, weight=weight)
    if H is None:
        raise ValueError("assisted operator needs observations")
    gram = (H.T @ H).tocsr()
    rhs = np.asarray(H.T @ np.asarray(y).ravel()).ravel()
    A = (system.velocity_matrix + gamma * weight * gram).tocsr()
    b = system.rhs_trace + gamma * weight * rhs
    return A, b, dict(pure=False, gamma=float(gamma), weight=float(weight), gram_nnz=int(gram.nnz))
