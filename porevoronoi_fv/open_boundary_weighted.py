"""Open-boundary (reservoir-cell) complexes on top of the unchanged modules geodesic_face_operator,
hybrid_voronoi_trace, pore_ownership and velocity_recovery.

An image block is padded by a one-voxel shell of artificial fluid that forms ONE extra cell R with the largest id.
R is skipped in assembly (no viscous term, no stabilisation, no body force), and its mass row is the row the
hybrid_voronoi_trace MINRES path removes (hybrid_voronoi_trace.py:1341). Block cells therefore exchange flux with
the exterior through boundary trace modes that are constrained from the block side only, and the exterior pressure
is the reference (p_R = 0 before hybrid_voronoi_trace centres p). Because every facelet appears in exactly two rows
of the full D, D^T 1 = 0 and the centring does not change D^T p. With open=False the builder reproduces the closed
(optionally x-periodic) complex of cell_complex / pore_ownership.

assemble() is hybrid_voronoi_trace.assemble_moment_constrained_hybrid_stokes (hybrid_voronoi_trace.py:1102-1269)
copied line for line, with one change: cells in `skip` get their recovery and divergence rows but contribute no
local matrix and no body force.

This module is open_boundary.py with one further change: assemble() takes an optional per-cell stabilisation
multiplier tau_cell, so that cell i is weighted by tau_i * 2 nu A / h instead of 2 nu A / h.  tau_cell=None and
tau_cell=1 must reproduce open_boundary.assemble BITWISE; a bitwise check on a drainage block-window confirmed
this (check script not released). Nothing else differs from open_boundary.py.
"""
from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy import sparse

import hybrid_voronoi_trace
import pore_ownership
import velocity_recovery


def _edges(labels, nc, periodic_x):
    keys = []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(lo)], labels[tuple(hi)]
        ok = (a >= 0) & (b >= 0) & (a != b)
        keys.append(np.minimum(a[ok], b[ok]).astype(np.int64) * nc + np.maximum(a[ok], b[ok]))
    if periodic_x:
        a, b = labels[:, :, -1], labels[:, :, 0]
        ok = (a >= 0) & (b >= 0) & (a != b)
        keys.append(np.minimum(a[ok], b[ok]).astype(np.int64) * nc + np.maximum(a[ok], b[ok]))
    e = np.unique(np.concatenate(keys))
    return e // nc, e % nc


def build(mask_block, site_flats, *, h=1.0, basis="connected_p1", open_faces=True, periodic_x=False):
    """mask_block: bool (Z, Y, X); site_flats: sorted unique flat indices into mask_block, all inside the mask."""
    t0 = time.perf_counter()
    site_flats = np.asarray(site_flats, np.int64)
    if not np.array_equal(site_flats, np.unique(site_flats)):
        raise ValueError("sites must be sorted unique flats")
    labels_b, dist_b = pore_ownership.exact_owners(mask_block, site_flats, h)
    t_own = time.perf_counter() - t0
    nb = int(labels_b.max()) + 1
    if open_faces:
        if periodic_x:
            raise ValueError("open faces and periodic_x are exclusive")
        shape = tuple(s + 2 for s in mask_block.shape)
        inner = (slice(1, -1),) * 3
        mask = np.ones(shape, bool)
        mask[inner] = mask_block
        labels = np.full(shape, nb, np.int32)
        labels[inner] = labels_b
        dist = np.zeros(shape)
        dist[inner] = dist_b
        nc, off, res = nb + 1, 1, nb
    else:
        mask, labels, dist, nc, off, res = mask_block, labels_b, dist_b, nb, 0, None
    pore = np.flatnonzero(mask.ravel())
    coords = np.column_stack(np.unravel_index(pore, mask.shape))[:, ::-1]
    xyz = (coords + 0.5) * h
    own = labels.ravel()[pore]
    counts = np.bincount(own, minlength=nc)
    centroid = np.column_stack([np.bincount(own, weights=xyz[:, a], minlength=nc) / counts for a in range(3)])
    owner, neigh = _edges(labels, nc, periodic_x)
    geom = SimpleNamespace(mask=mask, labels=labels, dist=dist, n_cells=nc, volume=counts * h ** 3,
                           centroid=centroid, owner=owner, neigh=neigh)
    t1 = time.perf_counter()
    trace = hybrid_voronoi_trace.build_hybrid_trace_geometry({"cp": velocity_recovery.HostAPI()}, geom,
                                                             SimpleNamespace(voxel_size=h, periodic_x=periodic_x),
                                                             trace_basis=basis)
    site_cell = labels_b.ravel()[site_flats]
    return dict(geom=geom, trace=trace, pore=pore, xyz=xyz, n_block=nb, reservoir=res, offset=off, h=h,
                shape_block=tuple(mask_block.shape), site_cell=site_cell,
                each_site_owns_its_cell=bool(np.array_equal(np.sort(site_cell), np.arange(nb))),
                timing=dict(ownership_s=t_own, trace_geometry_s=time.perf_counter() - t1),
                sizes=dict(n_cells=nc, n_block_cells=nb, n_facelets=int(trace.n_facelets),
                           n_modes=int(trace.n_trace_modes), n_dofs=int(3 * trace.n_trace_modes),
                           n_wall=int(trace.wall_cell.size), block_voxels=int(mask_block.sum())))


def assemble(trace, *, viscosity, body_force, skip=(), viscous_form="symmetric_gradient", tau_cell=None):
    """hybrid_voronoi_trace.assemble_moment_constrained_hybrid_stokes with `skip` cells contributing no local matrix /
    body force.

    tau_cell: None (== 1 everywhere) or an array of length trace.n_cells multiplying the cell stabilisation
    weight only.  Nothing else in the assembly depends on it."""
    start_time = time.perf_counter()
    skip = set(int(s) for s in skip)
    tau_vec = None if tau_cell is None else np.asarray(tau_cell, dtype=np.float64).reshape(int(trace.n_cells))
    area = trace.voxel_size * trace.voxel_size
    n_modes = trace.n_trace_modes
    n_trace_dofs = 3 * n_modes
    body_force = np.asarray(body_force, dtype=np.float64).reshape(3)
    mi = hybrid_voronoi_trace._minimum_image_x
    owner_r = mi(trace.face_centroid - trace.cell_centroid[trace.face_owner], trace.domain_length_x, trace.periodic_x)
    neigh_r = mi(trace.face_centroid - trace.cell_centroid[trace.face_neigh], trace.domain_length_x, trace.periodic_x)
    wall_r = mi(trace.wall_centroid - trace.cell_centroid[trace.wall_cell], trace.domain_length_x, trace.periodic_x)
    n_facelets = trace.n_facelets
    incident_cell = np.concatenate([trace.face_owner, trace.face_neigh, trace.wall_cell])
    incident_face = np.concatenate([np.arange(n_facelets, dtype=np.int32), np.arange(n_facelets, dtype=np.int32),
                                    np.full(trace.wall_cell.size, -1, dtype=np.int32)])
    incident_normal = np.concatenate([trace.face_normal, -trace.face_normal, trace.wall_normal], axis=0)
    incident_r = np.concatenate([owner_r, neigh_r, wall_r], axis=0)
    order = np.argsort(incident_cell, kind="stable")
    incident_cell = incident_cell[order]
    incident_face = incident_face[order]
    incident_normal = incident_normal[order]
    incident_r = incident_r[order]
    cell_start = np.searchsorted(incident_cell, np.arange(trace.n_cells + 1))
    matrix_rows, matrix_cols, matrix_values = [], [], []
    divergence_rows, divergence_cols, divergence_values = [], [], []
    body_force_matrix = np.zeros((n_trace_dofs, 3), dtype=np.float64)
    cell_recovery = []
    for cell in range(trace.n_cells):
        first = int(cell_start[cell])
        last = int(cell_start[cell + 1])
        cell_faces = incident_face[first:last]
        cell_normals = incident_normal[first:last]
        cell_r = incident_r[first:last]
        mode_set = set()
        for facelet in cell_faces[cell_faces >= 0]:
            for mode in trace.face_mode_ids[int(facelet)]:
                if mode >= 0:
                    mode_set.add(int(mode))
        local_modes = np.asarray(sorted(mode_set), dtype=np.int32)
        if local_modes.size == 0:
            cell_recovery.append((np.empty(0, dtype=np.int64), np.zeros((3, 0))))
            continue
        local_index = {int(mode): index for index, mode in enumerate(local_modes)}
        n_local_dofs = 3 * int(local_modes.size)
        recovery = np.zeros((3, n_local_dofs), dtype=np.float64)
        gradient_basis = np.zeros((n_local_dofs, 3, 3), dtype=np.float64)
        divergence = np.zeros(n_local_dofs, dtype=np.float64)
        volume = float(trace.cell_volume[cell])
        for facelet, normal, position in zip(cell_faces, cell_normals, cell_r):
            if facelet < 0:
                continue
            ids = trace.face_mode_ids[int(facelet)]
            values = trace.face_mode_values[int(facelet)]
            for mode, value in zip(ids, values):
                if mode < 0:
                    continue
                local_mode = local_index[int(mode)]
                block = slice(3 * local_mode, 3 * local_mode + 3)
                recovery[:, block] += (area * value / volume) * np.outer(position, normal)
                divergence[block] += area * value * normal
                for component in range(3):
                    gradient_basis[3 * local_mode + component, component, :] += (area * value / volume) * normal
        global_dofs = (3 * local_modes[:, None] + np.arange(3, dtype=np.int64)[None, :]).reshape(-1)
        if cell not in skip:
            if viscous_form == "full_gradient":
                local_matrix = viscosity * volume * np.einsum("aij,bij->ab", gradient_basis, gradient_basis)
                stabilization_weight = viscosity * area / trace.voxel_size
                if tau_vec is not None:
                    stabilization_weight = stabilization_weight * float(tau_vec[cell])
            elif viscous_form == "symmetric_gradient":
                symmetric_basis = 0.5 * (gradient_basis + np.transpose(gradient_basis, (0, 2, 1)))
                local_matrix = 2.0 * viscosity * volume * np.einsum("aij,bij->ab", symmetric_basis, symmetric_basis)
                stabilization_weight = 2.0 * viscosity * area / trace.voxel_size
                if tau_vec is not None:
                    stabilization_weight = stabilization_weight * float(tau_vec[cell])
            else:
                raise ValueError(f"Unsupported viscous form: {viscous_form}")
            for facelet, position in zip(cell_faces, cell_r):
                trace_map = np.zeros((3, n_local_dofs), dtype=np.float64)
                if facelet >= 0:
                    ids = trace.face_mode_ids[int(facelet)]
                    values = trace.face_mode_values[int(facelet)]
                    for mode, value in zip(ids, values):
                        if mode < 0:
                            continue
                        local_mode = local_index[int(mode)]
                        block = slice(3 * local_mode, 3 * local_mode + 3)
                        trace_map[:, block] += value * np.eye(3)
                gradient_at_face = np.einsum("aij,j->ia", gradient_basis, position)
                residual_map = trace_map - recovery - gradient_at_face
                local_matrix += stabilization_weight * (residual_map.T @ residual_map)
            matrix_rows.append(np.repeat(global_dofs, n_local_dofs))
            matrix_cols.append(np.tile(global_dofs, n_local_dofs))
            matrix_values.append(local_matrix.reshape(-1))
            body_force_matrix[global_dofs] += volume * recovery.T
        nonzero = np.flatnonzero(divergence)
        divergence_rows.extend([cell] * int(nonzero.size))
        divergence_cols.extend(global_dofs[nonzero].tolist())
        divergence_values.extend(divergence[nonzero].tolist())
        cell_recovery.append((global_dofs, recovery))
    rows = np.concatenate(matrix_rows)
    cols = np.concatenate(matrix_cols)
    values = np.concatenate(matrix_values)
    velocity_matrix = sparse.coo_matrix((values, (rows, cols)), shape=(n_trace_dofs, n_trace_dofs)).tocsr()
    velocity_matrix = (0.5 * (velocity_matrix + velocity_matrix.T)).tocsr()
    divergence_matrix = sparse.coo_matrix((divergence_values, (divergence_rows, divergence_cols)),
                                          shape=(trace.n_cells, n_trace_dofs)).tocsr()
    rhs_trace = body_force_matrix @ body_force
    return hybrid_voronoi_trace.HybridTraceSystem(
        trace_geometry=trace, velocity_matrix=velocity_matrix,
        divergence_matrix=divergence_matrix, body_force_matrix=body_force_matrix,
        rhs_trace=rhs_trace, cell_recovery=tuple(cell_recovery), viscosity=float(viscosity),
        body_force=body_force, viscous_form=viscous_form,
        assembly_time_s=float(time.perf_counter() - start_time))


def solve(system, A, b, *, rtol=1e-10, maxiter=200000, refinement_steps=1, solver="minres"):
    """Solve min 0.5 z'Az - b'z s.t. D z = 0 through the unmodified hybrid_voronoi_trace factorisation/solve
    (as stokes_solve.paper_solve)."""
    nz = A.shape[0]
    solved = replace(system, velocity_matrix=A.tocsr(), rhs_trace=b,
                     body_force_matrix=np.column_stack([b, np.zeros(nz), np.zeros(nz)]),
                     body_force=np.array([1.0, 0.0, 0.0]))
    t0 = time.perf_counter()
    fac = hybrid_voronoi_trace.build_hybrid_trace_factorization(solved, solver=solver, iterative_rtol=rtol,
                                                                iterative_maxiter=maxiter,
                                                                iterative_refinement_steps=refinement_steps)
    t1 = time.perf_counter()
    r = hybrid_voronoi_trace.solve_moment_constrained_hybrid_stokes(solved, factorization=fac)
    receipt = {k: r[k] for k in ("linear_solver", "linear_solver_iterations", "linear_solver_info",
                                 "linear_solver_relative_residual", "mass_inf_per_volume", "n_trace_dofs")
               if k in r}
    receipt.update(factorization_s=t1 - t0, solve_s=time.perf_counter() - t1)
    return r["trace_coefficients"].ravel().copy(), r, receipt


def to_phys(p_zyx, part):
    """Block voxel coordinates (z, y, x; voxel centres at integers) -> physical xyz of the (padded) complex."""
    p = np.asarray(p_zyx, np.float64) + part["offset"]
    return (p[:, ::-1] + 0.5) * part["h"]


def owners_of(p_zyx, part):
    """Owner cell of each point by the voxel it rounds to; -1 if that voxel is not a block-domain voxel."""
    c = np.rint(np.asarray(p_zyx, np.float64)).astype(np.int64)
    Z, Y, X = part["shape_block"]
    ok = np.all((c >= 0) & (c < np.array([Z, Y, X])), axis=1)
    out = np.full(c.shape[0], -1, np.int64)
    lab = part["geom"].labels
    o = part["offset"]
    out[ok] = lab[c[ok, 0] + o, c[ok, 1] + o, c[ok, 2] + o]
    if part["reservoir"] is not None:
        out[out == part["reservoir"]] = -1
    return out


def gamma_of(trace, alpha, nu, block_volume, reservoir=None):
    v = trace.cell_volume.copy()
    if reservoir is not None:
        v[reservoir] = 0.0
    return float(alpha) * nu * float(v.sum()) / (float(block_volume) ** (2.0 / 3.0))
