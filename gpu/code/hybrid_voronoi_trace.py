from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg, minres, splu

from geodesic_face_operator import (
    SIGNED_NORMAL_GROUP_VECTORS,
    _axis_signed_facelet_arrays,
    _periodic_x_signed_facelet_arrays,
)


TraceBasis = Literal["connected_p0", "connected_p1", "facelet_p0"]
ViscousForm = Literal["full_gradient", "symmetric_gradient"]
ProjectionSolver = Literal["auto", "direct_lu", "projected_cg"]
ForwardLinearSolver = Literal["direct_lu", "minres", "schur_cg"]


@dataclass(frozen=True)
class HybridTraceGeometry:
    n_cells: int
    voxel_size: float
    periodic_x: bool
    domain_length_x: float
    cell_volume: np.ndarray
    cell_centroid: np.ndarray
    face_key: np.ndarray
    face_group_id: np.ndarray
    face_owner: np.ndarray
    face_neigh: np.ndarray
    face_normal: np.ndarray
    face_centroid: np.ndarray
    face_r_owner_g: np.ndarray
    face_r_neigh_g: np.ndarray
    face_parent_edge: np.ndarray
    face_patch: np.ndarray
    patch_component_key: np.ndarray
    face_mode_ids: np.ndarray
    face_mode_values: np.ndarray
    mode_patch: np.ndarray
    wall_cell: np.ndarray
    wall_normal: np.ndarray
    wall_centroid: np.ndarray
    wall_r_g: np.ndarray
    stored_edge_sign_from_sorted: np.ndarray
    trace_basis: str

    @property
    def n_facelets(self) -> int:
        return int(self.face_key.size)

    @property
    def n_patches(self) -> int:
        return int(self.patch_component_key.size)

    @property
    def n_trace_modes(self) -> int:
        return int(self.mode_patch.size)


@dataclass(frozen=True)
class HybridTraceSystem:
    trace_geometry: HybridTraceGeometry
    velocity_matrix: sparse.csr_matrix
    divergence_matrix: sparse.csr_matrix
    body_force_matrix: np.ndarray
    rhs_trace: np.ndarray
    cell_recovery: tuple[tuple[np.ndarray, np.ndarray], ...]
    viscosity: float
    body_force: np.ndarray
    viscous_form: str
    assembly_time_s: float


@dataclass(frozen=True)
class HybridTraceFactorization:
    system: HybridTraceSystem
    kkt_matrix: sparse.spmatrix | None
    lu: Any
    solver_mode: str
    factorization_ordering: str
    preconditioner: LinearOperator | None
    reduced_divergence: sparse.csr_matrix | None
    pressure_operator: LinearOperator | None
    iterative_rtol: float
    iterative_maxiter: int
    iterative_refinement_steps: int
    solver_storage_nnz: int
    solver_storage_bytes: int
    factorization_time_s: float


@dataclass(frozen=True)
class HybridTraceProjectionOperators:
    trace_geometry: HybridTraceGeometry
    face_trace_matrix: sparse.csr_matrix
    divergence_matrix: sparse.csr_matrix
    cell_recovery_matrix: sparse.csr_matrix
    mean_velocity_matrix: sparse.csr_matrix
    mass_diagonal: np.ndarray
    mass_orthogonality_defect: float
    assembly_time_s: float


@dataclass(frozen=True)
class HybridTraceProjectionFactorization:
    operators: HybridTraceProjectionOperators
    constraint_matrix: sparse.csr_matrix
    pressure_kkt_matrix: sparse.csc_matrix | None
    pressure_matrix: sparse.csr_matrix | None
    pressure_component_count: int
    pressure_component_label: np.ndarray
    preserved_mean_components: tuple[int, ...]
    lu: Any
    solver_mode: str
    mean_pressure_coupling: np.ndarray | None
    mean_mass_matrix: np.ndarray | None
    mean_pressure_solution: np.ndarray | None
    reduced_mean_matrix: np.ndarray | None
    iterative_rtol: float
    iterative_maxiter: int
    setup_iteration_count_total: int
    setup_iteration_count_max: int
    setup_internal_relative_residual_max: float
    solver_storage_nnz: int
    solver_storage_bytes: int
    factorization_time_s: float


def _host(cp: Any, value: Any, dtype: Any | None = None) -> np.ndarray:
    array = cp.asnumpy(value) if hasattr(cp, "asnumpy") else np.asarray(value)
    return np.asarray(array, dtype=dtype)


def _minimum_image_x(delta: np.ndarray, domain_length_x: float, periodic_x: bool) -> np.ndarray:
    result = np.asarray(delta, dtype=np.float64).copy()
    if periodic_x and domain_length_x > 0.0:
        result[..., 0] -= domain_length_x * np.round(result[..., 0] / domain_length_x)
    return result


def _connected_facelet_patches(
    component_key: np.ndarray,
    group_id: np.ndarray,
    centroid: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a signed-normal group into coplanar edge-connected patches."""
    n_facelets = int(component_key.size)
    if n_facelets == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    scaled = centroid / float(voxel_size)
    normal_axis = (np.asarray(group_id, dtype=np.int64) // 2).astype(np.int64, copy=False)
    face_index = np.arange(n_facelets, dtype=np.int64)
    plane = np.rint(scaled[face_index, normal_axis]).astype(np.int64)
    tangential_first_axis = np.choose(normal_axis, [1, 0, 0])
    tangential_second_axis = np.choose(normal_axis, [2, 2, 1])
    first = np.floor(scaled[face_index, tangential_first_axis]).astype(np.int64)
    second = np.floor(scaled[face_index, tangential_second_axis]).astype(np.int64)
    records = np.rec.fromarrays(
        [component_key.astype(np.int64), normal_axis, plane, first, second],
        names="component,axis,plane,first,second",
    )
    order = np.argsort(records, kind="stable")
    sorted_records = records[order]
    if np.any(sorted_records[1:] == sorted_records[:-1]):
        raise RuntimeError("Connected facelet patch keys are not unique")

    edge_owner: list[np.ndarray] = []
    edge_neigh: list[np.ndarray] = []
    for field in ("first", "second"):
        query = sorted_records.copy()
        query[field] += 1
        position = np.searchsorted(sorted_records, query)
        valid = position < n_facelets
        safe_position = np.minimum(position, n_facelets - 1)
        valid &= sorted_records[safe_position] == query
        if np.any(valid):
            edge_owner.append(order[valid].astype(np.int32, copy=False))
            edge_neigh.append(order[position[valid]].astype(np.int32, copy=False))

    if edge_owner:
        owner = np.concatenate(edge_owner)
        neigh = np.concatenate(edge_neigh)
        adjacency = sparse.coo_matrix(
            (
                np.ones(2 * owner.size, dtype=np.int8),
                (
                    np.concatenate([owner, neigh]),
                    np.concatenate([neigh, owner]),
                ),
            ),
            shape=(n_facelets, n_facelets),
        ).tocsr()
        patch_count, face_patch = connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
    else:
        patch_count = n_facelets
        face_patch = np.arange(n_facelets, dtype=np.int32)
    first_facelet = np.full(int(patch_count), n_facelets, dtype=np.int64)
    np.minimum.at(first_facelet, face_patch, face_index)
    patch_component_key = component_key[first_facelet]
    return face_patch.astype(np.int32, copy=False), patch_component_key.astype(np.int64, copy=False)


def _trace_modes(
    trace_basis: TraceBasis,
    face_patch: np.ndarray,
    patch_component_key: np.ndarray,
    face_group_id: np.ndarray,
    face_centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_facelets = int(face_patch.size)
    if trace_basis == "facelet_p0":
        face_mode_ids = np.full((n_facelets, 3), -1, dtype=np.int32)
        face_mode_values = np.zeros((n_facelets, 3), dtype=np.float64)
        face_mode_ids[:, 0] = np.arange(n_facelets, dtype=np.int32)
        face_mode_values[:, 0] = 1.0
        return face_mode_ids, face_mode_values, np.arange(n_facelets, dtype=np.int32)

    patch_count = int(patch_component_key.size)
    patch = np.asarray(face_patch, dtype=np.int64)
    counts = np.bincount(patch, minlength=patch_count).astype(np.float64)
    if np.any(counts <= 0.0):
        raise RuntimeError("Trace patch numbering contains an empty patch")
    face_mode_ids = np.full((n_facelets, 3), -1, dtype=np.int32)
    face_mode_values = np.zeros((n_facelets, 3), dtype=np.float64)
    face_mode_ids[:, 0] = patch.astype(np.int32, copy=False)
    face_mode_values[:, 0] = 1.0
    if trace_basis == "connected_p0":
        return face_mode_ids, face_mode_values, np.arange(patch_count, dtype=np.int32)

    normal_axis = (np.asarray(face_group_id, dtype=np.int64) // 2).astype(np.int64, copy=False)
    tangential_first_axis = np.choose(normal_axis, [1, 0, 0])
    tangential_second_axis = np.choose(normal_axis, [2, 2, 1])
    face_index = np.arange(n_facelets, dtype=np.int64)
    first_coordinate = face_centroid[face_index, tangential_first_axis]
    second_coordinate = face_centroid[face_index, tangential_second_axis]
    first_sum = np.bincount(patch, weights=first_coordinate, minlength=patch_count)
    second_sum = np.bincount(patch, weights=second_coordinate, minlength=patch_count)
    first_vector = first_coordinate - (first_sum / counts)[patch]
    second_vector = second_coordinate - (second_sum / counts)[patch]
    first_norm2 = np.bincount(patch, weights=first_vector**2, minlength=patch_count)
    first_coordinate_norm2 = np.bincount(
        patch,
        weights=first_coordinate**2,
        minlength=patch_count,
    )
    tolerance2 = (32.0 * np.finfo(np.float64).eps) ** 2
    retain_first = first_norm2 > tolerance2 * np.maximum(first_coordinate_norm2, 1.0)
    first_scale = np.zeros(patch_count, dtype=np.float64)
    first_scale[retain_first] = np.sqrt(counts[retain_first] / first_norm2[retain_first])
    first_basis = first_vector * first_scale[patch]

    cross = np.bincount(
        patch,
        weights=first_vector * second_vector,
        minlength=patch_count,
    )
    projection = np.zeros(patch_count, dtype=np.float64)
    projection[retain_first] = cross[retain_first] / first_norm2[retain_first]
    second_vector = second_vector - projection[patch] * first_vector
    second_norm2 = np.bincount(patch, weights=second_vector**2, minlength=patch_count)
    second_coordinate_norm2 = np.bincount(
        patch,
        weights=second_coordinate**2,
        minlength=patch_count,
    )
    retain_second = second_norm2 > tolerance2 * np.maximum(second_coordinate_norm2, 1.0)
    second_scale = np.zeros(patch_count, dtype=np.float64)
    second_scale[retain_second] = np.sqrt(counts[retain_second] / second_norm2[retain_second])
    second_basis = second_vector * second_scale[patch]

    first_patch = np.flatnonzero(retain_first).astype(np.int32)
    second_patch = np.flatnonzero(retain_second).astype(np.int32)
    first_mode_by_patch = np.full(patch_count, -1, dtype=np.int32)
    second_mode_by_patch = np.full(patch_count, -1, dtype=np.int32)
    first_mode_by_patch[first_patch] = patch_count + np.arange(first_patch.size, dtype=np.int32)
    second_mode_by_patch[second_patch] = (
        patch_count + first_patch.size + np.arange(second_patch.size, dtype=np.int32)
    )
    first_face = retain_first[patch]
    second_face = retain_second[patch]
    face_mode_ids[first_face, 1] = first_mode_by_patch[patch[first_face]]
    face_mode_values[first_face, 1] = first_basis[first_face]
    second_column = np.where(first_face[second_face], 2, 1)
    second_face_index = face_index[second_face]
    face_mode_ids[second_face_index, second_column] = second_mode_by_patch[patch[second_face]]
    face_mode_values[second_face_index, second_column] = second_basis[second_face]
    mode_patch = np.concatenate(
        [
            np.arange(patch_count, dtype=np.int32),
            first_patch,
            second_patch,
        ]
    )
    return face_mode_ids, face_mode_values, mode_patch


def _wall_facelets(
    mask: np.ndarray,
    labels: np.ndarray,
    distance: np.ndarray,
    voxel_size: float,
    periodic_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    physical_axis = {2: 0, 1: 1, 0: 2}

    for array_axis in (2, 1, 0):
        for sign in (-1, 1):
            if array_axis == 2 and periodic_x:
                neighbour = np.roll(mask, shift=-sign, axis=array_axis)
            else:
                neighbour = np.zeros_like(mask, dtype=bool)
                source = [slice(None), slice(None), slice(None)]
                target = [slice(None), slice(None), slice(None)]
                if sign > 0:
                    source[array_axis] = slice(1, None)
                    target[array_axis] = slice(None, -1)
                else:
                    source[array_axis] = slice(None, -1)
                    target[array_axis] = slice(1, None)
                neighbour[tuple(target)] = mask[tuple(source)]

            wall = mask & ~neighbour
            z_index, y_index, x_index = np.nonzero(wall)
            wall_cells = labels[wall]
            valid = wall_cells >= 0
            z_index = z_index[valid]
            y_index = y_index[valid]
            x_index = x_index[valid]
            wall_cells = wall_cells[valid].astype(np.int32, copy=False)

            face_centroid = np.stack(
                [
                    (x_index + 0.5) * voxel_size,
                    (y_index + 0.5) * voxel_size,
                    (z_index + 0.5) * voxel_size,
                ],
                axis=1,
            )
            axis = physical_axis[array_axis]
            face_centroid[:, axis] += sign * 0.5 * voxel_size
            normal = np.zeros((wall_cells.size, 3), dtype=np.float64)
            normal[:, axis] = float(sign)

            cells.append(wall_cells)
            normals.append(normal)
            centroids.append(face_centroid)
            distances.append(distance[wall][valid].astype(np.float64) + 0.5 * voxel_size)

    return (
        np.concatenate(cells),
        np.concatenate(normals, axis=0),
        np.concatenate(centroids, axis=0),
        np.concatenate(distances),
    )


def build_hybrid_trace_geometry(
    ns: dict[str, Any],
    geom: Any,
    cfg: Any,
    *,
    trace_basis: TraceBasis = "connected_p0",
) -> HybridTraceGeometry:
    cp = ns["cp"]
    voxel_size = float(getattr(cfg, "voxel_size", 1.0))
    periodic_x = bool(getattr(cfg, "periodic_x", False))
    n_cells = int(geom.n_cells)

    parts = [
        _axis_signed_facelet_arrays(ns, geom, array_axis, voxel_size)
        for array_axis in (2, 1, 0)
    ]
    if periodic_x:
        parts.append(_periodic_x_signed_facelet_arrays(ns, geom, voxel_size))
    parts = [part for part in parts if int(part[0].size)]
    if not parts:
        raise RuntimeError("No intercell facelets are available for the hybrid trace space")

    face_key = _host(cp, cp.concatenate([part[0] for part in parts]), np.int64)
    face_group_id = _host(cp, cp.concatenate([part[1] for part in parts]), np.int8)
    face_r_owner_g = _host(cp, cp.concatenate([part[2] for part in parts]), np.float64)
    face_r_neigh_g = _host(cp, cp.concatenate([part[3] for part in parts]), np.float64)
    face_centroid = _host(cp, cp.concatenate([part[4] for part in parts], axis=0), np.float64)
    face_owner = face_key // n_cells
    face_neigh = face_key % n_cells
    normal_lookup = np.asarray(SIGNED_NORMAL_GROUP_VECTORS, dtype=np.float64)
    face_normal = normal_lookup[face_group_id.astype(np.int32)]
    component_key = face_key * np.int64(6) + face_group_id.astype(np.int64)

    if trace_basis == "facelet_p0":
        face_patch = np.arange(face_key.size, dtype=np.int32)
        patch_component_key = component_key.copy()
    else:
        connected_patch, connected_component_key = _connected_facelet_patches(
            component_key,
            face_group_id,
            face_centroid,
            voxel_size,
        )
        face_patch = connected_patch
        patch_component_key = connected_component_key
    face_mode_ids, face_mode_values, mode_patch = _trace_modes(
        trace_basis,
        face_patch,
        patch_component_key,
        face_group_id,
        face_centroid,
    )

    stored_owner = _host(cp, geom.owner, np.int64)
    stored_neigh = _host(cp, geom.neigh, np.int64)
    edge_key = np.minimum(stored_owner, stored_neigh) * n_cells + np.maximum(
        stored_owner, stored_neigh
    )
    edge_order = np.argsort(edge_key)
    face_position = np.searchsorted(edge_key[edge_order], face_key)
    face_parent_edge = edge_order[face_position]
    if not np.array_equal(edge_key[face_parent_edge], face_key):
        raise RuntimeError("At least one hybrid facelet has no stored aggregate edge")
    stored_edge_sign = np.where(stored_owner <= stored_neigh, 1.0, -1.0)

    mask = _host(cp, geom.mask, bool)
    labels = _host(cp, geom.labels, np.int32)
    distance = _host(cp, geom.dist, np.float64)
    cell_centroid = _host(cp, geom.centroid, np.float64)
    cell_volume = _host(cp, geom.volume, np.float64)
    wall_cell, wall_normal, wall_centroid, wall_r_g = _wall_facelets(
        mask,
        labels,
        distance,
        voxel_size,
        periodic_x,
    )

    return HybridTraceGeometry(
        n_cells=n_cells,
        voxel_size=voxel_size,
        periodic_x=periodic_x,
        domain_length_x=float(labels.shape[2]) * voxel_size,
        cell_volume=cell_volume,
        cell_centroid=cell_centroid,
        face_key=face_key,
        face_group_id=face_group_id,
        face_owner=face_owner.astype(np.int32),
        face_neigh=face_neigh.astype(np.int32),
        face_normal=face_normal,
        face_centroid=face_centroid,
        face_r_owner_g=face_r_owner_g,
        face_r_neigh_g=face_r_neigh_g,
        face_parent_edge=face_parent_edge.astype(np.int32),
        face_patch=face_patch,
        patch_component_key=patch_component_key,
        face_mode_ids=face_mode_ids,
        face_mode_values=face_mode_values,
        mode_patch=mode_patch,
        wall_cell=wall_cell,
        wall_normal=wall_normal,
        wall_centroid=wall_centroid,
        wall_r_g=wall_r_g,
        stored_edge_sign_from_sorted=stored_edge_sign,
        trace_basis=trace_basis,
    )


def build_hybrid_trace_projection_operators(
    trace: HybridTraceGeometry,
) -> HybridTraceProjectionOperators:
    """Build kinematic maps for constant-cell-state conservative trace projection."""
    start_time = time.perf_counter()
    n_facelets = trace.n_facelets
    n_modes = trace.n_trace_modes
    n_trace_dofs = 3 * n_modes
    face_index = np.arange(n_facelets, dtype=np.int64)
    component = np.arange(3, dtype=np.int64)

    trace_rows: list[np.ndarray] = []
    trace_cols: list[np.ndarray] = []
    trace_values: list[np.ndarray] = []
    for column in range(trace.face_mode_ids.shape[1]):
        mode = trace.face_mode_ids[:, column]
        valid = mode >= 0
        if not np.any(valid):
            continue
        valid_face = face_index[valid]
        valid_mode = mode[valid].astype(np.int64, copy=False)
        value = trace.face_mode_values[valid, column]
        trace_rows.append((3 * valid_face[:, None] + component[None, :]).reshape(-1))
        trace_cols.append((3 * valid_mode[:, None] + component[None, :]).reshape(-1))
        trace_values.append(np.repeat(value, 3))
    face_trace = sparse.coo_matrix(
        (
            np.concatenate(trace_values),
            (np.concatenate(trace_rows), np.concatenate(trace_cols)),
        ),
        shape=(3 * n_facelets, n_trace_dofs),
    ).tocsr()

    area = trace.voxel_size * trace.voxel_size
    normal_map = sparse.coo_matrix(
        (
            (area * trace.face_normal).reshape(-1),
            (
                np.repeat(face_index, 3),
                (3 * face_index[:, None] + component[None, :]).reshape(-1),
            ),
        ),
        shape=(n_facelets, 3 * n_facelets),
    ).tocsr()
    incidence = sparse.coo_matrix(
        (
            np.concatenate([np.ones(n_facelets), -np.ones(n_facelets)]),
            (
                np.concatenate([trace.face_owner, trace.face_neigh]),
                np.concatenate([face_index, face_index]),
            ),
        ),
        shape=(trace.n_cells, n_facelets),
    ).tocsr()
    divergence = (incidence @ normal_map @ face_trace).tocsr()

    owner_r = _minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_owner],
        trace.domain_length_x,
        trace.periodic_x,
    )
    neigh_r = _minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_neigh],
        trace.domain_length_x,
        trace.periodic_x,
    )
    recovery_rows: list[np.ndarray] = []
    recovery_cols: list[np.ndarray] = []
    recovery_values: list[np.ndarray] = []
    owner_volume = np.maximum(trace.cell_volume[trace.face_owner], 1.0e-300)
    neigh_volume = np.maximum(trace.cell_volume[trace.face_neigh], 1.0e-300)
    for output_component in range(3):
        for velocity_component in range(3):
            face_dof = 3 * face_index + velocity_component
            recovery_rows.extend(
                [
                    3 * trace.face_owner.astype(np.int64) + output_component,
                    3 * trace.face_neigh.astype(np.int64) + output_component,
                ]
            )
            recovery_cols.extend([face_dof, face_dof])
            recovery_values.extend(
                [
                    area
                    * owner_r[:, output_component]
                    * trace.face_normal[:, velocity_component]
                    / owner_volume,
                    -area
                    * neigh_r[:, output_component]
                    * trace.face_normal[:, velocity_component]
                    / neigh_volume,
                ]
            )
    face_to_cell_moment = sparse.coo_matrix(
        (
            np.concatenate(recovery_values),
            (np.concatenate(recovery_rows), np.concatenate(recovery_cols)),
        ),
        shape=(3 * trace.n_cells, 3 * n_facelets),
    ).tocsr()
    cell_recovery = (face_to_cell_moment @ face_trace).tocsr()
    total_volume = max(float(np.sum(trace.cell_volume)), 1.0e-300)
    mean_selector_rows = np.repeat(np.arange(3, dtype=np.int64), trace.n_cells)
    mean_selector_cols = np.concatenate(
        [3 * np.arange(trace.n_cells, dtype=np.int64) + component_index for component_index in range(3)]
    )
    mean_selector_values = np.tile(trace.cell_volume / total_volume, 3)
    mean_selector = sparse.coo_matrix(
        (mean_selector_values, (mean_selector_rows, mean_selector_cols)),
        shape=(3, 3 * trace.n_cells),
    ).tocsr()
    mean_velocity = (mean_selector @ cell_recovery).tocsr()

    mass = (area * (face_trace.T @ face_trace)).tocsr()
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    if np.any(~np.isfinite(mass_diagonal)) or np.any(mass_diagonal <= 0.0):
        raise RuntimeError("Trace projection mass matrix has a non-positive diagonal")
    off_diagonal = mass - sparse.diags(mass_diagonal, format="csr")
    mass_norm = max(float(sparse.linalg.norm(mass)), 1.0e-300)
    orthogonality_defect = float(sparse.linalg.norm(off_diagonal) / mass_norm)
    if orthogonality_defect > 1.0e-11:
        raise RuntimeError(
            f"Trace basis is not area-orthogonal enough for diagonal projection: {orthogonality_defect}"
        )

    return HybridTraceProjectionOperators(
        trace_geometry=trace,
        face_trace_matrix=face_trace,
        divergence_matrix=divergence,
        cell_recovery_matrix=cell_recovery,
        mean_velocity_matrix=mean_velocity,
        mass_diagonal=mass_diagonal,
        mass_orthogonality_defect=orthogonality_defect,
        assembly_time_s=float(time.perf_counter() - start_time),
    )


def build_hybrid_trace_projection_factorization(
    operators: HybridTraceProjectionOperators,
    *,
    preserve_mean_components: tuple[int, ...] = (),
    solver: ProjectionSolver = "auto",
    direct_cell_limit: int = 30_000,
    iterative_rtol: float = 1.0e-11,
    iterative_maxiter: int = 20_000,
) -> HybridTraceProjectionFactorization:
    start_time = time.perf_counter()
    trace = operators.trace_geometry
    divergence = operators.divergence_matrix
    preserved_components = tuple(sorted(set(int(value) for value in preserve_mean_components)))
    if any(value < 0 or value > 2 for value in preserved_components):
        raise ValueError(f"Mean-velocity components must be in [0, 2], got {preserved_components}")
    if solver not in ("auto", "direct_lu", "projected_cg"):
        raise ValueError(f"Unknown projection solver: {solver!r}")
    if int(direct_cell_limit) <= 0:
        raise ValueError("direct_cell_limit must be positive")
    if not (0.0 < float(iterative_rtol) < 1.0):
        raise ValueError("iterative_rtol must lie strictly between zero and one")
    if int(iterative_maxiter) <= 0:
        raise ValueError("iterative_maxiter must be positive")
    mean_constraints = operators.mean_velocity_matrix[list(preserved_components)]
    constraint_matrix = sparse.vstack([divergence, mean_constraints], format="csr")
    inverse_mass_diagonal = 1.0 / operators.mass_diagonal
    inverse_mass = sparse.diags(inverse_mass_diagonal, format="csr")

    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * trace.n_facelets, dtype=np.float64),
            (
                np.concatenate([trace.face_owner, trace.face_neigh]),
                np.concatenate([trace.face_neigh, trace.face_owner]),
            ),
        ),
        shape=(trace.n_cells, trace.n_cells),
    ).tocsr()
    component_count, component_label = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    selected_solver = (
        "direct_lu"
        if solver == "auto" and trace.n_cells <= int(direct_cell_limit)
        else "projected_cg"
        if solver == "auto"
        else str(solver)
    )

    if selected_solver == "projected_cg":
        pressure_matrix = (divergence @ inverse_mass @ divergence.T).tocsr()
        pressure_matrix.sum_duplicates()
        pressure_matrix.sort_indices()
        if preserved_components:
            weighted_mean_transpose = inverse_mass @ mean_constraints.T
            mean_pressure_coupling = np.asarray(
                (divergence @ weighted_mean_transpose).toarray(),
                dtype=np.float64,
            )
            mean_mass_matrix = np.asarray(
                (mean_constraints @ weighted_mean_transpose).toarray(),
                dtype=np.float64,
            )
        else:
            mean_pressure_coupling = np.zeros((trace.n_cells, 0), dtype=np.float64)
            mean_mass_matrix = np.zeros((0, 0), dtype=np.float64)
        storage_bytes = sum(
            array.nbytes
            for array in (pressure_matrix.data, pressure_matrix.indices, pressure_matrix.indptr)
        )
        storage_bytes += int(mean_pressure_coupling.nbytes + mean_mass_matrix.nbytes)
        preliminary = HybridTraceProjectionFactorization(
            operators=operators,
            constraint_matrix=constraint_matrix,
            pressure_kkt_matrix=None,
            pressure_matrix=pressure_matrix,
            pressure_component_count=int(component_count),
            pressure_component_label=np.asarray(component_label, dtype=np.int32),
            preserved_mean_components=preserved_components,
            lu=None,
            solver_mode=selected_solver,
            mean_pressure_coupling=mean_pressure_coupling,
            mean_mass_matrix=mean_mass_matrix,
            mean_pressure_solution=None,
            reduced_mean_matrix=None,
            iterative_rtol=float(iterative_rtol),
            iterative_maxiter=int(iterative_maxiter),
            setup_iteration_count_total=0,
            setup_iteration_count_max=0,
            setup_internal_relative_residual_max=0.0,
            solver_storage_nnz=int(pressure_matrix.nnz),
            solver_storage_bytes=int(storage_bytes),
            factorization_time_s=0.0,
        )
        setup_iterations: list[int] = []
        setup_residuals: list[float] = []
        mean_pressure_solution = np.empty_like(mean_pressure_coupling)
        for column in range(mean_pressure_coupling.shape[1]):
            mean_pressure_solution[:, column], iterations, residual = _solve_projected_pressure_cg(
                preliminary,
                mean_pressure_coupling[:, column],
            )
            setup_iterations.append(iterations)
            setup_residuals.append(residual)
        reduced_mean_matrix = mean_mass_matrix - mean_pressure_coupling.T @ mean_pressure_solution
        storage_bytes += int(mean_pressure_solution.nbytes + reduced_mean_matrix.nbytes)
        return replace(
            preliminary,
            mean_pressure_solution=mean_pressure_solution,
            reduced_mean_matrix=reduced_mean_matrix,
            setup_iteration_count_total=int(sum(setup_iterations)),
            setup_iteration_count_max=int(max(setup_iterations, default=0)),
            setup_internal_relative_residual_max=float(max(setup_residuals, default=0.0)),
            solver_storage_bytes=int(storage_bytes),
            factorization_time_s=float(time.perf_counter() - start_time),
        )

    pressure_matrix = (constraint_matrix @ inverse_mass @ constraint_matrix.T).tocsr()
    component_size = np.bincount(component_label, minlength=component_count).astype(np.float64)
    gauge = sparse.coo_matrix(
        (
            1.0 / np.sqrt(component_size[component_label]),
            (np.arange(trace.n_cells, dtype=np.int64), component_label),
        ),
        shape=(trace.n_cells, component_count),
    ).tocsr()
    zero_gauge = sparse.csr_matrix((component_count, component_count), dtype=np.float64)
    extended_gauge = sparse.vstack(
        [
            gauge,
            sparse.csr_matrix((len(preserved_components), component_count), dtype=np.float64),
        ],
        format="csr",
    )
    kkt = sparse.bmat(
        [[pressure_matrix, extended_gauge], [extended_gauge.T, zero_gauge]],
        format="csc",
    )
    lu = splu(kkt)
    storage_bytes = sum(
        array.nbytes
        for factor in (lu.L, lu.U)
        for array in (factor.data, factor.indices, factor.indptr)
    )
    return HybridTraceProjectionFactorization(
        operators=operators,
        constraint_matrix=constraint_matrix,
        pressure_kkt_matrix=kkt,
        pressure_matrix=None,
        pressure_component_count=int(component_count),
        pressure_component_label=np.asarray(component_label, dtype=np.int32),
        preserved_mean_components=preserved_components,
        lu=lu,
        solver_mode=selected_solver,
        mean_pressure_coupling=None,
        mean_mass_matrix=None,
        mean_pressure_solution=None,
        reduced_mean_matrix=None,
        iterative_rtol=float(iterative_rtol),
        iterative_maxiter=int(iterative_maxiter),
        setup_iteration_count_total=0,
        setup_iteration_count_max=0,
        setup_internal_relative_residual_max=0.0,
        solver_storage_nnz=int(lu.L.nnz + lu.U.nnz),
        solver_storage_bytes=int(storage_bytes),
        factorization_time_s=float(time.perf_counter() - start_time),
    )


def _remove_component_means(
    values: np.ndarray,
    component_label: np.ndarray,
    component_count: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    sums = np.bincount(component_label, weights=array, minlength=component_count)
    sizes = np.bincount(component_label, minlength=component_count)
    means = sums / np.maximum(sizes, 1)
    return array - means[component_label]


def _solve_projected_pressure_cg(
    factorization: HybridTraceProjectionFactorization,
    rhs: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    matrix = factorization.pressure_matrix
    if matrix is None:
        raise ValueError("Projected-CG pressure solve requires a pressure matrix")
    labels = factorization.pressure_component_label
    component_count = factorization.pressure_component_count
    projected_rhs = _remove_component_means(rhs, labels, component_count)
    rhs_norm = float(np.linalg.norm(projected_rhs))
    if rhs_norm <= 1.0e-300:
        return np.zeros_like(projected_rhs), 0, 0.0

    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise RuntimeError("Projected-CG pressure matrix has a non-positive diagonal")

    def project(values: np.ndarray) -> np.ndarray:
        return _remove_component_means(values, labels, component_count)

    operator = LinearOperator(
        matrix.shape,
        matvec=lambda values: project(matrix @ project(values)),
        dtype=np.float64,
    )
    preconditioner = LinearOperator(
        matrix.shape,
        matvec=lambda values: project(project(values) / diagonal),
        dtype=np.float64,
    )
    iteration_count = 0

    def count_iteration(_iterate: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = cg(
        operator,
        projected_rhs,
        x0=np.zeros_like(projected_rhs),
        rtol=factorization.iterative_rtol,
        atol=0.0,
        maxiter=factorization.iterative_maxiter,
        M=preconditioner,
        callback=count_iteration,
    )
    solution = project(solution)
    residual = project(matrix @ solution - projected_rhs)
    relative_residual = float(np.linalg.norm(residual) / rhs_norm)
    if info != 0 or relative_residual > max(20.0 * factorization.iterative_rtol, 1.0e-10):
        raise RuntimeError(
            "Projected-CG pressure solve did not reach the declared tolerance: "
            f"info={info}, iterations={iteration_count}, relative_residual={relative_residual:.3e}"
        )
    return solution, iteration_count, relative_residual


def project_constant_cell_states_to_conservative_trace(
    operators: HybridTraceProjectionOperators,
    cell_velocity: np.ndarray,
    *,
    factorization: HybridTraceProjectionFactorization | None = None,
    preserved_mean_target: Literal["none", "particle_state_mean", "face_target_mean"] = "none",
) -> dict[str, Any]:
    """Project constant cell states onto a shared trace with exact cell mass balance.

    The canonical formulation imposes no global mean-flow constraint. The two
    preserved-mean targets are retained only for controlled sensitivity tests.
    """
    call_start = time.perf_counter()
    trace = operators.trace_geometry
    target_cell_velocity = np.asarray(cell_velocity, dtype=np.float64)
    if target_cell_velocity.shape != (trace.n_cells, 3):
        raise ValueError(
            f"Expected cell velocity shape {(trace.n_cells, 3)}, got {target_cell_velocity.shape}"
        )
    if not np.all(np.isfinite(target_cell_velocity)):
        raise ValueError("Cell velocity contains non-finite values")
    if factorization is None:
        factorization = build_hybrid_trace_projection_factorization(operators)
    elif factorization.operators is not operators:
        raise ValueError("The supplied factorization belongs to different trace operators")

    length = np.maximum(trace.face_r_owner_g + trace.face_r_neigh_g, 1.0e-300)
    owner_weight = trace.face_r_neigh_g / length
    neigh_weight = trace.face_r_owner_g / length
    target_face_velocity = (
        owner_weight[:, None] * target_cell_velocity[trace.face_owner]
        + neigh_weight[:, None] * target_cell_velocity[trace.face_neigh]
    )
    area = trace.voxel_size * trace.voxel_size
    face_trace = operators.face_trace_matrix
    divergence = operators.divergence_matrix
    target_rhs = area * np.asarray(face_trace.T @ target_face_velocity.reshape(-1)).reshape(-1)
    unconstrained_coefficients = target_rhs / operators.mass_diagonal
    unconstrained_face_velocity = np.asarray(
        face_trace @ unconstrained_coefficients
    ).reshape(trace.n_facelets, 3)
    target_mean_velocity = np.sum(
        trace.cell_volume[:, None] * target_cell_velocity,
        axis=0,
    ) / max(float(np.sum(trace.cell_volume)), 1.0e-300)
    unconstrained_trace_mean_velocity = np.asarray(
        operators.mean_velocity_matrix @ unconstrained_coefficients
    ).reshape(3)
    if preserved_mean_target == "particle_state_mean":
        selected_mean_target = target_mean_velocity
    elif preserved_mean_target == "face_target_mean":
        selected_mean_target = unconstrained_trace_mean_velocity
    elif preserved_mean_target == "none":
        selected_mean_target = unconstrained_trace_mean_velocity
    else:
        raise ValueError(f"Unknown preserved mean target: {preserved_mean_target!r}")
    if preserved_mean_target == "none" and factorization.preserved_mean_components:
        raise ValueError(
            "preserved_mean_target='none' requires a factorization without mean constraints"
        )
    if preserved_mean_target != "none" and not factorization.preserved_mean_components:
        raise ValueError(
            f"preserved_mean_target={preserved_mean_target!r} requires at least one mean constraint"
        )
    constraint_target = np.concatenate(
        [
            np.zeros(trace.n_cells, dtype=np.float64),
            selected_mean_target[list(factorization.preserved_mean_components)],
        ]
    )
    initial_constraint_residual = np.asarray(
        factorization.constraint_matrix @ unconstrained_coefficients
    ).reshape(-1) - constraint_target

    solve_start = time.perf_counter()
    iterative_iterations: list[int] = []
    iterative_residuals: list[float] = []
    if factorization.solver_mode == "direct_lu":
        pressure_rhs = np.concatenate(
            [
                initial_constraint_residual,
                np.zeros(factorization.pressure_component_count, dtype=np.float64),
            ]
        )
        pressure_solution = factorization.lu.solve(pressure_rhs)
        multipliers = pressure_solution[: factorization.constraint_matrix.shape[0]]
    elif factorization.solver_mode == "projected_cg":
        cell_residual = initial_constraint_residual[: trace.n_cells]
        mean_residual = initial_constraint_residual[trace.n_cells :]
        pressure_from_cell_residual, iterations, residual = _solve_projected_pressure_cg(
            factorization,
            cell_residual,
        )
        iterative_iterations.append(iterations)
        iterative_residuals.append(residual)
        coupling = factorization.mean_pressure_coupling
        pressure_from_mean = factorization.mean_pressure_solution
        reduced_mean_matrix = factorization.reduced_mean_matrix
        if coupling is None or pressure_from_mean is None or reduced_mean_matrix is None:
            raise RuntimeError("Projected-CG factorization is missing mean-constraint operators")
        if coupling.shape[1]:
            reduced_mean_rhs = mean_residual - coupling.T @ pressure_from_cell_residual
            mean_multiplier = np.linalg.solve(reduced_mean_matrix, reduced_mean_rhs)
            pressure = pressure_from_cell_residual - pressure_from_mean @ mean_multiplier
            multipliers = np.concatenate([pressure, mean_multiplier])
        else:
            pressure = pressure_from_cell_residual
            multipliers = pressure
    else:
        raise RuntimeError(f"Unsupported projection solver mode: {factorization.solver_mode!r}")
    solve_time = float(time.perf_counter() - solve_start)
    pressure = multipliers[: trace.n_cells]
    correction = (
        np.asarray(factorization.constraint_matrix.T @ multipliers).reshape(-1)
        / operators.mass_diagonal
    )
    coefficients = unconstrained_coefficients - correction

    face_velocity = np.asarray(face_trace @ coefficients).reshape(trace.n_facelets, 3)
    face_flux_sorted = area * np.sum(face_velocity * trace.face_normal, axis=1)
    aggregate_flux_sorted = np.bincount(
        trace.face_parent_edge,
        weights=face_flux_sorted,
        minlength=trace.stored_edge_sign_from_sorted.size,
    )
    aggregate_flux_stored = aggregate_flux_sorted * trace.stored_edge_sign_from_sorted
    target_face_flux_sorted = area * np.sum(target_face_velocity * trace.face_normal, axis=1)
    target_aggregate_flux_sorted = np.bincount(
        trace.face_parent_edge,
        weights=target_face_flux_sorted,
        minlength=trace.stored_edge_sign_from_sorted.size,
    )
    target_aggregate_flux_stored = (
        target_aggregate_flux_sorted * trace.stored_edge_sign_from_sorted
    )
    unconstrained_face_flux_sorted = area * np.sum(
        unconstrained_face_velocity * trace.face_normal,
        axis=1,
    )
    unconstrained_aggregate_flux_sorted = np.bincount(
        trace.face_parent_edge,
        weights=unconstrained_face_flux_sorted,
        minlength=trace.stored_edge_sign_from_sorted.size,
    )
    unconstrained_aggregate_flux_stored = (
        unconstrained_aggregate_flux_sorted * trace.stored_edge_sign_from_sorted
    )
    recovered_cell_velocity = np.asarray(
        operators.cell_recovery_matrix @ coefficients
    ).reshape(trace.n_cells, 3)
    mass_residual = np.asarray(divergence @ coefficients).reshape(-1)
    recovered_mean_velocity = np.sum(
        trace.cell_volume[:, None] * recovered_cell_velocity,
        axis=0,
    ) / max(float(np.sum(trace.cell_volume)), 1.0e-300)
    preserved_mean_residual = (
        recovered_mean_velocity[list(factorization.preserved_mean_components)]
        - selected_mean_target[list(factorization.preserved_mean_components)]
    )

    correction_norm = float(
        np.sqrt(area * np.sum((face_velocity - target_face_velocity) ** 2))
    )
    target_norm = max(
        float(np.sqrt(area * np.sum(target_face_velocity**2))),
        1.0e-300,
    )
    final_constraint_residual = np.asarray(
        factorization.constraint_matrix @ coefficients
    ).reshape(-1) - constraint_target
    projection_system_relative_residual = float(
        np.linalg.norm(final_constraint_residual)
        / max(float(np.linalg.norm(initial_constraint_residual)), 1.0e-300)
    )
    return {
        "U": target_cell_velocity.copy(),
        "U_state": target_cell_velocity.copy(),
        "U_trace_moment": recovered_cell_velocity,
        "U_target": target_cell_velocity,
        "p_projection": pressure,
        "phi": aggregate_flux_stored,
        "phi_target_raw": target_aggregate_flux_stored,
        "phi_unconstrained_trace": unconstrained_aggregate_flux_stored,
        "face_velocity": face_velocity,
        "face_velocity_target": target_face_velocity,
        "face_velocity_unconstrained_trace": unconstrained_face_velocity,
        "face_flux_sorted": face_flux_sorted,
        "trace_coefficients": coefficients.reshape(trace.n_trace_modes, 3),
        "mass_residual": mass_residual,
        "mass_inf_per_volume": float(
            np.max(np.abs(mass_residual) / np.maximum(trace.cell_volume, 1.0e-300))
        ),
        "preserved_mean_components": factorization.preserved_mean_components,
        "preserved_mean_target": preserved_mean_target,
        "throughflow_constraint_active": bool(factorization.preserved_mean_components),
        "particle_state_mean_velocity": target_mean_velocity,
        "unconstrained_trace_mean_velocity": unconstrained_trace_mean_velocity,
        "selected_preserved_mean_velocity": selected_mean_target,
        "preserved_mean_velocity_residual_inf": float(
            np.max(np.abs(preserved_mean_residual)) if preserved_mean_residual.size else 0.0
        ),
        "trace_correction_relative_l2": correction_norm / target_norm,
        "mass_orthogonality_defect": operators.mass_orthogonality_defect,
        "operator_assembly_time_s": operators.assembly_time_s,
        "factorization_time_s": factorization.factorization_time_s,
        "solve_time_s": solve_time,
        "cached_call_time_s": float(time.perf_counter() - call_start),
        "factor_nnz": int(factorization.solver_storage_nnz),
        "factor_bytes": int(factorization.solver_storage_bytes),
        "projection_solver": factorization.solver_mode,
        "projection_solver_rtol": float(factorization.iterative_rtol),
        "projection_solver_maxiter": int(factorization.iterative_maxiter),
        "projection_solver_iteration_count_total": int(sum(iterative_iterations)),
        "projection_solver_iteration_count_max": int(max(iterative_iterations, default=0)),
        "projection_solver_internal_relative_residual_max": float(
            max(iterative_residuals, default=0.0)
        ),
        "projection_solver_setup_iteration_count_total": int(
            factorization.setup_iteration_count_total
        ),
        "projection_solver_setup_iteration_count_max": int(
            factorization.setup_iteration_count_max
        ),
        "projection_solver_setup_internal_relative_residual_max": float(
            factorization.setup_internal_relative_residual_max
        ),
        "projection_system_relative_residual": projection_system_relative_residual,
        "n_trace_modes": trace.n_trace_modes,
        "n_trace_dofs": 3 * trace.n_trace_modes,
        "n_pressure_components": factorization.pressure_component_count,
        "trace_basis": trace.trace_basis,
        "cell_velocity_definition": "constant_particle_state_per_ownership_cell",
        "trace_moment_velocity_definition": "constant_flux_first_moment_of_projected_trace",
        "interior_velocity_dofs_per_cell": 0,
        "solver_formulation_id": (
            f"constant_cell_state_conservative_interface_trace_{trace.trace_basis}_"
            + (
                f"periodic_{preserved_mean_target}_v3"
                if factorization.preserved_mean_components
                else "divergence_only_v2"
            )
        ),
    }
def assemble_moment_constrained_hybrid_stokes(
    trace: HybridTraceGeometry,
    *,
    viscosity: float,
    body_force: np.ndarray,
    viscous_form: ViscousForm = "symmetric_gradient",
) -> HybridTraceSystem:
    """Assemble a conservative interface-trace system with one constant cell velocity."""
    start_time = time.perf_counter()
    area = trace.voxel_size * trace.voxel_size
    n_modes = trace.n_trace_modes
    n_trace_dofs = 3 * n_modes
    body_force = np.asarray(body_force, dtype=np.float64).reshape(3)

    owner_r = _minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_owner],
        trace.domain_length_x,
        trace.periodic_x,
    )
    neigh_r = _minimum_image_x(
        trace.face_centroid - trace.cell_centroid[trace.face_neigh],
        trace.domain_length_x,
        trace.periodic_x,
    )
    wall_r = _minimum_image_x(
        trace.wall_centroid - trace.cell_centroid[trace.wall_cell],
        trace.domain_length_x,
        trace.periodic_x,
    )

    n_facelets = trace.n_facelets
    incident_cell = np.concatenate([trace.face_owner, trace.face_neigh, trace.wall_cell])
    incident_face = np.concatenate(
        [
            np.arange(n_facelets, dtype=np.int32),
            np.arange(n_facelets, dtype=np.int32),
            np.full(trace.wall_cell.size, -1, dtype=np.int32),
        ]
    )
    incident_normal = np.concatenate(
        [trace.face_normal, -trace.face_normal, trace.wall_normal], axis=0
    )
    incident_r = np.concatenate([owner_r, neigh_r, wall_r], axis=0)
    order = np.argsort(incident_cell, kind="stable")
    incident_cell = incident_cell[order]
    incident_face = incident_face[order]
    incident_normal = incident_normal[order]
    incident_r = incident_r[order]
    cell_start = np.searchsorted(incident_cell, np.arange(trace.n_cells + 1))

    matrix_rows: list[np.ndarray] = []
    matrix_cols: list[np.ndarray] = []
    matrix_values: list[np.ndarray] = []
    divergence_rows: list[int] = []
    divergence_cols: list[int] = []
    divergence_values: list[float] = []
    body_force_matrix = np.zeros((n_trace_dofs, 3), dtype=np.float64)
    cell_recovery: list[tuple[np.ndarray, np.ndarray]] = []

    for cell in range(trace.n_cells):
        first = int(cell_start[cell])
        last = int(cell_start[cell + 1])
        cell_faces = incident_face[first:last]
        cell_normals = incident_normal[first:last]
        cell_r = incident_r[first:last]

        mode_set: set[int] = set()
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
                    gradient_basis[3 * local_mode + component, component, :] += (
                        area * value / volume
                    ) * normal

        if viscous_form == "full_gradient":
            local_matrix = viscosity * volume * np.einsum(
                "aij,bij->ab", gradient_basis, gradient_basis
            )
            stabilization_weight = viscosity * area / trace.voxel_size
        elif viscous_form == "symmetric_gradient":
            symmetric_basis = 0.5 * (
                gradient_basis + np.transpose(gradient_basis, (0, 2, 1))
            )
            local_matrix = 2.0 * viscosity * volume * np.einsum(
                "aij,bij->ab", symmetric_basis, symmetric_basis
            )
            stabilization_weight = 2.0 * viscosity * area / trace.voxel_size
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

        global_dofs = (
            3 * local_modes[:, None] + np.arange(3, dtype=np.int64)[None, :]
        ).reshape(-1)
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
    velocity_matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(n_trace_dofs, n_trace_dofs)
    ).tocsr()
    velocity_matrix = (0.5 * (velocity_matrix + velocity_matrix.T)).tocsr()
    divergence_matrix = sparse.coo_matrix(
        (divergence_values, (divergence_rows, divergence_cols)),
        shape=(trace.n_cells, n_trace_dofs),
    ).tocsr()
    rhs_trace = body_force_matrix @ body_force

    return HybridTraceSystem(
        trace_geometry=trace,
        velocity_matrix=velocity_matrix,
        divergence_matrix=divergence_matrix,
        body_force_matrix=body_force_matrix,
        rhs_trace=rhs_trace,
        cell_recovery=tuple(cell_recovery),
        viscosity=float(viscosity),
        body_force=body_force,
        viscous_form=viscous_form,
        assembly_time_s=float(time.perf_counter() - start_time),
    )


def _sparse_storage_bytes(matrix: sparse.spmatrix) -> int:
    compressed = matrix.tocsr()
    return int(compressed.data.nbytes + compressed.indices.nbytes + compressed.indptr.nbytes)


def build_hybrid_trace_factorization(
    system: HybridTraceSystem,
    *,
    solver: ForwardLinearSolver = "direct_lu",
    iterative_rtol: float = 1.0e-10,
    iterative_maxiter: int = 20000,
    iterative_refinement_steps: int = 3,
    velocity_lu_ordering: str = "COLAMD",
) -> HybridTraceFactorization:
    trace = system.trace_geometry
    velocity = system.velocity_matrix
    divergence = system.divergence_matrix
    n_trace_dofs = int(velocity.shape[0])
    n_cells = trace.n_cells
    start = time.perf_counter()

    if solver == "direct_lu":
        gauge = sparse.csr_matrix(
            np.ones((n_cells, 1), dtype=np.float64) / float(n_cells)
        )
        zero_pressure = sparse.csr_matrix((n_cells, n_cells), dtype=np.float64)
        zero_trace_gauge = sparse.csr_matrix((n_trace_dofs, 1), dtype=np.float64)
        kkt = sparse.bmat(
            [
                [velocity, divergence.T, zero_trace_gauge],
                [divergence, zero_pressure, gauge],
                [zero_trace_gauge.T, gauge.T, None],
            ],
            format="csc",
        )
        lu = splu(kkt)
        storage_bytes = sum(
            _sparse_storage_bytes(factor) for factor in (lu.L, lu.U)
        )
        return HybridTraceFactorization(
            system=system,
            kkt_matrix=kkt,
            lu=lu,
            solver_mode=solver,
            factorization_ordering="COLAMD",
            preconditioner=None,
            reduced_divergence=None,
            pressure_operator=None,
            iterative_rtol=0.0,
            iterative_maxiter=0,
            iterative_refinement_steps=0,
            solver_storage_nnz=int(lu.L.nnz + lu.U.nnz),
            solver_storage_bytes=int(storage_bytes),
            factorization_time_s=float(time.perf_counter() - start),
        )

    if solver not in {"minres", "schur_cg"}:
        raise ValueError(f"Unknown forward linear solver: {solver!r}")
    if n_cells < 2:
        raise ValueError("Iterative pressure elimination requires at least two cells")
    if iterative_rtol <= 0.0:
        raise ValueError("Iterative relative tolerance must be positive")
    if iterative_maxiter <= 0:
        raise ValueError("Iterative maximum iteration count must be positive")
    if iterative_refinement_steps < 0:
        raise ValueError("Iterative refinement step count cannot be negative")

    # Remove one pressure equation and unknown. The omitted mass equation is
    # linearly dependent because every interior trace flux is shared.
    reduced_divergence = divergence[:-1].tocsr()
    velocity_diagonal = np.asarray(velocity.diagonal(), dtype=np.float64)
    velocity_scale = max(float(np.max(np.abs(velocity_diagonal))), 1.0)
    spectral_floor = max(
        100.0 * np.finfo(np.float64).eps * velocity_scale,
        np.finfo(np.float64).tiny,
    )
    if np.any(velocity_diagonal <= spectral_floor):
        raise RuntimeError("Hybrid velocity block has a non-positive diagonal")
    inverse_velocity_diagonal = 1.0 / velocity_diagonal
    schur_diagonal = np.asarray(
        reduced_divergence.multiply(reduced_divergence) @ inverse_velocity_diagonal
    ).reshape(-1)
    schur_scale = max(float(np.max(np.abs(schur_diagonal))), 1.0)
    schur_floor = max(
        100.0 * np.finfo(np.float64).eps * schur_scale,
        np.finfo(np.float64).tiny,
    )
    if np.any(schur_diagonal <= schur_floor):
        raise RuntimeError("Hybrid pressure Schur preconditioner has a non-positive diagonal")

    if solver == "minres":
        zero_pressure = sparse.csr_matrix(
            (n_cells - 1, n_cells - 1), dtype=np.float64
        )
        kkt = sparse.bmat(
            [[velocity, reduced_divergence.T], [reduced_divergence, zero_pressure]],
            format="csr",
        )
        inverse_preconditioner_diagonal = np.concatenate(
            [inverse_velocity_diagonal, 1.0 / schur_diagonal]
        )
        preconditioner = LinearOperator(
            kkt.shape,
            matvec=lambda vector: inverse_preconditioner_diagonal * vector,
            rmatvec=lambda vector: inverse_preconditioner_diagonal * vector,
            dtype=np.float64,
        )
        storage_bytes = _sparse_storage_bytes(kkt) + int(
            inverse_preconditioner_diagonal.nbytes
        )
        return HybridTraceFactorization(
            system=system,
            kkt_matrix=kkt,
            lu=None,
            solver_mode=solver,
            factorization_ordering="none",
            preconditioner=preconditioner,
            reduced_divergence=reduced_divergence,
            pressure_operator=None,
            iterative_rtol=float(iterative_rtol),
            iterative_maxiter=int(iterative_maxiter),
            iterative_refinement_steps=int(iterative_refinement_steps),
            solver_storage_nnz=int(kkt.nnz + inverse_preconditioner_diagonal.size),
            solver_storage_bytes=int(storage_bytes),
            factorization_time_s=float(time.perf_counter() - start),
        )

    allowed_orderings = {"COLAMD", "MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"}
    if velocity_lu_ordering not in allowed_orderings:
        raise ValueError(
            f"Unknown velocity LU ordering {velocity_lu_ordering!r}; "
            f"expected one of {sorted(allowed_orderings)}"
        )
    velocity_lu = splu(velocity.tocsc(), permc_spec=velocity_lu_ordering)

    def apply_pressure_operator(vector: np.ndarray) -> np.ndarray:
        lifted = np.asarray(reduced_divergence.T @ vector).reshape(-1)
        return np.asarray(
            reduced_divergence @ velocity_lu.solve(lifted)
        ).reshape(-1)

    pressure_operator = LinearOperator(
        (n_cells - 1, n_cells - 1),
        matvec=apply_pressure_operator,
        rmatvec=apply_pressure_operator,
        dtype=np.float64,
    )
    inverse_schur_diagonal = 1.0 / schur_diagonal
    pressure_preconditioner = LinearOperator(
        (n_cells - 1, n_cells - 1),
        matvec=lambda vector: inverse_schur_diagonal * vector,
        rmatvec=lambda vector: inverse_schur_diagonal * vector,
        dtype=np.float64,
    )
    storage_bytes = sum(
        _sparse_storage_bytes(factor) for factor in (velocity_lu.L, velocity_lu.U)
    )
    storage_bytes += _sparse_storage_bytes(reduced_divergence)
    storage_bytes += int(inverse_schur_diagonal.nbytes)
    return HybridTraceFactorization(
        system=system,
        kkt_matrix=None,
        lu=velocity_lu,
        solver_mode=solver,
        factorization_ordering=velocity_lu_ordering,
        preconditioner=pressure_preconditioner,
        reduced_divergence=reduced_divergence,
        pressure_operator=pressure_operator,
        iterative_rtol=float(iterative_rtol),
        iterative_maxiter=int(iterative_maxiter),
        iterative_refinement_steps=0,
        solver_storage_nnz=int(
            velocity_lu.L.nnz
            + velocity_lu.U.nnz
            + reduced_divergence.nnz
            + inverse_schur_diagonal.size
        ),
        solver_storage_bytes=int(storage_bytes),
        factorization_time_s=float(time.perf_counter() - start),
    )

def solve_moment_constrained_hybrid_stokes(
    system: HybridTraceSystem,
    *,
    factorization: HybridTraceFactorization | None = None,
    body_force: np.ndarray | None = None,
) -> dict[str, Any]:
    call_start = time.perf_counter()
    trace = system.trace_geometry
    velocity = system.velocity_matrix
    divergence = system.divergence_matrix
    n_trace_dofs = int(velocity.shape[0])
    n_cells = trace.n_cells
    if factorization is None:
        factorization = build_hybrid_trace_factorization(system)
    elif factorization.system is not system:
        raise ValueError("The supplied factorization belongs to a different hybrid trace system")
    active_body_force = (
        system.body_force
        if body_force is None
        else np.asarray(body_force, dtype=np.float64).reshape(3)
    )
    rhs_trace = system.body_force_matrix @ active_body_force
    solve_start = time.perf_counter()
    iteration_count = 0
    refinement_count = 0

    if factorization.solver_mode == "direct_lu":
        if factorization.lu is None:
            raise RuntimeError("Direct hybrid solve is missing its LU factorization")
        rhs = np.concatenate([rhs_trace, np.zeros(n_cells + 1, dtype=np.float64)])
        solution = factorization.lu.solve(rhs)
        pressure = solution[n_trace_dofs : n_trace_dofs + n_cells]
        linear_info = 0
        pressure_gauge = "zero_mean_lagrange_multiplier"
    elif factorization.solver_mode == "minres":
        if factorization.preconditioner is None:
            raise RuntimeError("MINRES hybrid solve is missing its preconditioner")
        rhs = np.concatenate([rhs_trace, np.zeros(n_cells - 1, dtype=np.float64)])

        def count_iteration(_iterate: np.ndarray) -> None:
            nonlocal iteration_count
            iteration_count += 1

        if factorization.kkt_matrix is None:
            raise RuntimeError("MINRES hybrid solve is missing its KKT operator")
        solution, linear_info = minres(
            factorization.kkt_matrix,
            rhs,
            rtol=factorization.iterative_rtol,
            maxiter=factorization.iterative_maxiter,
            M=factorization.preconditioner,
            callback=count_iteration,
            check=False,
        )
        residual_limit = max(20.0 * factorization.iterative_rtol, 1.0e-12)
        rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-300)
        while linear_info == 0 and refinement_count < factorization.iterative_refinement_steps:
            correction_rhs = np.asarray(
                rhs - factorization.kkt_matrix @ solution
            ).reshape(-1)
            if float(np.linalg.norm(correction_rhs)) / rhs_norm <= residual_limit:
                break
            correction, correction_info = minres(
                factorization.kkt_matrix,
                correction_rhs,
                rtol=factorization.iterative_rtol,
                maxiter=factorization.iterative_maxiter,
                M=factorization.preconditioner,
                callback=count_iteration,
                check=False,
            )
            solution += correction
            linear_info = correction_info
            refinement_count += 1
        pressure = np.concatenate(
            [solution[n_trace_dofs:], np.zeros(1, dtype=np.float64)]
        )
        pressure -= float(np.mean(pressure))
        pressure_gauge = "one_pressure_dof_eliminated_then_centered"
    elif factorization.solver_mode == "schur_cg":
        if (
            factorization.lu is None
            or factorization.preconditioner is None
            or factorization.reduced_divergence is None
            or factorization.pressure_operator is None
        ):
            raise RuntimeError("Schur-CG hybrid solve is missing solver operators")
        reduced_divergence = factorization.reduced_divergence
        unconstrained_trace = factorization.lu.solve(rhs_trace)
        pressure_rhs = np.asarray(
            reduced_divergence @ unconstrained_trace
        ).reshape(-1)

        def count_iteration(_iterate: np.ndarray) -> None:
            nonlocal iteration_count
            iteration_count += 1

        reduced_pressure, linear_info = cg(
            factorization.pressure_operator,
            pressure_rhs,
            rtol=factorization.iterative_rtol,
            atol=0.0,
            maxiter=factorization.iterative_maxiter,
            M=factorization.preconditioner,
            callback=count_iteration,
        )
        trace_solution = factorization.lu.solve(
            rhs_trace - reduced_divergence.T @ reduced_pressure
        )
        solution = np.concatenate([trace_solution, reduced_pressure])
        pressure = np.concatenate(
            [reduced_pressure, np.zeros(1, dtype=np.float64)]
        )
        pressure -= float(np.mean(pressure))
        rhs = np.concatenate([rhs_trace, np.zeros(n_cells - 1, dtype=np.float64)])
        pressure_gauge = "one_pressure_dof_eliminated_then_centered"
    else:
        raise RuntimeError(
            f"Unsupported hybrid linear solver mode: {factorization.solver_mode!r}"
        )

    solve_time = float(time.perf_counter() - solve_start)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Hybrid trace KKT solve returned a non-finite state")
    if factorization.kkt_matrix is not None:
        linear_residual = np.asarray(
            factorization.kkt_matrix @ solution - rhs
        ).reshape(-1)
    else:
        trace_solution = solution[:n_trace_dofs]
        linear_residual = np.concatenate(
            [
                np.asarray(
                    velocity @ trace_solution + divergence.T @ pressure - rhs_trace
                ).reshape(-1),
                np.asarray(divergence[:-1] @ trace_solution).reshape(-1),
            ]
        )
    linear_relative_residual = float(
        np.linalg.norm(linear_residual) / max(float(np.linalg.norm(rhs)), 1.0e-300)
    )
    if factorization.solver_mode in {"minres", "schur_cg"}:
        residual_limit = max(20.0 * factorization.iterative_rtol, 1.0e-12)
        if linear_info != 0 or linear_relative_residual > residual_limit:
            raise RuntimeError(
                f"{factorization.solver_mode} hybrid solve did not meet the declared "
                f"tolerance: info={linear_info}, iterations={iteration_count}, "
                f"relative_residual={linear_relative_residual:.3e}, "
                f"limit={residual_limit:.3e}"
            )

    trace_coefficients = solution[:n_trace_dofs].reshape(trace.n_trace_modes, 3)
    cell_velocity = np.zeros((n_cells, 3), dtype=np.float64)
    for cell, (global_dofs, recovery) in enumerate(system.cell_recovery):
        if global_dofs.size:
            cell_velocity[cell] = recovery @ solution[global_dofs]

    face_velocity = np.zeros((trace.n_facelets, 3), dtype=np.float64)
    for column in range(trace.face_mode_ids.shape[1]):
        mode = trace.face_mode_ids[:, column]
        valid = mode >= 0
        if np.any(valid):
            face_velocity[valid] += (
                trace.face_mode_values[valid, column, None]
                * trace_coefficients[mode[valid]]
            )
    face_flux_sorted = (
        trace.voxel_size * trace.voxel_size
    ) * np.sum(face_velocity * trace.face_normal, axis=1)
    aggregate_flux_sorted = np.bincount(
        trace.face_parent_edge,
        weights=face_flux_sorted,
        minlength=trace.stored_edge_sign_from_sorted.size,
    )
    aggregate_flux_stored = aggregate_flux_sorted * trace.stored_edge_sign_from_sorted

    mass_residual = np.asarray(divergence @ solution[:n_trace_dofs]).reshape(-1)
    momentum_residual = np.asarray(
        velocity @ solution[:n_trace_dofs]
        + divergence.T @ pressure
        - rhs_trace
    ).reshape(-1)
    symmetry_denominator = max(float(sparse.linalg.norm(velocity)), 1.0e-300)
    symmetry_defect = float(
        sparse.linalg.norm(velocity - velocity.T) / symmetry_denominator
    )
    call_time = float(time.perf_counter() - call_start)

    return {
        "U": cell_velocity,
        "p": pressure,
        "phi": aggregate_flux_stored,
        "face_velocity": face_velocity,
        "face_flux_sorted": face_flux_sorted,
        "trace_coefficients": trace_coefficients,
        "mass_residual": mass_residual,
        "momentum_residual": momentum_residual,
        "mass_inf_per_volume": float(
            np.max(np.abs(mass_residual) / np.maximum(trace.cell_volume, 1.0e-300))
        ),
        "momentum_residual_inf": float(np.max(np.abs(momentum_residual))),
        "velocity_matrix_symmetry_defect": symmetry_defect,
        "assembly_time_s": system.assembly_time_s,
        "factorization_time_s": factorization.factorization_time_s,
        "solve_time_s": solve_time,
        "cached_call_time_s": call_time,
        "factor_nnz": int(factorization.solver_storage_nnz),
        "factor_bytes": int(factorization.solver_storage_bytes),
        "linear_solver": factorization.solver_mode,
        "factorization_ordering": factorization.factorization_ordering,
        "linear_solver_iterations": int(iteration_count),
        "linear_solver_refinement_steps": int(refinement_count),
        "linear_solver_info": int(linear_info),
        "linear_solver_relative_residual": linear_relative_residual,
        "linear_solver_rtol": float(factorization.iterative_rtol),
        "linear_solver_maxiter": int(factorization.iterative_maxiter),
        "pressure_gauge": pressure_gauge,
        "n_trace_modes": trace.n_trace_modes,
        "n_trace_dofs": n_trace_dofs,
        "n_patches": trace.n_patches,
        "trace_basis": trace.trace_basis,
        "viscous_form": system.viscous_form,
        "body_force": active_body_force,
        "solver_formulation_id": (
            f"moment_constrained_hybrid_voronoi_{trace.trace_basis}_{system.viscous_form}_v1"
        ),
    }