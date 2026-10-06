from __future__ import annotations

import time
from dataclasses import replace
from typing import Any


SIGNED_NORMAL_GROUP_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")
SIGNED_NORMAL_GROUP_VECTORS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)


def _build_laplacian(ns: dict[str, Any], geom: Any, tproj: Any) -> Any:
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    owner = geom.owner.astype(cp.int32, copy=False)
    neigh = geom.neigh.astype(cp.int32, copy=False)
    vals = cp.concatenate([tproj, -tproj, tproj, -tproj]).astype(cp.float64)
    rows = cp.concatenate([owner, owner, neigh, neigh]).astype(cp.int32)
    cols = cp.concatenate([owner, neigh, neigh, owner]).astype(cp.int32)
    return cpx_sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(int(geom.n_cells), int(geom.n_cells)),
    ).tocsr()


def clone_geometry(
    ns: dict[str, Any],
    geom: Any,
    *,
    tproj: Any | None = None,
    twall: Any | None = None,
    w_owner: Any | None = None,
    w_neigh: Any | None = None,
    dvec: Any | None = None,
) -> Any:
    cp = ns["cp"]
    tproj_use = geom.tproj if tproj is None else cp.asarray(tproj, dtype=cp.float64)
    twall_use = geom.twall if twall is None else cp.asarray(twall, dtype=cp.float64)
    w_owner_use = geom.w_owner if w_owner is None else cp.asarray(w_owner, dtype=cp.float64)
    w_neigh_use = geom.w_neigh if w_neigh is None else cp.asarray(w_neigh, dtype=cp.float64)
    dvec_use = geom.dvec if dvec is None else cp.asarray(dvec, dtype=cp.float64)
    return replace(
        geom,
        tproj=tproj_use,
        twall=twall_use,
        w_owner=w_owner_use,
        w_neigh=w_neigh_use,
        dvec=dvec_use,
        laplacian=_build_laplacian(ns, geom, tproj_use),
    )


def _axis_facelet_arrays(ns: dict[str, Any], geom: Any, axis: int) -> tuple[Any, Any, Any]:
    cp = ns["cp"]
    labels = geom.labels.astype(cp.int32, copy=False)
    mask = geom.mask.astype(cp.bool_, copy=False)
    dist = geom.dist.astype(cp.float64, copy=False)
    shape = tuple(int(v) for v in labels.shape)
    half = 0.5 * float(getattr(geom, "_voxel_size_for_geodesic_metric", 1.0))

    lo = [slice(None), slice(None), slice(None)]
    hi = [slice(None), slice(None), slice(None)]
    lo[axis] = slice(0, shape[axis] - 1)
    hi[axis] = slice(1, shape[axis])

    lab0 = labels[tuple(lo)]
    lab1 = labels[tuple(hi)]
    m0 = mask[tuple(lo)]
    m1 = mask[tuple(hi)]
    d0 = dist[tuple(lo)]
    d1 = dist[tuple(hi)]
    valid = m0 & m1 & (lab0 >= 0) & (lab1 >= 0) & (lab0 != lab1)
    if not bool(cp.any(valid).get()):
        empty_i = cp.asarray([], dtype=cp.int64)
        empty_f = cp.asarray([], dtype=cp.float64)
        return empty_i, empty_f, empty_f

    a = lab0[valid].astype(cp.int64, copy=False)
    b = lab1[valid].astype(cp.int64, copy=False)
    da = d0[valid].astype(cp.float64, copy=False) + half
    db = d1[valid].astype(cp.float64, copy=False) + half
    owner_is_a = a < b
    lo_lab = cp.minimum(a, b)
    hi_lab = cp.maximum(a, b)
    keys = lo_lab * cp.int64(int(geom.n_cells)) + hi_lab
    ro = cp.where(owner_is_a, da, db)
    rn = cp.where(owner_is_a, db, da)
    return keys, ro, rn


def _periodic_x_facelet_arrays(ns: dict[str, Any], geom: Any) -> tuple[Any, Any, Any]:
    cp = ns["cp"]
    labels = geom.labels.astype(cp.int32, copy=False)
    mask = geom.mask.astype(cp.bool_, copy=False)
    dist = geom.dist.astype(cp.float64, copy=False)
    half = 0.5 * float(getattr(geom, "_voxel_size_for_geodesic_metric", 1.0))

    lab0 = labels[:, :, -1]
    lab1 = labels[:, :, 0]
    m0 = mask[:, :, -1]
    m1 = mask[:, :, 0]
    d0 = dist[:, :, -1]
    d1 = dist[:, :, 0]
    valid = m0 & m1 & (lab0 >= 0) & (lab1 >= 0) & (lab0 != lab1)
    if not bool(cp.any(valid).get()):
        empty_i = cp.asarray([], dtype=cp.int64)
        empty_f = cp.asarray([], dtype=cp.float64)
        return empty_i, empty_f, empty_f

    a = lab0[valid].astype(cp.int64, copy=False)
    b = lab1[valid].astype(cp.int64, copy=False)
    da = d0[valid].astype(cp.float64, copy=False) + half
    db = d1[valid].astype(cp.float64, copy=False) + half
    owner_is_a = a < b
    lo_lab = cp.minimum(a, b)
    hi_lab = cp.maximum(a, b)
    keys = lo_lab * cp.int64(int(geom.n_cells)) + hi_lab
    ro = cp.where(owner_is_a, da, db)
    rn = cp.where(owner_is_a, db, da)
    return keys, ro, rn


def _empty_signed_facelet_arrays(cp: Any) -> tuple[Any, Any, Any, Any, Any]:
    return (
        cp.asarray([], dtype=cp.int64),
        cp.asarray([], dtype=cp.int8),
        cp.asarray([], dtype=cp.float64),
        cp.asarray([], dtype=cp.float64),
        cp.empty((0, 3), dtype=cp.float64),
    )


def _axis_signed_facelet_arrays(
    ns: dict[str, Any],
    geom: Any,
    axis: int,
    h: float,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return facelets oriented from the lower cell label to the higher label."""
    cp = ns["cp"]
    labels = geom.labels.astype(cp.int32, copy=False)
    mask = geom.mask.astype(cp.bool_, copy=False)
    dist = geom.dist.astype(cp.float64, copy=False)
    shape = tuple(int(v) for v in labels.shape)
    half = 0.5 * h

    lo = [slice(None), slice(None), slice(None)]
    hi = [slice(None), slice(None), slice(None)]
    lo[axis] = slice(0, shape[axis] - 1)
    hi[axis] = slice(1, shape[axis])
    lab0 = labels[tuple(lo)]
    lab1 = labels[tuple(hi)]
    valid = (
        mask[tuple(lo)]
        & mask[tuple(hi)]
        & (lab0 >= 0)
        & (lab1 >= 0)
        & (lab0 != lab1)
    )
    if not bool(cp.any(valid).get()):
        return _empty_signed_facelet_arrays(cp)

    coord_zyx = cp.stack(cp.nonzero(valid), axis=1).astype(cp.float64)
    a = lab0[valid].astype(cp.int64, copy=False)
    b = lab1[valid].astype(cp.int64, copy=False)
    da = dist[tuple(lo)][valid].astype(cp.float64, copy=False) + half
    db = dist[tuple(hi)][valid].astype(cp.float64, copy=False) + half
    owner_is_a = a < b
    lo_lab = cp.minimum(a, b)
    hi_lab = cp.maximum(a, b)
    keys = lo_lab * cp.int64(int(geom.n_cells)) + hi_lab
    ro = cp.where(owner_is_a, da, db)
    rn = cp.where(owner_is_a, db, da)

    physical_axis = {2: 0, 1: 1, 0: 2}[int(axis)]
    group_id = (2 * physical_axis + cp.where(owner_is_a, 0, 1)).astype(cp.int8)
    centroid = cp.stack(
        [coord_zyx[:, 2] + 0.5, coord_zyx[:, 1] + 0.5, coord_zyx[:, 0] + 0.5],
        axis=1,
    )
    centroid[:, physical_axis] = coord_zyx[:, axis] + 1.0
    centroid *= h
    return keys, group_id, ro, rn, centroid


def _periodic_x_signed_facelet_arrays(
    ns: dict[str, Any],
    geom: Any,
    h: float,
) -> tuple[Any, Any, Any, Any, Any]:
    cp = ns["cp"]
    labels = geom.labels.astype(cp.int32, copy=False)
    mask = geom.mask.astype(cp.bool_, copy=False)
    dist = geom.dist.astype(cp.float64, copy=False)
    half = 0.5 * h

    lab0 = labels[:, :, -1]
    lab1 = labels[:, :, 0]
    valid = mask[:, :, -1] & mask[:, :, 0] & (lab0 >= 0) & (lab1 >= 0) & (lab0 != lab1)
    if not bool(cp.any(valid).get()):
        return _empty_signed_facelet_arrays(cp)

    coord_zy = cp.stack(cp.nonzero(valid), axis=1).astype(cp.float64)
    a = lab0[valid].astype(cp.int64, copy=False)
    b = lab1[valid].astype(cp.int64, copy=False)
    da = dist[:, :, -1][valid].astype(cp.float64, copy=False) + half
    db = dist[:, :, 0][valid].astype(cp.float64, copy=False) + half
    owner_is_a = a < b
    lo_lab = cp.minimum(a, b)
    hi_lab = cp.maximum(a, b)
    keys = lo_lab * cp.int64(int(geom.n_cells)) + hi_lab
    group_id = cp.where(owner_is_a, 0, 1).astype(cp.int8)
    ro = cp.where(owner_is_a, da, db)
    rn = cp.where(owner_is_a, db, da)
    domain_x = float(labels.shape[2]) * h
    centroid = cp.stack(
        [
            cp.full(coord_zy.shape[0], domain_x, dtype=cp.float64),
            (coord_zy[:, 1] + 0.5) * h,
            (coord_zy[:, 0] + 0.5) * h,
        ],
        axis=1,
    )
    return keys, group_id, ro, rn, centroid


def _incident_moment_matrix(
    cp: Any,
    n_cells: int,
    owner: Any,
    neigh: Any,
    area: Any,
    normal: Any,
) -> Any:
    moment = cp.zeros((n_cells, 3, 3), dtype=cp.float64)
    for a in range(3):
        for b in range(3):
            value = area * normal[:, a] * normal[:, b]
            cp.add.at(moment, (owner, a, b), value)
            cp.add.at(moment, (neigh, a, b), value)
    return moment


def _rank_condition_from_moment(cp: Any, moment: Any) -> tuple[Any, Any, Any, Any]:
    eigvals = cp.maximum(cp.linalg.eigvalsh(moment), 0.0)
    largest = eigvals[:, -1]
    tolerance = (3.0 * float(cp.finfo(cp.float64).eps)) * largest
    retained = eigvals > tolerance[:, None]
    rank = cp.sum(retained, axis=1).astype(cp.int8)
    condition = cp.where(
        rank == 3,
        largest / cp.maximum(eigvals[:, 0], cp.finfo(cp.float64).tiny),
        cp.inf,
    )
    return rank, condition, eigvals, tolerance


def build_signed_normal_face_groups(ns: dict[str, Any], geom: Any, cfg: Any) -> dict[str, Any]:
    """Preserve the six signed Cartesian normal groups hidden by edge aggregation."""
    cp = ns["cp"]
    n_cells = int(geom.n_cells)
    n_edges = int(geom.owner.size)
    h = float(getattr(cfg, "voxel_size", 1.0))
    area0 = h * h

    parts = [_axis_signed_facelet_arrays(ns, geom, axis, h) for axis in (2, 1, 0)]
    if bool(getattr(cfg, "periodic_x", False)):
        parts.append(_periodic_x_signed_facelet_arrays(ns, geom, h))
    parts = [part for part in parts if int(part[0].size)]
    if not parts:
        raise RuntimeError("No intercell facelets found for signed-normal grouping")

    face_key = cp.concatenate([part[0] for part in parts]).astype(cp.int64, copy=False)
    face_group = cp.concatenate([part[1] for part in parts]).astype(cp.int8, copy=False)
    face_ro = cp.concatenate([part[2] for part in parts]).astype(cp.float64, copy=False)
    face_rn = cp.concatenate([part[3] for part in parts]).astype(cp.float64, copy=False)
    face_centroid = cp.concatenate([part[4] for part in parts], axis=0).astype(cp.float64, copy=False)

    stored_owner = geom.owner.astype(cp.int64, copy=False)
    stored_neigh = geom.neigh.astype(cp.int64, copy=False)
    edge_owner = cp.minimum(stored_owner, stored_neigh)
    edge_neigh = cp.maximum(stored_owner, stored_neigh)
    stored_to_sorted_sign = cp.where(stored_owner <= stored_neigh, 1.0, -1.0).astype(cp.float64)
    edge_key = edge_owner * cp.int64(n_cells) + edge_neigh
    edge_order = cp.argsort(edge_key)
    sorted_edge_key = edge_key[edge_order]
    if n_edges > 1 and bool(cp.any(sorted_edge_key[1:] == sorted_edge_key[:-1]).get()):
        raise RuntimeError("Stored geometry has duplicate aggregate cell-pair edges")

    face_pos = cp.searchsorted(sorted_edge_key, face_key)
    face_in_range = face_pos < n_edges
    face_safe_pos = cp.minimum(face_pos, max(n_edges - 1, 0))
    face_matched = face_in_range & (sorted_edge_key[face_safe_pos] == face_key)
    if not bool(cp.all(face_matched).get()):
        raise RuntimeError(f"Signed-normal grouping found {int((~face_matched).sum().get())} unmapped facelets")
    face_parent = edge_order[face_safe_pos].astype(cp.int32)

    composite_key = face_key * cp.int64(6) + face_group.astype(cp.int64)
    unique_group_key, group_inverse = cp.unique(composite_key, return_inverse=True)
    n_groups = int(unique_group_key.size)
    group_edge_key = unique_group_key // cp.int64(6)
    group_id = (unique_group_key % cp.int64(6)).astype(cp.int8)
    group_pos = cp.searchsorted(sorted_edge_key, group_edge_key)
    group_parent = edge_order[group_pos].astype(cp.int32)
    group_owner = group_edge_key // cp.int64(n_cells)
    group_neigh = group_edge_key % cp.int64(n_cells)

    group_facelet_count = cp.bincount(group_inverse, minlength=n_groups).astype(cp.int32)
    group_area = group_facelet_count.astype(cp.float64) * area0
    group_ro = cp.bincount(group_inverse, weights=face_ro, minlength=n_groups) / cp.maximum(
        group_facelet_count, 1
    )
    group_rn = cp.bincount(group_inverse, weights=face_rn, minlength=n_groups) / cp.maximum(
        group_facelet_count, 1
    )
    group_centroid = cp.zeros((n_groups, 3), dtype=cp.float64)
    for component in range(3):
        group_centroid[:, component] = cp.bincount(
            group_inverse,
            weights=face_centroid[:, component],
            minlength=n_groups,
        ) / cp.maximum(group_facelet_count, 1)
    normal_lookup = cp.asarray(SIGNED_NORMAL_GROUP_VECTORS, dtype=cp.float64)
    group_normal = normal_lookup[group_id.astype(cp.int32)]
    group_avec = group_area[:, None] * group_normal

    edge_area_from_groups = cp.bincount(group_parent, weights=group_area, minlength=n_edges)
    theta = group_area / cp.maximum(edge_area_from_groups[group_parent], 1.0e-300)
    group_tproj_exact = theta * geom.tproj[group_parent].astype(cp.float64, copy=False)
    edge_tproj_from_groups = cp.bincount(group_parent, weights=group_tproj_exact, minlength=n_edges)
    edge_avec_from_groups = cp.zeros((n_edges, 3), dtype=cp.float64)
    for component in range(3):
        edge_avec_from_groups[:, component] = cp.bincount(
            group_parent,
            weights=group_avec[:, component],
            minlength=n_edges,
        )
    edge_avec_sorted = geom.avec.astype(cp.float64, copy=False) * stored_to_sorted_sign[:, None]

    edge_facelet_count = cp.bincount(face_parent, minlength=n_edges).astype(cp.int32)
    face_ell = face_ro + face_rn
    edge_ell_mean = cp.bincount(face_parent, weights=face_ell, minlength=n_edges) / cp.maximum(
        edge_facelet_count, 1
    )
    edge_ell_second = cp.bincount(face_parent, weights=face_ell * face_ell, minlength=n_edges) / cp.maximum(
        edge_facelet_count, 1
    )
    edge_ell_std = cp.sqrt(cp.maximum(edge_ell_second - edge_ell_mean * edge_ell_mean, 0.0))
    edge_ell_cv = edge_ell_std / cp.maximum(edge_ell_mean, 1.0e-300)
    edge_group_count = cp.bincount(group_parent, minlength=n_edges).astype(cp.int8)
    edge_area = geom.area.astype(cp.float64, copy=False)
    edge_chi = cp.linalg.norm(edge_avec_sorted, axis=1) / cp.maximum(edge_area, 1.0e-300)

    aggregate_normal = edge_avec_sorted / cp.maximum(edge_area[:, None], 1.0e-300)
    aggregate_moment = _incident_moment_matrix(
        cp,
        n_cells,
        edge_owner.astype(cp.int32),
        edge_neigh.astype(cp.int32),
        edge_area,
        aggregate_normal,
    )
    facelet_moment = _incident_moment_matrix(
        cp,
        n_cells,
        group_owner.astype(cp.int32),
        group_neigh.astype(cp.int32),
        group_area,
        group_normal,
    )
    aggregate_rank, aggregate_condition, aggregate_eigvals, aggregate_tol = _rank_condition_from_moment(
        cp, aggregate_moment
    )
    facelet_rank, facelet_condition, facelet_eigvals, facelet_tol = _rank_condition_from_moment(
        cp, facelet_moment
    )

    incident_area = cp.zeros(n_cells, dtype=cp.float64)
    incident_chi_area = cp.zeros(n_cells, dtype=cp.float64)
    incident_ell_area = cp.zeros(n_cells, dtype=cp.float64)
    incident_edge_count = cp.zeros(n_cells, dtype=cp.int32)
    incident_group_count = cp.zeros(n_cells, dtype=cp.int32)
    for cells in (edge_owner.astype(cp.int32), edge_neigh.astype(cp.int32)):
        cp.add.at(incident_area, cells, edge_area)
        cp.add.at(incident_chi_area, cells, edge_area * edge_chi)
        cp.add.at(incident_ell_area, cells, edge_area * edge_ell_mean)
        cp.add.at(incident_edge_count, cells, 1)
    for cells in (group_owner.astype(cp.int32), group_neigh.astype(cp.int32)):
        cp.add.at(incident_group_count, cells, 1)
    characteristic_length = cp.cbrt(cp.maximum(geom.volume.astype(cp.float64), 1.0e-300))
    cell_chi_area_mean = incident_chi_area / cp.maximum(incident_area, 1.0e-300)
    cell_geodesic_aspect = (
        incident_ell_area / cp.maximum(incident_area, 1.0e-300)
    ) / characteristic_length

    area_abs_error = cp.abs(edge_area_from_groups - edge_area)
    avec_abs_error = cp.linalg.norm(edge_avec_from_groups - edge_avec_sorted, axis=1)
    tproj_abs_error = cp.abs(edge_tproj_from_groups - geom.tproj.astype(cp.float64, copy=False))
    cp.cuda.Stream.null.synchronize()
    return {
        "group_names": SIGNED_NORMAL_GROUP_NAMES,
        "group_id": group_id,
        "group_owner": group_owner.astype(cp.int32),
        "group_neigh": group_neigh.astype(cp.int32),
        "group_parent": group_parent,
        "group_normal": group_normal,
        "group_area": group_area,
        "group_avec": group_avec,
        "group_centroid": group_centroid,
        "group_r_owner_g": group_ro,
        "group_r_neigh_g": group_rn,
        "group_ell_g": group_ro + group_rn,
        "group_facelet_count": group_facelet_count,
        "group_theta_area": theta,
        "group_tproj_exact": group_tproj_exact,
        "edge_owner_sorted": edge_owner.astype(cp.int32),
        "edge_neigh_sorted": edge_neigh.astype(cp.int32),
        "edge_stored_to_sorted_sign": stored_to_sorted_sign,
        "edge_area_from_groups": edge_area_from_groups,
        "edge_avec_sorted": edge_avec_sorted,
        "edge_avec_from_groups": edge_avec_from_groups,
        "edge_tproj_from_groups": edge_tproj_from_groups,
        "edge_facelet_count": edge_facelet_count,
        "edge_group_count": edge_group_count,
        "edge_chi": edge_chi,
        "edge_ell_mean": edge_ell_mean,
        "edge_ell_cv": edge_ell_cv,
        "aggregate_moment": aggregate_moment,
        "facelet_moment": facelet_moment,
        "aggregate_rank": aggregate_rank,
        "facelet_rank": facelet_rank,
        "aggregate_condition": aggregate_condition,
        "facelet_condition": facelet_condition,
        "aggregate_eigvals": aggregate_eigvals,
        "facelet_eigvals": facelet_eigvals,
        "aggregate_rank_tolerance": aggregate_tol,
        "facelet_rank_tolerance": facelet_tol,
        "cell_incident_edge_count": incident_edge_count,
        "cell_incident_group_count": incident_group_count,
        "cell_chi_area_mean": cell_chi_area_mean,
        "cell_geodesic_aspect": cell_geodesic_aspect,
        "cell_wall_coefficient_density": geom.twall.astype(cp.float64) / cp.maximum(
            geom.volume.astype(cp.float64), 1.0e-300
        ),
        "invariants": {
            "area_max_abs": float(cp.max(area_abs_error).get()),
            "area_max_rel": float(
                cp.max(area_abs_error / cp.maximum(cp.maximum(edge_area, edge_area_from_groups), 1.0e-300)).get()
            ),
            "area_vector_max_abs": float(cp.max(avec_abs_error).get()),
            "area_vector_max_rel_to_area": float(
                cp.max(avec_abs_error / cp.maximum(edge_area, 1.0e-300)).get()
            ),
            "tproj_exact_split_max_abs": float(cp.max(tproj_abs_error).get()),
            "tproj_exact_split_max_rel": float(
                cp.max(
                    tproj_abs_error
                    / cp.maximum(cp.abs(geom.tproj.astype(cp.float64, copy=False)), 1.0e-300)
                ).get()
            ),
        },
    }


def distribute_aggregate_flux_to_signed_groups(
    ns: dict[str, Any],
    groups: dict[str, Any],
    phi: Any,
    *,
    rule: str = "area",
    velocity_predictor: Any | None = None,
) -> dict[str, Any]:
    """Split each edge flux while preserving its aggregate value to roundoff."""
    cp = ns["cp"]
    parent = groups["group_parent"]
    parent_sign = groups["edge_stored_to_sorted_sign"]
    parent_phi_sorted = phi.astype(cp.float64, copy=False) * parent_sign
    theta = groups["group_theta_area"]
    rule_key = str(rule or "area").lower().strip()
    if rule_key in {"area", "area_only", "exact_area"}:
        group_phi = theta * parent_phi_sorted[parent]
        rule_id = "area_only_exact_aggregate"
    elif rule_key in {"predictor", "predictor_constrained", "raw_u_predictor"}:
        if velocity_predictor is None:
            raise ValueError("velocity_predictor is required for predictor-constrained subfluxes")
        owner = groups["group_owner"]
        neigh = groups["group_neigh"]
        ell = cp.maximum(groups["group_ell_g"], 1.0e-300)
        w_owner = groups["group_r_neigh_g"] / ell
        w_neigh = groups["group_r_owner_g"] / ell
        face_velocity = w_owner[:, None] * velocity_predictor[owner] + w_neigh[:, None] * velocity_predictor[neigh]
        predicted = groups["group_area"] * cp.sum(face_velocity * groups["group_normal"], axis=1)
        predicted_sum = cp.bincount(parent, weights=predicted, minlength=int(parent_phi_sorted.size))
        group_phi = predicted + theta * (parent_phi_sorted[parent] - predicted_sum[parent])
        rule_id = "predictor_constrained_exact_aggregate"
    else:
        raise ValueError(f"Unsupported signed-normal subflux rule: {rule}")

    collapsed_sorted = cp.bincount(parent, weights=group_phi, minlength=int(parent_phi_sorted.size))
    collapsed_stored = collapsed_sorted * parent_sign
    difference = collapsed_stored - phi.astype(cp.float64, copy=False)
    return {
        "group_phi": group_phi,
        "collapsed_phi": collapsed_stored,
        "rule_id": rule_id,
        "aggregate_max_abs": float(cp.max(cp.abs(difference)).get()),
        "aggregate_max_rel": float(
            cp.max(cp.abs(difference) / cp.maximum(cp.abs(phi.astype(cp.float64, copy=False)), 1.0e-300)).get()
        ),
    }


def reconstruct_rank_aware_velocity_from_signed_flux(
    ns: dict[str, Any],
    groups: dict[str, Any],
    group_phi: Any,
    *,
    unobservable_prior: Any | None = None,
) -> dict[str, Any]:
    """Solve the observable normal-moment system without a model ridge or wall penalty."""
    cp = ns["cp"]
    n_cells = int(groups["facelet_moment"].shape[0])
    owner = groups["group_owner"]
    neigh = groups["group_neigh"]
    normal = groups["group_normal"]
    moment = groups["facelet_moment"]
    rhs = cp.zeros((n_cells, 3), dtype=cp.float64)
    for component in range(3):
        value = group_phi * normal[:, component]
        cp.add.at(rhs, (owner, component), value)
        cp.add.at(rhs, (neigh, component), value)

    eigvals, eigvecs = cp.linalg.eigh(moment)
    eigvals = cp.maximum(eigvals, 0.0)
    tolerance = (3.0 * float(cp.finfo(cp.float64).eps)) * eigvals[:, -1]
    retained = eigvals > tolerance[:, None]
    inverse = cp.where(retained, 1.0 / cp.maximum(eigvals, cp.finfo(cp.float64).tiny), 0.0)
    spectral_rhs = cp.einsum("nji,nj->ni", eigvecs, rhs)
    velocity_observable = cp.einsum("nij,nj->ni", eigvecs, inverse * spectral_rhs)
    projector = cp.einsum("nik,nk,njk->nij", eigvecs, retained.astype(cp.float64), eigvecs)
    rank = cp.sum(retained, axis=1).astype(cp.int8)
    if unobservable_prior is None:
        velocity_completed = velocity_observable.copy()
        completion_id = "observable_projection_zero_nullspace"
    else:
        identity = cp.eye(3, dtype=cp.float64)[None, :, :]
        null_component = cp.einsum("nij,nj->ni", identity - projector, unobservable_prior)
        velocity_completed = velocity_observable + null_component
        completion_id = "observable_projection_plus_solve_state_nullspace"

    target = group_phi / cp.maximum(groups["group_area"], 1.0e-300)
    predicted_owner = cp.sum(velocity_completed[owner] * normal, axis=1)
    predicted_neigh = cp.sum(velocity_completed[neigh] * normal, axis=1)
    residual = cp.concatenate([predicted_owner - target, predicted_neigh - target])
    return {
        "U_observable": velocity_observable,
        "U_completed": velocity_completed,
        "observable_projector": projector,
        "rank": rank,
        "rank_tolerance": tolerance,
        "completion_id": completion_id,
        "normal_equation_residual_l2": float(cp.sqrt(cp.mean(residual * residual)).get()),
        "normal_equation_residual_inf": float(cp.max(cp.abs(residual)).get()),
    }


def build_geodesic_face_metric(
    ns: dict[str, Any],
    geom: Any,
    cfg: Any,
    *,
    operator_variant: str = "geodesic_face",
) -> dict[str, Any]:
    cp = ns["cp"]
    t0 = time.perf_counter()
    h = float(getattr(cfg, "voxel_size", 1.0))
    try:
        setattr(geom, "_voxel_size_for_geodesic_metric", h)
    except Exception:
        pass

    parts = [_axis_facelet_arrays(ns, geom, axis) for axis in (2, 1, 0)]
    if bool(getattr(cfg, "periodic_x", False)):
        parts.append(_periodic_x_facelet_arrays(ns, geom))

    key_parts = [p[0] for p in parts if int(p[0].size)]
    if not key_parts:
        raise RuntimeError("No intercell facelets found for geodesic face metric")
    keys = cp.concatenate(key_parts).astype(cp.int64, copy=False)
    ro_facelet = cp.concatenate([p[1] for p in parts if int(p[0].size)]).astype(cp.float64, copy=False)
    rn_facelet = cp.concatenate([p[2] for p in parts if int(p[0].size)]).astype(cp.float64, copy=False)
    area0 = h * h

    unique_keys, inv = cp.unique(keys, return_inverse=True)
    facelet_count = cp.bincount(inv, minlength=int(unique_keys.size)).astype(cp.float64)
    area_sum = facelet_count * area0
    ro_sum = cp.bincount(inv, weights=ro_facelet * area0, minlength=int(unique_keys.size)).astype(cp.float64)
    rn_sum = cp.bincount(inv, weights=rn_facelet * area0, minlength=int(unique_keys.size)).astype(cp.float64)

    owner = geom.owner.astype(cp.int64, copy=False)
    neigh = geom.neigh.astype(cp.int64, copy=False)
    lo_lab = cp.minimum(owner, neigh)
    hi_lab = cp.maximum(owner, neigh)
    face_keys = lo_lab * cp.int64(int(geom.n_cells)) + hi_lab
    pos = cp.searchsorted(unique_keys, face_keys)
    in_range = pos < int(unique_keys.size)
    safe_pos = cp.minimum(pos, max(int(unique_keys.size) - 1, 0))
    matched = in_range & (unique_keys[safe_pos] == face_keys)
    if not bool(cp.all(matched).get()):
        missing = int((~matched).sum().get())
        raise RuntimeError(f"Missing geodesic face metric accumulators for {missing} stored faces")

    ro = ro_sum[safe_pos] / cp.maximum(area_sum[safe_pos], 1.0e-300)
    rn = rn_sum[safe_pos] / cp.maximum(area_sum[safe_pos], 1.0e-300)
    ell = cp.maximum(ro + rn, 1.0e-12)
    w_owner = rn / ell
    w_neigh = ro / ell

    dvec0 = geom.dvec.astype(cp.float64, copy=False)
    dnorm0 = cp.maximum(cp.sqrt(cp.sum(dvec0 * dvec0, axis=1)), 1.0e-12)
    dvec_g = dvec0 * (ell / dnorm0)[:, None]

    area = cp.maximum(geom.area.astype(cp.float64, copy=False), 1.0e-300)
    tproj = area / ell
    tproj = cp.maximum(tproj, float(getattr(cfg, "transmissibility_floor", 1.0e-12)))
    cp.cuda.Stream.null.synchronize()

    variant = str(operator_variant or "geodesic_face").lower().strip()
    if variant in {"geodesic_weights", "geodesic_weights_only", "graph_geodesic_weights"}:
        tproj_use = geom.tproj.astype(cp.float64, copy=False)
        dvec_use = dvec0
        face_operator = "geodesic_weights"
        operator_tproj = "existing_face_transmissibility"
        operator_dvec = "physical_cell_to_cell_vector"
    elif variant in {"geodesic_face", "graph_geodesic_face", "all", "geodesic_all"}:
        tproj_use = tproj
        dvec_use = dvec_g
        face_operator = "geodesic_face"
        operator_tproj = "area_over_graph_geodesic_face_metric"
        operator_dvec = "physical_direction_with_graph_geodesic_length"
    else:
        raise ValueError(f"Unknown geodesic face operator variant: {operator_variant}")

    return {
        "tproj": tproj_use.astype(cp.float64, copy=False),
        "w_owner": w_owner.astype(cp.float64, copy=False),
        "w_neigh": w_neigh.astype(cp.float64, copy=False),
        "dvec": dvec_use.astype(cp.float64, copy=False),
        "ell_g": ell.astype(cp.float64, copy=False),
        "r_owner_g": ro.astype(cp.float64, copy=False),
        "r_neigh_g": rn.astype(cp.float64, copy=False),
        "facelet_count": facelet_count[safe_pos],
        "meta": {
            "face_operator": face_operator,
            "face_operator_variant": variant,
            "face_exchange_distance": "graph_geodesic_site_to_face",
            "face_interpolation_weights": "graph_geodesic_side_lengths",
            "operator_dvec": operator_dvec,
            "operator_tproj": operator_tproj,
            "operator_face_metric_s": float(time.perf_counter() - t0),
            "operator_ell_g_mean": float(cp.mean(ell).get()) if int(ell.size) else 0.0,
            "operator_ell_g_p90": float(cp.quantile(ell, 0.90).get()) if int(ell.size) else 0.0,
            "operator_tproj_mean": float(cp.mean(tproj_use).get()) if int(tproj_use.size) else 0.0,
            "operator_tproj_over_original_sum": float(
                (cp.sum(tproj_use) / cp.maximum(cp.sum(geom.tproj.astype(cp.float64)), 1.0e-300)).get()
            ),
            "darcy_readout_length": "physical_axis_flux_readout",
        },
    }


def physical_flux_readout(ns: dict[str, Any], geom: Any, phi: Any, physical_dvec: Any | None = None) -> Any:
    cp = ns["cp"]
    dvec = geom.dvec if physical_dvec is None else cp.asarray(physical_dvec, dtype=cp.float64)
    return cp.sum(phi[:, None] * dvec.astype(cp.float64), axis=0) / cp.maximum(cp.sum(geom.volume), 1.0e-300)
