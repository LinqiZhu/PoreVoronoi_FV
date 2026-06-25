from __future__ import annotations

import time
from dataclasses import replace
from typing import Any


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


def build_geodesic_face_metric(ns: dict[str, Any], geom: Any, cfg: Any) -> dict[str, Any]:
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

    return {
        "tproj": tproj.astype(cp.float64, copy=False),
        "w_owner": w_owner.astype(cp.float64, copy=False),
        "w_neigh": w_neigh.astype(cp.float64, copy=False),
        "dvec": dvec_g.astype(cp.float64, copy=False),
        "ell_g": ell.astype(cp.float64, copy=False),
        "r_owner_g": ro.astype(cp.float64, copy=False),
        "r_neigh_g": rn.astype(cp.float64, copy=False),
        "facelet_count": facelet_count[safe_pos],
        "meta": {
            "face_exchange_distance": "graph_geodesic_site_to_face",
            "face_interpolation_weights": "graph_geodesic_side_lengths",
            "operator_dvec": "physical_direction_with_graph_geodesic_length",
            "operator_tproj": "area_over_graph_geodesic_face_metric",
            "operator_face_metric_s": float(time.perf_counter() - t0),
            "operator_ell_g_mean": float(cp.mean(ell).get()) if int(ell.size) else 0.0,
            "operator_ell_g_p90": float(cp.quantile(ell, 0.90).get()) if int(ell.size) else 0.0,
            "operator_tproj_mean": float(cp.mean(tproj).get()) if int(tproj.size) else 0.0,
            "operator_tproj_over_original_sum": float(
                (cp.sum(tproj) / cp.maximum(cp.sum(geom.tproj.astype(cp.float64)), 1.0e-300)).get()
            ),
            "darcy_readout_length": "physical_axis_flux_readout",
        },
    }


def physical_flux_readout(ns: dict[str, Any], geom: Any, phi: Any, physical_dvec: Any | None = None) -> Any:
    cp = ns["cp"]
    dvec = geom.dvec if physical_dvec is None else cp.asarray(physical_dvec, dtype=cp.float64)
    return cp.sum(phi[:, None] * dvec.astype(cp.float64), axis=0) / cp.maximum(cp.sum(geom.volume), 1.0e-300)
