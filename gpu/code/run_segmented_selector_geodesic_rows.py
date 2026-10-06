from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(r".")
PACKAGE = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE / "code"
RUNNER_PATH = CODE_DIR / "flow_runner.py"
if str(CODE_DIR) in sys.path:
    sys.path.remove(str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR))

from geodesic_face_operator import (  # noqa: E402
    build_signed_normal_face_groups,
    distribute_aggregate_flux_to_signed_groups,
    reconstruct_rank_aware_velocity_from_signed_flux,
)

K_REF = {
    "bentheimer_sandstone_crop": 0.05225707669002652,
    "fibrous_filter_proxy": 0.1373584960111365,
}

REFERENCE_TOTAL_S = {
    "bentheimer_sandstone_crop": 242.67442080006003,
    "fibrous_filter_proxy": 256.2604510000674,
}


def load_runner():
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("flow_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_roi_environment() -> None:
    os.environ["PVFV_ROIJFA_C2_CORE"] = "1"
    os.environ["PVFV_ROIJFA_SPARSE_VOXELS"] = "1"
    os.environ["PVFV_ROIJFA_STAMPING_MODE"] = "D_c2_geodesic_ball"
    os.environ["PVFV_ROIJFA_TILE"] = "8,8,16"
    os.environ["PVFV_ROIJFA_REGULAR_STRIDE_HOTPATH"] = "0"
    os.environ["PVFV_LABEL_BACKEND"] = "roi_jfa"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _host_array(cp, value) -> np.ndarray:
    return np.asarray(cp.asnumpy(value))


def summarize_signed_geometry(ns: dict[str, Any], geom, groups: dict[str, Any]) -> dict[str, Any]:
    cp = ns["cp"]
    volume = geom.volume.astype(cp.float64, copy=False)
    total_volume = cp.maximum(cp.sum(volume), 1.0e-300)
    aggregate_rank3 = cp.sum(volume * (groups["aggregate_rank"] == 3)) / total_volume
    facelet_rank3 = cp.sum(volume * (groups["facelet_rank"] == 3)) / total_volume

    def finite_quantile(values, q: float) -> float:
        finite = values[cp.isfinite(values)]
        if int(finite.size) == 0:
            return float("nan")
        return float(cp.quantile(finite, q).get())

    chi = groups["edge_chi"]
    return {
        "signed_group_count": int(groups["group_id"].size),
        "edge_chi_mean": float(cp.mean(chi).get()),
        "edge_chi_min": float(cp.min(chi).get()),
        "edge_chi_p10": float(cp.quantile(chi, 0.10).get()),
        "edge_chi_p50": float(cp.quantile(chi, 0.50).get()),
        "edge_chi_p90": float(cp.quantile(chi, 0.90).get()),
        "aggregate_rank3_volume_fraction": float(aggregate_rank3.get()),
        "facelet_rank3_volume_fraction": float(facelet_rank3.get()),
        "aggregate_condition_p50_full_rank": finite_quantile(groups["aggregate_condition"], 0.50),
        "aggregate_condition_p90_full_rank": finite_quantile(groups["aggregate_condition"], 0.90),
        "facelet_condition_p50_full_rank": finite_quantile(groups["facelet_condition"], 0.50),
        "facelet_condition_p90_full_rank": finite_quantile(groups["facelet_condition"], 0.90),
        "graph_length_cv_mean": float(cp.mean(groups["edge_ell_cv"]).get()),
        "graph_length_cv_p90": float(cp.quantile(groups["edge_ell_cv"], 0.90).get()),
        **{f"invariant_{key}": value for key, value in groups["invariants"].items()},
    }


def geometry_group_audit_rows(
    ns: dict[str, Any],
    groups: dict[str, Any],
    *,
    case: str,
    wall_mode: str,
) -> list[dict[str, Any]]:
    cp = ns["cp"]
    fields = {
        key: _host_array(cp, groups[key])
        for key in (
            "group_id",
            "group_owner",
            "group_neigh",
            "group_parent",
            "group_normal",
            "group_area",
            "group_avec",
            "group_centroid",
            "group_r_owner_g",
            "group_r_neigh_g",
            "group_ell_g",
            "group_facelet_count",
            "group_theta_area",
            "group_tproj_exact",
        )
    }
    edge_chi = _host_array(cp, groups["edge_chi"])
    edge_group_count = _host_array(cp, groups["edge_group_count"])
    edge_ell_cv = _host_array(cp, groups["edge_ell_cv"])
    rows: list[dict[str, Any]] = []
    for group_index in range(int(fields["group_id"].size)):
        parent = int(fields["group_parent"][group_index])
        group_id = int(fields["group_id"][group_index])
        normal = fields["group_normal"][group_index]
        avec = fields["group_avec"][group_index]
        centroid = fields["group_centroid"][group_index]
        rows.append(
            {
                "case": case,
                "wall_mode": wall_mode,
                "group_index": group_index,
                "parent_edge": parent,
                "owner": int(fields["group_owner"][group_index]),
                "neigh": int(fields["group_neigh"][group_index]),
                "normal_group": groups["group_names"][group_id],
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "area": float(fields["group_area"][group_index]),
                "area_vector_x": float(avec[0]),
                "area_vector_y": float(avec[1]),
                "area_vector_z": float(avec[2]),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "r_owner_g": float(fields["group_r_owner_g"][group_index]),
                "r_neigh_g": float(fields["group_r_neigh_g"][group_index]),
                "ell_g": float(fields["group_ell_g"][group_index]),
                "facelet_count": int(fields["group_facelet_count"][group_index]),
                "theta_area": float(fields["group_theta_area"][group_index]),
                "tproj_exact": float(fields["group_tproj_exact"][group_index]),
                "parent_chi": float(edge_chi[parent]),
                "parent_group_count": int(edge_group_count[parent]),
                "parent_graph_length_cv": float(edge_ell_cv[parent]),
            }
        )
    return rows


def geometry_edge_audit_rows(
    ns: dict[str, Any],
    geom,
    groups: dict[str, Any],
    *,
    case: str,
    wall_mode: str,
) -> list[dict[str, Any]]:
    cp = ns["cp"]
    names = (
        "edge_owner_sorted",
        "edge_neigh_sorted",
        "edge_area_from_groups",
        "edge_avec_sorted",
        "edge_avec_from_groups",
        "edge_tproj_from_groups",
        "edge_facelet_count",
        "edge_group_count",
        "edge_chi",
        "edge_ell_mean",
        "edge_ell_cv",
    )
    data = {name: _host_array(cp, groups[name]) for name in names}
    area = _host_array(cp, geom.area)
    tproj = _host_array(cp, geom.tproj)
    rows: list[dict[str, Any]] = []
    for edge in range(int(area.size)):
        avec = data["edge_avec_sorted"][edge]
        avec_groups = data["edge_avec_from_groups"][edge]
        rows.append(
            {
                "case": case,
                "wall_mode": wall_mode,
                "edge": edge,
                "owner": int(data["edge_owner_sorted"][edge]),
                "neigh": int(data["edge_neigh_sorted"][edge]),
                "area": float(area[edge]),
                "area_from_groups": float(data["edge_area_from_groups"][edge]),
                "area_vector_x": float(avec[0]),
                "area_vector_y": float(avec[1]),
                "area_vector_z": float(avec[2]),
                "group_area_vector_x": float(avec_groups[0]),
                "group_area_vector_y": float(avec_groups[1]),
                "group_area_vector_z": float(avec_groups[2]),
                "chi": float(data["edge_chi"][edge]),
                "facelet_count": int(data["edge_facelet_count"][edge]),
                "signed_group_count": int(data["edge_group_count"][edge]),
                "graph_length_mean": float(data["edge_ell_mean"][edge]),
                "graph_length_cv": float(data["edge_ell_cv"][edge]),
                "tproj": float(tproj[edge]),
                "tproj_from_exact_split": float(data["edge_tproj_from_groups"][edge]),
            }
        )
    return rows


def geometry_cell_audit_rows(
    ns: dict[str, Any],
    geom,
    groups: dict[str, Any],
    *,
    case: str,
    wall_mode: str,
) -> list[dict[str, Any]]:
    cp = ns["cp"]
    names = (
        "aggregate_rank",
        "facelet_rank",
        "aggregate_condition",
        "facelet_condition",
        "aggregate_eigvals",
        "facelet_eigvals",
        "cell_incident_edge_count",
        "cell_incident_group_count",
        "cell_chi_area_mean",
        "cell_geodesic_aspect",
        "cell_wall_coefficient_density",
    )
    data = {name: _host_array(cp, groups[name]) for name in names}
    volume = _host_array(cp, geom.volume)
    rows: list[dict[str, Any]] = []
    for cell in range(int(geom.n_cells)):
        agg_eigs = data["aggregate_eigvals"][cell]
        face_eigs = data["facelet_eigvals"][cell]
        rows.append(
            {
                "case": case,
                "wall_mode": wall_mode,
                "cell": cell,
                "volume": float(volume[cell]),
                "aggregate_rank": int(data["aggregate_rank"][cell]),
                "facelet_rank": int(data["facelet_rank"][cell]),
                "aggregate_condition": float(data["aggregate_condition"][cell]),
                "facelet_condition": float(data["facelet_condition"][cell]),
                "aggregate_eigenvalue_0": float(agg_eigs[0]),
                "aggregate_eigenvalue_1": float(agg_eigs[1]),
                "aggregate_eigenvalue_2": float(agg_eigs[2]),
                "facelet_eigenvalue_0": float(face_eigs[0]),
                "facelet_eigenvalue_1": float(face_eigs[1]),
                "facelet_eigenvalue_2": float(face_eigs[2]),
                "incident_edge_count": int(data["cell_incident_edge_count"][cell]),
                "incident_signed_group_count": int(data["cell_incident_group_count"][cell]),
                "area_weighted_chi": float(data["cell_chi_area_mean"][cell]),
                "geodesic_aspect": float(data["cell_geodesic_aspect"][cell]),
                "wall_coefficient_density": float(data["cell_wall_coefficient_density"][cell]),
            }
        )
    return rows


def pressure_gradient_weight(ns: dict[str, Any], geom, cfg):
    cp = ns["cp"]
    mode = str(getattr(cfg, "pressure_gradient_weight", "tproj") or "tproj").lower().strip()
    if mode in {"", "tproj", "transmissibility"}:
        return geom.tproj.astype(cp.float64, copy=False)
    if mode == "area":
        return cp.maximum(geom.area.astype(cp.float64, copy=False), 1.0e-300)
    if mode == "sqrt_tproj_area":
        return cp.sqrt(
            cp.maximum(geom.tproj.astype(cp.float64, copy=False), 1.0e-300)
            * cp.maximum(geom.area.astype(cp.float64, copy=False), 1.0e-300)
        )
    if mode == "unit":
        return cp.ones_like(geom.tproj, dtype=cp.float64)
    raise ValueError(f"Unsupported pressure_gradient_weight: {mode}")


def make_cfg(runner, ns: dict[str, Any], out_dir: Path, args: argparse.Namespace):
    cfg = ns["pvfv_base_cfg"](out_dir=str(out_dir), profile="production")
    cfg.enable_convection = False
    cfg.pressure_gauge_eps = float(args.pressure_gauge_eps)
    cfg.tikhonov = float(args.tikhonov)
    cfg.velocity_reconstruct_from_flux = False
    cfg.velocity_reconstruction_lambda = 0.0
    cfg.velocity_reconstruction_tikhonov = float(args.velocity_reconstruction_tikhonov)
    setattr(cfg, "pressure_gradient_weight", str(args.pressure_gradient_weight))
    cfg.dt = float(args.dt)
    cfg.n_steps = int(args.n_steps)
    cfg.report_every = int(args.report_every)
    cfg.initial_velocity_mode = "zero"
    setattr(cfg, "solver_identity_audit", bool(args.solver_identity_audit))
    cloned = runner.clone_cfg(
        ns,
        cfg,
        out_dir=str(out_dir),
        transmissibility_ratio_clip=ns["_pb618_face_clip_for_mode"](str(args.face_mode)),
        explicit_nonorthogonal_correction=0.0,
    )
    cloned.tikhonov = float(args.tikhonov)
    cloned.velocity_reconstruct_from_flux = False
    cloned.velocity_reconstruction_lambda = 0.0
    cloned.velocity_reconstruction_tikhonov = float(args.velocity_reconstruction_tikhonov)
    setattr(cloned, "pressure_gradient_weight", str(args.pressure_gradient_weight))
    cloned.pressure_gauge_eps = float(args.pressure_gauge_eps)
    cloned.dt = float(args.dt)
    cloned.n_steps = int(args.n_steps)
    cloned.report_every = int(args.report_every)
    cloned.initial_velocity_mode = "zero"
    setattr(cloned, "solver_identity_audit", bool(args.solver_identity_audit))
    return cloned


def load_reference_npz(ns: dict[str, Any], path: Path, cfg):
    if not path.exists():
        return None, None
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    import numpy as np

    data = np.load(path, allow_pickle=True)
    n = int(data["volume"].shape[0])
    owner = cp.asarray(data["owner"].astype(np.int32))
    neigh = cp.asarray(data["neigh"].astype(np.int32))
    tproj = cp.asarray(data["tproj"].astype(np.float64))
    rows = cp.concatenate([owner, owner, neigh, neigh]).astype(cp.int32)
    cols = cp.concatenate([owner, neigh, neigh, owner]).astype(cp.int32)
    vals = cp.concatenate([tproj, -tproj, tproj, -tproj]).astype(cp.float64)
    laplacian = cpx_sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    eps = float(getattr(cfg, "pressure_gauge_eps", 0.0))
    if eps > 0.0:
        laplacian = laplacian + eps * cpx_sp.eye(n, dtype=cp.float64, format="csr")
    geom = ns["GPUFVMGeometry"](
        mask=cp.asarray(data["mask"].astype(bool)),
        labels=cp.asarray(data["labels"].astype(np.int32)),
        dist=cp.zeros(tuple(data["mask"].shape), dtype=cp.float64),
        n_cells=n,
        volume=cp.asarray(data["volume"].astype(np.float64)),
        centroid=cp.asarray(data["centroid"].astype(np.float64)),
        owner=owner,
        neigh=neigh,
        area=cp.asarray(data["area"].astype(np.float64)),
        avec=cp.asarray(data["avec"].astype(np.float64)),
        face_centroid=cp.asarray(data["face_centroid"].astype(np.float64)),
        dvec=cp.asarray(data["dvec"].astype(np.float64)),
        tproj=tproj,
        w_owner=cp.full(owner.shape, 0.5, dtype=cp.float64),
        w_neigh=cp.full(owner.shape, 0.5, dtype=cp.float64),
        twall=cp.asarray(data["twall"].astype(np.float64)),
        laplacian=laplacian,
    )
    result = {
        "U": cp.asarray(data["U"].astype(np.float64)),
        "p": cp.asarray(data["p"].astype(np.float64)),
        "phi": cp.asarray(data["phi"].astype(np.float64)),
        "div": cp.asarray(data["div"].astype(np.float64)),
        "K_eff_x": float(data["K_eff_x"]),
        "elapsed_s": float(data["elapsed_s"]) if "elapsed_s" in data else float(REFERENCE_TOTAL_S.get(path.stem.replace("__reference", ""), 0.0)),
        "pvfv_total_time_s": float(REFERENCE_TOTAL_S.get(path.stem.replace("__reference", ""), 0.0)),
    }
    return geom, result


def install_namespace(runner, args: argparse.Namespace):
    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    runner.install_runtime_cfg_attr_preservation(ns)
    runner.install_skip_zero_area_diagnostic(ns)
    runner.install_gpu_face_connected_split(ns, verify=False)
    runner.install_roi_backend(ns, runner.FLOW_NOTEBOOK.parent)
    runner.install_lsq_gradient_batched_compat(ns)

    fixed_wall_clone = ns["_pb618_clone_geometry_with_scaled_twall"]
    wall_args = types.SimpleNamespace(
        wall_closure_mode="global_comp_power",
        wall_ref_ccomp=float(args.wall_ref_ccomp),
        wall_length_exponent=float(args.wall_length_exponent),
        wall_size_exponent=float(args.wall_length_exponent) / 3.0,
        wall_scale_min=0.05,
        wall_scale_max=20.0,
    )
    runner.install_resolution_aware_wall_closure(ns, wall_args)
    global_wall_clone = ns["_pb618_clone_geometry_with_scaled_twall"]
    runner.install_geodesic_face_operator(ns, mode=str(args.face_operator))
    return ns, fixed_wall_clone, global_wall_clone


def build_seed(runner, ns: dict[str, Any], case: str):
    if case == "bentheimer_sandstone_crop":
        mask = runner.load_mask_npz(ns, runner.BENTHEIMER_INPUT)
        stride = (2, 4, 4)
        offset = (0, 1, 2)
        seed_spec = {
            "stage": "segmented_selector_geodesic_rows",
            "family": "stride_offset",
            "seed_id": "selected_stride_2x4x4_offset_0_1_2",
            "stride_zyx": stride,
            "offset_zyx": offset,
            "repeat": 0,
        }
        seed_flat = ns["pvfv_stride_offset_seed_flat_gpu"](mask, stride, offset)
        rule = "selected offset (0,1,2), stride (2,4,4)"
        return mask, seed_spec, seed_flat, rule

    if case == "fibrous_filter_proxy":
        mask = runner.load_mask_npz(ns, runner.FIBROUS_INPUT)
        stride = (2, 4, 4)
        half = ns["pvfv_offset_from_mode"](stride, "half")
        target = int(ns["pvfv_stride_offset_seed_flat_gpu"](mask, stride, half).size)
        seed_spec = {
            "stage": "segmented_selector_geodesic_rows",
            "family": "wall_biased",
            "seed_id": "wall_biased_stride_2x4x4_rng8400",
            "stride_zyx": stride,
            "offset_zyx": "wall_biased",
            "repeat": 0,
            "target_seed_count": target,
            "rng_seed": 8400,
        }
        seed_flat = ns["pvfv_wall_biased_seed_flat_gpu"](mask, target, rng_seed=8400)
        rule = "wall-biased deterministic candidate, target=half stride (2,4,4), rng_seed=8400"
        return mask, seed_spec, seed_flat, rule

    raise ValueError(f"Unsupported case: {case}")


def assemble_diagnostic_kkt_operators(ns: dict[str, Any], geom, cfg) -> dict[str, Any]:
    cp = ns["cp"]
    n = int(geom.n_cells)
    lap = geom.laplacian.toarray()
    eps = float(getattr(cfg, "pressure_gauge_eps", 0.0))
    if eps > 0.0:
        lap = lap - eps * cp.eye(n, dtype=cp.float64)
    a0 = float(cfg.nu) * (lap + cp.diag(geom.twall))
    a_vel = cp.kron(a0, cp.eye(3, dtype=cp.float64))

    owner = geom.owner
    neigh = geom.neigh
    dvec = geom.dvec
    weight = pressure_gradient_weight(ns, geom, cfg)

    moment = cp.zeros((n, 3, 3), dtype=cp.float64)
    coeff = cp.zeros((n, 3, n), dtype=cp.float64)
    for a in range(3):
        edge_coeff = weight * dvec[:, a]
        cp.add.at(coeff, (owner, a, neigh), edge_coeff)
        cp.add.at(coeff, (owner, a, owner), -edge_coeff)
        cp.add.at(coeff, (neigh, a, neigh), edge_coeff)
        cp.add.at(coeff, (neigh, a, owner), -edge_coeff)
        for b in range(3):
            mval = weight * dvec[:, a] * dvec[:, b]
            cp.add.at(moment, (owner, a, b), mval)
            cp.add.at(moment, (neigh, a, b), mval)

    diag = cp.arange(3)
    moment[:, diag, diag] += float(getattr(cfg, "tikhonov", 1.0e-12))
    grad = cp.einsum("iab,ibk->iak", cp.linalg.inv(moment), coeff).reshape((3 * n, n))
    pressure_coupling = grad * (geom.volume.repeat(3)[:, None] / float(cfg.rho))

    div = cp.zeros((n, 3 * n), dtype=cp.float64)
    for c in range(3):
        owner_coeff = geom.w_owner * geom.avec[:, c]
        neigh_coeff = geom.w_neigh * geom.avec[:, c]
        owner_col = 3 * owner + c
        neigh_col = 3 * neigh + c
        cp.add.at(div, (owner, owner_col), owner_coeff)
        cp.add.at(div, (owner, neigh_col), neigh_coeff)
        cp.add.at(div, (neigh, owner_col), -owner_coeff)
        cp.add.at(div, (neigh, neigh_col), -neigh_coeff)

    return {
        "laplacian_no_gauge": lap,
        "momentum_scalar": a0,
        "momentum": a_vel,
        "gradient": grad,
        "pressure_coupling": pressure_coupling,
        "divergence": div,
    }


def _gpu_array_sha256(cp, array) -> str:
    host = cp.asnumpy(cp.ascontiguousarray(array))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in host.shape)).encode("ascii"))
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(memoryview(host).cast("B"))
    return digest.hexdigest()


def _adjoint_defect(cp, left, right) -> dict[str, float | int]:
    plus = cp.linalg.norm(left - right)
    minus = cp.linalg.norm(left + right)
    denominator = cp.linalg.norm(left) + cp.linalg.norm(right)
    use_plus = bool((plus <= minus).get())
    numerator = plus if use_plus else minus
    return {
        "best_sign": 1 if use_plus else -1,
        "normalized_frobenius_defect": float((numerator / cp.maximum(denominator, 1.0e-300)).get()),
        "left_frobenius_norm": float(cp.linalg.norm(left).get()),
        "right_frobenius_norm": float(cp.linalg.norm(right).get()),
    }


def build_solver_identity_audit(ns: dict[str, Any], geom, cfg, operators, kkt) -> dict[str, Any]:
    cp = ns["cp"]
    n = int(geom.n_cells)
    gradient = operators["gradient"]
    pressure_coupling = operators["pressure_coupling"]
    divergence = operators["divergence"]
    a0 = operators["momentum_scalar"]
    lap = operators["laplacian_no_gauge"]

    kkt_norm = cp.linalg.norm(kkt)
    kkt_symmetry = cp.linalg.norm(kkt - kkt.T) / cp.maximum(2.0 * kkt_norm, 1.0e-300)
    a0_symmetry = cp.linalg.norm(a0 - a0.T) / cp.maximum(2.0 * cp.linalg.norm(a0), 1.0e-300)
    lap_symmetry = cp.linalg.norm(lap - lap.T) / cp.maximum(2.0 * cp.linalg.norm(lap), 1.0e-300)

    probe_count = min(4, max(n - 1, 1))
    index = cp.arange(n, dtype=cp.float64) + 1.0
    probes = []
    for probe_id in range(probe_count):
        frequency = float(probe_id + 1)
        probe = cp.sin(frequency * index) + cp.cos((frequency + 0.375) * index)
        probe -= cp.mean(probe)
        probes.append(probe)
    probe_matrix = cp.stack(probes, axis=1)
    velocity_rhs = (pressure_coupling @ probe_matrix).reshape(n, 3, probe_count)
    velocity_solution = cp.linalg.solve(a0, velocity_rhs.reshape(n, 3 * probe_count)).reshape(
        n, 3, probe_count
    )
    schur_action = divergence @ velocity_solution.reshape(3 * n, probe_count)
    lap_action = lap @ probe_matrix
    schur_probe_rows: list[dict[str, float | int]] = []
    for probe_id in range(probe_count):
        schur_probe = schur_action[:, probe_id]
        lap_probe = lap_action[:, probe_id]
        scale = cp.vdot(lap_probe, schur_probe).real / cp.maximum(cp.vdot(lap_probe, lap_probe).real, 1.0e-300)
        residual = schur_probe - scale * lap_probe
        schur_probe_rows.append(
            {
                "probe": int(probe_id),
                "best_laplacian_scale": float(scale.get()),
                "relative_residual_after_best_scale": float(
                    (cp.linalg.norm(residual) / cp.maximum(cp.linalg.norm(schur_probe), 1.0e-300)).get()
                ),
                "schur_rayleigh": float(
                    (cp.vdot(probe_matrix[:, probe_id], schur_probe).real / cp.maximum(
                        cp.vdot(probe_matrix[:, probe_id], probe_matrix[:, probe_id]).real,
                        1.0e-300,
                    )).get()
                ),
                "laplacian_rayleigh": float(
                    (cp.vdot(probe_matrix[:, probe_id], lap_probe).real / cp.maximum(
                        cp.vdot(probe_matrix[:, probe_id], probe_matrix[:, probe_id]).real,
                        1.0e-300,
                    )).get()
                ),
            }
        )

    exact_spectrum: dict[str, Any]
    if n <= 96:
        pressure_rhs = pressure_coupling.reshape(n, 3, n)
        pressure_velocity = cp.linalg.solve(a0, pressure_rhs.reshape(n, 3 * n)).reshape(n, 3, n)
        schur = divergence @ pressure_velocity.reshape(3 * n, n)
        schur_symmetric = 0.5 * (schur + schur.T)
        eigvals = cp.linalg.eigvalsh(schur_symmetric)
        exact_spectrum = {
            "computed": True,
            "matrix": "symmetric_part_of_D_Ainv_B",
            "eigenvalues": [float(v) for v in eigvals.get().tolist()],
        }
    else:
        exact_spectrum = {
            "computed": False,
            "reason": "full spectrum restricted to n_cells <= 96; deterministic Schur probes reported instead",
        }

    return {
        "audit_schema": "solver_identity_v1",
        "diagnostic_formulation_id": "diagnostic_dense_kkt_lsq_gradient_interpolated_divergence_v1",
        "production_formulation_id": "pseudo_time_pressure_correction_BTBt_v1",
        "n_cells": n,
        "n_edges": int(geom.owner.size),
        "pressure_gauge_eps": float(getattr(cfg, "pressure_gauge_eps", 0.0)),
        "gradient_vs_divergence_transpose": _adjoint_defect(cp, gradient, divergence.T),
        "pressure_coupling_vs_divergence_transpose": _adjoint_defect(cp, pressure_coupling, divergence.T),
        "kkt_symmetry_defect": float(kkt_symmetry.get()),
        "momentum_scalar_symmetry_defect": float(a0_symmetry.get()),
        "pressure_laplacian_symmetry_defect": float(lap_symmetry.get()),
        "schur_vs_pressure_laplacian_probes": schur_probe_rows,
        "schur_spectrum": exact_spectrum,
        "matrix_hashes_sha256": {
            "A_scalar": _gpu_array_sha256(cp, a0),
            "G_lsq": _gpu_array_sha256(cp, gradient),
            "B_pressure_coupling": _gpu_array_sha256(cp, pressure_coupling),
            "D_interpolated_divergence": _gpu_array_sha256(cp, divergence),
            "T_edge": _gpu_array_sha256(cp, geom.tproj),
            "pressure_laplacian_no_gauge": _gpu_array_sha256(cp, lap),
            "edge_owner": _gpu_array_sha256(cp, geom.owner),
            "edge_neigh": _gpu_array_sha256(cp, geom.neigh),
        },
        "matrix_shapes": {
            "A_scalar": list(a0.shape),
            "A_velocity_structured": [3 * n, 3 * n],
            "G_lsq": list(gradient.shape),
            "B_pressure_coupling": list(pressure_coupling.shape),
            "D_interpolated_divergence": list(divergence.shape),
            "T_edge": list(geom.tproj.shape),
        },
    }


def solve_monolithic_no_momentum_residual(ns: dict[str, Any], geom, cfg) -> dict[str, Any]:
    """Run the retained dense KKT as an explicitly diagnostic formulation."""
    cp = ns["cp"]
    n = int(geom.n_cells)
    t0 = time.perf_counter()
    operators = assemble_diagnostic_kkt_operators(ns, geom, cfg)
    a_vel = operators["momentum"]
    pressure_coupling = operators["pressure_coupling"]
    div = operators["divergence"]

    size = 4 * n + 1
    kkt = cp.zeros((size, size), dtype=cp.float64)
    rhs = cp.zeros(size, dtype=cp.float64)
    kkt[: 3 * n, : 3 * n] = a_vel
    kkt[: 3 * n, 3 * n : 4 * n] = pressure_coupling
    kkt[3 * n : 4 * n, : 3 * n] = div
    kkt[3 * n : 4 * n, 4 * n] = 1.0
    kkt[4 * n, 3 * n : 4 * n] = 1.0 / float(n)

    body = cp.asarray(cfg.body_force, dtype=cp.float64)
    rhs[: 3 * n] = (geom.volume[:, None] * body[None, :]).reshape(3 * n)

    identity_audit = None
    if bool(getattr(cfg, "solver_identity_audit", False)):
        identity_audit = build_solver_identity_audit(ns, geom, cfg, operators, kkt)

    sol = cp.linalg.solve(kkt, rhs)
    velocity = sol[: 3 * n].reshape((n, 3))
    pressure = sol[3 * n : 4 * n]
    phi = ns["face_flux_from_velocity_gpu"](velocity, geom)
    div_phi = ns["face_divergence_gpu"](phi, geom)
    cp.cuda.Stream.null.synchronize()

    mean_u = cp.sum(velocity * geom.volume[:, None], axis=0) / cp.maximum(cp.sum(geom.volume), 1.0e-300)
    k_mean = float((float(cfg.nu) * mean_u[0] / float(cfg.body_force[0])).get())
    kkt_momentum_residual = a_vel @ velocity.reshape(3 * n) + pressure_coupling @ pressure - rhs[: 3 * n]
    production_momentum_inf = float("nan")
    production_momentum_error = ""
    try:
        production_momentum = ns["steady_momentum_residual_gpu"](velocity, pressure, phi, geom, cfg)
        production_momentum_inf = float(cp.max(cp.linalg.norm(production_momentum, axis=1)).get())
    except Exception as exc:
        production_momentum_error = repr(exc)
    return {
        "U": velocity,
        "U_solve": velocity,
        "p": pressure,
        "phi": phi,
        "div": div_phi,
        "elapsed_s": float(time.perf_counter() - t0),
        "K_eff_x_mean_velocity": k_mean,
        "mass_inf_per_volume": float(cp.max(cp.abs(div_phi / geom.volume)).get()),
        "umax": float(cp.max(cp.linalg.norm(velocity, axis=1)).get()),
        "steady_momentum_inf": production_momentum_inf,
        "steady_momentum_evaluation_error": production_momentum_error,
        "diagnostic_kkt_momentum_residual_inf": float(cp.max(cp.abs(kkt_momentum_residual)).get()),
        "solver_formulation_id": "diagnostic_dense_kkt_lsq_gradient_interpolated_divergence_v1",
        "solver_identity": identity_audit,
    }


def solve_production_pressure_correction(ns: dict[str, Any], geom, cfg) -> dict[str, Any]:
    """Run the pseudo-time pressure-correction path described by the manuscript."""
    result = dict(ns["run_velocity_pressure_projection_gpu"](geom, cfg, initial_U=None, run_label="segmented_selector"))
    cp = ns["cp"]
    result["U_solve"] = result["U"]
    result["K_eff_x_mean_velocity"] = float(result.get("K_eff_x", float("nan")))
    result["umax"] = float(cp.max(cp.linalg.norm(result["U"], axis=1)).get())
    result["solver_formulation_id"] = "pseudo_time_pressure_correction_BTBt_v1"
    result["solver_identity"] = None
    result["diagnostic_kkt_momentum_residual_inf"] = float("nan")
    result["steady_momentum_evaluation_error"] = ""
    return result


def parse_float_list(text: str) -> list[float]:
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        return [0.0]
    return [float(item) for item in items]


def reconstruct_velocity_from_flux_gpu_compat(ns: dict[str, Any], phi, U_prior, geom, cfg):
    cp = ns["cp"]
    n = int(geom.n_cells)
    nf = int(geom.owner.size)
    M = cp.zeros((n, 3, 3), dtype=cp.float64)
    b = cp.zeros((n, 3), dtype=cp.float64)
    grid, block = ns["_threads_blocks"](nf)
    ns["_FLUX_RECON_ASSEMBLE_KERNEL"](
        grid,
        block,
        (
            nf,
            geom.owner,
            geom.neigh,
            geom.avec.reshape(-1),
            geom.area,
            phi,
            M.reshape(-1),
            b.reshape(-1),
        ),
    )
    lam = float(getattr(cfg, "velocity_reconstruction_lambda", 0.0))
    diag = cp.arange(3)
    velocity_tikhonov = float(
        getattr(cfg, "velocity_reconstruction_tikhonov", getattr(cfg, "tikhonov", 1.0e-12))
    )
    M[:, diag, diag] += lam + velocity_tikhonov
    b += lam * U_prior
    wall_mode = str(getattr(cfg, "velocity_reconstruction_wall_mode", "none") or "none").lower().strip()
    wall_multiplier = float(getattr(cfg, "velocity_reconstruction_wall_multiplier", 0.0))
    if wall_mode in {"none", "", "off"} or wall_multiplier == 0.0:
        pass
    elif wall_mode in {"inv_ccomp", "inverse_ccomp", "wall_inv_ccomp"}:
        ccomp = float(cp.sum(geom.mask).get()) / max(float(n), 1.0)
        M[:, diag, diag] += wall_multiplier * (1.0 / max(ccomp, 1.0e-300)) * geom.twall[:, None]
    elif wall_mode in {"twall_over_volume", "wall_over_volume"}:
        M[:, diag, diag] += wall_multiplier * (
            geom.twall[:, None] / cp.maximum(geom.volume[:, None], 1.0e-300)
        )
    elif wall_mode in {"normal_tensor", "tangent_tensor"}:
        ccomp = float(cp.sum(geom.mask).get()) / max(float(n), 1.0)
        wall_tensor_scale = wall_multiplier * (1.0 / max(ccomp, 1.0e-300))
        add_wall_tensor_penalty(ns, M, geom, cfg, wall_mode, wall_tensor_scale)
    else:
        raise ValueError(f"Unsupported velocity_reconstruction_wall_mode: {wall_mode}")
    U = cp.linalg.solve(M, b[:, :, None])[:, :, 0]
    if bool(getattr(cfg, "velocity_reconstruction_transverse_mean_zero", False)):
        mean = cp.sum(U * geom.volume[:, None], axis=0) / cp.maximum(cp.sum(geom.volume), 1.0e-300)
        U[:, 1] -= mean[1]
        U[:, 2] -= mean[2]
    return U


def add_wall_tensor_penalty(ns: dict[str, Any], M, geom, cfg, mode: str, scale: float) -> None:
    cp = ns["cp"]
    mask = geom.mask.astype(cp.bool_, copy=False)
    labels = geom.labels.astype(cp.int32, copy=False)
    centroid = geom.centroid.astype(cp.float64, copy=False)
    h = float(getattr(cfg, "voxel_size", 1.0))
    area0 = h * h
    floor_delta = max(float(getattr(cfg, "wall_distance_floor", 0.5)) * h, 1.0e-300)
    periodic_x = bool(getattr(cfg, "periodic_x", False))
    D, H, W = [int(v) for v in mask.shape]

    def add_faces(valid, axis: int, sign: float, face_coord):
        if int(cp.count_nonzero(valid).get()) == 0:
            return
        z, y, x = cp.nonzero(valid)
        lab = labels[z, y, x].astype(cp.int64)
        good = lab >= 0
        if int(cp.count_nonzero(good).get()) == 0:
            return
        z = z[good].astype(cp.float64)
        y = y[good].astype(cp.float64)
        x = x[good].astype(cp.float64)
        lab = lab[good]
        fcx = (x + 0.5) * h
        fcy = (y + 0.5) * h
        fcz = (z + 0.5) * h
        if axis == 0:
            fcx = face_coord(x) * h
        elif axis == 1:
            fcy = face_coord(y) * h
        else:
            fcz = face_coord(z) * h
        center = centroid[lab]
        if axis == 0:
            delta = (fcx - center[:, 0]) * sign
        elif axis == 1:
            delta = (fcy - center[:, 1]) * sign
        else:
            delta = (fcz - center[:, 2]) * sign
        coeff = float(scale) * area0 / cp.maximum(delta, floor_delta)
        if mode == "normal_tensor":
            cp.add.at(M, (lab, axis, axis), coeff)
        else:
            for comp in range(3):
                if comp != axis:
                    cp.add.at(M, (lab, comp, comp), coeff)

    # Axis index follows vector component order x,y,z while array order is z,y,x.
    if not periodic_x:
        valid = cp.zeros_like(mask)
        valid[:, :, 0] = mask[:, :, 0]
        add_faces(valid, 0, -1.0, lambda x: x * 0.0)
        valid = cp.zeros_like(mask)
        valid[:, :, W - 1] = mask[:, :, W - 1]
        add_faces(valid, 0, 1.0, lambda x: x * 0.0 + W)
    if W > 1:
        valid = cp.zeros_like(mask)
        valid[:, :, 1:] = mask[:, :, 1:] & (~mask[:, :, :-1])
        add_faces(valid, 0, -1.0, lambda x: x)
        valid = cp.zeros_like(mask)
        valid[:, :, :-1] = mask[:, :, :-1] & (~mask[:, :, 1:])
        add_faces(valid, 0, 1.0, lambda x: x + 1.0)
    valid = cp.zeros_like(mask)
    valid[:, 0, :] = mask[:, 0, :]
    add_faces(valid, 1, -1.0, lambda y: y * 0.0)
    valid = cp.zeros_like(mask)
    valid[:, H - 1, :] = mask[:, H - 1, :]
    add_faces(valid, 1, 1.0, lambda y: y * 0.0 + H)
    if H > 1:
        valid = cp.zeros_like(mask)
        valid[:, 1:, :] = mask[:, 1:, :] & (~mask[:, :-1, :])
        add_faces(valid, 1, -1.0, lambda y: y)
        valid = cp.zeros_like(mask)
        valid[:, :-1, :] = mask[:, :-1, :] & (~mask[:, 1:, :])
        add_faces(valid, 1, 1.0, lambda y: y + 1.0)
    valid = cp.zeros_like(mask)
    valid[0, :, :] = mask[0, :, :]
    add_faces(valid, 2, -1.0, lambda z: z * 0.0)
    valid = cp.zeros_like(mask)
    valid[D - 1, :, :] = mask[D - 1, :, :]
    add_faces(valid, 2, 1.0, lambda z: z * 0.0 + D)
    if D > 1:
        valid = cp.zeros_like(mask)
        valid[1:, :, :] = mask[1:, :, :] & (~mask[:-1, :, :])
        add_faces(valid, 2, -1.0, lambda z: z)
        valid = cp.zeros_like(mask)
        valid[:-1, :, :] = mask[:-1, :, :] & (~mask[1:, :, :])
        add_faces(valid, 2, 1.0, lambda z: z + 1.0)


def geometry_error_correlation_rows(
    ns: dict[str, Any],
    geom,
    groups: dict[str, Any],
    U_state,
    U_ref,
    *,
    case: str,
    wall_mode: str,
    solver_formulation_id: str,
    velocity_state_id: str,
) -> list[dict[str, Any]]:
    cp = ns["cp"]
    error_norm = _host_array(cp, cp.linalg.norm(U_state - U_ref, axis=1)).astype(np.float64)
    volume = _host_array(cp, geom.volume).astype(np.float64)
    metrics = {
        "aggregate_condition": _host_array(cp, groups["aggregate_condition"]),
        "facelet_condition": _host_array(cp, groups["facelet_condition"]),
        "aggregate_rank": _host_array(cp, groups["aggregate_rank"]),
        "facelet_rank": _host_array(cp, groups["facelet_rank"]),
        "area_weighted_chi": _host_array(cp, groups["cell_chi_area_mean"]),
        "geodesic_aspect": _host_array(cp, groups["cell_geodesic_aspect"]),
        "wall_coefficient_density": _host_array(cp, groups["cell_wall_coefficient_density"]),
        "incident_edge_count": _host_array(cp, groups["cell_incident_edge_count"]),
        "incident_signed_group_count": _host_array(cp, groups["cell_incident_group_count"]),
    }

    def correlations(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
        x = np.asarray(x[valid], dtype=np.float64)
        y = np.asarray(y[valid], dtype=np.float64)
        weights = np.asarray(weights[valid], dtype=np.float64)
        if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
            return float("nan"), float("nan"), float("nan")
        pearson = float(np.corrcoef(x, y)[0, 1])
        wsum = max(float(np.sum(weights)), np.finfo(np.float64).tiny)
        xmean = float(np.sum(weights * x) / wsum)
        ymean = float(np.sum(weights * y) / wsum)
        covariance = float(np.sum(weights * (x - xmean) * (y - ymean)) / wsum)
        xvar = float(np.sum(weights * (x - xmean) ** 2) / wsum)
        yvar = float(np.sum(weights * (y - ymean) ** 2) / wsum)
        weighted = covariance / max(np.sqrt(xvar * yvar), np.finfo(np.float64).tiny)
        try:
            from scipy.stats import spearmanr

            spearman = float(spearmanr(x, y).statistic)
        except Exception:
            spearman = float("nan")
        return pearson, weighted, spearman

    rows: list[dict[str, Any]] = []
    for metric_name, metric in metrics.items():
        pearson, weighted, spearman = correlations(np.asarray(metric), error_norm, volume)
        rows.append(
            {
                "case": case,
                "wall_mode": wall_mode,
                "solver_formulation_id": solver_formulation_id,
                "velocity_state_id": velocity_state_id,
                "geometry_metric": metric_name,
                "error_metric": "cell_velocity_error_norm",
                "pearson": pearson,
                "volume_weighted_pearson": weighted,
                "spearman": spearman,
                "n_finite": int(np.count_nonzero(np.isfinite(metric) & np.isfinite(error_norm))),
            }
        )
    return rows


def run_case(
    runner,
    ns: dict[str, Any],
    case: str,
    out_dir: Path,
    wall_clones: list[tuple[str, Any]],
    reference_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    cp = ns["cp"]
    cfg = make_cfg(runner, ns, out_dir, args)
    mask, seed_spec, seed_flat, rule = build_seed(runner, ns, case)

    print(f"[segmented-selector] build {case}: {rule}; S={int(seed_flat.size)}", flush=True)
    build_start = time.perf_counter()
    geom0, meta = ns["pvfv_build_geometry_from_seed_flat_timed"](
        mask,
        seed_flat,
        cfg,
        split_face_components=True,
        label_mode=str(args.label_backend),
        seed_spec=str(seed_spec["seed_id"]),
    )
    physical_dvec = ns.get("_pvfv_last_physical_dvec_readout")
    build_s = float(time.perf_counter() - build_start)
    ref_geom, ref_result = load_reference_npz(ns, reference_dir / f"{case}__reference.npz", cfg)

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, list[Any]] = {
        "geometry_group": [],
        "geometry_edge": [],
        "geometry_cell": [],
        "geometry_error_correlation": [],
        "orientation_flux": [],
        "solver_identity": [],
        "execution_record": [],
        "state_export": [],
    }
    for wall_mode, wall_clone in wall_clones:
        geom = wall_clone(geom0, 1.25)
        groups = None
        geometry_summary: dict[str, Any] = {}
        if bool(args.geometry_audit) or bool(args.signed_normal_reconstruct):
            print(f"[segmented-selector] signed-normal audit {case} wall={wall_mode}", flush=True)
            groups = build_signed_normal_face_groups(ns, geom, cfg)
            geometry_summary = summarize_signed_geometry(ns, geom, groups)
            artifacts["geometry_group"].extend(
                geometry_group_audit_rows(ns, groups, case=case, wall_mode=wall_mode)
            )
            artifacts["geometry_edge"].extend(
                geometry_edge_audit_rows(ns, geom, groups, case=case, wall_mode=wall_mode)
            )
            artifacts["geometry_cell"].extend(
                geometry_cell_audit_rows(ns, geom, groups, case=case, wall_mode=wall_mode)
            )

        print(
            f"[segmented-selector] solve {case} wall={wall_mode} "
            f"formulation={args.solver_formulation} Ncv={int(geom.n_cells)}",
            flush=True,
        )
        if str(args.solver_formulation) == "diagnostic_dense_kkt":
            res = solve_monolithic_no_momentum_residual(ns, geom, cfg)
        elif str(args.solver_formulation) == "production_pressure_correction":
            res = solve_production_pressure_correction(ns, geom, cfg)
        else:
            raise ValueError(f"Unsupported solver formulation: {args.solver_formulation}")
        solver_formulation_id = str(res["solver_formulation_id"])
        if bool(args.export_state_npz):
            state_path = out_dir / f"{case}__{wall_mode}__{args.solver_formulation}__state.npz"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                state_path,
                U=_host_array(cp, res["U"]),
                U_solve=_host_array(cp, res["U_solve"]),
                p=_host_array(cp, res["p"]),
                phi=_host_array(cp, res["phi"]),
                div=_host_array(cp, res["div"]),
                volume=_host_array(cp, geom.volume),
                owner=_host_array(cp, geom.owner),
                neigh=_host_array(cp, geom.neigh),
                solver_formulation_id=np.asarray(solver_formulation_id),
            )
            artifacts["state_export"].append(
                {
                    "case": case,
                    "wall_mode": wall_mode,
                    "solver_formulation_id": solver_formulation_id,
                    "path": str(state_path),
                }
            )
        artifacts["execution_record"].append(
            {
                "case": case,
                "wall_mode": wall_mode,
                "solver_formulation_id": solver_formulation_id,
                "solver_formulation_arg": str(args.solver_formulation),
                "n_cells": int(geom.n_cells),
                "n_edges": int(geom.owner.size),
                "elapsed_s": float(res["elapsed_s"]),
                "mass_inf_per_volume": float(res["mass_inf_per_volume"]),
                "steady_momentum_inf": float(res.get("steady_momentum_inf", float("nan"))),
                "steady_momentum_evaluation_error": str(
                    res.get("steady_momentum_evaluation_error", "")
                ),
                "diagnostic_kkt_momentum_residual_inf": float(
                    res.get("diagnostic_kkt_momentum_residual_inf", float("nan"))
                ),
                "steps_completed": int(res.get("steps_completed", 1)),
                "steady_converged": bool(res.get("steady_converged", False)),
                "solver_failure": str(res.get("solver_failure", "")),
                "pressure_gauge_eps": float(getattr(cfg, "pressure_gauge_eps", 0.0)),
                "dt": float(cfg.dt),
                "n_steps_requested": int(cfg.n_steps),
            }
        )
        if res.get("solver_identity") is not None:
            identity = dict(res["solver_identity"])
            identity.update({"case": case, "wall_mode": wall_mode})
            artifacts["solver_identity"].append(identity)

        if physical_dvec is not None:
            mean_flux_u = runner.physical_flux_readout(ns, geom, res["phi"], physical_dvec)
            k_eff = float((float(cfg.nu) * mean_flux_u[0] / float(cfg.body_force[0])).get())
        else:
            k_eff = float(res["K_eff_x_mean_velocity"])
        k_ref = float(K_REF[case])
        res_for_errors = dict(res)
        res_for_errors["K_eff_x"] = k_eff
        e_phi = float("nan")
        e_u_solve = float("nan")
        u_ref = None
        if ref_geom is not None and ref_result is not None:
            try:
                u_ref, _p_ref = ns["coarsen_voxel_reference_to_coarse_gpu"](ref_geom, ref_result, geom)
                e_u_solve = float(ns["pvfv_weighted_rel_l2_vec_gpu"](res["U_solve"], u_ref, geom.volume))
            except Exception as exc:
                print(f"[segmented-selector] velocity error failed for {case}/{wall_mode}: {exc!r}", flush=True)
            try:
                e_phi = float(ns["pvfv_flux_rel_error"](ref_geom, ref_result, geom, res_for_errors))
            except Exception as exc:
                print(f"[segmented-selector] flux error failed for {case}/{wall_mode}: {exc!r}", flush=True)
        op_meta = ns.get("_pvfv_last_face_operator_meta", {}) or {}
        velocity_variants: list[dict[str, Any]] = []
        if bool(args.velocity_reconstruct_from_flux):
            for recon_lambda in parse_float_list(str(args.velocity_reconstruction_lambdas)):
                for wall_multiplier in parse_float_list(str(args.velocity_reconstruction_wall_multipliers)):
                    setattr(cfg, "velocity_reconstruction_lambda", float(recon_lambda))
                    setattr(cfg, "velocity_reconstruction_wall_multiplier", float(wall_multiplier))
                    setattr(cfg, "velocity_reconstruction_wall_mode", str(args.velocity_reconstruction_wall_mode))
                    setattr(
                        cfg,
                        "velocity_reconstruction_transverse_mean_zero",
                        bool(args.velocity_reconstruction_transverse_mean_zero),
                    )
                    t_recon = time.perf_counter()
                    U_aggregate = reconstruct_velocity_from_flux_gpu_compat(
                        ns, res["phi"], res["U_solve"], geom, cfg
                    )
                    cp.cuda.Stream.null.synchronize()
                    velocity_variants.append(
                        {
                            "U_report": U_aggregate,
                            "U_observable": None,
                            "observable_projector": None,
                            "velocity_state_id": "U_phi_aggregate_face_normal_lsq",
                            "velocity_completion_id": "fixed_ridge_aggregate_lsq",
                            "reconstructed": True,
                            "reconstruction_lambda": float(recon_lambda),
                            "wall_multiplier": float(wall_multiplier),
                            "subflux_rule_id": "aggregate_edge_flux_no_split",
                            "subflux_aggregate_max_abs": 0.0,
                            "subflux_aggregate_max_rel": 0.0,
                            "rank3_volume_fraction": float("nan"),
                            "normal_residual_l2": float("nan"),
                            "normal_residual_inf": float("nan"),
                            "elapsed_s": float(time.perf_counter() - t_recon),
                            "rank": None,
                        }
                    )

        if bool(args.signed_normal_reconstruct):
            if groups is None:
                raise RuntimeError("Signed-normal reconstruction requires signed-normal geometry groups")
            for subflux_rule in [
                item.strip() for item in str(args.signed_normal_subflux_rules).split(",") if item.strip()
            ]:
                t_recon = time.perf_counter()
                split = distribute_aggregate_flux_to_signed_groups(
                    ns,
                    groups,
                    res["phi"],
                    rule=subflux_rule,
                    velocity_predictor=res["U_solve"],
                )
                reconstruction = reconstruct_rank_aware_velocity_from_signed_flux(
                    ns,
                    groups,
                    split["group_phi"],
                    unobservable_prior=res["U_solve"],
                )
                cp.cuda.Stream.null.synchronize()
                rank = reconstruction["rank"]
                rank3_volume_fraction = float(
                    (
                        cp.sum(geom.volume * (rank == 3))
                        / cp.maximum(cp.sum(geom.volume), 1.0e-300)
                    ).get()
                )
                velocity_variants.append(
                    {
                        "U_report": reconstruction["U_completed"],
                        "U_observable": reconstruction["U_observable"],
                        "observable_projector": reconstruction["observable_projector"],
                        "velocity_state_id": f"U_phi_signed_normal_{split['rule_id']}",
                        "velocity_completion_id": reconstruction["completion_id"],
                        "reconstructed": True,
                        "reconstruction_lambda": 0.0,
                        "wall_multiplier": 0.0,
                        "subflux_rule_id": split["rule_id"],
                        "subflux_aggregate_max_abs": split["aggregate_max_abs"],
                        "subflux_aggregate_max_rel": split["aggregate_max_rel"],
                        "rank3_volume_fraction": rank3_volume_fraction,
                        "normal_residual_l2": reconstruction["normal_equation_residual_l2"],
                        "normal_residual_inf": reconstruction["normal_equation_residual_inf"],
                        "elapsed_s": float(time.perf_counter() - t_recon),
                        "rank": rank,
                    }
                )
                group_phi_host = _host_array(cp, split["group_phi"])
                group_id_host = _host_array(cp, groups["group_id"])
                group_parent_host = _host_array(cp, groups["group_parent"])
                group_area_host = _host_array(cp, groups["group_area"])
                group_theta_host = _host_array(cp, groups["group_theta_area"])
                for group_index in range(int(group_phi_host.size)):
                    group_id = int(group_id_host[group_index])
                    artifacts["orientation_flux"].append(
                        {
                            "case": case,
                            "wall_mode": wall_mode,
                            "solver_formulation_id": solver_formulation_id,
                            "subflux_rule_id": split["rule_id"],
                            "group_index": group_index,
                            "parent_edge": int(group_parent_host[group_index]),
                            "normal_group": groups["group_names"][group_id],
                            "area": float(group_area_host[group_index]),
                            "theta_area": float(group_theta_host[group_index]),
                            "group_flux": float(group_phi_host[group_index]),
                            "reference_orientation_flux_available": False,
                            "orientation_flux_error": float("nan"),
                            "aggregate_max_abs": split["aggregate_max_abs"],
                            "aggregate_max_rel": split["aggregate_max_rel"],
                        }
                    )

        if not velocity_variants:
            velocity_variants.append(
                {
                    "U_report": res["U_solve"],
                    "U_observable": None,
                    "observable_projector": None,
                    "velocity_state_id": "U_solve",
                    "velocity_completion_id": "none",
                    "reconstructed": False,
                    "reconstruction_lambda": float("nan"),
                    "wall_multiplier": 0.0,
                    "subflux_rule_id": "none",
                    "subflux_aggregate_max_abs": 0.0,
                    "subflux_aggregate_max_rel": 0.0,
                    "rank3_volume_fraction": float("nan"),
                    "normal_residual_l2": float("nan"),
                    "normal_residual_inf": float("nan"),
                    "elapsed_s": 0.0,
                    "rank": None,
                }
            )

        for variant in velocity_variants:
                U_report = variant["U_report"]
                recon_lambda = variant["reconstruction_lambda"]
                wall_multiplier = variant["wall_multiplier"]
                setattr(cfg, "velocity_reconstruction_wall_multiplier", float(wall_multiplier))
                setattr(cfg, "velocity_reconstruction_wall_mode", str(args.velocity_reconstruction_wall_mode))
                setattr(
                    cfg,
                    "velocity_reconstruction_transverse_mean_zero",
                    bool(args.velocity_reconstruction_transverse_mean_zero),
                )
                reconstruct_s = float(variant["elapsed_s"])

                e_u = float("nan")
                e_u_observable = float("nan")
                e_u_observable_zero_full = float("nan")
                e_u_components = {"x": float("nan"), "y": float("nan"), "z": float("nan")}
                if u_ref is not None:
                    try:
                        e_u = float(ns["pvfv_weighted_rel_l2_vec_gpu"](U_report, u_ref, geom.volume))
                        for comp, name in enumerate(("x", "y", "z")):
                            err = U_report[:, comp] - u_ref[:, comp]
                            num = cp.sum(geom.volume * err * err)
                            den = cp.sum(geom.volume * u_ref[:, comp] * u_ref[:, comp])
                            e_u_components[name] = float(cp.sqrt(num / cp.maximum(den, 1.0e-300)).get())
                        if variant["U_observable"] is not None:
                            projected_ref = cp.einsum(
                                "nij,nj->ni", variant["observable_projector"], u_ref
                            )
                            e_u_observable = float(
                                ns["pvfv_weighted_rel_l2_vec_gpu"](
                                    variant["U_observable"], projected_ref, geom.volume
                                )
                            )
                            e_u_observable_zero_full = float(
                                ns["pvfv_weighted_rel_l2_vec_gpu"](
                                    variant["U_observable"], u_ref, geom.volume
                                )
                            )
                    except Exception as exc:
                        print(f"[segmented-selector] report velocity error failed for {case}/{wall_mode}: {exc!r}", flush=True)
                    if groups is not None:
                        artifacts["geometry_error_correlation"].extend(
                            geometry_error_correlation_rows(
                                ns,
                                geom,
                                groups,
                                U_report,
                                u_ref,
                                case=case,
                                wall_mode=wall_mode,
                                solver_formulation_id=solver_formulation_id,
                                velocity_state_id=str(variant["velocity_state_id"]),
                            )
                        )

                mean_u_report = cp.sum(U_report * geom.volume[:, None], axis=0) / cp.maximum(cp.sum(geom.volume), 1.0e-300)
                k_mean_report = float((float(cfg.nu) * mean_u_report[0] / float(cfg.body_force[0])).get())
                umax_report = float(cp.max(cp.linalg.norm(U_report, axis=1)).get())
                total_s = build_s + float(res["elapsed_s"]) + reconstruct_s
                row = {
                    "case": case,
                    "rule": rule,
                    "wall_mode": wall_mode,
                    "wall_beta": 1.25,
                    "face_mode": str(args.face_mode),
                    "face_operator_arg": str(args.face_operator),
                    "solver_formulation_arg": str(args.solver_formulation),
                    "solver_formulation_id": solver_formulation_id,
                    "solver_result_scope": (
                        "diagnostic_control"
                        if str(args.solver_formulation) == "diagnostic_dense_kkt"
                        else "production_pressure_correction_control"
                    ),
                    "velocity_state_id": str(variant["velocity_state_id"]),
                    "velocity_completion_id": str(variant["velocity_completion_id"]),
                    "pressure_gradient_weight": str(getattr(cfg, "pressure_gradient_weight", "tproj")),
                    "pressure_gauge_eps": float(getattr(cfg, "pressure_gauge_eps", 0.0)),
                    "dt": float(cfg.dt),
                    "n_steps_requested": int(cfg.n_steps),
                    "tikhonov": float(getattr(cfg, "tikhonov", float("nan"))),
                    "velocity_reconstructed_from_flux": bool(variant["reconstructed"]),
                    "velocity_reconstruction_lambda": float(recon_lambda),
                    "velocity_reconstruction_tikhonov": float(
                        getattr(cfg, "velocity_reconstruction_tikhonov", float("nan"))
                    ),
                    "velocity_reconstruction_wall_mode": str(getattr(cfg, "velocity_reconstruction_wall_mode", "")),
                    "velocity_reconstruction_wall_multiplier": float(
                        getattr(cfg, "velocity_reconstruction_wall_multiplier", float("nan"))
                    ),
                    "velocity_reconstruction_transverse_mean_zero": bool(
                        getattr(cfg, "velocity_reconstruction_transverse_mean_zero", False)
                    ),
                    "signed_normal_subflux_rule": str(variant["subflux_rule_id"]),
                    "subflux_aggregate_max_abs": float(variant["subflux_aggregate_max_abs"]),
                    "subflux_aggregate_max_rel": float(variant["subflux_aggregate_max_rel"]),
                    "velocity_reconstruction_rank3_volume_fraction": float(
                        variant["rank3_volume_fraction"]
                    ),
                    "velocity_normal_equation_residual_l2": float(variant["normal_residual_l2"]),
                    "velocity_normal_equation_residual_inf": float(variant["normal_residual_inf"]),
                    "S": int(seed_flat.size),
                    "N_cv": int(meta["N_cv"]),
                    "N_fl": int(meta["N_fl"]),
                    "C_comp": float(meta["C_comp"]),
                    "N_split": int(meta.get("N_split", 0)),
                    "face_connected_fraction": float(meta.get("face_connected_fraction", 1.0)),
                    "K_eff_x": k_eff,
                    "K_eff_x_mean_velocity": float(res["K_eff_x_mean_velocity"]),
                    "K_eff_x_report_velocity": k_mean_report,
                    "K_ref_x": k_ref,
                    "e_K_percent": 100.0 * abs(k_eff - k_ref) / max(abs(k_ref), 1.0e-300),
                    "e_u_percent": 100.0 * e_u,
                    "e_u_x_percent": 100.0 * e_u_components["x"],
                    "e_u_y_percent": 100.0 * e_u_components["y"],
                    "e_u_z_percent": 100.0 * e_u_components["z"],
                    "e_u_solve_percent": 100.0 * e_u_solve,
                    "e_u_phi_observable_percent": 100.0 * e_u_observable,
                    "e_u_phi_observable_zero_full_percent": 100.0 * e_u_observable_zero_full,
                    "e_phi_percent": 100.0 * e_phi,
                    "mass_inf_per_volume": float(res["mass_inf_per_volume"]),
                    "steady_momentum_inf": float(res.get("steady_momentum_inf", float("nan"))),
                    "steady_momentum_evaluation_error": str(
                        res.get("steady_momentum_evaluation_error", "")
                    ),
                    "diagnostic_kkt_momentum_residual_inf": float(
                        res.get("diagnostic_kkt_momentum_residual_inf", float("nan"))
                    ),
                    "steps_completed": int(res.get("steps_completed", 1)),
                    "steady_converged": bool(res.get("steady_converged", False)),
                    "solver_failure": str(res.get("solver_failure", "")),
                    "umax": umax_report,
                    "umax_solve": float(res["umax"]),
                    "t_build_s": build_s,
                    "t_solve_s": float(res["elapsed_s"]),
                    "t_velocity_reconstruction_s": reconstruct_s,
                    "t_total_s": total_s,
                    "t_ref_total_s": float(REFERENCE_TOTAL_S[case]),
                    "speedup_vs_ref": float(REFERENCE_TOTAL_S[case]) / max(total_s, 1.0e-300),
                    "face_operator": op_meta.get("face_operator", ""),
                    "face_operator_variant": op_meta.get("face_operator_variant", ""),
                    "operator_tproj": op_meta.get("operator_tproj", ""),
                    "operator_dvec": op_meta.get("operator_dvec", ""),
                    "face_exchange_distance": op_meta.get("face_exchange_distance", ""),
                    "face_interpolation_weights": op_meta.get("face_interpolation_weights", ""),
                    "operator_tproj_mean": op_meta.get("operator_tproj_mean", ""),
                    "operator_tproj_min": op_meta.get("operator_tproj_min", ""),
                    "operator_tproj_max": op_meta.get("operator_tproj_max", ""),
                    "roi_label_engine": meta.get("roi_label_engine", ""),
                    "label_mode": meta.get("label_mode", ""),
                    "diagnostic_note": (
                        "diagnostic dense KKT; not a production-table formulation"
                        if str(args.solver_formulation) == "diagnostic_dense_kkt"
                        else "manuscript pressure-correction control; convergence fields must pass before retention"
                    ),
                    **geometry_summary,
                }
                rows.append(row)
                print(
                    f"[segmented-selector] result {case} {wall_mode}: "
                    f"K={k_eff:.8g} eK={row['e_K_percent']:.2f}% "
                    f"eU={row['e_u_percent']:.2f}% ePhi={row['e_phi_percent']:.2f}% "
                    f"mass={row['mass_inf_per_volume']:.3e} t={total_s:.3f}s",
                    flush=True,
                )

    del mask
    cp.get_default_memory_pool().free_all_blocks()
    return rows, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retained segmented-selector rows with the final graph-geodesic face operator."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "segmented_selector_rows",
    )
    parser.add_argument(
        "--label-backend",
        default="roi_jfa",
        choices=["roi_jfa", "exact_geodesic", "exact_frontier_gpu"],
        help="Ownership backend used to build the retained cells.",
    )
    parser.add_argument(
        "--cases",
        default="bentheimer_sandstone_crop,fibrous_filter_proxy",
        help="Comma-separated case names.",
    )
    parser.add_argument(
        "--wall-modes",
        default="global_comp_power",
        help="Comma-separated wall modes: fixed_beta,global_comp_power.",
    )
    parser.add_argument(
        "--face-mode",
        default="overrelaxed_default",
        help="Face transmissibility clip mode passed to the notebook geometry builder.",
    )
    parser.add_argument(
        "--face-operator",
        default="geodesic_face",
        choices=["geodesic_face", "geodesic_weights", "geodesic_weights_only", "euclidean"],
        help="Cell-cell face operator variant.",
    )
    parser.add_argument(
        "--pressure-gradient-weight",
        default="tproj",
        choices=["tproj", "transmissibility", "area", "sqrt_tproj_area", "unit"],
        help="Face weight used in the monolithic pressure-gradient LSQ reconstruction.",
    )
    parser.add_argument(
        "--solver-formulation",
        default="diagnostic_dense_kkt",
        choices=["diagnostic_dense_kkt", "production_pressure_correction"],
        help="Select the diagnostic retained KKT or the manuscript pressure-correction formulation explicitly.",
    )
    parser.add_argument(
        "--solver-identity-audit",
        action="store_true",
        help="Export G/D/B/T hashes, adjoint defects, KKT symmetry, and Schur probes for the diagnostic KKT.",
    )
    parser.add_argument(
        "--export-state-npz",
        action="store_true",
        help="Export U_solve, p, phi, divergence, and incidence arrays for paired formulation comparisons.",
    )
    parser.add_argument(
        "--geometry-audit",
        action="store_true",
        help="Export signed-normal groups, chi, rank, condition, and geometric invariants without changing the solve.",
    )
    parser.add_argument(
        "--signed-normal-reconstruct",
        action="store_true",
        help="Report rank-aware U_phi from exact aggregate-preserving signed-normal subfluxes.",
    )
    parser.add_argument(
        "--signed-normal-subflux-rules",
        default="area",
        help="Comma-separated exact aggregate-preserving rules: area,predictor.",
    )
    parser.add_argument("--pressure-gauge-eps", type=float, default=1.0e-8)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--tikhonov", type=float, default=1.0e-12)
    parser.add_argument(
        "--velocity-reconstruct-from-flux",
        action="store_true",
        help="Report velocity after local flux-consistent reconstruction.",
    )
    parser.add_argument(
        "--velocity-reconstruction-lambdas",
        default="0.0",
        help="Comma-separated lambda values used when --velocity-reconstruct-from-flux is enabled.",
    )
    parser.add_argument(
        "--velocity-reconstruction-tikhonov",
        type=float,
        default=1.0e-10,
        help="Independent diagonal ridge for local flux-to-velocity reconstruction.",
    )
    parser.add_argument(
        "--velocity-reconstruction-wall-mode",
        default="none",
        choices=["none", "inv_ccomp", "twall_over_volume", "normal_tensor", "tangent_tensor"],
        help="Optional no-slip wall penalty used only in flux-to-velocity reconstruction.",
    )
    parser.add_argument(
        "--velocity-reconstruction-wall-multipliers",
        default="0.0",
        help="Comma-separated wall penalty multipliers used with --velocity-reconstruct-from-flux.",
    )
    parser.add_argument(
        "--velocity-reconstruction-transverse-mean-zero",
        action="store_true",
        help="Remove volume-mean transverse velocity components after flux-to-velocity reconstruction.",
    )
    parser.add_argument("--wall-length-exponent", type=float, default=0.15)
    parser.add_argument("--wall-ref-ccomp", type=float, default=64.0)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "outputs" / "reference_data" / "reference_data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_roi_environment()
    os.environ["PVFV_LABEL_BACKEND"] = str(args.label_backend)
    runner = load_runner()
    ns, fixed_wall_clone, global_wall_clone = install_namespace(runner, args)
    clones = {
        "fixed_beta": fixed_wall_clone,
        "global_comp_power": global_wall_clone,
    }
    wall_clones = []
    for mode in [item.strip() for item in str(args.wall_modes).split(",") if item.strip()]:
        if mode not in clones:
            raise ValueError(f"Unknown wall mode: {mode}")
        wall_clones.append((mode, clones[mode]))
    if not wall_clones:
        raise ValueError("No wall modes selected")

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, list[Any]] = {
        "geometry_group": [],
        "geometry_edge": [],
        "geometry_cell": [],
        "geometry_error_correlation": [],
        "orientation_flux": [],
        "solver_identity": [],
        "execution_record": [],
        "state_export": [],
    }
    for case in [item.strip() for item in str(args.cases).split(",") if item.strip()]:
        case_rows, case_artifacts = run_case(
            runner,
            ns,
            case,
            Path(args.out_dir),
            wall_clones,
            Path(args.reference_dir),
            args,
        )
        rows.extend(case_rows)
        for artifact_name, artifact_rows in case_artifacts.items():
            artifacts[artifact_name].extend(artifact_rows)
        write_csv(Path(args.out_dir) / "segmented_selector_geodesic_rows.csv", rows)

    out_csv = Path(args.out_dir) / "segmented_selector_geodesic_rows.csv"
    write_csv(out_csv, rows)
    csv_artifacts = {
        "geometry_group": "geometry_group_audit.csv",
        "geometry_edge": "geometry_edge_audit.csv",
        "geometry_cell": "geometry_cell_audit.csv",
        "geometry_error_correlation": "geometry_error_correlations.csv",
        "orientation_flux": "orientation_flux_metrics.csv",
    }
    written_artifacts: dict[str, Any] = {
        "segmented_selector_geodesic_rows.csv": len(rows),
    }
    for artifact_name, filename in csv_artifacts.items():
        if artifacts[artifact_name]:
            write_csv(Path(args.out_dir) / filename, artifacts[artifact_name])
            written_artifacts[filename] = len(artifacts[artifact_name])
    if artifacts["solver_identity"]:
        write_json(Path(args.out_dir) / "solver_identity.json", artifacts["solver_identity"])
        write_json(
            Path(args.out_dir) / "matrix_adjoint_test_report.json",
            [
                {
                    "case": item["case"],
                    "wall_mode": item["wall_mode"],
                    "gradient_vs_divergence_transpose": item["gradient_vs_divergence_transpose"],
                    "pressure_coupling_vs_divergence_transpose": item[
                        "pressure_coupling_vs_divergence_transpose"
                    ],
                    "kkt_symmetry_defect": item["kkt_symmetry_defect"],
                }
                for item in artifacts["solver_identity"]
            ],
        )
        written_artifacts["solver_identity.json"] = len(artifacts["solver_identity"])
        written_artifacts["matrix_adjoint_test_report.json"] = len(artifacts["solver_identity"])
    write_json(Path(args.out_dir) / "solver_execution_records.json", artifacts["execution_record"])
    written_artifacts["solver_execution_records.json"] = len(artifacts["execution_record"])
    if artifacts["state_export"]:
        write_json(Path(args.out_dir) / "state_export_manifest.json", artifacts["state_export"])
        written_artifacts["state_export_manifest.json"] = len(artifacts["state_export"])

    dual_metric_keys = (
        "case",
        "wall_mode",
        "solver_formulation_id",
        "velocity_state_id",
        "velocity_completion_id",
        "signed_normal_subflux_rule",
        "e_K_percent",
        "e_phi_percent",
        "e_u_solve_percent",
        "e_u_percent",
        "e_u_phi_observable_percent",
        "e_u_phi_observable_zero_full_percent",
        "e_u_x_percent",
        "e_u_y_percent",
        "e_u_z_percent",
        "velocity_reconstruction_rank3_volume_fraction",
        "subflux_aggregate_max_abs",
        "subflux_aggregate_max_rel",
        "mass_inf_per_volume",
        "steady_momentum_inf",
    )
    dual_rows = [{key: row.get(key, "") for key in dual_metric_keys} for row in rows]
    write_csv(Path(args.out_dir) / "velocity_dual_metrics.csv", dual_rows)
    written_artifacts["velocity_dual_metrics.csv"] = len(dual_rows)
    write_json(
        Path(args.out_dir) / "audit_artifact_manifest.json",
        {
            "schema": "methodology_upgrade_phase1_v1",
            "solver_formulation_arg": str(args.solver_formulation),
            "geometry_audit": bool(args.geometry_audit),
            "solver_identity_audit": bool(args.solver_identity_audit),
            "signed_normal_reconstruct": bool(args.signed_normal_reconstruct),
            "artifacts": written_artifacts,
        },
    )
    print(f"[segmented-selector] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
