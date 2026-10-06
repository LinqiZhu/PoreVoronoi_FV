from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "gpu" / "code"
sys.path.insert(0, str(CODE_DIR))

cp = pytest.importorskip("cupy")
import cupyx.scipy.sparse as cpx_sp  # noqa: E402

from run_segmented_selector_geodesic_rows import (  # noqa: E402
    assemble_diagnostic_kkt_operators,
    build_solver_identity_audit,
)
from flow_runner import install_monolithic_stokes_solver_modes  # noqa: E402


def test_small_solver_identity_audit_exports_exact_spectrum_and_hashes() -> None:
    owner = cp.asarray([0], dtype=cp.int32)
    neigh = cp.asarray([1], dtype=cp.int32)
    tproj = cp.asarray([2.0], dtype=cp.float64)
    rows = cp.concatenate([owner, owner, neigh, neigh])
    cols = cp.concatenate([owner, neigh, neigh, owner])
    vals = cp.asarray([2.0, -2.0, 2.0, -2.0], dtype=cp.float64)
    laplacian = cpx_sp.coo_matrix((vals, (rows, cols)), shape=(2, 2)).tocsr()
    geom = SimpleNamespace(
        n_cells=2,
        owner=owner,
        neigh=neigh,
        tproj=tproj,
        area=cp.asarray([1.0], dtype=cp.float64),
        avec=cp.asarray([[1.0, 0.0, 0.0]], dtype=cp.float64),
        dvec=cp.asarray([[1.0, 0.0, 0.0]], dtype=cp.float64),
        w_owner=cp.asarray([0.5], dtype=cp.float64),
        w_neigh=cp.asarray([0.5], dtype=cp.float64),
        volume=cp.asarray([1.0, 1.0], dtype=cp.float64),
        twall=cp.asarray([1.0, 1.0], dtype=cp.float64),
        laplacian=laplacian,
    )
    cfg = SimpleNamespace(
        nu=1.0,
        rho=1.0,
        pressure_gauge_eps=0.0,
        pressure_gradient_weight="area",
        tikhonov=1.0,
    )
    ns = {"cp": cp, "cpx_sp": cpx_sp}
    operators = assemble_diagnostic_kkt_operators(ns, geom, cfg)

    n = geom.n_cells
    size = 4 * n + 1
    kkt = cp.zeros((size, size), dtype=cp.float64)
    kkt[: 3 * n, : 3 * n] = operators["momentum"]
    kkt[: 3 * n, 3 * n : 4 * n] = operators["pressure_coupling"]
    kkt[3 * n : 4 * n, : 3 * n] = operators["divergence"]
    kkt[3 * n : 4 * n, 4 * n] = 1.0
    kkt[4 * n, 3 * n : 4 * n] = 1.0 / float(n)

    audit = build_solver_identity_audit(ns, geom, cfg, operators, kkt)

    assert audit["schur_spectrum"]["computed"] is True
    assert len(audit["schur_spectrum"]["eigenvalues"]) == 2
    assert len(audit["matrix_hashes_sha256"]) == 8
    assert all(len(value) == 64 for value in audit["matrix_hashes_sha256"].values())
    assert math.isfinite(audit["kkt_symmetry_defect"])
    assert math.isfinite(
        audit["pressure_coupling_vs_divergence_transpose"]["normalized_frobenius_defect"]
    )


def test_initial_velocity_mode_no_longer_switches_solver_formulation() -> None:
    calls: list[str] = []

    def production_solver(geom, cfg, initial_U=None, run_label: str = "main"):
        calls.append(run_label)
        return {"path": "production"}

    ns = {"cp": cp, "run_velocity_pressure_projection_gpu": production_solver}
    install_monolithic_stokes_solver_modes(ns)
    cfg = SimpleNamespace(
        initial_velocity_mode="monolithic_stokes",
        solver_formulation="production_pressure_correction",
    )

    result = ns["run_velocity_pressure_projection_gpu"](None, cfg, run_label="identity_test")

    assert result == {"path": "production"}
    assert calls == ["identity_test"]
