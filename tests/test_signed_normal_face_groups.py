from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "gpu" / "code"
sys.path.insert(0, str(CODE_DIR))

cp = pytest.importorskip("cupy")

from geodesic_face_operator import (  # noqa: E402
    build_signed_normal_face_groups,
    distribute_aggregate_flux_to_signed_groups,
    reconstruct_rank_aware_velocity_from_signed_flux,
)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(voxel_size=1.0, periodic_x=False)


def _cancelling_geometry(*, reverse_storage: bool = False) -> SimpleNamespace:
    labels = cp.asarray([[[0, 1, 0]]], dtype=cp.int32)
    owner = 1 if reverse_storage else 0
    neigh = 0 if reverse_storage else 1
    return SimpleNamespace(
        labels=labels,
        mask=cp.ones_like(labels, dtype=cp.bool_),
        dist=cp.zeros_like(labels, dtype=cp.float64),
        n_cells=2,
        owner=cp.asarray([owner], dtype=cp.int32),
        neigh=cp.asarray([neigh], dtype=cp.int32),
        area=cp.asarray([2.0], dtype=cp.float64),
        avec=cp.asarray([[0.0, 0.0, 0.0]], dtype=cp.float64),
        tproj=cp.asarray([3.0], dtype=cp.float64),
        volume=cp.asarray([1.0, 1.0], dtype=cp.float64),
        twall=cp.asarray([0.0, 0.0], dtype=cp.float64),
    )


def _single_x_geometry(*, reverse_storage: bool = False) -> SimpleNamespace:
    labels = cp.asarray([[[0, 1]]], dtype=cp.int32)
    owner = 1 if reverse_storage else 0
    neigh = 0 if reverse_storage else 1
    stored_avec = [-1.0, 0.0, 0.0] if reverse_storage else [1.0, 0.0, 0.0]
    return SimpleNamespace(
        labels=labels,
        mask=cp.ones_like(labels, dtype=cp.bool_),
        dist=cp.zeros_like(labels, dtype=cp.float64),
        n_cells=2,
        owner=cp.asarray([owner], dtype=cp.int32),
        neigh=cp.asarray([neigh], dtype=cp.int32),
        area=cp.asarray([1.0], dtype=cp.float64),
        avec=cp.asarray([stored_avec], dtype=cp.float64),
        tproj=cp.asarray([7.0], dtype=cp.float64),
        volume=cp.asarray([1.0, 1.0], dtype=cp.float64),
        twall=cp.asarray([0.0, 0.0], dtype=cp.float64),
    )


def test_signed_groups_recover_hidden_opposite_normals() -> None:
    groups = build_signed_normal_face_groups({"cp": cp}, _cancelling_geometry(), _cfg())

    np.testing.assert_array_equal(cp.asnumpy(groups["group_id"]), np.asarray([0, 1]))
    np.testing.assert_allclose(cp.asnumpy(groups["group_area"]), [1.0, 1.0])
    np.testing.assert_allclose(
        cp.asnumpy(groups["group_normal"]),
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(cp.asnumpy(groups["edge_chi"]), [0.0])
    np.testing.assert_array_equal(cp.asnumpy(groups["aggregate_rank"]), [0, 0])
    np.testing.assert_array_equal(cp.asnumpy(groups["facelet_rank"]), [1, 1])
    assert max(groups["invariants"].values()) == 0.0


def test_area_split_preserves_flux_and_pressure_laplacian() -> None:
    geom = _cancelling_geometry()
    groups = build_signed_normal_face_groups({"cp": cp}, geom, _cfg())
    split = distribute_aggregate_flux_to_signed_groups(
        {"cp": cp}, groups, cp.asarray([4.0], dtype=cp.float64), rule="area"
    )

    np.testing.assert_allclose(cp.asnumpy(split["group_phi"]), [2.0, 2.0])
    np.testing.assert_allclose(cp.asnumpy(split["collapsed_phi"]), [4.0])
    assert split["aggregate_max_abs"] == 0.0
    np.testing.assert_allclose(cp.asnumpy(groups["edge_tproj_from_groups"]), cp.asnumpy(geom.tproj))

    parent_laplacian = np.asarray([[3.0, -3.0], [-3.0, 3.0]])
    group_t = cp.asnumpy(groups["group_tproj_exact"])
    parallel_laplacian = np.asarray(
        [[group_t.sum(), -group_t.sum()], [-group_t.sum(), group_t.sum()]]
    )
    np.testing.assert_allclose(parallel_laplacian, parent_laplacian, rtol=0.0, atol=0.0)


def test_storage_orientation_reversal_is_exact() -> None:
    geom = _single_x_geometry(reverse_storage=True)
    groups = build_signed_normal_face_groups({"cp": cp}, geom, _cfg())
    stored_phi = cp.asarray([-5.0], dtype=cp.float64)
    split = distribute_aggregate_flux_to_signed_groups(
        {"cp": cp}, groups, stored_phi, rule="area"
    )

    np.testing.assert_allclose(cp.asnumpy(split["group_phi"]), [5.0])
    np.testing.assert_allclose(cp.asnumpy(split["collapsed_phi"]), [-5.0])
    np.testing.assert_allclose(cp.asnumpy(groups["edge_avec_sorted"]), [[1.0, 0.0, 0.0]])
    assert split["aggregate_max_abs"] == 0.0


def test_rank_aware_reconstruction_has_no_hidden_ridge() -> None:
    geom = _single_x_geometry()
    groups = build_signed_normal_face_groups({"cp": cp}, geom, _cfg())
    split = distribute_aggregate_flux_to_signed_groups(
        {"cp": cp}, groups, cp.asarray([5.0], dtype=cp.float64), rule="area"
    )
    prior = cp.asarray([[9.0, 2.0, 3.0], [8.0, 4.0, 6.0]], dtype=cp.float64)
    reconstruction = reconstruct_rank_aware_velocity_from_signed_flux(
        {"cp": cp}, groups, split["group_phi"], unobservable_prior=prior
    )

    np.testing.assert_array_equal(cp.asnumpy(reconstruction["rank"]), [1, 1])
    np.testing.assert_allclose(
        cp.asnumpy(reconstruction["U_observable"]),
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        cp.asnumpy(reconstruction["U_completed"]),
        [[5.0, 2.0, 3.0], [5.0, 4.0, 6.0]],
    )
    assert reconstruction["completion_id"] == "observable_projection_plus_solve_state_nullspace"


def test_predictor_constrained_split_remains_aggregate_exact() -> None:
    geom = _cancelling_geometry()
    groups = build_signed_normal_face_groups({"cp": cp}, geom, _cfg())
    predictor = cp.asarray([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=cp.float64)
    split = distribute_aggregate_flux_to_signed_groups(
        {"cp": cp},
        groups,
        cp.asarray([4.0], dtype=cp.float64),
        rule="predictor",
        velocity_predictor=predictor,
    )

    np.testing.assert_allclose(cp.asnumpy(split["collapsed_phi"]), [4.0], rtol=0.0, atol=1.0e-14)
    assert split["aggregate_max_abs"] <= 1.0e-14

