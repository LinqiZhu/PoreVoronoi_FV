from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "gpu" / "code"
sys.path.insert(0, str(CODE_DIR))

cp = pytest.importorskip("cupy")

from hybrid_voronoi_trace import (  # noqa: E402
    assemble_moment_constrained_hybrid_stokes,
    build_hybrid_trace_factorization,
    build_hybrid_trace_geometry,
    build_hybrid_trace_projection_factorization,
    build_hybrid_trace_projection_operators,
    project_constant_cell_states_to_conservative_trace,
    solve_moment_constrained_hybrid_stokes,
)


def _periodic_two_cell_geometry(cross_section: int = 1) -> SimpleNamespace:
    labels = np.zeros((cross_section, cross_section, 4), dtype=np.int32)
    labels[:, :, 2:] = 1
    volume = float(2 * cross_section * cross_section)
    centroid = np.asarray(
        [
            [1.0, 0.5 * cross_section, 0.5 * cross_section],
            [3.0, 0.5 * cross_section, 0.5 * cross_section],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        labels=cp.asarray(labels),
        mask=cp.ones(labels.shape, dtype=cp.bool_),
        dist=cp.zeros(labels.shape, dtype=cp.float64),
        n_cells=2,
        owner=cp.asarray([0], dtype=cp.int32),
        neigh=cp.asarray([1], dtype=cp.int32),
        volume=cp.asarray([volume, volume], dtype=cp.float64),
        centroid=cp.asarray(centroid),
    )


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(voxel_size=1.0, periodic_x=True)


def test_connected_p1_modes_are_area_orthogonal_on_planar_patches() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=2),
        _cfg(),
        trace_basis="connected_p1",
    )

    assert trace.n_facelets == 8
    assert trace.n_patches == 2
    assert trace.n_trace_modes == 6
    for patch in range(trace.n_patches):
        facelets = np.flatnonzero(trace.face_patch == patch)
        mode_ids = np.unique(trace.face_mode_ids[facelets][trace.face_mode_ids[facelets] >= 0])
        basis = np.zeros((facelets.size, mode_ids.size), dtype=np.float64)
        for row, facelet in enumerate(facelets):
            for mode, value in zip(
                trace.face_mode_ids[facelet], trace.face_mode_values[facelet]
            ):
                if mode >= 0:
                    basis[row, int(np.flatnonzero(mode_ids == mode)[0])] = value
        gram = basis.T @ basis
        np.testing.assert_allclose(gram, facelets.size * np.eye(3), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(np.sum(basis[:, 1:], axis=0), 0.0, rtol=0.0, atol=1.0e-12)


def test_cached_hybrid_solve_is_conservative_symmetric_and_linear() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=1),
        _cfg(),
        trace_basis="connected_p0",
    )
    system = assemble_moment_constrained_hybrid_stokes(
        trace,
        viscosity=1.0,
        body_force=np.asarray([1.0, 0.0, 0.0]),
        viscous_form="symmetric_gradient",
    )
    factorization = build_hybrid_trace_factorization(system)
    full = solve_moment_constrained_hybrid_stokes(system, factorization=factorization)
    half = solve_moment_constrained_hybrid_stokes(
        system,
        factorization=factorization,
        body_force=np.asarray([0.5, 0.0, 0.0]),
    )

    assert full["velocity_matrix_symmetry_defect"] == 0.0
    assert full["mass_inf_per_volume"] <= 1.0e-12
    assert full["momentum_residual_inf"] <= 1.0e-12
    assert full["n_trace_dofs"] == 6
    np.testing.assert_allclose(half["U"], 0.5 * full["U"], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(half["phi"], 0.5 * full["phi"], rtol=1.0e-12, atol=1.0e-12)


def test_minres_matches_direct_hybrid_state() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=1),
        _cfg(),
        trace_basis="connected_p0",
    )
    system = assemble_moment_constrained_hybrid_stokes(
        trace,
        viscosity=1.0,
        body_force=np.asarray([1.0, 0.0, 0.0]),
        viscous_form="symmetric_gradient",
    )
    direct = solve_moment_constrained_hybrid_stokes(
        system,
        factorization=build_hybrid_trace_factorization(system, solver="direct_lu"),
    )
    iterative = solve_moment_constrained_hybrid_stokes(
        system,
        factorization=build_hybrid_trace_factorization(
            system,
            solver="minres",
            iterative_rtol=1.0e-12,
            iterative_maxiter=5000,
        ),
    )

    schur = solve_moment_constrained_hybrid_stokes(
        system,
        factorization=build_hybrid_trace_factorization(
            system,
            solver="schur_cg",
            iterative_rtol=1.0e-12,
            iterative_maxiter=5000,
        ),
    )

    assert iterative["linear_solver"] == "minres"
    assert iterative["linear_solver_iterations"] > 0
    assert iterative["linear_solver_relative_residual"] <= 2.0e-11
    assert iterative["mass_inf_per_volume"] <= 1.0e-10
    assert iterative["momentum_residual_inf"] <= 1.0e-10
    np.testing.assert_allclose(iterative["U"], direct["U"], rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(
        iterative["phi"], direct["phi"], rtol=1.0e-10, atol=1.0e-11
    )
    np.testing.assert_allclose(
        iterative["p"], direct["p"], rtol=1.0e-10, atol=1.0e-11
    )
    assert schur["linear_solver"] == "schur_cg"
    assert schur["linear_solver_iterations"] > 0
    assert schur["linear_solver_relative_residual"] <= 2.0e-11
    assert schur["mass_inf_per_volume"] <= 1.0e-10
    assert schur["momentum_residual_inf"] <= 1.0e-10
    np.testing.assert_allclose(schur["U"], direct["U"], rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(schur["phi"], direct["phi"], rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(schur["p"], direct["p"], rtol=1.0e-10, atol=1.0e-11)


def test_cell_velocity_has_no_independent_interior_unknowns() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=1),
        _cfg(),
        trace_basis="connected_p0",
    )
    system = assemble_moment_constrained_hybrid_stokes(
        trace,
        viscosity=1.0,
        body_force=np.asarray([1.0, 0.0, 0.0]),
        viscous_form="symmetric_gradient",
    )

    assert system.velocity_matrix.shape == (3 * trace.n_trace_modes,) * 2
    assert system.body_force_matrix.shape == (3 * trace.n_trace_modes, 3)
    assert len(system.cell_recovery) == trace.n_cells
    assert all(recovery.shape[0] == 3 for _dofs, recovery in system.cell_recovery)


def test_constant_cell_state_trace_projection_preserves_uniform_periodic_flow() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=1),
        _cfg(),
        trace_basis="connected_p1",
    )
    operators = build_hybrid_trace_projection_operators(trace)
    factorization = build_hybrid_trace_projection_factorization(operators)
    target = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = project_constant_cell_states_to_conservative_trace(
        operators,
        target,
        factorization=factorization,
    )

    assert result["mass_inf_per_volume"] <= 1.0e-12
    assert result["interior_velocity_dofs_per_cell"] == 0
    assert result["trace_correction_relative_l2"] <= 1.0e-12
    np.testing.assert_allclose(result["U"], target, rtol=0.0, atol=1.0e-12)


def test_constant_cell_state_trace_projection_corrects_nonconservative_target() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=2),
        SimpleNamespace(voxel_size=1.0, periodic_x=False),
        trace_basis="connected_p1",
    )
    operators = build_hybrid_trace_projection_operators(trace)
    target = np.asarray([[1.0, 0.2, 0.0], [0.1, -0.3, 0.4]])
    result = project_constant_cell_states_to_conservative_trace(operators, target)

    assert result["mass_inf_per_volume"] <= 1.0e-12
    assert result["trace_correction_relative_l2"] > 0.0
    assert result["U"].shape == (2, 3)
    assert np.all(np.isfinite(result["U"]))
    np.testing.assert_array_equal(result["U"], target)
    assert not np.allclose(result["U_trace_moment"], target)


def test_periodic_trace_projection_can_preserve_particle_state_mean_throughflow() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=2),
        _cfg(),
        trace_basis="connected_p1",
    )
    operators = build_hybrid_trace_projection_operators(trace)
    target = np.asarray([[1.0, 0.2, 0.0], [0.1, -0.3, 0.4]])
    factorization = build_hybrid_trace_projection_factorization(
        operators,
        preserve_mean_components=(0,),
    )
    result = project_constant_cell_states_to_conservative_trace(
        operators,
        target,
        factorization=factorization,
        preserved_mean_target="particle_state_mean",
    )
    target_mean_x = float(np.sum(trace.cell_volume * target[:, 0]) / np.sum(trace.cell_volume))
    result_mean_x = float(np.sum(trace.cell_volume * result["U"][:, 0]) / np.sum(trace.cell_volume))

    assert result["mass_inf_per_volume"] <= 1.0e-12
    assert result["preserved_mean_velocity_residual_inf"] <= 1.0e-12
    np.testing.assert_allclose(result_mean_x, target_mean_x, rtol=0.0, atol=1.0e-12)


def test_periodic_trace_projection_can_preserve_preprojection_face_mean() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=2),
        _cfg(),
        trace_basis="connected_p1",
    )
    operators = build_hybrid_trace_projection_operators(trace)
    target = np.asarray([[1.0, 0.2, 0.0], [0.1, -0.3, 0.4]])
    factorization = build_hybrid_trace_projection_factorization(
        operators,
        preserve_mean_components=(0,),
    )
    result = project_constant_cell_states_to_conservative_trace(
        operators,
        target,
        factorization=factorization,
        preserved_mean_target="face_target_mean",
    )

    selected = np.asarray(result["selected_preserved_mean_velocity"])
    unconstrained = np.asarray(result["unconstrained_trace_mean_velocity"])
    np.testing.assert_allclose(selected[0], unconstrained[0], rtol=0.0, atol=1.0e-12)
    assert result["preserved_mean_velocity_residual_inf"] <= 1.0e-12


def test_default_periodic_projection_uses_only_cell_mass_constraints() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=2),
        _cfg(),
        trace_basis="connected_p1",
    )
    operators = build_hybrid_trace_projection_operators(trace)
    factorization = build_hybrid_trace_projection_factorization(operators)
    target = np.asarray([[1.0, 0.2, 0.0], [0.1, -0.3, 0.4]])
    result = project_constant_cell_states_to_conservative_trace(
        operators,
        target,
        factorization=factorization,
    )

    assert factorization.preserved_mean_components == ()
    assert result["preserved_mean_target"] == "none"
    assert result["throughflow_constraint_active"] is False
    assert result["mass_inf_per_volume"] <= 1.0e-12


def test_projected_cg_matches_direct_trace_projection() -> None:
    trace = build_hybrid_trace_geometry(
        {"cp": cp},
        _periodic_two_cell_geometry(cross_section=2),
        _cfg(),
        trace_basis="connected_p1",
    )
    operators = build_hybrid_trace_projection_operators(trace)
    direct = build_hybrid_trace_projection_factorization(operators, solver="direct_lu")
    iterative = build_hybrid_trace_projection_factorization(
        operators,
        solver="projected_cg",
        iterative_rtol=1.0e-12,
    )
    target = np.asarray([[1.0, 0.2, -0.1], [0.1, -0.3, 0.4]])

    direct_result = project_constant_cell_states_to_conservative_trace(
        operators,
        target,
        factorization=direct,
    )
    iterative_result = project_constant_cell_states_to_conservative_trace(
        operators,
        target,
        factorization=iterative,
    )

    assert iterative_result["projection_solver"] == "projected_cg"
    assert iterative_result["projection_system_relative_residual"] <= 1.0e-10
    np.testing.assert_allclose(
        iterative_result["U_trace_moment"],
        direct_result["U_trace_moment"],
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(iterative_result["phi"], direct_result["phi"], rtol=1.0e-10, atol=1.0e-11)
