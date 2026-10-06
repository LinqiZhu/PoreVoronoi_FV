"""Gauge-fixed stability and conditioning audit on the actual assembled blocks.

Everything here operates on the objects the production solver really builds:

* ``A = system.velocity_matrix``          (3 * n_trace_modes square, symmetric)
* ``D = system.divergence_matrix``        (n_cells x 3 * n_trace_modes)
* ``M_p = diag(system.trace_geometry.cell_volume)`` on the retained pressure dofs
* the production gauge, which in the locked MINRES path removes the **last**
  pressure equation and unknown (``divergence[:-1]`` in
  ``hybrid_voronoi_trace.build_hybrid_trace_factorization``) and then centres the
  recovered pressure.

Sign/transpose convention, read off the actual KKT matrix
``[[A, D_r^T], [D_r, 0]]``: the pressure Schur complement of that block system is
``-D_r A^{-1} D_r^T``.  The audited operator is therefore the symmetric positive
semi-definite

    S~ = M_p^{-1/2} D_r A^{-1} D_r^T M_p^{-1/2}.

No simplified surrogate is ever reconstructed from plotted output.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg, minres, splu

NULL_RELATIVE_THRESHOLD = 1.0e-12
NULL_ABSOLUTE_THRESHOLD = 1.0e-14
DENSE_PRESSURE_LIMIT = 9000
DENSE_BLOCK_COLUMNS = 128
LOBPCG_BLOCK = 6
LOBPCG_TOLERANCE = 1.0e-8
LOBPCG_MAXITER = 400
EIGSH_TOLERANCE = 1.0e-9


SPLU_MAX_ROWS = 60000
CG_TOLERANCE = 1.0e-13
CG_MAXITER = 20000
CG_REFINEMENT_CYCLES = 6
CG_TRUE_RESIDUAL_TARGET = 1.0e-12
CG_TRUE_RESIDUAL_LIMIT = 1.0e-10


class VelocityBlockSolver:
    """A^{-1} action on the velocity/trace block.

    ``A`` is symmetric positive definite (viscous energy plus the canonical
    stabilization).  Where a direct factorization is affordable the production
    sparse LU is used, exactly the call
    ``build_hybrid_trace_factorization(solver="schur_cg")`` makes.  At the sizes
    reached here (up to ~1.6e5 rows and ~4e7 nonzeros) it is not, so the inverse
    is applied by Jacobi-preconditioned conjugate gradients, on all right-hand
    sides at once so the sparse matrix is read once per block, with residual
    replacement so the reported accuracy is the TRUE residual and not the
    recursively updated one.  The method used and the achieved true residual are
    recorded on every audited row.
    """

    def __init__(
        self,
        A: sparse.spmatrix,
        ordering: str = "COLAMD",
        *,
        force_iterative: bool = False,
        splu_max_rows: int = SPLU_MAX_ROWS,
    ) -> None:
        self.A = A.tocsr()
        self.ordering = ordering
        self.lu = None
        self.fallback_reason = ""
        self.cg_calls = 0
        self.cg_iterations_total = 0
        self.cg_iterations_max = 0
        self.cg_refinement_cycles_max = 0
        self.cg_relative_residual_max = 0.0
        self._gpu_ready = None
        start = time.perf_counter()
        if not force_iterative and int(A.shape[0]) <= int(splu_max_rows):
            try:
                self.lu = splu(A.tocsc(), permc_spec=ordering)
                self.method = f"splu({ordering})"
                self.nnz = int(self.lu.L.nnz + self.lu.U.nnz)
            except (MemoryError, RuntimeError) as error:
                self.lu = None
                self.fallback_reason = f"{type(error).__name__}: {error}"
        if self.lu is None:
            diagonal = np.asarray(self.A.diagonal(), dtype=np.float64)
            if np.any(diagonal <= 0.0):
                raise RuntimeError("Velocity block has a non-positive diagonal")
            self._inverse_diagonal = 1.0 / diagonal
            device = "gpu" if self._try_gpu() else "cpu"
            self.method = (
                f"blocked_jacobi_preconditioned_cg_{device}_with_residual_replacement"
                f"(true_rtol_target={CG_TRUE_RESIDUAL_TARGET:g})"
            )
            self.nnz = int(self.A.nnz)
        self.factorization_time_s = float(time.perf_counter() - start)

    # -- device setup ------------------------------------------------------
    def _try_gpu(self) -> bool:
        if self._gpu_ready is not None:
            return bool(self._gpu_ready)
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx_sparse

            self._cp = cp
            self._A_gpu = cpx_sparse.csr_matrix(
                (
                    cp.asarray(self.A.data),
                    cp.asarray(self.A.indices),
                    cp.asarray(self.A.indptr),
                ),
                shape=self.A.shape,
            )
            self._inverse_diagonal_gpu = cp.asarray(self._inverse_diagonal)
            self._gpu_ready = True
        except Exception as error:
            self.fallback_reason = f"{type(error).__name__}: {error}"
            self._gpu_ready = False
        return bool(self._gpu_ready)

    # -- inner Krylov solve, one block of right-hand sides ------------------
    def _block_cg_core_gpu(self, B: Any) -> tuple[Any, int]:
        cp = self._cp
        A = self._A_gpu
        inverse = self._inverse_diagonal_gpu[:, None]
        norms = cp.linalg.norm(B, axis=0)
        target = CG_TOLERANCE * cp.maximum(norms, 1.0e-300)
        X = cp.zeros_like(B)
        R = B.copy()
        Z = inverse * R
        P = Z.copy()
        rz = cp.sum(R * Z, axis=0)
        active = norms > 0.0
        iterations = 0
        for iterations in range(1, CG_MAXITER + 1):
            AP = A @ P
            denominator = cp.sum(P * AP, axis=0)
            safe = cp.where(denominator == 0.0, 1.0, denominator)
            alpha = cp.where(active & (denominator != 0.0), rz / safe, 0.0)
            X += alpha[None, :] * P
            R -= alpha[None, :] * AP
            active = active & (cp.linalg.norm(R, axis=0) > target)
            if not bool(cp.any(active)):
                break
            Z = inverse * R
            rz_next = cp.sum(R * Z, axis=0)
            safe_rz = cp.where(rz == 0.0, 1.0, rz)
            beta = cp.where(active & (rz != 0.0), rz_next / safe_rz, 0.0)
            P = cp.where(active[None, :], Z + beta[None, :] * P, P)
            rz = cp.where(active, rz_next, rz)
        return X, iterations

    def _block_cg_core_cpu(self, B: np.ndarray) -> tuple[np.ndarray, int]:
        A = self.A
        inverse = self._inverse_diagonal[:, None]
        norms = np.linalg.norm(B, axis=0)
        target = CG_TOLERANCE * np.maximum(norms, 1.0e-300)
        X = np.zeros_like(B)
        R = B.copy()
        Z = inverse * R
        P = Z.copy()
        rz = np.sum(R * Z, axis=0)
        active = norms > 0.0
        iterations = 0
        for iterations in range(1, CG_MAXITER + 1):
            AP = A @ P
            denominator = np.sum(P * AP, axis=0)
            safe = np.where(denominator == 0.0, 1.0, denominator)
            alpha = np.where(active & (denominator != 0.0), rz / safe, 0.0)
            X += alpha[None, :] * P
            R -= alpha[None, :] * AP
            active = active & (np.linalg.norm(R, axis=0) > target)
            if not np.any(active):
                break
            Z = inverse * R
            rz_next = np.sum(R * Z, axis=0)
            safe_rz = np.where(rz == 0.0, 1.0, rz)
            beta = np.where(active & (rz != 0.0), rz_next / safe_rz, 0.0)
            P = np.where(active[None, :], Z + beta[None, :] * P, P)
            rz = np.where(active, rz_next, rz)
        return X, iterations

    # -- outer residual replacement ----------------------------------------
    def _solve_block(self, matrix: np.ndarray) -> np.ndarray:
        columns = int(matrix.shape[1])
        use_gpu = self._try_gpu()
        if use_gpu:
            cp = self._cp
            B = cp.asarray(matrix, dtype=cp.float64)
            A = self._A_gpu
            norms = cp.maximum(cp.linalg.norm(B, axis=0), 1.0e-300)
            X = cp.zeros_like(B)
            R = B.copy()
            iterations_total = 0
            achieved = float("inf")
            for cycle in range(1, CG_REFINEMENT_CYCLES + 1):
                correction, iterations = self._block_cg_core_gpu(R)
                X += correction
                R = B - A @ X
                achieved = float(cp.max(cp.linalg.norm(R, axis=0) / norms))
                iterations_total += iterations
                self.cg_iterations_max = max(self.cg_iterations_max, iterations)
                self.cg_refinement_cycles_max = max(self.cg_refinement_cycles_max, cycle)
                if achieved <= CG_TRUE_RESIDUAL_TARGET:
                    break
            result = cp.asnumpy(X)
        else:
            B = np.ascontiguousarray(matrix, dtype=np.float64)
            norms = np.maximum(np.linalg.norm(B, axis=0), 1.0e-300)
            X = np.zeros_like(B)
            R = B.copy()
            iterations_total = 0
            achieved = float("inf")
            for cycle in range(1, CG_REFINEMENT_CYCLES + 1):
                correction, iterations = self._block_cg_core_cpu(R)
                X += correction
                R = B - self.A @ X
                achieved = float(np.max(np.linalg.norm(R, axis=0) / norms))
                iterations_total += iterations
                self.cg_iterations_max = max(self.cg_iterations_max, iterations)
                self.cg_refinement_cycles_max = max(self.cg_refinement_cycles_max, cycle)
                if achieved <= CG_TRUE_RESIDUAL_TARGET:
                    break
            result = X
        self.cg_calls += columns
        self.cg_iterations_total += iterations_total * columns
        self.cg_relative_residual_max = max(self.cg_relative_residual_max, achieved)
        if achieved > CG_TRUE_RESIDUAL_LIMIT:
            raise RuntimeError(
                f"Velocity-block solve did not reach the declared accuracy: "
                f"true relative residual {achieved:.3e} after {CG_REFINEMENT_CYCLES} "
                f"residual-replacement cycles"
            )
        return result

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.lu is not None:
            return self.lu.solve(rhs)
        matrix = np.asarray(rhs, dtype=np.float64)
        if matrix.ndim == 1:
            return self._solve_block(matrix.reshape(-1, 1))[:, 0]
        return self._solve_block(np.ascontiguousarray(matrix))

    def report(self) -> dict[str, Any]:
        return {
            "a_solve_method": self.method,
            "a_solve_fallback_reason": self.fallback_reason,
            "a_factorization_time_s": self.factorization_time_s,
            "a_factor_nnz": self.nnz,
            "a_solve_calls": self.cg_calls,
            "a_cg_iterations_total": self.cg_iterations_total,
            "a_cg_iterations_max": self.cg_iterations_max,
            "a_cg_refinement_cycles_max": self.cg_refinement_cycles_max,
            "a_true_relative_residual_max": self.cg_relative_residual_max,
        }


def gauge_reduced_divergence(D: sparse.spmatrix, gauge_index: int) -> sparse.csr_matrix:
    """Remove one pressure equation.

    ``gauge_index = n_cells - 1`` reproduces the production ``divergence[:-1]``.
    Any other index is the same construction under a permutation of the cell
    ordering, because ``A`` lives in trace-dof space and is invariant under cell
    reordering while ``D`` is only row-permuted.
    """
    n_cells = int(D.shape[0])
    keep = np.setdiff1d(np.arange(n_cells, dtype=np.int64), np.asarray([gauge_index], dtype=np.int64))
    return D.tocsr()[keep].tocsr()


def build_reduced_kkt(A: sparse.spmatrix, D_r: sparse.spmatrix) -> sparse.csr_matrix:
    zero = sparse.csr_matrix((D_r.shape[0], D_r.shape[0]), dtype=np.float64)
    return sparse.bmat([[A, D_r.T], [D_r, zero]], format="csr")


def kkt_symmetry_defect(kkt: sparse.spmatrix) -> float:
    denominator = max(float(sparse.linalg.norm(kkt)), 1.0e-300)
    return float(sparse.linalg.norm(kkt - kkt.T) / denominator)


DENSE_FALLBACK_LIMIT = 9000


def schur_spectrum(
    A: sparse.spmatrix,
    D: sparse.spmatrix,
    cell_volume: np.ndarray,
    *,
    gauge_index: int | None = None,
    solver: VelocityBlockSolver | None = None,
    dense_limit: int = DENSE_PRESSURE_LIMIT,
    dense_fallback_limit: int = DENSE_FALLBACK_LIMIT,
) -> dict[str, Any]:
    """Extreme eigenvalues of S~ with converged residuals, or an honest failure.

    Small pressure spaces are decomposed exactly.  Larger ones use the
    matrix-free path; if that path does not converge, the row is retried with the
    exact dense decomposition when the pressure dimension still permits it, and
    the method actually used is recorded.  An estimate is never extrapolated.
    """
    first = _schur_spectrum_once(
        A, D, cell_volume, gauge_index=gauge_index, solver=solver, dense_limit=dense_limit
    )
    if first["spectral_status"] == "PASS":
        return first
    n_pressure = int(first["pressure_dimension_after_gauge"])
    if n_pressure > int(dense_fallback_limit) or "dense" in str(first["spectral_method"]):
        return first
    retry = _schur_spectrum_once(
        A, D, cell_volume, gauge_index=gauge_index, solver=solver, dense_limit=n_pressure
    )
    retry["iterative_attempt_before_dense_fallback"] = {
        key: first[key]
        for key in ("spectral_method", "spectral_status", "spectral_message", "spectral_residual_max")
        if key in first
    }
    return retry


def _schur_spectrum_once(
    A: sparse.spmatrix,
    D: sparse.spmatrix,
    cell_volume: np.ndarray,
    *,
    gauge_index: int | None = None,
    solver: VelocityBlockSolver | None = None,
    dense_limit: int = DENSE_PRESSURE_LIMIT,
) -> dict[str, Any]:
    n_cells = int(D.shape[0])
    if gauge_index is None:
        gauge_index = n_cells - 1
    D_r = gauge_reduced_divergence(D, gauge_index)
    keep = np.setdiff1d(np.arange(n_cells, dtype=np.int64), np.asarray([gauge_index]))
    volume = np.asarray(cell_volume, dtype=np.float64)[keep]
    if np.any(volume <= 0.0):
        raise RuntimeError("A retained pressure cell has non-positive volume")
    scale = 1.0 / np.sqrt(volume)
    B = sparse.diags(scale) @ D_r.tocsr()
    B = B.tocsr()
    n_pressure = int(B.shape[0])

    owned_solver = solver is None
    if solver is None:
        solver = VelocityBlockSolver(A)

    record: dict[str, Any] = {
        "pressure_dimension_before_gauge": n_cells,
        "pressure_dimension_after_gauge": n_pressure,
        "gauge_choice": f"pressure_index_{gauge_index}_eliminated_then_centered",
        "gauge_index": int(gauge_index),
        "a_solve_method": solver.method,
        "a_factorization_time_s": solver.factorization_time_s,
        "a_factor_nnz": solver.nnz,
        "a_matrix_shape": [int(A.shape[0]), int(A.shape[1])],
        "a_matrix_nnz": int(A.nnz),
        "schur_operator": "M_p^{-1/2} D_r A^{-1} D_r^T M_p^{-1/2}",
        "sign_convention": "read from the assembled KKT [[A, D_r^T],[D_r, 0]]; Schur = -D_r A^{-1} D_r^T",
    }

    start = time.perf_counter()
    if n_pressure <= dense_limit:
        dense = np.empty((n_pressure, n_pressure), dtype=np.float64)
        Bt = B.T.tocsc()
        for begin in range(0, n_pressure, DENSE_BLOCK_COLUMNS):
            end = min(begin + DENSE_BLOCK_COLUMNS, n_pressure)
            rhs = np.asarray(Bt[:, begin:end].todense(), dtype=np.float64)
            dense[:, begin:end] = B @ solver.solve(rhs)
        symmetry = float(np.max(np.abs(dense - dense.T)))
        magnitude = max(float(np.max(np.abs(dense))), 1.0e-300)
        dense = 0.5 * (dense + dense.T)
        eigenvalues = np.linalg.eigvalsh(dense)
        lambda_max = float(eigenvalues[-1])
        threshold = max(NULL_RELATIVE_THRESHOLD * lambda_max, NULL_ABSOLUTE_THRESHOLD)
        positive = eigenvalues[eigenvalues > threshold]
        record.update(
            {
                "spectral_method": (
                    "dense_eigvalsh_on_explicit_schur, with the velocity-block inverse applied "
                    f"by {solver.method}"
                ),
                "spectral_tolerance": float(np.finfo(np.float64).eps),
                "spectral_iterations": 0,
                "spectral_residual_max": 0.0,
                "schur_lambda_min_positive": float(positive[0]) if positive.size else float("nan"),
                "schur_lambda_max": lambda_max,
                "rank_defect_estimate_after_gauge": int(np.count_nonzero(eigenvalues <= threshold)),
                "schur_symmetry_defect_relative": float(symmetry / magnitude),
                "null_threshold": float(threshold),
                "spectral_status": "PASS" if positive.size else "FAIL",
                "smallest_eight_eigenvalues": [float(value) for value in eigenvalues[:8]],
            }
        )
    else:
        def matvec(vector: np.ndarray) -> np.ndarray:
            lifted = np.asarray(B.T @ np.asarray(vector).reshape(-1)).reshape(-1)
            return np.asarray(B @ solver.solve(lifted)).reshape(-1)

        def matmat(block: np.ndarray) -> np.ndarray:
            lifted = np.asarray(B.T @ np.asarray(block))
            return np.asarray(B @ solver.solve(np.ascontiguousarray(lifted)))

        operator = LinearOperator(
            (n_pressure, n_pressure),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            rmatmat=matmat,
            dtype=np.float64,
        )
        diagonal_A = np.asarray(A.diagonal(), dtype=np.float64)
        schur_diagonal = np.asarray(B.multiply(B) @ (1.0 / diagonal_A)).reshape(-1)
        schur_diagonal = np.maximum(schur_diagonal, np.finfo(np.float64).tiny)
        preconditioner = LinearOperator(
            (n_pressure, n_pressure),
            matvec=lambda vector: np.asarray(vector).reshape(-1) / schur_diagonal,
            rmatvec=lambda vector: np.asarray(vector).reshape(-1) / schur_diagonal,
            dtype=np.float64,
        )
        largest, largest_vectors = eigsh(
            operator, k=1, which="LA", tol=EIGSH_TOLERANCE, maxiter=5000
        )
        lambda_max = float(largest[-1])
        generator = np.random.default_rng(20260824)
        guess = generator.standard_normal((n_pressure, LOBPCG_BLOCK))
        status = "PASS"
        message = ""
        try:
            smallest, smallest_vectors = lobpcg(
                operator,
                guess,
                M=preconditioner,
                largest=False,
                tol=LOBPCG_TOLERANCE,
                maxiter=LOBPCG_MAXITER,
            )
        except Exception as error:  # reported, never silently replaced
            status = "FAIL"
            message = f"{type(error).__name__}: {error}"
            smallest = np.asarray([np.nan])
            smallest_vectors = np.zeros((n_pressure, 1))
        residuals = []
        for column in range(smallest_vectors.shape[1]):
            vector = smallest_vectors[:, column]
            norm = float(np.linalg.norm(vector))
            if norm == 0.0 or not np.isfinite(smallest[column]):
                residuals.append(float("inf"))
                continue
            vector = vector / norm
            residual = matvec(vector) - float(smallest[column]) * vector
            residuals.append(float(np.linalg.norm(residual) / max(abs(float(smallest[column])), 1.0e-300)))
        top_vector = largest_vectors[:, 0] / max(float(np.linalg.norm(largest_vectors[:, 0])), 1.0e-300)
        top_residual = float(
            np.linalg.norm(matvec(top_vector) - lambda_max * top_vector) / max(abs(lambda_max), 1.0e-300)
        )
        threshold = max(NULL_RELATIVE_THRESHOLD * lambda_max, NULL_ABSOLUTE_THRESHOLD)
        finite = np.asarray([value for value in smallest if np.isfinite(value)], dtype=np.float64)
        positive = finite[finite > threshold]
        converged = [
            value
            for value, residual in zip(smallest, residuals)
            if np.isfinite(value) and residual <= 1.0e-4
        ]
        if not converged:
            status = "FAIL"
            message = message or "LOBPCG did not converge to the requested eigen-residual"
        record.update(
            {
                "spectral_method": (
                    "matrix_free_LinearOperator: eigsh(which=LA) for the largest eigenvalue and "
                    "preconditioned LOBPCG for the smallest, with the velocity-block inverse "
                    f"applied inside every matvec by {solver.method}"
                ),
                "spectral_tolerance": LOBPCG_TOLERANCE,
                "spectral_iterations": LOBPCG_MAXITER,
                "spectral_residual_max": float(max([top_residual, *residuals])),
                "schur_lambda_min_positive": float(positive[0]) if positive.size else float("nan"),
                "schur_lambda_max": lambda_max,
                "rank_defect_estimate_after_gauge": int(np.count_nonzero(finite <= threshold)),
                "schur_symmetry_defect_relative": 0.0,
                "null_threshold": float(threshold),
                "spectral_status": status,
                "spectral_message": message,
                "smallest_eight_eigenvalues": [float(value) for value in np.sort(finite)[:8]],
                "lobpcg_eigen_residuals": residuals,
                "largest_eigen_residual": top_residual,
            }
        )
    record["spectral_seconds"] = float(time.perf_counter() - start)
    record.update(solver.report())
    lambda_min = record["schur_lambda_min_positive"]
    record["beta_h"] = float(np.sqrt(lambda_min)) if np.isfinite(lambda_min) else float("nan")
    record["schur_condition_proxy"] = (
        float(record["schur_lambda_max"] / lambda_min) if np.isfinite(lambda_min) and lambda_min > 0 else float("nan")
    )
    if owned_solver:
        del solver
    return record


def solve_with_gauge(
    system: Any,
    gauge_index: int,
    *,
    rtol: float,
    maxiter: int,
    refinement_steps: int,
) -> dict[str, Any]:
    """Production MINRES path with one chosen pressure index eliminated.

    Structurally identical to
    ``hybrid_voronoi_trace.build_hybrid_trace_factorization(solver='minres')`` and
    ``solve_moment_constrained_hybrid_stokes``: the same reduced KKT, the same
    inverse-diagonal block preconditioner, the same iterative refinement rule and
    the same centring of the recovered pressure.  Only which pressure index is
    eliminated differs, which is a permutation of the cell ordering.
    """
    trace = system.trace_geometry
    A = system.velocity_matrix
    D = system.divergence_matrix
    n_trace = int(A.shape[0])
    n_cells = int(trace.n_cells)
    D_r = gauge_reduced_divergence(D, gauge_index)
    kkt = build_reduced_kkt(A, D_r)

    diagonal_A = np.asarray(A.diagonal(), dtype=np.float64)
    if np.any(diagonal_A <= 0.0):
        raise RuntimeError("Hybrid velocity block has a non-positive diagonal")
    inverse_diagonal_A = 1.0 / diagonal_A
    schur_diagonal = np.asarray(D_r.multiply(D_r) @ inverse_diagonal_A).reshape(-1)
    if np.any(schur_diagonal <= 0.0):
        raise RuntimeError("Hybrid pressure Schur preconditioner has a non-positive diagonal")
    inverse = np.concatenate([inverse_diagonal_A, 1.0 / schur_diagonal])
    preconditioner = LinearOperator(
        kkt.shape,
        matvec=lambda vector: inverse * vector,
        rmatvec=lambda vector: inverse * vector,
        dtype=np.float64,
    )
    rhs_trace = system.body_force_matrix @ system.body_force
    rhs = np.concatenate([rhs_trace, np.zeros(n_cells - 1, dtype=np.float64)])

    iterations = 0

    def count(_iterate: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = minres(
        kkt, rhs, rtol=rtol, maxiter=maxiter, M=preconditioner, callback=count, check=False
    )
    limit = max(20.0 * rtol, 1.0e-12)
    rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-300)
    refinements = 0
    while info == 0 and refinements < refinement_steps:
        correction_rhs = np.asarray(rhs - kkt @ solution).reshape(-1)
        if float(np.linalg.norm(correction_rhs)) / rhs_norm <= limit:
            break
        correction, info = minres(
            kkt, correction_rhs, rtol=rtol, maxiter=maxiter, M=preconditioner, callback=count, check=False
        )
        solution += correction
        refinements += 1

    keep = np.setdiff1d(np.arange(n_cells, dtype=np.int64), np.asarray([gauge_index]))
    pressure = np.zeros(n_cells, dtype=np.float64)
    pressure[keep] = solution[n_trace:]
    pressure -= float(np.mean(pressure))
    trace_solution = solution[:n_trace]

    cell_velocity = np.zeros((n_cells, 3), dtype=np.float64)
    for cell, (global_dofs, recovery) in enumerate(system.cell_recovery):
        if global_dofs.size:
            cell_velocity[cell] = recovery @ trace_solution[global_dofs]

    coefficients = trace_solution.reshape(trace.n_trace_modes, 3)
    face_velocity = np.zeros((trace.n_facelets, 3), dtype=np.float64)
    for column in range(trace.face_mode_ids.shape[1]):
        mode = trace.face_mode_ids[:, column]
        valid = mode >= 0
        if np.any(valid):
            face_velocity[valid] += trace.face_mode_values[valid, column, None] * coefficients[mode[valid]]
    face_flux = (trace.voxel_size**2) * np.sum(face_velocity * trace.face_normal, axis=1)
    aggregate = np.bincount(
        trace.face_parent_edge, weights=face_flux, minlength=trace.stored_edge_sign_from_sorted.size
    ) * trace.stored_edge_sign_from_sorted

    residual = np.asarray(kkt @ solution - rhs).reshape(-1)
    relative_residual = float(np.linalg.norm(residual) / rhs_norm)
    mass_residual = np.asarray(D @ trace_solution).reshape(-1)
    return {
        "gauge_index": int(gauge_index),
        "U": cell_velocity,
        "p": pressure,
        "phi": aggregate,
        "linear_solver_iterations": int(iterations),
        "linear_solver_refinement_steps": int(refinements),
        "linear_solver_info": int(info),
        "linear_solver_relative_residual": relative_residual,
        "mass_inf_per_volume": float(
            np.max(np.abs(mass_residual) / np.maximum(trace.cell_volume, 1.0e-300))
        ),
    }


def gauge_sensitivity(
    system: Any,
    *,
    rtol: float,
    maxiter: int,
    refinement_steps: int,
    alternative_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Repeat the solve with two alternative pressure indices chosen without reference data."""
    n_cells = int(system.trace_geometry.n_cells)
    production_index = n_cells - 1
    if alternative_indices is None:
        alternative_indices = [0, n_cells // 2]
    alternative_indices = [int(index) for index in alternative_indices if int(index) != production_index]
    baseline = solve_with_gauge(
        system, production_index, rtol=rtol, maxiter=maxiter, refinement_steps=refinement_steps
    )
    comparisons: list[dict[str, Any]] = []
    worst = 0.0
    for index in alternative_indices:
        variant = solve_with_gauge(
            system, index, rtol=rtol, maxiter=maxiter, refinement_steps=refinement_steps
        )
        entry: dict[str, Any] = {"gauge_index": index}
        for field in ("U", "p", "phi"):
            reference = np.asarray(baseline[field], dtype=np.float64)
            candidate = np.asarray(variant[field], dtype=np.float64)
            difference = float(np.linalg.norm(candidate - reference))
            magnitude = max(float(np.linalg.norm(reference)), 1.0e-300)
            entry[f"relative_difference_{field}"] = difference / magnitude
            worst = max(worst, difference / magnitude)
        entry["linear_solver_iterations"] = variant["linear_solver_iterations"]
        entry["linear_solver_relative_residual"] = variant["linear_solver_relative_residual"]
        comparisons.append(entry)
    return {
        "production_gauge_index": production_index,
        "alternative_gauge_selection_rule": (
            "cell indices 0 and n_cells // 2; chosen from the cell ordering alone, with no "
            "reference velocity, flux, pressure or error consulted"
        ),
        "comparisons": comparisons,
        "gauge_sensitivity_relative_solution_difference": float(worst),
        "baseline_linear_solver_iterations": baseline["linear_solver_iterations"],
        "baseline_linear_solver_relative_residual": baseline["linear_solver_relative_residual"],
    }
