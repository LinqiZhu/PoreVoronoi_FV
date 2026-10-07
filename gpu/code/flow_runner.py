from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(r".")
FLOW_NOTEBOOK = PACKAGE_ROOT / "notebooks" / "flow_solver.ipynb"
BENTHEIMER_INPUT = ROOT / "examples/segmented_masks" / "bentheimer_dry_crop_16x96x96_origin_1562_59_349_pore0.npz"
FIBROUS_INPUT = ROOT / "examples/segmented_masks" / "fibrous_filter_proxy_16x64x64.npz"
OUT_ROOT = ROOT / "outputs/flow_runs"
STEADY_SCALAR_INITIAL_MODES = {"steady_scalar_x", "scalar_stokes_x", "poisson_x"}
STEADY_SCALAR_SOLVER_MODES = {
    "steady_scalar_direct",
    "scalar_stokes_direct",
    "poisson_direct",
    "steady_scalar_projected",
    "scalar_stokes_projected",
    "poisson_projected",
    "steady_scalar_projected_reconstruct",
    "scalar_stokes_projected_reconstruct",
    "poisson_projected_reconstruct",
}
MONOLITHIC_STOKES_SOLVER_MODES = {"monolithic_stokes", "stokes_monolithic"}
PRODUCTION_PRESSURE_CORRECTION_FORMULATION = "production_pressure_correction"
DIAGNOSTIC_DENSE_KKT_FORMULATION = "diagnostic_dense_kkt"
SOLVER_FORMULATIONS = {
    PRODUCTION_PRESSURE_CORRECTION_FORMULATION,
    DIAGNOSTIC_DENSE_KKT_FORMULATION,
}
DENSE_LU_SOLVER_MODES = {"dense_lu", "gpu_dense_lu", "lu"}
PRESSURE_GRADIENT_WEIGHT_MODES = {"tproj", "transmissibility", "area", "sqrt_tproj_area", "unit"}

try:
    from geodesic_face_operator import build_geodesic_face_metric, clone_geometry, physical_flux_readout
except ModuleNotFoundError:
    _LOCAL_GEODESIC_MODULE_DIR = Path(
        r"porevoronoi_fv"
    )
    if _LOCAL_GEODESIC_MODULE_DIR.exists():
        sys.path.insert(0, str(_LOCAL_GEODESIC_MODULE_DIR))
    from geodesic_face_operator import build_geodesic_face_metric, clone_geometry, physical_flux_readout


def pressure_gradient_weight(ns: dict[str, Any], geom: Any, cfg: Any):
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


def read_notebook_code_cells(path: Path, cell_ids: list[int]) -> list[tuple[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for idx in cell_ids:
        cell = data["cells"][idx]
        if cell.get("cell_type") != "code":
            continue
        out.append((idx, "".join(cell.get("source", []))))
    return out


def load_flow_namespace(path: Path) -> dict[str, Any]:
    module_name = "__flow_namespace__"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    ns: dict[str, Any] = module.__dict__
    ns["__file__"] = str(path)
    # Cells 1, 3, 5, 7, and 9 define the GPU solver, manuscript data export,
    # seed-influence suite, and half-offset/GFPS admissible seed protocol.  The
    # final run-control cell is intentionally skipped here.
    for idx, code in read_notebook_code_cells(path, [1, 3, 5, 7, 9]):
        print(f"[flow] loading notebook cell {idx}")
        exec(compile(code, f"{path}:cell{idx}", "exec"), ns)
    return ns


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    try:
        total = float(seconds)
    except Exception:
        return "?"
    if not math.isfinite(total) or total < 0:
        return "?"
    total_i = int(round(total))
    h, rem = divmod(total_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


class FlowProgress:
    def __init__(self, *, enabled: bool, total_units: int, every_s: float) -> None:
        self.enabled = bool(enabled)
        self.total_units = max(0, int(total_units))
        self.every_s = max(0.0, float(every_s))
        self.start_s = time.perf_counter()
        self.completed = 0
        self.lock = threading.Lock()

    def emit(self, text: str) -> None:
        if self.enabled:
            print(text, flush=True)

    def begin(self, label: str) -> float:
        now = time.perf_counter()
        with self.lock:
            done = int(self.completed)
            total = int(self.total_units)
        eta = self.eta_seconds(now=now)
        self.emit(
            f"[flow][progress] start {done + 1}/{total}: {label}; "
            f"elapsed={format_duration(now - self.start_s)}, eta={format_duration(eta)}"
        )
        return now

    def complete(self, label: str, started_s: float, *, extra: str = "") -> None:
        now = time.perf_counter()
        with self.lock:
            self.completed += 1
            done = int(self.completed)
            total = int(self.total_units)
        eta = self.eta_seconds(now=now)
        suffix = f"; {extra}" if extra else ""
        pct = (100.0 * done / total) if total else 100.0
        self.emit(
            f"[flow][progress] done {done}/{total} ({pct:.1f}%): {label}; "
            f"stage={format_duration(now - started_s)}, "
            f"elapsed={format_duration(now - self.start_s)}, "
            f"eta={format_duration(eta)}{suffix}"
        )

    def eta_seconds(self, *, now: float | None = None) -> float | None:
        now = time.perf_counter() if now is None else float(now)
        with self.lock:
            done = int(self.completed)
            total = int(self.total_units)
        if not self.enabled or total <= 0 or done <= 0:
            return None
        avg_s = max(0.0, now - self.start_s) / float(done)
        return max(0, total - done) * avg_s

    def heartbeat(self, label: str, started_s: float) -> "ProgressHeartbeat":
        return ProgressHeartbeat(self, label, started_s)


class ProgressHeartbeat:
    def __init__(self, meter: FlowProgress, label: str, started_s: float) -> None:
        self.meter = meter
        self.label = label
        self.started_s = float(started_s)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "ProgressHeartbeat":
        if self.meter.enabled and self.meter.every_s > 0:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=0.2)
        return False

    def _run(self) -> None:
        every = max(1.0, float(self.meter.every_s))
        while not self.stop_event.wait(every):
            now = time.perf_counter()
            with self.meter.lock:
                done = int(self.meter.completed)
                total = int(self.meter.total_units)
            self.meter.emit(
                f"[flow][progress] running {done + 1}/{total}: {self.label}; "
                f"stage_elapsed={format_duration(now - self.started_s)}, "
                f"total_elapsed={format_duration(now - self.meter.start_s)}, "
                f"eta_after_completed={format_duration(self.meter.eta_seconds(now=now))}"
            )


def progress_enabled_from_args(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "quiet_progress", False)):
        return False
    env = str(os.environ.get("PVFV_FLOW_PROGRESS", "1")).lower().strip()
    return env not in {"0", "false", "no", "off"}


def coarse_row_progress_extra(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    try:
        pieces.append(f"eK={100.0 * abs(float(row.get('e_K', 0.0))):.2f}%")
    except Exception:
        pass
    try:
        t_total = float(row.get("t_total_s", 0.0))
        t_ref = float(row.get("t_ref_total_s", 0.0))
        if t_total > 0 and t_ref > 0:
            pieces.append(f"speedup={t_ref / t_total:.1f}x")
            pieces.append(f"run={format_duration(t_total)}")
    except Exception:
        pass
    try:
        pieces.append(f"label={1000.0 * float(row.get('t_label_s', 0.0)):.2f}ms")
    except Exception:
        pass
    return ", ".join(pieces)


def load_mask_npz(ns: dict[str, Any], path: Path):
    cp = ns["cp"]
    data = np.load(path, allow_pickle=True)
    mask_np = np.asarray(data["mask"], dtype=bool)
    return cp.asarray(mask_np)


def install_half_only_density(ns: dict[str, Any]) -> None:
    original = ns["pvfv_half_gfps_seed_specs_for_density"]

    def half_only(mask, stride_zyx, *args, **kwargs):
        try:
            specs = original(mask, stride_zyx, *args, **kwargs)
        except TypeError:
            specs = original(mask, stride_zyx, ["half"])
        return [
            item for item in specs
            if str(item[0].get("family", "")).lower() == "stride_half_admissible"
        ]

    ns["pvfv_half_gfps_seed_specs_for_density"] = half_only


def install_runtime_cfg_attr_preservation(ns: dict[str, Any]) -> None:
    """Preserve runner-only config attributes across notebook pvfv_clone_cfg calls."""
    original = ns["pvfv_clone_cfg"]
    if getattr(original, "_pvfv_preserves_runtime_attrs", False):
        return

    runtime_attrs = (
        "linear_solver_mode",
        "momentum_solver_mode",
        "projection_interval",
        "pvfv_progress_label",
    )

    def clone_with_runtime_attrs(cfg=None, **updates):
        out = original(cfg, **updates)
        if cfg is not None:
            for attr in runtime_attrs:
                if hasattr(cfg, attr) and not hasattr(out, attr):
                    setattr(out, attr, getattr(cfg, attr))
        for attr in runtime_attrs:
            if attr in updates:
                setattr(out, attr, updates[attr])
        return out

    clone_with_runtime_attrs._pvfv_preserves_runtime_attrs = True  # type: ignore[attr-defined]
    ns["pvfv_clone_cfg"] = clone_with_runtime_attrs
    print("[flow] runtime cfg attributes preserved across notebook clones")


def install_skip_zero_area_diagnostic(ns: dict[str, Any]) -> None:
    def skipped_zero_area_count(*_args, **_kwargs) -> int:
        return -1

    ns["pvfv_zero_area_candidate_count_cpu"] = skipped_zero_area_count
    print("[flow] zero-area CPU diagnostic skipped; N_0A is reported as -1")


def install_steady_scalar_initial_guess(ns: dict[str, Any]) -> None:
    """Add a GPU scalar Stokes warm-start mode for coarse pressure polishing."""
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    original = ns["make_initial_velocity_gpu"]

    if "_pvfv_steady_scalar_velocity_gpu" not in ns:
        def solve_scalar_velocity_x(geom, cfg):
            n = int(geom.n_cells)
            U = cp.zeros((n, 3), dtype=cp.float64)
            fx = float(getattr(cfg, "body_force", (0.0, 0.0, 0.0))[0])
            if abs(fx) <= 1.0e-300:
                return U

            # Scalar steady diffusion-wall balance:
            #     nu * (L_face + T_wall) u_x = V * f_x
            # This is the coarse steady Stokes backbone used either as a
            # physics-aware initial field or as a direct projected solve.
            A = geom.laplacian + cpx_sp.diags(geom.twall, 0, shape=(n, n), dtype=cp.float64, format="csr")
            rhs = geom.volume * fx / max(float(cfg.nu), 1.0e-300)
            M = None
            if bool(getattr(cfg, "use_jacobi_preconditioner", True)) and "_make_jacobi_preconditioner_gpu" in ns:
                M = ns["_make_jacobi_preconditioner_gpu"](A)
            try:
                sol, info = ns["_cg_gpu_tol"](
                    A,
                    rhs,
                    min(float(getattr(cfg, "momentum_tol", 1.0e-8)), 1.0e-8),
                    int(getattr(cfg, "momentum_maxiter", 5000)),
                    M=M,
                )
                if int(info) != 0 and hasattr(ns.get("cpx_spla"), "spsolve"):
                    sol = ns["cpx_spla"].spsolve(A, rhs)
            except Exception:
                if hasattr(ns.get("cpx_spla"), "spsolve"):
                    sol = ns["cpx_spla"].spsolve(A, rhs)
                else:
                    raise
            U[:, 0] = sol
            return U

        ns["_pvfv_steady_scalar_velocity_gpu"] = solve_scalar_velocity_x

    if getattr(original, "_pvfv_steady_scalar_initial", False):
        return

    def warm_start(geom, cfg):
        mode = str(getattr(cfg, "initial_velocity_mode", "zero")).lower().strip()
        if mode not in STEADY_SCALAR_INITIAL_MODES:
            return original(geom, cfg)
        return ns["_pvfv_steady_scalar_velocity_gpu"](geom, cfg)

    warm_start._pvfv_steady_scalar_initial = True  # type: ignore[attr-defined]
    ns["make_initial_velocity_gpu"] = warm_start
    print("[flow] steady-scalar GPU initial guess mode installed")


def install_steady_scalar_solver_modes(ns: dict[str, Any]) -> None:
    """Add direct/projected GPU steady-scalar coarse solver modes."""
    install_steady_scalar_initial_guess(ns)
    cp = ns["cp"]
    original = ns["run_velocity_pressure_projection_gpu"]

    if getattr(original, "_pvfv_steady_scalar_solver_modes", False):
        return

    def finalize_result(geom, cfg, U, p, phi, elapsed_s: float, *, run_label: str,
                        steps_completed: int, solver_failure: str = "", failure_step: int = -1):
        div = ns["face_divergence_gpu"](phi, geom)
        mass_inf = float(cp.max(cp.abs(div / geom.volume)).get())
        mass_l2 = float(cp.sqrt(cp.mean((div / geom.volume) ** 2)).get())
        mean_U = cp.sum(U * geom.volume[:, None], axis=0) / cp.sum(geom.volume)
        mean_U_host = [float(v) for v in mean_U.get()]
        fx = float(cfg.body_force[0])
        k_eff_x = float((cfg.nu * mean_U[0] / fx).get()) if abs(fx) > 0 else float("nan")
        mom = ns["steady_momentum_residual_gpu"](U, p, phi, geom, cfg)
        mom_inf = float(cp.max(cp.linalg.norm(mom, axis=1)).get())
        energy = float(ns["kinetic_energy_gpu"](U, geom).get())
        cfl = float(ns["convective_cfl_gpu"](phi, geom, cfg).get())
        diffusion_number = float(ns["diffusion_stiffness_number_gpu"](geom, cfg).get())
        steady_converged = (
            not solver_failure
            and mass_inf < max(float(getattr(cfg, "projection_tol", 1.0e-8)) * 10.0, 1.0e-12)
            and mom_inf < max(10.0 * float(getattr(cfg, "steady_tol", 1.0e-8)), 1.0e-10)
        )
        return dict(
            U=U,
            p=p,
            phi=phi,
            div=div,
            elapsed_s=float(elapsed_s),
            mass_inf_per_volume=float(mass_inf),
            mass_l2_per_volume=float(mass_l2),
            mean_U_x=mean_U_host[0],
            mean_U_y=mean_U_host[1],
            mean_U_z=mean_U_host[2],
            K_eff_x=float(k_eff_x),
            n_cells=int(geom.n_cells),
            n_faces=int(geom.owner.size),
            last_mass_inf=float(mass_inf),
            steady_converged=bool(steady_converged),
            final_rel_dU=0.0,
            final_rel_dU_ref=0.0,
            steady_momentum_inf=float(mom_inf),
            steps_completed=int(steps_completed),
            failure_step=int(failure_step),
            solver_failure=str(solver_failure),
            implicit_diffusion=True,
            enable_convection=bool(cfg.enable_convection),
            energy_initial=float(energy),
            energy_final=float(energy),
            energy_max=float(energy),
            max_cfl=float(cfl),
            max_convection_inf=0.0,
            max_diffusion_number=float(diffusion_number),
            nan_or_inf_detected=not all(math.isfinite(v) for v in (mass_inf, mass_l2, k_eff_x, mom_inf, energy, cfl)),
            run_label=str(run_label),
            history=[],
            scalar_solver_mode=str(getattr(cfg, "initial_velocity_mode", "")),
        )

    def run_scalar_mode(geom, cfg, initial_U=None, run_label: str = "main"):
        mode = str(getattr(cfg, "initial_velocity_mode", "zero")).lower().strip()
        if mode not in STEADY_SCALAR_SOLVER_MODES:
            return original(geom, cfg, initial_U=initial_U, run_label=run_label)

        t0 = time.perf_counter()
        U = initial_U.copy() if initial_U is not None else ns["_pvfv_steady_scalar_velocity_gpu"](geom, cfg)
        p = cp.zeros(int(geom.n_cells), dtype=cp.float64)
        phi = ns["face_flux_from_velocity_gpu"](U, geom)
        solver_failure = ""
        failure_step = -1
        steps_completed = 0

        if "projected" in mode:
            steps_completed = 1
            try:
                M_p = None
                if bool(getattr(cfg, "use_jacobi_preconditioner", True)) and "_make_jacobi_preconditioner_gpu" in ns:
                    M_p = ns["_make_jacobi_preconditioner_gpu"](geom.laplacian)
                div_star = ns["face_divergence_gpu"](phi, geom)
                rhs = -(float(cfg.rho) / float(cfg.dt)) * div_star
                pcorr, _ = ns["solve_pressure_correction_gpu"](rhs, geom, cfg, M_p=M_p)
                phi = ns["correct_flux_gpu"](phi, pcorr, geom, cfg)
                grad_pcorr = ns["lsq_gradient_gpu"](pcorr, geom, cfg)
                U = U - float(cfg.dt / cfg.rho) * grad_pcorr
                if "reconstruct" in mode:
                    U = ns["reconstruct_velocity_from_flux_gpu"](phi, U, geom, cfg)
                p = pcorr - cp.mean(pcorr)
            except RuntimeError as exc:
                solver_failure = str(exc)
                failure_step = 1
                if bool(getattr(cfg, "raise_on_solver_failure", True)):
                    raise
                print(f"[flow] {run_label} scalar projected solver failure: {solver_failure}")

        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        return finalize_result(
            geom,
            cfg,
            U,
            p,
            phi,
            elapsed,
            run_label=run_label,
            steps_completed=steps_completed,
            solver_failure=solver_failure,
            failure_step=failure_step,
        )

    run_scalar_mode._pvfv_steady_scalar_solver_modes = True  # type: ignore[attr-defined]
    run_scalar_mode.__name__ = getattr(original, "__name__", "run_velocity_pressure_projection_gpu")
    run_scalar_mode.__doc__ = getattr(original, "__doc__", None)
    ns["run_velocity_pressure_projection_gpu"] = run_scalar_mode
    print("[flow] steady-scalar direct/projected solver modes installed")


def install_monolithic_stokes_solver_modes(ns: dict[str, Any]) -> None:
    """Add an explicitly selected diagnostic dense Stokes saddle-point solve."""
    cp = ns["cp"]
    original = ns["run_velocity_pressure_projection_gpu"]
    if getattr(original, "_pvfv_monolithic_stokes_solver_modes", False):
        return

    def mode_enabled(cfg) -> bool:
        formulation = str(
            getattr(cfg, "solver_formulation", PRODUCTION_PRESSURE_CORRECTION_FORMULATION)
        ).lower().strip()
        return formulation == DIAGNOSTIC_DENSE_KKT_FORMULATION

    def build_lsq_gradient_matrix(geom, cfg):
        n = int(geom.n_cells)
        owner = geom.owner
        neigh = geom.neigh
        dvec = geom.dvec
        weights = pressure_gradient_weight(ns, geom, cfg)
        M = cp.zeros((n, 3, 3), dtype=cp.float64)
        C = cp.zeros((n, 3, n), dtype=cp.float64)
        for a in range(3):
            coeff_a = weights * dvec[:, a]
            cp.add.at(C, (owner, a, neigh), coeff_a)
            cp.add.at(C, (owner, a, owner), -coeff_a)
            cp.add.at(C, (neigh, a, neigh), coeff_a)
            cp.add.at(C, (neigh, a, owner), -coeff_a)
            for b in range(3):
                mval = weights * dvec[:, a] * dvec[:, b]
                cp.add.at(M, (owner, a, b), mval)
                cp.add.at(M, (neigh, a, b), mval)
        diag = cp.arange(3)
        M[:, diag, diag] += float(cfg.tikhonov)
        Minv = cp.linalg.inv(M)
        G_cell = cp.einsum("iab,ibk->iak", Minv, C)
        return G_cell.reshape((3 * n, n))

    def build_divergence_matrix(geom):
        n = int(geom.n_cells)
        owner = geom.owner
        neigh = geom.neigh
        D = cp.zeros((n, 3 * n), dtype=cp.float64)
        for c in range(3):
            coeff_owner = geom.w_owner * geom.avec[:, c]
            coeff_neigh = geom.w_neigh * geom.avec[:, c]
            col_owner = 3 * owner + c
            col_neigh = 3 * neigh + c
            cp.add.at(D, (owner, col_owner), coeff_owner)
            cp.add.at(D, (owner, col_neigh), coeff_neigh)
            cp.add.at(D, (neigh, col_owner), -coeff_owner)
            cp.add.at(D, (neigh, col_neigh), -coeff_neigh)
        return D

    def finalize_result(geom, cfg, U, p, phi, elapsed_s: float, run_label: str, U_momentum=None):
        div = ns["face_divergence_gpu"](phi, geom)
        mass_inf = float(cp.max(cp.abs(div / geom.volume)).get())
        mass_l2 = float(cp.sqrt(cp.mean((div / geom.volume) ** 2)).get())
        mean_U = cp.sum(U * geom.volume[:, None], axis=0) / cp.sum(geom.volume)
        mean_U_host = [float(v) for v in mean_U.get()]
        fx = float(cfg.body_force[0])
        k_eff_x = float((float(cfg.nu) * mean_U[0] / fx).get()) if abs(fx) > 0 else float("nan")
        U_for_momentum = U if U_momentum is None else U_momentum
        mom = ns["steady_momentum_residual_gpu"](U_for_momentum, p, phi, geom, cfg)
        mom_inf = float(cp.max(cp.linalg.norm(mom, axis=1)).get())
        energy = float(ns["kinetic_energy_gpu"](U, geom).get())
        cfl = float(ns["convective_cfl_gpu"](phi, geom, cfg).get())
        diffusion_number = float(ns["diffusion_stiffness_number_gpu"](geom, cfg).get())
        steady_converged = (
            mass_inf < max(float(getattr(cfg, "projection_tol", 1.0e-8)) * 10.0, 1.0e-12)
            and mom_inf < max(10.0 * float(getattr(cfg, "steady_tol", 1.0e-7)), 1.0e-6)
        )
        return dict(
            U=U,
            p=p,
            phi=phi,
            div=div,
            elapsed_s=float(elapsed_s),
            mass_inf_per_volume=float(mass_inf),
            mass_l2_per_volume=float(mass_l2),
            mean_U_x=mean_U_host[0],
            mean_U_y=mean_U_host[1],
            mean_U_z=mean_U_host[2],
            K_eff_x=float(k_eff_x),
            n_cells=int(geom.n_cells),
            n_faces=int(geom.owner.size),
            last_mass_inf=float(mass_inf),
            steady_converged=bool(steady_converged),
            final_rel_dU=0.0,
            final_rel_dU_ref=0.0,
            steady_momentum_inf=float(mom_inf),
            steps_completed=1,
            failure_step=-1,
            solver_failure="",
            implicit_diffusion=True,
            enable_convection=bool(cfg.enable_convection),
            energy_initial=float(energy),
            energy_final=float(energy),
            energy_max=float(energy),
            max_cfl=float(cfl),
            max_convection_inf=0.0,
            max_diffusion_number=float(diffusion_number),
            nan_or_inf_detected=not all(math.isfinite(v) for v in (mass_inf, mass_l2, k_eff_x, mom_inf, energy, cfl)),
            run_label=str(run_label),
            history=[],
            monolithic_stokes=True,
            velocity_reconstructed_from_flux=bool(getattr(cfg, "velocity_reconstruct_from_flux", False)),
            velocity_reconstruction_lambda=float(getattr(cfg, "velocity_reconstruction_lambda", float("nan"))),
        )

    def run_monolithic(geom, cfg, initial_U=None, run_label: str = "main"):
        if not mode_enabled(cfg):
            return original(geom, cfg, initial_U=initial_U, run_label=run_label)

        t0 = time.perf_counter()
        n = int(geom.n_cells)
        L = geom.laplacian.toarray()
        eps = float(getattr(cfg, "pressure_gauge_eps", 0.0))
        if eps > 0.0:
            L = L - eps * cp.eye(n, dtype=cp.float64)
        A0 = float(cfg.nu) * (L + cp.diag(geom.twall))
        A = cp.kron(A0, cp.eye(3, dtype=cp.float64))
        G = build_lsq_gradient_matrix(geom, cfg)
        B = G * (geom.volume.repeat(3)[:, None] / float(cfg.rho))
        D = build_divergence_matrix(geom)

        system_size = 4 * n + 1
        mat = cp.zeros((system_size, system_size), dtype=cp.float64)
        rhs = cp.zeros(system_size, dtype=cp.float64)
        mat[: 3 * n, : 3 * n] = A
        mat[: 3 * n, 3 * n : 4 * n] = B
        mat[3 * n : 4 * n, : 3 * n] = D
        mat[3 * n : 4 * n, 4 * n] = 1.0
        mat[4 * n, 3 * n : 4 * n] = 1.0 / float(n)
        body = cp.asarray(cfg.body_force, dtype=cp.float64)
        rhs[: 3 * n] = (geom.volume[:, None] * body[None, :]).reshape(3 * n)

        sol = cp.linalg.solve(mat, rhs)
        U = sol[: 3 * n].reshape((n, 3))
        p = sol[3 * n : 4 * n]
        phi = ns["face_flux_from_velocity_gpu"](U, geom)
        U_solve = U
        if bool(getattr(cfg, "velocity_reconstruct_from_flux", False)):
            U = ns["reconstruct_velocity_from_flux_gpu"](phi, U_solve, geom, cfg)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        result = finalize_result(geom, cfg, U, p, phi, elapsed, run_label, U_momentum=U_solve)
        result["solver_formulation_id"] = (
            "diagnostic_dense_kkt_lsq_gradient_interpolated_divergence_v1"
        )
        result["solver_result_scope"] = "diagnostic_control"
        return result

    run_monolithic._pvfv_monolithic_stokes_solver_modes = True  # type: ignore[attr-defined]
    run_monolithic.__name__ = getattr(original, "__name__", "run_velocity_pressure_projection_gpu")
    run_monolithic.__doc__ = getattr(original, "__doc__", None)
    ns["run_velocity_pressure_projection_gpu"] = run_monolithic
    print("[flow] explicit diagnostic dense-KKT formulation installed")


def install_direct_sparse_linear_solver(ns: dict[str, Any]) -> None:
    """Add a cfg.linear_solver_mode='direct' option for coarse GPU sparse solves."""
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    cpx_spla = ns.get("cpx_spla")
    if cpx_spla is None or not hasattr(cpx_spla, "spsolve"):
        raise RuntimeError("cupyx.scipy.sparse.linalg.spsolve is not available in this kernel")

    original_pressure = ns["solve_pressure_correction_gpu"]
    original_momentum = ns["implicit_momentum_predictor_gpu"]

    if getattr(original_pressure, "_pvfv_direct_sparse_linear_solver", False):
        return

    def direct_enabled(cfg) -> bool:
        return str(getattr(cfg, "linear_solver_mode", "cg")).lower().strip() in {"direct", "spsolve"}

    def solve_pressure(rhs, geom, cfg, M_p=None):
        if not direct_enabled(cfg):
            return original_pressure(rhs, geom, cfg, M_p=M_p)
        rhs = rhs - cp.mean(rhs)
        n = int(geom.n_cells)
        eps = float(getattr(cfg, "pressure_gauge_eps", 1.0e-12))
        A = geom.laplacian + cpx_sp.eye(n, dtype=cp.float64, format="csr") * eps
        pcorr = cpx_spla.spsolve(A, rhs)
        pcorr = pcorr - cp.mean(pcorr)
        return pcorr, 0

    def solve_momentum(U, p, phi, geom, cfg, A_vel, M_vel=None):
        if not direct_enabled(cfg):
            return original_momentum(U, p, phi, geom, cfg, A_vel, M_vel=M_vel)
        gradp = ns["lsq_gradient_gpu"](p, geom, cfg)
        if bool(cfg.enable_convection):
            Cterm = ns["convection_term_gpu"](U, phi, geom)
        else:
            Cterm = cp.zeros_like(U)
        if "nonorthogonal_diffusion_correction_gpu" in ns:
            Dnoc = ns["nonorthogonal_diffusion_correction_gpu"](U, geom, cfg)
        else:
            Dnoc = cp.zeros_like(U)
        body = cp.asarray(cfg.body_force, dtype=cp.float64)[None, :]
        rhs = (
            (geom.volume[:, None] / float(cfg.dt)) * U
            + geom.volume[:, None] * (body - gradp / float(cfg.rho) - Cterm + Dnoc)
        )
        Ustar = cp.empty_like(U)
        for c in range(3):
            Ustar[:, c] = cpx_spla.spsolve(A_vel, rhs[:, c])
        return Ustar

    solve_pressure._pvfv_direct_sparse_linear_solver = True  # type: ignore[attr-defined]
    solve_momentum._pvfv_direct_sparse_linear_solver = True  # type: ignore[attr-defined]
    ns["solve_pressure_correction_gpu"] = solve_pressure
    ns["implicit_momentum_predictor_gpu"] = solve_momentum
    print("[flow] direct sparse linear solver mode installed")


def install_dense_lu_linear_solver(ns: dict[str, Any]) -> None:
    """Use cached GPU dense LU factors for small coarse pressure/momentum solves."""
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    cpx_linalg = importlib.import_module("cupyx.scipy.linalg")
    original_pressure = ns["solve_pressure_correction_gpu"]
    original_momentum = ns["implicit_momentum_predictor_gpu"]

    if getattr(original_pressure, "_pvfv_dense_lu_linear_solver", False):
        return

    pressure_cache: dict[tuple[int, int, float], tuple[Any, Any]] = {}
    momentum_cache: dict[int, tuple[Any, Any]] = {}

    def dense_enabled(cfg) -> bool:
        return str(getattr(cfg, "linear_solver_mode", "cg")).lower().strip() in DENSE_LU_SOLVER_MODES

    def pressure_factor(geom, cfg):
        n = int(geom.n_cells)
        eps = float(getattr(cfg, "pressure_gauge_eps", 1.0e-12))
        key = (id(geom.laplacian), n, eps)
        cached = pressure_cache.get(key)
        if cached is None or cached[0] is not geom.laplacian:
            A = geom.laplacian + cpx_sp.eye(n, dtype=cp.float64, format="csr") * eps
            factor = cpx_linalg.lu_factor(A.toarray())
            pressure_cache[key] = (geom.laplacian, factor)
            return factor
        return cached[1]

    def momentum_factor(A_vel):
        key = id(A_vel)
        cached = momentum_cache.get(key)
        if cached is None or cached[0] is not A_vel:
            factor = cpx_linalg.lu_factor(A_vel.toarray())
            momentum_cache[key] = (A_vel, factor)
            return factor
        return cached[1]

    def solve_pressure(rhs, geom, cfg, M_p=None):
        if not dense_enabled(cfg):
            return original_pressure(rhs, geom, cfg, M_p=M_p)
        rhs = rhs - cp.mean(rhs)
        pcorr = cpx_linalg.lu_solve(pressure_factor(geom, cfg), rhs)
        pcorr = pcorr - cp.mean(pcorr)
        return pcorr, 0

    def solve_momentum(U, p, phi, geom, cfg, A_vel, M_vel=None):
        if not dense_enabled(cfg):
            return original_momentum(U, p, phi, geom, cfg, A_vel, M_vel=M_vel)
        gradp = ns["lsq_gradient_gpu"](p, geom, cfg)
        Cterm = ns["convection_term_gpu"](U, phi, geom) if bool(cfg.enable_convection) else cp.zeros_like(U)
        if "nonorthogonal_diffusion_correction_gpu" in ns:
            Dnoc = ns["nonorthogonal_diffusion_correction_gpu"](U, geom, cfg)
        else:
            Dnoc = cp.zeros_like(U)
        body = cp.asarray(cfg.body_force, dtype=cp.float64)[None, :]
        rhs = (
            (geom.volume[:, None] / float(cfg.dt)) * U
            + geom.volume[:, None] * (body - gradp / float(cfg.rho) - Cterm + Dnoc)
        )
        return cpx_linalg.lu_solve(momentum_factor(A_vel), rhs)

    solve_pressure._pvfv_dense_lu_linear_solver = True  # type: ignore[attr-defined]
    solve_momentum._pvfv_dense_lu_linear_solver = True  # type: ignore[attr-defined]
    ns["solve_pressure_correction_gpu"] = solve_pressure
    ns["implicit_momentum_predictor_gpu"] = solve_momentum
    ns["_pvfv_dense_lu_pressure_cache"] = pressure_cache
    ns["_pvfv_dense_lu_momentum_cache"] = momentum_cache
    print("[flow] cached GPU dense-LU linear solver mode installed")


def install_lsq_gradient_batched_compat(ns: dict[str, Any]) -> None:
    """Use an explicit batched 3x3 solve for the existing LSQ gradient kernel."""
    cp = ns["cp"]
    kernel = ns.get("_LSQ_ASSEMBLE_KERNEL")
    threads_blocks = ns.get("_threads_blocks")
    if kernel is None or threads_blocks is None:
        raise RuntimeError("LSQ gradient kernel dependencies are unavailable")
    original = ns["lsq_gradient_gpu"]
    if getattr(original, "_pvfv_lsq_batched_compat", False):
        return

    def lsq_gradient_gpu(scalar, geom, cfg):
        n = int(geom.n_cells)
        nf = int(geom.owner.size)
        M = cp.zeros((n, 3, 3), dtype=cp.float64)
        b = cp.zeros((n, 3), dtype=cp.float64)
        grid, block = threads_blocks(nf)
        kernel(
            grid,
            block,
            (
                nf,
                geom.owner,
                geom.neigh,
                geom.dvec.reshape(-1),
                pressure_gradient_weight(ns, geom, cfg),
                scalar,
                M.reshape(-1),
                b.reshape(-1),
            ),
        )
        diag = cp.arange(3)
        M[:, diag, diag] += float(cfg.tikhonov)

        a = M[:, 0, 0]
        bb = M[:, 0, 1]
        c = M[:, 0, 2]
        d = M[:, 1, 1]
        e = M[:, 1, 2]
        f = M[:, 2, 2]
        det = a * (d * f - e * e) - bb * (bb * f - c * e) + c * (bb * e - c * d)
        det = cp.where(cp.abs(det) > 1.0e-300, det, cp.sign(det + 1.0e-300) * 1.0e-300)
        inv00 = (d * f - e * e) / det
        inv01 = (c * e - bb * f) / det
        inv02 = (bb * e - c * d) / det
        inv11 = (a * f - c * c) / det
        inv12 = (bb * c - a * e) / det
        inv22 = (a * d - bb * bb) / det

        out = cp.empty_like(b)
        rhs0 = b[:, 0]
        rhs1 = b[:, 1]
        rhs2 = b[:, 2]
        out[:, 0] = inv00 * rhs0 + inv01 * rhs1 + inv02 * rhs2
        out[:, 1] = inv01 * rhs0 + inv11 * rhs1 + inv12 * rhs2
        out[:, 2] = inv02 * rhs0 + inv12 * rhs1 + inv22 * rhs2
        return out

    lsq_gradient_gpu._pvfv_lsq_batched_compat = True  # type: ignore[attr-defined]
    lsq_gradient_gpu.__name__ = getattr(original, "__name__", "lsq_gradient_gpu")
    lsq_gradient_gpu.__doc__ = getattr(original, "__doc__", None)
    ns["lsq_gradient_gpu"] = lsq_gradient_gpu
    print("[flow] existing LSQ gradient kernel uses explicit batched 3x3 solve")


def install_block_momentum_cg(ns: dict[str, Any]) -> None:
    """Solve the three momentum components as one GPU multi-RHS CG problem."""
    cp = ns["cp"]
    original = ns["implicit_momentum_predictor_gpu"]
    if getattr(original, "_pvfv_block_momentum_cg", False):
        return

    def block_enabled(cfg) -> bool:
        mode = str(getattr(cfg, "momentum_solver_mode", "cg")).lower().strip()
        linear = str(getattr(cfg, "linear_solver_mode", "cg")).lower().strip()
        return mode in {"block_cg", "multi_rhs_cg", "spmm_cg"} and linear not in {"direct", "spsolve"}

    def block_cg(A, B, tol: float, maxiter: int):
        B = cp.asarray(B, dtype=cp.float64)
        X = cp.zeros_like(B)
        R = B.copy()
        bnorm = cp.linalg.norm(B, axis=0)
        active = bnorm > 1.0e-300
        if not bool(cp.any(active).get()):
            return X, 0

        diag = A.diagonal().astype(cp.float64)
        inv_diag = 1.0 / cp.where(cp.abs(diag) > 1.0e-300, diag, 1.0)
        Z = R * inv_diag[:, None]
        P = Z.copy()
        rz = cp.sum(R * Z, axis=0)
        target = cp.maximum(float(tol) * bnorm, 1.0e-300)
        info = int(maxiter)

        for _it in range(1, int(maxiter) + 1):
            AP = A.dot(P)
            denom = cp.sum(P * AP, axis=0)
            alpha = cp.where(active, rz / cp.where(cp.abs(denom) > 1.0e-300, denom, 1.0), 0.0)
            X = X + P * alpha[None, :]
            R = R - AP * alpha[None, :]
            res = cp.linalg.norm(R, axis=0)
            active_new = res > target
            if not bool(cp.any(active_new).get()):
                info = 0
                break
            Z = R * inv_diag[:, None]
            rz_new = cp.sum(R * Z, axis=0)
            beta = cp.where(active_new, rz_new / cp.where(cp.abs(rz) > 1.0e-300, rz, 1.0), 0.0)
            P = Z + P * beta[None, :]
            rz = rz_new
            active = active_new
        return X, info

    def solve_momentum(U, p, phi, geom, cfg, A_vel, M_vel=None):
        if not block_enabled(cfg):
            return original(U, p, phi, geom, cfg, A_vel, M_vel=M_vel)

        gradp = ns["lsq_gradient_gpu"](p, geom, cfg)
        Cterm = ns["convection_term_gpu"](U, phi, geom) if bool(cfg.enable_convection) else cp.zeros_like(U)
        if "nonorthogonal_diffusion_correction_gpu" in ns:
            Dnoc = ns["nonorthogonal_diffusion_correction_gpu"](U, geom, cfg)
        else:
            Dnoc = cp.zeros_like(U)
        body = cp.asarray(cfg.body_force, dtype=cp.float64)[None, :]
        rhs = (
            (geom.volume[:, None] / float(cfg.dt)) * U
            + geom.volume[:, None] * (body - gradp / float(cfg.rho) - Cterm + Dnoc)
        )
        Ustar, info = block_cg(
            A_vel,
            rhs,
            float(getattr(cfg, "momentum_tol", 1.0e-8)),
            int(getattr(cfg, "momentum_maxiter", 5000)),
        )
        if int(info) != 0:
            raise RuntimeError(f"GPU block momentum CG did not converge; info={int(info)}")
        return Ustar

    solve_momentum._pvfv_block_momentum_cg = True  # type: ignore[attr-defined]
    ns["implicit_momentum_predictor_gpu"] = solve_momentum
    print("[flow] block multi-RHS momentum CG mode installed")


def install_projection_interval_solver(ns: dict[str, Any]) -> None:
    """Add cfg.projection_interval>1 support for cheaper coarse projection loops."""
    cp = ns["cp"]
    original = ns["run_velocity_pressure_projection_gpu"]
    if getattr(original, "_pvfv_projection_interval_solver", False):
        return

    def finite_scalar(x: float) -> bool:
        return math.isfinite(float(x))

    def run_with_interval(geom, cfg, initial_U=None, run_label: str = "main"):
        interval = int(getattr(cfg, "projection_interval", 1))
        mode = str(getattr(cfg, "initial_velocity_mode", "zero")).lower().strip()
        if interval <= 1 or mode in STEADY_SCALAR_SOLVER_MODES:
            return original(geom, cfg, initial_U=initial_U, run_label=run_label)

        n = int(geom.n_cells)
        U = initial_U.copy() if initial_U is not None else ns["make_initial_velocity_gpu"](geom, cfg)
        p = cp.zeros(n, dtype=cp.float64)
        phi = ns["face_flux_from_velocity_gpu"](U, geom)
        A_vel = ns["build_implicit_velocity_matrix_gpu"](geom, cfg) if bool(cfg.implicit_diffusion) else None
        M_vel = ns["_make_jacobi_preconditioner_gpu"](A_vel) if (A_vel is not None and bool(getattr(cfg, "use_jacobi_preconditioner", True))) else None
        M_p = ns["_make_jacobi_preconditioner_gpu"](geom.laplacian) if bool(getattr(cfg, "use_jacobi_preconditioner", True)) else None

        t0 = time.perf_counter()
        last_mass_inf = None
        steady_converged = False
        rel_change = float("inf")
        rel_change_ref = float("inf")
        mom_inf = float("inf")
        U_prev = U.copy()
        U0_norm_inf = float(cp.max(cp.linalg.norm(U, axis=1)).get())
        energy_initial = float(ns["kinetic_energy_gpu"](U, geom).get())
        energy_max = energy_initial
        energy_now = energy_initial
        max_cfl = 0.0
        max_convection_inf = 0.0
        max_diffusion_number = float(ns["diffusion_stiffness_number_gpu"](geom, cfg).get())
        nan_or_inf_detected = False
        history = []
        solver_failure = ""
        failure_step = -1
        steps_completed = 0
        projection_count = 0
        cfl_now = float(ns["convective_cfl_gpu"](phi, geom, cfg).get())

        for step in range(1, int(cfg.n_steps) + 1):
            try:
                if bool(cfg.implicit_diffusion):
                    Ustar = ns["implicit_momentum_predictor_gpu"](U, p, phi, geom, cfg, A_vel, M_vel=M_vel)
                else:
                    gradp = ns["lsq_gradient_gpu"](p, geom, cfg)
                    Dterm = ns["diffusion_term_gpu"](U, geom, cfg)
                    Cterm = ns["convection_term_gpu"](U, phi, geom) if bool(cfg.enable_convection) else cp.zeros_like(U)
                    body = cp.asarray(cfg.body_force, dtype=cp.float64)[None, :]
                    Ustar = U + float(cfg.dt) * (-Cterm + Dterm + body - gradp / float(cfg.rho))

                phi_star = ns["face_flux_from_velocity_gpu"](Ustar, geom)
                do_project = (step % interval == 0) or (step == int(cfg.n_steps))
                if do_project:
                    div_star = ns["face_divergence_gpu"](phi_star, geom)
                    rhs = -(float(cfg.rho) / float(cfg.dt)) * div_star
                    pcorr, _ = ns["solve_pressure_correction_gpu"](rhs, geom, cfg, M_p=M_p)
                    phi = ns["correct_flux_gpu"](phi_star, pcorr, geom, cfg)
                    grad_pcorr = ns["lsq_gradient_gpu"](pcorr, geom, cfg)
                    U = Ustar - float(cfg.dt / cfg.rho) * grad_pcorr
                    if bool(cfg.velocity_reconstruct_from_flux):
                        U = ns["reconstruct_velocity_from_flux_gpu"](phi, U, geom, cfg)
                    p = p + pcorr
                    p = p - cp.mean(p)
                    projection_count += 1
                else:
                    U = Ustar
                    phi = phi_star
                steps_completed = int(step)
            except RuntimeError as exc:
                solver_failure = str(exc)
                failure_step = int(step)
                nan_or_inf_detected = True
                if bool(getattr(cfg, "raise_on_solver_failure", True)):
                    raise
                print(f"[flow] {run_label} projection-interval solver failure at step {step}: {solver_failure}")
                break

            do_diag = bool(getattr(cfg, "diagnostics_every_step", False)) or (
                cfg.report_every and (step % int(cfg.report_every) == 0 or step == 1 or step == int(cfg.n_steps))
            )
            if do_diag:
                energy_now = float(ns["kinetic_energy_gpu"](U, geom).get())
                cfl_now = float(ns["convective_cfl_gpu"](phi, geom, cfg).get())
                energy_max = max(float(energy_max), energy_now)
                max_cfl = max(float(max_cfl), cfl_now)
                if not (finite_scalar(energy_now) and finite_scalar(cfl_now)):
                    nan_or_inf_detected = True
                if bool(cfg.enable_convection):
                    Cdiag = ns["convection_term_gpu"](U, phi, geom)
                    conv_inf_now = float(cp.max(cp.linalg.norm(Cdiag, axis=1)).get())
                else:
                    conv_inf_now = 0.0
                max_convection_inf = max(float(max_convection_inf), float(conv_inf_now))
                if bool(getattr(cfg, "diagnostics_every_step", False)):
                    history.append({"step": int(step), "energy": energy_now, "cfl": cfl_now, "convection_inf": conv_inf_now})

            if cfg.report_every and (step % int(cfg.report_every) == 0 or step == 1 or step == int(cfg.n_steps)):
                div = ns["face_divergence_gpu"](phi, geom)
                mass_inf_gpu = cp.max(cp.abs(div / geom.volume))
                last_mass_inf = float(mass_inf_gpu.get())
                umax = float(cp.max(cp.linalg.norm(U, axis=1)).get())
                dU = cp.max(cp.linalg.norm(U - U_prev, axis=1))
                Un = cp.maximum(cp.max(cp.linalg.norm(U, axis=1)), 1.0e-300)
                rel_change = float((dU / Un).get())
                rel_denom_ref = max(float(U0_norm_inf), float(Un.get()), 1.0e-300)
                rel_change_ref = float(dU.get()) / rel_denom_ref
                mom = ns["steady_momentum_residual_gpu"](U, p, phi, geom, cfg)
                mom_inf = float(cp.max(cp.linalg.norm(mom, axis=1)).get())
                print(
                    f"[flow] {run_label} step {step:5d}/{cfg.n_steps}: "
                    f"projection_interval={interval}, projections={projection_count}, "
                    f"mass_inf/V={last_mass_inf:.3e}, umax={umax:.6e}, "
                    f"rel_dU={rel_change:.3e}, rel_dU_ref={rel_change_ref:.3e}, "
                    f"mom_inf={mom_inf:.3e}, CFL={cfl_now:.3e}, E={energy_now:.3e}"
                )
                U_prev = U.copy()
                if (
                    bool(getattr(cfg, "stop_on_steady", True))
                    and step >= int(cfg.steady_min_steps)
                    and rel_change < float(cfg.steady_tol)
                    and mom_inf < max(10.0 * float(cfg.steady_tol), 1.0e-10)
                ):
                    steady_converged = True
                    print(f"[flow] steady convergence reached at step {step}: rel_dU={rel_change:.3e}, mom_inf={mom_inf:.3e}")
                    break

        elapsed = time.perf_counter() - t0
        div = ns["face_divergence_gpu"](phi, geom)
        mass_inf = float(cp.max(cp.abs(div / geom.volume)).get())
        mass_l2 = float(cp.sqrt(cp.mean((div / geom.volume) ** 2)).get())
        mean_U = cp.sum(U * geom.volume[:, None], axis=0) / cp.sum(geom.volume)
        mean_U_host = [float(v) for v in mean_U.get()]
        fx = float(cfg.body_force[0])
        k_eff_x = float((cfg.nu * mean_U[0] / fx).get()) if abs(fx) > 0 else float("nan")
        mom = ns["steady_momentum_residual_gpu"](U, p, phi, geom, cfg)
        mom_inf = float(cp.max(cp.linalg.norm(mom, axis=1)).get())
        energy_final = float(ns["kinetic_energy_gpu"](U, geom).get())
        return dict(
            U=U,
            p=p,
            phi=phi,
            div=div,
            elapsed_s=float(elapsed),
            mass_inf_per_volume=float(mass_inf),
            mass_l2_per_volume=float(mass_l2),
            mean_U_x=mean_U_host[0],
            mean_U_y=mean_U_host[1],
            mean_U_z=mean_U_host[2],
            K_eff_x=float(k_eff_x),
            n_cells=n,
            n_faces=int(geom.owner.size),
            last_mass_inf=last_mass_inf if last_mass_inf is not None else mass_inf,
            steady_converged=bool(steady_converged),
            final_rel_dU=float(rel_change),
            final_rel_dU_ref=float(rel_change_ref),
            steady_momentum_inf=float(mom_inf),
            steps_completed=int(steps_completed),
            failure_step=int(failure_step),
            solver_failure=str(solver_failure),
            implicit_diffusion=bool(cfg.implicit_diffusion),
            enable_convection=bool(cfg.enable_convection),
            energy_initial=float(energy_initial),
            energy_final=float(energy_final),
            energy_max=float(energy_max),
            max_cfl=float(max_cfl),
            max_convection_inf=float(max_convection_inf),
            max_diffusion_number=float(max_diffusion_number),
            nan_or_inf_detected=bool(nan_or_inf_detected),
            run_label=str(run_label),
            history=history,
            projection_interval=int(interval),
            projection_count=int(projection_count),
        )

    run_with_interval._pvfv_projection_interval_solver = True  # type: ignore[attr-defined]
    ns["run_velocity_pressure_projection_gpu"] = run_with_interval
    print("[flow] projection-interval solver mode installed")


def install_gpu_face_connected_split(ns: dict[str, Any], *, verify: bool = False) -> None:
    """Replace CPU 3D face-connected relabeling with a GPU union-find equivalent."""
    cp = ns["cp"]
    original = ns.get("pvfv_face_connected_reindex_cpu")
    if original is None:
        raise RuntimeError("pvfv_face_connected_reindex_cpu is not available")

    init_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void pvfv_face_split_init(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ labels,
        int* __restrict__ parent,
        const int nvox
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;
        parent[idx] = (mask[idx] && labels[idx] >= 0) ? idx : -1;
    }
    ''', "pvfv_face_split_init")

    union_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    int pvfv_find_root(int* parent, int x) {
        int p = parent[x];
        int guard = 0;
        while (p >= 0 && p != x && guard < 4096) {
            x = p;
            p = parent[x];
            ++guard;
        }
        return p;
    }

    extern "C" __device__ __forceinline__
    void pvfv_try_union(
        int* parent,
        const int* labels,
        int a,
        int b,
        int* changed
    ) {
        if (a == b) return;
        if (parent[a] < 0 || parent[b] < 0) return;
        if (labels[a] != labels[b]) return;

        int guard = 0;
        while (guard < 64) {
            int ra = pvfv_find_root(parent, a);
            int rb = pvfv_find_root(parent, b);
            if (ra < 0 || rb < 0 || ra == rb) return;
            int hi = (ra > rb) ? ra : rb;
            int lo = (ra > rb) ? rb : ra;
            int old = atomicMin(&parent[hi], lo);
            if (old == hi || old == lo) {
                atomicExch(changed, 1);
                return;
            }
            atomicExch(changed, 1);
            ++guard;
        }
    }

    extern "C" __global__
    void pvfv_face_split_union6(
        int* __restrict__ parent,
        const int* __restrict__ labels,
        int* __restrict__ changed,
        const int D,
        const int H,
        const int W,
        const int nvox
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;
        if (parent[idx] < 0) return;

        const int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        if (z + 1 < D) pvfv_try_union(parent, labels, idx, idx + HW, changed);
        if (y + 1 < H) pvfv_try_union(parent, labels, idx, idx + W, changed);
        if (W > 1) {
            int nx_idx = (x + 1 < W) ? (idx + 1) : (idx - (W - 1));
            pvfv_try_union(parent, labels, idx, nx_idx, changed);
        }
    }
    ''', "pvfv_face_split_union6")

    compress_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    int pvfv_find_root_compress(int* parent, int x) {
        int p = parent[x];
        int guard = 0;
        while (p >= 0 && p != x && guard < 4096) {
            x = p;
            p = parent[x];
            ++guard;
        }
        return p;
    }

    extern "C" __global__
    void pvfv_face_split_compress(
        int* __restrict__ parent,
        const int nvox
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;
        if (parent[idx] < 0) return;
        int r = pvfv_find_root_compress(parent, idx);
        parent[idx] = r;
    }
    ''', "pvfv_face_split_compress")

    def gpu_face_connected_reindex(mask_gpu, labels_gpu):
        mask_u8 = cp.asarray(mask_gpu, dtype=cp.uint8)
        labels = cp.ascontiguousarray(cp.asarray(labels_gpu, dtype=cp.int32))
        D, H, W = [int(v) for v in labels.shape]
        nvox = int(D * H * W)
        mask_flat = cp.ascontiguousarray(mask_u8.ravel())
        labels_flat = cp.ascontiguousarray(labels.ravel())

        threads = 256
        blocks = (nvox + threads - 1) // threads
        parent = cp.empty(nvox, dtype=cp.int32)
        changed = cp.zeros(1, dtype=cp.int32)

        init_kernel((blocks,), (threads,), (mask_flat, labels_flat, parent, np.int32(nvox)))

        max_iters_env = os.environ.get("PVFV_GPU_FACE_SPLIT_MAX_ITERS", "")
        max_iters = int(max_iters_env) if max_iters_env else max(16, 2 * max(D, H, W))
        used_iters = 0
        for it in range(max_iters):
            changed.fill(0)
            union_kernel(
                (blocks,), (threads,),
                (parent, labels_flat, changed, np.int32(D), np.int32(H), np.int32(W), np.int32(nvox)),
            )
            compress_kernel((blocks,), (threads,), (parent, np.int32(nvox)))
            used_iters = it + 1
            if int(changed.get()[0]) == 0:
                break
        else:
            if str(os.environ.get("PVFV_GPU_FACE_SPLIT_FALLBACK", "1")).lower().strip() not in {"0", "false", "no", "off"}:
                return original(mask_gpu, labels_gpu)

        valid_idx = cp.where(parent >= 0)[0]
        n_valid = int(valid_idx.size)
        if n_valid == 0:
            info = {
                "n_original_labels": 0,
                "n_cv": 0,
                "n_split_extra": 0,
                "n_face_disconnected_labels": 0,
                "face_connected_fraction": 1.0,
            }
            ns["_pvfv_last_gpu_face_split"] = {"used": True, "iters": int(used_iters), "n_valid": 0}
            return cp.full(labels.shape, -1, dtype=cp.int32), info

        roots_valid = parent[valid_idx]
        unique_roots, inverse = cp.unique(roots_valid, return_inverse=True)
        new_flat = cp.full(nvox, -1, dtype=cp.int32)
        new_flat[valid_idx] = inverse.astype(cp.int32, copy=False)

        comp_labels = labels_flat[unique_roots]
        orig_labels, comp_counts = cp.unique(comp_labels, return_counts=True)
        comp_count = int(unique_roots.size)
        n_orig = int(orig_labels.size)
        n_disconnected = int(cp.count_nonzero(comp_counts > 1).get())
        info = {
            "n_original_labels": n_orig,
            "n_cv": comp_count,
            "n_split_extra": int(comp_count - n_orig),
            "n_face_disconnected_labels": n_disconnected,
            "face_connected_fraction": float((n_orig - n_disconnected) / max(n_orig, 1)),
        }
        new_labels = new_flat.reshape(labels.shape)
        ns["_pvfv_last_gpu_face_split"] = {
            "used": True,
            "iters": int(used_iters),
            "n_valid": int(n_valid),
            "n_cv": int(comp_count),
        }

        if verify:
            cpu_labels, cpu_info = original(mask_gpu, labels_gpu)
            same_labels = bool(cp.all(new_labels == cpu_labels).get())
            same_info = all(
                info.get(k) == cpu_info.get(k)
                for k in ("n_original_labels", "n_cv", "n_split_extra", "n_face_disconnected_labels")
            ) and abs(float(info["face_connected_fraction"]) - float(cpu_info["face_connected_fraction"])) < 1.0e-15
            if not (same_labels and same_info):
                mismatch = int(cp.count_nonzero(new_labels != cpu_labels).get())
                raise RuntimeError(
                    f"GPU face split verification failed: mismatched_voxels={mismatch}, "
                    f"gpu_info={info}, cpu_info={cpu_info}"
                )
            ns["_pvfv_last_gpu_face_split"]["verified"] = True
        return new_labels, info

    gpu_face_connected_reindex._pvfv_gpu_face_split = True  # type: ignore[attr-defined]
    ns["pvfv_face_connected_reindex_cpu"] = gpu_face_connected_reindex
    try:
        warm_mask = cp.ones((1, 1, 2), dtype=cp.uint8)
        warm_labels = cp.zeros((1, 1, 2), dtype=cp.int32)
        gpu_face_connected_reindex(warm_mask, warm_labels)
        cp.cuda.Stream.null.synchronize()
        ns["_pvfv_last_gpu_face_split"] = {"used": False}
    except Exception as exc:
        print(f"[flow] GPU face split warm-up skipped: {exc}")
    print(f"[flow] GPU 3D face-connected split installed; verify={bool(verify)}")


def install_cuda_geometry_builders(ns: dict[str, Any]) -> None:
    """Install RawKernel front-ends for Voronoi-to-FV geometry construction."""
    cp = ns["cp"]
    aggregate_faces_by_key = ns.get("_aggregate_faces_by_key")
    cfg_cls = ns.get("PB615Config")
    if aggregate_faces_by_key is None or cfg_cls is None:
        raise RuntimeError("geometry dependencies are not available in the notebook namespace")

    cell_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void pvfv_cell_moments_kernel(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ labels,
        double* __restrict__ count,
        double* __restrict__ sx,
        double* __restrict__ sy,
        double* __restrict__ sz,
        const int D,
        const int H,
        const int W,
        const int n_cells,
        const double h
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;
        if (!mask[idx]) return;
        int lab = labels[idx];
        if (lab < 0 || lab >= n_cells) return;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;

        atomicAdd(&count[lab], 1.0);
        atomicAdd(&sx[lab], ((double)x + 0.5) * h);
        atomicAdd(&sy[lab], ((double)y + 0.5) * h);
        atomicAdd(&sz[lab], ((double)z + 0.5) * h);
    }
    ''', "pvfv_cell_moments_kernel")

    wall_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    void pvfv_add_wall(
        int lab,
        double fcx,
        double fcy,
        double fcz,
        double nx,
        double ny,
        double nz,
        const double* __restrict__ centroid,
        double* __restrict__ twall,
        double area0,
        double floor_delta
    ) {
        double cx = centroid[3 * lab + 0];
        double cy = centroid[3 * lab + 1];
        double cz = centroid[3 * lab + 2];
        double delta = (fcx - cx) * nx + (fcy - cy) * ny + (fcz - cz) * nz;
        if (delta < floor_delta) delta = floor_delta;
        atomicAdd(&twall[lab], area0 / delta);
    }

    extern "C" __global__
    void pvfv_wall_moments_kernel(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ labels,
        const double* __restrict__ centroid,
        double* __restrict__ twall,
        const int D,
        const int H,
        const int W,
        const int n_cells,
        const int periodic_x,
        const double h,
        const double wall_floor
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (idx >= nvox) return;
        if (!mask[idx]) return;
        int lab = labels[idx];
        if (lab < 0 || lab >= n_cells) return;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;
        double area0 = h * h;
        double floor_delta = wall_floor * h;

        double xc = ((double)x + 0.5) * h;
        double yc = ((double)y + 0.5) * h;
        double zc = ((double)z + 0.5) * h;

        if (!periodic_x && x == 0) {
            pvfv_add_wall(lab, 0.0, yc, zc, -1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (!periodic_x && x + 1 == W) {
            pvfv_add_wall(lab, (double)W * h, yc, zc, 1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (x > 0 && !mask[idx - 1]) {
            pvfv_add_wall(lab, (double)x * h, yc, zc, -1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (x + 1 < W && !mask[idx + 1]) {
            pvfv_add_wall(lab, ((double)x + 1.0) * h, yc, zc, 1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }

        if (y == 0) {
            pvfv_add_wall(lab, xc, 0.0, zc, 0.0, -1.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (y + 1 == H) {
            pvfv_add_wall(lab, xc, (double)H * h, zc, 0.0, 1.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (y > 0 && !mask[idx - W]) {
            pvfv_add_wall(lab, xc, (double)y * h, zc, 0.0, -1.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (y + 1 < H && !mask[idx + W]) {
            pvfv_add_wall(lab, xc, ((double)y + 1.0) * h, zc, 0.0, 1.0, 0.0, centroid, twall, area0, floor_delta);
        }

        if (z == 0) {
            pvfv_add_wall(lab, xc, yc, 0.0, 0.0, 0.0, -1.0, centroid, twall, area0, floor_delta);
        }
        if (z + 1 == D) {
            pvfv_add_wall(lab, xc, yc, (double)D * h, 0.0, 0.0, 1.0, centroid, twall, area0, floor_delta);
        }
        if (z > 0 && !mask[idx - HW]) {
            pvfv_add_wall(lab, xc, yc, (double)z * h, 0.0, 0.0, -1.0, centroid, twall, area0, floor_delta);
        }
        if (z + 1 < D && !mask[idx + HW]) {
            pvfv_add_wall(lab, xc, yc, ((double)z + 1.0) * h, 0.0, 0.0, 1.0, centroid, twall, area0, floor_delta);
        }
    }
    ''', "pvfv_wall_moments_kernel")

    face_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    void pvfv_emit_facelet(
        int lab_i,
        int lab_j,
        int n_cells,
        int axis,
        double fcx,
        double fcy,
        double fcz,
        double area0,
        long long* __restrict__ keys,
        double* __restrict__ ax,
        double* __restrict__ ay,
        double* __restrict__ az,
        double* __restrict__ xc,
        double* __restrict__ yc,
        double* __restrict__ zc,
        double* __restrict__ area,
        int* __restrict__ counter
    ) {
        if (lab_i < 0 || lab_j < 0 || lab_i == lab_j) return;
        int lo = lab_i < lab_j ? lab_i : lab_j;
        int hi = lab_i < lab_j ? lab_j : lab_i;
        double sign = (lab_i == lo) ? 1.0 : -1.0;
        int pos = atomicAdd(counter, 1);
        keys[pos] = ((long long)lo) * ((long long)n_cells) + (long long)hi;
        ax[pos] = (axis == 0) ? sign * area0 : 0.0;
        ay[pos] = (axis == 1) ? sign * area0 : 0.0;
        az[pos] = (axis == 2) ? sign * area0 : 0.0;
        xc[pos] = fcx;
        yc[pos] = fcy;
        zc[pos] = fcz;
        area[pos] = area0;
    }

    extern "C" __global__
    void pvfv_positive_facelets_emit_kernel(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ labels,
        long long* __restrict__ keys,
        double* __restrict__ ax,
        double* __restrict__ ay,
        double* __restrict__ az,
        double* __restrict__ xc,
        double* __restrict__ yc,
        double* __restrict__ zc,
        double* __restrict__ area,
        int* __restrict__ counter,
        const int D,
        const int H,
        const int W,
        const int n_cells,
        const int periodic_x,
        const double h
    ) {
        int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int nvox = D * H * W;
        if (tid >= 3 * nvox) return;
        int axis = tid / nvox;
        int idx = tid - axis * nvox;
        if (!mask[idx]) return;
        int lab_i = labels[idx];
        if (lab_i < 0 || lab_i >= n_cells) return;

        int HW = H * W;
        int z = idx / HW;
        int rem = idx - z * HW;
        int y = rem / W;
        int x = rem - y * W;
        int nb = -1;
        double area0 = h * h;
        double fcx = ((double)x + 0.5) * h;
        double fcy = ((double)y + 0.5) * h;
        double fcz = ((double)z + 0.5) * h;

        if (axis == 0) {
            if (x + 1 < W) {
                nb = idx + 1;
            } else if (periodic_x) {
                nb = idx - (W - 1);
            } else {
                return;
            }
            fcx = ((double)x + 1.0) * h;
        } else if (axis == 1) {
            if (y + 1 >= H) return;
            nb = idx + W;
            fcy = ((double)y + 1.0) * h;
        } else {
            if (z + 1 >= D) return;
            nb = idx + HW;
            fcz = ((double)z + 1.0) * h;
        }

        if (nb < 0 || !mask[nb]) return;
        int lab_j = labels[nb];
        if (lab_j < 0 || lab_j >= n_cells) return;
        pvfv_emit_facelet(
            lab_i, lab_j, n_cells, axis, fcx, fcy, fcz, area0,
            keys, ax, ay, az, xc, yc, zc, area, counter
        );
    }
    ''', "pvfv_positive_facelets_emit_kernel")

    def _grid_1d(n: int, threads: int = 256) -> tuple[tuple[int], tuple[int]]:
        return ((max(1, int((int(n) + threads - 1) // threads)),), (threads,))

    def build_cells_cuda(mask, labels, n_cells: int, cfg):
        h = float(cfg.voxel_size)
        labels_i = cp.ascontiguousarray(cp.asarray(labels, dtype=cp.int32))
        mask_u8 = cp.ascontiguousarray(cp.asarray(mask, dtype=cp.uint8))
        D, H, W = [int(v) for v in labels_i.shape]
        n = int(n_cells)
        nvox = int(D * H * W)
        count = cp.zeros(n, dtype=cp.float64)
        sx = cp.zeros(n, dtype=cp.float64)
        sy = cp.zeros(n, dtype=cp.float64)
        sz = cp.zeros(n, dtype=cp.float64)
        cell_kernel(
            *_grid_1d(nvox),
            (mask_u8.ravel(), labels_i.ravel(), count, sx, sy, sz,
             np.int32(D), np.int32(H), np.int32(W), np.int32(n), np.float64(h)),
        )
        if str(os.environ.get("PVFV_GEOM_VALIDATE_EMPTY_CELLS", "0")).lower().strip() in {"1", "true", "yes", "on"}:
            if bool(cp.any(count <= 0).get()):
                raise RuntimeError("At least one seed has no assigned voxel; remove empty seeds or rebuild labels.")
        safe_count = cp.maximum(count, 1.0e-300)
        volume = count * (h ** 3)
        centroid = cp.stack([sx / safe_count, sy / safe_count, sz / safe_count], axis=1)
        return volume, centroid

    def build_wall_moments_cuda(mask, labels, centroid, n_cells: int, cfg):
        h = float(cfg.voxel_size)
        labels_i = cp.ascontiguousarray(cp.asarray(labels, dtype=cp.int32))
        mask_u8 = cp.ascontiguousarray(cp.asarray(mask, dtype=cp.uint8))
        centroid_f = cp.ascontiguousarray(cp.asarray(centroid, dtype=cp.float64))
        D, H, W = [int(v) for v in labels_i.shape]
        n = int(n_cells)
        nvox = int(D * H * W)
        twall = cp.zeros(n, dtype=cp.float64)
        wall_kernel(
            *_grid_1d(nvox),
            (mask_u8.ravel(), labels_i.ravel(), centroid_f.reshape(-1), twall,
             np.int32(D), np.int32(H), np.int32(W), np.int32(n),
             np.int32(1 if bool(cfg.periodic_x) else 0), np.float64(h),
             np.float64(float(cfg.wall_distance_floor))),
        )
        return twall

    def build_positive_area_faces_cuda(mask, labels, n_cells: int, cfg):
        h = float(cfg.voxel_size)
        labels_i = cp.ascontiguousarray(cp.asarray(labels, dtype=cp.int32))
        mask_u8 = cp.ascontiguousarray(cp.asarray(mask, dtype=cp.uint8))
        D, H, W = [int(v) for v in labels_i.shape]
        n = int(n_cells)
        nvox = int(D * H * W)
        nmax = max(1, 3 * nvox)
        keys = cp.empty(nmax, dtype=cp.int64)
        ax = cp.empty(nmax, dtype=cp.float64)
        ay = cp.empty(nmax, dtype=cp.float64)
        az = cp.empty(nmax, dtype=cp.float64)
        xc = cp.empty(nmax, dtype=cp.float64)
        yc = cp.empty(nmax, dtype=cp.float64)
        zc = cp.empty(nmax, dtype=cp.float64)
        area = cp.empty(nmax, dtype=cp.float64)
        counter = cp.zeros(1, dtype=cp.int32)
        face_kernel(
            *_grid_1d(3 * nvox),
            (mask_u8.ravel(), labels_i.ravel(), keys, ax, ay, az, xc, yc, zc,
             area, counter, np.int32(D), np.int32(H), np.int32(W), np.int32(n),
             np.int32(1 if bool(cfg.periodic_x) else 0), np.float64(h)),
        )
        n_facelets = int(counter.get()[0])
        if n_facelets <= 0:
            raise RuntimeError("No positive-area intercell faces were found; increase seed count or check mask.")
        return aggregate_faces_by_key(
            keys[:n_facelets], ax[:n_facelets], ay[:n_facelets], az[:n_facelets],
            xc[:n_facelets], yc[:n_facelets], zc[:n_facelets], area[:n_facelets], n,
        )

    ns["build_cells_gpu"] = build_cells_cuda
    ns["build_wall_moments_gpu"] = build_wall_moments_cuda
    ns["build_positive_area_faces_gpu"] = build_positive_area_faces_cuda
    ns["_pvfv_cuda_geometry_builders"] = {"used": True, "mode": "rawkernel_cell_wall_facelets"}

    try:
        cfg = cfg_cls()
        warm_mask = cp.ones((2, 2, 2), dtype=cp.uint8)
        warm_labels = cp.zeros((2, 2, 2), dtype=cp.int32)
        warm_labels[:, :, 1] = 1
        vol, cen = build_cells_cuda(warm_mask, warm_labels, 2, cfg)
        _ = build_positive_area_faces_cuda(warm_mask, warm_labels, 2, cfg)
        _ = build_wall_moments_cuda(warm_mask, warm_labels, cen, 2, cfg)
        cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        print(f"[flow] CUDA geometry builder warm-up skipped: {exc}")
    print("[flow] CUDA RawKernel geometry builders installed")


def install_label_backend(ns: dict[str, Any], notebook_dir: Path) -> str:
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    backend = importlib.import_module("exact_frontier_backend")

    cp = ns["cp"]
    time_mod = ns.get("time", time)
    original = ns["pvfv_build_geometry_from_seed_flat_timed"]
    exact_frontier_names = {"exact_frontier_gpu", "exact-frontier-gpu", "frontier_gpu", "frontier-dijkstra-gpu"}
    installed = str(os.environ.get("PVFV_LABEL_BACKEND", "exact_frontier_gpu")).lower().strip()
    if installed != "exact_geodesic" and installed not in exact_frontier_names:
        raise ValueError(f"PVFV_LABEL_BACKEND={installed!r}; use exact_frontier_gpu or exact_geodesic")
    kernel_bundle: dict[str, Any] = {}

    try:
        kernel_bundle["exact_frontier_kernel"] = backend.build_exact_frontier_bfs6_kernel()
        kernel_bundle["exact_frontier_init_kernel"] = backend.build_exact_frontier_bfs6_init_kernel()
        warm_mask = cp.ones((2, 2, 2), dtype=cp.uint8)
        warm_seeds = cp.asarray([[0, 0, 0]], dtype=cp.int32)
        backend.exact_frontier_dijkstra_gpu_6(
            warm_mask,
            warm_seeds,
            kernel=kernel_bundle["exact_frontier_kernel"],
            init_kernel=kernel_bundle["exact_frontier_init_kernel"],
            validate=False,
            collect_stats=False,
            return_float64=False,
        )
        cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        print(f"[flow] exact frontier kernel prebuild skipped: {exc}")
        kernel_bundle = {}

    def wrapped(mask, seed_flat, cfg, *, split_face_components=True,
                label_mode="exact_geodesic", seed_spec="custom"):
        requested = str(os.environ.get("PVFV_LABEL_BACKEND", "exact_frontier_gpu")).lower().strip()
        if requested == "exact_geodesic":
            return original(
                mask,
                seed_flat,
                cfg,
                split_face_components=split_face_components,
                label_mode=label_mode,
                seed_spec=seed_spec,
            )
        if requested not in exact_frontier_names:
            raise ValueError(f"PVFV_LABEL_BACKEND={requested!r}; use exact_frontier_gpu or exact_geodesic")

        t0 = time_mod.perf_counter()
        seed_flat = cp.asarray(seed_flat, dtype=cp.int64)
        cp.cuda.Stream.null.synchronize()
        t_seed = time_mod.perf_counter() - t0

        d, h, w = [int(v) for v in mask.shape]
        mask_u8 = mask.astype(cp.uint8, copy=False)
        z = seed_flat // cp.int64(h * w)
        rem = seed_flat - z * cp.int64(h * w)
        y = rem // cp.int64(w)
        x = rem - y * cp.int64(w)
        seeds_zyx = cp.stack([z, y, x], axis=1).astype(cp.int32, copy=False)

        t1 = time_mod.perf_counter()
        labels, dist = backend.exact_frontier_dijkstra_gpu_6(
            mask_u8,
            seeds_zyx,
            kernel=kernel_bundle.get("exact_frontier_kernel"),
            init_kernel=kernel_bundle.get("exact_frontier_init_kernel"),
            validate=False,
            collect_stats=False,
            return_float64=False,
        )
        labels = labels.astype(cp.int32, copy=False)
        dist = dist.astype(cp.float32, copy=False)
        cp.cuda.Stream.null.synchronize()
        t_label = time_mod.perf_counter() - t1

        n_seed = int(seed_flat.size)
        t_split0 = time_mod.perf_counter()
        ns["_pvfv_last_gpu_face_split"] = {"used": False}
        split_info = {
            "n_original_labels": n_seed,
            "n_cv": n_seed,
            "n_split_extra": 0,
            "n_face_disconnected_labels": 0,
            "face_connected_fraction": 1.0,
        }
        if split_face_components:
            labels, split_info = ns["pvfv_face_connected_reindex_cpu"](mask, labels)
        cp.cuda.Stream.null.synchronize()
        t_split = time_mod.perf_counter() - t_split0
        gpu_split_meta = ns.get("_pvfv_last_gpu_face_split", {})
        if not isinstance(gpu_split_meta, dict):
            gpu_split_meta = {}

        n0a = ns["pvfv_zero_area_candidate_count_cpu"](mask, labels)
        t2 = time_mod.perf_counter()
        geom = ns["pvfv_build_geometry_from_labels_ncells_gpu"](
            mask, labels, dist, int(split_info["n_cv"]), cfg
        )
        cp.cuda.Stream.null.synchronize()
        t_mom = time_mod.perf_counter() - t2

        n_fl = int(mask.sum().get())
        meta = {
            "S": n_seed,
            "N_cv": int(geom.n_cells),
            "N_fl": n_fl,
            "C_comp": float(n_fl / max(int(geom.n_cells), 1)),
            "t_seed_s": float(t_seed),
            "t_label_s": float(t_label),
            "t_split_s": float(t_split),
            "t_part_s": float(t_seed + t_label + t_split),
            "t_mom_s": float(t_mom),
            "N_split": int(split_info["n_split_extra"]),
            "N_face_disconnected_labels": int(split_info["n_face_disconnected_labels"]),
            "face_connected_fraction": float(split_info["face_connected_fraction"]),
            "N_0A": int(n0a),
            "label_mode": "exact_frontier_gpu",
            "distance_connectivity": int(cfg.distance_connectivity),
            "seed_spec": str(seed_spec),
            "gpu_face_split_used": bool(gpu_split_meta.get("used", False)),
            "gpu_face_split_iters": int(gpu_split_meta.get("iters", 0) or 0),
            "gpu_face_split_verified": bool(gpu_split_meta.get("verified", False)),
            "exact_frontier_tinit_s": float(getattr(backend, "EXACT_FRONTIER_LAST_TINIT_WALL", 0.0)),
            "exact_frontier_titer_s": float(getattr(backend, "EXACT_FRONTIER_LAST_TITER_WALL", 0.0)),
            "exact_frontier_tpred_s": float(getattr(backend, "EXACT_FRONTIER_LAST_WALL_TIME", 0.0)),
            "exact_frontier_iters": int(getattr(backend, "EXACT_FRONTIER_LAST_ITERS", 0)),
            "exact_frontier_max_frontier": int(getattr(backend, "EXACT_FRONTIER_LAST_MAX_FRONTIER", 0)),
            "exact_frontier_max_dist": int(getattr(backend, "EXACT_FRONTIER_LAST_MAX_DIST", 0)),
        }
        ns["_pvfv_last_label_meta"] = dict(meta)
        progress_env = str(os.environ.get("PVFV_FLOW_PROGRESS", "1")).lower().strip()
        if progress_env not in {"0", "false", "no", "off"}:
            progress_label = str(getattr(cfg, "pvfv_progress_label", "partition"))
            print(
                f"[flow][exact_frontier_gpu] {progress_label}: "
                f"S={n_seed} N_fl={n_fl:,} N_cv={int(geom.n_cells)} "
                f"seed={1000.0 * t_seed:.2f}ms "
                f"label={1000.0 * t_label:.2f}ms "
                f"split={1000.0 * t_split:.2f}ms "
                f"geom={1000.0 * t_mom:.2f}ms "
                f"gpu_split_iters={int(meta['gpu_face_split_iters'])}",
                flush=True,
            )
        return geom, meta

    ns["pvfv_build_geometry_from_seed_flat_timed"] = wrapped
    print(f"[flow] label backend: {installed} installed for seeded coarse partitions")
    return "exact_geodesic" if installed == "exact_geodesic" else "exact_frontier_gpu"


def install_geodesic_face_operator(ns: dict[str, Any], *, mode: str) -> None:
    mode_key = str(mode or "geodesic_face").lower().strip()
    if mode_key in {"", "euclidean", "default", "off", "none"}:
        ns["_pvfv_face_operator_mode"] = "euclidean"
        print("[flow] face operator: Euclidean/default geometry")
        return
    if mode_key not in {
        "geodesic_face",
        "graph_geodesic_face",
        "geodesic_all",
        "geodesic_weights",
        "geodesic_weights_only",
        "graph_geodesic_weights",
    }:
        raise ValueError(f"Unknown face operator mode: {mode}")

    cp = ns["cp"]
    original_build = ns["pvfv_build_geometry_from_seed_flat_timed"]
    if getattr(original_build, "_pvfv_geodesic_face_operator", False):
        ns["_pvfv_face_operator_mode"] = mode_key
        return

    def wrapped_build(*args, **kwargs):
        geom, meta = original_build(*args, **kwargs)
        cfg = kwargs.get("cfg")
        if cfg is None and len(args) >= 3:
            cfg = args[2]
        if cfg is None:
            raise RuntimeError("Cannot locate cfg while applying geodesic face operator")
        physical_dvec = geom.dvec.copy()
        op_t0 = time.perf_counter()
        metric = build_geodesic_face_metric(ns, geom, cfg, operator_variant=mode_key)
        geom = clone_geometry(
            ns,
            geom,
            tproj=metric["tproj"],
            w_owner=metric["w_owner"],
            w_neigh=metric["w_neigh"],
            dvec=metric["dvec"],
        )
        cp.cuda.Stream.null.synchronize()
        op_meta = dict(metric["meta"])
        op_meta.update(
            {
                "face_operator": op_meta.get("face_operator", mode_key),
                "operator_closure": "geodesic_face_operator",
                "operator_closure_s": float(time.perf_counter() - op_t0),
                "darcy_readout_length": "physical_axis_flux_readout",
            }
        )
        meta.update(op_meta)
        ns["_pvfv_last_physical_dvec_readout"] = physical_dvec
        ns["_pvfv_last_face_operator_meta"] = dict(op_meta)
        return geom, meta

    wrapped_build._pvfv_geodesic_face_operator = True  # type: ignore[attr-defined]
    wrapped_build.__name__ = getattr(original_build, "__name__", "pvfv_build_geometry_from_seed_flat_timed")
    wrapped_build.__doc__ = getattr(original_build, "__doc__", None)
    ns["pvfv_build_geometry_from_seed_flat_timed"] = wrapped_build

    original_run = ns["pvfv_run_flow_case_seeded"]
    if not getattr(original_run, "_pvfv_physical_flux_readout", False):

        def wrapped_run(
            case_name,
            mask,
            seed_flat,
            seed_spec,
            cfg,
            ref_geom,
            ref_result,
            **kwargs,
        ):
            export_data = bool(kwargs.get("export_data", True))
            out = kwargs.get("out")
            panel_hint = str(kwargs.get("panel_hint", "seeded"))
            call_kwargs = dict(kwargs)
            call_kwargs["export_data"] = False
            row, geom, res = original_run(
                case_name,
                mask,
                seed_flat,
                seed_spec,
                cfg,
                ref_geom,
                ref_result,
                **call_kwargs,
            )
            physical_dvec = ns.get("_pvfv_last_physical_dvec_readout")
            op_meta = ns.get("_pvfv_last_face_operator_meta", {})
            if physical_dvec is not None and "phi" in res:
                mean_flux_u = physical_flux_readout(ns, geom, res["phi"], physical_dvec)
                fx = float(getattr(cfg, "body_force", [0.0])[0])
                if abs(fx) > 0.0:
                    k_flux = float((float(cfg.nu) * mean_flux_u[0] / fx).get())
                    row["K_eff_x_solver_reported"] = float(res.get("K_eff_x", float("nan")))
                    res["K_eff_x_solver_reported"] = float(res.get("K_eff_x", float("nan")))
                    row["K_eff_x"] = k_flux
                    res["K_eff_x"] = k_flux
                    ref_k = float(ref_result.get("K_eff_x", float("nan")))
                    row["e_K"] = abs(k_flux - ref_k) / max(abs(ref_k), 1.0e-300)
                    row["mean_U_x_flux_readout"] = float(mean_flux_u[0].get())
                    row["mean_U_y_flux_readout"] = float(mean_flux_u[1].get())
                    row["mean_U_z_flux_readout"] = float(mean_flux_u[2].get())
            if isinstance(op_meta, dict):
                row.update(op_meta)
            if "pvfv_quality_flags" in ns:
                row = ns["pvfv_quality_flags"](row)
            if export_data and out is not None:
                pseudo_stride = tuple(seed_spec.get("stride_zyx", (0, 0, 0))) if seed_spec.get("stride_zyx", None) else (0, 0, 0)
                products = ns["pvfv_save_flow_products"](
                    Path(out),
                    case_name,
                    tuple(pseudo_stride),
                    str(kwargs.get("run_tag", "seeded")),
                    ref_geom,
                    ref_result,
                    geom,
                    res,
                    row,
                    panel_hint=panel_hint,
                )
                row.update(products)
                row["data_manifest"] = str(Path(products["run_arrays_npz"]).parent / "run_data_manifest.json")
            return row, geom, res

        wrapped_run._pvfv_physical_flux_readout = True  # type: ignore[attr-defined]
        wrapped_run.__name__ = getattr(original_run, "__name__", "pvfv_run_flow_case_seeded")
        wrapped_run.__doc__ = getattr(original_run, "__doc__", None)
        ns["pvfv_run_flow_case_seeded"] = wrapped_run

    ns["_pvfv_face_operator_mode"] = mode_key
    print(f"[flow] face operator: {mode_key} with physical-axis Darcy readout")


def install_runtime_profile(ns: dict[str, Any], *, synchronize: bool = True) -> dict[str, dict[str, float]]:
    """Install lightweight timing wrappers around the main GPU pipeline stages."""
    cp = ns["cp"]
    profile: dict[str, dict[str, float]] = {}

    def record(name: str, elapsed: float) -> None:
        item = profile.setdefault(name, {"calls": 0.0, "seconds": 0.0})
        item["calls"] += 1.0
        item["seconds"] += float(elapsed)

    def wrap(name: str) -> None:
        original = ns.get(name)
        if original is None or getattr(original, "_pvfv_profiled", False):
            return

        def profiled(*args, **kwargs):
            if synchronize:
                cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if synchronize:
                    cp.cuda.Stream.null.synchronize()
                record(name, time.perf_counter() - t0)

        profiled._pvfv_profiled = True  # type: ignore[attr-defined]
        profiled.__name__ = getattr(original, "__name__", name)
        profiled.__doc__ = getattr(original, "__doc__", None)
        ns[name] = profiled

    for fname in [
        "pvfv_build_reference",
        "pvfv_run_flow_case_seeded",
        "pvfv_build_geometry_from_seed_flat_timed",
        "pvfv_build_geometry_from_labels_ncells_gpu",
        "pvfv_face_connected_reindex_cpu",
        "pvfv_zero_area_candidate_count_cpu",
        "run_velocity_pressure_projection_gpu",
        "implicit_momentum_predictor_gpu",
        "solve_pressure_correction_gpu",
        "lsq_gradient_gpu",
        "diffusion_term_gpu",
        "convection_term_gpu",
        "face_flux_from_velocity_gpu",
        "face_divergence_gpu",
        "correct_flux_gpu",
        "steady_momentum_residual_gpu",
        "coarsen_voxel_reference_to_coarse_gpu",
        "pvfv_flux_rel_error",
        "pvfv_save_flow_products",
    ]:
        wrap(fname)

    ns["_pvfv_runtime_profile"] = profile
    ns["_pvfv_runtime_profile_sync"] = bool(synchronize)
    print(f"[flow] runtime profile enabled; synchronize={bool(synchronize)}")
    return profile


def install_resolution_aware_wall_closure(ns: dict[str, Any], args: argparse.Namespace) -> None:
    """Optionally replace scalar wall-beta scaling by a size-aware closure."""
    mode = str(getattr(args, "wall_closure_mode", "fixed_beta") or "fixed_beta").lower().strip()
    if mode in {"", "fixed", "fixed_beta", "scalar", "scalar_beta"}:
        ns["_pvfv_wall_closure_mode"] = "fixed_beta"
        ns["_pvfv_wall_closure_params"] = {}
        return

    if mode not in {"global_comp_power", "cell_volume_power"}:
        raise ValueError(f"Unsupported wall closure mode: {mode}")
    if "_pb618_clone_geometry_with_scaled_twall" not in ns:
        raise RuntimeError("Notebook namespace does not expose wall scaling hook")

    cp = ns["cp"]
    Geometry = ns["GPUFVMGeometry"]
    ref_ccomp = max(float(getattr(args, "wall_ref_ccomp", 64.0) or 64.0), 1.0e-300)
    raw_length_exponent = getattr(args, "wall_length_exponent", None)
    if raw_length_exponent is None:
        comp_exponent = float(getattr(args, "wall_size_exponent", 0.0) or 0.0)
        length_exponent = 3.0 * comp_exponent
    else:
        length_exponent = float(raw_length_exponent)
        comp_exponent = length_exponent / 3.0
    scale_min = max(float(getattr(args, "wall_scale_min", 0.05) or 0.05), 0.0)
    scale_max = max(float(getattr(args, "wall_scale_max", 20.0) or 20.0), scale_min)

    def clone_with_resolution_aware_wall(geom, wall_beta: float):
        beta = float(wall_beta)
        n_cells = max(int(geom.n_cells), 1)
        n_fl = cp.maximum(cp.sum(geom.mask).astype(cp.float64), 1.0)
        total_volume = cp.maximum(cp.sum(geom.volume), 1.0e-300)
        voxel_volume = total_volume / n_fl
        ccomp = total_volume / (voxel_volume * float(n_cells))

        if mode == "global_comp_power":
            scale = cp.asarray((ccomp / ref_ccomp) ** comp_exponent, dtype=cp.float64)
        else:
            ref_volume = ref_ccomp * voxel_volume
            scale = cp.power(cp.maximum(geom.volume / cp.maximum(ref_volume, 1.0e-300), 1.0e-300), comp_exponent)
        scale = cp.clip(scale, scale_min, scale_max)
        twall = geom.twall * beta * scale

        try:
            scale_mean = float(cp.mean(scale).get())
            scale_min_obs = float(cp.min(scale).get())
            scale_max_obs = float(cp.max(scale).get())
            ccomp_obs = float(ccomp.get())
        except Exception:
            scale_mean = scale_min_obs = scale_max_obs = ccomp_obs = float("nan")
        ns["_pvfv_last_wall_closure_meta"] = {
            "wall_closure_mode": mode,
            "wall_ref_ccomp": float(ref_ccomp),
            "wall_size_exponent": float(comp_exponent),
            "wall_length_exponent": float(length_exponent),
            "wall_scale_min": float(scale_min),
            "wall_scale_max": float(scale_max),
            "wall_scale_mean": scale_mean,
            "wall_scale_min_observed": scale_min_obs,
            "wall_scale_max_observed": scale_max_obs,
            "wall_ccomp_observed": ccomp_obs,
        }

        return Geometry(
            mask=geom.mask,
            labels=geom.labels,
            dist=geom.dist,
            n_cells=geom.n_cells,
            volume=geom.volume,
            centroid=geom.centroid,
            owner=geom.owner,
            neigh=geom.neigh,
            area=geom.area,
            avec=geom.avec,
            face_centroid=geom.face_centroid,
            dvec=geom.dvec,
            tproj=geom.tproj,
            w_owner=geom.w_owner,
            w_neigh=geom.w_neigh,
            twall=twall,
            laplacian=geom.laplacian,
        )

    ns["_pb618_clone_geometry_with_scaled_twall"] = clone_with_resolution_aware_wall
    ns["_pvfv_wall_closure_mode"] = mode
    ns["_pvfv_wall_closure_params"] = {
        "wall_ref_ccomp": float(ref_ccomp),
        "wall_size_exponent": float(comp_exponent),
        "wall_length_exponent": float(length_exponent),
        "wall_scale_min": float(scale_min),
        "wall_scale_max": float(scale_max),
    }
    print(
        "[flow] resolution-aware wall closure installed: "
        f"mode={mode} eta_w={length_exponent:g} comp_exponent={comp_exponent:g} ref_ccomp={ref_ccomp:g} "
        f"clip=[{scale_min:g},{scale_max:g}]"
    )


def runtime_profile_summary(ns: dict[str, Any]) -> dict[str, Any]:
    profile = ns.get("_pvfv_runtime_profile")
    if not isinstance(profile, dict) or not profile:
        return {}
    rows = []
    total = 0.0
    for name, item in profile.items():
        calls = float(item.get("calls", 0.0))
        seconds = float(item.get("seconds", 0.0))
        total += seconds
        rows.append({
            "stage": str(name),
            "calls": int(calls),
            "seconds": seconds,
            "avg_seconds": seconds / max(calls, 1.0),
        })
    rows.sort(key=lambda row: float(row["seconds"]), reverse=True)
    return {
        "synchronized_timing": bool(ns.get("_pvfv_runtime_profile_sync", False)),
        "note": "Nested timings are intentionally inclusive and may double count child stages.",
        "inclusive_seconds_sum": total,
        "stages": rows,
    }


def build_case_specs(ns: dict[str, Any], profile: str, bentheimer_npz: Path, fibrous_npz: Path) -> list[dict[str, Any]]:
    plan = ns["pvfv_half_gfps_case_plan"](profile)
    shape = tuple(int(v) for v in plan["shape_zyx"])
    synthetic_strides = [tuple(int(x) for x in s) for s in plan["density_strides"]]
    proxy_strides = [(4, 8, 8), (2, 6, 6), (2, 4, 4)]
    return [
        {
            "case": "orthogonal_duct",
            "paper_case": "Orthogonal duct",
            "mask_kind": "procedural",
            "shape_zyx": shape,
            "strides": synthetic_strides,
            "main_text_case": True,
        },
        {
            "case": "skewed_duct",
            "paper_case": "Skewed duct",
            "mask_kind": "procedural",
            "shape_zyx": shape,
            "strides": synthetic_strides,
            "main_text_case": True,
        },
        {
            "case": "A_thin_wall",
            "paper_case": "Thin-wall synthetic",
            "mask_kind": "procedural",
            "shape_zyx": shape,
            "strides": synthetic_strides,
            "main_text_case": True,
        },
        {
            "case": "B_narrow_throat",
            "paper_case": "Narrow-throat synthetic",
            "mask_kind": "procedural",
            "shape_zyx": shape,
            "strides": synthetic_strides,
            "main_text_case": True,
        },
        {
            "case": "C_maze",
            "paper_case": "Maze synthetic",
            "mask_kind": "procedural",
            "shape_zyx": shape,
            "strides": synthetic_strides,
            "main_text_case": True,
        },
        {
            "case": "bentheimer_sandstone_crop",
            "paper_case": "Bentheimer segmented sandstone crop",
            "mask_kind": "npz_mask",
            "mask_npz": str(bentheimer_npz),
            "strides": proxy_strides,
            "main_text_case": True,
        },
        {
            "case": "fibrous_filter_proxy",
            "paper_case": "Fibrous filter proxy",
            "mask_kind": "npz_mask",
            "mask_npz": str(fibrous_npz),
            "strides": proxy_strides,
            "main_text_case": True,
        },
    ]


def filter_case_specs(specs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = list(specs)
    if getattr(args, "case_filter", ""):
        wanted = {
            item.strip()
            for item in str(args.case_filter).split(",")
            if item.strip()
        }
        out = [
            spec for spec in out
            if spec["case"] in wanted or spec["paper_case"] in wanted
        ]
    max_strides = int(getattr(args, "max_strides_per_case", 0) or 0)
    if max_strides > 0:
        clipped = []
        for spec in out:
            spec2 = dict(spec)
            spec2["strides"] = list(spec2["strides"])[:max_strides]
            clipped.append(spec2)
        out = clipped
    if getattr(args, "stride_indices", ""):
        indices = [
            int(item.strip())
            for item in str(args.stride_indices).split(",")
            if item.strip()
        ]
        selected = []
        for spec in out:
            spec2 = dict(spec)
            strides = list(spec2["strides"])
            spec2["strides"] = [strides[i] for i in indices if 0 <= i < len(strides)]
            selected.append(spec2)
        out = selected
    if not out:
        raise ValueError("Case filter removed all cases")
    if any(not spec.get("strides") for spec in out):
        raise ValueError("Stride filter removed all strides for at least one case")
    return out


def make_mask(ns: dict[str, Any], spec: dict[str, Any]):
    if spec["mask_kind"] == "procedural":
        return ns["pvfv_make_mask_gpu"](spec["case"], tuple(spec["shape_zyx"]))
    if spec["mask_kind"] == "npz_mask":
        return load_mask_npz(ns, Path(spec["mask_npz"]))
    raise ValueError(f"Unknown mask kind: {spec['mask_kind']}")


def summarize_case_specs(ns: dict[str, Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_specs = ns["pvfv_half_gfps_seed_specs_for_density"]
    cp = ns["cp"]
    for spec in specs:
        mask = make_mask(ns, spec)
        n_fl = int(mask.sum().get())
        rows.append({
            "case": spec["case"],
            "paper_case": spec["paper_case"],
            "mask_kind": spec["mask_kind"],
            "shape_zyx": "x".join(str(int(v)) for v in mask.shape),
            "N_fl": n_fl,
            "porosity": float(n_fl / int(np.prod(tuple(int(v) for v in mask.shape)))),
            "strides": ";".join(str(tuple(s)) for s in spec["strides"]),
            "seed_counts": ";".join(
                str(int(seed_specs(mask, tuple(s), ["half"])[0][1].size))
                for s in spec["strides"]
            ),
        })
        del mask
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def final_score(row: dict[str, Any]) -> float:
    def val(key: str, default: float = 0.0) -> float:
        try:
            x = float(row.get(key, default))
            return x if math.isfinite(x) else default
        except Exception:
            return default
    return val("e_K", 1.0e9) + 0.25 * val("e_phi", 0.0) + 0.1 * val("e_u", 0.0)


def choose_final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    final: list[dict[str, Any]] = []
    for case, group in by_case.items():
        valid = [
            r for r in group
            if str(r.get("valid_physical", "True")).lower() in {"true", "1"}
        ]
        if not valid:
            valid = group
        best = min(valid, key=final_score)
        final.append(best)
    order = {
        "orthogonal_duct": 0,
        "skewed_duct": 1,
        "A_thin_wall": 2,
        "B_narrow_throat": 3,
        "C_maze": 4,
        "bentheimer_sandstone_crop": 5,
        "fibrous_filter_proxy": 6,
    }
    return sorted(final, key=lambda r: order.get(str(r["case"]), 999))


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item) for item in parse_csv_values(text)]


def parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def token_float(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def token_text(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).strip()) or "blank"


def clone_cfg(ns: dict[str, Any], cfg, **updates):
    if "pvfv_clone_cfg" in ns:
        return ns["pvfv_clone_cfg"](cfg, **updates)
    data = {k: getattr(cfg, k) for k in ns["PB615Config"].__dataclass_fields__.keys()}
    data.update(updates)
    return ns["PB615Config"](**data)


def apply_common_cfg_overrides(cfg, args: argparse.Namespace, *, prefix: str):
    dt = getattr(args, f"{prefix}_dt", None)
    n_steps = getattr(args, f"{prefix}_n_steps", None)
    report_every = getattr(args, f"{prefix}_report_every", None)
    initial_velocity_mode = getattr(args, f"{prefix}_initial_velocity_mode", None)
    solver_formulation = getattr(
        args,
        f"{prefix}_solver_formulation",
        PRODUCTION_PRESSURE_CORRECTION_FORMULATION,
    )
    if dt is not None:
        cfg.dt = float(dt)
    if n_steps is not None:
        cfg.n_steps = int(n_steps)
    if report_every is not None:
        cfg.report_every = int(report_every)
    if initial_velocity_mode:
        cfg.initial_velocity_mode = str(initial_velocity_mode)
    setattr(cfg, "solver_formulation", str(solver_formulation))
    if bool(getattr(args, "disable_convection", False)):
        cfg.enable_convection = False
    if getattr(args, "body_force_x", None) is not None:
        bf = tuple(float(v) for v in cfg.body_force)
        cfg.body_force = (float(args.body_force_x), bf[1], bf[2])
    if getattr(args, "wall_distance_floor", None) is not None:
        cfg.wall_distance_floor = float(args.wall_distance_floor)
    if getattr(args, "pressure_gauge_eps", None) is not None:
        cfg.pressure_gauge_eps = float(args.pressure_gauge_eps)
    if getattr(args, "transmissibility_floor", None) is not None:
        cfg.transmissibility_floor = float(args.transmissibility_floor)
    return cfg


def flow_dt_values(cfg, args: argparse.Namespace) -> list[float]:
    vals = parse_csv_floats(getattr(args, "coarse_dt_values", ""))
    if vals:
        return vals
    if getattr(args, "coarse_dt", None) is not None:
        return [float(args.coarse_dt)]
    return [float(cfg.dt)]


def flow_n_step_values(cfg, args: argparse.Namespace) -> list[int]:
    vals = parse_csv_ints(getattr(args, "coarse_n_steps_values", ""))
    if vals:
        return vals
    if getattr(args, "coarse_n_steps", None) is not None:
        return [int(args.coarse_n_steps)]
    return [int(cfg.n_steps)]


def flow_initial_velocity_modes(cfg, args: argparse.Namespace) -> list[str]:
    vals = parse_csv_values(getattr(args, "coarse_initial_velocity_modes", ""))
    return vals if vals else [str(getattr(cfg, "initial_velocity_mode", "zero"))]


def flow_projection_intervals(cfg, args: argparse.Namespace) -> list[int]:
    vals = parse_csv_ints(getattr(args, "coarse_projection_interval_values", ""))
    if vals:
        return [max(1, int(v)) for v in vals]
    return [max(1, int(getattr(args, "coarse_projection_interval", getattr(cfg, "projection_interval", 1))))]


def flow_face_modes(args: argparse.Namespace) -> list[str]:
    vals = parse_csv_values(getattr(args, "face_modes", ""))
    return vals if vals else [str(args.face_mode)]


def run_flow(ns: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cp = ns["cp"]
    profile = str(args.profile).lower()
    out = Path(args.out_dir).resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ns["pvfv_ensure_dirs"](out)
    base_cfg = ns["pvfv_base_cfg"](out_dir=str(out), profile=profile)
    ref_cfg = apply_common_cfg_overrides(
        clone_cfg(ns, base_cfg, out_dir=str(out)),
        args,
        prefix="reference",
    )
    setattr(ref_cfg, "linear_solver_mode", str(getattr(args, "reference_linear_solver", "cg")))
    setattr(ref_cfg, "momentum_solver_mode", str(getattr(args, "reference_momentum_solver", "cg")))
    setattr(ref_cfg, "projection_interval", max(1, int(getattr(args, "reference_projection_interval", 1))))
    coarse_template_cfg = apply_common_cfg_overrides(
        clone_cfg(ns, base_cfg, out_dir=str(out)),
        args,
        prefix="coarse",
    )
    setattr(coarse_template_cfg, "pressure_gradient_weight", str(getattr(args, "pressure_gradient_weight", "tproj")))
    setattr(coarse_template_cfg, "linear_solver_mode", str(getattr(args, "coarse_linear_solver", "cg")))
    setattr(coarse_template_cfg, "momentum_solver_mode", str(getattr(args, "coarse_momentum_solver", "cg")))
    setattr(coarse_template_cfg, "projection_interval", max(1, int(getattr(args, "coarse_projection_interval", 1))))
    plan = ns["pvfv_half_gfps_case_plan"](profile)
    face_modes = flow_face_modes(args)
    dt_values = flow_dt_values(coarse_template_cfg, args)
    n_step_values = flow_n_step_values(coarse_template_cfg, args)
    initial_velocity_modes = flow_initial_velocity_modes(coarse_template_cfg, args)
    projection_intervals = flow_projection_intervals(coarse_template_cfg, args)

    wall_plan = {
        "shape_zyx": plan["shape_zyx"],
        "fixed_stride": plan["fixed_stride"],
        "beta_values": plan["beta_values"],
    }
    if args.skip_wall_calibration:
        beta_star = float(args.fixed_beta_star)
        wall = {"beta_star": beta_star, "summary": {"fixed_beta_star": beta_star}}
        print(f"[flow] wall calibration skipped; fixed beta_star={beta_star:g}")
    else:
        print("[flow] wall calibration")
        wall = ns["pvfv_run_wall_calibration"](ref_cfg, wall_plan, out)
        beta_star = float(wall["beta_star"])
        print(f"[flow] beta_star={beta_star:g}")

    specs = filter_case_specs(
        build_case_specs(ns, profile, Path(args.bentheimer_npz), Path(args.fibrous_npz)),
        args,
    )
    ref_rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    spec_rows = summarize_case_specs(ns, specs)
    write_csv(out / "flow_case_spec_preflight.csv", spec_rows)
    flow_units_per_stride = max(
        1,
        len(face_modes)
        * len(initial_velocity_modes)
        * len(projection_intervals)
        * len(dt_values)
        * len(n_step_values),
    )
    total_units = sum(1 + len(spec["strides"]) * flow_units_per_stride for spec in specs)
    progress = FlowProgress(
        enabled=progress_enabled_from_args(args),
        total_units=total_units,
        every_s=float(getattr(args, "progress_every_s", 30.0) or 0.0),
    )
    progress.emit(
        f"[flow][progress] plan: cases={len(specs)}, "
        f"references={len(specs)}, coarse_runs={total_units - len(specs)}, "
        f"heartbeat={format_duration(progress.every_s)}"
    )

    for spec in specs:
        case = spec["case"]
        print(f"\n[flow] reference for {case}")
        mask = make_mask(ns, spec)
        stage_label = f"reference {case}"
        stage_t0 = progress.begin(stage_label)
        with progress.heartbeat(stage_label, stage_t0):
            ref_geom, ref_res, ref_row = ns["pvfv_build_reference"](case, mask, ref_cfg, out=out)
        progress.complete(stage_label, stage_t0)
        ref_row["paper_case"] = spec["paper_case"]
        ref_row["mask_kind"] = spec["mask_kind"]
        ref_row["main_text_case"] = bool(spec["main_text_case"])
        ref_row["mask_npz"] = spec.get("mask_npz", "")
        ref_row["cfg_dt"] = float(ref_cfg.dt)
        ref_row["cfg_n_steps"] = int(ref_cfg.n_steps)
        ref_row["cfg_enable_convection"] = bool(ref_cfg.enable_convection)
        ref_rows.append(ref_row)

        for stride in spec["strides"]:
            candidates = ns["pvfv_half_gfps_seed_specs_for_density"](mask, tuple(stride), ["half"])
            candidates = [
                item for item in candidates
                if str(item[0].get("family", "")).lower() == "stride_half_admissible"
            ]
            if not candidates:
                raise RuntimeError(f"No half-offset seed spec for {case}, stride={stride}")
            seed_spec, seed_flat = candidates[0]
            for face_mode in face_modes:
                for initial_velocity_mode in initial_velocity_modes:
                    for projection_interval in projection_intervals:
                        for dt in dt_values:
                            for n_steps in n_step_values:
                                flow_cfg = clone_cfg(
                                    ns,
                                    coarse_template_cfg,
                                    out_dir=str(out),
                                    dt=float(dt),
                                    n_steps=int(n_steps),
                                    initial_velocity_mode=str(initial_velocity_mode),
                                )
                                setattr(flow_cfg, "linear_solver_mode", str(getattr(args, "coarse_linear_solver", "cg")))
                                setattr(flow_cfg, "momentum_solver_mode", str(getattr(args, "coarse_momentum_solver", "cg")))
                                setattr(flow_cfg, "projection_interval", int(projection_interval))
                                setattr(flow_cfg, "pressure_gradient_weight", str(getattr(args, "pressure_gradient_weight", "tproj")))
                                setattr(flow_cfg, "velocity_reconstruct_from_flux", bool(getattr(args, "coarse_velocity_reconstruct_from_flux", False)))
                                if getattr(args, "coarse_velocity_reconstruction_lambda", None) is not None:
                                    setattr(flow_cfg, "velocity_reconstruction_lambda", float(args.coarse_velocity_reconstruction_lambda))
                                run_tag = f"flow_{seed_spec['seed_id']}"
                                if (
                                    face_mode != "overrelaxed_default"
                                    or str(initial_velocity_mode) != str(base_cfg.initial_velocity_mode)
                                    or int(projection_interval) != 1
                                    or abs(float(dt) - float(base_cfg.dt)) > 1.0e-15
                                    or int(n_steps) != int(base_cfg.n_steps)
                                    or str(getattr(flow_cfg, "pressure_gradient_weight", "tproj")) != "tproj"
                                    or bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False))
                                ):
                                    run_tag += (
                                        f"__{face_mode}"
                                        f"__ivm{token_text(str(initial_velocity_mode))}"
                                        f"__pint{int(projection_interval)}"
                                        f"__dt{token_float(float(dt))}__n{int(n_steps)}"
                                    )
                                    if str(getattr(flow_cfg, "pressure_gradient_weight", "tproj")) != "tproj":
                                        run_tag += f"__pgw{token_text(str(getattr(flow_cfg, 'pressure_gradient_weight', 'tproj')))}"
                                    if bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False)):
                                        run_tag += f"__vrecflux_lam{token_float(float(getattr(flow_cfg, 'velocity_reconstruction_lambda', 0.0)))}"
                                print(
                                    f"[flow] {case} {run_tag} S={int(seed_flat.size)} "
                                    f"face={face_mode} ivm={str(flow_cfg.initial_velocity_mode)} "
                                    f"formulation={str(getattr(flow_cfg, 'solver_formulation', PRODUCTION_PRESSURE_CORRECTION_FORMULATION))} "
                                    f"pint={int(projection_interval)} "
                                    f"dt={float(flow_cfg.dt):g} n_steps={int(flow_cfg.n_steps)} "
                                    f"pgw={str(getattr(flow_cfg, 'pressure_gradient_weight', 'tproj'))}"
                                )
                                stage_label = f"{case} {run_tag}"
                                setattr(flow_cfg, "pvfv_progress_label", stage_label)
                                stage_t0 = progress.begin(stage_label)
                                with progress.heartbeat(stage_label, stage_t0):
                                    row, _geom, _res = ns["pvfv_run_flow_case_seeded"](
                                        case,
                                        mask,
                                        seed_flat,
                                        seed_spec,
                                        flow_cfg,
                                        ref_geom,
                                        ref_res,
                                        wall_beta=beta_star,
                                        face_mode=face_mode,
                                        reconstruct=bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False)),
                                        run_tag=run_tag,
                                        out=out,
                                        export_data=not bool(getattr(args, "no_export_data", False)),
                                        panel_hint="flow_density",
                                    )
                                row["paper_case"] = spec["paper_case"]
                                row["mask_kind"] = spec["mask_kind"]
                                row["main_text_case"] = bool(spec["main_text_case"])
                                row["mask_npz"] = spec.get("mask_npz", "")
                                row["cfg_dt"] = float(flow_cfg.dt)
                                row["cfg_n_steps"] = int(flow_cfg.n_steps)
                                row["cfg_enable_convection"] = bool(flow_cfg.enable_convection)
                                row["initial_velocity_mode"] = str(flow_cfg.initial_velocity_mode)
                                row["solver_formulation"] = str(
                                    getattr(
                                        flow_cfg,
                                        "solver_formulation",
                                        PRODUCTION_PRESSURE_CORRECTION_FORMULATION,
                                    )
                                )
                                row["solver_formulation_id"] = str(
                                    res.get(
                                        "solver_formulation_id",
                                        "pseudo_time_pressure_correction_BTBt_v1",
                                    )
                                )
                                row["solver_result_scope"] = str(
                                    res.get(
                                        "solver_result_scope",
                                        "production_pressure_correction_control",
                                    )
                                )
                                row["momentum_solver_mode"] = str(getattr(flow_cfg, "momentum_solver_mode", "cg"))
                                row["projection_interval"] = int(projection_interval)
                                row["pressure_gradient_weight"] = str(getattr(flow_cfg, "pressure_gradient_weight", "tproj"))
                                row["velocity_reconstruct_from_flux"] = bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False))
                                row["velocity_reconstruction_lambda"] = float(getattr(flow_cfg, "velocity_reconstruction_lambda", float("nan")))
                                row["wall_closure_mode"] = str(ns.get("_pvfv_wall_closure_mode", "fixed_beta"))
                                wall_meta = ns.get("_pvfv_last_wall_closure_meta", {})
                                if isinstance(wall_meta, dict):
                                    for key in (
                                        "wall_ref_ccomp",
                                        "wall_size_exponent",
                                        "wall_length_exponent",
                                        "wall_scale_min",
                                        "wall_scale_max",
                                        "wall_scale_mean",
                                        "wall_scale_min_observed",
                                        "wall_scale_max_observed",
                                        "wall_ccomp_observed",
                                    ):
                                        if key in wall_meta:
                                            row[key] = wall_meta[key]
                                label_meta = ns.get("_pvfv_last_label_meta", {})
                                if isinstance(label_meta, dict):
                                    for key in (
                                        "t_seed_s", "t_label_s", "t_split_s",
                                        "gpu_face_split_used", "gpu_face_split_iters", "gpu_face_split_verified",
                                        "cuda_geometry_builders_used", "cuda_geometry_builder_mode",
                                        "exact_frontier_tinit_s", "exact_frontier_titer_s",
                                        "exact_frontier_tpred_s", "exact_frontier_iters",
                                        "exact_frontier_max_frontier", "exact_frontier_max_dist",
                                    ):
                                        if key in label_meta:
                                            row[key] = label_meta[key]
                                progress.complete(
                                    stage_label,
                                    stage_t0,
                                    extra=coarse_row_progress_extra(row),
                                )
                                density_rows.append(row)

        del mask
        cp.get_default_memory_pool().free_all_blocks()

    write_csv(out / "pvfv_reference_rows_all.csv", ref_rows)
    write_csv(out / "pvfv_seed_density_sweep.csv", density_rows)
    write_csv(out / "pvfv_flow_sweep_abc.csv", density_rows)
    final_rows = choose_final_rows(density_rows)
    write_csv(out / "table5_final_accuracy_cost_summary.csv", final_rows)
    write_csv(out / "table_flow_final_accuracy_cost_summary.csv", final_rows)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner": str(Path(__file__).resolve()),
        "flow_notebook": str(FLOW_NOTEBOOK),
        "profile": profile,
        "out_dir": str(out),
        "label_backend": args.label_backend,
        "face_operator": str(args.face_operator),
        "darcy_readout_length": "physical_axis_flux_readout",
        "density_family": "stride_half_admissible",
        "case_policy": "7 main-text cases; Data-derived seeds reserved for SI/sensitivity",
        "beta_star": beta_star,
        "wall_closure": {
            "mode": str(ns.get("_pvfv_wall_closure_mode", "fixed_beta")),
            "params": ns.get("_pvfv_wall_closure_params", {}),
        },
        "reference_cfg": {
            "dt": float(ref_cfg.dt),
            "n_steps": int(ref_cfg.n_steps),
            "enable_convection": bool(ref_cfg.enable_convection),
            "initial_velocity_mode": str(ref_cfg.initial_velocity_mode),
            "solver_formulation": str(
                getattr(ref_cfg, "solver_formulation", PRODUCTION_PRESSURE_CORRECTION_FORMULATION)
            ),
            "body_force": [float(v) for v in ref_cfg.body_force],
            "wall_distance_floor": float(ref_cfg.wall_distance_floor),
            "pressure_gauge_eps": float(ref_cfg.pressure_gauge_eps),
            "transmissibility_floor": float(ref_cfg.transmissibility_floor),
            "linear_solver_mode": str(getattr(ref_cfg, "linear_solver_mode", "cg")),
            "momentum_solver_mode": str(getattr(ref_cfg, "momentum_solver_mode", "cg")),
            "projection_interval": int(getattr(ref_cfg, "projection_interval", 1)),
        },
        "coarse_cfg_scan": {
            "face_modes": face_modes,
            "dt_values": dt_values,
            "n_step_values": n_step_values,
            "initial_velocity_modes": initial_velocity_modes,
            "projection_intervals": projection_intervals,
            "enable_convection": bool(coarse_template_cfg.enable_convection),
            "initial_velocity_mode": str(coarse_template_cfg.initial_velocity_mode),
            "solver_formulation": str(
                getattr(
                    coarse_template_cfg,
                    "solver_formulation",
                    PRODUCTION_PRESSURE_CORRECTION_FORMULATION,
                )
            ),
            "body_force": [float(v) for v in coarse_template_cfg.body_force],
            "wall_distance_floor": float(coarse_template_cfg.wall_distance_floor),
            "pressure_gauge_eps": float(coarse_template_cfg.pressure_gauge_eps),
            "transmissibility_floor": float(coarse_template_cfg.transmissibility_floor),
            "pressure_gradient_weight": str(getattr(coarse_template_cfg, "pressure_gradient_weight", "tproj")),
            "linear_solver_mode": str(getattr(coarse_template_cfg, "linear_solver_mode", "cg")),
            "momentum_solver_mode": str(getattr(coarse_template_cfg, "momentum_solver_mode", "cg")),
            "projection_interval": int(getattr(coarse_template_cfg, "projection_interval", 1)),
            "velocity_reconstruct_from_flux": bool(getattr(args, "coarse_velocity_reconstruct_from_flux", False)),
            "velocity_reconstruction_lambda": (
                None
                if getattr(args, "coarse_velocity_reconstruction_lambda", None) is None
                else float(args.coarse_velocity_reconstruction_lambda)
            ),
        },
        "runtime_policy": {
            "export_data": not bool(getattr(args, "no_export_data", False)),
            "face_operator": str(getattr(args, "face_operator", "geodesic_face")),
            "darcy_readout_length": "physical_axis_flux_readout",
            "skip_zero_area_diagnostic": bool(getattr(args, "skip_zero_area_diagnostic", False)),
            "paper_fast_coarse": bool(getattr(args, "paper_fast_coarse", False)),
            "pressure_gradient_weight": str(getattr(args, "pressure_gradient_weight", "tproj")),
            "steady_scalar_initial_guess": bool(getattr(args, "steady_scalar_initial_guess", False)),
            "steady_scalar_solver": str(getattr(args, "steady_scalar_solver", "")),
            "coarse_velocity_reconstruct_from_flux": bool(getattr(args, "coarse_velocity_reconstruct_from_flux", False)),
            "coarse_velocity_reconstruction_lambda": (
                None
                if getattr(args, "coarse_velocity_reconstruction_lambda", None) is None
                else float(args.coarse_velocity_reconstruction_lambda)
            ),
            "gpu_face_split": bool(getattr(args, "gpu_face_split", False)),
            "gpu_face_split_verify": bool(getattr(args, "gpu_face_split_verify", False)),
            "cuda_geometry_builders": bool(ns.get("_pvfv_cuda_geometry_builders", {}).get("used", False)),
            "cuda_geometry_builder_mode": str(ns.get("_pvfv_cuda_geometry_builders", {}).get("mode", "")),
        },
        "case_specs": specs,
        "n_reference_rows": len(ref_rows),
        "n_density_rows": len(density_rows),
        "n_final_rows": len(final_rows),
        "outputs": {
            "case_spec_preflight": str(out / "flow_case_spec_preflight.csv"),
            "reference_rows": str(out / "pvfv_reference_rows_all.csv"),
            "density_sweep": str(out / "pvfv_seed_density_sweep.csv"),
            "final_table": str(out / "table_flow_final_accuracy_cost_summary.csv"),
        },
    }
    profile_summary = runtime_profile_summary(ns)
    if profile_summary:
        profile_path = out / "runtime_profile.json"
        profile_path.write_text(json.dumps(profile_summary, indent=2), encoding="utf-8")
        manifest["outputs"]["runtime_profile"] = str(profile_path)
    (out / "flow_runner_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("PVFV_FLOW_PROFILE", "production"))
    ap.add_argument("--label-backend", default=os.environ.get("PVFV_LABEL_BACKEND", "exact_frontier_gpu"),
                    choices=["exact_frontier_gpu", "exact_geodesic"])
    ap.add_argument("--face-operator", default=os.environ.get("PVFV_FACE_OPERATOR", "geodesic_face"),
                    choices=["geodesic_face", "geodesic_weights", "geodesic_weights_only", "euclidean"],
                    help="Cell-cell face metric used by the production operator; wall closure remains separately controlled.")
    ap.add_argument("--pressure-gradient-weight", default=os.environ.get("PVFV_PRESSURE_GRADIENT_WEIGHT", "tproj"),
                    choices=["tproj", "transmissibility", "area", "sqrt_tproj_area", "unit"],
                    help="Face weight used in the cell pressure-gradient least-squares reconstruction.")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT / "flow_runner")
    ap.add_argument("--bentheimer-npz", type=Path, default=BENTHEIMER_INPUT)
    ap.add_argument("--fibrous-npz", type=Path, default=FIBROUS_INPUT)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--case-filter", default="",
                    help="Comma-separated internal or paper case names to run.")
    ap.add_argument("--max-strides-per-case", type=int, default=0,
                    help="For smoke tests, keep only the first N seed-density strides per case.")
    ap.add_argument("--stride-indices", default="",
                    help="Comma-separated zero-based stride indices to run, e.g. '1,2' for medium/low.")
    ap.add_argument("--skip-wall-calibration", action="store_true",
                    help="Use --fixed-beta-star instead of running wall calibration.")
    ap.add_argument("--fixed-beta-star", type=float, default=1.25,
                    help="Beta used when --skip-wall-calibration is set.")
    ap.add_argument("--wall-closure-mode", default=os.environ.get("PVFV_WALL_CLOSURE_MODE", "fixed_beta"),
                    choices=["fixed_beta", "global_comp_power", "cell_volume_power"],
                    help="Wall transmissibility scaling used after the scalar beta anchor.")
    ap.add_argument("--wall-size-exponent", type=float, default=float(os.environ.get("PVFV_WALL_SIZE_EXPONENT", "0.0")),
                    help="Compression-ratio exponent p for resolution-aware wall scaling; kept for compatibility.")
    ap.add_argument("--wall-length-exponent", type=float, default=(
        None if os.environ.get("PVFV_WALL_LENGTH_EXPONENT", "") == "" else float(os.environ["PVFV_WALL_LENGTH_EXPONENT"])
    ),
                    help="Length-scale wall-resolution exponent eta_w. If provided, p=eta_w/3.")
    ap.add_argument("--wall-ref-ccomp", type=float, default=float(os.environ.get("PVFV_WALL_REF_CCOMP", "64.0")),
                    help="Reference compression/cell-volume scale for size-aware wall closure.")
    ap.add_argument("--wall-scale-min", type=float, default=float(os.environ.get("PVFV_WALL_SCALE_MIN", "0.05")),
                    help="Lower clip for size-aware wall scale.")
    ap.add_argument("--wall-scale-max", type=float, default=float(os.environ.get("PVFV_WALL_SCALE_MAX", "20.0")),
                    help="Upper clip for size-aware wall scale.")
    ap.add_argument("--face-mode", default="overrelaxed_default",
                    help="Single coarse face mode to run.")
    ap.add_argument("--face-modes", default="",
                    help="Comma-separated coarse face modes; overrides --face-mode.")
    ap.add_argument("--reference-dt", type=float, default=None,
                    help="Override reference solve pseudo-time step.")
    ap.add_argument("--reference-n-steps", type=int, default=None,
                    help="Override reference solve maximum steps.")
    ap.add_argument("--reference-report-every", type=int, default=None,
                    help="Override reference report interval.")
    ap.add_argument("--reference-initial-velocity-mode", default=None,
                    help="Override reference initial velocity mode.")
    ap.add_argument(
        "--reference-solver-formulation",
        default=PRODUCTION_PRESSURE_CORRECTION_FORMULATION,
        choices=sorted(SOLVER_FORMULATIONS),
        help="Mathematical formulation for the reference solve; diagnostic KKT is never a production-table row.",
    )
    ap.add_argument("--coarse-dt", type=float, default=None,
                    help="Override coarse solve pseudo-time step.")
    ap.add_argument("--coarse-dt-values", default="",
                    help="Comma-separated coarse dt values to scan after one reference solve.")
    ap.add_argument("--coarse-n-steps", type=int, default=None,
                    help="Override coarse solve maximum steps.")
    ap.add_argument("--coarse-n-steps-values", default="",
                    help="Comma-separated coarse n_steps values to scan after one reference solve.")
    ap.add_argument("--coarse-report-every", type=int, default=None,
                    help="Override coarse report interval.")
    ap.add_argument("--coarse-initial-velocity-mode", default=None,
                    help="Override coarse initial velocity mode.")
    ap.add_argument("--coarse-initial-velocity-modes", default="",
                    help="Comma-separated coarse initial-guess modes to scan after one reference solve.")
    ap.add_argument(
        "--coarse-solver-formulation",
        default=PRODUCTION_PRESSURE_CORRECTION_FORMULATION,
        choices=sorted(SOLVER_FORMULATIONS),
        help="Mathematical formulation for coarse solves; separate from the initial velocity mode.",
    )
    ap.add_argument("--disable-convection", action="store_true",
                    help="Disable convection in both reference and coarse solves for stability diagnostics.")
    ap.add_argument("--body-force-x", type=float, default=None,
                    help="Override x-directed body force in both reference and coarse solves.")
    ap.add_argument("--wall-distance-floor", type=float, default=None,
                    help="Override wall distance floor in both reference and coarse solves.")
    ap.add_argument("--pressure-gauge-eps", type=float, default=None,
                    help="Override pressure gauge regularization in both reference and coarse solves.")
    ap.add_argument("--transmissibility-floor", type=float, default=None,
                    help="Override transmissibility floor in both reference and coarse solves.")
    ap.add_argument("--reference-linear-solver", default="cg", choices=["cg", "direct", "spsolve", "dense_lu", "gpu_dense_lu", "lu"],
                    help="Linear solver mode for the reference flow solve.")
    ap.add_argument("--coarse-linear-solver", default="cg", choices=["cg", "direct", "spsolve", "dense_lu", "gpu_dense_lu", "lu"],
                    help="Linear solver mode for coarse flow solves.")
    ap.add_argument("--reference-momentum-solver", default="cg", choices=["cg", "block_cg", "multi_rhs_cg", "spmm_cg"],
                    help="Momentum predictor solver mode for the reference flow solve.")
    ap.add_argument("--coarse-momentum-solver", default="cg", choices=["cg", "block_cg", "multi_rhs_cg", "spmm_cg"],
                    help="Momentum predictor solver mode for coarse flow solves.")
    ap.add_argument("--reference-projection-interval", type=int, default=1,
                    help="Pressure projection interval for the reference solve.")
    ap.add_argument("--coarse-projection-interval", type=int, default=1,
                    help="Pressure projection interval for coarse solves.")
    ap.add_argument("--coarse-projection-interval-values", default="",
                    help="Comma-separated projection intervals to scan after one reference solve.")
    ap.add_argument("--paper-fast-coarse", action="store_true",
                    help="Use the current fast Stokes coarse preset when explicit coarse settings are absent.")
    ap.add_argument("--steady-scalar-initial-guess", action="store_true",
                    help="Use a direct GPU scalar Stokes warm-start for coarse solves.")
    ap.add_argument("--steady-scalar-solver", default="",
                    choices=["", "direct", "projected", "projected_reconstruct"],
                    help="Replace the coarse iterative solve by a GPU steady-scalar direct/projected solve.")
    ap.add_argument("--coarse-velocity-reconstruct-from-flux", action="store_true",
                    help="Report coarse cell velocities from a flux-consistent local reconstruction after the coarse solve.")
    ap.add_argument("--coarse-velocity-reconstruction-lambda", type=float, default=None,
                    help="Regularization weight for flux-consistent coarse velocity reconstruction; use 0 for flux-only reporting.")
    ap.add_argument("--gpu-face-split", action="store_true",
                    help="Use GPU 3D face-connected component relabeling instead of the CPU pass.")
    ap.add_argument("--gpu-face-split-verify", action="store_true",
                    help="Compare GPU face split against the original CPU pass and fail on mismatch.")
    ap.add_argument("--cuda-geometry-builders", action="store_true",
                    help="Use experimental RawKernel geometry builders; leave off for SI-audited scientific runs.")
    ap.add_argument("--no-export-data", action="store_true",
                    help="Skip heavy per-run NumPy/CSV products during parameter scans.")
    ap.add_argument("--skip-zero-area-diagnostic", action="store_true",
                    help="Skip the optional CPU zero-area diagnostic; reports N_0A=-1.")
    ap.add_argument("--runtime-profile", action="store_true",
                    help="Write inclusive timing diagnostics for major GPU/CPU pipeline stages.")
    ap.add_argument("--runtime-profile-no-sync", action="store_true",
                    help="Do not synchronize around profiled GPU calls. Faster but less exact.")
    ap.add_argument("--progress-every-s", type=float,
                    default=float(os.environ.get("PVFV_FLOW_PROGRESS_EVERY_S", "30")),
                    help="Heartbeat interval for lightweight stage progress and ETA printing.")
    ap.add_argument("--quiet-progress", action="store_true",
                    help="Disable lightweight stage progress, ETA, and label summary lines.")
    args = ap.parse_args()
    if bool(args.quiet_progress):
        os.environ["PVFV_FLOW_PROGRESS"] = "0"
    else:
        os.environ.setdefault("PVFV_FLOW_PROGRESS", "1")
    os.environ["PVFV_FLOW_PROGRESS_EVERY_S"] = str(max(0.0, float(args.progress_every_s)))
    if args.paper_fast_coarse:
        if args.coarse_dt is None and not args.coarse_dt_values:
            args.coarse_dt = 5.0
        if args.coarse_n_steps is None and not args.coarse_n_steps_values:
            args.coarse_n_steps = 100
        if args.coarse_report_every is None:
            args.coarse_report_every = 100
        args.disable_convection = True
    if args.steady_scalar_initial_guess and not args.coarse_initial_velocity_mode:
        args.coarse_initial_velocity_mode = "steady_scalar_x"
    if args.steady_scalar_solver and not args.coarse_initial_velocity_mode:
        args.coarse_initial_velocity_mode = f"steady_scalar_{args.steady_scalar_solver}"
    requested_initial_modes = parse_csv_values(args.coarse_initial_velocity_modes) or [
        str(args.coarse_initial_velocity_mode or "")
    ]
    legacy_solver_modes = [
        mode for mode in requested_initial_modes if str(mode).lower().strip() in MONOLITHIC_STOKES_SOLVER_MODES
    ]
    if legacy_solver_modes:
        ap.error(
            "monolithic_stokes is a solver formulation, not an initial velocity mode; "
            "use --coarse-solver-formulation diagnostic_dense_kkt and an actual initial-guess mode"
        )

    if not Path(args.bentheimer_npz).exists():
        raise FileNotFoundError(args.bentheimer_npz)
    if not Path(args.fibrous_npz).exists():
        raise FileNotFoundError(args.fibrous_npz)

    ns = load_flow_namespace(FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    install_half_only_density(ns)
    install_runtime_cfg_attr_preservation(ns)
    install_lsq_gradient_batched_compat(ns)
    if args.cuda_geometry_builders:
        install_cuda_geometry_builders(ns)
    install_resolution_aware_wall_closure(ns, args)
    coarse_modes = [
        str(mode).lower().strip()
        for mode in (parse_csv_values(args.coarse_initial_velocity_modes) or [str(args.coarse_initial_velocity_mode or "")])
        if str(mode).strip()
    ]
    if (
        str(args.coarse_solver_formulation) == DIAGNOSTIC_DENSE_KKT_FORMULATION
        or str(args.reference_solver_formulation) == DIAGNOSTIC_DENSE_KKT_FORMULATION
    ):
        install_monolithic_stokes_solver_modes(ns)
    if any(mode in STEADY_SCALAR_SOLVER_MODES for mode in coarse_modes):
        install_steady_scalar_solver_modes(ns)
    elif args.steady_scalar_initial_guess or any(mode in STEADY_SCALAR_INITIAL_MODES for mode in coarse_modes):
        install_steady_scalar_initial_guess(ns)
    if str(args.reference_linear_solver).lower().strip() in {"direct", "spsolve"} or str(args.coarse_linear_solver).lower().strip() in {"direct", "spsolve"}:
        install_direct_sparse_linear_solver(ns)
    if str(args.reference_linear_solver).lower().strip() in DENSE_LU_SOLVER_MODES or str(args.coarse_linear_solver).lower().strip() in DENSE_LU_SOLVER_MODES:
        install_dense_lu_linear_solver(ns)
    if str(args.reference_momentum_solver).lower().strip() in {"block_cg", "multi_rhs_cg", "spmm_cg"} or str(args.coarse_momentum_solver).lower().strip() in {"block_cg", "multi_rhs_cg", "spmm_cg"}:
        install_block_momentum_cg(ns)
    projection_intervals_for_install = parse_csv_ints(args.coarse_projection_interval_values) or [int(args.coarse_projection_interval)]
    if int(args.reference_projection_interval) > 1 or any(int(v) > 1 for v in projection_intervals_for_install):
        install_projection_interval_solver(ns)
    if args.skip_zero_area_diagnostic:
        install_skip_zero_area_diagnostic(ns)
    if args.gpu_face_split:
        install_gpu_face_connected_split(ns, verify=bool(args.gpu_face_split_verify))
    os.environ["PVFV_LABEL_BACKEND"] = str(args.label_backend)
    backend = "exact_geodesic"
    if args.label_backend == "exact_frontier_gpu":
        backend = install_label_backend(ns, FLOW_NOTEBOOK.parent)
    args.label_backend = backend
    install_geodesic_face_operator(ns, mode=str(args.face_operator))
    if args.runtime_profile:
        install_runtime_profile(ns, synchronize=not bool(args.runtime_profile_no_sync))

    specs = filter_case_specs(
        build_case_specs(ns, args.profile, Path(args.bentheimer_npz), Path(args.fibrous_npz)),
        args,
    )
    if args.dry_run:
        rows = summarize_case_specs(ns, specs)
        out = Path(args.out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "flow_case_spec_preflight.csv", rows)
        print(json.dumps({
            "dry_run": True,
            "label_backend": backend,
            "out_dir": str(out),
            "case_rows": rows,
        }, indent=2))
        return

    manifest = run_flow(ns, args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
