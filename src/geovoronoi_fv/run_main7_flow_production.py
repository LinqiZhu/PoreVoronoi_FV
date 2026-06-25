from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(os.environ.get("GEOVORONOI_FV_ROOT", Path(__file__).resolve().parents[2])).resolve()
ROOT = Path(os.environ.get("GEOVORONOI_FV_WORK_ROOT", PACKAGE_ROOT)).resolve()
FLOW_NOTEBOOK = ROOT / "notebooks" / "TAGVFV_CMAME_ABC_half_gfps_admissible_seed_suite.ipynb"
BENTHEIMER_INPUT = ROOT / "production_inputs" / "bentheimer_dry_crop_16x96x96_origin_1562_59_349_pore0.npz"
FIBROUS_INPUT = ROOT / "production_inputs" / "fibrous_filter_proxy_16x64x64_from_review_mask.npz"
OUT_ROOT = ROOT / "production_runs"
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
DENSE_LU_SOLVER_MODES = {"dense_lu", "gpu_dense_lu", "lu"}

try:
    from geodesic_face_operator import build_geodesic_face_metric, clone_geometry, physical_flux_readout
except ModuleNotFoundError:
    _LOCAL_GEODESIC_MODULE_DIR = PACKAGE_ROOT / "src" / "geovoronoi_fv"
    if _LOCAL_GEODESIC_MODULE_DIR.exists():
        sys.path.insert(0, str(_LOCAL_GEODESIC_MODULE_DIR))
    from geodesic_face_operator import build_geodesic_face_metric, clone_geometry, physical_flux_readout


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
    module_name = "__main7_flow_namespace__"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    ns: dict[str, Any] = module.__dict__
    ns["__file__"] = str(path)
    # Cells 1, 3, 5, 7, and 9 define the GPU solver, manuscript data export,
    # seed-influence suite, and half-offset/GFPS admissible seed protocol.  The
    # final run-control cell is intentionally skipped here.
    for idx, code in read_notebook_code_cells(path, [1, 3, 5, 7, 9]):
        print(f"[MAIN7] loading notebook cell {idx}")
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


class Main7Progress:
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
            f"[MAIN7][progress] start {done + 1}/{total}: {label}; "
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
            f"[MAIN7][progress] done {done}/{total} ({pct:.1f}%): {label}; "
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
    def __init__(self, meter: Main7Progress, label: str, started_s: float) -> None:
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
                f"[MAIN7][progress] running {done + 1}/{total}: {self.label}; "
                f"stage_elapsed={format_duration(now - self.started_s)}, "
                f"total_elapsed={format_duration(now - self.meter.start_s)}, "
                f"eta_after_completed={format_duration(self.meter.eta_seconds(now=now))}"
            )


def progress_enabled_from_args(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "quiet_progress", False)):
        return False
    env = str(os.environ.get("CMAME_MAIN7_PROGRESS", "1")).lower().strip()
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
        pieces.append(f"roi={1000.0 * float(row.get('roi_tpred_s', 0.0)):.2f}ms")
    except Exception:
        try:
            pieces.append(f"part={1000.0 * float(row.get('t_part_s', 0.0)):.2f}ms")
        except Exception:
            pass
    return ", ".join(pieces)


def load_mask_npz(ns: dict[str, Any], path: Path):
    cp = ns["cp"]
    data = np.load(path, allow_pickle=True)
    mask_np = np.asarray(data["mask"], dtype=bool)
    return cp.asarray(mask_np)


def install_half_only_density(ns: dict[str, Any]) -> None:
    original = ns["cmame_half_gfps_seed_specs_for_density"]

    def half_only(mask, stride_zyx, *args, **kwargs):
        try:
            specs = original(mask, stride_zyx, *args, **kwargs)
        except TypeError:
            specs = original(mask, stride_zyx, ["half"])
        return [
            item for item in specs
            if str(item[0].get("family", "")).lower() == "stride_half_admissible"
        ]

    ns["cmame_half_gfps_seed_specs_for_density"] = half_only


def install_runtime_cfg_attr_preservation(ns: dict[str, Any]) -> None:
    """Preserve runner-only config attributes across notebook cmame_clone_cfg calls."""
    original = ns["cmame_clone_cfg"]
    if getattr(original, "_cmame_preserves_runtime_attrs", False):
        return

    runtime_attrs = (
        "linear_solver_mode",
        "momentum_solver_mode",
        "projection_interval",
        "cmame_progress_label",
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

    clone_with_runtime_attrs._cmame_preserves_runtime_attrs = True  # type: ignore[attr-defined]
    ns["cmame_clone_cfg"] = clone_with_runtime_attrs
    print("[MAIN7] runtime cfg attributes preserved across notebook clones")


def install_skip_zero_area_diagnostic(ns: dict[str, Any]) -> None:
    def skipped_zero_area_count(*_args, **_kwargs) -> int:
        return -1

    ns["cmame_zero_area_candidate_count_cpu"] = skipped_zero_area_count
    print("[MAIN7] zero-area CPU diagnostic skipped; N_0A is reported as -1")


def install_steady_scalar_initial_guess(ns: dict[str, Any]) -> None:
    """Add a GPU scalar Stokes warm-start mode for coarse pressure polishing."""
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    original = ns["make_initial_velocity_gpu"]

    if "_cmame_steady_scalar_velocity_gpu" not in ns:
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

        ns["_cmame_steady_scalar_velocity_gpu"] = solve_scalar_velocity_x

    if getattr(original, "_cmame_steady_scalar_initial", False):
        return

    def warm_start(geom, cfg):
        mode = str(getattr(cfg, "initial_velocity_mode", "zero")).lower().strip()
        if mode not in STEADY_SCALAR_INITIAL_MODES:
            return original(geom, cfg)
        return ns["_cmame_steady_scalar_velocity_gpu"](geom, cfg)

    warm_start._cmame_steady_scalar_initial = True  # type: ignore[attr-defined]
    ns["make_initial_velocity_gpu"] = warm_start
    print("[MAIN7] steady-scalar GPU initial guess mode installed")


def install_steady_scalar_solver_modes(ns: dict[str, Any]) -> None:
    """Add direct/projected GPU steady-scalar coarse solver modes."""
    install_steady_scalar_initial_guess(ns)
    cp = ns["cp"]
    original = ns["run_velocity_pressure_projection_gpu"]

    if getattr(original, "_cmame_steady_scalar_solver_modes", False):
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
        U = initial_U.copy() if initial_U is not None else ns["_cmame_steady_scalar_velocity_gpu"](geom, cfg)
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
                print(f"[MAIN7] {run_label} scalar projected solver failure: {solver_failure}")

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

    run_scalar_mode._cmame_steady_scalar_solver_modes = True  # type: ignore[attr-defined]
    run_scalar_mode.__name__ = getattr(original, "__name__", "run_velocity_pressure_projection_gpu")
    run_scalar_mode.__doc__ = getattr(original, "__doc__", None)
    ns["run_velocity_pressure_projection_gpu"] = run_scalar_mode
    print("[MAIN7] steady-scalar direct/projected solver modes installed")


def install_monolithic_stokes_solver_modes(ns: dict[str, Any]) -> None:
    """Add a one-shot coarse steady Stokes saddle-point solve for hard rock cases."""
    cp = ns["cp"]
    original = ns["run_velocity_pressure_projection_gpu"]
    if getattr(original, "_cmame_monolithic_stokes_solver_modes", False):
        return

    def mode_enabled(cfg) -> bool:
        mode = str(getattr(cfg, "initial_velocity_mode", "zero")).lower().strip()
        return mode in MONOLITHIC_STOKES_SOLVER_MODES

    def build_lsq_gradient_matrix(geom, cfg):
        n = int(geom.n_cells)
        owner = geom.owner
        neigh = geom.neigh
        dvec = geom.dvec
        weights = geom.tproj
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
        return finalize_result(geom, cfg, U, p, phi, elapsed, run_label, U_momentum=U_solve)

    run_monolithic._cmame_monolithic_stokes_solver_modes = True  # type: ignore[attr-defined]
    run_monolithic.__name__ = getattr(original, "__name__", "run_velocity_pressure_projection_gpu")
    run_monolithic.__doc__ = getattr(original, "__doc__", None)
    ns["run_velocity_pressure_projection_gpu"] = run_monolithic
    print("[MAIN7] monolithic steady Stokes solver mode installed")


def install_direct_sparse_linear_solver(ns: dict[str, Any]) -> None:
    """Add a cfg.linear_solver_mode='direct' option for coarse GPU sparse solves."""
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    cpx_spla = ns.get("cpx_spla")
    if cpx_spla is None or not hasattr(cpx_spla, "spsolve"):
        raise RuntimeError("cupyx.scipy.sparse.linalg.spsolve is not available in this kernel")

    original_pressure = ns["solve_pressure_correction_gpu"]
    original_momentum = ns["implicit_momentum_predictor_gpu"]

    if getattr(original_pressure, "_cmame_direct_sparse_linear_solver", False):
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

    solve_pressure._cmame_direct_sparse_linear_solver = True  # type: ignore[attr-defined]
    solve_momentum._cmame_direct_sparse_linear_solver = True  # type: ignore[attr-defined]
    ns["solve_pressure_correction_gpu"] = solve_pressure
    ns["implicit_momentum_predictor_gpu"] = solve_momentum
    print("[MAIN7] direct sparse linear solver mode installed")


def install_dense_lu_linear_solver(ns: dict[str, Any]) -> None:
    """Use cached GPU dense LU factors for small coarse pressure/momentum solves."""
    cp = ns["cp"]
    cpx_sp = ns["cpx_sp"]
    cpx_linalg = importlib.import_module("cupyx.scipy.linalg")
    original_pressure = ns["solve_pressure_correction_gpu"]
    original_momentum = ns["implicit_momentum_predictor_gpu"]

    if getattr(original_pressure, "_cmame_dense_lu_linear_solver", False):
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

    solve_pressure._cmame_dense_lu_linear_solver = True  # type: ignore[attr-defined]
    solve_momentum._cmame_dense_lu_linear_solver = True  # type: ignore[attr-defined]
    ns["solve_pressure_correction_gpu"] = solve_pressure
    ns["implicit_momentum_predictor_gpu"] = solve_momentum
    ns["_cmame_dense_lu_pressure_cache"] = pressure_cache
    ns["_cmame_dense_lu_momentum_cache"] = momentum_cache
    print("[MAIN7] cached GPU dense-LU linear solver mode installed")


def install_lsq_gradient_batched_compat(ns: dict[str, Any]) -> None:
    """Use an explicit batched 3x3 solve for the existing LSQ gradient kernel."""
    cp = ns["cp"]
    kernel = ns.get("_LSQ_ASSEMBLE_KERNEL")
    threads_blocks = ns.get("_threads_blocks")
    if kernel is None or threads_blocks is None:
        raise RuntimeError("LSQ gradient kernel dependencies are unavailable")
    original = ns["lsq_gradient_gpu"]
    if getattr(original, "_cmame_lsq_batched_compat", False):
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
                geom.tproj,
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

    lsq_gradient_gpu._cmame_lsq_batched_compat = True  # type: ignore[attr-defined]
    lsq_gradient_gpu.__name__ = getattr(original, "__name__", "lsq_gradient_gpu")
    lsq_gradient_gpu.__doc__ = getattr(original, "__doc__", None)
    ns["lsq_gradient_gpu"] = lsq_gradient_gpu
    print("[MAIN7] existing LSQ gradient kernel uses explicit batched 3x3 solve")


def install_block_momentum_cg(ns: dict[str, Any]) -> None:
    """Solve the three momentum components as one GPU multi-RHS CG problem."""
    cp = ns["cp"]
    original = ns["implicit_momentum_predictor_gpu"]
    if getattr(original, "_cmame_block_momentum_cg", False):
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

    solve_momentum._cmame_block_momentum_cg = True  # type: ignore[attr-defined]
    ns["implicit_momentum_predictor_gpu"] = solve_momentum
    print("[MAIN7] block multi-RHS momentum CG mode installed")


def install_projection_interval_solver(ns: dict[str, Any]) -> None:
    """Add cfg.projection_interval>1 support for cheaper coarse projection loops."""
    cp = ns["cp"]
    original = ns["run_velocity_pressure_projection_gpu"]
    if getattr(original, "_cmame_projection_interval_solver", False):
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
                print(f"[MAIN7] {run_label} projection-interval solver failure at step {step}: {solver_failure}")
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
                    f"[MAIN7] {run_label} step {step:5d}/{cfg.n_steps}: "
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
                    print(f"[MAIN7] steady convergence reached at step {step}: rel_dU={rel_change:.3e}, mom_inf={mom_inf:.3e}")
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

    run_with_interval._cmame_projection_interval_solver = True  # type: ignore[attr-defined]
    ns["run_velocity_pressure_projection_gpu"] = run_with_interval
    print("[MAIN7] projection-interval solver mode installed")


def install_gpu_face_connected_split(ns: dict[str, Any], *, verify: bool = False) -> None:
    """Replace CPU 3D face-connected relabeling with a GPU union-find equivalent."""
    cp = ns["cp"]
    original = ns.get("cmame_face_connected_reindex_cpu")
    if original is None:
        raise RuntimeError("cmame_face_connected_reindex_cpu is not available")

    init_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cmame_face_split_init(
        const unsigned char* __restrict__ mask,
        const int* __restrict__ labels,
        int* __restrict__ parent,
        const int nvox
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;
        parent[idx] = (mask[idx] && labels[idx] >= 0) ? idx : -1;
    }
    ''', "cmame_face_split_init")

    union_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    int cmame_find_root(int* parent, int x) {
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
    void cmame_try_union(
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
            int ra = cmame_find_root(parent, a);
            int rb = cmame_find_root(parent, b);
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
    void cmame_face_split_union6(
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

        if (z + 1 < D) cmame_try_union(parent, labels, idx, idx + HW, changed);
        if (y + 1 < H) cmame_try_union(parent, labels, idx, idx + W, changed);
        if (W > 1) {
            int nx_idx = (x + 1 < W) ? (idx + 1) : (idx - (W - 1));
            cmame_try_union(parent, labels, idx, nx_idx, changed);
        }
    }
    ''', "cmame_face_split_union6")

    compress_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    int cmame_find_root_compress(int* parent, int x) {
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
    void cmame_face_split_compress(
        int* __restrict__ parent,
        const int nvox
    ) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nvox) return;
        if (parent[idx] < 0) return;
        int r = cmame_find_root_compress(parent, idx);
        parent[idx] = r;
    }
    ''', "cmame_face_split_compress")

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

        max_iters_env = os.environ.get("CMAME_GPU_FACE_SPLIT_MAX_ITERS", "")
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
            if str(os.environ.get("CMAME_GPU_FACE_SPLIT_FALLBACK", "1")).lower().strip() not in {"0", "false", "no", "off"}:
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
            ns["_cmame_last_gpu_face_split"] = {"used": True, "iters": int(used_iters), "n_valid": 0}
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
        ns["_cmame_last_gpu_face_split"] = {
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
            ns["_cmame_last_gpu_face_split"]["verified"] = True
        return new_labels, info

    gpu_face_connected_reindex._cmame_gpu_face_split = True  # type: ignore[attr-defined]
    ns["cmame_face_connected_reindex_cpu"] = gpu_face_connected_reindex
    try:
        warm_mask = cp.ones((1, 1, 2), dtype=cp.uint8)
        warm_labels = cp.zeros((1, 1, 2), dtype=cp.int32)
        gpu_face_connected_reindex(warm_mask, warm_labels)
        cp.cuda.Stream.null.synchronize()
        ns["_cmame_last_gpu_face_split"] = {"used": False}
    except Exception as exc:
        print(f"[MAIN7] GPU face split warm-up skipped: {exc}")
    print(f"[MAIN7] GPU 3D face-connected split installed; verify={bool(verify)}")


def install_cuda_geometry_builders(ns: dict[str, Any]) -> None:
    """Install RawKernel front-ends for Voronoi-to-FV geometry construction."""
    cp = ns["cp"]
    aggregate_faces_by_key = ns.get("_aggregate_faces_by_key")
    cfg_cls = ns.get("PB615Config")
    if aggregate_faces_by_key is None or cfg_cls is None:
        raise RuntimeError("geometry dependencies are not available in the notebook namespace")

    cell_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cmame_cell_moments_kernel(
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
    ''', "cmame_cell_moments_kernel")

    wall_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    void cmame_add_wall(
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
    void cmame_wall_moments_kernel(
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
            cmame_add_wall(lab, 0.0, yc, zc, -1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (!periodic_x && x + 1 == W) {
            cmame_add_wall(lab, (double)W * h, yc, zc, 1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (x > 0 && !mask[idx - 1]) {
            cmame_add_wall(lab, (double)x * h, yc, zc, -1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (x + 1 < W && !mask[idx + 1]) {
            cmame_add_wall(lab, ((double)x + 1.0) * h, yc, zc, 1.0, 0.0, 0.0, centroid, twall, area0, floor_delta);
        }

        if (y == 0) {
            cmame_add_wall(lab, xc, 0.0, zc, 0.0, -1.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (y + 1 == H) {
            cmame_add_wall(lab, xc, (double)H * h, zc, 0.0, 1.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (y > 0 && !mask[idx - W]) {
            cmame_add_wall(lab, xc, (double)y * h, zc, 0.0, -1.0, 0.0, centroid, twall, area0, floor_delta);
        }
        if (y + 1 < H && !mask[idx + W]) {
            cmame_add_wall(lab, xc, ((double)y + 1.0) * h, zc, 0.0, 1.0, 0.0, centroid, twall, area0, floor_delta);
        }

        if (z == 0) {
            cmame_add_wall(lab, xc, yc, 0.0, 0.0, 0.0, -1.0, centroid, twall, area0, floor_delta);
        }
        if (z + 1 == D) {
            cmame_add_wall(lab, xc, yc, (double)D * h, 0.0, 0.0, 1.0, centroid, twall, area0, floor_delta);
        }
        if (z > 0 && !mask[idx - HW]) {
            cmame_add_wall(lab, xc, yc, (double)z * h, 0.0, 0.0, -1.0, centroid, twall, area0, floor_delta);
        }
        if (z + 1 < D && !mask[idx + HW]) {
            cmame_add_wall(lab, xc, yc, ((double)z + 1.0) * h, 0.0, 0.0, 1.0, centroid, twall, area0, floor_delta);
        }
    }
    ''', "cmame_wall_moments_kernel")

    face_kernel = cp.RawKernel(r'''
    extern "C" __device__ __forceinline__
    void cmame_emit_facelet(
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
    void cmame_positive_facelets_emit_kernel(
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
        cmame_emit_facelet(
            lab_i, lab_j, n_cells, axis, fcx, fcy, fcz, area0,
            keys, ax, ay, az, xc, yc, zc, area, counter
        );
    }
    ''', "cmame_positive_facelets_emit_kernel")

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
        if str(os.environ.get("CMAME_GEOM_VALIDATE_EMPTY_CELLS", "0")).lower().strip() in {"1", "true", "yes", "on"}:
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
    ns["_cmame_cuda_geometry_builders"] = {"used": True, "mode": "rawkernel_cell_wall_facelets"}

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
        print(f"[MAIN7] CUDA geometry builder warm-up skipped: {exc}")
    print("[MAIN7] CUDA RawKernel geometry builders installed")


def install_roi_backend(ns: dict[str, Any], notebook_dir: Path) -> str:
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    try:
        backend = importlib.import_module("cmame_roi_jfa_backend")
    except Exception as exc:
        print(f"[MAIN7] ROI-JFA backend import failed; exact labels retained: {exc}")
        return "exact_geodesic"

    cp = ns["cp"]
    time_mod = ns.get("time", time)
    original = ns["cmame_build_geometry_from_seed_flat_timed"]
    roi_profile_gpu = str(os.environ.get("CMAME_ROIJFA_PROFILE", "0")).lower().strip() in {"1", "true", "yes", "on"}
    roi_tile_default = os.environ.get("CMAME_ROIJFA_TILE", "8,8,16")
    roi_skip_split = str(os.environ.get("CMAME_ROIJFA_SKIP_SPLIT", "0")).lower().strip() in {"1", "true", "yes", "on"}
    roi_cache_clearance = str(os.environ.get("CMAME_ROIJFA_CACHE_CLEARANCE", "1")).lower().strip() not in {"0", "false", "no", "off"}
    roi_active_list = str(os.environ.get("CMAME_ROIJFA_ACTIVE_LIST", "1")).lower().strip() not in {"0", "false", "no", "off"}
    roi_active_final_iters = max(0, int(os.environ.get("CMAME_ROIJFA_ACTIVE_FINAL_ITERS", "0") or 0))
    roi_active_final_band_iters = max(0, int(os.environ.get("CMAME_ROIJFA_ACTIVE_FINAL_BAND_ITERS", "1") or 1))
    roi_active_final_halo_iters = max(0, int(os.environ.get("CMAME_ROIJFA_ACTIVE_FINAL_HALO_ITERS", "1") or 1))
    roi_active_final_enabled = bool(
        roi_active_final_iters > 0
        and hasattr(backend, "geodesic_voronoi_roi_jfa_with_active_final")
    )
    roi_c2_core = str(os.environ.get("CMAME_ROIJFA_C2_CORE", "0")).lower().strip() in {"1", "true", "yes", "on"}
    roi_c2_margin = float(os.environ.get("CMAME_ROIJFA_C2_MARGIN", "1.0e-6"))
    roi_c2_los = str(os.environ.get("CMAME_ROIJFA_C2_LOS", "0")).lower().strip() in {"1", "true", "yes", "on"}
    roi_stamping_mode = str(os.environ.get("CMAME_ROIJFA_STAMPING_MODE", "")).strip()
    roi_sparse_voxels = str(os.environ.get("CMAME_ROIJFA_SPARSE_VOXELS", "0")).lower().strip() in {"1", "true", "yes", "on"}
    roi_regular_hotpath_mode = str(os.environ.get("CMAME_ROIJFA_REGULAR_STRIDE_HOTPATH", "auto")).lower().strip()
    roi_regular_hotpath = roi_regular_hotpath_mode not in {"0", "false", "no", "off"}
    roi_regular_hotpath_verify = str(os.environ.get("CMAME_ROIJFA_REGULAR_STRIDE_HOTPATH_VERIFY", "0")).lower().strip() in {"1", "true", "yes", "on"}
    roi_regular_nearest_min_voxels = max(0, int(os.environ.get("CMAME_ROIJFA_NEAREST_CERT_MIN_VOXELS", "200000") or 0))
    if roi_active_final_iters > 0 and not roi_active_final_enabled:
        print("[MAIN7] active-final ROI-JFA requested but unavailable; falling back to plain ROI-JFA")
    if roi_sparse_voxels and not hasattr(backend, "geodesic_voronoi_roi_jfa_sparse_voxels"):
        print("[MAIN7] sparse voxel ROI-JFA requested but unavailable; falling back to tile ROI-JFA")
        roi_sparse_voxels = False
    roi_kernel_bundle: dict[str, Any] = {}
    roi_clearance_cache: dict[tuple[int, tuple[int, int, int]], Any] = {}

    try:
        if hasattr(backend, "build_exact_frontier_bfs6_kernel"):
            roi_kernel_bundle["exact_frontier_kernel"] = backend.build_exact_frontier_bfs6_kernel()
        if hasattr(backend, "build_exact_frontier_bfs6_init_kernel"):
            roi_kernel_bundle["exact_frontier_init_kernel"] = backend.build_exact_frontier_bfs6_init_kernel()
        roi_kernel_bundle["stamping_kernel"] = backend.build_seed_stamping_los_parallel_packed_kernel()
        roi_kernel_bundle["tiles_dual_kernel"] = backend.build_tiles_dual_3d_kernel()
        if roi_active_list:
            roi_kernel_bundle["roi_step_kernels"] = {
                "kernel": backend.build_geodesic_roi_jfa_step_active_list_kernels_3d(use_int_offset=True)[0],
                "kernel_full": backend.build_geodesic_roi_jfa_step_kernels_3d(use_int_offset=True)[0],
                "mode": "adaptive_active_list",
            }
        else:
            roi_kernel_bundle["roi_step_kernels"] = backend.build_geodesic_roi_jfa_step_kernels_3d(use_int_offset=True)
        roi_kernel_bundle["active_tiles_kernel"] = backend.build_active_tiles_kernel_3d()
        if roi_active_list:
            roi_kernel_bundle["apply_kernel"] = {
                "kernel": backend.build_apply_roi_tile_updates_active_list_kernel_3d(),
                "kernel_full": backend.build_apply_roi_tile_updates_kernel_3d(),
                "mode": "adaptive_active_list",
            }
        else:
            roi_kernel_bundle["apply_kernel"] = backend.build_apply_roi_tile_updates_kernel_3d()
        roi_kernel_bundle["active_fallback_kernel"] = backend.build_fallback_full_roi_kernel_3d()
        roi_kernel_bundle["closure_kernel"] = backend.build_roi_closure_jump1_packed_kernel_3d()
        roi_kernel_bundle["relax_kernel"] = backend.build_local_relax_packed_kernel_3d()
        roi_kernel_bundle["los_kmax_init_kernel"] = backend.build_los_kmax_init_kernel_3d()
        roi_kernel_bundle["los_kmax_update_kernel"] = backend.build_los_kmax_update_kernel_3d()
        if hasattr(backend, "build_regular_stride_l1_lattice_clearance_label_kernel_3d"):
            roi_kernel_bundle["regular_stride_clearance_kernel"] = backend.build_regular_stride_l1_lattice_clearance_label_kernel_3d()
        if hasattr(backend, "build_regular_stride_l1_lattice_label_kernel_3d"):
            roi_kernel_bundle["regular_stride_lattice_kernel"] = backend.build_regular_stride_l1_lattice_label_kernel_3d()
        if hasattr(backend, "build_regular_stride_l1_lattice_nearest_clearance_check_kernel_3d"):
            roi_kernel_bundle["regular_stride_nearest_check_kernel"] = backend.build_regular_stride_l1_lattice_nearest_clearance_check_kernel_3d()
        if hasattr(backend, "build_regular_stride_l1_lattice_nearest_certified_label_kernel_3d"):
            roi_kernel_bundle["regular_stride_nearest_cert_kernel"] = backend.build_regular_stride_l1_lattice_nearest_certified_label_kernel_3d()
        if hasattr(backend, "build_regular_stride_lattice_lut_fill_kernel"):
            roi_kernel_bundle["regular_stride_lut_fill_kernel"] = backend.build_regular_stride_lattice_lut_fill_kernel()
        if "exact_frontier_kernel" in roi_kernel_bundle:
            warm_mask = cp.ones((2, 2, 2), dtype=cp.uint8)
            warm_seeds = cp.asarray([[0, 0, 0]], dtype=cp.int32)
            backend.exact_frontier_dijkstra_gpu_6(
                warm_mask,
                warm_seeds,
                kernel=roi_kernel_bundle["exact_frontier_kernel"],
                init_kernel=roi_kernel_bundle.get("exact_frontier_init_kernel"),
                validate=False,
                collect_stats=False,
                return_float64=False,
            )
            cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        print(f"[MAIN7] ROI-JFA kernel prebuild skipped: {exc}")
        roi_kernel_bundle = {}

    def infer_regular_stride(seed_spec: Any) -> tuple[int, int, int] | None:
        text = str(seed_spec)
        match = re.search(r"stride_(\d+)x(\d+)x(\d+)_half", text)
        if not match:
            return None
        return tuple(int(match.group(i)) for i in range(1, 4))

    def infer_protocol_radius_override(seed_spec: Any) -> tuple[float | None, str]:
        if str(os.environ.get("CMAME_ROIJFA_PROTOCOL_RADIUS", "1")).lower().strip() in {"0", "false", "no", "off"}:
            return None, ""
        text = str(seed_spec)
        match = re.search(r"stride_(\d+)x(\d+)x(\d+)_half", text)
        if not match:
            return None, ""
        stride = tuple(int(match.group(i)) for i in range(1, 4))
        delta_r = float(os.environ.get("CMAME_ROIJFA_DELTA_R", "1.0"))
        radius = max(0.0, 0.5 * float(min(stride)) - delta_r)
        return radius, f"regular_half_stride_{stride[0]}x{stride[1]}x{stride[2]}"

    def build_regular_lattice_lut(seeds_zyx, shape: tuple[int, int, int], stride: tuple[int, int, int]):
        d, h, w = [int(v) for v in shape]
        n_seed = int(seeds_zyx.shape[0])
        if n_seed <= 0:
            raise ValueError("regular-stride hotpath received no seeds")
        first_seed = [int(v) for v in cp.asnumpy(seeds_zyx[0, :])]
        offsets = [int(first_seed[axis]) % int(stride[axis]) for axis in range(3)]
        oz, oy, ox = offsets
        dims = (
            0 if oz >= d else (d - 1 - oz) // int(stride[0]) + 1,
            0 if oy >= h else (h - 1 - oy) // int(stride[1]) + 1,
            0 if ox >= w else (w - 1 - ox) // int(stride[2]) + 1,
        )
        if min(dims) <= 0:
            raise ValueError("regular-stride hotpath lattice has an empty dimension")
        lut = cp.empty(dims, dtype=cp.int32)
        lut.fill(cp.int32(-1))
        bad = cp.zeros(1, dtype=cp.int32)
        kernel = roi_kernel_bundle.get("regular_stride_lut_fill_kernel")
        if kernel is None:
            raise RuntimeError("regular-stride hotpath LUT fill kernel is unavailable")
        threads = 256
        blocks = (n_seed + threads - 1) // threads
        kernel(
            (int(blocks),),
            (int(threads),),
            (
                seeds_zyx.reshape(-1),
                lut.ravel(),
                bad,
                int(n_seed),
                int(dims[0]), int(dims[1]), int(dims[2]),
                int(stride[0]), int(stride[1]), int(stride[2]),
                int(oz), int(oy), int(ox),
            ),
        )
        if int(bad.get()[0]) != 0:
            raise ValueError("regular-stride hotpath seed coordinates are not on the inferred lattice")
        return lut, (oz, oy, ox)

    if roi_regular_hotpath and roi_kernel_bundle.get("regular_stride_clearance_kernel") is not None:
        try:
            warm_mask = cp.ones((2, 2, 2), dtype=cp.uint8)
            warm_seeds = cp.asarray([[0, 0, 0]], dtype=cp.int32)
            warm_lut, warm_offset = build_regular_lattice_lut(warm_seeds, (2, 2, 2), (1, 1, 1))
            warm_labels, warm_dist, _warm_bad = backend.regular_stride_l1_lattice_nearest_certified_label_gpu_6(
                warm_mask,
                warm_lut,
                (1, 1, 1),
                warm_offset,
                kernel=roi_kernel_bundle.get("regular_stride_nearest_cert_kernel"),
                profile_gpu=False,
            )
            warm_clear_labels, warm_clear_dist = backend.regular_stride_l1_lattice_clearance_label_gpu_6(
                warm_mask,
                warm_lut,
                (1, 1, 1),
                warm_offset,
                kernel=roi_kernel_bundle.get("regular_stride_clearance_kernel"),
                profile_gpu=False,
            )
            _ = warm_labels.astype(cp.int32, copy=False)
            _ = warm_dist.astype(cp.float64, copy=False)
            _ = warm_clear_labels.astype(cp.int32, copy=False)
            _ = warm_clear_dist.astype(cp.float64, copy=False)
            if roi_regular_hotpath_verify and hasattr(backend, "exact_frontier_dijkstra_gpu_6"):
                _ = backend.exact_frontier_dijkstra_gpu_6(
                    warm_mask, warm_seeds, validate=True, collect_stats=False, return_float64=False
                )
            cp.cuda.Stream.null.synchronize()
        except Exception as exc:
            print(f"[MAIN7] regular-stride hotpath warm-up skipped: {exc}")

    def wrapped(mask, seed_flat, cfg, *, split_face_components=True,
                label_mode="exact_geodesic", seed_spec="custom"):
        requested = str(os.environ.get("CMAME_LABEL_BACKEND", "roi_jfa")).lower().strip()
        use_roi_jfa = requested in {"roi_jfa", "roi-jfa", "roijfa"}
        use_exact_frontier = requested in {"exact_frontier_gpu", "exact-frontier-gpu", "frontier_gpu", "frontier-dijkstra-gpu"}
        if not use_roi_jfa and not use_exact_frontier:
            return original(
                mask,
                seed_flat,
                cfg,
                split_face_components=split_face_components,
                label_mode=label_mode,
                seed_spec=seed_spec,
            )

        t0 = time_mod.perf_counter()
        seed_flat = cp.asarray(seed_flat, dtype=cp.int64)
        cp.cuda.Stream.null.synchronize()
        t_seed = time_mod.perf_counter() - t0

        d, h, w = [int(v) for v in mask.shape]
        mask_u8 = mask.astype(cp.uint8, copy=False)
        mask_flat_u8 = mask_u8.ravel()
        z = seed_flat // cp.int64(h * w)
        rem = seed_flat - z * cp.int64(h * w)
        y = rem // cp.int64(w)
        x = rem - y * cp.int64(w)
        seeds_zyx = cp.stack([z, y, x], axis=1).astype(cp.int32, copy=False)
        selected_label_mode = "exact_frontier_gpu" if use_exact_frontier else "roi_jfa"
        regular_stride = infer_regular_stride(seed_spec)
        regular_hotpath_available = bool(
            use_roi_jfa
            and
            roi_regular_hotpath
            and regular_stride is not None
            and int(getattr(cfg, "distance_connectivity", 0)) == 6
            and hasattr(backend, "regular_stride_l1_lattice_clearance_label_gpu_6")
            and roi_kernel_bundle.get("regular_stride_clearance_kernel") is not None
        )
        regular_nearest_cert_available = bool(
            regular_hotpath_available
            and hasattr(backend, "regular_stride_l1_lattice_nearest_certified_label_gpu_6")
            and roi_kernel_bundle.get("regular_stride_nearest_cert_kernel") is not None
        )

        clearance_kmax27 = None
        if roi_cache_clearance and not regular_hotpath_available:
            clearance_key = (int(mask_u8.data.ptr), (d, h, w))
            clearance_kmax27 = roi_clearance_cache.get(clearance_key)
            if clearance_kmax27 is None:
                maxdim = int(max(d, h, w))
                max_k = int(maxdim.bit_length() - 1)
                clearance_kmax27 = backend.precompute_los_kmax_27dirs_3d(
                    mask_flat_u8,
                    d, h, w,
                    max_k=max_k,
                    init_kernel=roi_kernel_bundle.get("los_kmax_init_kernel"),
                    update_kernel=roi_kernel_bundle.get("los_kmax_update_kernel"),
                    verbose=False,
                )
                roi_clearance_cache[clearance_key] = clearance_kmax27

        t1 = time_mod.perf_counter()
        radius_override, radius_mode = infer_protocol_radius_override(seed_spec)
        previous_radius_override = os.environ.get("CMAME_ROIJFA_RADIUS_OVERRIDE")
        if radius_override is not None:
            os.environ["CMAME_ROIJFA_RADIUS_OVERRIDE"] = f"{float(radius_override):.17g}"
        elif previous_radius_override is not None:
            os.environ.pop("CMAME_ROIJFA_RADIUS_OVERRIDE", None)
        try:
            fast_lut_s = 0.0
            fast_hotpath_s = 0.0
            fast_verify_mismatch = -1
            fast_uncertified = -1
            fast_nearest_bad = -1
            fast_lattice_kernel_s = 0.0
            fast_nearest_check_s = 0.0
            fast_clearance_kernel_s = 0.0
            regular_strategy = "none"
            regular_offset = ()
            roi_label_engine = "exact_frontier_gpu" if use_exact_frontier else "roi_jfa"
            if use_exact_frontier:
                labels, dist = backend.exact_frontier_dijkstra_gpu_6(
                    mask_u8,
                    seeds_zyx,
                    kernel=roi_kernel_bundle.get("exact_frontier_kernel"),
                    init_kernel=roi_kernel_bundle.get("exact_frontier_init_kernel"),
                    validate=False,
                    collect_stats=False,
                    return_float64=False,
                )
            elif regular_hotpath_available:
                lut_t0 = time_mod.perf_counter()
                lattice_lut, regular_offset = build_regular_lattice_lut(seeds_zyx, (d, h, w), regular_stride)
                cp.cuda.Stream.null.synchronize()
                fast_lut_s = time_mod.perf_counter() - lut_t0
                hot_t0 = time_mod.perf_counter()
                mode_forces_nearest = roi_regular_hotpath_mode in {"nearest", "nearest_cert", "nearest_certified", "fused"}
                mode_threshold_nearest = roi_regular_hotpath_mode in {"auto_nearest", "threshold_nearest"}
                mode_forces_clearance = roi_regular_hotpath_mode in {"clearance", "clearance_only", "full_clearance"}
                use_nearest_cert = bool(
                    regular_nearest_cert_available
                    and not mode_forces_clearance
                    and (mode_forces_nearest or (mode_threshold_nearest and int(mask_u8.size) >= int(roi_regular_nearest_min_voxels)))
                )
                if use_nearest_cert:
                    regular_strategy = "nearest_cert_then_clearance"
                    labels, dist, fast_nearest_bad = backend.regular_stride_l1_lattice_nearest_certified_label_gpu_6(
                        mask_u8,
                        lattice_lut,
                        regular_stride,
                        regular_offset,
                        kernel=roi_kernel_bundle.get("regular_stride_nearest_cert_kernel"),
                        profile_gpu=bool(roi_profile_gpu),
                    )
                    fast_lattice_kernel_s = float(getattr(backend, "REGULAR_STRIDE_L1_LAST_NEAREST_CERT_TIME", 0.0))
                    fast_nearest_check_s = 0.0
                    fast_nearest_bad = int(fast_nearest_bad)
                    fast_uncertified = fast_nearest_bad
                    if fast_nearest_bad == 0:
                        roi_label_engine = "regular_stride_nearest_certified"
                    else:
                        labels, dist = backend.regular_stride_l1_lattice_clearance_label_gpu_6(
                            mask_u8,
                            lattice_lut,
                            regular_stride,
                            regular_offset,
                            kernel=roi_kernel_bundle.get("regular_stride_clearance_kernel"),
                            profile_gpu=bool(roi_profile_gpu),
                        )
                        fast_clearance_kernel_s = float(getattr(backend, "REGULAR_STRIDE_L1_LAST_KERNEL_TIME", 0.0))
                        fast_uncertified = int(getattr(backend, "REGULAR_STRIDE_L1_LAST_UNCERTIFIED", 0))
                else:
                    regular_strategy = "clearance_direct"
                    labels, dist = backend.regular_stride_l1_lattice_clearance_label_gpu_6(
                        mask_u8,
                        lattice_lut,
                        regular_stride,
                        regular_offset,
                        kernel=roi_kernel_bundle.get("regular_stride_clearance_kernel"),
                        profile_gpu=bool(roi_profile_gpu),
                    )
                    fast_clearance_kernel_s = float(getattr(backend, "REGULAR_STRIDE_L1_LAST_KERNEL_TIME", 0.0))
                    fast_uncertified = int(getattr(backend, "REGULAR_STRIDE_L1_LAST_UNCERTIFIED", 0))
                fast_hotpath_s = time_mod.perf_counter() - hot_t0
                if fast_uncertified != 0:
                    labels, dist = backend.exact_frontier_dijkstra_gpu_6(
                        mask_u8, seeds_zyx, validate=True, collect_stats=False, return_float64=False
                    )
                    roi_label_engine = "regular_stride_clearance_fallback_exact"
                else:
                    if fast_nearest_bad != 0:
                        roi_label_engine = "regular_stride_clearance"
                    elif regular_strategy == "clearance_direct":
                        roi_label_engine = "regular_stride_clearance_direct"
                    if roi_regular_hotpath_verify:
                        exact_labels, exact_dist = backend.exact_frontier_dijkstra_gpu_6(
                            mask_u8, seeds_zyx, validate=True, collect_stats=False, return_float64=False
                        )
                        fast_verify_mismatch = int(cp.count_nonzero((mask_u8 != 0) & (labels != exact_labels)).get())
                        if fast_verify_mismatch != 0:
                            raise RuntimeError(
                                f"regular-stride hotpath verification failed: {fast_verify_mismatch} label mismatches"
                            )
            else:
                if roi_sparse_voxels:
                    roi_solver = backend.geodesic_voronoi_roi_jfa_sparse_voxels
                elif roi_active_final_enabled:
                    roi_solver = backend.geodesic_voronoi_roi_jfa_with_active_final
                else:
                    roi_solver = backend.geodesic_voronoi_roi_jfa
                roi_kwargs = {
                    "tile_size": tuple(int(v) for v in roi_tile_default.split(",")),
                    "delta_r": float(os.environ.get("CMAME_ROIJFA_DELTA_R", "1.0")),
                    "eta_max": float(os.environ.get("CMAME_ROIJFA_ETA_MAX", "0.8")),
                    "r_tile": int(os.environ.get("CMAME_ROIJFA_R_TILE", "1")),
                    "verbose": False,
                    "viz_policy": "none",
                    "n_relax_after": int(os.environ.get("CMAME_ROIJFA_RELAX_AFTER", "1")),
                    "profile_gpu": bool(roi_profile_gpu),
                    "return_records": False,
                    "clearance_kmax27": clearance_kmax27,
                    "stamping_kernel": roi_kernel_bundle.get("stamping_kernel"),
                    "tiles_dual_kernel": roi_kernel_bundle.get("tiles_dual_kernel"),
                    "roi_step_kernels": roi_kernel_bundle.get("roi_step_kernels"),
                    "active_tiles_kernel": roi_kernel_bundle.get("active_tiles_kernel"),
                    "apply_kernel": roi_kernel_bundle.get("apply_kernel"),
                    "active_fallback_kernel": roi_kernel_bundle.get("active_fallback_kernel"),
                    "closure_kernel": roi_kernel_bundle.get("closure_kernel"),
                    "relax_kernel": roi_kernel_bundle.get("relax_kernel"),
                    "los_kmax_init_kernel": roi_kernel_bundle.get("los_kmax_init_kernel"),
                    "los_kmax_update_kernel": roi_kernel_bundle.get("los_kmax_update_kernel"),
                    "use_active_list_step": roi_active_list,
                }
                if roi_stamping_mode:
                    roi_kwargs["stamping_kernel"] = roi_stamping_mode
                if roi_active_final_enabled:
                    roi_kwargs.update({
                        "active_final_iters": int(roi_active_final_iters),
                        "active_final_band_iters": int(roi_active_final_band_iters),
                        "active_final_halo_iters": int(roi_active_final_halo_iters),
                    })
                labels, dist, _tile_roi, _roi_mask = roi_solver(mask_u8, seeds_zyx, **roi_kwargs)
        finally:
            if previous_radius_override is None:
                os.environ.pop("CMAME_ROIJFA_RADIUS_OVERRIDE", None)
            else:
                os.environ["CMAME_ROIJFA_RADIUS_OVERRIDE"] = previous_radius_override
        labels = labels.astype(cp.int32, copy=False)
        if regular_hotpath_available or use_exact_frontier:
            dist = dist.astype(cp.float32, copy=False)
        else:
            dist = dist.astype(cp.float64, copy=False)
        cp.cuda.Stream.null.synchronize()
        t_label = time_mod.perf_counter() - t1

        n_seed = int(seed_flat.size)
        t_split0 = time_mod.perf_counter()
        ns["_cmame_last_gpu_face_split"] = {"used": False}
        split_info = {
            "n_original_labels": n_seed,
            "n_cv": n_seed,
            "n_split_extra": 0,
            "n_face_disconnected_labels": 0,
            "face_connected_fraction": 1.0,
        }
        if split_face_components and not roi_skip_split:
            labels, split_info = ns["cmame_face_connected_reindex_cpu"](mask, labels)
        cp.cuda.Stream.null.synchronize()
        t_split = time_mod.perf_counter() - t_split0
        gpu_split_meta = ns.get("_cmame_last_gpu_face_split", {})
        if not isinstance(gpu_split_meta, dict):
            gpu_split_meta = {}

        n0a = ns["cmame_zero_area_candidate_count_cpu"](mask, labels)
        t2 = time_mod.perf_counter()
        geom = ns["cmame_build_geometry_from_labels_ncells_gpu"](
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
            "label_mode": str(selected_label_mode),
            "roi_label_engine": str(roi_label_engine),
            "distance_connectivity": int(cfg.distance_connectivity),
            "seed_spec": str(seed_spec),
            "roi_regular_stride_hotpath": bool(regular_hotpath_available),
            "roi_regular_stride_hotpath_verify": bool(roi_regular_hotpath_verify),
            "roi_regular_strategy": str(regular_strategy),
            "roi_regular_nearest_min_voxels": int(roi_regular_nearest_min_voxels),
            "roi_regular_stride": "" if regular_stride is None else str(tuple(int(v) for v in regular_stride)),
            "roi_regular_offset": "" if not regular_hotpath_available else str(tuple(int(v) for v in regular_offset)),
            "roi_regular_uncertified": int(fast_uncertified),
            "roi_regular_nearest_bad": int(fast_nearest_bad),
            "roi_regular_verify_mismatch": int(fast_verify_mismatch),
            "roi_regular_lut_s": float(fast_lut_s),
            "roi_regular_hotpath_s": float(fast_hotpath_s),
            "roi_regular_lattice_kernel_s": float(fast_lattice_kernel_s),
            "roi_regular_nearest_check_s": float(fast_nearest_check_s),
            "roi_regular_clearance_kernel_s": float(fast_clearance_kernel_s),
            "roi_regular_kernel_s": float(getattr(backend, "REGULAR_STRIDE_L1_LAST_KERNEL_TIME", 0.0)) if regular_hotpath_available else 0.0,
            "roi_tile_size": str(tuple(int(v) for v in roi_tile_default.split(","))),
            "roi_profile_gpu": bool(roi_profile_gpu),
            "roi_cache_clearance": bool(roi_cache_clearance),
            "roi_skip_face_split": bool(roi_skip_split),
            "roi_active_final_enabled": bool(roi_active_final_enabled),
            "roi_active_final_iters": int(roi_active_final_iters if roi_active_final_enabled else 0),
            "roi_active_final_band_iters": int(roi_active_final_band_iters),
            "roi_active_final_halo_iters": int(roi_active_final_halo_iters),
            "roi_c2_core": bool(roi_c2_core),
            "roi_c2_margin": float(roi_c2_margin),
            "roi_c2_los": bool(roi_c2_los),
            "roi_c2_los_used": bool(False if (regular_hotpath_available or use_exact_frontier) else getattr(backend, "ROI_JFA_LAST_C2_LOS_USED", 0)),
            "roi_stamping_mode": str(roi_stamping_mode),
            "roi_sparse_voxels": bool(roi_sparse_voxels),
            "roi_radii_mode": str(radius_mode),
            "roi_radius_override": "" if radius_override is None else float(radius_override),
            "gpu_face_split_used": bool(gpu_split_meta.get("used", False)),
            "gpu_face_split_iters": int(gpu_split_meta.get("iters", 0) or 0),
            "gpu_face_split_verified": bool(gpu_split_meta.get("verified", False)),
            "roi_tstamp_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TSTAMP_WALL", 0.0)),
            "roi_tradii_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TRADII_WALL", 0.0)),
            "roi_tcand_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TCAND_WALL", 0.0)),
            "roi_tinit_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TINIT_WALL", 0.0)),
            "roi_tbubble_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TBUBBLE_WALL", 0.0)),
            "roi_tfilter_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TFILTER_WALL", 0.0)),
            "roi_tdecode_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TDECODE_WALL", 0.0)),
            "roi_tc2_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TC2_WALL", 0.0)),
            "roi_c2_count": 0 if (regular_hotpath_available or use_exact_frontier) else int(getattr(backend, "ROI_JFA_LAST_C2_COUNT", 0)),
            "roi_bubble_iters": 0 if (regular_hotpath_available or use_exact_frontier) else int(getattr(backend, "ROI_JFA_LAST_BUBBLE_ITERS", 0)),
            "roi_max_radius": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_MAX_RADIUS", 0.0)),
            "roi_tjfa_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TJFA_WALL", 0.0)),
            "roi_tclose_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TCLOSE_WALL", 0.0)),
            "roi_trelax_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TRELAX_WALL", 0.0)),
            "roi_tactive_final_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TAFINAL_WALL", 0.0)),
            "roi_tactive_final_gpu_s": 0.0 if (regular_hotpath_available or use_exact_frontier) else float(getattr(backend, "ROI_JFA_LAST_TAFINAL_GPU_TIME", 0.0)),
            "roi_tpred_s": (
                float(fast_lut_s + fast_hotpath_s)
                if regular_hotpath_available else
                (0.0 if use_exact_frontier else float(getattr(backend, "ROI_JFA_LAST_TPRED_WALL", 0.0)))
            ),
            "exact_frontier_tinit_s": float(getattr(backend, "EXACT_FRONTIER_LAST_TINIT_WALL", 0.0)) if use_exact_frontier else 0.0,
            "exact_frontier_titer_s": float(getattr(backend, "EXACT_FRONTIER_LAST_TITER_WALL", 0.0)) if use_exact_frontier else 0.0,
            "exact_frontier_tpred_s": float(getattr(backend, "EXACT_FRONTIER_LAST_WALL_TIME", 0.0)) if use_exact_frontier else 0.0,
            "exact_frontier_iters": int(getattr(backend, "EXACT_FRONTIER_LAST_ITERS", 0)) if use_exact_frontier else 0,
            "exact_frontier_max_frontier": int(getattr(backend, "EXACT_FRONTIER_LAST_MAX_FRONTIER", 0)) if use_exact_frontier else 0,
            "exact_frontier_max_dist": int(getattr(backend, "EXACT_FRONTIER_LAST_MAX_DIST", 0)) if use_exact_frontier else 0,
        }
        ns["_cmame_last_roi_meta"] = dict(meta)
        progress_env = str(os.environ.get("CMAME_MAIN7_PROGRESS", "1")).lower().strip()
        if progress_env not in {"0", "false", "no", "off"}:
            progress_label = str(getattr(cfg, "cmame_progress_label", "partition"))
            print(
                f"[MAIN7][{selected_label_mode}] {progress_label}: "
                f"S={n_seed} N_fl={n_fl:,} N_cv={int(geom.n_cells)} "
                f"seed={1000.0 * t_seed:.2f}ms "
                f"label={1000.0 * t_label:.2f}ms "
                f"split={1000.0 * t_split:.2f}ms "
                f"geom={1000.0 * t_mom:.2f}ms "
                f"gpu_split_iters={int(meta['gpu_face_split_iters'])}",
                flush=True,
            )
        return geom, meta

    ns["cmame_build_geometry_from_seed_flat_timed"] = wrapped
    installed = str(os.environ.get("CMAME_LABEL_BACKEND", "roi_jfa")).lower().strip()
    print(f"[MAIN7] label backend: {installed} installed for seeded coarse partitions")
    return "exact_frontier_gpu" if installed == "exact_frontier_gpu" else "roi_jfa"


def install_geodesic_face_operator(ns: dict[str, Any], *, mode: str) -> None:
    mode_key = str(mode or "geodesic_face").lower().strip()
    if mode_key in {"", "euclidean", "default", "off", "none"}:
        ns["_cmame_face_operator_mode"] = "euclidean"
        print("[MAIN7] face operator: Euclidean/default geometry")
        return
    if mode_key not in {"geodesic_face", "graph_geodesic_face"}:
        raise ValueError(f"Unknown face operator mode: {mode}")

    cp = ns["cp"]
    original_build = ns["cmame_build_geometry_from_seed_flat_timed"]
    if getattr(original_build, "_cmame_geodesic_face_operator", False):
        ns["_cmame_face_operator_mode"] = "geodesic_face"
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
        metric = build_geodesic_face_metric(ns, geom, cfg)
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
                "face_operator": "geodesic_face",
                "operator_closure": "geodesic_face_operator",
                "operator_closure_s": float(time.perf_counter() - op_t0),
                "darcy_readout_length": "physical_axis_flux_readout",
            }
        )
        meta.update(op_meta)
        ns["_cmame_last_physical_dvec_readout"] = physical_dvec
        ns["_cmame_last_face_operator_meta"] = dict(op_meta)
        return geom, meta

    wrapped_build._cmame_geodesic_face_operator = True  # type: ignore[attr-defined]
    wrapped_build.__name__ = getattr(original_build, "__name__", "cmame_build_geometry_from_seed_flat_timed")
    wrapped_build.__doc__ = getattr(original_build, "__doc__", None)
    ns["cmame_build_geometry_from_seed_flat_timed"] = wrapped_build

    original_run = ns["cmame_run_flow_case_seeded"]
    if not getattr(original_run, "_cmame_physical_flux_readout", False):

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
            physical_dvec = ns.get("_cmame_last_physical_dvec_readout")
            op_meta = ns.get("_cmame_last_face_operator_meta", {})
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
            if "cmame_quality_flags" in ns:
                row = ns["cmame_quality_flags"](row)
            if export_data and out is not None:
                pseudo_stride = tuple(seed_spec.get("stride_zyx", (0, 0, 0))) if seed_spec.get("stride_zyx", None) else (0, 0, 0)
                products = ns["cmame_save_flow_products"](
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

        wrapped_run._cmame_physical_flux_readout = True  # type: ignore[attr-defined]
        wrapped_run.__name__ = getattr(original_run, "__name__", "cmame_run_flow_case_seeded")
        wrapped_run.__doc__ = getattr(original_run, "__doc__", None)
        ns["cmame_run_flow_case_seeded"] = wrapped_run

    ns["_cmame_face_operator_mode"] = "geodesic_face"
    print("[MAIN7] face operator: graph-geodesic cell-cell exchange with physical-axis Darcy readout")


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
        if original is None or getattr(original, "_cmame_profiled", False):
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

        profiled._cmame_profiled = True  # type: ignore[attr-defined]
        profiled.__name__ = getattr(original, "__name__", name)
        profiled.__doc__ = getattr(original, "__doc__", None)
        ns[name] = profiled

    for fname in [
        "cmame_build_reference",
        "cmame_run_flow_case_seeded",
        "cmame_build_geometry_from_seed_flat_timed",
        "cmame_build_geometry_from_labels_ncells_gpu",
        "cmame_face_connected_reindex_cpu",
        "cmame_zero_area_candidate_count_cpu",
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
        "cmame_flux_rel_error",
        "cmame_save_flow_products",
    ]:
        wrap(fname)

    ns["_cmame_runtime_profile"] = profile
    ns["_cmame_runtime_profile_sync"] = bool(synchronize)
    print(f"[MAIN7] runtime profile enabled; synchronize={bool(synchronize)}")
    return profile


def install_resolution_aware_wall_closure(ns: dict[str, Any], args: argparse.Namespace) -> None:
    """Optionally replace scalar wall-beta scaling by a size-aware closure."""
    mode = str(getattr(args, "wall_closure_mode", "fixed_beta") or "fixed_beta").lower().strip()
    if mode in {"", "fixed", "fixed_beta", "scalar", "scalar_beta"}:
        ns["_cmame_wall_closure_mode"] = "fixed_beta"
        ns["_cmame_wall_closure_params"] = {}
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
        ns["_cmame_last_wall_closure_meta"] = {
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
    ns["_cmame_wall_closure_mode"] = mode
    ns["_cmame_wall_closure_params"] = {
        "wall_ref_ccomp": float(ref_ccomp),
        "wall_size_exponent": float(comp_exponent),
        "wall_length_exponent": float(length_exponent),
        "wall_scale_min": float(scale_min),
        "wall_scale_max": float(scale_max),
    }
    print(
        "[MAIN7] resolution-aware wall closure installed: "
        f"mode={mode} eta_w={length_exponent:g} comp_exponent={comp_exponent:g} ref_ccomp={ref_ccomp:g} "
        f"clip=[{scale_min:g},{scale_max:g}]"
    )


def runtime_profile_summary(ns: dict[str, Any]) -> dict[str, Any]:
    profile = ns.get("_cmame_runtime_profile")
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
        "synchronized_timing": bool(ns.get("_cmame_runtime_profile_sync", False)),
        "note": "Nested timings are intentionally inclusive and may double count child stages.",
        "inclusive_seconds_sum": total,
        "stages": rows,
    }


def build_case_specs(ns: dict[str, Any], profile: str, bentheimer_npz: Path, fibrous_npz: Path) -> list[dict[str, Any]]:
    plan = ns["cmame_half_gfps_case_plan"](profile)
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
        return ns["cmame_make_mask_gpu"](spec["case"], tuple(spec["shape_zyx"]))
    if spec["mask_kind"] == "npz_mask":
        return load_mask_npz(ns, Path(spec["mask_npz"]))
    raise ValueError(f"Unknown mask kind: {spec['mask_kind']}")


def summarize_case_specs(ns: dict[str, Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_specs = ns["cmame_half_gfps_seed_specs_for_density"]
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
    if "cmame_clone_cfg" in ns:
        return ns["cmame_clone_cfg"](cfg, **updates)
    data = {k: getattr(cfg, k) for k in ns["PB615Config"].__dataclass_fields__.keys()}
    data.update(updates)
    return ns["PB615Config"](**data)


def apply_common_cfg_overrides(cfg, args: argparse.Namespace, *, prefix: str):
    dt = getattr(args, f"{prefix}_dt", None)
    n_steps = getattr(args, f"{prefix}_n_steps", None)
    report_every = getattr(args, f"{prefix}_report_every", None)
    initial_velocity_mode = getattr(args, f"{prefix}_initial_velocity_mode", None)
    if dt is not None:
        cfg.dt = float(dt)
    if n_steps is not None:
        cfg.n_steps = int(n_steps)
    if report_every is not None:
        cfg.report_every = int(report_every)
    if initial_velocity_mode:
        cfg.initial_velocity_mode = str(initial_velocity_mode)
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


def run_main7(ns: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cp = ns["cp"]
    profile = str(args.profile).lower()
    out = Path(args.out_dir).resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ns["cmame_ensure_dirs"](out)
    base_cfg = ns["cmame_base_cfg"](out_dir=str(out), profile=profile)
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
    setattr(coarse_template_cfg, "linear_solver_mode", str(getattr(args, "coarse_linear_solver", "cg")))
    setattr(coarse_template_cfg, "momentum_solver_mode", str(getattr(args, "coarse_momentum_solver", "cg")))
    setattr(coarse_template_cfg, "projection_interval", max(1, int(getattr(args, "coarse_projection_interval", 1))))
    plan = ns["cmame_half_gfps_case_plan"](profile)
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
        print(f"[MAIN7] wall calibration skipped; fixed beta_star={beta_star:g}")
    else:
        print("[MAIN7] wall calibration")
        wall = ns["cmame_run_wall_calibration"](ref_cfg, wall_plan, out)
        beta_star = float(wall["beta_star"])
        print(f"[MAIN7] beta_star={beta_star:g}")

    specs = filter_case_specs(
        build_case_specs(ns, profile, Path(args.bentheimer_npz), Path(args.fibrous_npz)),
        args,
    )
    ref_rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    spec_rows = summarize_case_specs(ns, specs)
    write_csv(out / "main7_case_spec_preflight.csv", spec_rows)
    flow_units_per_stride = max(
        1,
        len(face_modes)
        * len(initial_velocity_modes)
        * len(projection_intervals)
        * len(dt_values)
        * len(n_step_values),
    )
    total_units = sum(1 + len(spec["strides"]) * flow_units_per_stride for spec in specs)
    progress = Main7Progress(
        enabled=progress_enabled_from_args(args),
        total_units=total_units,
        every_s=float(getattr(args, "progress_every_s", 30.0) or 0.0),
    )
    progress.emit(
        f"[MAIN7][progress] plan: cases={len(specs)}, "
        f"references={len(specs)}, coarse_runs={total_units - len(specs)}, "
        f"heartbeat={format_duration(progress.every_s)}"
    )

    for spec in specs:
        case = spec["case"]
        print(f"\n[MAIN7] reference for {case}")
        mask = make_mask(ns, spec)
        stage_label = f"reference {case}"
        stage_t0 = progress.begin(stage_label)
        with progress.heartbeat(stage_label, stage_t0):
            ref_geom, ref_res, ref_row = ns["cmame_build_reference"](case, mask, ref_cfg, out=out)
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
            candidates = ns["cmame_half_gfps_seed_specs_for_density"](mask, tuple(stride), ["half"])
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
                                setattr(flow_cfg, "velocity_reconstruct_from_flux", bool(getattr(args, "coarse_velocity_reconstruct_from_flux", False)))
                                if getattr(args, "coarse_velocity_reconstruction_lambda", None) is not None:
                                    setattr(flow_cfg, "velocity_reconstruction_lambda", float(args.coarse_velocity_reconstruction_lambda))
                                run_tag = f"main7_{seed_spec['seed_id']}"
                                if (
                                    face_mode != "overrelaxed_default"
                                    or str(initial_velocity_mode) != str(base_cfg.initial_velocity_mode)
                                    or int(projection_interval) != 1
                                    or abs(float(dt) - float(base_cfg.dt)) > 1.0e-15
                                    or int(n_steps) != int(base_cfg.n_steps)
                                    or bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False))
                                ):
                                    run_tag += (
                                        f"__{face_mode}"
                                        f"__ivm{token_text(str(initial_velocity_mode))}"
                                        f"__pint{int(projection_interval)}"
                                        f"__dt{token_float(float(dt))}__n{int(n_steps)}"
                                    )
                                    if bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False)):
                                        run_tag += f"__vrecflux_lam{token_float(float(getattr(flow_cfg, 'velocity_reconstruction_lambda', 0.0)))}"
                                print(
                                    f"[MAIN7] {case} {run_tag} S={int(seed_flat.size)} "
                                    f"face={face_mode} ivm={str(flow_cfg.initial_velocity_mode)} "
                                    f"pint={int(projection_interval)} "
                                    f"dt={float(flow_cfg.dt):g} n_steps={int(flow_cfg.n_steps)}"
                                )
                                stage_label = f"{case} {run_tag}"
                                setattr(flow_cfg, "cmame_progress_label", stage_label)
                                stage_t0 = progress.begin(stage_label)
                                with progress.heartbeat(stage_label, stage_t0):
                                    row, _geom, _res = ns["cmame_run_flow_case_seeded"](
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
                                        panel_hint="main7_density",
                                    )
                                row["paper_case"] = spec["paper_case"]
                                row["mask_kind"] = spec["mask_kind"]
                                row["main_text_case"] = bool(spec["main_text_case"])
                                row["mask_npz"] = spec.get("mask_npz", "")
                                row["cfg_dt"] = float(flow_cfg.dt)
                                row["cfg_n_steps"] = int(flow_cfg.n_steps)
                                row["cfg_enable_convection"] = bool(flow_cfg.enable_convection)
                                row["initial_velocity_mode"] = str(flow_cfg.initial_velocity_mode)
                                row["momentum_solver_mode"] = str(getattr(flow_cfg, "momentum_solver_mode", "cg"))
                                row["projection_interval"] = int(projection_interval)
                                row["velocity_reconstruct_from_flux"] = bool(getattr(flow_cfg, "velocity_reconstruct_from_flux", False))
                                row["velocity_reconstruction_lambda"] = float(getattr(flow_cfg, "velocity_reconstruction_lambda", float("nan")))
                                row["wall_closure_mode"] = str(ns.get("_cmame_wall_closure_mode", "fixed_beta"))
                                wall_meta = ns.get("_cmame_last_wall_closure_meta", {})
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
                                roi_meta = ns.get("_cmame_last_roi_meta", {})
                                if isinstance(roi_meta, dict):
                                    for key in (
                                        "t_seed_s", "t_label_s", "t_split_s",
                                        "roi_tile_size", "roi_profile_gpu", "roi_cache_clearance", "roi_skip_face_split",
                                        "roi_active_final_enabled", "roi_active_final_iters",
                                        "roi_active_final_band_iters", "roi_active_final_halo_iters",
                                        "roi_c2_core", "roi_c2_margin", "roi_c2_los", "roi_c2_los_used",
                                        "roi_stamping_mode", "roi_sparse_voxels",
                                        "roi_radii_mode", "roi_radius_override",
                                        "gpu_face_split_used", "gpu_face_split_iters", "gpu_face_split_verified",
                                        "cuda_geometry_builders_used", "cuda_geometry_builder_mode",
                                        "roi_tstamp_s", "roi_tradii_s", "roi_tcand_s", "roi_tinit_s",
                                        "roi_tbubble_s", "roi_tfilter_s", "roi_tdecode_s",
                                        "roi_tc2_s", "roi_c2_count",
                                        "roi_bubble_iters", "roi_max_radius",
                                    "roi_tjfa_s", "roi_tclose_s", "roi_trelax_s",
                                    "roi_tactive_final_s", "roi_tactive_final_gpu_s", "roi_tpred_s",
                                    "exact_frontier_tinit_s", "exact_frontier_titer_s",
                                    "exact_frontier_tpred_s", "exact_frontier_iters",
                                    "exact_frontier_max_frontier", "exact_frontier_max_dist",
                                    ):
                                        if key in roi_meta:
                                            row[key] = roi_meta[key]
                                progress.complete(
                                    stage_label,
                                    stage_t0,
                                    extra=coarse_row_progress_extra(row),
                                )
                                density_rows.append(row)

        del mask
        cp.get_default_memory_pool().free_all_blocks()

    write_csv(out / "cmame_reference_rows_all.csv", ref_rows)
    write_csv(out / "cmame_seed_density_sweep.csv", density_rows)
    write_csv(out / "cmame_flow_sweep_abc.csv", density_rows)
    final_rows = choose_final_rows(density_rows)
    write_csv(out / "table5_final_accuracy_cost_summary.csv", final_rows)
    write_csv(out / "table_main7_final_accuracy_cost_summary.csv", final_rows)

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
            "mode": str(ns.get("_cmame_wall_closure_mode", "fixed_beta")),
            "params": ns.get("_cmame_wall_closure_params", {}),
        },
        "reference_cfg": {
            "dt": float(ref_cfg.dt),
            "n_steps": int(ref_cfg.n_steps),
            "enable_convection": bool(ref_cfg.enable_convection),
            "initial_velocity_mode": str(ref_cfg.initial_velocity_mode),
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
            "body_force": [float(v) for v in coarse_template_cfg.body_force],
            "wall_distance_floor": float(coarse_template_cfg.wall_distance_floor),
            "pressure_gauge_eps": float(coarse_template_cfg.pressure_gauge_eps),
            "transmissibility_floor": float(coarse_template_cfg.transmissibility_floor),
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
            "steady_scalar_initial_guess": bool(getattr(args, "steady_scalar_initial_guess", False)),
            "steady_scalar_solver": str(getattr(args, "steady_scalar_solver", "")),
            "coarse_velocity_reconstruct_from_flux": bool(getattr(args, "coarse_velocity_reconstruct_from_flux", False)),
            "coarse_velocity_reconstruction_lambda": (
                None
                if getattr(args, "coarse_velocity_reconstruction_lambda", None) is None
                else float(args.coarse_velocity_reconstruction_lambda)
            ),
            "roi_jfa_tile": str(getattr(args, "roi_jfa_tile", "") or os.environ.get("CMAME_ROIJFA_TILE", "8,8,16")),
            "roi_jfa_profile": bool(getattr(args, "roi_jfa_profile", False)),
            "roi_jfa_skip_face_split": bool(getattr(args, "roi_jfa_skip_face_split", False)),
            "roi_jfa_active_list": str(os.environ.get("CMAME_ROIJFA_ACTIVE_LIST", "1")).lower().strip() not in {"0", "false", "no", "off"},
            "roi_jfa_active_list_threshold": float(os.environ.get("CMAME_ROIJFA_ACTIVE_LIST_THRESHOLD", "0.75")),
            "roi_jfa_active_final_iters": int(os.environ.get("CMAME_ROIJFA_ACTIVE_FINAL_ITERS", "0") or 0),
            "roi_jfa_active_final_band_iters": int(os.environ.get("CMAME_ROIJFA_ACTIVE_FINAL_BAND_ITERS", "1") or 1),
            "roi_jfa_active_final_halo_iters": int(os.environ.get("CMAME_ROIJFA_ACTIVE_FINAL_HALO_ITERS", "1") or 1),
            "roi_jfa_c2_core": str(os.environ.get("CMAME_ROIJFA_C2_CORE", "0")).lower().strip() in {"1", "true", "yes", "on"},
            "roi_jfa_c2_margin": float(os.environ.get("CMAME_ROIJFA_C2_MARGIN", "1.0e-6")),
            "roi_jfa_c2_los": str(os.environ.get("CMAME_ROIJFA_C2_LOS", "0")).lower().strip() in {"1", "true", "yes", "on"},
            "roi_jfa_stamping_mode": str(os.environ.get("CMAME_ROIJFA_STAMPING_MODE", "")),
            "roi_jfa_sparse_voxels": str(os.environ.get("CMAME_ROIJFA_SPARSE_VOXELS", "0")).lower().strip() in {"1", "true", "yes", "on"},
            "gpu_face_split": bool(getattr(args, "gpu_face_split", False)),
            "gpu_face_split_verify": bool(getattr(args, "gpu_face_split_verify", False)),
            "cuda_geometry_builders": bool(ns.get("_cmame_cuda_geometry_builders", {}).get("used", False)),
            "cuda_geometry_builder_mode": str(ns.get("_cmame_cuda_geometry_builders", {}).get("mode", "")),
            "roi_jfa_relax_after": int(getattr(args, "roi_jfa_relax_after", -1)),
        },
        "case_specs": specs,
        "n_reference_rows": len(ref_rows),
        "n_density_rows": len(density_rows),
        "n_final_rows": len(final_rows),
        "outputs": {
            "case_spec_preflight": str(out / "main7_case_spec_preflight.csv"),
            "reference_rows": str(out / "cmame_reference_rows_all.csv"),
            "density_sweep": str(out / "cmame_seed_density_sweep.csv"),
            "final_table": str(out / "table_main7_final_accuracy_cost_summary.csv"),
        },
    }
    profile_summary = runtime_profile_summary(ns)
    if profile_summary:
        profile_path = out / "runtime_profile.json"
        profile_path.write_text(json.dumps(profile_summary, indent=2), encoding="utf-8")
        manifest["outputs"]["runtime_profile"] = str(profile_path)
    (out / "main7_flow_production_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("CMAME_FLOW_PROFILE", "production"))
    ap.add_argument("--label-backend", default=os.environ.get("CMAME_LABEL_BACKEND", "roi_jfa"),
                    choices=["roi_jfa", "exact_geodesic", "exact_frontier_gpu"])
    ap.add_argument("--face-operator", default=os.environ.get("CMAME_FACE_OPERATOR", "geodesic_face"),
                    choices=["geodesic_face", "euclidean"],
                    help="Cell-cell face metric used by the production operator; wall closure remains separately controlled.")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT / "main7_flow_production")
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
    ap.add_argument("--wall-closure-mode", default=os.environ.get("CMAME_WALL_CLOSURE_MODE", "fixed_beta"),
                    choices=["fixed_beta", "global_comp_power", "cell_volume_power"],
                    help="Wall transmissibility scaling used after the scalar beta anchor.")
    ap.add_argument("--wall-size-exponent", type=float, default=float(os.environ.get("CMAME_WALL_SIZE_EXPONENT", "0.0")),
                    help="Compression-ratio exponent p for resolution-aware wall scaling; kept for compatibility.")
    ap.add_argument("--wall-length-exponent", type=float, default=(
        None if os.environ.get("CMAME_WALL_LENGTH_EXPONENT", "") == "" else float(os.environ["CMAME_WALL_LENGTH_EXPONENT"])
    ),
                    help="Length-scale wall-resolution exponent eta_w. If provided, p=eta_w/3.")
    ap.add_argument("--wall-ref-ccomp", type=float, default=float(os.environ.get("CMAME_WALL_REF_CCOMP", "64.0")),
                    help="Reference compression/cell-volume scale for size-aware wall closure.")
    ap.add_argument("--wall-scale-min", type=float, default=float(os.environ.get("CMAME_WALL_SCALE_MIN", "0.05")),
                    help="Lower clip for size-aware wall scale.")
    ap.add_argument("--wall-scale-max", type=float, default=float(os.environ.get("CMAME_WALL_SCALE_MAX", "20.0")),
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
                    help="Comma-separated coarse initial/solver modes to scan after one reference solve.")
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
    ap.add_argument("--roi-jfa-tile", default="",
                    help="ROI-JFA tile size as 'z,y,x'. Defaults to 8,8,16 in production runner.")
    ap.add_argument("--roi-jfa-profile", action="store_true",
                    help="Enable ROI-JFA CUDA-event profiling; disabled by default for production timing.")
    ap.add_argument("--roi-jfa-skip-face-split", action="store_true",
                    help="Skip the optional CPU face-connected relabel pass after ROI-JFA.")
    ap.add_argument("--roi-jfa-disable-active-list", action="store_true",
                    help="Disable adaptive launch-level active-list ROI-JFA tile kernels.")
    ap.add_argument("--roi-jfa-active-list-threshold", type=float, default=None,
                    help="Use active-list kernels only when active_tiles/nTiles is below this threshold.")
    ap.add_argument("--roi-jfa-active-final-iters", type=int, default=None,
                    help="Run tile-boundary active-front final ROI-JFA refinement for N iterations.")
    ap.add_argument("--roi-jfa-active-final-band-iters", type=int, default=None,
                    help="Tile-lattice boundary-band dilations for active-front final refinement.")
    ap.add_argument("--roi-jfa-active-final-halo-iters", type=int, default=None,
                    help="Tile-lattice dirty-halo dilations for active-front final refinement.")
    ap.add_argument("--roi-jfa-c2-core", action="store_true",
                    help="Enable certified second-competitor ROI core expansion.")
    ap.add_argument("--roi-jfa-c2-margin", type=float, default=None,
                    help="Safety margin for certified second-competitor ROI core expansion.")
    ap.add_argument("--roi-jfa-c2-los", action="store_true",
                    help="Use cached 27-direction LOS-kmax tables for the C2 canonical-path certificate.")
    ap.add_argument("--roi-jfa-stamping-mode", default="",
                    help="Override ROI-JFA stamping mode, e.g. D_c2_geodesic_ball.")
    ap.add_argument("--roi-jfa-sparse-voxels", action="store_true",
                    help="Use sparse voxel-list execution for the certified ROI set.")
    ap.add_argument("--roi-jfa-regular-stride-hotpath", action="store_true",
                    help="Use the clearance-aware regular-stride CUDA hot path when the seed protocol and graph allow it.")
    ap.add_argument("--roi-jfa-regular-stride-hotpath-verify", action="store_true",
                    help="Verify the regular-stride hot path against exact GPU frontier labels and fail on mismatch.")
    ap.add_argument("--gpu-face-split", action="store_true",
                    help="Use GPU 3D face-connected component relabeling instead of the CPU pass.")
    ap.add_argument("--gpu-face-split-verify", action="store_true",
                    help="Compare GPU face split against the original CPU pass and fail on mismatch.")
    ap.add_argument("--cuda-geometry-builders", action="store_true",
                    help="Use experimental RawKernel geometry builders; leave off for SI-audited scientific runs.")
    ap.add_argument("--roi-jfa-relax-after", type=int, default=-1,
                    help="Override ROI-JFA final local relax count.")
    ap.add_argument("--no-export-data", action="store_true",
                    help="Skip heavy per-run NumPy/CSV products during parameter scans.")
    ap.add_argument("--skip-zero-area-diagnostic", action="store_true",
                    help="Skip the optional CPU zero-area diagnostic; reports N_0A=-1.")
    ap.add_argument("--runtime-profile", action="store_true",
                    help="Write inclusive timing diagnostics for major GPU/CPU pipeline stages.")
    ap.add_argument("--runtime-profile-no-sync", action="store_true",
                    help="Do not synchronize around profiled GPU calls. Faster but less exact.")
    ap.add_argument("--progress-every-s", type=float,
                    default=float(os.environ.get("CMAME_MAIN7_PROGRESS_EVERY_S", "30")),
                    help="Heartbeat interval for lightweight stage progress and ETA printing.")
    ap.add_argument("--quiet-progress", action="store_true",
                    help="Disable lightweight stage progress, ETA, and ROI summary lines.")
    args = ap.parse_args()
    if bool(args.quiet_progress):
        os.environ["CMAME_MAIN7_PROGRESS"] = "0"
    else:
        os.environ.setdefault("CMAME_MAIN7_PROGRESS", "1")
    os.environ["CMAME_MAIN7_PROGRESS_EVERY_S"] = str(max(0.0, float(args.progress_every_s)))
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
    if args.roi_jfa_tile:
        os.environ["CMAME_ROIJFA_TILE"] = str(args.roi_jfa_tile)
    else:
        os.environ.setdefault("CMAME_ROIJFA_TILE", "8,8,16")
    os.environ["CMAME_ROIJFA_PROFILE"] = "1" if bool(args.roi_jfa_profile) else os.environ.get("CMAME_ROIJFA_PROFILE", "0")
    os.environ["CMAME_ROIJFA_STAMP_PROFILE"] = "1" if bool(args.roi_jfa_profile) else os.environ.get("CMAME_ROIJFA_STAMP_PROFILE", "0")
    if args.roi_jfa_skip_face_split:
        os.environ["CMAME_ROIJFA_SKIP_SPLIT"] = "1"
    os.environ["CMAME_ROIJFA_ACTIVE_LIST"] = "0" if bool(args.roi_jfa_disable_active_list) else os.environ.get("CMAME_ROIJFA_ACTIVE_LIST", "1")
    if args.roi_jfa_active_list_threshold is not None:
        os.environ["CMAME_ROIJFA_ACTIVE_LIST_THRESHOLD"] = str(float(args.roi_jfa_active_list_threshold))
    else:
        os.environ.setdefault("CMAME_ROIJFA_ACTIVE_LIST_THRESHOLD", "0.75")
    if args.roi_jfa_active_final_iters is not None:
        os.environ["CMAME_ROIJFA_ACTIVE_FINAL_ITERS"] = str(max(0, int(args.roi_jfa_active_final_iters)))
    else:
        os.environ.setdefault("CMAME_ROIJFA_ACTIVE_FINAL_ITERS", "0")
    if args.roi_jfa_active_final_band_iters is not None:
        os.environ["CMAME_ROIJFA_ACTIVE_FINAL_BAND_ITERS"] = str(max(0, int(args.roi_jfa_active_final_band_iters)))
    else:
        os.environ.setdefault("CMAME_ROIJFA_ACTIVE_FINAL_BAND_ITERS", "1")
    if args.roi_jfa_active_final_halo_iters is not None:
        os.environ["CMAME_ROIJFA_ACTIVE_FINAL_HALO_ITERS"] = str(max(0, int(args.roi_jfa_active_final_halo_iters)))
    else:
        os.environ.setdefault("CMAME_ROIJFA_ACTIVE_FINAL_HALO_ITERS", "1")
    if bool(args.roi_jfa_c2_core):
        os.environ["CMAME_ROIJFA_C2_CORE"] = "1"
        os.environ.setdefault("CMAME_ROIJFA_STAMPING_MODE", "D_c2_geodesic_ball")
    else:
        os.environ.setdefault("CMAME_ROIJFA_C2_CORE", "0")
    if args.roi_jfa_c2_margin is not None:
        os.environ["CMAME_ROIJFA_C2_MARGIN"] = str(float(args.roi_jfa_c2_margin))
    else:
        os.environ.setdefault("CMAME_ROIJFA_C2_MARGIN", "1.0e-6")
    if bool(args.roi_jfa_c2_los):
        os.environ["CMAME_ROIJFA_C2_LOS"] = "1"
    else:
        os.environ.setdefault("CMAME_ROIJFA_C2_LOS", "0")
    if args.roi_jfa_stamping_mode:
        os.environ["CMAME_ROIJFA_STAMPING_MODE"] = str(args.roi_jfa_stamping_mode)
    if bool(args.roi_jfa_sparse_voxels):
        os.environ["CMAME_ROIJFA_SPARSE_VOXELS"] = "1"
    else:
        os.environ.setdefault("CMAME_ROIJFA_SPARSE_VOXELS", "0")
    if bool(args.roi_jfa_regular_stride_hotpath):
        os.environ["CMAME_ROIJFA_REGULAR_STRIDE_HOTPATH"] = "1"
    else:
        os.environ.setdefault("CMAME_ROIJFA_REGULAR_STRIDE_HOTPATH", "0")
    if bool(args.roi_jfa_regular_stride_hotpath_verify):
        os.environ["CMAME_ROIJFA_REGULAR_STRIDE_HOTPATH_VERIFY"] = "1"
    else:
        os.environ.setdefault("CMAME_ROIJFA_REGULAR_STRIDE_HOTPATH_VERIFY", "0")
    if int(args.roi_jfa_relax_after) >= 0:
        os.environ["CMAME_ROIJFA_RELAX_AFTER"] = str(int(args.roi_jfa_relax_after))

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
    if any(mode in MONOLITHIC_STOKES_SOLVER_MODES for mode in coarse_modes):
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
    os.environ["CMAME_LABEL_BACKEND"] = str(args.label_backend)
    backend = "exact_geodesic"
    if args.label_backend in {"roi_jfa", "exact_frontier_gpu"}:
        backend = install_roi_backend(ns, FLOW_NOTEBOOK.parent)
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
        write_csv(out / "main7_case_spec_preflight.csv", rows)
        print(json.dumps({
            "dry_run": True,
            "label_backend": backend,
            "out_dir": str(out),
            "case_rows": rows,
        }, indent=2))
        return

    manifest = run_main7(ns, args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()




