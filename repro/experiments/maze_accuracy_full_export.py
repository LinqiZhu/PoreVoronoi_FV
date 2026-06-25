from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


from _paths import REPO_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
BASE = REPO_ROOT
TOOLS = SRC_ROOT
NOTEBOOK_DIR = BASE / "notebooks"
OUT_DIR = ROOT / "outputs" / "maze_accuracy_full_export_2026-06-06"
PKG = ROOT / "outputs" / "paper_ready_data_package_2026-06-06"
FIG = ROOT / "outputs" / "figure_ready_data_package_2026-06-06"

sys.path.insert(0, str(TOOLS))
import run_main7_flow_production as runner  # noqa: E402


CASE = "C_maze"
WALL_BETA = 1.25
STRIDE = (2, 2, 4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_reference_cfg(ns: dict[str, Any]) -> Any:
    cfg = ns["cmame_base_cfg"](out_dir=str(OUT_DIR), profile="production")
    cfg = runner.clone_cfg(ns, cfg, out_dir=str(OUT_DIR))
    cfg.dt = 2.0
    cfg.n_steps = int(os.environ.get("CMAME_MAZE_ACCURACY_REF_STEPS", "900"))
    cfg.report_every = max(1, int(cfg.n_steps))
    cfg.enable_convection = False
    cfg.initial_velocity_mode = "zero"
    setattr(cfg, "linear_solver_mode", "cg")
    setattr(cfg, "momentum_solver_mode", "cg")
    setattr(cfg, "projection_interval", 1)
    return cfg


def make_coarse_cfg(ns: dict[str, Any], solver_mode: str, momentum_mode: str) -> Any:
    cfg = ns["cmame_base_cfg"](out_dir=str(OUT_DIR), profile="production")
    cfg = runner.clone_cfg(
        ns,
        cfg,
        out_dir=str(OUT_DIR),
        transmissibility_ratio_clip=ns["_pb618_face_clip_for_mode"]("overrelaxed_default"),
        explicit_nonorthogonal_correction=0.0,
    )
    cfg.dt = 5.0
    cfg.n_steps = int(os.environ.get("CMAME_MAZE_ACCURACY_NSTEPS", "100"))
    cfg.report_every = max(1, min(20, int(cfg.n_steps)))
    cfg.enable_convection = False
    cfg.initial_velocity_mode = "zero"
    cfg.pressure_gauge_eps = float(os.environ.get("CMAME_MAZE_ACCURACY_GAUGE", "1.0e-12"))
    setattr(cfg, "linear_solver_mode", solver_mode)
    setattr(cfg, "momentum_solver_mode", momentum_mode)
    setattr(cfg, "projection_interval", 1)
    return cfg


def half_seed(ns: dict[str, Any], mask):
    candidates = ns["cmame_half_gfps_seed_specs_for_density"](mask, STRIDE, ["half"])
    candidates = [
        item for item in candidates
        if str(item[0].get("family", "")).lower() == "stride_half_admissible"
    ]
    if not candidates:
        raise RuntimeError(f"No half-offset candidate for {STRIDE}")
    return candidates[0]


def install_solver_hooks(ns: dict[str, Any], solver_mode: str, momentum_mode: str) -> None:
    mode = solver_mode.lower().strip()
    if mode in {"dense_lu", "gpu_dense_lu", "lu"}:
        runner.install_dense_lu_linear_solver(ns)
    if mode in {"direct", "spsolve"}:
        runner.install_direct_sparse_linear_solver(ns)
    if momentum_mode.lower().strip() in {"block_cg", "multi_rhs_cg", "spmm_cg"}:
        runner.install_block_momentum_cg(ns)


def install_debug_wrappers(ns: dict[str, Any]) -> None:
    cp = ns["cp"]

    def wrap(name: str) -> None:
        original = ns.get(name)
        if original is None or getattr(original, "_maze_accuracy_debug_wrapped", False):
            return

        def wrapped(*args, **kwargs):
            print(f"[maze-debug] before {name}", flush=True)
            cp.cuda.Stream.null.synchronize()
            out = original(*args, **kwargs)
            cp.cuda.Stream.null.synchronize()
            print(f"[maze-debug] after {name}", flush=True)
            return out

        wrapped._maze_accuracy_debug_wrapped = True  # type: ignore[attr-defined]
        ns[name] = wrapped

    for fname in [
        "make_initial_velocity_gpu",
        "face_flux_from_velocity_gpu",
        "build_implicit_velocity_matrix_gpu",
        "_make_jacobi_preconditioner_gpu",
        "diffusion_stiffness_number_gpu",
        "implicit_momentum_predictor_gpu",
        "lsq_gradient_gpu",
        "convection_term_gpu",
        "nonorthogonal_diffusion_correction_gpu",
        "solve_pressure_correction_gpu",
        "face_divergence_gpu",
        "correct_flux_gpu",
        "steady_momentum_residual_gpu",
        "coarsen_voxel_reference_to_coarse_gpu",
        "cmame_weighted_rel_l2_vec_gpu",
        "cmame_phi_ref_on_coarse_faces_cpu",
        "cmame_flux_rel_error",
        "cmame_save_flow_products",
    ]:
        wrap(fname)
    print("[maze-debug] projection debug wrappers installed", flush=True)


def install_safe_flux_error(ns: dict[str, Any]) -> None:
    cp = ns["cp"]

    def safe_flux_rel_error(ref_geom, ref_result, coarse_geom, coarse_result) -> float:
        print("[maze-safe-flux] aggregating reference flux", flush=True)
        phi_ref = ns["cmame_phi_ref_on_coarse_faces_cpu"](ref_geom, ref_result, coarse_geom)
        print(f"[maze-safe-flux] phi_ref length={len(phi_ref)}", flush=True)
        phi_gpu = coarse_result["phi"]
        cp.cuda.Stream.null.synchronize()
        phi = phi_gpu.get().astype(float, copy=False)
        print(f"[maze-safe-flux] phi length={len(phi)}", flush=True)
        if phi.shape != phi_ref.shape:
            raise ValueError(f"flux length mismatch: coarse={phi.shape}, ref={phi_ref.shape}")
        print(
            f"[maze-safe-flux] finite coarse={bool(np.isfinite(phi).all())} "
            f"finite ref={bool(np.isfinite(phi_ref).all())}",
            flush=True,
        )
        diff = phi - phi_ref
        den = float(np.sqrt(np.sum(phi_ref * phi_ref, dtype=np.float64)))
        num = float(np.sqrt(np.sum(diff * diff, dtype=np.float64)))
        print(f"[maze-safe-flux] num={num:.6e} den={den:.6e}", flush=True)
        return float(num / max(float(den), 1.0e-300))

    ns["cmame_flux_rel_error"] = safe_flux_rel_error
    print("[maze-safe-flux] safe flux-error override installed", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    solver_mode = os.environ.get("CMAME_MAZE_ACCURACY_SOLVER", "dense_lu").strip()
    momentum_mode = os.environ.get("CMAME_MAZE_ACCURACY_MOMENTUM", "cg").strip()
    export_data = os.environ.get("CMAME_MAZE_ACCURACY_EXPORT", "1").strip().lower() not in {"0", "false", "no"}

    os.environ["CMAME_ROIJFA_C2_CORE"] = "1"
    os.environ["CMAME_ROIJFA_C2_MARGIN"] = "1.0e-6"
    os.environ["CMAME_ROIJFA_SPARSE_VOXELS"] = "1"
    os.environ["CMAME_ROIJFA_STAMPING_MODE"] = "D_c2_geodesic_ball"
    os.environ["CMAME_ROIJFA_TILE"] = "8,8,16"
    os.environ.setdefault("CMAME_MAIN7_PROGRESS", "1")

    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    runner.install_runtime_cfg_attr_preservation(ns)
    install_solver_hooks(ns, solver_mode, momentum_mode)
    runner.install_skip_zero_area_diagnostic(ns)
    runner.install_gpu_face_connected_split(ns, verify=False)
    runner.install_roi_backend(ns, NOTEBOOK_DIR)
    install_safe_flux_error(ns)
    if os.environ.get("CMAME_MAZE_ACCURACY_DEBUG", "0").strip().lower() in {"1", "true", "yes"}:
        install_debug_wrappers(ns)

    specs = {
        str(spec["case"]): spec
        for spec in runner.build_case_specs(ns, "production", runner.BENTHEIMER_INPUT, runner.FIBROUS_INPUT)
    }
    spec = specs[CASE]
    mask = runner.make_mask(ns, spec)
    ref_cfg = make_reference_cfg(ns)
    coarse_cfg = make_coarse_cfg(ns, solver_mode, momentum_mode)

    print(
        f"[maze-accuracy] reference solve; coarse solver={solver_mode}, "
        f"momentum={momentum_mode}, stride={STRIDE}, "
        f"n_steps={coarse_cfg.n_steps}, export={export_data}",
        flush=True,
    )
    t0 = time.perf_counter()
    ref_geom, ref_res, _ref_row = ns["cmame_build_reference"](CASE, mask, ref_cfg, out=OUT_DIR)
    print(
        f"[maze-accuracy] reference K={float(ref_res['K_eff_x']):.8g} "
        f"time={time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    seed_spec, seed_flat = half_seed(ns, mask)
    run_tag = (
        f"maze_accuracy_stride_{'x'.join(map(str, STRIDE))}_half__"
        f"{solver_mode.lower().replace(' ', '_')}"
    )
    print(f"[maze-accuracy] run {run_tag} S={int(seed_flat.size)}", flush=True)
    row, _geom, _res = ns["cmame_run_flow_case_seeded"](
        CASE,
        mask,
        seed_flat,
        seed_spec,
        coarse_cfg,
        ref_geom,
        ref_res,
        wall_beta=WALL_BETA,
        face_mode="overrelaxed_default",
        run_tag=run_tag,
        out=OUT_DIR,
        export_data=export_data,
        panel_hint="maze_accuracy_full_export",
    )
    row["paper_case"] = spec["paper_case"]
    row["mask_kind"] = spec["mask_kind"]
    row["main_text_case"] = bool(spec["main_text_case"])
    row["cfg_dt"] = float(coarse_cfg.dt)
    row["cfg_n_steps"] = int(coarse_cfg.n_steps)
    row["cfg_enable_convection"] = bool(coarse_cfg.enable_convection)
    row["initial_velocity_mode"] = str(coarse_cfg.initial_velocity_mode)
    row["linear_solver_mode"] = str(getattr(coarse_cfg, "linear_solver_mode", "cg"))
    row["momentum_solver_mode"] = str(getattr(coarse_cfg, "momentum_solver_mode", "cg"))
    row["projection_interval"] = int(getattr(coarse_cfg, "projection_interval", 1))
    row["e_K_percent"] = 100.0 * float(row["e_K"])
    row["e_u_percent"] = 100.0 * float(row["e_u"])
    row["e_phi_percent"] = 100.0 * float(row["e_phi"])
    row["speedup_vs_ref"] = float(row["S_pipe"])
    row["full_export_scope"] = "full-field exported C_maze denser accuracy seed row"
    row["manuscript_use"] = (
        "main accuracy-cost maze repair row; retained balanced repair remains the faster trade-off"
    )

    rows = [row]
    suffix = "" if export_data and int(coarse_cfg.n_steps) == 100 else f"__n{int(coarse_cfg.n_steps)}__export{int(export_data)}"
    write_csv(OUT_DIR / f"maze_accuracy_full_export_rows{suffix}.csv", rows)
    if export_data and int(coarse_cfg.n_steps) == 100:
        write_csv(PKG / "maze_accuracy_full_export_rows.csv", rows)
        write_csv(FIG / "plot_maze_accuracy_full_export_rows.csv", rows)
    print(
        f"[maze-accuracy] done {run_tag}: eK={row['e_K_percent']:.2f}% "
        f"eu={row['e_u_percent']:.2f}% ephi={row['e_phi_percent']:.2f}% "
        f"speedup={row['speedup_vs_ref']:.1f}x total={float(row['t_total_s']):.3f}s",
        flush=True,
    )
    print(f"[maze-accuracy] wrote {OUT_DIR / f'maze_accuracy_full_export_rows{suffix}.csv'}", flush=True)


if __name__ == "__main__":
    main()



