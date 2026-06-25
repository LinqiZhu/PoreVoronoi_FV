from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any


from _paths import REPO_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
BASE = REPO_ROOT
TOOLS = SRC_ROOT
NOTEBOOK_DIR = BASE / "notebooks"
OUT_DIR = ROOT / "outputs" / "maze_repaired_full_export_2026-06-06"
PKG = ROOT / "outputs" / "paper_ready_data_package_2026-06-06"
FIG = ROOT / "outputs" / "figure_ready_data_package_2026-06-06"

sys.path.insert(0, str(TOOLS))
import run_main7_flow_production as runner  # noqa: E402


CASE = "C_maze"
WALL_BETA = 1.25
# The retained manuscript replacement row is the balanced repair.  The denser
# (2,2,4) accuracy-focused row is exported by maze_accuracy_full_export.py.
REPAIR_STRIDES = [(2, 3, 4)]


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
    cfg.n_steps = 900
    cfg.report_every = 900
    cfg.enable_convection = False
    cfg.initial_velocity_mode = "zero"
    setattr(cfg, "linear_solver_mode", "cg")
    setattr(cfg, "momentum_solver_mode", "cg")
    setattr(cfg, "projection_interval", 1)
    return cfg


def make_coarse_cfg(ns: dict[str, Any]) -> Any:
    cfg = ns["cmame_base_cfg"](out_dir=str(OUT_DIR), profile="production")
    cfg = runner.clone_cfg(
        ns,
        cfg,
        out_dir=str(OUT_DIR),
        transmissibility_ratio_clip=ns["_pb618_face_clip_for_mode"]("overrelaxed_default"),
        explicit_nonorthogonal_correction=0.0,
    )
    cfg.dt = 5.0
    cfg.n_steps = 100
    cfg.report_every = 100
    cfg.enable_convection = False
    cfg.initial_velocity_mode = "zero"
    setattr(cfg, "linear_solver_mode", "dense_lu")
    setattr(cfg, "momentum_solver_mode", "cg")
    setattr(cfg, "projection_interval", 1)
    return cfg


def half_seed(ns: dict[str, Any], mask, stride: tuple[int, int, int]):
    candidates = ns["cmame_half_gfps_seed_specs_for_density"](mask, stride, ["half"])
    candidates = [
        item for item in candidates
        if str(item[0].get("family", "")).lower() == "stride_half_admissible"
    ]
    if not candidates:
        raise RuntimeError(f"No half-offset candidate for {stride}")
    return candidates[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CMAME_ROIJFA_C2_CORE"] = "1"
    os.environ["CMAME_ROIJFA_C2_MARGIN"] = "1.0e-6"
    os.environ["CMAME_ROIJFA_SPARSE_VOXELS"] = "1"
    os.environ["CMAME_ROIJFA_STAMPING_MODE"] = "D_c2_geodesic_ball"
    os.environ["CMAME_ROIJFA_TILE"] = "8,8,16"
    os.environ.setdefault("CMAME_MAIN7_PROGRESS", "1")

    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    runner.install_runtime_cfg_attr_preservation(ns)
    runner.install_dense_lu_linear_solver(ns)
    runner.install_skip_zero_area_diagnostic(ns)
    runner.install_gpu_face_connected_split(ns, verify=False)
    runner.install_roi_backend(ns, NOTEBOOK_DIR)

    specs = {
        str(spec["case"]): spec
        for spec in runner.build_case_specs(ns, "production", runner.BENTHEIMER_INPUT, runner.FIBROUS_INPUT)
    }
    spec = specs[CASE]
    mask = runner.make_mask(ns, spec)
    ref_cfg = make_reference_cfg(ns)
    coarse_cfg = make_coarse_cfg(ns)

    print("[maze-full] reference solve", flush=True)
    t0 = time.perf_counter()
    ref_geom, ref_res, ref_row = ns["cmame_build_reference"](CASE, mask, ref_cfg, out=OUT_DIR)
    print(
        f"[maze-full] reference K={float(ref_res['K_eff_x']):.8g} "
        f"time={time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for stride in REPAIR_STRIDES:
        seed_spec, seed_flat = half_seed(ns, mask, stride)
        run_tag = f"maze_repaired_stride_{'x'.join(map(str, stride))}_half"
        print(f"[maze-full] run {run_tag} S={int(seed_flat.size)}", flush=True)
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
            export_data=True,
            panel_hint="maze_repair_full_export",
        )
        row["paper_case"] = spec["paper_case"]
        row["mask_kind"] = spec["mask_kind"]
        row["main_text_case"] = bool(spec["main_text_case"])
        row["cfg_dt"] = float(coarse_cfg.dt)
        row["cfg_n_steps"] = int(coarse_cfg.n_steps)
        row["cfg_enable_convection"] = bool(coarse_cfg.enable_convection)
        row["initial_velocity_mode"] = str(coarse_cfg.initial_velocity_mode)
        row["momentum_solver_mode"] = str(getattr(coarse_cfg, "momentum_solver_mode", "cg"))
        row["projection_interval"] = int(getattr(coarse_cfg, "projection_interval", 1))
        row["e_K_percent"] = 100.0 * float(row["e_K"])
        row["e_u_percent"] = 100.0 * float(row["e_u"])
        row["e_phi_percent"] = 100.0 * float(row["e_phi"])
        row["speedup_vs_ref"] = float(row["S_pipe"])
        row["full_export_scope"] = "full-field exported C_maze repaired seed row"
        rows.append(row)
        write_csv(OUT_DIR / "maze_repaired_full_export_rows.csv", rows)
        write_csv(PKG / "maze_repaired_full_export_rows.csv", rows)
        write_csv(FIG / "plot_maze_repaired_full_export_rows.csv", rows)
        print(
            f"[maze-full] done {run_tag}: eK={row['e_K_percent']:.2f}% "
            f"eu={row['e_u_percent']:.2f}% ephi={row['e_phi_percent']:.2f}% "
            f"speedup={row['speedup_vs_ref']:.1f}x",
            flush=True,
        )

    print(f"[maze-full] wrote {OUT_DIR / 'maze_repaired_full_export_rows.csv'}", flush=True)


if __name__ == "__main__":
    main()



