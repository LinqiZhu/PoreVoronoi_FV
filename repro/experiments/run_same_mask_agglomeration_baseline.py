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
OUT_DIR = ROOT / "outputs" / "same_mask_agglomeration_baseline_2026-06-06"
PKG = ROOT / "outputs" / "paper_ready_data_package_2026-06-06"
FIG = ROOT / "outputs" / "figure_ready_data_package_2026-06-06"
SNIP = PKG / "tex_snippets"

sys.path.insert(0, str(TOOLS))
import run_main7_flow_production as runner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bentheimer_monolithic_stokes_test import solve_monolithic  # noqa: E402


K_REF = {
    "A_thin_wall": 7.929508571134987,
    "B_narrow_throat": 1.908398990239913,
    "C_maze": 1.0113186496784738,
    "bentheimer_sandstone_crop": 0.05225707669002652,
    "fibrous_filter_proxy": 0.1373584960111365,
}

REFERENCE_TOTAL_S = {
    "A_thin_wall": 106.69208619988058,
    "B_narrow_throat": 94.66889699990861,
    "C_maze": 99.16445940011181,
    "bentheimer_sandstone_crop": 242.67442080006003,
    "fibrous_filter_proxy": 256.2604510000674,
}

CASE_PLAN: list[dict[str, Any]] = [
    {
        "case": "A_thin_wall",
        "geovoronoi_rule": "regular half-offset (4,4,4)",
        "seed_family": "stride_half_admissible",
        "stride": (4, 4, 4),
        "agglomeration_block": (4, 4, 4),
    },
    {
        "case": "B_narrow_throat",
        "geovoronoi_rule": "regular half-offset (4,4,4)",
        "seed_family": "stride_half_admissible",
        "stride": (4, 4, 4),
        "agglomeration_block": (4, 4, 4),
    },
    {
        "case": "C_maze",
        "geovoronoi_rule": "regular half-offset (4,4,4)",
        "seed_family": "stride_half_admissible",
        "stride": (4, 4, 4),
        "agglomeration_block": (4, 4, 4),
    },
    {
        "case": "bentheimer_sandstone_crop",
        "geovoronoi_rule": "mask-selector origin-offset (2,4,4)",
        "seed_family": "stride_offset_origin",
        "stride": (2, 4, 4),
        "agglomeration_block": (4, 4, 4),
    },
    {
        "case": "fibrous_filter_proxy",
        "geovoronoi_rule": "mask-selector wall-biased fast (2,4,4)",
        "seed_family": "wall_biased",
        "stride": (2, 4, 4),
        "agglomeration_block": (4, 4, 4),
    },
]


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


def case_specs(ns: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = runner.build_case_specs(ns, "production", runner.BENTHEIMER_INPUT, runner.FIBROUS_INPUT)
    return {str(spec["case"]): spec for spec in specs}


def make_cfg(ns: dict[str, Any]) -> Any:
    cfg = ns["cmame_base_cfg"](out_dir=str(OUT_DIR), profile="production")
    cfg.enable_convection = False
    cfg.pressure_gauge_eps = 1.0e-8
    cfg = runner.clone_cfg(
        ns,
        cfg,
        out_dir=str(OUT_DIR),
        transmissibility_ratio_clip=ns["_pb618_face_clip_for_mode"]("overrelaxed_default"),
        explicit_nonorthogonal_correction=0.0,
    )
    return cfg


def geovoronoi_seed(ns: dict[str, Any], mask, item: dict[str, Any]):
    case = str(item["case"])
    stride = tuple(item["stride"])
    family = str(item["seed_family"])
    if family == "stride_half_admissible":
        candidates = ns["cmame_half_gfps_seed_specs_for_density"](mask, stride, ["half"])
        candidates = [
            cand for cand in candidates
            if str(cand[0].get("family", "")).lower() == "stride_half_admissible"
        ]
        if not candidates:
            raise RuntimeError(f"No half seed for {case} stride={stride}")
        return candidates[0]
    if family == "stride_offset_origin":
        off = ns["cmame_offset_from_mode"](stride, "origin")
        spec = {
            "stage": "same_mask_agglomeration_baseline",
            "family": "stride_offset",
            "seed_id": f"same_mask_position_stride_{'x'.join(map(str, stride))}_origin",
            "stride_zyx": tuple(stride),
            "offset_zyx": tuple(off),
            "repeat": 0,
        }
        return spec, ns["cmame_stride_offset_seed_flat_gpu"](mask, stride, off)
    if family == "wall_biased":
        off = ns["cmame_offset_from_mode"](stride, "half")
        base_count = int(ns["cmame_stride_offset_seed_flat_gpu"](mask, stride, off).size)
        spec = {
            "stage": "same_mask_agglomeration_baseline",
            "family": "wall_biased",
            "seed_id": f"same_mask_wall_biased_{case}_r0",
            "stride_zyx": tuple(stride),
            "offset_zyx": "wall_biased",
            "repeat": 0,
            "target_seed_count": base_count,
        }
        return spec, ns["cmame_wall_biased_seed_flat_gpu"](mask, base_count, rng_seed=8400)
    raise ValueError(f"Unknown seed family: {family}")


def block_labels(cp, mask, block: tuple[int, int, int]):
    d, h, w = [int(v) for v in mask.shape]
    bz, by, bx = [int(v) for v in block]
    nz = (d + bz - 1) // bz
    ny = (h + by - 1) // by
    nx = (w + bx - 1) // bx
    z, y, x = cp.indices((d, h, w), dtype=cp.int32)
    labels = (z // bz) * (ny * nx) + (y // by) * nx + (x // bx)
    labels = labels.astype(cp.int32, copy=False)
    return cp.where(mask.astype(bool), labels, cp.int32(-1))


def build_agglomeration_geometry(ns: dict[str, Any], mask, cfg, block: tuple[int, int, int]):
    cp = ns["cp"]
    t0 = time.perf_counter()
    labels = block_labels(cp, mask, block)
    cp.cuda.Stream.null.synchronize()
    t_label = time.perf_counter() - t0

    t_split0 = time.perf_counter()
    labels, split_info = ns["cmame_face_connected_reindex_cpu"](mask, labels)
    cp.cuda.Stream.null.synchronize()
    t_split = time.perf_counter() - t_split0

    t_geom0 = time.perf_counter()
    dist = cp.zeros(mask.shape, dtype=cp.float64)
    geom = ns["cmame_build_geometry_from_labels_ncells_gpu"](
        mask, labels, dist, int(split_info["n_cv"]), cfg
    )
    cp.cuda.Stream.null.synchronize()
    t_geom = time.perf_counter() - t_geom0

    n_fl = int(mask.sum().get())
    meta = {
        "S": "",
        "N_cv": int(geom.n_cells),
        "N_fl": n_fl,
        "C_comp": float(n_fl / max(int(geom.n_cells), 1)),
        "N_split": int(split_info["n_split_extra"]),
        "N_face_disconnected_labels": int(split_info["n_face_disconnected_labels"]),
        "face_connected_fraction": float(split_info["face_connected_fraction"]),
        "t_label_s": float(t_label),
        "t_split_s": float(t_split),
        "t_mom_s": float(t_geom),
        "t_build_s": float(t_label + t_split + t_geom),
    }
    return geom, meta


def build_geovoronoi_geometry(ns: dict[str, Any], mask, cfg, seed_spec, seed_flat):
    t0 = time.perf_counter()
    geom, meta = ns["cmame_build_geometry_from_seed_flat_timed"](
        mask,
        seed_flat,
        cfg,
        split_face_components=True,
        label_mode="roi_jfa",
        seed_spec=seed_spec,
    )
    meta["t_build_s"] = float(time.perf_counter() - t0)
    return geom, meta


def solve_row(
    ns: dict[str, Any],
    *,
    spec: dict[str, Any],
    method: str,
    construction_rule: str,
    geom,
    meta: dict[str, Any],
    cfg,
    wall_beta: float,
) -> dict[str, Any]:
    cp = ns["cp"]
    case = str(spec["case"])
    if abs(float(wall_beta) - 1.0) > 1.0e-15:
        geom = ns["_pb618_clone_geometry_with_scaled_twall"](geom, float(wall_beta))
    t0 = time.perf_counter()
    res = solve_monolithic(ns, geom, cfg)
    t_solve = float(time.perf_counter() - t0)
    t_total = float(meta.get("t_build_s", 0.0)) + t_solve
    k_ref = float(K_REF[case])
    e_k = abs(float(res["K_eff_x"]) - k_ref) / max(abs(k_ref), 1.0e-300)
    ref_total = float(REFERENCE_TOTAL_S[case])
    return {
        "case": case,
        "paper_case": str(spec.get("paper_case", case)),
        "method": method,
        "construction_rule": construction_rule,
        "wall_beta": float(wall_beta),
        "N_fl": int(meta.get("N_fl", int(geom.mask.sum().get()))),
        "S": meta.get("S", ""),
        "N_cv": int(meta["N_cv"]),
        "compression_C": float(meta.get("C_comp", int(meta.get("N_fl", 0)) / max(int(meta["N_cv"]), 1))),
        "N_split": int(meta.get("N_split", 0) or 0),
        "face_connected_fraction": float(meta.get("face_connected_fraction", 1.0) or 1.0),
        "K_eff_x": float(res["K_eff_x"]),
        "K_ref_x": k_ref,
        "e_K_percent": 100.0 * e_k,
        "mass_inf_per_volume": float(res["mass_inf_per_volume"]),
        "steady_momentum_inf": float(res["steady_momentum_inf"]),
        "umax": float(res["umax"]),
        "t_build_s": float(meta.get("t_build_s", 0.0)),
        "t_solve_s": t_solve,
        "t_total_s": t_total,
        "t_ref_total_s": ref_total,
        "speedup_vs_ref": ref_total / max(t_total, 1.0e-300),
        "roi_tpred_s": float(meta.get("roi_tpred_s", 0.0) or 0.0),
        "claim_boundary": "same-mask current-operator construction comparison; not a literature-implementation timing claim",
    }


def fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_summary(rows: list[dict[str, Any]]) -> None:
    write_csv(OUT_DIR / "same_mask_agglomeration_baseline_rows.csv", rows)
    write_csv(PKG / "same_mask_agglomeration_baseline.csv", rows)
    write_csv(FIG / "plot_same_mask_agglomeration_baseline.csv", rows)

    tex_lines = []
    for row in rows:
        tex_lines.append(
            " & ".join(
                [
                    tex_escape(row["paper_case"]),
                    tex_escape(row["method"]),
                    tex_escape(row["construction_rule"]),
                    str(int(row["N_cv"])),
                    fmt(row["compression_C"], 1),
                    fmt(row["e_K_percent"], 2),
                    fmt(row["speedup_vs_ref"], 1),
                ]
            )
            + r" \\"
        )
    SNIP.mkdir(parents=True, exist_ok=True)
    (SNIP / "table_same_mask_agglomeration_rows.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    md = [
        "# Same-mask Block-agglomeration Baseline",
        "",
        "Date: 2026-06-06",
        "",
        "This audit compares the current GeoVoronoi-FV construction with a conservative block-voxel agglomeration baseline on the same masks, the same locked voxel-reference permeabilities, and the same monolithic coarse-Stokes solve. The block baseline is a current-operator construction comparison, not a claim about a fully optimized implementation of the cited voxel-agglomeration literature.",
        "",
        "| Case | Method | Rule | N_cv | C | eK (%) | Speedup |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['paper_case']} | {row['method']} | {row['construction_rule']} | "
            f"{int(row['N_cv'])} | {fmt(row['compression_C'], 1)} | {fmt(row['e_K_percent'], 2)} | {fmt(row['speedup_vs_ref'], 1)} |"
        )
    md.extend(
        [
            "",
            "## Use boundary",
            "",
            "- Safe use: same-mask construction comparison under the current coarse operator.",
            "- Unsafe use: claiming a definitive speed benchmark against all voxel-agglomeration implementations.",
            "- Interpretation should focus on whether prescribed geodesic/state sites improve accuracy at comparable compression on the retained masks.",
            "- The retained result is nuanced: block agglomeration is competitive on regular synthetic masks but fails badly on the two segmented complex morphologies.",
            "",
        ]
    )
    (OUT_DIR / "same_mask_agglomeration_baseline.md").write_text("\n".join(md), encoding="utf-8")
    (PKG / "same_mask_agglomeration_baseline.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CMAME_ROIJFA_C2_CORE"] = "1"
    os.environ["CMAME_ROIJFA_SPARSE_VOXELS"] = "1"
    os.environ["CMAME_ROIJFA_STAMPING_MODE"] = "D_c2_geodesic_ball"
    os.environ["CMAME_ROIJFA_TILE"] = "8,8,16"
    os.environ.setdefault("CMAME_MAIN7_PROGRESS", "1")

    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    runner.install_runtime_cfg_attr_preservation(ns)
    runner.install_gpu_face_connected_split(ns, verify=False)
    runner.install_roi_backend(ns, NOTEBOOK_DIR)

    specs = case_specs(ns)
    rows: list[dict[str, Any]] = []
    wall_beta = 1.25
    total_jobs = 2 * len(CASE_PLAN)
    job = 0
    start_all = time.perf_counter()
    for item in CASE_PLAN:
        case = str(item["case"])
        spec = specs[case]
        mask = runner.make_mask(ns, spec)
        cfg = make_cfg(ns)

        job += 1
        print(f"[agglom] {job}/{total_jobs} GeoVoronoi {case}", flush=True)
        t0 = time.perf_counter()
        seed_spec, seed_flat = geovoronoi_seed(ns, mask, item)
        geom, meta = build_geovoronoi_geometry(ns, mask, cfg, seed_spec, seed_flat)
        meta["S"] = int(seed_flat.size)
        row = solve_row(
            ns,
            spec=spec,
            method="GeoVoronoi-FV",
            construction_rule=str(item["geovoronoi_rule"]),
            geom=geom,
            meta=meta,
            cfg=cfg,
            wall_beta=wall_beta,
        )
        rows.append(row)
        write_summary(rows)
        elapsed = time.perf_counter() - start_all
        avg = elapsed / max(job, 1)
        eta = avg * max(total_jobs - job, 0)
        print(
            f"[agglom] done GeoVoronoi {case}: eK={row['e_K_percent']:.2f}% "
            f"speedup={row['speedup_vs_ref']:.1f}x stage={time.perf_counter()-t0:.1f}s "
            f"eta={eta/60:.1f} min",
            flush=True,
        )

        job += 1
        print(f"[agglom] {job}/{total_jobs} Block agglomeration {case}", flush=True)
        t0 = time.perf_counter()
        geom, meta = build_agglomeration_geometry(ns, mask, cfg, tuple(item["agglomeration_block"]))
        row = solve_row(
            ns,
            spec=spec,
            method="Block agglomeration",
            construction_rule=f"axis-aligned block {tuple(item['agglomeration_block'])}",
            geom=geom,
            meta=meta,
            cfg=cfg,
            wall_beta=wall_beta,
        )
        rows.append(row)
        write_summary(rows)
        elapsed = time.perf_counter() - start_all
        avg = elapsed / max(job, 1)
        eta = avg * max(total_jobs - job, 0)
        print(
            f"[agglom] done Block {case}: eK={row['e_K_percent']:.2f}% "
            f"speedup={row['speedup_vs_ref']:.1f}x stage={time.perf_counter()-t0:.1f}s "
            f"eta={eta/60:.1f} min",
            flush=True,
        )

    print(f"[agglom] wrote {OUT_DIR / 'same_mask_agglomeration_baseline_rows.csv'}", flush=True)
    print(f"[agglom] wrote {PKG / 'same_mask_agglomeration_baseline.csv'}", flush=True)


if __name__ == "__main__":
    main()



