from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import cupy as cp

from _paths import REPO_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))

import cmame_roi_jfa_backend as backend
import run_main7_flow_production as runner


OUT = ROOT / "outputs" / "current_large_scale_roi_jfa_audit_2026-06-06"
BASE_SHAPE = (24, 24, 64)
STRIDE = (4, 4, 4)


def log(message: str) -> None:
    print(f"[large-roi] {message}", flush=True)


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [{str(k): v for k, v in row.items()} for row in csv.DictReader(f)]


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def set_roi_env() -> dict[str, str | None]:
    keys = [
        "CMAME_ROIJFA_C2_CORE",
        "CMAME_ROIJFA_SPARSE_VOXELS",
        "CMAME_ROIJFA_RADIUS_OVERRIDE",
        "CMAME_ROIJFA_ACTIVE_LIST_THRESHOLD",
    ]
    old = {key: os.environ.get(key) for key in keys}
    os.environ["CMAME_ROIJFA_C2_CORE"] = "1"
    os.environ["CMAME_ROIJFA_SPARSE_VOXELS"] = "1"
    os.environ["CMAME_ROIJFA_RADIUS_OVERRIDE"] = "1.0"
    os.environ["CMAME_ROIJFA_ACTIVE_LIST_THRESHOLD"] = "0.75"
    return old


def restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def seed_flat_to_zyx(seed_flat, shape: tuple[int, int, int]):
    d, h, w = shape
    seed_flat = cp.asarray(seed_flat, dtype=cp.int64)
    z = seed_flat // cp.int64(h * w)
    rem = seed_flat - z * cp.int64(h * w)
    y = rem // cp.int64(w)
    x = rem - y * cp.int64(w)
    return cp.stack([z, y, x], axis=1).astype(cp.int32, copy=False)


def run_case(ns: dict[str, Any], case: str, scale: int) -> dict[str, Any]:
    case_t0 = time.perf_counter()
    shape = tuple(int(v * scale) for v in BASE_SHAPE)
    log(f"start case={case} scale={scale} shape={shape}")
    log(f"  building mask")
    mask = ns["cmame_make_mask_gpu"](case, shape)
    mask_u8 = mask.astype(cp.uint8, copy=False)
    log(f"  selecting half-GFPS seeds")
    seed_spec, seed_flat = ns["cmame_half_gfps_seed_specs_for_density"](mask, STRIDE, ["half"])[0]
    seeds = seed_flat_to_zyx(seed_flat, shape)
    n_fl = int(mask.sum().get())
    n_vox = int(shape[0] * shape[1] * shape[2])
    n_seed = int(seed_flat.size)
    tile_size = (8, 8, 16)

    cp.cuda.Device().synchronize()
    old = set_roi_env()
    try:
        log(f"  ROI-JFA start, N_fl={n_fl}, S={n_seed}, tile={tile_size}")
        t0 = time.perf_counter()
        labels, dist, tile_roi, roi_mask = backend.geodesic_voronoi_roi_jfa_sparse_voxels(
            mask_u8,
            seeds,
            tile_size=tile_size,
            delta_r=1.0,
            eta_max=0.8,
            r_tile=1,
            verbose=False,
            viz_policy="none",
            n_relax_after=1,
            profile_gpu=True,
            return_records=False,
            max_refine_iters=256,
            use_active_list_step=True,
            stamping_kernel="D_c2_geodesic_ball",
        )
        cp.cuda.Device().synchronize()
        label_wall_s = time.perf_counter() - t0
        log(f"  ROI-JFA done in {label_wall_s:.3f}s")
    finally:
        restore_env(old)

    roi_voxels = int(cp.count_nonzero(roi_mask).get())
    roi_tiles = int(cp.count_nonzero(tile_roi).get())
    total_tiles = (
        ((shape[0] + tile_size[0] - 1) // tile_size[0])
        * ((shape[1] + tile_size[1] - 1) // tile_size[1])
        * ((shape[2] + tile_size[2] - 1) // tile_size[2])
    )

    cp.cuda.Device().synchronize()
    log(f"  face-connected split start")
    ts0 = time.perf_counter()
    split_labels, split_info = ns["cmame_face_connected_reindex_cpu"](mask, labels.astype(cp.int32, copy=False))
    cp.cuda.Device().synchronize()
    split_wall_s = time.perf_counter() - ts0
    log(f"  face-connected split done in {split_wall_s:.3f}s")
    gpu_split_meta = ns.get("_cmame_last_gpu_face_split", {})
    if not isinstance(gpu_split_meta, dict):
        gpu_split_meta = {}

    row = {
        "case": case,
        "scale_factor": int(scale),
        "shape_zyx": "x".join(str(v) for v in shape),
        "N_vox": n_vox,
        "N_fl": n_fl,
        "porosity": float(n_fl / max(n_vox, 1)),
        "stride_zyx": str(STRIDE),
        "S": n_seed,
        "N_cv_after_split": int(split_info["n_cv"]),
        "N_split": int(split_info["n_split_extra"]),
        "face_connected_fraction": float(split_info["face_connected_fraction"]),
        "compression_C": float(n_fl / max(int(split_info["n_cv"]), 1)),
        "tile_size": str(tile_size),
        "roi_voxels": roi_voxels,
        "roi_fraction_of_fluid": float(roi_voxels / max(n_fl, 1)),
        "roi_tiles": roi_tiles,
        "roi_tile_fraction": float(roi_tiles / max(total_tiles, 1)),
        "roi_tstamp_s": float(getattr(backend, "ROI_JFA_LAST_TSTAMP_WALL", 0.0)),
        "roi_tradii_s": float(getattr(backend, "ROI_JFA_LAST_TRADII_WALL", 0.0)),
        "roi_tcand_s": float(getattr(backend, "ROI_JFA_LAST_TCAND_WALL", 0.0)),
        "roi_tinit_s": float(getattr(backend, "ROI_JFA_LAST_TINIT_WALL", 0.0)),
        "roi_tbubble_s": float(getattr(backend, "ROI_JFA_LAST_TBUBBLE_WALL", 0.0)),
        "roi_tfilter_s": float(getattr(backend, "ROI_JFA_LAST_TFILTER_WALL", 0.0)),
        "roi_tdecode_s": float(getattr(backend, "ROI_JFA_LAST_TDECODE_WALL", 0.0)),
        "roi_tc2_s": float(getattr(backend, "ROI_JFA_LAST_TC2_WALL", 0.0)),
        "roi_c2_count": int(getattr(backend, "ROI_JFA_LAST_C2_COUNT", 0)),
        "roi_tjfa_s": float(getattr(backend, "ROI_JFA_LAST_TJFA_WALL", 0.0)),
        "roi_tclose_s": float(getattr(backend, "ROI_JFA_LAST_TCLOSE_WALL", 0.0)),
        "roi_trelax_s": float(getattr(backend, "ROI_JFA_LAST_TRELAX_WALL", 0.0)),
        "roi_tpred_s": float(getattr(backend, "ROI_JFA_LAST_TPRED_WALL", 0.0)),
        "label_wall_s": float(label_wall_s),
        "split_wall_s": float(split_wall_s),
        "label_plus_split_s": float(label_wall_s + split_wall_s),
        "gpu_face_split_used": bool(gpu_split_meta.get("used", False)),
        "gpu_face_split_iters": int(gpu_split_meta.get("iters", 0) or 0),
        "distance_audit": "accelerated_only_not_exact_global",
        "method": "current_c2_sparse_roi_jfa_plus_gpu_face_split",
    }
    del labels, dist, tile_roi, roi_mask, split_labels, mask, mask_u8, seeds, seed_flat
    cp.get_default_memory_pool().free_all_blocks()
    log(f"done case={case} scale={scale} total={time.perf_counter() - case_t0:.3f}s")
    return row


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Current large-scale ROI-JFA audit, 2026-06-06",
        "",
        "This is a data-only scaling audit. It runs current C2-sparse ROI-JFA and GPU face-connected splitting only; it does not run the voxel reference or the coarse flow solve.",
        "",
        "| Case | Scale | Shape | N_fl | S | N_cv | ROI/fluid | C2 count | ROI pred. | split | label+split |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('case', '')} | {as_int(row, 'scale_factor')} | {row.get('shape_zyx', '')} | "
            f"{as_int(row, 'N_fl')} | {as_int(row, 'S')} | {as_int(row, 'N_cv_after_split')} | "
            f"{as_float(row, 'roi_fraction_of_fluid'):.3%} | {as_int(row, 'roi_c2_count')} | "
            f"{as_float(row, 'roi_tpred_s'):.3f}s | {as_float(row, 'split_wall_s'):.3f}s | "
            f"{as_float(row, 'label_plus_split_s'):.3f}s |"
        )
    lines.extend([
        "",
        "Use in manuscript:",
        "",
        "- These rows can replace legacy large-scale ROI-JFA timing data only as accelerated-only current-protocol scaling evidence.",
        "- They should not be used as flow-accuracy evidence because no voxel reference or coarse flow solve is run in this audit.",
        "- The exact-distance partition audit remains the correctness anchor for final-sized cases.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="orthogonal_duct,skewed_duct,C_maze")
    ap.add_argument("--scales", default="1,2")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    cases = parse_csv(args.cases)
    scales = parse_int_csv(args.scales)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    runner.install_half_only_density(ns)
    runner.install_gpu_face_connected_split(ns, verify=False)

    # Warm up the CUDA kernels on a tiny case so the first recorded row is less
    # affected by just-in-time compilation.
    if not args.no_warmup:
        _ = run_case(ns, "orthogonal_duct", 1)
    rows: list[dict[str, Any]] = read_existing(out_dir / "current_large_scale_roi_jfa_rows.csv")
    completed = {
        (str(row.get("case", "")), int(float(row.get("scale_factor", -1))))
        for row in rows
        if str(row.get("case", ""))
    }
    requested = [(case, int(scale)) for case in cases for scale in scales]
    pending = [task for task in requested if task not in completed]
    log(f"requested={len(requested)}, existing={len(requested) - len(pending)}, pending={len(pending)}")
    batch_t0 = time.perf_counter()
    done_now = 0
    for case in cases:
        for scale in scales:
            if (case, int(scale)) in completed:
                log(f"keeping existing case={case} scale={scale}")
                continue
            task_t0 = time.perf_counter()
            rows.append(run_case(ns, case, scale))
            done_now += 1
            write_csv(out_dir / "current_large_scale_roi_jfa_rows.csv", rows)
            write_md(out_dir / "current_large_scale_roi_jfa_audit.md", rows)
            elapsed = time.perf_counter() - batch_t0
            avg = elapsed / max(done_now, 1)
            eta = avg * max(len(pending) - done_now, 0)
            log(
                f"progress {done_now}/{len(pending)} new rows; "
                f"last={format_eta(time.perf_counter() - task_t0)}, ETA={format_eta(eta)}"
            )
    write_csv(out_dir / "current_large_scale_roi_jfa_rows.csv", rows)
    write_md(out_dir / "current_large_scale_roi_jfa_audit.md", rows)
    log(f"wrote {out_dir / 'current_large_scale_roi_jfa_rows.csv'}")
    log(f"wrote {out_dir / 'current_large_scale_roi_jfa_audit.md'}")


if __name__ == "__main__":
    main()


