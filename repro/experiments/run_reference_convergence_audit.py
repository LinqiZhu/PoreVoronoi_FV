from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


from _paths import REPO_ROOT, OUTPUTS_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
LOCAL_OUTPUTS = OUTPUTS_ROOT
RUNNER_PATH = SRC_ROOT / "run_main7_flow_production.py"


def import_runner():
    spec = importlib.util.spec_from_file_location("cmame_main7_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def existing_reference_summary() -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for csv_path in sorted(LOCAL_OUTPUTS.rglob("cmame_reference_rows_all.csv")):
        for row in read_csv(csv_path):
            case = row.get("case", "")
            k_ref = as_float(row.get("K_ref_x"))
            if not case or not math.isfinite(k_ref):
                continue
            grouped[case].append(
                {
                    "K_ref_x": k_ref,
                    "steady_converged": str(row.get("steady_converged", "")).strip().lower(),
                    "steps_completed": as_float(row.get("steps_completed")),
                    "source_csv": str(csv_path.relative_to(ROOT)),
                }
            )
    out: dict[str, dict[str, Any]] = {}
    for case, rows in grouped.items():
        vals = [float(r["K_ref_x"]) for r in rows]
        k_mean = float(np.mean(vals))
        k_min = float(np.min(vals))
        k_max = float(np.max(vals))
        rel_span = (k_max - k_min) / max(abs(k_mean), 1.0e-300)
        out[case] = {
            "locked_ref_count": len(rows),
            "locked_K_ref_mean": k_mean,
            "locked_K_ref_min": k_min,
            "locked_K_ref_max": k_max,
            "locked_K_ref_rel_span": rel_span,
            "locked_steady_true": sum(1 for r in rows if r["steady_converged"] in {"true", "1", "yes"}),
            "locked_steady_false": sum(1 for r in rows if r["steady_converged"] in {"false", "0", "no"}),
            "locked_sources": "; ".join(str(r["source_csv"]) for r in rows),
        }
    return out


def comma_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def make_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="C_maze")
    ap.add_argument("--profile", default="production")
    ap.add_argument("--reference-n-steps", type=int, default=1800)
    ap.add_argument("--reference-dt", type=float, default=None)
    ap.add_argument("--reference-report-every", type=int, default=300)
    ap.add_argument("--reference-linear-solver", default="cg")
    ap.add_argument("--reference-momentum-solver", default="cg")
    ap.add_argument("--reference-projection-interval", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=LOCAL_OUTPUTS / "reference_convergence_audit_2026-06-06")
    ap.add_argument("--bentheimer-npz", type=Path, default=RUNNER_PATH.parent.parent / "production_inputs" / "bentheimer_dry_crop_16x96x96_origin_1562_59_349_pore0.npz")
    ap.add_argument("--fibrous-npz", type=Path, default=RUNNER_PATH.parent.parent / "production_inputs" / "fibrous_filter_proxy_16x64x64_from_review_mask.npz")
    ap.add_argument("--force", action="store_true", help="Rerun cases even if the same case and step count already exist in the audit CSV.")
    return ap.parse_args()


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Reference Convergence Audit",
        "",
        "Date: 2026-06-06",
        "",
        "This audit runs new traditional voxel/grid reference solves only. It does not run the ROI-JFA/coarse solver.",
        "",
        "| Case | locked refs | locked K_ref | audit steps | audit K_ref | change vs locked | mass residual | momentum residual | steady | runtime |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        locked_count = int(row.get("locked_ref_count", 0))
        locked_k = as_float(row.get("locked_K_ref_mean"))
        audit_k = as_float(row.get("audit_K_ref_x"))
        rel = as_float(row.get("rel_change_vs_locked"))
        mass = as_float(row.get("audit_mass_inf_per_volume"))
        mom = as_float(row.get("audit_steady_momentum_inf"))
        lines.append(
            "| {case} | {locked_count} | {locked_k:.8g} | {steps} | {audit_k:.8g} | {rel:.3%} | {mass:.3e} | {mom:.3e} | {steady} | {runtime:.1f}s |".format(
                case=row.get("case", ""),
                locked_count=locked_count,
                locked_k=locked_k,
                steps=int(row.get("audit_cfg_n_steps", -1)),
                audit_k=audit_k,
                rel=rel,
                mass=mass,
                mom=mom,
                steady=row.get("audit_steady_converged", ""),
                runtime=as_float(row.get("audit_runtime_s")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If `change vs locked` is small, the existing locked reference is numerically stable even if the old steady flag was not triggered.",
            "- If `steady` remains false, the final manuscript should present this as a reference-convergence audit instead of claiming the old 900-step row alone proves full steady convergence.",
            "- Seed-position and seed-count sweeps should compare against the locked reference; they should not be described as rerunning a fresh grid reference for every seed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merged_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {
        str(row.get("case", "")): row
        for row in existing_rows
        if row.get("case", "")
    }
    for row in new_rows:
        by_case[str(row.get("case", ""))] = row
    return sorted(by_case.values(), key=lambda r: str(r.get("case", "")))


def main() -> None:
    args = make_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "reference_convergence_audit_rows.csv"
    md_path = out / "reference_convergence_audit.md"

    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        existing_rows = read_csv(csv_path)
    existing_by_key = {
        (str(row.get("case", "")), int(as_float(row.get("audit_cfg_n_steps"), -1))): row
        for row in existing_rows
        if row.get("case", "")
    }

    locked = existing_reference_summary()
    runner = import_runner()
    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    runner.install_runtime_cfg_attr_preservation(ns)

    base_cfg = ns["cmame_base_cfg"](out_dir=str(out), profile=str(args.profile))
    ref_cfg = runner.clone_cfg(ns, base_cfg, out_dir=str(out))
    if args.reference_dt is not None:
        ref_cfg.dt = float(args.reference_dt)
    ref_cfg.n_steps = int(args.reference_n_steps)
    ref_cfg.report_every = int(args.reference_report_every)
    setattr(ref_cfg, "linear_solver_mode", str(args.reference_linear_solver))
    setattr(ref_cfg, "momentum_solver_mode", str(args.reference_momentum_solver))
    setattr(ref_cfg, "projection_interval", max(1, int(args.reference_projection_interval)))

    dummy_filter_args = argparse.Namespace(case_filter=args.cases, max_strides_per_case=0, stride_indices="")
    specs = runner.filter_case_specs(
        runner.build_case_specs(ns, str(args.profile), Path(args.bentheimer_npz), Path(args.fibrous_npz)),
        dummy_filter_args,
    )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        case = str(spec["case"])
        existing_key = (case, int(args.reference_n_steps))
        if not bool(args.force) and existing_key in existing_by_key:
            print(f"[reference-audit] keeping existing {case} n_steps={int(args.reference_n_steps)}")
            continue
        print(f"[reference-audit] running {case} n_steps={int(ref_cfg.n_steps)}")
        mask = runner.make_mask(ns, spec)
        tic = time.perf_counter()
        _geom, _res, ref_row = ns["cmame_build_reference"](case, mask, ref_cfg, out=out)
        runtime_s = time.perf_counter() - tic
        summary = locked.get(case, {})
        locked_k = as_float(summary.get("locked_K_ref_mean"))
        audit_k = as_float(ref_row.get("K_ref_x"))
        rel_change = abs(audit_k - locked_k) / max(abs(locked_k), 1.0e-300) if math.isfinite(locked_k) and math.isfinite(audit_k) else math.nan
        row = {
            "case": case,
            "paper_case": spec.get("paper_case", case),
            "mask_kind": spec.get("mask_kind", ""),
            "audit_K_ref_x": audit_k,
            "rel_change_vs_locked": rel_change,
            "audit_mass_inf_per_volume": as_float(ref_row.get("mass_inf_per_volume")),
            "audit_steady_momentum_inf": as_float(ref_row.get("steady_momentum_inf")),
            "audit_steady_converged": bool(ref_row.get("steady_converged", False)),
            "audit_steps_completed": int(ref_row.get("steps_completed", -1)),
            "audit_runtime_s": runtime_s,
            "audit_cfg_dt": float(ref_cfg.dt),
            "audit_cfg_n_steps": int(ref_cfg.n_steps),
            "audit_reference_npz": ref_row.get("reference_npz", ""),
            **summary,
        }
        rows.append(row)
        current_rows = merged_rows(existing_rows, rows)
        write_csv(csv_path, current_rows)
        write_markdown(md_path, current_rows)
        print(f"[reference-audit] checkpoint wrote {case}")
        del mask
        ns["cp"].get_default_memory_pool().free_all_blocks()

    final_rows = merged_rows(existing_rows, rows)
    write_csv(csv_path, final_rows)
    write_markdown(md_path, final_rows)
    print(f"[reference-audit] wrote {csv_path}")
    print(f"[reference-audit] wrote {md_path}")


if __name__ == "__main__":
    main()


