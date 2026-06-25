from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any


from _paths import REPO_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
BASE = REPO_ROOT
TOOLS = SRC_ROOT
NOTEBOOK_DIR = BASE / "notebooks"
OUT_DIR = ROOT / "outputs" / "maze_seed_repair_scan_2026-06-06"

sys.path.insert(0, str(TOOLS))
import run_main7_flow_production as runner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_density_position_scan as scan  # noqa: E402


CASE = "C_maze"
K_REF = 1.0113186496784738
REFERENCE_TOTAL_S = 99.16445940011181


def write_rows(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT_DIR / "maze_seed_repair_scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_candidate(out: list[dict[str, Any]], seen: set[str], spec: dict[str, Any], seed_flat, *, source: str) -> None:
    seed_id = str(spec.get("seed_id", f"{source}_{len(out)}"))
    key = f"{seed_id}|{int(seed_flat.size)}|{source}"
    if key in seen:
        return
    seen.add(key)
    row = dict(spec)
    row["seed_source"] = source
    row["_seed_flat"] = seed_flat
    out.append(row)


def maze_candidates(ns: dict[str, Any], mask) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    # The baseline row uses (4,4,4).  The extra rows test whether the maze
    # error is seed-density/placement driven while keeping the same FV operator.
    density_strides = [(4, 4, 4), (3, 4, 4), (2, 4, 4), (2, 3, 4), (2, 2, 4)]
    for stride in density_strides:
        for spec, seed_flat in ns["cmame_half_gfps_seed_specs_for_density"](mask, stride, ["half"]):
            spec = dict(spec)
            spec["scan_panel"] = "maze_density"
            add_candidate(rows, seen, spec, seed_flat, source="half_density")

        half_offset = ns["cmame_offset_from_mode"](stride, "half")
        base_count = int(ns["cmame_stride_offset_seed_flat_gpu"](mask, stride, half_offset).size)

        origin_offset = ns["cmame_offset_from_mode"](stride, "origin")
        add_candidate(
            rows,
            seen,
            {
                "stage": "maze_seed_repair",
                "family": "stride_offset",
                "seed_id": f"maze_origin_stride_{'x'.join(map(str, stride))}",
                "stride_zyx": tuple(stride),
                "offset_zyx": tuple(origin_offset),
                "repeat": 0,
                "target_seed_count": base_count,
                "scan_panel": "maze_position",
            },
            ns["cmame_stride_offset_seed_flat_gpu"](mask, stride, origin_offset),
            source="origin_position",
        )

        if "cmame_gfps_seed_flat_gpu" in ns:
            min_sep = max(1.0, 0.50 * float(max(1, min(stride))))
            add_candidate(
                rows,
                seen,
                {
                    "stage": "maze_seed_repair",
                    "family": "gfps_admissible",
                    "seed_id": f"maze_gfps_stride_{'x'.join(map(str, stride))}_r0",
                    "stride_zyx": tuple(stride),
                    "offset_zyx": "gfps",
                    "repeat": 0,
                    "target_seed_count": base_count,
                    "scan_panel": "maze_position",
                    "min_sep": min_sep,
                },
                ns["cmame_gfps_seed_flat_gpu"](
                    mask,
                    base_count,
                    rng_seed=9700 + 100 * min(stride) + 3 * sum(stride),
                    min_clearance=1.0,
                    min_sep=min_sep,
                    batch_rounds=10,
                ),
                source="gfps_position",
            )

        if "cmame_wall_biased_seed_flat_gpu" in ns and stride in {(4, 4, 4), (2, 4, 4)}:
            add_candidate(
                rows,
                seen,
                {
                    "stage": "maze_seed_repair",
                    "family": "wall_biased",
                    "seed_id": f"maze_wall_biased_stride_{'x'.join(map(str, stride))}_r0",
                    "stride_zyx": tuple(stride),
                    "offset_zyx": "wall_biased",
                    "repeat": 0,
                    "target_seed_count": base_count,
                    "scan_panel": "maze_position",
                },
                ns["cmame_wall_biased_seed_flat_gpu"](mask, base_count, rng_seed=9800 + 11 * sum(stride)),
                source="wall_biased_position",
            )

    # Keep the first repair pass small enough to finish quickly on the desktop.
    out: list[dict[str, Any]] = []
    for row in rows:
        seed_flat = row["_seed_flat"]
        if int(seed_flat.size) <= 2600:
            out.append(row)
        else:
            print(f"[maze-repair] skip {row.get('seed_id')} S={int(seed_flat.size)} > 2600", flush=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan.OUT_DIR = OUT_DIR
    scan.K_REF[CASE] = K_REF
    scan.REFERENCE_TOTAL_S[CASE] = REFERENCE_TOTAL_S

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

    specs = scan.case_specs(ns)
    mask = runner.make_mask(ns, specs[CASE])
    candidates = maze_candidates(ns, mask)
    rows: list[dict[str, Any]] = []
    print(f"[maze-repair] {len(candidates)} C_maze seed candidates", flush=True)
    for idx, seed_row in enumerate(candidates, start=1):
        print(
            f"[maze-repair] {idx}/{len(candidates)} {seed_row.get('seed_id')} "
            f"S={int(seed_row['_seed_flat'].size)}",
            flush=True,
        )
        row = scan.run_one(ns, specs[CASE], seed_row, CASE, mask, wall_beta=1.25)
        rows.append(row)
        write_rows(rows)

    rows_sorted = sorted(rows, key=lambda r: float(r["e_K"]))
    print("[maze-repair] best rows:", flush=True)
    for row in rows_sorted[:8]:
        print(
            f"[maze-repair] {row['seed_id']} S={row['S']} "
            f"eK={100.0 * float(row['e_K']):.2f}% speedup={float(row['speedup_est_vs_ref']):.1f}x "
            f"K={float(row['K_eff_x']):.6g}",
            flush=True,
        )
    print(f"[maze-repair] wrote {OUT_DIR / 'maze_seed_repair_scan.csv'}", flush=True)


if __name__ == "__main__":
    main()



