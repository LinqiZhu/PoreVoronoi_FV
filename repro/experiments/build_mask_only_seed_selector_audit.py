from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


from _paths import REPO_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
BASE = REPO_ROOT
TOOLS = SRC_ROOT
OUT = ROOT / "outputs" / "paper_ready_data_package_2026-06-06"
FIG = ROOT / "outputs" / "figure_ready_data_package_2026-06-06"
SNIP = OUT / "tex_snippets"

sys.path.insert(0, str(TOOLS))
import run_main7_flow_production as runner  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def find_row(rows: list[dict[str, str]], *, case: str, **criteria: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("case") != case:
            continue
        ok = True
        for key, value in criteria.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    return None


def mask_wall_metrics(mask_np: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask_np, dtype=bool)
    fluid = int(mask.sum())
    total = int(mask.size)
    contacts = np.zeros_like(mask, dtype=np.uint8)
    contact_faces = 0
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        ma = mask[tuple(a)]
        mb = mask[tuple(b)]
        boundary = ma != mb
        contact_faces += int(boundary.sum())
        contacts[tuple(a)] += (boundary & ma).astype(np.uint8)
        contacts[tuple(b)] += (boundary & mb).astype(np.uint8)

    fluid_contacts = contacts[mask]
    if fluid_contacts.size:
        near_wall_fraction = float(np.mean(fluid_contacts > 0))
        mean_solid_neighbors = float(np.mean(fluid_contacts))
        p90_solid_neighbors = float(np.quantile(fluid_contacts, 0.90))
        p99_solid_neighbors = float(np.quantile(fluid_contacts, 0.99))
    else:
        near_wall_fraction = 0.0
        mean_solid_neighbors = 0.0
        p90_solid_neighbors = 0.0
        p99_solid_neighbors = 0.0

    return {
        "total_voxels": total,
        "fluid_voxels": fluid,
        "porosity": float(fluid / max(total, 1)),
        "near_wall_fraction": near_wall_fraction,
        "mean_solid_neighbors": mean_solid_neighbors,
        "p90_solid_neighbors": p90_solid_neighbors,
        "p99_solid_neighbors": p99_solid_neighbors,
        "contact_faces_per_fluid": float(contact_faces / max(fluid, 1)),
    }


def selector_from_metrics(metrics: dict[str, float | int], *, accuracy_mode: bool) -> dict[str, str]:
    porosity = float(metrics["porosity"])
    near_wall = float(metrics["near_wall_fraction"])
    mean_contacts = float(metrics["mean_solid_neighbors"])

    if near_wall >= 0.35 and porosity >= 0.45:
        if accuracy_mode:
            return {
                "selector_class": "fibrous_high_wall_contact",
                "seed_family": "gfps_admissible",
                "seed_rule": "GFPS denser streamwise sites, stride (1,4,4)",
                "selector_reason": "near-wall fluid fraction >= 0.35 with moderate/high porosity",
            }
        return {
            "selector_class": "fibrous_high_wall_contact",
            "seed_family": "wall_biased",
            "seed_rule": "wall-biased sites at the half-stride target count",
            "selector_reason": "near-wall fluid fraction >= 0.35 with moderate/high porosity",
        }
    if porosity < 0.45 and mean_contacts >= 0.25:
        return {
            "selector_class": "granular_low_porosity",
            "seed_family": "stride_offset",
            "seed_rule": "origin-offset structured sites, stride (2,4,4)",
            "selector_reason": "low porosity with nontrivial solid-contact density",
        }
    return {
        "selector_class": "structured_or_open_channel",
        "seed_family": "stride_half_admissible",
        "seed_rule": "regular half-offset structured sites",
        "selector_reason": "low near-wall fraction or high-porosity structured geometry",
    }


def evidence_row_for_selection(
    *,
    case: str,
    selection: dict[str, str],
    main_rows: list[dict[str, str]],
    protocol_rows: list[dict[str, str]],
) -> tuple[dict[str, str] | None, str, str]:
    family = selection["seed_family"]
    if case == "fibrous_filter_proxy" and family == "wall_biased":
        row = find_row(protocol_rows, case=case, protocol_label="fibrous_fast_fixed_beta")
        return row, "morphology_aware_seed_protocol_candidate.csv", "fast fibrous mask-only selector validation row"
    if case == "fibrous_filter_proxy" and family == "gfps_admissible":
        row = find_row(protocol_rows, case=case, protocol_label="fibrous_accurate_fixed_beta")
        return row, "morphology_aware_seed_protocol_candidate.csv", "accurate fibrous mask-only selector validation row"
    if case == "bentheimer_sandstone_crop" and family == "stride_offset":
        row = find_row(protocol_rows, case=case, protocol_label="bentheimer_default_offset")
        return row, "morphology_aware_seed_protocol_candidate.csv", "granular-rock mask-only selector validation row"
    row = find_row(main_rows, case=case)
    return row, "main_solver_table_for_table1.csv", "main-table structured-site row"


def row_metrics_from_evidence(row: dict[str, str] | None) -> dict[str, Any]:
    if row is None:
        return {
            "selected_S": "",
            "selected_N_cv": "",
            "selected_e_K_percent": "",
            "selected_speedup_vs_ref": "",
            "selected_wall_beta": "",
            "selected_K_eff_x": "",
            "selected_K_ref": "",
        }

    e_k = as_float(row, "e_K_percent", np.nan)
    if not np.isfinite(e_k):
        e_k = 100.0 * as_float(row, "e_K", np.nan)

    speedup = as_float(row, "speedup_vs_ref", np.nan)
    if not np.isfinite(speedup):
        speedup = as_float(row, "speedup_est_vs_ref", np.nan)

    return {
        "selected_S": int(float(row.get("S", 0) or 0)),
        "selected_N_cv": int(float(row.get("N_cv", 0) or 0)),
        "selected_e_K_percent": e_k,
        "selected_speedup_vs_ref": speedup,
        "selected_wall_beta": as_float(row, "wall_beta", as_float(row, "wall_beta_fixed", np.nan)),
        "selected_K_eff_x": as_float(row, "K_eff_x", np.nan),
        "selected_K_ref": as_float(row, "K_ref", np.nan),
    }


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def build_tex_rows(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["paper_case"]),
                    fmt_num(row["porosity"], 3),
                    fmt_num(row["near_wall_fraction"], 3),
                    tex_escape(row["selector_class"]),
                    tex_escape(row["selected_seed_family"]),
                    fmt_num(row["selected_e_K_percent"], 2),
                    fmt_num(row["selected_speedup_vs_ref"], 1),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    main_rows = read_csv(OUT / "main_solver_table_for_table1.csv")
    protocol_rows = read_csv(OUT / "morphology_aware_seed_protocol_candidate.csv")

    ns = runner.load_flow_namespace(runner.FLOW_NOTEBOOK)
    ns["require_cuda_gpu"]()
    specs = runner.build_case_specs(ns, "production", runner.BENTHEIMER_INPUT, runner.FIBROUS_INPUT)
    cp = ns["cp"]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        case = str(spec["case"])
        mask_cp = runner.make_mask(ns, spec)
        metrics = mask_wall_metrics(cp.asnumpy(mask_cp))
        for policy, accuracy_mode in (("fast", False), ("accurate", True)):
            if case != "fibrous_filter_proxy" and policy == "accurate":
                continue
            selection = selector_from_metrics(metrics, accuracy_mode=accuracy_mode)
            evidence, source_file, evidence_role = evidence_row_for_selection(
                case=case,
                selection=selection,
                main_rows=main_rows,
                protocol_rows=protocol_rows,
            )
            perf = row_metrics_from_evidence(evidence)
            rows.append(
                {
                    "case": case,
                    "paper_case": spec.get("paper_case", case),
                    "policy": policy,
                    "selector_class": selection["selector_class"],
                    "selected_seed_family": selection["seed_family"],
                    "selected_seed_rule": selection["seed_rule"],
                    "selector_reason": selection["selector_reason"],
                    **metrics,
                    **perf,
                    "evidence_source": source_file,
                    "evidence_role": evidence_role,
                    "selection_uses_K_ref": False,
                    "claim_boundary": (
                        "mask-only selector audit; validation uses locked K_ref only after the seed rule is fixed"
                    ),
                }
            )

    out_csv = OUT / "mask_only_seed_selector_audit.csv"
    write_csv(out_csv, rows)
    write_csv(FIG / "plot_mask_only_seed_selector_audit.csv", rows)
    SNIP.mkdir(parents=True, exist_ok=True)
    (SNIP / "table_mask_only_seed_selector_rows.tex").write_text(build_tex_rows(rows), encoding="utf-8")

    md_lines = [
        "# Mask-only Seed Selector Audit",
        "",
        "Date: 2026-06-06",
        "",
        "This audit fixes a seed-family selector from mask morphology only, then reports the already-computed validation rows. The selector does not use permeability or velocity error to choose a seed family; locked references are used only after selection to quantify the result.",
        "",
        "## Selector rule",
        "",
        "- `fibrous_high_wall_contact`: if near-wall fluid fraction is at least 0.35 and porosity is at least 0.45, use wall-biased sites for a fast policy, or denser GFPS `(1,4,4)` for an accurate fibrous policy.",
        "- `granular_low_porosity`: if porosity is below 0.45 and the mean solid-neighbour count is at least 0.25, use origin-offset structured sites.",
        "- `structured_or_open_channel`: otherwise use regular half-offset structured sites.",
        "",
        "## Validation rows",
        "",
        "| Case | Policy | Porosity | Near-wall fraction | Selector | Seed family | eK (%) | Speedup | Evidence |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {case} | {policy} | {porosity} | {near_wall} | {selector} | {family} | {ek} | {speedup} | {source} |".format(
                case=row["paper_case"],
                policy=row["policy"],
                porosity=fmt_num(row["porosity"], 3),
                near_wall=fmt_num(row["near_wall_fraction"], 3),
                selector=row["selector_class"],
                family=row["selected_seed_family"],
                ek=fmt_num(row["selected_e_K_percent"], 2),
                speedup=fmt_num(row["selected_speedup_vs_ref"], 1),
                source=row["evidence_source"],
            )
        )
    md_lines.extend(
        [
            "",
            "## Manuscript use",
            "",
            "- Safe stronger use: seed placement can be framed as a mask-conditioned protocol constraint rather than a reference-tuned post hoc choice.",
            "- Boundary: this is not yet a universal automatic selector for arbitrary unseen rocks; it is an audit over the retained manuscript masks and the already validated candidate rows.",
            "- Fibrous interpretation: the mask-only high-wall-contact class explains why regular half-offset sites fail on fibrous media and why wall-biased/GFPS candidates are retained as fast/accurate policies.",
            "",
        ]
    )
    (OUT / "mask_only_seed_selector_audit.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[mask-selector] wrote {out_csv}")
    print(f"[mask-selector] wrote {OUT / 'mask_only_seed_selector_audit.md'}")


if __name__ == "__main__":
    main()



