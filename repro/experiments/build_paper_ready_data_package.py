from __future__ import annotations

import csv
from pathlib import Path


from _paths import REPO_ROOT, SRC_ROOT, add_source_path
add_source_path()
ROOT = REPO_ROOT
OUT = ROOT / "outputs" / "paper_ready_data_package_2026-06-06"

MAIN_SELECTED = ROOT / "outputs" / "final_main_table_selected_2026-06-06.csv"
MAIN_PLUS_FIB = ROOT / "outputs" / "final_main_table_selected_plus_fibrous_si_2026-06-06.csv"
HYDRAULIC = ROOT / "outputs" / "hydraulic_network_closure_production" / "hydraulic_network_closure_all_cases.csv"
FIB_BETA = ROOT / "outputs" / "fibrous_beta_sensitivity_summary_2026-06-06.csv"
FIB_SCAN = ROOT / "outputs" / "fibrous_monolithic_beta_scan" / "fibrous_monolithic_beta_scan.csv"
BENT_BETA2 = ROOT / "outputs" / "bentheimer_monolithic_stokes_beta2_stride1" / "table_main7_final_accuracy_cost_summary.csv"
SEED_SCAN = ROOT / "outputs" / "seed_density_position_scan" / "seed_density_position_scan.csv"
FIB_SEED_REPEATS = ROOT / "outputs" / "fibrous_seed_position_repeats" / "fibrous_seed_position_repeats.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({str(k).strip().strip('"'): v for k, v in row.items()})
        return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: str | float) -> float:
    return 100.0 * float(value)


def as_float(value: str | float) -> float:
    return float(value)


def build_main_table() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(MAIN_SELECTED):
        rows.append(
            {
                "case": row["case"],
                "paper_case": row["paper_case"],
                "role": "main_text_general_solver",
                "solver_closure": row["initial_velocity_mode"],
                "stride_zyx": row["stride_zyx"],
                "S": int(float(row["S"])),
                "N_cv": int(float(row["N_cv"])),
                "compression_C": as_float(row["C_comp"]),
                "wall_beta": as_float(row["wall_beta"]),
                "K_eff_x": as_float(row["K_eff_x"]),
                "K_ref_x": as_float(row["K_ref_x"]),
                "e_K_percent": pct(row["e_K"]),
                "e_u_percent": pct(row["e_u"]),
                "e_phi_percent": pct(row["e_phi"]),
                "mass_inf_per_volume": as_float(row["mass_inf_per_volume"]),
                "steady_momentum_inf": as_float(row["steady_momentum_inf"]),
                "t_total_s": as_float(row["t_total_s"]),
                "t_ref_total_s": as_float(row["t_ref_total_s"]),
                "speedup_vs_ref": as_float(row["S_pipe"]),
                "roi_tpred_ms": 1000.0 * as_float(row["roi_tpred_s"]),
                "recommended_use": (
                    "main permeability/mass/timing row; avoid pointwise velocity claim"
                    if row["case"] == "bentheimer_sandstone_crop"
                    else "main benchmark row"
                ),
            }
        )
    return rows


def find_row(rows: list[dict[str, str]], *, case: str, seed_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("case") == case and row.get("seed_id") == seed_id:
            return row
    raise KeyError(f"missing seed row: {case} {seed_id}")


def build_seed_protocol_table() -> list[dict[str, object]]:
    scan_rows = read_csv(SEED_SCAN)
    repeat_rows = read_csv(FIB_SEED_REPEATS)
    selected = [
        {
            "protocol_label": "fibrous_fast_fixed_beta",
            "row": find_row(
                scan_rows,
                case="fibrous_filter_proxy",
                seed_id="position_wall_biased_fibrous_filter_proxy_r0",
            ),
            "seed_rule": "wall-biased first deterministic candidate at target count equal to half stride",
            "selection_rule": "fixed seed-family and fixed RNG seed before validation",
            "role": "current-method fast fibrous candidate",
            "source_csv": "outputs/seed_density_position_scan/seed_density_position_scan.csv",
        },
        {
            "protocol_label": "fibrous_accurate_fixed_beta",
            "row": find_row(
                repeat_rows,
                case="fibrous_filter_proxy",
                seed_id="repeat_gfps_stride_1x4x4_r0",
            ),
            "seed_rule": "GFPS first deterministic candidate at denser streamwise stride",
            "selection_rule": "fixed seed-family and first repeat before validation",
            "role": "current-method accurate fibrous candidate",
            "source_csv": "outputs/fibrous_seed_position_repeats/fibrous_seed_position_repeats.csv",
        },
        {
            "protocol_label": "duct_default_regular",
            "row": find_row(scan_rows, case="orthogonal_duct", seed_id="stride_4x4x4_half"),
            "seed_rule": "regular half-offset stride seed",
            "selection_rule": "fixed structured seed protocol",
            "role": "duct-safe default",
            "source_csv": "outputs/seed_density_position_scan/seed_density_position_scan.csv",
        },
        {
            "protocol_label": "bentheimer_default_offset",
            "row": find_row(
                scan_rows,
                case="bentheimer_sandstone_crop",
                seed_id="position_stride_2x4x4_origin",
            ),
            "seed_rule": "origin-offset structured stride seed",
            "selection_rule": "fixed structured seed protocol",
            "role": "granular-rock default",
            "source_csv": "outputs/seed_density_position_scan/seed_density_position_scan.csv",
        },
    ]
    out = []
    for item in selected:
        row = item["row"]
        out.append(
            {
                "protocol_label": item["protocol_label"],
                "case": row["case"],
                "seed_rule": item["seed_rule"],
                "seed_family": row["seed_family"],
                "stride_zyx": row["stride_zyx"],
                "selection_rule": item["selection_rule"],
                "wall_beta": as_float(row["wall_beta_fixed"]),
                "S": int(float(row["S"])),
                "N_cv": int(float(row["N_cv"])),
                "K_eff_x": as_float(row["K_eff_x"]),
                "K_ref": as_float(row["K_ref"]),
                "e_K": as_float(row["e_K"]),
                "total_s": as_float(row["total_s"]),
                "reference_total_s": as_float(row["reference_total_s"]),
                "speedup_est_vs_ref": as_float(row["speedup_est_vs_ref"]),
                "role": item["role"],
                "source_csv": item["source_csv"],
            }
        )
    return out


def build_site_sensitivity_table() -> list[dict[str, object]]:
    scan_rows = read_csv(SEED_SCAN)
    selected = [
        {
            "case_label": "Orthogonal duct",
            "row": find_row(scan_rows, case="orthogonal_duct", seed_id="stride_4x4x4_half"),
            "site_rule": r"regular half \((4,4,4)\)",
            "main_reading": "structured sites are adequate",
        },
        {
            "case_label": "Orthogonal duct",
            "row": find_row(
                scan_rows,
                case="orthogonal_duct",
                seed_id="position_wall_biased_orthogonal_duct_r0",
            ),
            "site_rule": r"wall-biased \((4,4,4)\)",
            "main_reading": "fibrous rule is unnecessary",
        },
        {
            "case_label": "Bentheimer crop",
            "row": find_row(
                scan_rows,
                case="bentheimer_sandstone_crop",
                seed_id="position_stride_2x4x4_origin",
            ),
            "site_rule": r"origin-offset \((2,4,4)\)",
            "main_reading": "structured offset transfers",
        },
        {
            "case_label": "Bentheimer crop",
            "row": find_row(
                scan_rows,
                case="bentheimer_sandstone_crop",
                seed_id="gfps_target_stride_2x4x4",
            ),
            "site_rule": r"GFPS \((2,4,4)\)",
            "main_reading": "generic GFPS is not always safer",
        },
        {
            "case_label": "Fibrous filter",
            "row": find_row(scan_rows, case="fibrous_filter_proxy", seed_id="stride_2x4x4_half"),
            "site_rule": r"regular half \((2,4,4)\)",
            "main_reading": "misses the fibrous connectivity scale",
        },
        {
            "case_label": "Fibrous filter",
            "row": find_row(
                scan_rows,
                case="fibrous_filter_proxy",
                seed_id="position_wall_biased_fibrous_filter_proxy_r0",
            ),
            "site_rule": r"wall-biased \((2,4,4)\)",
            "main_reading": "placement dominates at fixed count",
        },
        {
            "case_label": "Fibrous filter",
            "row": find_row(scan_rows, case="fibrous_filter_proxy", seed_id="gfps_target_stride_1x4x4"),
            "site_rule": r"GFPS \((1,4,4)\)",
            "main_reading": "higher site density recovers accuracy",
        },
    ]
    out = []
    for item in selected:
        row = item["row"]
        out.append(
            {
                "case_label": item["case_label"],
                "case": row["case"],
                "site_rule": item["site_rule"],
                "seed_id": row["seed_id"],
                "seed_family": row["seed_family"],
                "wall_beta": as_float(row["wall_beta_fixed"]),
                "S": int(float(row["S"])),
                "N_cv": int(float(row["N_cv"])),
                "K_eff_x": as_float(row["K_eff_x"]),
                "K_ref": as_float(row["K_ref"]),
                "e_K": as_float(row["e_K"]),
                "speedup_est_vs_ref": as_float(row["speedup_est_vs_ref"]),
                "main_reading": item["main_reading"],
                "source_csv": "outputs/seed_density_position_scan/seed_density_position_scan.csv",
            }
        )
    return out


def tex_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def tex_sci(value: float) -> str:
    return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def write_table_tex_snippets(
    main_rows: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
    site_rows: list[dict[str, object]],
) -> None:
    tex_dir = OUT / "tex_snippets"
    tex_dir.mkdir(parents=True, exist_ok=True)

    case_names = {
        "orthogonal_duct": "Orthogonal duct",
        "skewed_duct": "Skewed duct",
        "A_thin_wall": "Thin-wall synthetic",
        "B_narrow_throat": "Narrow-throat synthetic",
        "C_maze": "Maze synthetic stress case",
        "bentheimer_sandstone_crop": "Bentheimer segmented crop",
    }
    solver_names = {
        "zero": "dense LU",
        "monolithic_stokes": "monolithic Stokes",
    }
    main_lines = []
    for row in main_rows:
        main_lines.append(
            "{case} & {solver} & {S:d} & {N:d} & {C:.1f} & {eK:.2f} & {eu:.2f} & {ephi:.2f} & {mass} & {t:.3f} & {sp:.1f} \\\\".format(
                case=case_names.get(str(row["case"]), str(row["paper_case"])),
                solver=solver_names.get(str(row["solver_closure"]), str(row["solver_closure"])),
                S=int(row["S"]),
                N=int(row["N_cv"]),
                C=float(row["compression_C"]),
                eK=float(row["e_K_percent"]),
                eu=float(row["e_u_percent"]),
                ephi=float(row["e_phi_percent"]),
                mass=tex_sci(float(row["mass_inf_per_volume"])),
                t=float(row["t_total_s"]),
                sp=float(row["speedup_vs_ref"]),
            )
        )
    (tex_dir / "table_final_accuracy_cost_rows.tex").write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    seed_case_names = {
        "fibrous_fast_fixed_beta": "Fibrous filter",
        "fibrous_accurate_fixed_beta": "Fibrous filter",
        "duct_default_regular": "Orthogonal duct",
        "bentheimer_default_offset": "Bentheimer crop",
    }
    seed_rule_names = {
        "fibrous_fast_fixed_beta": r"wall-biased \((2,4,4)\)",
        "fibrous_accurate_fixed_beta": r"GFPS \((1,4,4)\)",
        "duct_default_regular": r"regular half \((4,4,4)\)",
        "bentheimer_default_offset": r"origin-offset \((2,4,4)\)",
    }
    seed_reading = {
        "fibrous_fast_fixed_beta": "placement dominates at fixed count",
        "fibrous_accurate_fixed_beta": "higher site density recovers accuracy",
        "duct_default_regular": "structured sites are adequate",
        "bentheimer_default_offset": "structured offset transfers",
    }
    seed_lines = []
    for row in seed_rows:
        label = str(row["protocol_label"])
        seed_lines.append(
            "{case} & {rule} & {S:d} & {N:d} & {eK:.4f} & {sp:.1f} & {reading} \\\\".format(
                case=seed_case_names[label],
                rule=seed_rule_names[label],
                S=int(row["S"]),
                N=int(row["N_cv"]),
                eK=float(row["e_K"]),
                sp=float(row["speedup_est_vs_ref"]),
                reading=seed_reading[label],
            )
        )
    (tex_dir / "table_site_sensitivity_candidate_rows.tex").write_text("\n".join(seed_lines) + "\n", encoding="utf-8")

    site_lines = []
    for row in site_rows:
        site_lines.append(
            "{case} & {rule} & {S:d} & {N:d} & {eK:.4f} & {sp:.1f} & {reading} \\\\".format(
                case=str(row["case_label"]),
                rule=str(row["site_rule"]),
                S=int(row["S"]),
                N=int(row["N_cv"]),
                eK=float(row["e_K"]),
                sp=float(row["speedup_est_vs_ref"]),
                reading=str(row["main_reading"]),
            )
        )
    (tex_dir / "table_site_sensitivity_full_rows.tex").write_text("\n".join(site_lines) + "\n", encoding="utf-8")


def build_optional_fibrous_row() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(MAIN_PLUS_FIB):
        if row["case"] != "fibrous_filter_proxy":
            continue
        rows.append(
            {
                "case": row["case"],
                "paper_case": row["paper_case"],
                "role": "supplementary_tuned_monolithic_robustness",
                "solver_closure": row["initial_velocity_mode"],
                "stride_zyx": row["stride_zyx"],
                "S": int(float(row["S"])),
                "N_cv": int(float(row["N_cv"])),
                "compression_C": as_float(row["C_comp"]),
                "wall_beta": as_float(row["wall_beta"]),
                "K_eff_x": as_float(row["K_eff_x"]),
                "K_ref_x": as_float(row["K_ref_x"]),
                "e_K_percent": pct(row["e_K"]),
                "e_u_percent": pct(row["e_u"]),
                "e_phi_percent": pct(row["e_phi"]),
                "mass_inf_per_volume": as_float(row["mass_inf_per_volume"]),
                "steady_momentum_inf": as_float(row["steady_momentum_inf"]),
                "t_total_s": as_float(row["t_total_s"]),
                "t_ref_total_s": as_float(row["t_ref_total_s"]),
                "speedup_vs_ref": as_float(row["S_pipe"]),
                "roi_tpred_ms": 1000.0 * as_float(row["roi_tpred_s"]),
                "recommended_use": "SI robustness; tuned wall beta, not universal fixed-beta claim",
            }
        )
    return rows


def build_hydraulic_real_table() -> list[dict[str, object]]:
    out = []
    for row in read_csv(HYDRAULIC):
        if row["closure"] != "patch_throat_size_factor":
            continue
        if row["case"] not in {"bentheimer_sandstone_crop", "fibrous_filter_proxy"}:
            continue
        out.append(
            {
                "case": row["case"],
                "paper_case": row["paper_case"],
                "role": "no_reference_permeability_only_closure",
                "closure": row["closure"],
                "stride_zyx": row["stride_zyx"],
                "S": int(float(row["S"])),
                "N_cv": int(float(row["N_cv"])),
                "compression_C": as_float(row["C_comp"]),
                "K_eff_x": as_float(row["K_eff_x"]),
                "K_ref_x": as_float(row["K_ref_x"]),
                "e_K_percent": pct(row["e_K"]),
                "mass_inf_per_volume": as_float(row["mass_inf_per_volume"]),
                "t_total_s": as_float(row["t_total_s"]),
                "t_ref_total_s": as_float(row["t_ref_total_s"]),
                "speedup_vs_ref": as_float(row["speedup_vs_ref"]),
                "roi_tpred_ms": 1000.0 * as_float(row["roi_tpred_s"]),
                "network_solve_ms": 1000.0 * as_float(row["network_solve_s"]),
                "face_eta_mean": as_float(row["face_eta_mean"]),
                "face_eta_p10": as_float(row["face_eta_p10"]),
                "recommended_use": "prior-tech screening only; not part of current ROI-JFA/coarse-Stokes method",
            }
        )
    return out


def build_rejected_closure_table() -> list[dict[str, object]]:
    out = []
    for row in read_csv(HYDRAULIC):
        if row["closure"] == "patch_throat_size_factor" and row["case"] in {"bentheimer_sandstone_crop", "fibrous_filter_proxy"}:
            status = "reference_only_not_current_method"
            reason = "accurate in these complex networks but changes solver family away from current coarse-Stokes method"
        elif row["closure"] == "patch_throat_size_factor":
            status = "not_for_main"
            reason = "underpredicts regular/structured duct cases because artificial patch walls dominate"
        elif "component_merged" in row["closure"] and "periodic_lift" not in row["closure"]:
            status = "rejected"
            reason = "overpredicts permeability after periodic wrap faces are included"
        else:
            status = "rejected"
            reason = "tested periodic-lift formulation cancels the macroscopic drive in this graph orientation"
        out.append(
            {
                "case": row["case"],
                "closure": row["closure"],
                "status": status,
                "K_eff_x": as_float(row["K_eff_x"]),
                "K_ref_x": as_float(row["K_ref_x"]),
                "e_K_percent": pct(row["e_K"]),
                "mass_inf_per_volume": as_float(row["mass_inf_per_volume"]),
                "t_total_s": as_float(row["t_total_s"]),
                "reason": reason,
            }
        )
    return out


def build_sensitivity_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fib_rows = []
    source = FIB_BETA if FIB_BETA.exists() else FIB_SCAN
    for row in read_csv(source):
        beta = row.get("beta") or row.get("wall_beta")
        k_eff = row.get("K_eff_x") or row.get("K_eff")
        e_k_percent = row.get("e_K_percent")
        e_k = row.get("e_K") or row.get("e_K_x")
        fib_rows.append(
            {
                "case": "fibrous_filter_proxy",
                "beta": as_float(beta),
                "K_eff_x": as_float(k_eff),
                "e_K_percent": as_float(e_k_percent) if e_k_percent is not None else pct(e_k),
                "mass_inf_per_volume": as_float(row["mass_inf_per_volume"]),
                "recommended_use": "SI wall-closure sensitivity only",
            }
        )
    bent_rows = []
    if BENT_BETA2.exists():
        for row in read_csv(BENT_BETA2):
            bent_rows.append(
                {
                    "case": row["case"],
                    "paper_case": row["paper_case"],
                    "stride_zyx": row["stride_zyx"],
                    "wall_beta": as_float(row["wall_beta"]),
                    "K_eff_x": as_float(row["K_eff_x"]),
                    "K_ref_x": as_float(row["K_ref_x"]),
                    "e_K_percent": pct(row["e_K"]),
                    "mass_inf_per_volume": as_float(row["mass_inf_per_volume"]),
                    "t_total_s": as_float(row["t_total_s"]),
                    "speedup_vs_ref": as_float(row["S_pipe"]),
                    "recommended_use": "SI sensitivity; good permeability match but rock-specific beta",
                }
            )
    return fib_rows, bent_rows


def write_claim_matrix() -> None:
    rows = [
        {
            "claim": "ROI-JFA C2+sparse partitioning supports high compression and fast geometry construction",
            "evidence_file": "main_solver_table_for_table1.csv",
            "status": "supported",
            "guardrail": "Report ROI timing and compression; do not imply LOS-C2 is faster.",
        },
        {
            "claim": "General coarse Stokes solver gives accurate permeability on procedural cases",
            "evidence_file": "main_solver_table_for_table1.csv",
            "status": "supported",
            "guardrail": "Maze is a stress case with 26.85% permeability error; present honestly.",
        },
        {
            "claim": "Bentheimer real-rock permeability transfer is below 10% error with large speedup",
            "evidence_file": "main_solver_table_for_table1.csv",
            "status": "supported",
            "guardrail": "Do not claim pointwise velocity accuracy for Bentheimer.",
        },
        {
            "claim": "A PNM-style hydraulic-size-factor closure can replace or extend the current ROI-JFA/coarse-Stokes method",
            "evidence_file": "real_complex_no_reference_hydraulic_closure.csv",
            "status": "reference_only_not_current_method",
            "guardrail": "Useful prior-tech screening result but not part of the current method or contribution.",
        },
        {
            "claim": "One fixed wall beta is universal across porous materials",
            "evidence_file": "fibrous_wall_beta_sensitivity.csv",
            "status": "not_supported",
            "guardrail": "Fibrous beta sensitivity shows beta=1.25 is inaccurate there.",
        },
        {
            "claim": "One fixed seed family is universal across porous materials",
            "evidence_file": "morphology_aware_seed_protocol_candidate.csv",
            "status": "not_supported",
            "guardrail": "Fibrous benefits from wall-biased or denser GFPS sites, while ducts and Bentheimer prefer structured or offset sites.",
        },
        {
            "claim": "Fibrous fixed-beta accuracy can be improved by ROI-JFA site count and placement",
            "evidence_file": "morphology_aware_seed_protocol_candidate.csv",
            "status": "supported_candidate",
            "guardrail": "Do not select the best repeat by K_ref; use fixed or geometry-defined seed protocol before validation.",
        },
        {
            "claim": "Voxel agglomeration is a final quantitative baseline under the current production protocol",
            "evidence_file": "none_fresh_final",
            "status": "not_currently_supported",
            "guardrail": "Keep agglomeration as positioning only until same-mask final-protocol runs are regenerated.",
        },
        {
            "claim": "Component-merged throat closure improves the network method",
            "evidence_file": "hydraulic_closure_variant_decision_table.csv",
            "status": "rejected",
            "guardrail": "It overpredicts after periodic wrap faces are included.",
        },
    ]
    write_csv(OUT / "claim_to_evidence_matrix.csv", rows)


def write_manifest() -> None:
    text = """# Paper-Ready Data Package

Date: 2026-06-06

This folder separates manuscript-ready numerical evidence from exploratory or rejected runs.

## Files

- `main_solver_table_for_table1.csv`: preferred main-text numerical table. It contains five procedural ROI-JFA/coarse-Stokes rows plus the Bentheimer monolithic-Stokes real-rock row.
- `fibrous_tuned_monolithic_si_row.csv`: optional SI row for fibrous robustness under beta=2.0.
- `real_complex_no_reference_hydraulic_closure.csv`: exploratory prior-tech reference only. It records a PNM-style hydraulic-size-factor test and must not be presented as a replacement for the ROI-JFA/coarse-Stokes method.
- `hydraulic_closure_variant_decision_table.csv`: all tested PNM-style variants with accepted/rejected status.
- `fibrous_wall_beta_sensitivity.csv`: beta sensitivity for the fibrous proxy.
- `bentheimer_beta2_sensitivity_row.csv`: optional Bentheimer beta=2 sensitivity row.
- `claim_to_evidence_matrix.csv`: claim-level guidance to prevent overclaiming.
- `morphology_aware_seed_protocol_candidate.csv`: fixed-beta current-method seed-protocol candidates from the seed count/placement scan.
- `site_sensitivity_table_for_tex.csv`: full fixed-beta site count/placement table matching the current LaTeX sensitivity table.
- `tex_snippets/`: rows generated from the current CSVs for the main accuracy-cost table and site-sensitivity table.

## Main recommendation

Use `main_solver_table_for_table1.csv` for the main table. Keep `real_complex_no_reference_hydraulic_closure.csv` and the other hydraulic-network variants as prior-tech screening evidence only, not as the paper's method. The current method remains ROI-JFA geometry reduction coupled to the coarse Stokes/monolithic solver.

Use `morphology_aware_seed_protocol_candidate.csv` for the prescribed-site count/placement sensitivity discussion. These rows keep `beta=1.25` fixed and change only the site family/count. They are current-method evidence, but the final manuscript should avoid choosing a seed repeat by reference error.

## Guardrails

- Do not claim pointwise velocity accuracy for Bentheimer or fibrous.
- Do not claim the PNM-style hydraulic closure replaces or extends the current method unless a separate method section is intentionally added later.
- Do not claim a universal fixed wall beta across porous materials.
- Do not claim a universal seed family across porous materials.
- Do not use old agglomeration-review rows as final quantitative evidence under the current production protocol.
- Do not use old Bentheimer pressure-correction runs as final evidence.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = build_main_table()
    seed_rows = build_seed_protocol_table()
    site_rows = build_site_sensitivity_table()
    write_csv(OUT / "main_solver_table_for_table1.csv", main_rows)
    write_csv(OUT / "morphology_aware_seed_protocol_candidate.csv", seed_rows)
    write_csv(OUT / "site_sensitivity_table_for_tex.csv", site_rows)
    write_csv(OUT / "fibrous_tuned_monolithic_si_row.csv", build_optional_fibrous_row())
    write_csv(OUT / "real_complex_no_reference_hydraulic_closure.csv", build_hydraulic_real_table())
    write_csv(OUT / "hydraulic_closure_variant_decision_table.csv", build_rejected_closure_table())
    fib_rows, bent_rows = build_sensitivity_tables()
    write_csv(OUT / "fibrous_wall_beta_sensitivity.csv", fib_rows)
    write_csv(OUT / "bentheimer_beta2_sensitivity_row.csv", bent_rows)
    write_claim_matrix()
    write_table_tex_snippets(main_rows, seed_rows, site_rows)
    write_manifest()
    print(f"[paper-ready] wrote {OUT}")


if __name__ == "__main__":
    main()


