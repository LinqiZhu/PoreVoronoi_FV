from __future__ import annotations

import csv
import html
import json
from pathlib import Path

MM_W = 183.0
MM_H = 76.0
PT_TO_MM = 0.3527777778

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "cmame_artifacts" / "figures"
SOURCE = ROOT / "cmame_artifacts" / "figure_source_data"

FONT_FAMILY = "Helvetica, Arial, Liberation Sans, sans-serif"
FONT_PT = {
    "letter": 8.0,
    "title": 7.0,
    "annotation": 5.5,
    "case": 5.8,
    "axis": 6.0,
    "tick": 5.4,
    "legend": 5.4,
    "endpoint": 5.5,
    "setup": 5.4,
}
FONT = {k: v * PT_TO_MM for k, v in FONT_PT.items()}
COL = {
    "text": "#222222",
    "secondary": "#666666",
    "axis": "#3A3A3A",
    "grid": "#D8D8D8",
    "ownership": "#009E73",
    "closure": "#0072B2",
    "geometry": "#E69F00",
    "white": "#FFFFFF",
    "missing": "#777777",
    "reference": "#D62728",
}

CASES = ["Orthogonal", "Skewed", "Thin wall", "Narrow throat", "Maze"]
SPEEDUP = {
    "Orthogonal": 11.3,
    "Skewed": 9.3,
    "Thin wall": 10.4,
    "Narrow throat": 9.2,
    "Maze": 9.1,
}
CUDA_MS = {
    "Orthogonal": 0.100,
    "Skewed": 0.124,
    "Thin wall": 0.105,
    "Narrow throat": 0.123,
    "Maze": 0.123,
}
CASE_SOURCE_MAP = {
    "Orthogonal": "orthogonal_duct",
    "Skewed": "skewed_duct",
    "Thin wall": "A_thin_wall",
    "Narrow throat": "B_narrow_throat",
    "Maze": "C_maze",
}
CONSTRUCTION_CANDIDATES = [
    ROOT / "cmame_artifacts" / "figure_source_data" / "roi_jfa_construction_timing_current_path_gpu_split_with_maze_2026-06-24.csv",
    ROOT / "cmame_artifacts" / "figure_source_data" / "roi_jfa_construction_timing_current_path_gpu_split_2026-06-10.csv",
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    weight: str = "400",
    fill: str = COL["text"],
    anchor: str = "start",
    baseline: str = "middle",
    italic: bool = False,
    cls: str = "",
) -> str:
    style = f"font-family:{FONT_FAMILY};font-size:{size:.4f};font-weight:{weight};fill:{fill};"
    if italic:
        style += "font-style:italic;"
    class_attr = f' class="{esc(cls)}"' if cls else ""
    return (
        f'<text{class_attr} x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}" '
        f'dominant-baseline="{baseline}" style="{style}">{esc(value)}</text>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float,
    dash: str | None = None,
    cap: str = "butt",
    cls: str = "",
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    class_attr = f' class="{esc(cls)}"' if cls else ""
    return (
        f'<line{class_attr} x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{stroke}" stroke-width="{width:.3f}" stroke-linecap="{cap}"{dash_attr}/>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    sw: float = 0.0,
    cls: str = "",
) -> str:
    class_attr = f' class="{esc(cls)}"' if cls else ""
    stroke_attr = f' stroke="{stroke}" stroke-width="{sw:.3f}"' if stroke != "none" else ' stroke="none"'
    return (
        f'<rect{class_attr} x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" '
        f'fill="{fill}"{stroke_attr}/>'
    )


def read_build_rows() -> tuple[list[dict[str, float | str]], Path, list[dict[str, float | str]]]:
    source = next((path for path in CONSTRUCTION_CANDIDATES if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Missing construction timing CSV for Figure 04")
    with source.open(newline="", encoding="utf-8-sig") as fh:
        raw_rows = list(csv.DictReader(fh))
    by_case = {row["case"]: row for row in raw_rows}
    rows: list[dict[str, float | str]] = []
    checks: list[dict[str, float | str]] = []
    for display, raw_case in CASE_SOURCE_MAP.items():
        if raw_case not in by_case:
            raise ValueError(f"Missing construction timing row for {raw_case}")
        raw = by_case[raw_case]
        ownership = float(raw["seed_ms"]) + float(raw["label_ms"])
        closure = float(raw["split_ms"])
        geometry = float(raw["geometry_ms"])
        setup = float(raw["kernel_ms"])
        build = float(raw["build_ms"])
        stage_sum = ownership + closure + geometry
        diff = abs(stage_sum - build)
        if diff > 5e-4:
            raise AssertionError(f"Stage sum mismatch for {display}: {stage_sum} vs {build}")
        rows.append(
            {
                "case": display,
                "setup_ms": setup,
                "ownership_ms": ownership,
                "closure_ms": closure,
                "geometry_ms": geometry,
                "build_ms": build,
            }
        )
        checks.append(
            {
                "case": display,
                "stage_sum_ms": stage_sum,
                "build_ms": build,
                "absolute_difference_ms": diff,
            }
        )
    return rows, source, checks


def make_source_table(build_rows: list[dict[str, float | str]], construction_source: Path) -> Path:
    SOURCE.mkdir(parents=True, exist_ok=True)
    out = SOURCE / "Figure_04_ownership_speed_audit_source.csv"
    try:
        construction_source_label = construction_source.relative_to(ROOT).as_posix()
    except ValueError:
        construction_source_label = construction_source.name
    fields = [
        "panel",
        "case",
        "speedup_exact_frontier_over_roi_jfa",
        "roi_jfa_cuda_ms_table_value",
        "exact_gpu_ms_from_displayed_ratio",
        "setup_ms",
        "ownership_ms",
        "cell_closure_ms",
        "geometry_assembly_ms",
        "post_setup_build_ms",
        "source",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            writer.writerow(
                {
                    "panel": "a",
                    "case": case,
                    "speedup_exact_frontier_over_roi_jfa": SPEEDUP[case],
                    "roi_jfa_cuda_ms_table_value": CUDA_MS[case],
                    "exact_gpu_ms_from_displayed_ratio": SPEEDUP[case] * CUDA_MS[case],
                    "source": "geovoronoi_fv_cmame_text_master_submission_audit_final.tex Table~\\ref{tab:roi-jfa-cpu-reference}",
                }
            )
        for row in build_rows:
            writer.writerow(
                {
                    "panel": "b",
                    "case": row["case"],
                    "setup_ms": row["setup_ms"],
                    "ownership_ms": row["ownership_ms"],
                    "cell_closure_ms": row["closure_ms"],
                    "geometry_assembly_ms": row["geometry_ms"],
                    "post_setup_build_ms": row["build_ms"],
                    "source": construction_source_label,
                }
            )
    return out


def build_svg(build_rows: list[dict[str, float | str]]) -> str:
    elems: list[str] = []
    elems.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{MM_W:.1f}mm" height="{MM_H:.1f}mm" viewBox="0 0 {MM_W:.1f} {MM_H:.1f}" version="1.1">'
    )
    elems.append('<defs><style><![CDATA[text { font-family: Helvetica, Arial, Liberation Sans, sans-serif; } .panel-letter { font-weight:700; } .panel-title { font-weight:600; }]]></style>')
    elems.append('<linearGradient id="grad-ownership" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#28B391"/><stop offset="45%" stop-color="#009E73"/><stop offset="100%" stop-color="#007A59"/></linearGradient>')
    elems.append('<linearGradient id="grad-closure" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#2A87BF"/><stop offset="45%" stop-color="#0072B2"/><stop offset="100%" stop-color="#00557F"/></linearGradient>')
    elems.append('<linearGradient id="grad-geometry" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#EAB13A"/><stop offset="45%" stop-color="#D79A13"/><stop offset="100%" stop-color="#B87900"/></linearGradient>')
    elems.append('</defs>')
    elems.append(rect(0, 0, MM_W, MM_H, fill="#FFFFFF", cls="background"))

    title_y = 6.0
    panel_a_letter_x = 6.5
    panel_b_letter_x = 91.5
    panel_title_dx = 6.2
    elems.append(text(panel_a_letter_x, title_y, "a", size=FONT["letter"], weight="700", baseline="alphabetic", cls="panel-letter"))
    elems.append(text(panel_a_letter_x + panel_title_dx, title_y, "Exact agreement and ownership speedup", size=FONT["title"], weight="600", baseline="alphabetic", cls="panel-title"))
    elems.append(text(panel_b_letter_x, title_y, "b", size=FONT["letter"], weight="700", baseline="alphabetic", cls="panel-letter"))
    elems.append(text(panel_b_letter_x + panel_title_dx, title_y, "Post-setup build-time decomposition", size=FONT["title"], weight="600", baseline="alphabetic", cls="panel-title"))

    ax1 = {"left": 28.0, "top": 18.2, "width": 58.0, "height": 41.5, "xmax": 12.4}
    ax2 = {"left": 110.5, "top": 18.2, "width": 62.0, "height": 41.5, "xmax": 15.4}
    row_step = ax1["height"] / len(CASES)
    bar_h = row_step * 0.42

    def x1(v: float) -> float:
        return ax1["left"] + v / ax1["xmax"] * ax1["width"]

    def x2(v: float) -> float:
        return ax2["left"] + v / ax2["xmax"] * ax2["width"]

    def yrow(i: int) -> float:
        return ax1["top"] + (i + 0.5) * row_step

    ax1_bottom = ax1["top"] + ax1["height"]
    ax2_bottom = ax2["top"] + ax2["height"]
    case_gap = 2.4

    # Panel a axes and grid: sparse, tick-locked vertical guides only.
    for tick in [4, 8, 12]:
        elems.append(line(x1(tick), ax1["top"], x1(tick), ax1_bottom, stroke=COL["grid"], width=0.12, cls="gridline panel-a-grid"))
    elems.append(line(ax1["left"], ax1["top"], ax1["left"], ax1_bottom, stroke=COL["axis"], width=0.18, cls="axis y-axis panel-a-axis"))
    elems.append(line(ax1["left"], ax1_bottom, ax1["left"] + ax1["width"], ax1_bottom, stroke=COL["axis"], width=0.18, cls="axis panel-a-axis"))
    for tick in [0, 4, 8, 12]:
        elems.append(line(x1(tick), ax1_bottom, x1(tick), ax1_bottom + 1.1, stroke=COL["axis"], width=0.16, cls="tick panel-a-tick"))
        elems.append(text(x1(tick), ax1_bottom + 3.7, str(tick), size=FONT["tick"], anchor="middle", fill=COL["axis"], cls="tick-label panel-a-tick-label"))

    # Panel a data are exact-match rows; the red 1x reference is drawn above bars.
    for i, case in enumerate(CASES):
        y = yrow(i)
        val = SPEEDUP[case]
        elems.append(text(ax1["left"] - case_gap, y, case, size=FONT["case"], anchor="end", cls="case-label panel-a-case"))
        elems.append(rect(x1(0.0), y - bar_h / 2, x1(val) - x1(0.0), bar_h, fill="url(#grad-ownership)", cls="speedup-bar"))
        elems.append(text(x1(val) - 0.9, y, f"{val:.1f}{chr(215)}", size=FONT["endpoint"], anchor="end", fill=COL["white"], cls="endpoint-label speedup-label"))
    elems.append(line(x1(1.0), ax1["top"], x1(1.0), ax1_bottom, stroke=COL["reference"], width=0.24, dash="1.15,1.0", cls="reference-line"))
    elems.append(text(x1(1.0), ax1["top"] - 1.35, f"1{chr(215)}", size=FONT["annotation"], anchor="middle", fill=COL["reference"], cls="reference-label"))
    elems.append(text(ax1["left"] + ax1["width"] / 2, 71.3, "Ownership speedup (exact-frontier / ROI-JFA)", size=FONT["axis"], anchor="middle", cls="axis-label panel-a-axis-label"))

    # Panel b legend: same baseline, equal swatches, compact spacing inside the panel width.
    legend_y = 13.0
    legend_items = [
        ("Ownership", "url(#grad-ownership)", 110.5),
        ("Closure", "url(#grad-closure)", 128.0),
        ("Moments + facelets", "url(#grad-geometry)", 142.2),
    ]
    for label, color, lx in legend_items:
        elems.append(rect(lx, legend_y - 1.0, 2.0, 2.0, fill=color, cls="legend-swatch"))
        elems.append(text(lx + 2.9, legend_y, label, size=FONT["legend"], baseline="middle", cls="legend-label"))

    # Panel b axes and grid share the row geometry of panel a; no setup-value column is drawn.
    for tick in [4, 8, 12]:
        elems.append(line(x2(tick), ax2["top"], x2(tick), ax2_bottom, stroke=COL["grid"], width=0.12, cls="gridline panel-b-grid"))
    elems.append(line(ax2["left"], ax2["top"], ax2["left"], ax2_bottom, stroke=COL["axis"], width=0.18, cls="axis y-axis panel-b-axis"))
    elems.append(line(ax2["left"], ax2_bottom, ax2["left"] + ax2["width"], ax2_bottom, stroke=COL["axis"], width=0.18, cls="axis panel-b-axis"))
    for tick in [0, 4, 8, 12]:
        elems.append(line(x2(tick), ax2_bottom, x2(tick), ax2_bottom + 1.1, stroke=COL["axis"], width=0.16, cls="tick panel-b-tick"))
        elems.append(text(x2(tick), ax2_bottom + 3.7, str(tick), size=FONT["tick"], anchor="middle", fill=COL["axis"], cls="tick-label panel-b-tick-label"))
    elems.append(text(ax2["left"] + ax2["width"] / 2, 71.3, "Post-setup build time (ms)", size=FONT["axis"], anchor="middle", cls="axis-label panel-b-axis-label"))

    build_by_case = {str(row["case"]): row for row in build_rows}
    for i, case in enumerate(CASES):
        y = yrow(i)
        elems.append(text(ax2["left"] - case_gap, y, case, size=FONT["case"], anchor="end", cls="case-label panel-b-case"))
        row = build_by_case[case]
        left = 0.0
        for key, color in [("ownership_ms", "url(#grad-ownership)"), ("closure_ms", "url(#grad-closure)"), ("geometry_ms", "url(#grad-geometry)")]:
            width = float(row[key])
            elems.append(rect(x2(left), y - bar_h / 2, x2(left + width) - x2(left), bar_h, fill=color, stroke=COL["white"], sw=0.16, cls=f"bar-{key}"))
            left += width
        elems.append(text(min(x2(float(row["build_ms"])) + 0.8, ax2["left"] + ax2["width"] + 0.7), y, f"{float(row['build_ms']):.2f}", size=FONT["endpoint"], anchor="start", cls="total-label"))

    elems.append("</svg>")
    return "\n".join(elems) + "\n"

def write_report(
    svg_path: Path,
    png_path: Path,
    preview_path: Path,
    pdf_path: Path,
    source_table: Path,
    construction_source: Path,
    sum_checks: list[dict[str, float | str]],
) -> Path:
    svg_text = svg_path.read_text(encoding="utf-8")
    report = {
        "figure": "Figure_04_ownership_speed_audit",
        "canvas_mm": {"width": MM_W, "height": MM_H},
        "workflow": "native editable SVG generated for Inkscape editing; PDF/PNG previews are derived from the current SVG",
        "font_family_declared": FONT_FAMILY,
        "font_sizes_pt": FONT_PT,
        "data_sources": {
            "panel_a_speedup": "geovoronoi_fv_cmame_text_master_submission_audit_final.tex Table~\\ref{tab:roi-jfa-cpu-reference}",
            "panel_b_construction_timing": construction_source.relative_to(ROOT).as_posix() if construction_source.is_relative_to(ROOT) else construction_source.name,
            "source_table": source_table.relative_to(ROOT).as_posix() if source_table.is_relative_to(ROOT) else source_table.name,
        },
        "uncertainty_data_available": False,
        "axes_rectangles_mm": {
            "panel_a": {"left": 28.0, "top": 18.2, "width": 58.0, "height": 41.5},
            "panel_b": {"left": 110.5, "top": 18.2, "width": 62.0, "height": 41.5},
            "setup_column": "removed from figure; setup_ms remains in source data and manuscript table",
        },
        "checks": [
            {"name": "svg_width_exact_183mm", "passed": 'width="183.0mm"' in svg_text},
            {"name": "svg_height_exact_76mm", "passed": 'height="76.0mm"' in svg_text},
            {"name": "svg_editable_text_present", "passed": "<text" in svg_text},
            {"name": "svg_contains_no_raster_image", "passed": "<image" not in svg_text.lower()},
            {"name": "panel_a_uses_horizontal_bars", "passed": "speedup-bar" in svg_text and "speedup-dot" not in svg_text},
            {"name": "setup_column_removed", "passed": "Setup (ms)" not in svg_text and "setup-value" not in svg_text and "setup-header" not in svg_text},
            {"name": "reference_line_and_label_red", "passed": "#D62728" in svg_text and "reference-line" in svg_text and "reference-label" in svg_text},
            {"name": "both_panel_y_axes_present", "passed": svg_text.count("y-axis") >= 2},
            {"name": "case_label_columns_consistent", "passed": "panel-a-case" in svg_text and "panel-b-case" in svg_text},
            {"name": "legend_swatch_count", "passed": svg_text.count("legend-swatch") == 3},
            {"name": "maze_stage_resolved_present", "passed": "Maze" in {str(row["case"]) for row in sum_checks} and "missing-note" not in svg_text},
            {"name": "stacked_bar_widths_use_unrounded_source", "passed": all(float(row["absolute_difference_ms"]) <= 5e-4 for row in sum_checks), "evidence": sum_checks},
            {
                "name": "outputs_written",
                "passed": svg_path.exists() and pdf_path.exists(),
                "evidence": {
                    "svg": svg_path.relative_to(ROOT).as_posix() if svg_path.is_relative_to(ROOT) else svg_path.name,
                    "png": png_path.relative_to(ROOT).as_posix() if png_path.is_relative_to(ROOT) else png_path.name,
                    "pdf": pdf_path.relative_to(ROOT).as_posix() if pdf_path.is_relative_to(ROOT) else pdf_path.name,
                    "png_preview_exists": png_path.exists() and preview_path.exists(),
                },
            },
        ],
    }
    report_path = FIG / "Figure_04_layout_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def build_figure04() -> dict[str, str]:
    FIG.mkdir(parents=True, exist_ok=True)
    build_rows, construction_source, sum_checks = read_build_rows()
    source_table = make_source_table(build_rows, construction_source)
    svg_path = FIG / "Figure_04_ownership_speed_audit.svg"
    pdf_path = FIG / "Figure_04_ownership_speed_audit.pdf"
    png_path = FIG / "Figure_04_ownership_speed_audit.png"
    preview_path = FIG / "Figure_04_ownership_speed_audit_preview.png"
    warning_path = FIG / "Figure_04_export_warning.txt"
    svg_path.write_text(build_svg(build_rows), encoding="utf-8")
    if warning_path.exists():
        warning_path.unlink()
    export_notes: list[str] = []
    try:
        import cairosvg

        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path), output_width=MM_W * 72 / 25.4, output_height=MM_H * 72 / 25.4)
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=round(MM_W / 25.4 * 300), output_height=round(MM_H / 25.4 * 300))
        preview_path.write_bytes(png_path.read_bytes())
        export_notes.append("CairoSVG exported PDF and PNG")
    except Exception as cairo_exc:
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPDF

            drawing = svg2rlg(str(svg_path))
            renderPDF.drawToFile(drawing, str(pdf_path))
            export_notes.append("CairoSVG unavailable; svglib/reportlab exported PDF from current SVG")
        except Exception as pdf_exc:
            export_notes.append(f"PDF export failed after CairoSVG failure: {pdf_exc!r}")
        export_notes.append(f"PNG export not available inside this Python runtime: {cairo_exc!r}")
        warning_path.write_text("\n".join(export_notes), encoding="utf-8")
    report_path = write_report(svg_path, png_path, preview_path, pdf_path, source_table, construction_source, sum_checks)
    return {"svg": svg_path.relative_to(ROOT).as_posix() if svg_path.is_relative_to(ROOT) else svg_path.name, "pdf": pdf_path.relative_to(ROOT).as_posix() if pdf_path.is_relative_to(ROOT) else pdf_path.name, "png": png_path.relative_to(ROOT).as_posix() if png_path.is_relative_to(ROOT) else png_path.name, "preview": str(preview_path), "report": str(report_path)}


if __name__ == "__main__":
    print(json.dumps(build_figure04(), indent=2))

