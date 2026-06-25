#!/usr/bin/env python3
"""Draw Figure 07: sampled-state projection onto a conservative face-flux field.

This version is a full redraw.  It avoids the previous dashboard/card language
and uses a fixed 183 mm x 100 mm grid, explicit panel coordinates, and a small
edge table so panel (c) cannot contradict the values it encodes.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["mathtext.default"] = "it"


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cmame_artifacts" / "figures"
DATA_DIR = ROOT / "cmame_artifacts" / "figure_source_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SVG = OUT_DIR / "Figure_07_state_projection.svg"
PDF = OUT_DIR / "Figure_07_state_projection.pdf"
PNG = OUT_DIR / "Figure_07_state_projection.png"
PNG_GRAY = OUT_DIR / "Figure_07_state_projection.grayscale.png"
CAPTION = OUT_DIR / "Figure_07_state_projection.caption.tex"
CHECKS = OUT_DIR / "Figure_07_state_projection_checks.txt"
SCENE_JSON = DATA_DIR / "Figure_07_state_projection_scene.json"
SYMBOL_JSON = DATA_DIR / "Figure_07_state_projection_symbols.json"
EDGE_CSV = DATA_DIR / "Figure_07_state_projection_edge_table.csv"

INKSCAPE = Path(os.environ.get("INKSCAPE_BIN", "inkscape"))

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

W_MM, H_MM = 183.0, 59.0
PT = 0.352777778
PANEL_MARGIN = 6.0
PANEL_GAP = 5.0
PANEL_W = (W_MM - 2 * PANEL_MARGIN - 2 * PANEL_GAP) / 3

PANELS = {
    "a": {"x": PANEL_MARGIN, "w": PANEL_W},
    "b": {"x": PANEL_MARGIN + PANEL_W + PANEL_GAP, "w": PANEL_W},
    "c": {"x": PANEL_MARGIN + 2 * (PANEL_W + PANEL_GAP), "w": PANEL_W},
}
TITLE_Y = 8.4
BODY_Y = 19.0
BODY_H = 36.0
FLOW_Y = 33.2
SECTION_TITLE_Y = BODY_Y + 1.0
BODY_LABEL_Y = 44.8
LEGEND_Y = 51.1
SECTION_TITLE_PT = 6.15
BODY_LABEL_PT = 5.90
MINOR_LABEL_PT = 5.45
FLOW_ARROW_LEN = 2.70
FLOW_ARROW_WIDTH = 0.42

COL = {
    "ink": "#20242A",
    "muted": "#5D6874",
    "grid": "#CCD6DD",
    "light_grid": "#E6ECEF",
    "hair": "#BFC9D0",
    "panel_rule": "#D8E0E5",
    "cell_blue": "#CFE7EC",
    "cell_lav": "#DDD2EA",
    "cell_peach": "#F3DDC8",
    "edge_blue": "#2C6F9E",
    "edge_orange": "#B86C28",
    "edge_green": "#2F8C78",
    "edge_value": "#116A5B",
    "edge_purple": "#6D56A5",
    "node": "#27313A",
    "white": "#FFFFFF",
}

SYMBOLS = {
    "cell_graph": "G_f",
    "cell_face_set": "F_ij",
    "facelet": "gamma",
    "target_facelet_flux": "phi_gamma^{t,loc}",
    "target_edge_flux": "phi_e^t",
    "projected_flux": "phi^{proj}",
    "projection_residual": "r^{proj}",
    "incidence": "B",
    "weight": "W",
    "node_state": "u_i",
}

NODE_POS = {
    "A": (0.14, 0.22),
    "B": (0.50, 0.18),
    "C": (0.86, 0.25),
    "D": (0.22, 0.62),
    "E": (0.56, 0.56),
    "F": (0.82, 0.72),
}

# The projected values are deliberately non-zero while B^T phi = 0, avoiding
# the earlier contradiction where zero residual was shown as zero edge flux.
EDGE_TABLE = [
    {"edge_id": "e1", "u": "A", "v": "B", "target": 0.66, "projected": 0.25, "target_label": "above", "projected_label": "above"},
    {"edge_id": "e2", "u": "B", "v": "C", "target": -0.52, "projected": -0.20, "target_label": "above", "projected_label": "above"},
    {"edge_id": "e3", "u": "A", "v": "D", "target": 0.59, "projected": -0.25, "target_label": "left", "projected_label": "left"},
    {"edge_id": "e4", "u": "B", "v": "E", "target": 0.11, "projected": 0.45, "target_label": "right_high", "projected_label": "right_high"},
    {"edge_id": "e5", "u": "D", "v": "E", "target": -0.11, "projected": -0.25, "target_label": "below", "projected_label": "below"},
    {"edge_id": "e6", "u": "E", "v": "F", "target": 0.14, "projected": 0.20, "target_label": "below_right_far", "projected_label": "below_right_far"},
    {"edge_id": "e7", "u": "C", "v": "F", "target": -0.11, "projected": -0.20, "target_label": "right_lower", "projected_label": "right_low"},
]


def svg_el(tag: str, **attrs) -> ET.Element:
    return ET.Element(f"{{{NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attrs.items() if v is not None})


def sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    child = svg_el(tag, **attrs)
    parent.append(child)
    return child


def style(**items) -> str:
    return ";".join(f"{k.replace('_', '-')}:{v}" for k, v in items.items() if v is not None)


def add_text(
    parent: ET.Element,
    x: float,
    y: float,
    s: str,
    size_pt: float = 6.5,
    weight: str = "400",
    fill: str = COL["ink"],
    anchor: str = "start",
    italic: bool = False,
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> ET.Element:
    st = style(
        font_family="Helvetica,Arial,sans-serif",
        font_size=f"{size_pt * PT:.3f}px",
        font_weight=weight,
        font_style="italic" if italic else "normal",
        fill=fill,
        stroke=stroke,
        stroke_width=f"{stroke_width:.3f}" if stroke_width is not None else None,
        paint_order="stroke fill" if stroke is not None else None,
        stroke_linejoin="round" if stroke is not None else None,
    )
    e = sub(parent, "text", x=f"{x:.3f}", y=f"{y:.3f}", text_anchor=anchor, style=st)
    e.text = s
    return e


def add_section_title(parent: ET.Element, x: float, y: float, s: str, anchor: str = "middle") -> ET.Element:
    return add_text(parent, x, y, s, size_pt=SECTION_TITLE_PT, weight="400", fill=COL["muted"], anchor=anchor)


def add_body_label(parent: ET.Element, x: float, y: float, s: str, fill: str = COL["muted"], anchor: str = "middle") -> ET.Element:
    return add_text(parent, x, y, s, size_pt=BODY_LABEL_PT, weight="400", fill=fill, anchor=anchor)


def add_minor_label(parent: ET.Element, x: float, y: float, s: str, fill: str = COL["muted"], anchor: str = "middle") -> ET.Element:
    return add_text(parent, x, y, s, size_pt=MINOR_LABEL_PT, weight="400", fill=fill, anchor=anchor)


def add_math(
    parent: ET.Element,
    x: float,
    y: float,
    tex: str,
    size_pt: float = 6.5,
    fill: str = COL["ink"],
    anchor: str = "start",
) -> ET.Element:
    """Add TeX-like math as editable vector paths in SVG coordinates."""
    prop = FontProperties(family="DejaVu Sans")
    text_path = TextPath((0, 0), f"${tex}$", size=size_pt, prop=prop, usetex=False)
    verts = text_path.vertices
    if len(verts) == 0:
        return sub(parent, "g")
    min_x = float(verts[:, 0].min())
    max_x = float(verts[:, 0].max())
    # Avoid TextPath.get_extents() here: on this Windows/Matplotlib stack it can
    # trigger a native crash for a single gamma glyph.
    if anchor == "middle":
        x_shift = x - (min_x + (max_x - min_x) / 2.0) * PT
    elif anchor == "end":
        x_shift = x - max_x * PT
    else:
        x_shift = x - min_x * PT
    y_shift = y

    def map_xy(px: float, py: float) -> tuple[float, float]:
        return x_shift + px * PT, y_shift - py * PT

    chunks: list[str] = []
    for verts, code in text_path.iter_segments():
        if code == MplPath.MOVETO:
            px, py = map_xy(verts[0], verts[1])
            chunks.append(f"M {px:.3f} {py:.3f}")
        elif code == MplPath.LINETO:
            px, py = map_xy(verts[0], verts[1])
            chunks.append(f"L {px:.3f} {py:.3f}")
        elif code == MplPath.CURVE3:
            x1, y1 = map_xy(verts[0], verts[1])
            x2, y2 = map_xy(verts[2], verts[3])
            chunks.append(f"Q {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}")
        elif code == MplPath.CURVE4:
            x1, y1 = map_xy(verts[0], verts[1])
            x2, y2 = map_xy(verts[2], verts[3])
            x3, y3 = map_xy(verts[4], verts[5])
            chunks.append(f"C {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {x3:.3f} {y3:.3f}")
        elif code == MplPath.CLOSEPOLY:
            chunks.append("Z")
    return sub(parent, "path", d=" ".join(chunks), fill=fill, stroke="none")


def add_symbol_text(
    parent: ET.Element,
    x: float,
    y: float,
    pieces: list[dict[str, object]],
    size_pt: float = 6.0,
    fill: str = COL["ink"],
    anchor: str = "start",
) -> None:
    """Draw simple math labels with real SVG tspans instead of TeX source text."""
    char_w = size_pt * PT * 0.52
    sub_scale = 0.68
    width = 0.0
    for piece in pieces:
        text = str(piece["text"])
        role = str(piece.get("role", "main"))
        width += len(text) * char_w * (sub_scale if role in {"sub", "sup"} else 1.0)
    if anchor == "middle":
        x0 = x - width / 2.0
    elif anchor == "end":
        x0 = x - width
    else:
        x0 = x
    st = style(
        font_family="Helvetica,Arial,sans-serif",
        font_size=f"{size_pt * PT:.3f}px",
        font_weight="400",
        font_style="italic",
        fill=fill,
    )
    e = sub(parent, "text", x=f"{x0:.3f}", y=f"{y:.3f}", text_anchor="start", style=st)
    e.text = ""
    for piece in pieces:
        text = str(piece["text"])
        role = str(piece.get("role", "main"))
        if role == "sub":
            sub(e, "tspan", baseline_shift="sub", font_size=f"{size_pt * PT * sub_scale:.3f}px").text = text
        elif role == "sup":
            sub(e, "tspan", baseline_shift="super", font_size=f"{size_pt * PT * sub_scale:.3f}px").text = text
        else:
            sub(e, "tspan").text = text


def sym(parent: ET.Element, x: float, y: float, name: str, size_pt: float = 6.0, fill: str = COL["ink"], anchor: str = "start") -> None:
    library: dict[str, list[dict[str, object]]] = {
        "G_f": [{"text": "G"}, {"text": "f", "role": "sub"}],
        "Omega_i": [{"text": "Ω"}, {"text": "i", "role": "sub"}],
        "gamma": [{"text": "γ"}],
        "F_ij": [{"text": "F"}, {"text": "ij", "role": "sub"}],
        "phi_e": [{"text": "φ"}, {"text": "e", "role": "sub"}],
        "phi_gamma_v": [{"text": "φ"}, {"text": "γ", "role": "sub"}, {"text": "v", "role": "sup"}],
        "phi_gamma_p": [{"text": "φ"}, {"text": "γ", "role": "sub"}, {"text": "p", "role": "sup"}],
        "B_phi": [{"text": "B"}, {"text": "T", "role": "sup"}, {"text": "φ"}],
    }
    add_symbol_text(parent, x, y, library[name], size_pt=size_pt, fill=fill, anchor=anchor)


def add_line(parent: ET.Element, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 0.35, dash: str | None = None) -> ET.Element:
    return sub(
        parent,
        "line",
        x1=f"{x1:.3f}",
        y1=f"{y1:.3f}",
        x2=f"{x2:.3f}",
        y2=f"{y2:.3f}",
        stroke=color,
        stroke_width=f"{width:.3f}",
        stroke_linecap="round",
        stroke_dasharray=dash,
        fill="none",
    )


def add_rect(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = "none",
    stroke: str = COL["hair"],
    sw: float = 0.35,
    rx: float = 0.0,
    opacity: float | None = None,
) -> ET.Element:
    return sub(
        parent,
        "rect",
        x=f"{x:.3f}",
        y=f"{y:.3f}",
        width=f"{w:.3f}",
        height=f"{h:.3f}",
        fill=fill,
        stroke=stroke,
        stroke_width=f"{sw:.3f}",
        rx=f"{rx:.3f}",
        opacity=f"{opacity:.3f}" if opacity is not None else None,
    )


def add_circle(parent: ET.Element, cx: float, cy: float, r: float, fill: str, stroke: str = COL["white"], sw: float = 0.35) -> ET.Element:
    return sub(parent, "circle", cx=f"{cx:.3f}", cy=f"{cy:.3f}", r=f"{r:.3f}", fill=fill, stroke=stroke, stroke_width=f"{sw:.3f}")


def arrow_marker(defs: ET.Element, marker_id: str, color: str) -> None:
    marker = sub(defs, "marker", id=marker_id, viewBox="0 0 10 10", refX="8", refY="5", markerWidth="4.2", markerHeight="4.2", orient="auto-start-reverse")
    sub(marker, "path", d="M 0 0 L 10 5 L 0 10 z", fill=color, stroke="none")


def add_arrow(parent: ET.Element, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 0.55, marker: str = "arrowGrey") -> None:
    sub(
        parent,
        "line",
        x1=f"{x1:.3f}",
        y1=f"{y1:.3f}",
        x2=f"{x2:.3f}",
        y2=f"{y2:.3f}",
        stroke=color,
        stroke_width=f"{width:.3f}",
        stroke_linecap="round",
        marker_end=f"url(#{marker})",
    )


def add_flow_arrow(parent: ET.Element, cx: float, y: float) -> None:
    add_arrow(
        parent,
        cx - FLOW_ARROW_LEN / 2,
        y,
        cx + FLOW_ARROW_LEN / 2,
        y,
        "#9AA6B0",
        width=FLOW_ARROW_WIDTH,
        marker="arrowGrey",
    )


def fmt_flux(v: float) -> str:
    if abs(v) < 0.005:
        return "0.00"
    return f"{'+' if v > 0 else '−'}{abs(v):.2f}"


def grid(parent: ET.Element, x: float, y: float, cols: int, rows: int, s: float, stroke: str = COL["grid"], sw: float = 0.25, fill: str = "#FAFCFD") -> None:
    add_rect(parent, x, y, cols * s, rows * s, fill=fill, stroke=stroke, sw=sw)
    for k in range(1, cols):
        add_line(parent, x + k * s, y, x + k * s, y + rows * s, stroke, sw)
    for k in range(1, rows):
        add_line(parent, x, y + k * s, x + cols * s, y + k * s, stroke, sw)


def map_nodes(x: float, y: float, w: float, h: float) -> dict[str, tuple[float, float]]:
    return {k: (x + px * w, y + py * h) for k, (px, py) in NODE_POS.items()}


def edge_label_position(nodes: dict[str, tuple[float, float]], edge: dict[str, object], which: str) -> tuple[float, float, str]:
    u, v = str(edge["u"]), str(edge["v"])
    x1, y1 = nodes[u]
    x2, y2 = nodes[v]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    orient = str(edge[f"{which}_label"])
    offsets = {
        "above": (0.0, -0.95, "middle"),
        "below": (0.0, 1.25, "middle"),
        "left": (-1.05, 0.15, "end"),
        "right": (1.05, 0.15, "start"),
        "right_high": (1.25, -0.95, "start"),
        "right_low": (1.15, 0.95, "start"),
        "right_lower": (1.15, 2.10, "start"),
        "below_left": (-0.75, 1.10, "end"),
        "below_left_far": (-0.85, 1.25, "end"),
        "below_right_far": (0.95, 1.25, "start"),
    }
    dx, dy, anchor = offsets[orient]
    return mx + dx, my + dy, anchor


def draw_flux_network(root: ET.Element, x: float, y: float, w: float, h: float, value_key: str, title: str) -> None:
    nodes = map_nodes(x, y, w, h)
    if value_key == "target":
        add_section_title(root, x + w / 2, SECTION_TITLE_Y, "Target flux field")
    else:
        add_section_title(root, x + w / 2, SECTION_TITLE_Y, "Conservative flux field")
    for e in EDGE_TABLE:
        x1, y1 = nodes[str(e["u"])]
        x2, y2 = nodes[str(e["v"])]
        add_line(root, x1, y1, x2, y2, "#9EAAB3", 0.35)
    for e in EDGE_TABLE:
        val = float(e[value_key])
        if abs(val) < 0.005:
            continue
        u, v = str(e["u"]), str(e["v"])
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        if val < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
        frac = 0.25
        xa = x1 + (x2 - x1) * frac
        ya = y1 + (y2 - y1) * frac
        xb = x1 + (x2 - x1) * 0.73
        yb = y1 + (y2 - y1) * 0.73
        add_arrow(root, xa, ya, xb, yb, COL["edge_green"], width=0.58, marker="arrowGreen")
    for name, (cx, cy) in nodes.items():
        add_circle(root, cx, cy, 0.72, "#FFFFFF", stroke=COL["node"], sw=0.35)
    label_key = "target" if value_key == "target" else "projected"
    for e in EDGE_TABLE:
        lx, ly, anchor = edge_label_position(nodes, e, label_key)
        add_text(
            root,
            lx,
            ly,
            fmt_flux(float(e[value_key])),
            size_pt=4.75,
            weight="500",
            fill=COL["edge_value"],
            anchor=anchor,
            stroke=COL["white"],
            stroke_width=0.75,
        )


def draw_panel_a(root: ET.Element) -> None:
    x, w = PANELS["a"]["x"], PANELS["a"]["w"]
    y0 = BODY_Y + 1.0
    s = 5.1
    grid_w = 3 * s
    left_center = x + 11.85
    right_center = x + w - 11.85
    left_x = left_center - grid_w / 2
    right_x = right_center - grid_w / 2
    gy = y0 + 4.2

    add_section_title(root, left_center, SECTION_TITLE_Y, "Observed samples")
    add_section_title(root, right_center, SECTION_TITLE_Y, "Completed on FV graph")

    grid(root, left_x, gy, 3, 3, s, fill="#FBFDFE")
    add_rect(root, left_x, gy, s, 2 * s, fill="#D7ECF1", stroke=COL["edge_blue"], sw=0.38)
    observations = [
        (left_x + 0.55 * s, gy + 0.65 * s, 0.75, -0.70),
        (left_x + 1.60 * s, gy + 1.10 * s, 0.95, -0.35),
        (left_x + 0.82 * s, gy + 2.30 * s, -0.55, 0.52),
        (left_x + 2.28 * s, gy + 0.45 * s, 0.62, 0.56),
    ]
    for px, py, dx, dy in observations:
        norm = max((dx * dx + dy * dy) ** 0.5, 0.01)
        ux, uy = dx / norm, dy / norm
        add_arrow(root, px + ux * 1.25, py + uy * 1.25, px + ux * 2.05, py + uy * 2.05, COL["node"], width=0.38, marker="arrowGrey")
    for px, py, _dx, _dy in observations:
        add_circle(root, px, py, 0.60, COL["node"], stroke=COL["white"], sw=0.18)
    add_body_label(root, left_center, BODY_LABEL_Y, "samples at fixed cells")

    grid(root, right_x, gy, 3, 3, s, fill="#FBFDFE")
    cell_centers = {(c, r): (right_x + (c + 0.5) * s, gy + (r + 0.5) * s) for r in range(3) for c in range(3)}
    direct_nodes = [(0, 0), (1, 1), (2, 0), (2, 2)]
    complete_nodes = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2)]
    links = [((0, 0), (1, 1)), ((1, 1), (2, 2)), ((1, 1), (2, 1)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2))]
    for a, b in links:
        (x1, y1), (x2, y2) = cell_centers[a], cell_centers[b]
        add_line(root, x1, y1, x2, y2, "#9EACB7", 0.32)
    (x1, y1), (x2, y2) = cell_centers[(0, 0)], cell_centers[(1, 1)]
    add_line(root, x1, y1, x2, y2, COL["edge_orange"], 0.72)
    for n in complete_nodes:
        cx, cy = cell_centers[n]
        add_circle(root, cx, cy, 0.72, "#FFFFFF", stroke=COL["edge_blue"], sw=0.42)
    for n in direct_nodes:
        cx, cy = cell_centers[n]
        add_circle(root, cx, cy, 0.78, COL["edge_blue"], stroke=COL["white"], sw=0.35)
    add_body_label(root, right_center, BODY_LABEL_Y, "harmonic fill on graph")

    add_circle(root, x + 14.0, gy + 24.7, 0.65, COL["edge_blue"], stroke=COL["white"], sw=0.25)
    add_minor_label(root, x + 15.2, LEGEND_Y, "observed", anchor="start")
    add_circle(root, x + 28.6, gy + 24.7, 0.65, "#FFFFFF", stroke=COL["edge_blue"], sw=0.4)
    add_minor_label(root, x + 29.8, LEGEND_Y, "completed", anchor="start")


def draw_panel_b(root: ET.Element) -> None:
    x = PANELS["b"]["x"]
    y0 = BODY_Y + 1.0
    geom_center = x + 9.00
    est_center = x + PANELS["b"]["w"] / 2
    blend_center = x + PANELS["b"]["w"] - 9.00
    geom_x = geom_center - 7.4
    est_x = est_center - 7.15
    blend_x = blend_center - 7.0
    gy = y0 + 5.0

    add_section_title(root, geom_center, SECTION_TITLE_Y, "Local edge")
    add_section_title(root, est_center, SECTION_TITLE_Y, "Estimates")
    add_section_title(root, blend_center, SECTION_TITLE_Y, "Aggregation")

    add_rect(root, geom_x, gy, 7.3, 16.0, fill=COL["cell_blue"], stroke=COL["edge_blue"], sw=0.35)
    add_rect(root, geom_x + 7.5, gy, 7.3, 16.0, fill=COL["cell_lav"], stroke=COL["edge_purple"], sw=0.35)
    add_rect(root, geom_x + 7.12, gy + 2.6, 0.78, 10.8, fill=COL["edge_orange"], stroke="none")
    add_line(root, geom_x + 3.0, gy + 11.6, geom_x + 7.5, gy + 8.5, COL["edge_blue"], 0.55)
    add_line(root, geom_x + 12.2, gy + 11.6, geom_x + 7.5, gy + 8.5, COL["edge_purple"], 0.55)
    add_circle(root, geom_x + 3.0, gy + 11.6, 0.58, COL["node"], stroke=COL["white"], sw=0.22)
    add_circle(root, geom_x + 12.2, gy + 11.6, 0.58, COL["node"], stroke=COL["white"], sw=0.22)
    add_circle(root, geom_x + 7.5, gy + 8.5, 0.52, COL["edge_orange"], stroke=COL["white"], sw=0.22)
    add_body_label(root, geom_center, BODY_LABEL_Y, "facelet")

    add_rect(root, est_x, gy + 0.8, 14.3, 6.1, fill="#FFFFFF", stroke=COL["hair"], sw=0.35, rx=0.35)
    add_rect(root, est_x, gy + 9.2, 14.3, 6.1, fill="#FFFFFF", stroke=COL["hair"], sw=0.35, rx=0.35)
    add_minor_label(root, est_x + 7.15, gy + 4.55, "baseline flux", fill=COL["edge_blue"])
    add_minor_label(root, est_x + 7.15, gy + 12.95, "sample flux", fill=COL["edge_purple"])

    merge_cx, merge_cy = blend_x + 4.5, gy + 8.2
    sum_cx = blend_x + 12.2
    add_line(root, est_x + 14.3, gy + 3.7, merge_cx - 1.35, merge_cy - 0.95, COL["hair"], 0.32)
    add_line(root, est_x + 14.3, gy + 12.1, merge_cx - 1.35, merge_cy + 0.95, COL["hair"], 0.32)
    add_circle(root, merge_cx, merge_cy, 1.34, "#FFFFFF", stroke=COL["edge_green"], sw=0.50)
    label_x = sum_cx + 3.20
    add_minor_label(root, label_x, merge_cy - 2.05, "weight", fill=COL["edge_green"])
    add_line(root, merge_cx + 1.55, merge_cy, sum_cx - 1.95, merge_cy, COL["edge_green"], 0.62)
    add_circle(root, sum_cx, merge_cy, 1.25, "#FFFFFF", stroke=COL["edge_green"], sw=0.55)
    add_minor_label(root, sum_cx, merge_cy + 0.50, "\u03a3", fill=COL["edge_green"])
    add_minor_label(root, label_x, merge_cy + 3.55, "target", fill=COL["edge_green"])
    add_body_label(root, blend_center, BODY_LABEL_Y, "facelet sum")


def draw_panel_c(root: ET.Element) -> None:
    x = PANELS["c"]["x"]
    y0 = BODY_Y + 1.0
    graph_w = 15.6
    graph_h = 21.2
    graph_y = FLOW_Y - graph_h / 2
    box_w = 11.3
    box_h = 10.6
    left_center = x + 10.00
    box_center = x + PANELS["c"]["w"] / 2
    right_center = x + PANELS["c"]["w"] - 10.00
    left = (left_center - graph_w / 2, graph_y, graph_w, graph_h)
    box = (box_center - box_w / 2, FLOW_Y - box_h / 2, box_w, box_h)
    right = (right_center - graph_w / 2, graph_y, graph_w, graph_h)
    draw_flux_network(root, *left, value_key="target", title="")
    bx, by, bw, bh = box
    add_rect(root, bx, by, bw, bh, fill="#FFFFFF", stroke=COL["hair"], sw=0.42, rx=0.35)
    add_text(root, bx + bw / 2, by + 4.45, "closest", size_pt=5.30, anchor="middle", fill=COL["muted"])
    add_text(root, bx + bw / 2, by + 7.25, "conservative", size_pt=4.85, anchor="middle", fill=COL["muted"])
    flow_y = graph_y + graph_h / 2
    add_flow_arrow(root, (left[0] + left[2] + bx) / 2, flow_y)
    add_flow_arrow(root, (bx + bw + right[0]) / 2, flow_y)
    draw_flux_network(root, *right, value_key="projected", title="")
    add_body_label(root, right[0] + right[2] / 2, BODY_LABEL_Y, "cell balance = 0", fill=COL["edge_green"])


def compute_projected_residual() -> dict[str, float]:
    res = {n: 0.0 for n in NODE_POS}
    for edge in EDGE_TABLE:
        u, v = str(edge["u"]), str(edge["v"])
        val = float(edge["projected"])
        res[u] -= val
        res[v] += val
    return res


def write_caption() -> None:
    CAPTION.write_text(
        r"""\caption{\textbf{Projection of sampled states to a conservative face-flux field.}
The sampled state is first attached to fixed GeoVoronoi--FV cells and completed on the finite-volume graph.  For each cell--cell edge, facelet-local reconstructions provide baseline and sample-supported flux estimates, which are blended and summed over \(\mathcal F_{ij}\) to form a target edge flux \(\phi_e^t\).  The final projection solves the weighted closest conservative edge field subject to \(B^\top\phi=0\).  In panel (c), arrow direction follows the sign of the edge flux and the adjacent numbers give the edge-flux values; \(r^{\mathrm{proj}}=0\) denotes conservative cell balance after projection, not zero edge flux.}""",
        encoding="utf-8",
    )


def run_exports() -> None:
    if not INKSCAPE.exists():
        raise FileNotFoundError(f"Inkscape not found: {INKSCAPE}")
    subprocess.run([str(INKSCAPE), str(SVG), "--export-type=pdf", f"--export-filename={PDF}"], check=True)
    subprocess.run([str(INKSCAPE), str(SVG), "--export-type=png", "--export-dpi=600", f"--export-filename={PNG}"], check=True)
    subprocess.run([str(INKSCAPE), str(SVG), "--export-type=png", "--export-dpi=600", "--export-background=#FFFFFF", f"--export-filename={PNG_GRAY}"], check=True)


def write_checks(root: ET.Element) -> None:
    svg_text = SVG.read_text(encoding="utf-8")
    forbidden = ["Gf", "Fij", "φet", "φproj", "rproj"]
    bad = [tok for tok in forbidden if tok in svg_text]
    if bad:
        raise AssertionError(f"Forbidden tokens in SVG: {bad}")
    if "<image" in svg_text:
        raise AssertionError("SVG contains raster image tags")

    graph_w = 15.6
    graph_h = 21.2
    graph_y = FLOW_Y - graph_h / 2
    panel_c = PANELS["c"]
    target_nodes = map_nodes(panel_c["x"] + 10.00 - graph_w / 2, graph_y, graph_w, graph_h)
    projected_nodes = map_nodes(panel_c["x"] + panel_c["w"] - 10.00 - graph_w / 2, graph_y, graph_w, graph_h)
    expected_dx = projected_nodes["A"][0] - target_nodes["A"][0]
    for key in NODE_POS:
        dx = projected_nodes[key][0] - target_nodes[key][0]
        dy = projected_nodes[key][1] - target_nodes[key][1]
        assert abs(dx - expected_dx) < 1e-9 and abs(dy) < 1e-9, "Panel c networks are not identical translations"

    arrow_rows = []
    for edge in EDGE_TABLE:
        for k in ("target", "projected"):
            lab = fmt_flux(float(edge[k]))
            if lab != "0.00":
                assert (lab[0] == "+" or lab[0] == "−") and len(lab.split(".")[1]) == 2
            arrow_rows.append({"edge_id": edge["edge_id"], "value_key": k, "value": edge[k], "label": lab, "colored_arrow": abs(float(edge[k])) >= 0.005})

    projected_residual = compute_projected_residual()
    max_abs_resid = max(abs(v) for v in projected_residual.values())
    assert max_abs_resid < 1e-12, "Projected fluxes are not conservative"

    report = {
        "canvas_mm": [W_MM, H_MM],
        "panel_x_ranges_mm": {k: [v["x"], v["x"] + v["w"]] for k, v in PANELS.items()},
        "body_zone_mm": [BODY_Y, BODY_Y + BODY_H],
        "formula_zone_mm": None,
        "forbidden_tokens": bad,
        "svg_image_tags": svg_text.count("<image"),
        "projected_residual": projected_residual,
        "max_abs_projected_residual": max_abs_resid,
        "flux_arrow_consistency": arrow_rows,
        "note": "Generated verification report for Figure 07 state projection.",
    }
    CHECKS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    root = svg_el("svg", width=f"{W_MM}mm", height=f"{H_MM}mm", viewBox=f"0 0 {W_MM} {H_MM}", version="1.1")
    defs = sub(root, "defs")
    arrow_marker(defs, "arrowGrey", "#8D99A4")
    arrow_marker(defs, "arrowGreen", COL["edge_green"])

    sub(root, "rect", x="0", y="0", width=f"{W_MM}", height=f"{H_MM}", fill="#FFFFFF", stroke="none")
    add_text(root, W_MM / 2, 7.2, "State samples are projected onto a conservative face-flux field", size_pt=8.6, weight="600", anchor="middle")

    for key, title in [("a", "Sampled states"), ("b", "Face interpolation"), ("c", "Conservative projection")]:
        x, w = PANELS[key]["x"], PANELS[key]["w"]
        add_text(root, x - 0.2, TITLE_Y + 3.3, key, size_pt=8.0, weight="700")
        add_text(root, x + w / 2, TITLE_Y + 3.3, title, size_pt=7.0, weight="500", anchor="middle")
    add_flow_arrow(root, PANELS["a"]["x"] + PANELS["a"]["w"] + PANEL_GAP / 2, FLOW_Y)
    add_flow_arrow(root, PANELS["b"]["x"] + PANELS["b"]["w"] + PANEL_GAP / 2, FLOW_Y)

    draw_panel_a(root)
    draw_panel_b(root)
    draw_panel_c(root)

    SVG.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    SCENE_JSON.write_text(json.dumps({"canvas_mm": [W_MM, H_MM], "panels": PANELS, "vertical_zones_mm": {"body_y": BODY_Y, "body_h": BODY_H}, "flux_edges": EDGE_TABLE, "symbols": SYMBOLS}, indent=2), encoding="utf-8")
    SYMBOL_JSON.write_text(json.dumps(SYMBOLS, indent=2), encoding="utf-8")
    with EDGE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EDGE_TABLE[0].keys()))
        writer.writeheader()
        writer.writerows(EDGE_TABLE)

    write_caption()
    run_exports()
    write_checks(root)


if __name__ == "__main__":
    main()

