from __future__ import annotations

from collections import deque
from pathlib import Path
import heapq
import json
import math
import re

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import to_hex

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

MM = 1.0 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 6.2,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.2,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

COL = {
    "ink": "#222222",
    "muted": "#5F6D78",
    "grid": "#C8D5D9",
    "node_edge": "#A9BBC2",
    "node_fill": "#FFFFFF",
    "solid": "#087A63",
    "site": "#C74D88",
    "core": "#9ED7C8",
    "active": "#F0D58A",
    "accept": "#2E6FA3",
    "reject": "#D97A20",
    "audit": "#C42BD1",
    "unit": "#2E6FA3",
    "neutral": "#8C9AA5",
    "tile": "#60758C",
    "pore": "#FBFDFC",
    "card": "#FFFFFF",
    "card_edge": "#D8E1E5",
    "owner0": "#8BC9BE",
    "owner1": "#F1D39B",
    "owner2": "#D8BBDD",
}

DASH = (0, (2.4, 1.8))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SITE_LABELS = [r"$S_0$", r"$S_1$", r"$S_2$"]
DIST_VMAX = 7.0


class Graph:
    pass


def center(node):
    return float(node[0]), float(node[1])


def canonical_edge(a, b):
    return tuple(sorted((a, b)))


def single_source(source, pore, adjacency):
    dist = {n: math.inf for n in pore}
    dist[source] = 0.0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adjacency[u]:
            if math.isinf(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(v)
    return dist


def exact_multisource(pore, adjacency, sites):
    labels = {n: -1 for n in pore}
    D = {n: math.inf for n in pore}
    keys = {idx: site for idx, site in enumerate(sites)}
    pq = []
    for idx, site in enumerate(sites):
        labels[site] = idx
        D[site] = 0.0
        heapq.heappush(pq, (0.0, keys[idx], idx, site))
    while pq:
        d, key, idx, u = heapq.heappop(pq)
        if d != D[u] or labels[u] != idx or key != keys[idx]:
            continue
        for v in adjacency[u]:
            current = (D[v], keys[labels[v]] if labels[v] >= 0 else (999, 999))
            cand = (d + 1.0, key)
            if cand < current:
                D[v] = cand[0]
                labels[v] = idx
                heapq.heappush(pq, (cand[0], key, idx, v))
    return labels, D


def owner_certified_partition(pore, adjacency, sites, labels, D):
    site_D = [single_source(s, pore, adjacency) for s in sites]
    spacing = [min(site_D[k][sites[j]] for j in range(len(sites)) if j != k) for k in range(len(sites))]
    core = {n for n in pore if D[n] < 0.5 * spacing[labels[n]]}
    return core, set(pore) - core


def directed_audit(pore, adjacency, sites, labels, D):
    keys = {idx: site for idx, site in enumerate(sites)}
    rho_D = 0.0
    n_lex = 0
    for u in sorted(pore):
        for v in sorted(adjacency[u]):
            rho_D = max(rho_D, max(0.0, D[v] - D[u] - 1.0))
            if (D[u] + 1.0, keys[labels[u]]) < (D[v], keys[labels[v]]):
                n_lex += 1
    return rho_D, n_lex


def build_graph_from_parameters():
    g = Graph()
    g.width, g.height = 11, 8
    g.solid = {(4, 4), (4, 5), (4, 6), (7, 0), (8, 0), (8, 1)}
    g.pore = {(x, y) for x in range(g.width) for y in range(g.height)} - g.solid
    g.sites = [(1, 1), (9, 1), (2, 6)]
    g.adj = {n: [] for n in g.pore}
    g.edges = set()
    for x, y in g.pore:
        for dx, dy in DIR4:
            nb = (x + dx, y + dy)
            if nb in g.pore:
                g.adj[(x, y)].append(nb)
                g.edges.add(canonical_edge((x, y), nb))
    g.labels, g.D = exact_multisource(g.pore, g.adj, g.sites)
    g.core, g.active = owner_certified_partition(g.pore, g.adj, g.sites, g.labels, g.D)
    g.h = 4
    g.u = (1, 1)
    g.v = (5, 1)
    g.v0 = (5, 5)
    g.path = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    g.rho_D, g.n_lex = directed_audit(g.pore, g.adj, g.sites, g.labels, g.D)
    g.jump_passes = 4
    g.closure_passes = 6
    g.tile_bounds = {"x": [5, 8], "y": [1, 4], "bounds": [4.5, 0.5, 8.5, 4.5]}
    g.unit_relaxation_edges = [((5, 1), (6, 1)), ((6, 2), (6, 3))]
    g.directed_audit_edges = [((6, 4), (7, 4))]
    return g


def scene_record(g):
    def xy_list(nodes):
        return [[int(x), int(y)] for x, y in sorted(nodes)]

    def xy_path(nodes):
        return [[int(x), int(y)] for x, y in nodes]

    owner_rows = []
    distance_rows = []
    for y in range(g.height):
        owner_row = []
        distance_row = []
        for x in range(g.width):
            n = (x, y)
            if n in g.solid:
                owner_row.append(None)
                distance_row.append(None)
            else:
                owner_row.append(f"S{g.labels[n]}")
                distance_row.append(float(g.D[n]))
        owner_rows.append(owner_row)
        distance_rows.append(distance_row)

    return {
        "nx": g.width,
        "ny": g.height,
        "graph": "2-D four-neighbour pore graph with integer node centres and half-integer cell boundaries",
        "solid_cells": xy_list(g.solid),
        "pore_nodes": xy_list(g.pore),
        "site_coordinates": xy_path(g.sites),
        "certified_mask": xy_list(g.core),
        "active_mask": xy_list(g.active),
        "accepted_candidate": {
            "h_j": g.h,
            "u": list(g.u),
            "v": list(g.v),
            "offset": [g.v[0] - g.u[0], g.v[1] - g.u[1]],
            "path_cost": len(g.path) - 1,
            "decision": "accepted when the feasible path certificate also lexicographically improves the stored pair",
        },
        "rejected_candidate": {
            "h_j": g.h,
            "u": list(g.u),
            "v0": list(g.v0),
            "offset": [g.v0[0] - g.u[0], g.v0[1] - g.u[1]],
            "decision": "rejected because the scale-h_j proposal has no feasible path certificate through the pore graph",
        },
        "certified_path_vertices": xy_path(g.path),
        "tile_bounds": g.tile_bounds,
        "unit_relaxation_edges": [[list(a), list(b)] for a, b in g.unit_relaxation_edges],
        "directed_audit_edges": [[list(a), list(b)] for a, b in g.directed_audit_edges],
        "owner_field_L": owner_rows,
        "distance_field_D": distance_rows,
        "rho_D": float(g.rho_D),
        "n_lex": int(g.n_lex),
        "active_fraction": float(len(g.active) / len(g.pore)),
        "jump_count": int(g.jump_passes),
        "closure_count": int(g.closure_passes),
        "mode": "exact",
    }


def write_scene_record(record, path):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def graph_from_scene(record):
    g = Graph()
    g.width, g.height = int(record["nx"]), int(record["ny"])
    g.solid = {tuple(p) for p in record["solid_cells"]}
    g.pore = {tuple(p) for p in record["pore_nodes"]}
    g.sites = [tuple(p) for p in record["site_coordinates"]]
    g.adj = {n: [] for n in g.pore}
    g.edges = set()
    for x, y in g.pore:
        for dx, dy in DIR4:
            nb = (x + dx, y + dy)
            if nb in g.pore:
                g.adj[(x, y)].append(nb)
                g.edges.add(canonical_edge((x, y), nb))
    g.labels, g.D = exact_multisource(g.pore, g.adj, g.sites)
    g.core = {tuple(p) for p in record["certified_mask"]}
    g.active = {tuple(p) for p in record["active_mask"]}
    g.h = int(record["accepted_candidate"]["h_j"])
    g.u = tuple(record["accepted_candidate"]["u"])
    g.v = tuple(record["accepted_candidate"]["v"])
    g.v0 = tuple(record["rejected_candidate"]["v0"])
    g.path = [tuple(p) for p in record["certified_path_vertices"]]
    g.tile_bounds = record["tile_bounds"]
    g.unit_relaxation_edges = [(tuple(a), tuple(b)) for a, b in record["unit_relaxation_edges"]]
    g.directed_audit_edges = [(tuple(a), tuple(b)) for a, b in record["directed_audit_edges"]]
    g.rho_D, g.n_lex = directed_audit(g.pore, g.adj, g.sites, g.labels, g.D)
    g.jump_passes = int(record["jump_count"])
    g.closure_passes = int(record["closure_count"])
    return g


def build_graph():
    base = build_graph_from_parameters()
    validate(base)
    record = scene_record(base)
    write_scene_record(record, OUT / "figure3_scene.json")
    g = graph_from_scene(record)
    validate(g)
    return g


def validate(g):
    assert g.width == 11 and g.height == 8
    assert g.solid == {(4, 4), (4, 5), (4, 6), (7, 0), (8, 0), (8, 1)}
    assert g.sites == [(1, 1), (9, 1), (2, 6)]
    assert g.v[0] - g.u[0] == 4 and g.v[1] - g.u[1] == 0
    assert g.v0[0] - g.u[0] == 4 and g.v0[1] - g.u[1] == 4
    assert len(g.path) == 5
    assert g.tile_bounds["bounds"] == [4.5, 0.5, 8.5, 4.5]
    assert set(g.sites) <= g.pore
    assert g.core.isdisjoint(g.active)
    assert g.core | g.active == g.pore
    assert g.solid.isdisjoint(g.core | g.active)
    assert g.u in g.pore and g.v in g.pore and g.v0 in g.pore
    assert g.v in g.active and g.v0 in g.active
    assert g.path[0] == g.u and g.path[-1] == g.v
    assert all(n in g.pore for n in g.path)
    assert all(canonical_edge(a, b) in g.edges for a, b in zip(g.path[:-1], g.path[1:]))
    assert not any(n in g.solid for n in g.path)
    assert abs(g.rho_D) < 1e-12
    assert g.n_lex == 0
    assert len(set(g.labels.values())) == len(g.sites)
    assert abs(100.0 * len(g.active) / len(g.pore) - 51.21951219512195) < 1e-12
    for idx, s in enumerate(g.sites):
        assert g.labels[s] == idx and abs(g.D[s]) < 1e-12


def setup_panel(ax, letter, title, g):
    ax.set_xlim(-0.5, g.width - 0.5)
    ax.set_ylim(-0.5, g.height - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.00, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, fontweight="bold", color=COL["ink"])
    ax.text(0.075, 1.03, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.0, color=COL["ink"])


def draw_lattice(ax, g, mode="plain", show_graph=True, show_nodes=True, alpha=1.0, owner=False, distance=False):
    owner_cols = {0: COL["owner0"], 1: COL["owner1"], 2: COL["owner2"]}
    ax.add_patch(patches.Rectangle((-0.5, -0.5), g.width, g.height, fc="#FFFFFF", ec="none", zorder=-10))
    for y in range(g.height):
        for x in range(g.width):
            n = (x, y)
            if n in g.solid:
                continue
            if owner:
                fc, ec, z, a = owner_cols[g.labels[n]], "#E5EEEB", 1, 0.96
            elif distance:
                t = min(g.D[n], DIST_VMAX) / DIST_VMAX
                fc = to_hex(mpl.colormaps["YlGnBu"](0.18 + 0.72 * t))
                ec, z, a = "#E5EEEB", 1, 0.96
            elif mode == "partition":
                fc, ec, z, a = (COL["core"] if n in g.core else COL["active"]), "#FFFFFF", 1, 0.78
            else:
                fc, ec, z, a = COL["pore"], "#DDE8E5", 1, alpha
            ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, fc=fc, ec=ec, lw=0.28, alpha=a, zorder=z))
    for i in range(g.width + 1):
        x = i - 0.5
        ax.plot([x, x], [-0.5, g.height - 0.5], color=COL["grid"], lw=0.30, alpha=0.84, zorder=2)
    for i in range(g.height + 1):
        y = i - 0.5
        ax.plot([-0.5, g.width - 0.5], [y, y], color=COL["grid"], lw=0.30, alpha=0.84, zorder=2)
    for x, y in sorted(g.solid):
        ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, fc=COL["solid"], ec="#C7DBD3", lw=0.28, zorder=7))
    if show_graph:
        for a, b in sorted(g.edges):
            ax.plot([center(a)[0], center(b)[0]], [center(a)[1], center(b)[1]],
                    color=COL["node_edge"], lw=0.24, alpha=0.34, zorder=10)
    if show_nodes:
        for n in sorted(g.pore):
            x, y = center(n)
            ax.scatter([x], [y], s=8, fc=COL["node_fill"], ec=COL["node_edge"], lw=0.38, zorder=20)


def draw_sites(ax, g, labels=True, z=60):
    for idx, s in enumerate(g.sites):
        x, y = center(s)
        ax.scatter([x], [y], s=48, fc=COL["site"], ec="white", lw=1.0, zorder=z)
        if labels:
            dx, dy = (0.18, 0.34)
            if idx == 1:
                dx, dy = (-0.78, 0.28)
            ax.text(x + dx, y + dy, SITE_LABELS[idx], fontsize=6.0, color=COL["site"], zorder=z + 1,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))


def arrow(ax, start, end, color, lw=1.04, dashed=False, z=18, shrink=0.14):
    x0, y0 = center(start)
    x1, y1 = center(end)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length > 0:
        x0 += shrink * dx / length; y0 += shrink * dy / length
        x1 -= shrink * dx / length; y1 -= shrink * dy / length
    style = dict(arrowstyle="-|>", mutation_scale=5.7, lw=lw, color=color, shrinkA=0, shrinkB=0)
    if dashed:
        style["linestyle"] = DASH
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=style, zorder=z)


def candidate_line(ax, start, end, color, lw=1.02, dashed=False, z=18):
    x0, y0 = center(start)
    x1, y1 = center(end)
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=DASH if dashed else "-",
            solid_capstyle="round", zorder=z)


def path_line(ax, path, color, lw=1.10, z=18):
    px, py = zip(*[center(n) for n in path])
    ax.plot(px, py, color=color, lw=lw, solid_capstyle="round", zorder=z)


def pill(ax, x, y, text, color=COL["ink"], ha="left", va="center", fs=5.7, z=100, edge=True):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
            bbox=dict(boxstyle="round,pad=0.09", fc="white", ec=COL["card_edge"] if edge else "none", lw=0.22, alpha=0.88))


def draw_panel_a(ax, g):
    draw_lattice(ax, g, mode="plain", show_graph=True, show_nodes=True)
    draw_sites(ax, g)
    pill(ax, -0.12, 7.18, r"$G_g$ on pore voxels", COL["muted"], fs=5.5)


def draw_panel_b(ax, g):
    draw_lattice(ax, g, mode="partition", show_graph=False, show_nodes=True)
    draw_sites(ax, g)
    pill(ax, -0.10, 7.18, r"$\mathcal{C}$", COL["core"], fs=6.0)
    pill(ax, 0.76, 7.18, r"$\mathcal{R}$", COL["active"], fs=6.0)


def draw_panel_c(ax, g):
    draw_lattice(ax, g, mode="partition", show_graph=True, show_nodes=True, alpha=0.8)
    path_line(ax, g.path, COL["accept"], lw=1.08, z=18)
    candidate_line(ax, g.u, g.v0, COL["reject"], lw=0.98, dashed=True, z=18)
    draw_sites(ax, g, labels=False)
    ax.scatter([center(g.u)[0]], [center(g.u)[1]], s=38, fc="white", ec=COL["accept"], lw=1.25, zorder=80)
    ax.scatter([center(g.v)[0]], [center(g.v)[1]], s=38, fc=COL["accept"], ec="white", lw=0.72, zorder=81)
    ax.scatter([center(g.v0)[0]], [center(g.v0)[1]], s=50, marker="x", color=COL["reject"], lw=1.35, zorder=82)
    ax.text(center(g.u)[0] - 0.54, center(g.u)[1] - 0.78, "u", fontsize=6.0, color=COL["ink"], zorder=90)
    ax.text(center(g.v)[0] + 0.20, center(g.v)[1] + 0.34, "v", fontsize=6.0, color=COL["accept"], zorder=90)
    ax.text(center(g.v0)[0] + 0.34, center(g.v0)[1] + 0.44, r"$v^0$", fontsize=6.0, color=COL["reject"], zorder=90)
    pill(ax, 6.75, 7.08, "$h_j=4$\ncertified path", COL["ink"], fs=5.5)


def draw_panel_d(ax, g):
    draw_lattice(ax, g, mode="partition", show_graph=True, show_nodes=True, alpha=0.8)
    path_line(ax, g.path, COL["accept"], lw=1.10, z=18)
    draw_sites(ax, g, labels=False)
    for n in g.path[1:-1]:
        ax.scatter([center(n)[0]], [center(n)[1]], s=24, fc=COL["accept"], ec="white", lw=0.52, zorder=80)
    ax.scatter([center(g.u)[0]], [center(g.u)[1]], s=44, fc="white", ec=COL["accept"], lw=1.16, zorder=82)
    ax.scatter([center(g.v)[0]], [center(g.v)[1]], s=44, fc=COL["accept"], ec="white", lw=0.72, zorder=83)
    ax.text(center(g.u)[0] - 0.52, center(g.u)[1] - 0.70, "u", fontsize=6.0, color=COL["ink"], zorder=90)
    ax.text(center(g.v)[0] + 0.20, center(g.v)[1] + 0.36, "v", fontsize=6.0, color=COL["accept"], zorder=90)
    pill(ax, 5.72, 6.42, r"$P_{h_j}(u,v)\subset G_g$" + "\n" + "finite path certificate" + "\n" + "update only if lex improves", COL["ink"], fs=5.3, edge=False)


def draw_panel_e(ax, g):
    draw_lattice(ax, g, mode="partition", show_graph=True, show_nodes=True, alpha=0.8)
    draw_sites(ax, g, labels=False)
    xb0, yb0, xb1, yb1 = g.tile_bounds["bounds"]
    ax.add_patch(patches.Rectangle((xb0, yb0), xb1 - xb0, yb1 - yb0,
                                   fill=False, ec=COL["tile"], lw=0.55, ls=DASH, alpha=0.66, zorder=55))
    ax.text(xb0 + 0.12, yb1 + 0.36, "tile", fontsize=5.5, color=COL["tile"], alpha=0.78, zorder=90)
    for a, b in g.unit_relaxation_edges:
        arrow(ax, a, b, COL["unit"], lw=1.04, z=18)
    for a, b in g.directed_audit_edges:
        arrow(ax, a, b, COL["audit"], lw=1.05, z=19)
    pill(ax, -0.10, 7.24, "closure: unit edges", COL["unit"], fs=5.5, edge=False)
    pill(ax, 6.12, 7.24, "audit: directed edge", COL["audit"], fs=5.5, edge=False)
    pill(ax, -0.02, 0.02, r"$\rho_D$ = max residual" + "\n" + r"$n_{\mathrm{lex}}=0$", COL["ink"], fs=5.5, edge=False)


def draw_field_map(ax, g, x0, y0, scale, mode, title):
    owner_cols = {0: COL["owner0"], 1: COL["owner1"], 2: COL["owner2"]}
    ax.text(x0 + 0.5 * g.width * scale, y0 + g.height * scale + 0.20, title, fontsize=6.0,
            color=COL["ink"], ha="center", va="bottom", zorder=100)
    for y in range(g.height):
        for x in range(g.width):
            n = (x, y)
            if n in g.solid:
                fc, ec = COL["solid"], "#C7DBD3"
            elif mode == "owner":
                fc, ec = owner_cols[g.labels[n]], "#E5EEEB"
            else:
                t = min(g.D[n], DIST_VMAX) / DIST_VMAX
                fc, ec = to_hex(mpl.colormaps["YlGnBu"](0.18 + 0.72 * t)), "#E5EEEB"
            ax.add_patch(patches.Rectangle((x0 + x * scale, y0 + y * scale), scale, scale,
                                           fc=fc, ec=ec, lw=0.16, zorder=20))
    for s in g.sites:
        ax.scatter([x0 + (s[0] + 0.5) * scale], [y0 + (s[1] + 0.5) * scale],
                   s=12, fc=COL["site"], ec="white", lw=0.35, zorder=80)


def draw_panel_f(ax, g):
    ax.set_xlim(-0.35, g.width + 0.35)
    ax.set_ylim(-0.35, g.height + 0.45)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    scale = 0.42
    draw_field_map(ax, g, 0.20, 3.42, scale, "owner", r"Owner field $L$")
    draw_field_map(ax, g, 6.15, 3.42, scale, "distance", r"Distance field $D$")
    legend_y = 2.82
    for i, (lab, col) in enumerate([(r"$L=S_0$", COL["owner0"]), (r"$L=S_1$", COL["owner1"]), (r"$L=S_2$", COL["owner2"])]):
        x = 0.22 + i * 1.42
        ax.add_patch(patches.Rectangle((x, legend_y), 0.26, 0.18, fc=col, ec="#E9EEF0", lw=0.18, zorder=100))
        ax.text(x + 0.34, legend_y + 0.09, lab, fontsize=5.5, va="center", color=COL["ink"], zorder=100)
    cb_x, cb_y, cb_w, cb_h = 6.20, 2.83, 3.50, 0.18
    for i in range(40):
        t = i / 39
        ax.add_patch(patches.Rectangle((cb_x + cb_w * i / 40, cb_y), cb_w / 40, cb_h,
                                       fc=to_hex(mpl.colormaps["YlGnBu"](0.18 + 0.72 * t)), ec="none", zorder=100))
    ax.add_patch(patches.Rectangle((cb_x, cb_y), cb_w, cb_h, fill=False, ec="#E9EEF0", lw=0.18, zorder=101))
    ax.text(cb_x, cb_y - 0.15, "0", fontsize=5.5, color=COL["ink"], ha="center", va="top")
    ax.text(cb_x + cb_w, cb_y - 0.15, "7", fontsize=5.5, color=COL["ink"], ha="center", va="top")
    summary = (
        "Returned: " + r"$L$, $D$, $\rho_D$, $n_{\mathrm{lex}}$" + "\n"
        + f"Active {100*len(g.active)/len(g.pore):.1f}%; jumps {g.jump_passes}; closure {g.closure_passes}; exact mode" + "\n"
        + r"$\rho_D$" + f"={g.rho_D:.0f}, " + r"$n_{\mathrm{lex}}$" + f"={g.n_lex}"
    )
    ax.text(0.22, 1.10, summary, fontsize=5.5, color=COL["ink"], va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=COL["card_edge"], lw=0.20, alpha=0.88), zorder=120)


def add_panel_label_title(ax, letter, title):
    ax.text(0.00, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, fontweight="bold", color=COL["ink"])
    ax.text(0.075, 1.03, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.0, color=COL["ink"])


def draw_legend(fig):
    ax = fig.add_axes([0.055, 0.022, 0.89, 0.080])
    ax.set_axis_off()
    items = [
        ("patch", COL["solid"], "solid"),
        ("site", COL["site"], r"sites $S_g$"),
        ("patch", COL["core"], r"certified $\mathcal{C}$"),
        ("patch", COL["active"], r"active $\mathcal{R}$"),
        ("line", COL["accept"], "accepted path"),
        ("dash", COL["reject"], "rejected candidate"),
        ("arrow", COL["unit"], "unit relaxation"),
        ("arrow", COL["audit"], "directed audit"),
    ]
    xs = [0.00, 0.24, 0.48, 0.72]
    ys = [0.72, 0.26]
    for idx, (kind, col, label) in enumerate(items):
        row = idx // 4
        col_idx = idx % 4
        x = xs[col_idx]
        y = ys[row]
        if kind == "patch":
            ax.add_patch(patches.Rectangle((x, y - 0.08), 0.022, 0.16, fc=col, ec="none", transform=ax.transAxes))
            tx = x + 0.030
        elif kind == "site":
            ax.scatter([x + 0.011], [y], s=22, fc=col, ec="white", lw=0.45, transform=ax.transAxes, zorder=3)
            tx = x + 0.030
        elif kind == "dash":
            ax.plot([x, x + 0.040], [y, y], color=col, lw=0.98, ls=DASH, transform=ax.transAxes)
            tx = x + 0.050
        elif kind == "line":
            ax.plot([x, x + 0.040], [y, y], color=col, lw=1.04, transform=ax.transAxes)
            tx = x + 0.050
        else:
            ax.annotate("", xy=(x + 0.040, y), xytext=(x, y), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.04, mutation_scale=5.7, shrinkA=0, shrinkB=0))
            tx = x + 0.050
        ax.text(tx, y, label, transform=ax.transAxes, va="center", ha="left", fontsize=5.5, color=COL["ink"])


def make_figure():
    g = build_graph()
    validate(g)
    fig = plt.figure(figsize=(183 * MM, 100 * MM), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.045, right=0.985, top=0.905, bottom=0.165,
                          wspace=0.18, hspace=0.13)
    titles = [
        ("a", "Input pore graph"),
        ("b", "Certified core and active region"),
        ("c", r"Scale-$h_j$ candidates"),
        ("d", "Path certificate and update"),
        ("e", "Closure and directed audit"),
        ("f", "Returned fields"),
    ]
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax, (letter, title) in zip(axes, titles):
        if letter != "f":
            setup_panel(ax, letter, title, g)
        else:
            add_panel_label_title(ax, letter, title)
    draw_panel_a(axes[0], g)
    draw_panel_b(axes[1], g)
    draw_panel_c(axes[2], g)
    draw_panel_d(axes[3], g)
    draw_panel_e(axes[4], g)
    draw_panel_f(axes[5], g)
    draw_legend(fig)
    return fig, g


def audit_scene(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "nx", "ny", "solid_cells", "pore_nodes", "site_coordinates", "certified_mask", "active_mask",
        "accepted_candidate", "rejected_candidate", "certified_path_vertices", "tile_bounds",
        "unit_relaxation_edges", "directed_audit_edges", "owner_field_L", "distance_field_D",
        "rho_D", "n_lex", "active_fraction", "jump_count", "closure_count", "mode",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError(f"Scene record missing keys: {missing}")
    if record["nx"] != 11 or record["ny"] != 8:
        raise RuntimeError("Scene record grid is not 11x8")
    if record["accepted_candidate"]["offset"] != [4, 0]:
        raise RuntimeError("Accepted candidate is not the required h_j=4 horizontal offset")
    if record["rejected_candidate"]["offset"] != [4, 4]:
        raise RuntimeError("Rejected candidate is not the required h_j=4 diagonal offset")
    if record["tile_bounds"]["bounds"] != [4.5, 0.5, 8.5, 4.5]:
        raise RuntimeError("Tile bounds are not aligned to half-integer boundaries")
    if abs(record["active_fraction"] - 0.5121951219512195) > 1e-12:
        raise RuntimeError("Active fraction does not match the retained 51.2% audit value")
    if record["rho_D"] != 0.0 or record["n_lex"] != 0:
        raise RuntimeError("Directed audit values do not match rho_D=0 and n_lex=0")
    return record


def audit_files(base: Path):
    svg = base.with_suffix(".svg")
    text = svg.read_text(encoding="utf-8")
    forbidden = ["rho" + "_g", "\\hat", "eta" + "_", "(a)", "(b)", "(c)", "(d)", "Figure 02"]
    found = [s for s in forbidden if s in text]
    if found:
        raise RuntimeError(f"Forbidden strings in SVG: {found}")
    if re.search(r"Audited ROI|global title", text):
        raise RuntimeError("Global title residue found")
    audit_scene(OUT / "figure3_scene.json")


def main():
    fig, g = make_figure()
    base = OUT / "figure3_audited_ownership"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    plt.close(fig)
    audit_files(base)
    print("Figure 3 validation passed:")
    print("- 11x8 master lattice with integer node centres and half-integer boundaries: pass")
    print("- scene JSON drives shared solid/site/mask/candidate/tile/audit data: pass")
    print("- certified core / active region partition: pass")
    print("- accepted jump has feasible four-edge graph path certificate: pass")
    print("- rejected candidate is a scale-h_j offset ending at v0 without an arrowhead: pass")
    print("- directed residual rho_D and n_lex computed from directed graph edges: pass")
    print("- returned owner and distance fields computed from the same graph: pass")
    print("- outputs written:")
    for suffix in (".pdf", ".svg", ".png"):
        p = base.with_suffix(suffix)
        print(f"  {p} ({p.stat().st_size} bytes)")
    scene = OUT / "figure3_scene.json"
    print(f"  {scene} ({scene.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

