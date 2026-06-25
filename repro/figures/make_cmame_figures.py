from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper_ready_data_package_2026-06-06"
FIG = ROOT / "cmame_artifacts" / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

COL = {
    "ink": "#1f252b",
    "muted": "#6b7280",
    "grid": "#d7dde5",
    "fluid": "#f8fafc",
    "solid": "#2f3845",
    "teal": "#2a9d8f",
    "blue": "#3a6ea5",
    "orange": "#e07a3f",
    "red": "#c94c4c",
    "purple": "#7b5ea7",
    "green": "#4f9d69",
    "gold": "#c99a2e",
}

FIG2_COL = {
    "domain": "#f5f9f6",
    "frame": "#6fa985",
    "grid": "#dce9e1",
    "solid": "#009E73",
    "solid_edge": "#b9d8ca",
    "site": "#CC79A7",
    "site_shadow": "#CC79A7",
    "path": "#245B73",
    "stack_solid": "#5f6874",
    "cell_i": "#D8A760",
    "cell_j": "#8AA6C8",
    "cell_k": "#B98FBD",
    "wall": "#B9A14E",
    "interface": "#E040FB",
}

FIG2_CELL_PALETTE = ["#7DB6A3", "#D8A760", "#B98FBD", "#97B770", "#B9A14E", "#8AA6C8"]


def save(fig: plt.Figure, name: str, png: bool = False) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    if png:
        fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def label(ax, txt: str) -> None:
    ax.text(
        0.0,
        1.02,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        color=COL["ink"],
    )


def clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_voxel_grid(ax, mask: np.ndarray, labels_arr: np.ndarray | None = None) -> None:
    h, w = mask.shape
    if labels_arr is None:
        img = np.where(mask, 0, 1)
        cmap = ListedColormap([COL["fluid"], COL["solid"]])
        ax.imshow(img, cmap=cmap, origin="lower", extent=[0, w, 0, h])
    else:
        colors = [COL["solid"], "#dceef2", "#f4e1d0", "#e7def2", "#e1ead7"]
        img = np.where(mask, labels_arr + 1, 0)
        ax.imshow(img, cmap=ListedColormap(colors), origin="lower", extent=[0, w, 0, h])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, w + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, h + 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    clean(ax)


def small_mask() -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = 12, 18
    mask = np.ones((h, w), dtype=bool)
    mask[2:10, 8:10] = False
    mask[5:7, 8:10] = True
    mask[0:2, 3:6] = False
    mask[9:12, 13:16] = False
    sites = [(3, 3), (14, 3), (4, 9), (13, 8)]
    return mask, sites


def bfs_labels(mask: np.ndarray, sites: list[tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    lab = np.full((h, w), -1, dtype=int)
    dist = np.full((h, w), np.inf)
    q: deque[tuple[int, int]] = deque()
    for k, (x, y) in enumerate(sites):
        if mask[y, x]:
            lab[y, x] = k
            dist[y, x] = 0
            q.append((x, y))
    for x, y in q:
        pass
    while q:
        x, y = q.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                nd = dist[y, x] + 1
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    lab[ny, nx] = lab[y, x]
                    q.append((nx, ny))
    return lab


DIR3 = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]


def small_mask_3d() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    nz, ny, nx = 6, 8, 11
    mask = np.ones((nz, ny, nx), dtype=bool)
    mask[:, 1:7, 5] = False
    mask[2:4, 3:5, 5] = True
    mask[0:2, 0:3, 2:4] = False
    mask[4:6, 5:8, 8:10] = False
    sites = [(2, 2, 3), (8, 2, 3), (2, 6, 4), (8, 5, 1)]
    return mask, sites


def bfs_labels_3d(mask: np.ndarray, sites: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = mask.shape
    lab = np.full(mask.shape, -1, dtype=int)
    dist = np.full(mask.shape, np.inf)
    q: deque[tuple[int, int, int]] = deque()
    for k, (x, y, z) in enumerate(sites):
        if mask[z, y, x]:
            lab[z, y, x] = k
            dist[z, y, x] = 0
            q.append((x, y, z))
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in DIR3:
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz and mask[nz_, ny_, nx_]:
                nd = dist[z, y, x] + 1
                if nd < dist[nz_, ny_, nx_]:
                    dist[nz_, ny_, nx_] = nd
                    lab[nz_, ny_, nx_] = lab[z, y, x]
                    q.append((nx_, ny_, nz_))
    return lab, dist


def shortest_path_3d(mask: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = mask.shape
    q: deque[tuple[int, int, int]] = deque([start])
    prev: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    while q:
        x, y, z = q.popleft()
        if (x, y, z) == goal:
            break
        for dx, dy, dz in DIR3:
            nb = (x + dx, y + dy, z + dz)
            bx, by, bz = nb
            if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and mask[bz, by, bx] and nb not in prev:
                prev[nb] = (x, y, z)
                q.append(nb)
    if goal not in prev:
        return np.empty((0, 3))
    path = []
    cur: tuple[int, int, int] | None = goal
    while cur is not None:
        path.append((cur[0] + 0.5, cur[1] + 0.5, cur[2] + 0.5))
        cur = prev[cur]
    return np.asarray(path[::-1])


def face_vertices(x: int, y: int, z: int, direction: tuple[int, int, int]) -> list[tuple[float, float, float]]:
    dx, dy, dz = direction
    if dx == 1:
        return [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
    if dx == -1:
        return [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)]
    if dy == 1:
        return [(x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)]
    if dy == -1:
        return [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]
    if dz == 1:
        return [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
    return [(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)]


def add_faces(ax, faces: list[list[tuple[float, float, float]]], color: str, alpha: float, lw: float = 0.12, edge: str = "white") -> None:
    if not faces:
        return
    poly = Poly3DCollection(faces, facecolors=color, edgecolors=edge, linewidths=lw, alpha=alpha)
    poly.set_zsort("average")
    ax.add_collection3d(poly)


LIGHT_DIR = np.array([-0.45, -0.55, 0.78])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def face_normal(face: list[tuple[float, float, float]]) -> np.ndarray:
    pts = np.asarray(face, dtype=float)
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[1])
    norm = np.linalg.norm(normal)
    return normal / norm if norm else np.array([0.0, 0.0, 1.0])


def lit_color(base: str, normal: np.ndarray, alpha: float, ambient: float = 0.66, diffuse: float = 0.52) -> tuple[float, float, float, float]:
    rgb = np.asarray(to_rgb(base), dtype=float)
    intensity = ambient + diffuse * max(float(np.dot(normal, LIGHT_DIR)), 0.0)
    if intensity <= 1.0:
        shaded = rgb * intensity
    else:
        shaded = rgb + (1.0 - rgb) * min(intensity - 1.0, 0.32)
    return float(shaded[0]), float(shaded[1]), float(shaded[2]), alpha


def add_faces_lit(
    ax,
    faces: list[list[tuple[float, float, float]]],
    color: str,
    alpha: float,
    lw: float = 0.12,
    edge: str = "white",
    ambient: float = 0.66,
    diffuse: float = 0.52,
) -> None:
    if not faces:
        return
    facecolors = [lit_color(color, face_normal(face), alpha, ambient=ambient, diffuse=diffuse) for face in faces]
    poly = Poly3DCollection(faces, facecolors=facecolors, edgecolors=edge, linewidths=lw)
    poly.set_zsort("average")
    ax.add_collection3d(poly)


def domain_box_faces(mask: np.ndarray) -> list[list[tuple[float, float, float]]]:
    nz, ny, nx = mask.shape
    return [
        [(0, 0, 0), (0, ny, 0), (nx, ny, 0), (nx, 0, 0)],
        [(0, 0, nz), (nx, 0, nz), (nx, ny, nz), (0, ny, nz)],
        [(0, 0, 0), (nx, 0, 0), (nx, 0, nz), (0, 0, nz)],
        [(0, ny, 0), (0, ny, nz), (nx, ny, nz), (nx, ny, 0)],
        [(0, 0, 0), (0, 0, nz), (0, ny, nz), (0, ny, 0)],
        [(nx, 0, 0), (nx, ny, 0), (nx, ny, nz), (nx, 0, nz)],
    ]


def draw_open_domain_frame(ax, mask: np.ndarray) -> None:
    nz, ny, nx = mask.shape
    frame = "#b9d2dd"
    grid = "#d6e5ec"

    # Bottom and back reference planes give depth without covering the objects.
    for x in range(nx + 1):
        ax.plot([x, x], [0, ny], [0, 0], color=grid, lw=0.30, alpha=0.78)
        ax.plot([x, x], [ny, ny], [0, nz], color=grid, lw=0.30, alpha=0.62)
    for y in range(ny + 1):
        ax.plot([0, nx], [y, y], [0, 0], color=grid, lw=0.30, alpha=0.78)
    for z in range(nz + 1):
        ax.plot([0, nx], [ny, ny], [z, z], color=grid, lw=0.30, alpha=0.62)

    # Draw only the rear/bottom box edges, leaving the front open.
    edges = [
        ((0, 0, 0), (nx, 0, 0)),
        ((0, 0, 0), (0, ny, 0)),
        ((nx, 0, 0), (nx, ny, 0)),
        ((0, ny, 0), (nx, ny, 0)),
        ((0, ny, 0), (0, ny, nz)),
        ((nx, ny, 0), (nx, ny, nz)),
        ((0, ny, nz), (nx, ny, nz)),
    ]
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=frame, lw=0.72, alpha=0.95)


def draw_cad_domain_box(ax, mask: np.ndarray) -> None:
    nz, ny, nx = mask.shape
    faces = domain_box_faces(mask)
    normals = [face_normal(face) for face in faces]
    # CAD-like x-ray box: keep the transparent volume but make rear/side
    # faces darker than the illuminated upper/front faces.
    for face, normal in zip(faces, normals):
        facing = max(float(np.dot(normal, LIGHT_DIR)), 0.0)
        alpha = 0.11 + 0.10 * facing
        add_faces_lit(
            ax,
            [face],
            "#9ecbd7",
            alpha=alpha,
            lw=0.48,
            edge="#b7d4df",
            ambient=0.76,
            diffuse=0.46,
        )
    frame = "#8fb7c6"
    grid = "#c4dae4"
    for z in range(nz + 1):
        ax.plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [z] * 5, color=grid, lw=0.24, alpha=0.42)
    for x in range(nx + 1):
        ax.plot([x, x], [0, 0], [0, nz], color=grid, lw=0.20, alpha=0.28)
        ax.plot([x, x], [ny, ny], [0, nz], color=grid, lw=0.22, alpha=0.50)
    for y in range(ny + 1):
        ax.plot([0, 0], [y, y], [0, nz], color=grid, lw=0.20, alpha=0.28)
        ax.plot([nx, nx], [y, y], [0, nz], color=grid, lw=0.22, alpha=0.50)
    corners = [
        (0, 0, 0), (nx, 0, 0), (nx, ny, 0), (0, ny, 0),
        (0, 0, nz), (nx, 0, nz), (nx, ny, nz), (0, ny, nz),
    ]
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edge_pairs:
        a, b = corners[i], corners[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=frame, lw=0.72, alpha=0.92)


def draw_mask_surface_3d(
    ax,
    mask: np.ndarray,
    labels_arr: np.ndarray | None,
    alpha: float,
    boundary_between_labels: bool,
    label_subset: set[int] | None = None,
    lit: bool = False,
) -> None:
    palette = FIG2_CELL_PALETTE
    nz, ny, nx = mask.shape
    faces_by_label: dict[int, list[list[tuple[float, float, float]]]] = {}
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    continue
                lab = int(labels_arr[z, y, x]) if labels_arr is not None else 0
                if label_subset is not None and lab not in label_subset:
                    continue
                for d in DIR3:
                    dx, dy, dz = d
                    bx, by, bz = x + dx, y + dy, z + dz
                    outside = not (0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz)
                    solid = False if outside else not mask[bz, by, bx]
                    diff_lab = (
                        False
                        if outside or solid or labels_arr is None
                        else int(labels_arr[bz, by, bx]) != lab
                    )
                    if outside or solid or (boundary_between_labels and diff_lab):
                        faces_by_label.setdefault(lab, []).append(face_vertices(x, y, z, d))
    for lab, faces in faces_by_label.items():
        add = add_faces_lit if lit else add_faces
        add(ax, faces, palette[lab % len(palette)], alpha=alpha, lw=0.10, edge="#f8fafc")


def draw_solid_surface_3d(
    ax,
    mask: np.ndarray,
    alpha: float = 0.34,
    lit: bool = False,
    color: str = COL["solid"],
    ambient: float = 0.66,
    diffuse: float = 0.52,
) -> None:
    solid = ~mask
    nz, ny, nx = mask.shape
    faces: list[list[tuple[float, float, float]]] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not solid[z, y, x]:
                    continue
                for d in DIR3:
                    dx, dy, dz = d
                    bx, by, bz = x + dx, y + dy, z + dz
                    outside = not (0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz)
                    if outside or not solid[bz, by, bx]:
                        faces.append(face_vertices(x, y, z, d))
    add = add_faces_lit if lit else add_faces
    if lit:
        add(ax, faces, color, alpha=alpha, lw=0.14, edge="#eef3f7", ambient=ambient, diffuse=diffuse)
    else:
        add(ax, faces, color, alpha=alpha, lw=0.10, edge="#ffffff")


def intercell_facelets(mask: np.ndarray, labels_arr: np.ndarray) -> tuple[list[list[tuple[float, float, float]]], dict[tuple[int, int], list[np.ndarray]]]:
    faces: list[list[tuple[float, float, float]]] = []
    by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    nz, ny, nx = mask.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    continue
                lab = int(labels_arr[z, y, x])
                for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    dx, dy, dz = d
                    bx, by, bz = x + dx, y + dy, z + dz
                    if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and mask[bz, by, bx]:
                        nb_lab = int(labels_arr[bz, by, bx])
                        if nb_lab != lab:
                            face = face_vertices(x, y, z, d)
                            faces.append(face)
                            pair = tuple(sorted((lab, nb_lab)))
                            by_pair.setdefault(pair, []).append(np.mean(np.asarray(face), axis=0))
    return faces, by_pair


def wall_facelets(mask: np.ndarray, labels_arr: np.ndarray, labels_keep: set[int]) -> list[list[tuple[float, float, float]]]:
    nz, ny, nx = mask.shape
    faces: list[list[tuple[float, float, float]]] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x] or int(labels_arr[z, y, x]) not in labels_keep:
                    continue
                for d in DIR3:
                    dx, dy, dz = d
                    bx, by, bz = x + dx, y + dy, z + dz
                    if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and not mask[bz, by, bx]:
                        faces.append(face_vertices(x, y, z, d))
    return faces


def setup_3d_panel(ax, mask: np.ndarray, panel: str, title: str) -> None:
    nz, ny, nx = mask.shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=24, azim=-54)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.text2D(0.00, 0.98, panel, transform=ax.transAxes, fontweight="bold", color=COL["ink"])
    ax.set_title(title, pad=2, fontsize=8.2)


def setup_render_panel(ax, panel: str, title: str) -> None:
    ax.set_axis_off()
    ax.text(0.00, 0.98, panel, transform=ax.transAxes, fontweight="bold", color=COL["ink"], ha="left", va="top")
    ax.set_title(title, pad=2, fontsize=8.2)


def draw_site_mask_key(ax) -> None:
    ax.set_axis_off()
    y0 = 0.52
    entries = [
        (0.03, "traversable $V$", FIG2_COL["domain"], FIG2_COL["frame"], "box"),
        (0.39, "solid", FIG2_COL["solid"], FIG2_COL["solid"], "box"),
        (0.61, "fixed sites $S_g$", FIG2_COL["site"], FIG2_COL["site"], "dot"),
    ]
    for x0, text, face, edge, kind in entries:
        if kind == "dot":
            ax.scatter([x0], [y0], s=22, transform=ax.transAxes, c=face, edgecolors="white", linewidths=0.5, zorder=20)
        else:
            ax.add_patch(
                patches.Rectangle(
                    (x0 - 0.013, y0 - 0.125),
                    0.044,
                    0.235,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.8,
                    alpha=0.90,
                    zorder=20,
                )
            )
        ax.text(x0 + 0.043, y0, text, transform=ax.transAxes, ha="left", va="center", fontsize=6.0, color=COL["ink"], zorder=21)


def draw_site_to_cell_key(ax) -> None:
    ax.set_axis_off()
    entries = [
        (0.02, 0.66, "fluid $V$", FIG2_COL["domain"], FIG2_COL["frame"], "box"),
        (0.23, 0.66, "solid", FIG2_COL["solid"], FIG2_COL["solid"], "box"),
        (0.36, 0.66, "sites $S_g$", FIG2_COL["site"], FIG2_COL["site"], "dot"),
        (0.02, 0.24, "FV cells $\\Omega_k$", FIG2_COL["cell_j"], FIG2_COL["cell_j"], "box"),
        (0.27, 0.24, "facelets $\\Gamma_{ij}$", FIG2_COL["interface"], FIG2_COL["interface"], "box"),
        (0.52, 0.24, "geodesic path", FIG2_COL["path"], FIG2_COL["path"], "line"),
    ]
    for x0, y0, text, face, edge, kind in entries:
        if kind == "dot":
            ax.scatter([x0 + 0.010], [y0], s=22, transform=ax.transAxes, c=face, edgecolors="white", linewidths=0.5, zorder=20)
            text_x = x0 + 0.045
        elif kind == "line":
            ax.plot([x0 - 0.002, x0 + 0.036], [y0, y0], transform=ax.transAxes, color=edge, lw=1.4, solid_capstyle="round", zorder=20)
            ax.scatter([x0 + 0.017], [y0], s=14, transform=ax.transAxes, c=FIG2_COL["site"], edgecolors="white", linewidths=0.35, zorder=21)
            text_x = x0 + 0.052
        else:
            ax.add_patch(
                patches.Rectangle(
                    (x0 - 0.003, y0 - 0.105),
                    0.035,
                    0.205,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.8,
                    alpha=0.90,
                    zorder=20,
                )
            )
            text_x = x0 + 0.045
        ax.text(text_x, y0, text, transform=ax.transAxes, ha="left", va="center", fontsize=6.0, color=COL["ink"], zorder=21)


def crop_near_white_image(src: Path, dst: Path, pad: int = 90) -> None:
    from PIL import Image

    img = Image.open(src).convert("RGBA")
    arr = np.asarray(img.convert("RGB"))
    keep = np.any(arr < 250, axis=2)
    if not np.any(keep):
        img.save(dst)
        return
    ys, xs = np.where(keep)
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad, img.width)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad, img.height)
    img.crop((left, top, right, bottom)).save(dst)


def adjacent_to_solid(mask: np.ndarray) -> np.ndarray:
    nz, ny, nx = mask.shape
    out = np.zeros(mask.shape, dtype=bool)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    continue
                for dx, dy, dz in DIR3:
                    bx, by, bz = x + dx, y + dy, z + dz
                    if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and not mask[bz, by, bx]:
                        out[z, y, x] = True
                        break
    return out


def render_site_mask_cad(
    mask: np.ndarray,
    sites: list[tuple[int, int, int]],
    labels_arr: np.ndarray | None = None,
    section_z: int | None = None,
    path: np.ndarray | None = None,
    ownership_labels_keep: set[int] | None = None,
    cell_labels_keep: set[int] | None = None,
    cell_interface_pair: tuple[int, int] | None = None,
    wall_label_keep: int | None = None,
    graph_site_pair: tuple[int, int] | None = None,
    cell_multi_opacity: float | None = None,
    solid_opacity_override: float | None = None,
    facelet_pull: float = 0.36,
    wall_facelet_pull: float = 0.32,
    graph_pull: float = 1.20,
    cell_voxelized: bool = False,
    site_indices: set[int] | None = None,
    out_stub: str = "Figure_03a_site_mask_cad",
) -> Path | None:
    try:
        import pyvista as pv
    except ImportError:
        return None

    FIG.mkdir(parents=True, exist_ok=True)
    raw = FIG / f"{out_stub}_raw.png"
    out = FIG / f"{out_stub}.png"
    nz, ny, nx = mask.shape

    pl = pv.Plotter(off_screen=True, window_size=(1800, 1300))
    pl.set_background("white")

    edge_color = "#347a58"
    grid_color = FIG2_COL["grid"]
    solid_color = FIG2_COL["solid"]
    site_color = FIG2_COL["site"]

    camera_position = np.asarray((15.5, -14.5, 9.5), dtype=float)
    camera_target = np.asarray((nx / 2, ny / 2, nz / 2), dtype=float)
    view_dir = camera_target - camera_position
    view_dir = view_dir / np.linalg.norm(view_dir)

    def add_shell_face(
        vertices,
        opacity,
        color="#c7ead2",
        edge="#6fa985",
        edge_width=1.05,
        show_edges=True,
    ) -> None:
        points = np.asarray(vertices, dtype=float)
        plane = pv.PolyData(points, np.asarray([4, 0, 1, 2, 3]))
        pl.add_mesh(
            plane,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge,
            line_width=edge_width,
            lighting=False,
        )

    eps = 0.018
    shell_fill = "#d8f0df"
    shell_edge = "#347a58"
    add_shell_face([(0, ny + eps, 0), (nx, ny + eps, 0), (nx, ny + eps, nz), (0, ny + eps, nz)], 0.20, shell_fill, shell_edge, 1.22)
    add_shell_face([(-eps, 0, 0), (-eps, ny, 0), (-eps, ny, nz), (-eps, 0, nz)], 0.15, shell_fill, shell_edge, 1.12)
    add_shell_face([(0, 0, -eps), (nx, 0, -eps), (nx, ny, -eps), (0, ny, -eps)], 0.19, "#e7f5ea", shell_edge, 1.18)
    add_shell_face([(0, 0, nz + eps), (0, ny, nz + eps), (nx, ny, nz + eps), (nx, 0, nz + eps)], 0.18, shell_fill, shell_edge, 1.18)

    near_shell_faces = [
        ([(nx + eps, 0, 0), (nx + eps, 0, nz), (nx + eps, ny, nz), (nx + eps, ny, 0)], 0.31),
        ([(0, -eps, 0), (0, -eps, nz), (nx, -eps, nz), (nx, -eps, 0)], 0.30),
    ]
    top_lid_vertices = [(0, 0, nz + 2.6 * eps), (0, ny, nz + 2.6 * eps), (nx, ny, nz + 2.6 * eps), (nx, 0, nz + 2.6 * eps)]

    def add_line(p0, p1, color=grid_color, width=1.0, opacity=0.16) -> None:
        pl.add_mesh(pv.Line(p0, p1), color=color, line_width=width, opacity=opacity, lighting=False)

    def shrunken_cube_faces(x: int, y: int, z: int, gap: float = 0.055) -> list[list[tuple[float, float, float]]]:
        x0, x1 = x + gap, x + 1 - gap
        y0, y1 = y + gap, y + 1 - gap
        z0, z1 = z + gap, z + 1 - gap
        return [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],
            [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],
            [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
            [(x1, y1, z0), (x1, y1, z1), (x0, y1, z1), (x0, y1, z0)],
            [(x0, y1, z0), (x0, y1, z1), (x0, y0, z1), (x0, y0, z0)],
        ]

    y_cover = -eps * 1.7
    x_cover = nx + eps * 1.7

    for x in range(nx + 1):
        add_line((x, 0, 0), (x, ny, 0), opacity=0.11)
    for y in range(ny + 1):
        add_line((0, y, 0), (nx, y, 0), opacity=0.11)

    corners = [
        (0, 0, 0),
        (nx, 0, 0),
        (nx, ny, 0),
        (0, ny, 0),
        (0, 0, nz),
        (nx, 0, nz),
        (nx, ny, nz),
        (0, ny, nz),
    ]
    edge_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    edge_depths = []
    for i, j in edge_pairs:
        midpoint = (np.asarray(corners[i], dtype=float) + np.asarray(corners[j], dtype=float)) / 2
        edge_depths.append(float(np.dot(midpoint - camera_position, view_dir)))
    min_edge_depth = min(edge_depths)
    max_edge_depth = max(edge_depths)
    edge_span = max(max_edge_depth - min_edge_depth, 1e-9)
    for (i, j), depth in sorted(zip(edge_pairs, edge_depths), key=lambda item: item[1], reverse=True):
        if (i, j) == (4, 5):
            continue
        near_weight = 1 - (depth - min_edge_depth) / edge_span
        width = 2.5 + 3.3 * near_weight
        opacity = 0.30 + 0.65 * near_weight
        add_line(corners[i], corners[j], color=edge_color, width=width, opacity=opacity)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    sh = pv.Plane(
                        center=(x + 0.5, y + 0.5, 0.018),
                        direction=(0, 0, 1),
                        i_size=0.94,
                        j_size=0.94,
                        i_resolution=1,
                        j_resolution=1,
                    )
                    pl.add_mesh(sh, color="#243240", opacity=0.030, lighting=False)

    if labels_arr is not None and section_z is not None:
        zc = section_z + 0.515
        for y in range(ny):
            for x in range(nx):
                if mask[section_z, y, x]:
                    lab = int(labels_arr[section_z, y, x])
                    tile = pv.Plane(
                        center=(x + 0.5, y + 0.5, zc),
                        direction=(0, 0, 1),
                        i_size=0.94,
                        j_size=0.94,
                        i_resolution=1,
                        j_resolution=1,
                    )
                    pl.add_mesh(
                        tile,
                        color=FIG2_CELL_PALETTE[lab % len(FIG2_CELL_PALETTE)],
                        opacity=0.54,
                        show_edges=True,
                        edge_color="#ffffff",
                        line_width=0.52,
                        lighting=False,
                    )
        section_edge = "#245B73"
        for p0, p1 in [
            ((0, 0, zc + 0.012), (nx, 0, zc + 0.012)),
            ((nx, 0, zc + 0.012), (nx, ny, zc + 0.012)),
            ((nx, ny, zc + 0.012), (0, ny, zc + 0.012)),
            ((0, ny, zc + 0.012), (0, 0, zc + 0.012)),
        ]:
            pl.add_mesh(pv.Line(p0, p1), color=section_edge, line_width=2.2, opacity=0.32, lighting=False)

    solid_points: list[tuple[float, float, float]] = []
    solid_faces: list[list[int]] = []
    solid_face_vertices = {
        (1, 0, 0): lambda x, y, z: [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)],
        (-1, 0, 0): lambda x, y, z: [(x, y + 1, z), (x, y, z), (x, y, z + 1), (x, y + 1, z + 1)],
        (0, 1, 0): lambda x, y, z: [(x + 1, y + 1, z), (x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1)],
        (0, -1, 0): lambda x, y, z: [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)],
        (0, 0, 1): lambda x, y, z: [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)],
        (0, 0, -1): lambda x, y, z: [(x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z), (x, y, z)],
    }
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    for dx, dy, dz in DIR3:
                        bx, by, bz = x + dx, y + dy, z + dz
                        if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and not mask[bz, by, bx]:
                            continue
                        base = len(solid_points)
                        solid_points.extend(solid_face_vertices[(dx, dy, dz)](x, y, z))
                        solid_faces.append([4, base, base + 1, base + 2, base + 3])
    if solid_points:
        solid_mesh = pv.PolyData(np.asarray(solid_points, dtype=float), np.asarray(solid_faces, dtype=np.int64).ravel())
        solid_opacity = 1.0
        if section_z is not None:
            solid_opacity = 0.84
        if ownership_labels_keep is not None:
            solid_opacity = 0.50
        if cell_labels_keep is not None:
            solid_opacity = 1.0
        if solid_opacity_override is not None:
            solid_opacity = solid_opacity_override
        pl.add_mesh(
            solid_mesh,
            color=solid_color,
            show_edges=False,
            opacity=solid_opacity,
            smooth_shading=False,
            ambient=0.24,
            diffuse=0.78,
            specular=0.055,
            specular_power=18,
        )

    if labels_arr is not None and ownership_labels_keep is not None:
        for lab_keep in sorted(ownership_labels_keep):
            pts: list[tuple[float, float, float]] = []
            polys: list[list[int]] = []
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        if not mask[z, y, x] or int(labels_arr[z, y, x]) != lab_keep:
                            continue
                        for d in DIR3:
                            dx, dy, dz = d
                            bx, by, bz = x + dx, y + dy, z + dz
                            if (
                                0 <= bx < nx
                                and 0 <= by < ny
                                and 0 <= bz < nz
                                and mask[bz, by, bx]
                                and int(labels_arr[bz, by, bx]) == lab_keep
                            ):
                                continue
                            base = len(pts)
                            pts.extend(face_vertices(x, y, z, d))
                            polys.append([4, base, base + 1, base + 2, base + 3])
            if pts:
                owner_mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(polys, dtype=np.int64).ravel())
                pl.add_mesh(
                    owner_mesh,
                    color=FIG2_CELL_PALETTE[lab_keep % len(FIG2_CELL_PALETTE)],
                    show_edges=True,
                    edge_color="#ffffff",
                    line_width=0.23,
                    opacity=0.28,
                    smooth_shading=False,
                    ambient=0.32,
                    diffuse=0.66,
                    specular=0.02,
                    specular_power=10,
                )

    if labels_arr is not None and cell_labels_keep is not None:
        cell_colors = [FIG2_COL["cell_i"], "#8AA6C8", FIG2_COL["cell_k"], FIG2_COL["wall"]]
        multi_cell_mode = len(cell_labels_keep) > 1

        def add_oriented_face_mesh(
            faces: list[list[tuple[float, float, float]]],
            color: str,
            opacity: float,
            show_edges: bool,
            edge_width: float,
            ambient: float,
            diffuse: float,
        ) -> None:
            if not faces:
                return
            pts: list[tuple[float, float, float]] = []
            polys: list[list[int]] = []
            for face in faces:
                base = len(pts)
                pts.extend(face)
                polys.append([4, base, base + 1, base + 2, base + 3])
            mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(polys, dtype=np.int64).ravel())
            pl.add_mesh(
                mesh,
                color=color,
                show_edges=show_edges,
                edge_color="#f8fbf9",
                line_width=edge_width,
                opacity=opacity,
                smooth_shading=False,
                ambient=ambient,
                diffuse=diffuse,
                specular=0.035,
                specular_power=14,
            )

        for color_index, lab_keep in enumerate(sorted(cell_labels_keep)):
            front_faces: list[list[tuple[float, float, float]]] = []
            side_faces: list[list[tuple[float, float, float]]] = []
            back_faces: list[list[tuple[float, float, float]]] = []
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        if not mask[z, y, x] or int(labels_arr[z, y, x]) != lab_keep:
                            continue
                        if cell_voxelized:
                            faces_to_add = shrunken_cube_faces(x, y, z)
                        else:
                            faces_to_add = []
                            for d in DIR3:
                                dx, dy, dz = d
                                bx, by, bz = x + dx, y + dy, z + dz
                                if (
                                    0 <= bx < nx
                                    and 0 <= by < ny
                                    and 0 <= bz < nz
                                    and mask[bz, by, bx]
                                    and int(labels_arr[bz, by, bx]) == lab_keep
                                ):
                                    continue
                                faces_to_add.append(face_vertices(x, y, z, d))
                        for face in faces_to_add:
                            center = np.mean(np.asarray(face, dtype=float), axis=0)
                            toward_camera = camera_position - center
                            toward_camera /= max(float(np.linalg.norm(toward_camera)), 1e-9)
                            facing = float(np.dot(face_normal(face), toward_camera))
                            if facing > 0.18:
                                front_faces.append(face)
                            elif facing > -0.25:
                                side_faces.append(face)
                            else:
                                back_faces.append(face)

            cell_color = cell_colors[color_index % len(cell_colors)]
            if multi_cell_mode:
                alpha = 1.00 if cell_multi_opacity is None else cell_multi_opacity
                add_oriented_face_mesh(back_faces, cell_color, alpha, False, 0.0, 0.26, 0.74)
                add_oriented_face_mesh(side_faces, cell_color, alpha, True, 0.24, 0.26, 0.74)
                add_oriented_face_mesh(front_faces, cell_color, alpha, True, 0.42, 0.24, 0.76)
            else:
                add_oriented_face_mesh(back_faces, cell_color, 0.18, False, 0.0, 0.34, 0.54)
                add_oriented_face_mesh(side_faces, cell_color, 0.74, True, 0.36, 0.32, 0.66)
                add_oriented_face_mesh(front_faces, cell_color, 1.00, True, 0.72, 0.30, 0.74)

        if wall_label_keep is not None:
            wall_faces: list[list[tuple[float, float, float]]] = []
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        if not mask[z, y, x] or int(labels_arr[z, y, x]) != wall_label_keep:
                            continue
                        for d in DIR3:
                            dx, dy, dz = d
                            bx, by, bz = x + dx, y + dy, z + dz
                            if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and not mask[bz, by, bx]:
                                face = face_vertices(x, y, z, d)
                                center = np.mean(np.asarray(face, dtype=float), axis=0)
                                toward_camera = camera_position - center
                                toward_camera /= max(float(np.linalg.norm(toward_camera)), 1e-9)
                                shifted = [
                                    tuple(np.asarray(vertex, dtype=float) + wall_facelet_pull * toward_camera)
                                    for vertex in face
                                ]
                                wall_faces.append(shifted)
            add_oriented_face_mesh(wall_faces, "#F0E6A8", 0.66, True, 0.78, 0.56, 0.44)

        if cell_interface_pair is not None:
            interface_pair = set(cell_interface_pair)
            interface_faces: list[list[tuple[float, float, float]]] = []
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        if not mask[z, y, x] or int(labels_arr[z, y, x]) not in interface_pair:
                            continue
                        lab = int(labels_arr[z, y, x])
                        for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                            dx, dy, dz = d
                            bx, by, bz = x + dx, y + dy, z + dz
                            if (
                                0 <= bx < nx
                                and 0 <= by < ny
                                and 0 <= bz < nz
                                and mask[bz, by, bx]
                                and int(labels_arr[bz, by, bx]) in interface_pair
                                and int(labels_arr[bz, by, bx]) != lab
                            ):
                                face = face_vertices(x, y, z, d)
                                center = np.mean(np.asarray(face, dtype=float), axis=0)
                                toward_camera = camera_position - center
                                toward_camera /= max(float(np.linalg.norm(toward_camera)), 1e-9)
                                shifted = [
                                    tuple(np.asarray(vertex, dtype=float) + facelet_pull * toward_camera)
                                    for vertex in face
                                ]
                                interface_faces.append(shifted)
            add_oriented_face_mesh(interface_faces, "#D7D0F0", 0.66, True, 0.78, 0.56, 0.44)

    if path is not None and len(path) > 1:
        path_points = np.asarray(path, dtype=float)
        for a, b in zip(path_points[:-1], path_points[1:]):
            segment = pv.Line(tuple(a), tuple(b))
            try:
                halo = segment.tube(radius=0.090, n_sides=18)
                pl.add_mesh(
                    halo,
                    color="#ffffff",
                    opacity=0.68,
                    smooth_shading=True,
                    ambient=0.32,
                    diffuse=0.68,
                    specular=0.02,
                    specular_power=8,
                )
                tube = segment.tube(radius=0.052, n_sides=18)
                pl.add_mesh(
                    tube,
                    color=FIG2_COL["path"],
                    opacity=1.0,
                    smooth_shading=True,
                    ambient=0.16,
                    diffuse=0.84,
                    specular=0.10,
                    specular_power=18,
                )
            except Exception:
                pl.add_mesh(segment, color=FIG2_COL["path"], line_width=7.0, opacity=1.0, lighting=False)

    visible_sites = [(idx, site) for idx, site in enumerate(sites) if site_indices is None or idx in site_indices]

    for _, (x, y, z) in visible_sites:
        cx, cy, cz = x + 0.5, y + 0.5, z + 0.5
        pl.add_mesh(
            pv.Line((cx, cy, 0.05), (cx, cy, max(0.05, cz - 0.30))),
            color=site_color,
            line_width=3.1,
            opacity=0.22,
            lighting=False,
        )
        shadow = pv.Disc(center=(cx, cy, 0.03), inner=0.0, outer=0.27, normal=(0, 0, 1), r_res=1, c_res=64)
        pl.add_mesh(shadow, color=FIG2_COL["site_shadow"], opacity=0.13, lighting=False)

    if graph_site_pair is not None:
        i, j = graph_site_pair
        if 0 <= i < len(sites) and 0 <= j < len(sites):
            p0 = np.asarray(sites[i], dtype=float) + 0.5
            p1 = np.asarray(sites[j], dtype=float) + 0.5
            toward_camera = -view_dir
            lift = np.asarray((0.0, 0.0, 0.05))
            p0 = p0 + graph_pull * toward_camera + lift
            p1 = p1 + graph_pull * toward_camera + lift
            segment = pv.Line(tuple(p0), tuple(p1))
            try:
                halo = segment.tube(radius=0.086, n_sides=24)
                pl.add_mesh(
                    halo,
                    color="#ffffff",
                    opacity=0.82,
                    smooth_shading=True,
                    ambient=0.30,
                    diffuse=0.70,
                    specular=0.05,
                    specular_power=12,
                )
                tube = segment.tube(radius=0.054, n_sides=24)
                pl.add_mesh(
                    tube,
                    color=COL["ink"],
                    opacity=1.0,
                    smooth_shading=True,
                    ambient=0.18,
                    diffuse=0.82,
                    specular=0.08,
                    specular_power=18,
                )
            except Exception:
                pl.add_mesh(segment, color=COL["ink"], line_width=7.0, opacity=1.0, lighting=False)

    for _, (x, y, z) in visible_sites:
        sphere = pv.Sphere(radius=0.30, center=(x + 0.5, y + 0.5, z + 0.5), theta_resolution=48, phi_resolution=24)
        pl.add_mesh(
            sphere,
            color=site_color,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.78,
            specular=0.34,
            specular_power=24,
        )

    if cell_labels_keep is None:
        add_shell_face(top_lid_vertices, 0.28, "#d8f0df", shell_edge, 0.75, False)
        if section_z is None:
            add_line((0, -eps, nz), (nx, -eps, nz), color=edge_color, width=2.8, opacity=0.62)

        for vertices, opacity in near_shell_faces:
            add_shell_face(vertices, opacity, "#d8f0df", "#8bbd9d", 0.55, False)

    pl.remove_all_lights()
    up_hint = np.asarray((0, 0, 1), dtype=float)
    screen_right = np.cross(view_dir, up_hint)
    screen_right = screen_right / np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, view_dir)
    screen_up = screen_up / np.linalg.norm(screen_up)
    key_light_pos = camera_target - 7.5 * view_dir - 10.5 * screen_right + 9.5 * screen_up
    fill_light_pos = camera_target - 4.5 * view_dir + 9.0 * screen_right + 2.8 * screen_up
    rim_light_pos = camera_target + 6.0 * view_dir - 2.0 * screen_right + 6.0 * screen_up
    pl.add_light(pv.Light(position=tuple(key_light_pos), focal_point=tuple(camera_target), color="white", intensity=0.92))
    pl.add_light(pv.Light(position=tuple(fill_light_pos), focal_point=tuple(camera_target), color="#eef8f3", intensity=0.16))
    pl.add_light(pv.Light(position=tuple(rim_light_pos), focal_point=tuple(camera_target), color="#f7fbf8", intensity=0.08))
    try:
        pl.enable_ssao(radius=1.15, bias=0.018, kernel_size=256)
    except Exception:
        pass
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    try:
        pl.enable_depth_peeling()
    except Exception:
        pass

    pl.camera_position = [tuple(camera_position), tuple(camera_target), (0, 0, 1)]
    pl.camera.parallel_projection = False
    pl.camera.view_angle = 37.0
    pl.show(screenshot=str(raw), auto_close=True)
    crop_near_white_image(raw, out, pad=90)
    return out


def plot_sites_3d(ax, sites: list[tuple[int, int, int]], color: str = COL["orange"], size: float = 42, depthshade: bool = False) -> None:
    pts = np.asarray([(x + 0.5, y + 0.5, z + 0.5) for x, y, z in sites])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, edgecolors="white", linewidths=0.8, depthshade=depthshade)


def plot_site_spheres(ax, sites: list[tuple[int, int, int]], color: str = COL["orange"], radius: float = 0.28) -> None:
    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 9)
    uu, vv = np.meshgrid(u, v)
    sx = np.cos(uu) * np.sin(vv)
    sy = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)
    for x, y, z in sites:
        ax.plot_surface(
            x + 0.5 + radius * sx,
            y + 0.5 + radius * sy,
            z + 0.5 + radius * sz,
            color=color,
            edgecolor="white",
            linewidth=0.08,
            shade=True,
            antialiased=True,
            alpha=1.0,
        )


def draw_site_depth_guides(
    ax,
    sites: list[tuple[int, int, int]],
    color: str = COL["orange"],
    alpha: float = 0.42,
) -> None:
    for x, y, z in sites:
        px, py, pz = x + 0.5, y + 0.5, z + 0.5
        ax.plot([px, px], [py, py], [0.02, pz - 0.05], color=color, lw=0.62, alpha=alpha)
        ax.scatter([px], [py], [0.02], s=14, c=color, alpha=alpha * 0.52, edgecolors="none", depthshade=False)


Z_GAP = 1.18


def z_stack(z: int | float) -> float:
    return float(z) * Z_GAP


def layer_tile(x: int, y: int, z: int) -> list[tuple[float, float, float]]:
    zz = z_stack(z)
    return [(x, y, zz), (x + 1, y, zz), (x + 1, y + 1, zz), (x, y + 1, zz)]


def draw_layer_frame(ax, mask: np.ndarray) -> None:
    nz, ny, nx = mask.shape
    for z in range(nz):
        zz = z_stack(z)
        ax.plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [zz] * 5, color="#c6ced8", lw=0.55, alpha=0.75)
    for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]:
        ax.plot([x, x], [y, y], [0, z_stack(nz - 1)], color="#d8dee7", lw=0.35, alpha=0.55)


def draw_slice_stack(
    ax,
    mask: np.ndarray,
    labels_arr: np.ndarray | None = None,
    fluid_alpha: float = 0.52,
    solid_alpha: float = 0.88,
    show_fluid: bool = True,
    solid_color: str = FIG2_COL["stack_solid"],
    solid_edge: str = "#f8fafc",
) -> None:
    palette = FIG2_CELL_PALETTE
    nz, ny, nx = mask.shape
    fluid_faces: dict[int, list[list[tuple[float, float, float]]]] = {}
    solid_faces: list[list[tuple[float, float, float]]] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if mask[z, y, x]:
                    if show_fluid:
                        lab = int(labels_arr[z, y, x]) if labels_arr is not None else 0
                        fluid_faces.setdefault(lab, []).append(layer_tile(x, y, z))
                else:
                    solid_faces.append(layer_tile(x, y, z))
    draw_layer_frame(ax, mask)
    if show_fluid:
        for lab, faces in fluid_faces.items():
            color = "#edf6f8" if labels_arr is None else palette[lab % len(palette)]
            add_faces(ax, faces, color, alpha=fluid_alpha, lw=0.18, edge="white")
    add_faces(ax, solid_faces, solid_color, alpha=solid_alpha, lw=0.15, edge=solid_edge)


def draw_label_tiles_stack(
    ax,
    mask: np.ndarray,
    labels_arr: np.ndarray,
    labels_keep: set[int],
    alpha: float = 0.74,
) -> None:
    palette = FIG2_CELL_PALETTE
    faces_by_lab: dict[int, list[list[tuple[float, float, float]]]] = {}
    nz, ny, nx = mask.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if mask[z, y, x] and int(labels_arr[z, y, x]) in labels_keep:
                    lab = int(labels_arr[z, y, x])
                    faces_by_lab.setdefault(lab, []).append(layer_tile(x, y, z))
    for lab, faces in faces_by_lab.items():
        add_faces(ax, faces, palette[lab % len(palette)], alpha=alpha, lw=0.18, edge="white")


def plot_sites_stack(ax, sites: list[tuple[int, int, int]], color: str = COL["orange"], size: float = 38) -> None:
    pts = np.asarray([(x + 0.5, y + 0.5, z_stack(z) + 0.08) for x, y, z in sites])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, edgecolors="white", linewidths=0.7, depthshade=False)


def setup_stack_panel(ax, mask: np.ndarray, panel: str, title: str) -> None:
    nz, ny, nx = mask.shape
    ax.set_xlim(-0.2, nx + 0.2)
    ax.set_ylim(-0.2, ny + 0.2)
    ax.set_zlim(-0.25, z_stack(nz - 1) + 0.55)
    ax.set_box_aspect((nx, ny, z_stack(nz - 1) + 0.8))
    ax.view_init(elev=28, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.text2D(0.00, 0.98, panel, transform=ax.transAxes, fontweight="bold", color=COL["ink"])
    ax.set_title(title, pad=1, fontsize=8.0)


def section_tile(x: int, y: int, z0: float = 0.0) -> list[tuple[float, float, float]]:
    return [(x, y, z0), (x + 1, y, z0), (x + 1, y + 1, z0), (x, y + 1, z0)]


def section_prism(x: int, y: int, z0: float = 0.0, h: float = 0.34) -> list[list[tuple[float, float, float]]]:
    return [
        [(x, y, z0 + h), (x + 1, y, z0 + h), (x + 1, y + 1, z0 + h), (x, y + 1, z0 + h)],
        [(x, y, z0), (x, y, z0 + h), (x + 1, y, z0 + h), (x + 1, y, z0)],
        [(x + 1, y, z0), (x + 1, y, z0 + h), (x + 1, y + 1, z0 + h), (x + 1, y + 1, z0)],
        [(x + 1, y + 1, z0), (x + 1, y + 1, z0 + h), (x, y + 1, z0 + h), (x, y + 1, z0)],
        [(x, y + 1, z0), (x, y + 1, z0 + h), (x, y, z0 + h), (x, y, z0)],
    ]


def setup_ownership_section_panel(ax, mask: np.ndarray, panel: str, title: str) -> None:
    _, ny, nx = mask.shape
    ax.set_xlim(-0.10, nx + 0.12)
    ax.set_ylim(-0.08, ny + 0.12)
    ax.set_zlim(-0.10, 0.82)
    ax.set_box_aspect((nx, ny, 1.35), zoom=1.24)
    ax.view_init(elev=37, azim=-57)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.text2D(0.00, 0.98, panel, transform=ax.transAxes, fontweight="bold", color=COL["ink"])
    ax.set_title(title, pad=0, fontsize=7.8)


def draw_ownership_section(
    ax,
    mask: np.ndarray,
    labels_arr: np.ndarray,
    sites: list[tuple[int, int, int]],
    z: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    _, ny, nx = mask.shape
    palette = FIG2_CELL_PALETTE
    faces_by_lab: dict[int, list[list[tuple[float, float, float]]]] = {}
    solid_faces: list[list[tuple[float, float, float]]] = []
    for y in range(ny):
        for x in range(nx):
            if mask[z, y, x]:
                lab = int(labels_arr[z, y, x])
                faces_by_lab.setdefault(lab, []).append(section_tile(x, y, 0.0))
            else:
                solid_faces.extend(section_prism(x, y, 0.0, 0.38))

    for lab, faces in faces_by_lab.items():
        add_faces(ax, faces, palette[lab % len(palette)], alpha=0.82, lw=0.24, edge="#ffffff")
    add_faces_lit(ax, solid_faces, FIG2_COL["stack_solid"], alpha=0.86, lw=0.18, edge="#f1f5f9", ambient=0.58, diffuse=0.50)

    frame_color = "#9aa8b6"
    ax.plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [0.012] * 5, color=frame_color, lw=0.72, alpha=0.78)

    visible_sites = [(x, y, zz) for x, y, zz in sites if zz == z]
    if visible_sites:
        pts = np.asarray([(x + 0.5, y + 0.5, 0.15) for x, y, _ in visible_sites])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=38, c=COL["ink"], edgecolors="white", linewidths=0.85, depthshade=False, zorder=8)
        ax.text(visible_sites[0][0] + 0.72, visible_sites[0][1] + 0.32, 0.34, "$s_i$", color=COL["ink"], fontsize=7.4)

    target = (5, 4, z)
    path = shortest_path_3d(mask, sites[0], target)
    if len(path):
        sec_path = path.copy()
        sec_path[:, 2] = 0.28
        ax.plot(sec_path[:, 0], sec_path[:, 1], sec_path[:, 2], color=FIG2_COL["path"], lw=2.75, zorder=10)
        ax.scatter(sec_path[:, 0], sec_path[:, 1], sec_path[:, 2], s=9, c=FIG2_COL["path"], depthshade=False, zorder=11)
    probe = np.asarray([target[0] + 0.5, target[1] + 0.5, 0.31])
    ax.scatter([probe[0]], [probe[1]], [probe[2]], s=48, c="white", edgecolors=FIG2_COL["path"], linewidths=1.2, depthshade=False, zorder=12)
    ax.text(probe[0] + 0.28, probe[1] + 0.18, probe[2] + 0.08, "$L(x)=s_i$", color=FIG2_COL["path"], fontsize=7.0)
    return path, probe


def setup_cell_reveal_panel(ax, mask: np.ndarray, panel: str, title: str) -> None:
    nz, ny, nx = mask.shape
    ax.set_xlim(-0.35, nx + 0.20)
    ax.set_ylim(-0.35, ny + 0.25)
    ax.set_zlim(-0.20, nz + 0.25)
    ax.set_box_aspect((nx, ny, nz), zoom=1.04)
    ax.view_init(elev=24, azim=-43)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.text2D(0.00, 0.98, panel, transform=ax.transAxes, fontweight="bold", color=COL["ink"])
    ax.set_title(title, pad=1, fontsize=8.2)


def draw_geodesic_cell_reveal(
    ax,
    mask: np.ndarray,
    labels_arr: np.ndarray,
    sites: list[tuple[int, int, int]],
    label_id: int = 0,
) -> None:
    nz, ny, nx = mask.shape
    cell_voxels = {
        (x, y, z)
        for z in range(nz)
        for y in range(ny)
        for x in range(nx)
        if mask[z, y, x] and int(labels_arr[z, y, x]) == label_id
    }
    cell_faces = voxel_faces(cell_voxels, (0.0, 0.0, 0.0))

    solid_faces: list[list[tuple[float, float, float]]] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if mask[z, y, x]:
                    continue
                for d in DIR3:
                    bx, by, bz = x + d[0], y + d[1], z + d[2]
                    if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and not mask[bz, by, bx]:
                        continue
                    solid_faces.append(face_vertices(x, y, z, d))

    draw_open_domain_frame(ax, mask)
    add_faces_lit(ax, solid_faces, FIG2_COL["solid"], alpha=0.12, lw=0.14, edge="#cde2d7", ambient=0.55, diffuse=0.45)
    add_faces_lit(ax, cell_faces, FIG2_COL["cell_i"], alpha=0.72, lw=0.22, edge="#fffaf0", ambient=0.60, diffuse=0.48)

    sx, sy, sz = sites[label_id]
    site = np.asarray([sx + 0.5, sy + 0.5, sz + 0.5])
    ax.scatter([site[0]], [site[1]], [site[2]], s=54, c=COL["ink"], edgecolors="white", linewidths=0.75, depthshade=False)
    ax.text(site[0] + 0.35, site[1] + 0.25, site[2] + 0.30, "$s_i$", color=COL["ink"], fontsize=7.8)
    ax.text(1.00, 6.10, 5.60, "$\\Omega_i$", color=COL["ink"], fontsize=8.2)


def input_object_legend(ax) -> None:
    handles = [
        patches.Patch(facecolor=FIG2_COL["domain"], edgecolor=FIG2_COL["frame"], alpha=0.70, label="traversable $V$"),
        patches.Patch(facecolor=FIG2_COL["solid"], edgecolor="white", alpha=0.90, label="solid"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FIG2_COL["site"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.5,
            label="sites $S_g$",
        ),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.03),
        borderpad=0.18,
        labelspacing=0.15,
        handlelength=0.9,
        handletextpad=0.36,
        frameon=True,
        fancybox=False,
        fontsize=5.2,
    )
    leg.get_frame().set_edgecolor("#d7dde5")
    leg.get_frame().set_linewidth(0.45)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.86)


def stack_path(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return path
    out = path.copy()
    out[:, 2] = (out[:, 2] - 0.5) * Z_GAP + 0.78
    return out


def ownership_probe_path(
    mask: np.ndarray,
    labels_arr: np.ndarray,
    dist: np.ndarray,
    sites: list[tuple[int, int, int]],
    owner: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask & (labels_arr == owner))
    if len(coords) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    z, y, x = coords[np.argmax(dist[mask & (labels_arr == owner)])]
    path = shortest_path_3d(mask, sites[owner], (int(x), int(y), int(z)))
    return path, np.asarray([x + 0.5, y + 0.5, z_stack(int(z)) + 0.08])


def translated_face(face: list[tuple[float, float, float]], offset: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    ox, oy, oz = offset
    return [(x + ox, y + oy, z + oz) for x, y, z in face]


def voxel_faces(voxels: set[tuple[int, int, int]], offset: tuple[float, float, float]) -> list[list[tuple[float, float, float]]]:
    faces: list[list[tuple[float, float, float]]] = []
    for x, y, z in voxels:
        for d in DIR3:
            nb = (x + d[0], y + d[1], z + d[2])
            if nb not in voxels:
                faces.append(translated_face(face_vertices(x, y, z, d), offset))
    return faces


def merged_voxel_faces(voxels: set[tuple[int, int, int]], offset: tuple[float, float, float]) -> list[list[tuple[float, float, float]]]:
    def rectangles(units: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
        remaining = set(units)
        rects: list[tuple[int, int, int, int]] = []
        while remaining:
            u0, v0 = min(remaining, key=lambda item: (item[1], item[0]))
            u1 = u0
            while (u1 + 1, v0) in remaining:
                u1 += 1
            v1 = v0
            while all((u, v1 + 1) in remaining for u in range(u0, u1 + 1)):
                v1 += 1
            for u in range(u0, u1 + 1):
                for v in range(v0, v1 + 1):
                    remaining.remove((u, v))
            rects.append((u0, u1 + 1, v0, v1 + 1))
        return rects

    face_units: dict[tuple[int, int, int], dict[int, set[tuple[int, int]]]] = {d: {} for d in DIR3}
    for x, y, z in voxels:
        for d in DIR3:
            dx, dy, dz = d
            if (x + dx, y + dy, z + dz) in voxels:
                continue
            if dx:
                plane = x + (1 if dx > 0 else 0)
                uv = (y, z)
            elif dy:
                plane = y + (1 if dy > 0 else 0)
                uv = (x, z)
            else:
                plane = z + (1 if dz > 0 else 0)
                uv = (x, y)
            face_units[d].setdefault(plane, set()).add(uv)

    faces: list[list[tuple[float, float, float]]] = []
    for d, planes in face_units.items():
        dx, dy, dz = d
        for plane, units in planes.items():
            for u0, u1, v0, v1 in rectangles(units):
                if dx == 1:
                    face = [(plane, u0, v0), (plane, u1, v0), (plane, u1, v1), (plane, u0, v1)]
                elif dx == -1:
                    face = [(plane, u0, v0), (plane, u0, v1), (plane, u1, v1), (plane, u1, v0)]
                elif dy == 1:
                    face = [(u0, plane, v0), (u0, plane, v1), (u1, plane, v1), (u1, plane, v0)]
                elif dy == -1:
                    face = [(u0, plane, v0), (u1, plane, v0), (u1, plane, v1), (u0, plane, v1)]
                elif dz == 1:
                    face = [(u0, v0, plane), (u1, v0, plane), (u1, v1, plane), (u0, v1, plane)]
                else:
                    face = [(u0, v0, plane), (u0, v1, plane), (u1, v1, plane), (u1, v0, plane)]
                faces.append(translated_face(face, offset))
    return faces


def prescribed_cell_demo() -> tuple[
    set[tuple[int, int, int]],
    set[tuple[int, int, int]],
    list[list[tuple[float, float, float]]],
    list[list[tuple[float, float, float]]],
    np.ndarray,
    np.ndarray,
]:
    left = {(x, y, z) for x in range(0, 3) for y in range(0, 3) for z in range(0, 3)}
    right = {(x, y, z) for x in range(3, 6) for y in range(0, 3) for z in range(0, 3)}
    interface = []
    wall = []
    for y in range(0, 3):
        for z in range(0, 3):
            interface.append(face_vertices(2, y, z, (1, 0, 0)))
    for x, y, z in sorted(left):
        if y == 0 and x < 2:
            wall.append(face_vertices(x, y, z, (0, -1, 0)))
    si = np.array([1.50, 1.50, 1.50])
    sj = np.array([4.50, 1.50, 1.50])
    return left, right, interface, wall, si, sj


def merge_coplanar_unit_facelets(
    faces: list[list[tuple[float, float, float]]],
) -> list[tuple[int, list[tuple[float, float, float]]]]:
    """Merge coplanar unit facelets into larger rectangles for cleaner CAD rendering."""

    def rectangles(units: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
        remaining = set(units)
        rects: list[tuple[int, int, int, int]] = []
        while remaining:
            u0, v0 = min(remaining, key=lambda item: (item[1], item[0]))
            u1 = u0
            while (u1 + 1, v0) in remaining:
                u1 += 1
            v1 = v0
            while all((u, v1 + 1) in remaining for u in range(u0, u1 + 1)):
                v1 += 1
            for u in range(u0, u1 + 1):
                for v in range(v0, v1 + 1):
                    remaining.remove((u, v))
            rects.append((u0, u1 + 1, v0, v1 + 1))
        return rects

    grouped: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for face in faces:
        pts = np.asarray(face, dtype=float)
        axis = int(np.argmin(pts.max(axis=0) - pts.min(axis=0)))
        plane = int(round(float(pts[0, axis])))
        uv_axes = [a for a in range(3) if a != axis]
        u0 = int(round(float(pts[:, uv_axes[0]].min())))
        v0 = int(round(float(pts[:, uv_axes[1]].min())))
        grouped.setdefault((axis, plane), set()).add((u0, v0))

    merged: list[tuple[int, list[tuple[float, float, float]]]] = []
    for (axis, plane), units in grouped.items():
        uv_axes = [a for a in range(3) if a != axis]
        for u0, u1, v0, v1 in rectangles(units):
            corners: list[tuple[float, float, float]] = []
            for uu, vv in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]:
                p = [0.0, 0.0, 0.0]
                p[axis] = float(plane)
                p[uv_axes[0]] = float(uu)
                p[uv_axes[1]] = float(vv)
                corners.append(tuple(p))
            merged.append((axis, corners))
    return merged


def solidify_facelet(
    axis: int,
    face: list[tuple[float, float, float]],
    thickness: float = 0.085,
) -> list[list[tuple[float, float, float]]]:
    n = np.zeros(3, dtype=float)
    n[axis] = 1.0
    pts = [np.asarray(p, dtype=float) for p in face]
    front = [p + 0.5 * thickness * n for p in pts]
    back = [p - 0.5 * thickness * n for p in pts]
    faces = [[tuple(p) for p in front], [tuple(p) for p in reversed(back)]]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([tuple(front[i]), tuple(front[j]), tuple(back[j]), tuple(back[i])])
    return faces


def transform_faces(
    faces: list[list[tuple[float, float, float]]],
    center: np.ndarray,
    scale: float,
    source_center: np.ndarray,
) -> list[list[tuple[float, float, float]]]:
    return [
        [tuple(center + scale * (np.asarray(vertex, dtype=float) - source_center)) for vertex in face]
        for face in faces
    ]


def _render_facelets_graph_cad_exploded(
    mask: np.ndarray,
    sites: list[tuple[int, int, int]],
    labels_arr: np.ndarray,
    raw: Path,
    out: Path,
    pv,
    pair: tuple[int, int] | None = None,
) -> Path:
    nz, ny, nx = mask.shape
    camera_position = np.asarray((11.0, -12.0, 7.6), dtype=float)
    camera_target = np.asarray((0.35, 0.05, 1.45), dtype=float)
    view_dir = camera_target - camera_position
    view_dir = view_dir / np.linalg.norm(view_dir)
    view_to_camera = -view_dir

    target_pair = tuple(sorted(pair)) if pair is not None else (0, 1)
    contact_faces_by_label: dict[int, list[list[tuple[float, float, float]]]] = {0: [], 1: [], 2: [], 3: []}
    highlighted_interface_faces: list[list[tuple[float, float, float]]] = []
    highlighted_pair_faces_by_label: dict[int, list[list[tuple[float, float, float]]]] = {0: [], 1: [], 2: [], 3: []}
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    continue
                lab = int(labels_arr[z, y, x])
                for d in DIR3:
                    dx, dy, dz = d
                    bx, by, bz = x + dx, y + dy, z + dz
                    if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz:
                        if not mask[bz, by, bx]:
                            contact_faces_by_label.setdefault(lab, []).append(face_vertices(x, y, z, d))
                        else:
                            nb_lab = int(labels_arr[bz, by, bx])
                            if nb_lab != lab and (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                                face = face_vertices(x, y, z, d)
                                visible_lab = lab if float(np.dot(np.asarray(d, dtype=float), view_to_camera)) >= 0.0 else nb_lab
                                contact_faces_by_label.setdefault(visible_lab, []).append(face)
                                if tuple(sorted((lab, nb_lab))) == target_pair:
                                    highlighted_interface_faces.append(face)
                                    highlighted_pair_faces_by_label.setdefault(lab, []).append(face)
                                    highlighted_pair_faces_by_label.setdefault(nb_lab, []).append(face)

    contact_solid_face_sets: dict[int, list[list[tuple[float, float, float]]]] = {}
    for lab_id, faces in contact_faces_by_label.items():
        solids_for_label: list[list[tuple[float, float, float]]] = []
        for axis, face in merge_coplanar_unit_facelets(faces):
            solids_for_label.extend(solidify_facelet(axis, face))
        if solids_for_label:
            contact_solid_face_sets[lab_id] = solids_for_label
    contact_solid_faces = [face for faces in contact_solid_face_sets.values() for face in faces]
    highlighted_interface_solid_faces: list[list[tuple[float, float, float]]] = []
    for axis, face in merge_coplanar_unit_facelets(highlighted_interface_faces):
        highlighted_interface_solid_faces.extend(solidify_facelet(axis, face, thickness=0.105))

    solid_voxels = {
        (x, y, z)
        for z in range(nz)
        for y in range(ny)
        for x in range(nx)
        if not mask[z, y, x]
    }
    obstacle_unit_faces = [
        face_vertices(x, y, z, d)
        for x, y, z in solid_voxels
        for d in DIR3
        if (x + d[0], y + d[1], z + d[2]) not in solid_voxels
    ]
    obstacle_faces = merged_voxel_faces(solid_voxels, (0.0, 0.0, 0.0))
    source_pts = np.asarray([pt for face in obstacle_faces + contact_solid_faces for pt in face], dtype=float)
    source_center = 0.5 * (source_pts.min(axis=0) + source_pts.max(axis=0))
    mother_center = np.asarray((0.0, 0.05, 1.35), dtype=float)
    mother_obstacle_faces = transform_faces(obstacle_faces, mother_center, 1.0, source_center)
    mother_obstacle_unit_faces = transform_faces(obstacle_unit_faces, mother_center, 1.0, source_center)
    mother_interface_faces = transform_faces(contact_solid_faces, mother_center, 1.0, source_center)
    mother_highlight_interface_faces = transform_faces(highlighted_interface_solid_faces, mother_center, 1.0, source_center)

    base_vectors = {
        2: np.asarray((-2.20, 1.65, 0.0), dtype=float),
        3: np.asarray((2.35, 1.55, 0.0), dtype=float),
        0: np.asarray((-2.35, -1.75, 0.0), dtype=float),
        1: np.asarray((2.45, -1.65, 0.0), dtype=float),
    }
    previous_factor = 1.22
    extra_voxels = 3.0
    explode_vectors: dict[int, np.ndarray] = {}
    for lab_id, vector in base_vectors.items():
        old_vector = vector * previous_factor
        explode_vectors[lab_id] = old_vector + extra_voxels * old_vector / np.linalg.norm(old_vector)

    up_hint = np.asarray((0, 0, 1), dtype=float)
    screen_right = np.cross(view_dir, up_hint)
    screen_right = screen_right / np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, view_dir)
    screen_up = screen_up / np.linalg.norm(screen_up)

    cell_color_by_label = {
        0: FIG2_COL["cell_i"],
        1: FIG2_COL["cell_j"],
        2: FIG2_COL["cell_k"],
        3: FIG2_COL["wall"],
    }
    edge_color_by_label = {
        0: "#fff4d6",
        1: "#eef6fb",
        2: "#f5ebf5",
        3: "#eff8df",
    }
    def brighten(color: str, amount: float = 0.26) -> str:
        rgb = np.asarray(to_rgb(color), dtype=float)
        rgb = rgb + (1.0 - rgb) * amount
        return "#%02x%02x%02x" % tuple(np.clip(np.round(rgb * 255), 0, 255).astype(int))

    def facelet_grid_sides(
        face: list[tuple[float, float, float]],
        thickness: float = 0.085,
    ) -> list[list[tuple[float, float, float]]]:
        normal = face_normal(face)
        pts = [np.asarray(p, dtype=float) for p in face]
        front = [tuple(p + 0.5 * thickness * normal) for p in pts]
        back = [tuple(p - 0.5 * thickness * normal) for p in reversed(pts)]
        return [front, back]

    facelet_color_by_label = {lab_id: brighten(color, 0.28) for lab_id, color in cell_color_by_label.items()}
    mother_interface_face_sets = [
        (
            transform_faces(faces, mother_center, 1.0, source_center),
            facelet_color_by_label.get(lab_id, FIG2_COL["interface"]),
            FIG2_COL["interface"],
        )
        for lab_id, faces in contact_solid_face_sets.items()
    ]
    mother_interface_grid_sets = [
        transform_faces(
            [side for face in faces for side in facelet_grid_sides(face)],
            mother_center,
            1.0,
            source_center,
        )
        for faces in contact_faces_by_label.values()
    ]

    cell_face_sets: list[
        tuple[
            list[list[tuple[float, float, float]]],
            list[list[tuple[float, float, float]]],
            str,
            str,
            int,
            np.ndarray | None,
        ]
    ] = []
    site_marker_positions: list[tuple[int, np.ndarray]] = []
    cell_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lab_id in [2, 3, 0, 1]:
        voxels = {
            (x, y, z)
            for z in range(nz)
            for y in range(ny)
            for x in range(nx)
            if mask[z, y, x] and int(labels_arr[z, y, x]) == lab_id
        }
        if not voxels:
            continue
        assembled_faces = transform_faces(
            merged_voxel_faces(voxels, (0.0, 0.0, 0.0)),
            mother_center,
            1.0,
            source_center,
        )
        assembled_unit_faces = transform_faces(
            [
                face_vertices(x, y, z, d)
                for x, y, z in voxels
                for d in DIR3
                if (x + d[0], y + d[1], z + d[2]) not in voxels
            ],
            mother_center,
            1.0,
            source_center,
        )
        moved_faces = [
            [tuple(np.asarray(vertex, dtype=float) + explode_vectors[lab_id]) for vertex in face]
            for face in assembled_faces
        ]
        moved_unit_faces = [
            [tuple(np.asarray(vertex, dtype=float) + explode_vectors[lab_id]) for vertex in face]
            for face in assembled_unit_faces
        ]
        pts = np.asarray([pt for face in moved_faces for pt in face], dtype=float)
        cell_bounds[lab_id] = (pts.min(axis=0), pts.max(axis=0))
        site_pos = None
        if lab_id in (0, 1):
            site_pos = np.asarray(sites[lab_id], dtype=float) + 0.5 - source_center + mother_center + explode_vectors[lab_id]
            site_marker_positions.append((lab_id, site_pos))
        cell_face_sets.append((moved_faces, moved_unit_faces, cell_color_by_label[lab_id], edge_color_by_label[lab_id], lab_id, site_pos))
    highlighted_cell_face_sets: list[list[list[tuple[float, float, float]]]] = []
    for lab_id, faces in highlighted_pair_faces_by_label.items():
        if not faces or lab_id not in explode_vectors:
            continue
        merged_faces = [face for _, face in merge_coplanar_unit_facelets(faces)]
        base_faces = transform_faces(merged_faces, mother_center, 1.0, source_center)
        moved_faces = [
            [
                tuple(np.asarray(vertex, dtype=float) + explode_vectors[lab_id] + 0.018 * view_to_camera)
                for vertex in face
            ]
            for face in base_faces
        ]
        highlighted_cell_face_sets.append(moved_faces)

    pl = pv.Plotter(off_screen=True, window_size=(2700, 1600))
    pl.set_background("white")

    def add_face_mesh(
        faces: list[list[tuple[float, float, float]]],
        color: str,
        edge: str,
        opacity: float,
        line_width: float,
        ambient: float,
        diffuse: float,
        specular: float = 0.035,
        show_edges: bool = True,
    ) -> None:
        if not faces:
            return
        pts: list[tuple[float, float, float]] = []
        polys: list[list[int]] = []
        for face in faces:
            base = len(pts)
            pts.extend(face)
            polys.append([4, base, base + 1, base + 2, base + 3])
        mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(polys, dtype=np.int64).ravel())
        pl.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge,
            line_width=line_width,
            smooth_shading=False,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=16,
        )

    def add_surface_grid_lines(
        faces: list[list[tuple[float, float, float]]],
        color: str = "#ffffff",
    ) -> None:
        points: list[tuple[float, float, float]] = []
        lines: list[int] = []
        seen: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
        for face in faces:
            normal = face_normal(face)
            shifted = [tuple(np.asarray(vertex, dtype=float) + 0.012 * normal) for vertex in face]
            for idx in range(4):
                a0 = tuple(np.round(np.asarray(face[idx], dtype=float), 5))
                b0 = tuple(np.round(np.asarray(face[(idx + 1) % 4], dtype=float), 5))
                key = tuple(sorted((a0, b0)))
                if key in seen:
                    continue
                seen.add(key)
                base = len(points)
                points.append(shifted[idx])
                points.append(shifted[(idx + 1) % 4])
                lines.extend([2, base, base + 1])
        if not points:
            return
        mesh = pv.PolyData(np.asarray(points, dtype=float))
        mesh.lines = np.asarray(lines, dtype=np.int64)
        pl.add_mesh(
            mesh,
            color=color,
            opacity=0.58,
            line_width=0.85,
            render_lines_as_tubes=False,
            lighting=False,
        )

    def is_outer_shell_face(
        face: list[tuple[float, float, float]],
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        minv, maxv = bounds
        normal = face_normal(face)
        axis = int(np.argmax(np.abs(normal)))
        center = np.mean(np.asarray(face, dtype=float), axis=0)
        target = maxv[axis] if normal[axis] > 0.0 else minv[axis]
        return abs(float(center[axis] - target)) < 1.0e-5

    for faces, unit_faces, color, edge, lab_id, site_pos in cell_face_sets:
        faces_to_draw = faces
        unit_faces_to_draw = unit_faces
        bounds = cell_bounds[lab_id]
        if lab_id in (0, 1) and site_pos is not None:
            _, maxv = bounds
            def in_site_opening(face: list[tuple[float, float, float]]) -> bool:
                return (
                    abs(face_normal(face)[2]) < 0.25
                    and float(np.dot(face_normal(face), view_to_camera)) > 0.12
                    and abs(float(np.dot(np.mean(np.asarray(face), axis=0) - site_pos, screen_right))) < 3.10
                    and abs(float(np.dot(np.mean(np.asarray(face), axis=0) - site_pos, screen_up))) < 2.40
                )

            def opens_site(face: list[tuple[float, float, float]]) -> bool:
                return (
                    in_site_opening(face)
                    and not (maxv is not None and float(np.mean(np.asarray(face), axis=0)[0]) > float(maxv[0]) - 1.0e-6)
                )

            open_indices = {idx for idx, face in enumerate(faces) if opens_site(face)}
            faces_to_draw = [face for idx, face in enumerate(faces) if idx not in open_indices]
            unit_faces_to_draw = [
                face
                for face in unit_faces
                if not opens_site(face) and face_normal(face)[2] > -0.75
            ]
        add_face_mesh(faces_to_draw, color, edge, 1.0, 0.0, 0.34, 0.68, 0.045, False)
        grid_faces = [face for face in unit_faces_to_draw if is_outer_shell_face(face, bounds)]
        if lab_id in (0, 1):
            _, maxv = bounds
            grid_faces = [
                face
                for face in grid_faces
                if (
                    face_normal(face)[2] > 0.75
                    or (
                        face_normal(face)[0] > 0.75
                        and abs(float(np.mean(np.asarray(face), axis=0)[0]) - float(maxv[0])) < 1.0e-5
                    )
                )
            ]
        add_surface_grid_lines(grid_faces)
    for faces in highlighted_cell_face_sets:
        add_face_mesh(faces, FIG2_COL["interface"], "#ffffff", 1.0, 0.80, 0.42, 0.70, 0.09, True)

    add_face_mesh(mother_obstacle_faces, FIG2_COL["solid"], FIG2_COL["solid"], 1.0, 0.0, 0.34, 0.66, 0.035, False)
    add_surface_grid_lines(mother_obstacle_unit_faces)
    for faces, color, edge in mother_interface_face_sets:
        add_face_mesh(faces, color, edge, 1.0, 0.55, 0.46, 0.68, 0.075, True)
    for faces in mother_interface_grid_sets:
        add_surface_grid_lines(faces)

    for _, node in site_marker_positions:
        pl.add_mesh(
            pv.Sphere(radius=0.34, center=tuple(node), theta_resolution=48, phi_resolution=24),
            color=FIG2_COL["site"],
            smooth_shading=True,
            ambient=0.42,
            diffuse=0.66,
            specular=0.18,
            specular_power=22,
        )

    pl.remove_all_lights()
    key_light_pos = camera_target - 7.5 * view_dir - 10.5 * screen_right + 9.5 * screen_up
    fill_light_pos = camera_target - 4.5 * view_dir + 9.0 * screen_right + 2.8 * screen_up
    rim_light_pos = camera_target + 6.0 * view_dir - 2.0 * screen_right + 6.0 * screen_up
    pl.add_light(pv.Light(position=tuple(key_light_pos), focal_point=tuple(camera_target), color="white", intensity=0.92))
    pl.add_light(pv.Light(position=tuple(fill_light_pos), focal_point=tuple(camera_target), color="#eef8f3", intensity=0.16))
    pl.add_light(pv.Light(position=tuple(rim_light_pos), focal_point=tuple(camera_target), color="#f7fbf8", intensity=0.08))
    try:
        pl.enable_ssao(radius=1.0, bias=0.018, kernel_size=256)
    except Exception:
        pass
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    pl.camera_position = [tuple(camera_position), tuple(camera_target), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 9.05
    pl.show(screenshot=str(raw), auto_close=True)
    crop_near_white_image(raw, out, pad=70)
    return out


def render_facelets_graph_cad(
    mask: np.ndarray | None = None,
    sites: list[tuple[int, int, int]] | None = None,
    labels_arr: np.ndarray | None = None,
    pair: tuple[int, int] | None = None,
    out_stub: str = "Figure_03d_facelets_graph_cad",
) -> Path | None:
    try:
        import pyvista as pv
    except ImportError:
        return None

    FIG.mkdir(parents=True, exist_ok=True)
    raw = FIG / f"{out_stub}_raw.png"
    out = FIG / f"{out_stub}.png"

    if mask is not None and sites is not None and labels_arr is not None:
        return _render_facelets_graph_cad_exploded(mask, sites, labels_arr, raw, out, pv, pair)

        # Panel (d) is an exploded CAD view.  The four cells below are generated
        # in one assembled coordinate system and then translated outward, so
        # removing the explosion vectors reconstructs a single rectangular
        # cell block attached to the same interface face.
        pl = pv.Plotter(off_screen=True, window_size=(2000, 1500))
        pl.set_background("white")

        def add_box(
            bounds: tuple[float, float, float, float, float, float],
            color: str,
            ambient: float = 0.28,
            diffuse: float = 0.76,
            specular: float = 0.045,
        ) -> None:
            mesh = pv.Box(bounds=bounds)
            pl.add_mesh(
                mesh,
                color=color,
                opacity=1.0,
                show_edges=False,
                smooth_shading=False,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                specular_power=18,
            )

        def shifted_bounds(
            bounds: tuple[float, float, float, float, float, float],
            shift: tuple[float, float, float],
        ) -> tuple[float, float, float, float, float, float]:
            dx, dy, dz = shift
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            return (xmin + dx, xmax + dx, ymin + dy, ymax + dy, zmin + dz, zmax + dz)

        def add_site(center: tuple[float, float, float]) -> None:
            pl.add_mesh(
                pv.Sphere(radius=0.34, center=center, theta_resolution=48, phi_resolution=24),
                color=FIG2_COL["site"],
                smooth_shading=True,
                ambient=0.42,
                diffuse=0.66,
                specular=0.18,
                specular_power=22,
            )

        x0, x1 = 0.0, 2.35
        y0, y1, y2 = -3.2, 0.0, 3.2
        z0, z1, z2 = 0.0, 2.2, 4.4
        face_thickness = 0.12
        add_box(
            (-face_thickness, face_thickness, y0, y2, z0, z2),
            FIG2_COL["interface"],
            ambient=0.42,
            diffuse=0.70,
            specular=0.075,
        )
        # A pair of solid-contact patches on the same face keeps the wall/solid
        # meaning visible while preserving the assembled block scale.
        add_box((-0.14, 0.14, -2.9, -1.15, 3.35, 4.28), FIG2_COL["solid"], ambient=0.32, diffuse=0.74, specular=0.035)
        add_box((-0.14, 0.14, 1.15, 2.9, 3.35, 4.28), FIG2_COL["solid"], ambient=0.32, diffuse=0.74, specular=0.035)
        add_box((-0.14, 0.14, -2.95, -1.55, 0.15, 1.05), FIG2_COL["solid"], ambient=0.32, diffuse=0.74, specular=0.035)

        assembled_cells = {
            2: ((x0, x1, y0, y1, z1, z2), FIG2_COL["cell_k"], (-2.55, -1.35, 1.05)),
            3: ((x0, x1, y1, y2, z1, z2), FIG2_COL["wall"], (2.55, 1.35, 1.05)),
            0: ((x0, x1, y0, y1, z0, z1), FIG2_COL["cell_i"], (-2.75, -1.50, -1.05)),
            1: ((x0, x1, y1, y2, z0, z1), FIG2_COL["cell_j"], (2.75, 1.50, -1.05)),
        }
        site_centres: list[tuple[float, float, float]] = []
        for lab_id, (bounds, color, shift) in assembled_cells.items():
            add_box(shifted_bounds(bounds, shift), color)
            if lab_id in (0, 1):
                xmin, xmax, ymin, ymax, zmin, zmax = shifted_bounds(bounds, shift)
                site_centres.append((xmin + 0.55 * (xmax - xmin), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)))
        for center in site_centres:
            add_site(center)

        camera_position = np.asarray((7.5, -9.2, 6.2), dtype=float)
        camera_target = np.asarray((1.05, 0.0, 2.05), dtype=float)
        view_dir = camera_target - camera_position
        view_dir = view_dir / np.linalg.norm(view_dir)
        up_hint = np.asarray((0, 0, 1), dtype=float)
        screen_right = np.cross(view_dir, up_hint)
        screen_right = screen_right / np.linalg.norm(screen_right)
        screen_up = np.cross(screen_right, view_dir)
        screen_up = screen_up / np.linalg.norm(screen_up)

        pl.remove_all_lights()
        key_light_pos = camera_target - 7.5 * view_dir - 10.5 * screen_right + 9.5 * screen_up
        fill_light_pos = camera_target - 4.5 * view_dir + 9.0 * screen_right + 2.8 * screen_up
        rim_light_pos = camera_target + 6.0 * view_dir - 2.0 * screen_right + 6.0 * screen_up
        pl.add_light(pv.Light(position=tuple(key_light_pos), focal_point=tuple(camera_target), color="white", intensity=0.92))
        pl.add_light(pv.Light(position=tuple(fill_light_pos), focal_point=tuple(camera_target), color="#eef8f3", intensity=0.16))
        pl.add_light(pv.Light(position=tuple(rim_light_pos), focal_point=tuple(camera_target), color="#f7fbf8", intensity=0.08))
        try:
            pl.enable_ssao(radius=1.0, bias=0.018, kernel_size=256)
        except Exception:
            pass
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        pl.camera_position = [tuple(camera_position), tuple(camera_target), (0, 0, 1)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = 6.4
        pl.show(screenshot=str(raw), auto_close=True)
        crop_near_white_image(raw, out, pad=95)
        return out

        nz, ny, nx = mask.shape
        camera_position = np.asarray((4.2, -19.0, 9.8), dtype=float)
        camera_target = np.asarray((0.0, 0.0, 1.25), dtype=float)
        view_dir = camera_target - camera_position
        view_dir = view_dir / np.linalg.norm(view_dir)
        view_to_camera = camera_position - camera_target
        view_to_camera = view_to_camera / np.linalg.norm(view_to_camera)
        up_hint = np.asarray((0, 0, 1), dtype=float)
        screen_right = np.cross(view_dir, up_hint)
        screen_right = screen_right / np.linalg.norm(screen_right)
        screen_up = np.cross(screen_right, view_dir)
        screen_up = screen_up / np.linalg.norm(screen_up)
        labels_present = [2, 3, 0, 1]
        layout_targets = [
            np.asarray((-5.05, 3.20, 0.0), dtype=float),
            np.asarray((5.05, 3.20, 0.0), dtype=float),
            np.asarray((-5.05, -3.20, 0.0), dtype=float),
            np.asarray((5.05, -3.20, 0.0), dtype=float),
        ]
        cell_color_by_label = {
            0: FIG2_COL["cell_i"],
            1: FIG2_COL["cell_j"],
            2: FIG2_COL["cell_k"],
            3: FIG2_COL["wall"],
        }
        edge_color_by_label = {
            0: "#fff4d6",
            1: "#eef6fb",
            2: "#f5ebf5",
            3: "#eff8df",
        }
        cell_face_sets: list[tuple[list[list[tuple[float, float, float]]], str, str, int, np.ndarray | None]] = []
        site_marker_positions: list[tuple[int, np.ndarray]] = []
        cell_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        label_offsets: dict[int, np.ndarray] = {}
        for pos, lab_id in enumerate(labels_present):
            voxels = {
                (x, y, z)
                for z in range(nz)
                for y in range(ny)
                for x in range(nx)
                if mask[z, y, x] and int(labels_arr[z, y, x]) == lab_id
            }
            if not voxels:
                continue
            coords = np.asarray([(x + 0.5, y + 0.5, z + 0.5) for x, y, z in voxels], dtype=float)
            center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
            offset = tuple(layout_targets[pos] - center)
            label_offsets[lab_id] = np.asarray(offset, dtype=float)
            site_pos = None
            if lab_id in (0, 1):
                site_pos = np.asarray(sites[lab_id], dtype=float) + 0.5 + np.asarray(offset)
            faces_for_cell = merged_voxel_faces(voxels, offset)
            if faces_for_cell:
                pts_for_bounds = np.asarray([pt for face in faces_for_cell for pt in face], dtype=float)
                cell_bounds[lab_id] = (pts_for_bounds.min(axis=0), pts_for_bounds.max(axis=0))
            cell_face_sets.append((faces_for_cell, cell_color_by_label[lab_id], edge_color_by_label[lab_id], lab_id, site_pos))
            if lab_id in (0, 1) and site_pos is not None:
                site_marker_positions.append((lab_id, site_pos))

        mother_obstacle_faces: list[list[tuple[float, float, float]]] = []
        mother_interface_faces: list[list[tuple[float, float, float]]] = []
        all_interface_faces, _ = intercell_facelets(mask, labels_arr)
        contact_faces = all_interface_faces + wall_facelets(mask, labels_arr, {0, 1, 2, 3})
        contact_solid_faces: list[list[tuple[float, float, float]]] = []
        for axis, face in merge_coplanar_unit_facelets(contact_faces):
            contact_solid_faces.extend(solidify_facelet(axis, face))
        solid_voxels = {
            (x, y, z)
            for z in range(nz)
            for y in range(ny)
            for x in range(nx)
            if not mask[z, y, x]
        }
        obstacle_faces = merged_voxel_faces(solid_voxels, (0.0, 0.0, 0.0))
        mother_faces = obstacle_faces + contact_solid_faces
        if mother_faces:
            pts = np.asarray([pt for face in mother_faces for pt in face], dtype=float)
            source_center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            mother_center = np.asarray((0.0, 0.05, 1.35), dtype=float)
            mother_scale = 1.0
            mother_obstacle_faces = transform_faces(obstacle_faces, mother_center, mother_scale, source_center)
            mother_interface_faces = transform_faces(contact_solid_faces, mother_center, mother_scale, source_center)

        pl = pv.Plotter(off_screen=True, window_size=(2000, 1500))
        pl.set_background("white")

        def add_face_mesh(
            faces: list[list[tuple[float, float, float]]],
            color: str,
            edge: str,
            opacity: float,
            line_width: float,
            ambient: float,
            diffuse: float,
            specular: float = 0.035,
            show_edges: bool = True,
        ) -> None:
            if not faces:
                return
            pts: list[tuple[float, float, float]] = []
            polys: list[list[int]] = []
            for face in faces:
                base = len(pts)
                pts.extend(face)
                polys.append([4, base, base + 1, base + 2, base + 3])
            mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(polys, dtype=np.int64).ravel())
            pl.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                edge_color=edge,
                line_width=line_width,
                smooth_shading=False,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                specular_power=16,
            )

        def add_tube(p0: np.ndarray, p1: np.ndarray, radius: float, color: str, opacity: float = 1.0) -> None:
            line = pv.Line(tuple(p0), tuple(p1))
            tube = line.tube(radius=radius, n_sides=32)
            pl.add_mesh(
                tube,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                ambient=0.22,
                diffuse=0.74,
                specular=0.08,
                specular_power=18,
            )

        for faces, color, edge, lab_id, site_pos in cell_face_sets:
            faces_to_draw = faces
            if lab_id in (0, 1) and site_pos is not None:
                minv, maxv = cell_bounds.get(lab_id, (None, None))
                open_indices = {
                    idx
                    for idx, face in enumerate(faces)
                    if abs(face_normal(face)[2]) < 0.2
                    and float(np.dot(face_normal(face), view_to_camera)) > 0.12
                    and abs(float(np.dot(np.mean(np.asarray(face), axis=0) - site_pos, screen_right))) < 3.05
                    and abs(float(np.dot(np.mean(np.asarray(face), axis=0) - site_pos, screen_up))) < 2.35
                    and not (maxv is not None and float(np.mean(np.asarray(face), axis=0)[0]) > float(maxv[0]) - 1.0e-6)
                }
                faces_to_draw = [face for idx, face in enumerate(faces) if idx not in open_indices]
            add_face_mesh(
                faces_to_draw,
                color,
                edge,
                1.0,
                0.0,
                0.34,
                0.68,
                0.045,
                show_edges=False,
            )

        if mother_obstacle_faces:
            add_face_mesh(
                mother_obstacle_faces,
                FIG2_COL["solid"],
                FIG2_COL["solid"],
                1.0,
                0.0,
                0.34,
                0.66,
                0.035,
                show_edges=False,
            )

        if mother_interface_faces:
            add_face_mesh(
                mother_interface_faces,
                FIG2_COL["interface"],
                FIG2_COL["interface"],
                1.0,
                0.0,
                0.46,
                0.68,
                0.075,
                show_edges=False,
            )

        site_lookup = {lab_id: node for lab_id, node in site_marker_positions}

        for _, node in site_marker_positions:
            pl.add_mesh(
                pv.Sphere(radius=0.44, center=tuple(node), theta_resolution=48, phi_resolution=24),
                color=FIG2_COL["site"],
                smooth_shading=True,
                ambient=0.42,
                diffuse=0.66,
                specular=0.18,
                specular_power=22,
            )

        pl.remove_all_lights()
        key_light_pos = camera_target - 7.5 * view_dir - 10.5 * screen_right + 9.5 * screen_up
        fill_light_pos = camera_target - 4.5 * view_dir + 9.0 * screen_right + 2.8 * screen_up
        rim_light_pos = camera_target + 6.0 * view_dir - 2.0 * screen_right + 6.0 * screen_up
        pl.add_light(pv.Light(position=tuple(key_light_pos), focal_point=tuple(camera_target), color="white", intensity=0.92))
        pl.add_light(pv.Light(position=tuple(fill_light_pos), focal_point=tuple(camera_target), color="#eef8f3", intensity=0.16))
        pl.add_light(pv.Light(position=tuple(rim_light_pos), focal_point=tuple(camera_target), color="#f7fbf8", intensity=0.08))
        try:
            pl.enable_ssao(radius=1.0, bias=0.018, kernel_size=256)
        except Exception:
            pass
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        try:
            pl.enable_depth_peeling()
        except Exception:
            pass
        pl.camera_position = [tuple(camera_position), tuple(camera_target), (0, 0, 1)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = 10.2
        pl.show(screenshot=str(raw), auto_close=True)
        crop_near_white_image(raw, out, pad=95)
        return out

    left, right, interface, wall, si, sj = prescribed_cell_demo()
    gap = 0.92
    left_offset = (-0.40, 0.28, 0.12)
    right_offset = (gap, 0.28, 0.12)
    interface_offset = (gap * 0.50, 0.28, 0.12)
    wall_offset = (-0.40, 0.20, 0.12)
    left_faces = voxel_faces(left, left_offset)
    right_faces = voxel_faces(right, right_offset)
    interface_faces = [translated_face(f, interface_offset) for f in interface]
    wall_faces = [translated_face(f, wall_offset) for f in wall]
    graph_y = left_offset[1] - 0.36
    si_node = np.asarray((si[0] + left_offset[0], graph_y, si[2] + left_offset[2]))
    sj_node = np.asarray((sj[0] + right_offset[0], graph_y, sj[2] + right_offset[2]))
    camera_position = np.asarray((8.6, -7.6, 5.2), dtype=float)
    camera_target = np.asarray((3.05, 1.55, 1.86), dtype=float)

    pl = pv.Plotter(off_screen=True, window_size=(2000, 1500))
    pl.set_background("white")

    def add_face_mesh(
        faces: list[list[tuple[float, float, float]]],
        color: str,
        edge: str,
        opacity: float = 1.0,
        line_width: float = 0.7,
        ambient: float = 0.26,
        diffuse: float = 0.74,
        specular: float = 0.04,
    ) -> None:
        if not faces:
            return
        pts: list[tuple[float, float, float]] = []
        polys: list[list[int]] = []
        for face in faces:
            base = len(pts)
            pts.extend(face)
            polys.append([4, base, base + 1, base + 2, base + 3])
        mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(polys, dtype=np.int64).ravel())
        pl.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=True,
            edge_color=edge,
            line_width=line_width,
            smooth_shading=False,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=18,
        )

    add_face_mesh(left_faces, FIG2_COL["cell_i"], "#fff4d6", line_width=0.64)
    add_face_mesh(right_faces, FIG2_COL["cell_j"], "#eef6fb", line_width=0.64)
    add_face_mesh(wall_faces, FIG2_COL["wall"], "#fff8dc", opacity=0.96, line_width=1.08, ambient=0.30, diffuse=0.72)
    add_face_mesh(interface_faces, FIG2_COL["interface"], "#ffffff", opacity=0.98, line_width=1.22, ambient=0.30, diffuse=0.72)

    def add_tube(p0, p1, radius: float, color: str, opacity: float = 1.0) -> None:
        line = pv.Line(tuple(p0), tuple(p1))
        tube = line.tube(radius=radius, n_sides=32)
        pl.add_mesh(
            tube,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.80,
            specular=0.08,
            specular_power=18,
        )

    add_tube(si_node, sj_node, 0.074, "white", 1.0)
    add_tube(si_node, sj_node, 0.050, COL["ink"], 1.0)

    for node in [si_node, sj_node]:
        pl.add_mesh(
            pv.Sphere(radius=0.155, center=tuple(node), theta_resolution=48, phi_resolution=24),
            color="white",
            smooth_shading=True,
            ambient=0.24,
            diffuse=0.70,
            specular=0.12,
            specular_power=18,
        )
        pl.add_mesh(
            pv.Sphere(radius=0.105, center=tuple(node), theta_resolution=48, phi_resolution=24),
            color=FIG2_COL["site"],
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.74,
            specular=0.10,
            specular_power=18,
        )

    view_dir = camera_target - camera_position
    view_dir = view_dir / np.linalg.norm(view_dir)
    up_hint = np.asarray((0, 0, 1), dtype=float)
    screen_right = np.cross(view_dir, up_hint)
    screen_right = screen_right / np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, view_dir)
    screen_up = screen_up / np.linalg.norm(screen_up)

    pl.remove_all_lights()
    key_light_pos = camera_target - 6.5 * view_dir - 8.0 * screen_right + 7.6 * screen_up
    fill_light_pos = camera_target - 3.8 * view_dir + 7.2 * screen_right + 2.4 * screen_up
    rim_light_pos = camera_target + 4.5 * view_dir - 1.8 * screen_right + 5.0 * screen_up
    pl.add_light(pv.Light(position=tuple(key_light_pos), focal_point=tuple(camera_target), color="white", intensity=0.96))
    pl.add_light(pv.Light(position=tuple(fill_light_pos), focal_point=tuple(camera_target), color="#eef8f3", intensity=0.14))
    pl.add_light(pv.Light(position=tuple(rim_light_pos), focal_point=tuple(camera_target), color="#f7fbf8", intensity=0.08))

    try:
        pl.enable_ssao(radius=1.0, bias=0.018, kernel_size=256)
    except Exception:
        pass
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    pl.camera_position = [tuple(camera_position), tuple(camera_target), (0, 0, 1)]
    pl.camera.parallel_projection = False
    pl.camera.view_angle = 35.0
    pl.show(screenshot=str(raw), auto_close=True)
    crop_near_white_image(raw, out, pad=95)
    return out


def local_cell_faces(
    mask: np.ndarray,
    labels_arr: np.ndarray,
    keep_labels: set[int],
    roi: tuple[slice, slice, slice],
    offsets: dict[int, tuple[float, float, float]],
) -> dict[int, list[list[tuple[float, float, float]]]]:
    zsl, ysl, xsl = roi
    faces_by_lab: dict[int, list[list[tuple[float, float, float]]]] = {}
    nz, ny, nx = mask.shape
    zr = range(*zsl.indices(nz))
    yr = range(*ysl.indices(ny))
    xr = range(*xsl.indices(nx))
    roi_set = {(x, y, z) for z in zr for y in yr for x in xr}
    for z in zr:
        for y in yr:
            for x in xr:
                if not mask[z, y, x]:
                    continue
                lab = int(labels_arr[z, y, x])
                if lab not in keep_labels:
                    continue
                for d in DIR3:
                    dx, dy, dz = d
                    nb = (x + dx, y + dy, z + dz)
                    outside = nb not in roi_set or not (0 <= nb[0] < nx and 0 <= nb[1] < ny and 0 <= nb[2] < nz)
                    if outside or not mask[nb[2], nb[1], nb[0]] or int(labels_arr[nb[2], nb[1], nb[0]]) != lab:
                        faces_by_lab.setdefault(lab, []).append(translated_face(face_vertices(x, y, z, d), offsets.get(lab, (0, 0, 0))))
    return faces_by_lab


def local_facelets(
    mask: np.ndarray,
    labels_arr: np.ndarray,
    pair: tuple[int, int],
    roi: tuple[slice, slice, slice],
    offset: tuple[float, float, float],
) -> list[list[tuple[float, float, float]]]:
    zsl, ysl, xsl = roi
    nz, ny, nx = mask.shape
    faces: list[list[tuple[float, float, float]]] = []
    for z in range(*zsl.indices(nz)):
        for y in range(*ysl.indices(ny)):
            for x in range(*xsl.indices(nx)):
                if not mask[z, y, x]:
                    continue
                lab = int(labels_arr[z, y, x])
                for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    dx, dy, dz = d
                    bx, by, bz = x + dx, y + dy, z + dz
                    if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz and mask[bz, by, bx]:
                        nb_lab = int(labels_arr[bz, by, bx])
                        if tuple(sorted((lab, nb_lab))) == pair:
                            faces.append(translated_face(face_vertices(x, y, z, d), offset))
    return faces


def fig_site_to_cell_complex() -> None:
    mask, sites = small_mask_3d()
    lab, dist = bfs_labels_3d(mask, sites)
    faces, face_pairs = intercell_facelets(mask, lab)
    pair = (0, 1) if (0, 1) in face_pairs else max(face_pairs, key=lambda p: len(face_pairs[p]))
    cad_panel = render_site_mask_cad(mask, sites)
    ownership_path = shortest_path_3d(mask, sites[0], sites[1])
    ownership_panel = render_site_mask_cad(
        mask,
        sites,
        labels_arr=lab,
        path=ownership_path,
        out_stub="Figure_03b_ownership_cad",
    )
    cells_panel = render_site_mask_cad(
        mask,
        sites,
        labels_arr=lab,
        cell_labels_keep={int(v) for v in np.unique(lab[mask])},
        site_indices=set(),
        out_stub="Figure_03c_cells_cad",
    )
    facelet_panel = render_facelets_graph_cad(mask=mask, sites=sites, labels_arr=lab, pair=pair)
    fig = plt.figure(figsize=(7.55, 5.25))
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 0.16, 1.26, 1.26],
        wspace=0.035,
        hspace=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0:2], projection=None if cad_panel else "3d")
    ax_b = fig.add_subplot(gs[0, 3:5], projection=None if ownership_panel else "3d")
    ax_c = fig.add_subplot(gs[1, 0:2], projection=None if cells_panel else "3d")
    ax_d = fig.add_subplot(gs[1, 2:5], projection=None if facelet_panel else "3d")
    cad_panel_inset = [0.00, 0.13, 1.00, 0.82]

    if cad_panel:
        setup_render_panel(ax_a, "(a)", "input: $V$ and fixed $S_g$")
        img_ax = ax_a.inset_axes(cad_panel_inset)
        img_ax.imshow(plt.imread(cad_panel))
        img_ax.set_axis_off()
        ax_a.set_anchor("C")
    else:
        setup_3d_panel(ax_a, mask, "(a)", "input: $V$ and fixed $S_g$")
        draw_cad_domain_box(ax_a, mask)
        draw_solid_surface_3d(ax_a, mask, alpha=0.98, lit=True, color=FIG2_COL["solid"], ambient=0.42, diffuse=1.12)
        draw_site_depth_guides(ax_a, sites, color=FIG2_COL["site"], alpha=0.24)
        plot_site_spheres(ax_a, sites, color=FIG2_COL["site"], radius=0.28)
        plot_sites_3d(ax_a, sites, color=FIG2_COL["site"], size=28, depthshade=False)
        ax_a.text(0.72, 3.35, 4.18, "$S_g$", color=FIG2_COL["site"], fontsize=7.2)
        ax_a.view_init(elev=30, azim=-43)
        ax_a.set_proj_type("persp", focal_length=0.84)

    if ownership_panel:
        setup_render_panel(ax_b, "(b)", "graph-geodesic reachability")
        img_ax = ax_b.inset_axes(cad_panel_inset)
        img_ax.imshow(plt.imread(ownership_panel))
        img_ax.set_axis_off()
        ax_b.set_anchor("C")
    else:
        setup_ownership_section_panel(ax_b, mask, "(b)", "ownership $L(x)$")
        draw_ownership_section(ax_b, mask, lab, sites, z=3)

    if cells_panel:
        setup_render_panel(ax_c, "(c)", "finite-volume cells $\\Omega_k$")
        img_ax = ax_c.inset_axes(cad_panel_inset)
        img_ax.imshow(plt.imread(cells_panel))
        img_ax.set_axis_off()
        ax_c.set_anchor("C")
    else:
        setup_cell_reveal_panel(ax_c, mask, "(c)", "cell $\\Omega_i$")
        draw_geodesic_cell_reveal(ax_c, mask, lab, sites, label_id=0)

    if facelet_panel:
        setup_render_panel(ax_d, "(d)", "FV cells and facelets")
        img_ax = ax_d.inset_axes([0.01, 0.04, 0.98, 0.90])
        img_ax.imshow(plt.imread(facelet_panel))
        img_ax.set_axis_off()
        label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.18}
        ax_d.text(0.50, 0.78, "$\\Omega_\\ell$", transform=ax_d.transAxes, color=COL["ink"], fontsize=7.8, bbox=label_box, zorder=31)
        ax_d.text(0.89, 0.70, "$\\Omega_m$", transform=ax_d.transAxes, color=COL["ink"], fontsize=7.8, bbox=label_box, zorder=31)
        ax_d.text(0.12, 0.28, "$\\Omega_i$", transform=ax_d.transAxes, color=COL["ink"], fontsize=7.8, bbox=label_box, zorder=31)
        ax_d.text(0.70, 0.25, "$\\Omega_j$", transform=ax_d.transAxes, color=COL["ink"], fontsize=7.8, bbox=label_box, zorder=31)
        ax_d.text(0.235, 0.525, "$s_i$", transform=ax_d.transAxes, color=FIG2_COL["site"], fontsize=7.4, bbox=label_box, zorder=31)
        ax_d.text(0.555, 0.305, "$s_j$", transform=ax_d.transAxes, color=FIG2_COL["site"], fontsize=7.4, bbox=label_box, zorder=31)
        ax_d.text(
            0.512,
            0.458,
            "$\\Gamma_{ij}$",
            transform=ax_d.transAxes,
            color="white",
            fontsize=7.0,
            ha="center",
            va="center",
            bbox={"facecolor": FIG2_COL["interface"], "edgecolor": "none", "alpha": 0.86, "pad": 0.10},
            zorder=32,
        )
        ax_d.set_anchor("C")
    else:
        setup_render_panel(ax_d, "(d)", "facelets and graph $G_f$")

    key_ax = fig.add_axes([0.040, 0.012, 0.62, 0.075])
    draw_site_to_cell_key(key_ax)
    fig.subplots_adjust(left=0.012, right=0.992, bottom=0.092, top=0.965)
    save(fig, "Figure_03_site_to_cell_complex", png=True)


def fig_ownership_speed_audit() -> None:
    timing = pd.DataFrame(
        {
            "case": ["Orthogonal", "Skewed", "Thin wall", "Narrow throat", "Maze"],
            "speedup": [11.3, 9.3, 10.4, 9.2, 9.1],
            "t_cuda_ms": [0.100, 0.124, 0.105, 0.123, 0.123],
            "mismatch": [0, 0, 0, 0, 0],
        }
    )
    build = pd.DataFrame(
        {
            "case": ["Orthogonal", "Skewed", "Thin wall", "Narrow throat"],
            "Setup": [0.08, 0.05, 0.04, 0.04],
            "Ownership": [0.76, 0.43, 0.43, 0.51],
            "Components": [3.51, 1.85, 1.87, 2.43],
            "Geometry": [8.19, 7.56, 7.92, 7.37],
        }
    )
    fig, axs = plt.subplots(1, 2, figsize=(8.9, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})
    axs[0].bar(timing["case"], timing["speedup"], color=COL["teal"])
    axs[0].axhspan(9.1, 11.3, color=COL["teal"], alpha=0.10)
    axs[0].set_ylabel("Speedup vs exact GPU labels")
    axs[0].set_ylim(0, 13)
    axs[0].tick_params(axis="x", rotation=35)
    axs[0].text(0.02, 0.95, "0 strict label mismatches\nin all audited rows", transform=axs[0].transAxes, va="top")
    label(axs[0], "(a)")
    bottom = np.zeros(len(build))
    cols = [COL["muted"], COL["blue"], COL["orange"], COL["green"]]
    for c, color in zip(["Setup", "Ownership", "Components", "Geometry"], cols):
        axs[1].bar(build["case"], build[c], bottom=bottom, label=c, color=color)
        bottom += build[c].to_numpy()
    axs[1].set_ylabel("Construction time (ms)")
    axs[1].tick_params(axis="x", rotation=25)
    axs[1].legend(ncol=2, frameon=False, loc="upper right")
    axs[1].text(0.02, 0.95, "site-to-cell build: 9.83--12.47 ms", transform=axs[1].transAxes, va="top")
    label(axs[1], "(b)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Ownership audit and production construction timing", y=1.03, fontweight="bold")
    save(fig, "Figure_04_ownership_speed_audit")


def fig_geodesic_face_operator() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.15), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    clean(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.add_patch(patches.Rectangle((0.5, 0.5), 9.0, 5.0, fc="#eef5f7", ec=COL["grid"], lw=1.0))
    blocks = [(4.2, 0.5, 1.1, 2.2), (4.2, 3.3, 1.1, 2.2), (6.2, 1.7, 1.1, 2.5)]
    for b in blocks:
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], fc=COL["solid"], ec=COL["solid"]))
    ax.plot(2.0, 3.0, "*", ms=12, color=COL["blue"], mec="white", mew=0.8)
    ax.plot(8.4, 3.0, "*", ms=12, color=COL["orange"], mec="white", mew=0.8)
    ax.plot([2.0, 8.4], [3.0, 3.0], "--", color=COL["red"], lw=1.4, label="Euclidean chord")
    path = np.array([[2.0, 3.0], [3.4, 3.0], [3.4, 1.25], [5.75, 1.25], [5.75, 4.75], [8.4, 4.75], [8.4, 3.0]])
    ax.plot(path[:, 0], path[:, 1], "-", color=COL["teal"], lw=2.2, label="graph-geodesic site-to-face path")
    ax.plot([5.3, 5.3], [2.75, 3.25], color=COL["ink"], lw=2.8)
    ax.text(5.45, 3.08, "facelet", va="center")
    ax.text(1.5, 2.45, "owner site", color=COL["blue"])
    ax.text(7.6, 2.45, "neighbour site", color=COL["orange"])
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Obstructed exchange length")
    label(ax, "(a)")

    audit = pd.DataFrame(
        {
            "metric": ["Euclidean\nface metric", "Geodesic\nface metric"],
            "eK": [27.19, 2.95],
            "K": [0.7363, 0.9815],
        }
    )
    axs[1].bar(audit["metric"], audit["eK"], color=[COL["red"], COL["teal"]])
    axs[1].set_ylabel("Permeability error (%)")
    axs[1].set_ylim(0, 36)
    for i, row in audit.iterrows():
        axs[1].text(i, row.eK + 1.0, f"K={row.K:.4f}", ha="center")
    axs[1].text(
        0.98,
        0.97,
        "same maze mask\nsame sites\nsame face graph",
        transform=axs[1].transAxes,
        ha="right",
        va="top",
        color=COL["muted"],
    )
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].grid(axis="y", color=COL["grid"], lw=0.6)
    axs[1].set_title("Metric audit")
    label(axs[1], "(b)")
    fig.suptitle("Graph-geodesic face metric for cell--cell exchange", y=1.03, fontweight="bold")
    save(fig, "Figure_05_geodesic_face_operator")
    save(fig_geodesic_face_operator_si(), "Figure_S04_intercell_metric_audit")


def fig_geodesic_face_operator_si() -> plt.Figure:
    audit = pd.DataFrame(
        {
            "metric": ["Euclidean face metric", "Geodesic face metric"],
            "Kx": [0.7363, 0.9815],
            "eK": [27.19, 2.95],
            "ratio": [1.000, 1.333],
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.75))
    colors = [COL["red"], COL["teal"]]
    axs[0].bar(audit["metric"], audit["Kx"], color=colors)
    axs[0].axhline(1.01131865, color=COL["ink"], lw=1.1, ls="--", label="voxel reference")
    axs[0].set_ylabel("$K_x$")
    axs[0].tick_params(axis="x", rotation=18)
    axs[0].legend(frameon=False)
    label(axs[0], "(a)")
    axs[1].bar(audit["metric"], audit["eK"], color=colors)
    axs[1].set_ylabel("$e_K$ (%)")
    axs[1].tick_params(axis="x", rotation=18)
    label(axs[1], "(b)")
    axs[2].bar(audit["metric"], audit["ratio"], color=colors)
    axs[2].set_ylabel("$K_x/K_x^{Euc}$")
    axs[2].tick_params(axis="x", rotation=18)
    label(axs[2], "(c)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Intercell metric audit on the maze stress case", y=1.04, fontweight="bold")
    return fig


def fig_wall_facelet_closure() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.3, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    clean(ax)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    cell = patches.Polygon([[1.1, 1.0], [4.3, 0.8], [5.2, 2.2], [4.6, 4.1], [1.4, 3.8]], closed=True, fc="#dceef2", ec=COL["blue"], lw=1.4)
    solid = patches.Rectangle((5.15, 0.45), 1.7, 4.2, fc=COL["solid"], ec=COL["solid"])
    ax.add_patch(cell)
    ax.add_patch(solid)
    facelets = [((4.8, 1.25), (5.25, 1.45)), ((5.0, 2.25), (5.35, 2.55)), ((4.75, 3.35), (5.25, 3.55))]
    for (x1, y1), (x2, y2) in facelets:
        ax.plot([x1, x2], [y1, y2], color=COL["orange"], lw=3)
        ax.arrow((x1 + x2) / 2, (y1 + y2) / 2, -0.55, 0.0, head_width=0.08, head_length=0.12, color=COL["orange"], length_includes_head=True)
    ax.plot(2.7, 2.45, "o", color=COL["blue"], ms=5)
    ax.text(2.85, 2.55, "cell i", color=COL["blue"])
    ax.text(5.55, 4.2, "solid", color="white")
    ax.set_title("Wall facelets on assembled cell")
    label(ax, "(a)")

    ax = axs[1]
    clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.05, 0.72, 0.9, 0.18, "wall facelet areas $A_g^w$"),
        (0.05, 0.48, 0.9, 0.18, "effective normal distances $\\delta_{i,g}^w$"),
        (0.05, 0.24, 0.9, 0.18, "scale transfer $(\\ell_c/\\ell_*)^{\\alpha_w}$"),
        (0.05, 0.02, 0.9, 0.16, "$T_i^{wall}=\\beta_w(\\ell_c/\\ell_*)^{\\alpha_w}\\sum_g A_g^w/\\delta_{i,g}^w$"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="#f8fafc", ec=COL["grid"], lw=1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center")
    for y0, y1 in [(0.72, 0.66), (0.48, 0.42), (0.24, 0.18)]:
        ax.annotate("", (0.5, y1), (0.5, y0), arrowprops=dict(arrowstyle="-|>", color=COL["muted"], lw=0.9))
    ax.set_title("Area-over-distance wall contribution")
    label(ax, "(b)")
    fig.suptitle("Wall facelet closure for unresolved no-slip resistance", y=1.04, fontweight="bold")
    save(fig, "Figure_06_wall_facelet_closure")


def fig_state_projection() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.8))
    titles = ["Sampled states", "Face interpolation", "Conservative correction"]
    for ax, title in zip(axs, titles):
        clean(ax)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title(title)
    cells = [
        patches.Rectangle((0.5, 0.6), 1.5, 2.8, fc="#dceef2", ec=COL["blue"]),
        patches.Rectangle((2.1, 0.6), 1.7, 2.8, fc="#e7def2", ec=COL["purple"]),
        patches.Rectangle((3.9, 0.6), 1.6, 2.8, fc="#f4e1d0", ec=COL["orange"]),
    ]
    for ax in axs:
        for c in cells:
            ax.add_patch(patches.Rectangle(c.get_xy(), c.get_width(), c.get_height(), fc=c.get_facecolor(), ec=c.get_edgecolor(), alpha=0.9))
    pts = [(1.0, 1.2), (1.3, 2.7), (2.7, 1.5), (3.2, 2.5), (4.7, 1.6), (5.1, 2.9)]
    for x, y in pts:
        axs[0].plot(x, y, "o", color=COL["ink"], ms=3.8)
        axs[0].arrow(x, y, 0.28, 0.15, head_width=0.07, color=COL["ink"], length_includes_head=True)
    axs[0].text(0.55, 3.62, "$S_m$ fixed", color=COL["ink"])
    for x in [2.05, 3.85]:
        axs[1].plot([x, x], [0.6, 3.4], color=COL["red"], lw=2)
        axs[1].arrow(x - 0.45, 2.0, 0.35, 0.0, head_width=0.08, color=COL["teal"], length_includes_head=True)
        axs[1].arrow(x + 0.45, 2.0, -0.35, 0.0, head_width=0.08, color=COL["orange"], length_includes_head=True)
    axs[1].text(2.55, 3.62, "site-to-face weights", ha="center")
    for x, y, dx in [(1.2, 2.0, 0.8), (3.0, 2.2, 0.75), (4.6, 1.8, 0.65)]:
        axs[2].arrow(x, y, dx, 0, head_width=0.12, color=COL["teal"], length_includes_head=True)
    for x in [2.05, 3.85]:
        axs[2].plot([x, x], [0.6, 3.4], color=COL["red"], lw=2.1)
    axs[2].text(1.0, 3.62, "$\\nabla\\cdot q=0$ after projection")
    for t, ax in zip(["(a)", "(b)", "(c)"], axs):
        label(ax, t)
    fig.suptitle("State samples are projected onto a conservative face-flux field", y=1.04, fontweight="bold")
    save(fig, "Figure_07_state_projection")


def porous_texture(seed: int, shape=(28, 40), phi=0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=shape)
    for _ in range(6):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    return z > np.quantile(z, 1 - phi)


def draw_case_thumb(ax, name: str) -> None:
    clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if "Orthogonal" in name:
        ax.add_patch(patches.Rectangle((0.08, 0.2), 0.84, 0.6, fc="#dceef2", ec=COL["blue"], lw=1))
    elif "Skewed" in name:
        ax.add_patch(patches.Polygon([[0.15, 0.2], [0.92, 0.2], [0.75, 0.8], [0.0, 0.8]], fc="#dceef2", ec=COL["blue"], lw=1))
    elif "Thin-wall" in name:
        ax.add_patch(patches.Rectangle((0.06, 0.17), 0.88, 0.66, fc="#dceef2", ec=COL["blue"], lw=1))
        ax.add_patch(patches.Rectangle((0.47, 0.17), 0.055, 0.66, fc=COL["solid"]))
    elif "Narrow" in name:
        ax.add_patch(patches.Rectangle((0.06, 0.2), 0.36, 0.6, fc="#dceef2", ec=COL["blue"], lw=1))
        ax.add_patch(patches.Rectangle((0.58, 0.2), 0.36, 0.6, fc="#dceef2", ec=COL["blue"], lw=1))
        ax.add_patch(patches.Rectangle((0.42, 0.43), 0.16, 0.14, fc="#dceef2", ec=COL["blue"], lw=1))
    elif "Maze" in name:
        ax.add_patch(patches.Rectangle((0.05, 0.15), 0.9, 0.7, fc="#dceef2", ec=COL["blue"], lw=1))
        for x, y, w, h in [(0.2, 0.15, 0.08, 0.48), (0.42, 0.37, 0.08, 0.48), (0.64, 0.15, 0.08, 0.48)]:
            ax.add_patch(patches.Rectangle((x, y), w, h, fc=COL["solid"]))
    else:
        seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
        tex = porous_texture(seed, phi=0.42 if "Berea" in name else 0.32)
        ax.imshow(tex, cmap=ListedColormap([COL["solid"], "#dceef2"]), origin="lower", extent=[0.05, 0.95, 0.15, 0.85])
    ax.text(0.5, 0.04, name, ha="center", va="bottom", fontsize=7.3)


def fig_case_atlas() -> None:
    cases = [
        ("Bentheimer\nstate transfer", "mainline"),
        ("Public Berea\nstate transfer", "mainline"),
        ("Orthogonal\nduct", "mechanism"),
        ("Skewed\nduct", "mechanism"),
        ("Thin-wall\nsynthetic", "mechanism"),
        ("Narrow-throat\nsynthetic", "mechanism"),
        ("Maze\nsynthetic", "stress"),
        ("Bentheimer\ncrop", "segmented"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(9.4, 4.4))
    for ax, (name, role) in zip(axs.ravel(), cases):
        draw_case_thumb(ax, name)
        ax.text(0.05, 0.93, role, ha="left", va="top", fontsize=7, color=COL["ink"], bbox=dict(fc="white", ec=COL["grid"], pad=1.5))
    fig.suptitle("Validation case families and evidence roles", y=1.02, fontweight="bold")
    save(fig, "Figure_08_case_atlas")


def fig_experimental_protocol() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.03, 0.60, 0.18, 0.22, "Binary mask\nand reference field"),
        (0.26, 0.60, 0.18, 0.22, "Sampled states\n$S_m$"),
        (0.49, 0.60, 0.18, 0.22, "Auxiliary mask sites\n$S_a$"),
        (0.72, 0.60, 0.22, 0.22, "GeoVoronoi-FV\nsite-to-cell operator"),
        (0.72, 0.18, 0.22, 0.22, "Same-mask metrics\n$e_K$, $e_\\phi$, $e_u$, time"),
        (0.49, 0.18, 0.18, 0.22, "Conservative\nprojection"),
        (0.26, 0.18, 0.18, 0.22, "Face-local\nstate map"),
        (0.03, 0.18, 0.18, 0.22, "Locked protocol\nand audit record"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018", fc="#f8fafc", ec=COL["grid"], lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center")
    arrows = [((0.21, 0.71), (0.26, 0.71)), ((0.44, 0.71), (0.49, 0.71)), ((0.67, 0.71), (0.72, 0.71)), ((0.83, 0.60), (0.83, 0.40)), ((0.72, 0.29), (0.67, 0.29)), ((0.49, 0.29), (0.44, 0.29)), ((0.26, 0.29), (0.21, 0.29))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.1, color=COL["muted"]))
    ax.text(0.055, 0.91, "Bentheimer: particle-track states from DNS velocity field", color=COL["blue"], fontweight="bold")
    ax.text(0.055, 0.86, "Berea: particles advected in voxel-reference field", color=COL["orange"], fontweight="bold")
    ax.text(0.49, 0.08, "Only declared state density changes in the Berea support ladder; mask, auxiliary rule and operator remain fixed.", ha="center", color=COL["muted"])
    fig.suptitle("Experimental data generation and comparison protocol", y=1.02, fontweight="bold")
    save(fig, "Figure_10_experimental_protocol")


def fig_support_density_response() -> None:
    df = pd.read_csv(DATA / "protocol_refinement_berea_rerun_rows.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.0))
    x = df["support"]
    for col, color, lab in [
        ("e_K_percent", COL["blue"], "$e_K$"),
        ("e_phi_percent", COL["orange"], "$e_\\phi$"),
        ("e_u_percent", COL["teal"], "$e_u$"),
    ]:
        axs[0].plot(x, df[col], "o-", lw=1.5, color=color, label=lab)
    axs[0].axvline(1000, color=COL["ink"], ls="--", lw=1)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Support windows")
    axs[0].set_ylabel("Error (%)")
    axs[0].legend(frameon=False)
    axs[0].set_title("Accuracy response")
    label(axs[0], "(a)")
    axs[1].plot(x, df["C_comp"], "o-", color=COL["purple"], lw=1.5)
    axs[1].axvline(1000, color=COL["ink"], ls="--", lw=1)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Support windows")
    axs[1].set_ylabel("Mean voxel-grid cells per GVF cell")
    axs[1].set_title("Controlled-space ratio")
    label(axs[1], "(b)")
    axs[2].bar(df["support"].astype(str), df["t_total_s"], color=COL["green"])
    axs[2].set_xlabel("Support windows")
    axs[2].set_ylabel("Projection task time (s)")
    axs[2].tick_params(axis="x", rotation=30)
    retained = df[df["support"] == 1000].iloc[0]
    axs[2].annotate(
        f"retained\n{retained.task_speedup:.0f}x",
        xy=(1, retained.t_total_s),
        xytext=(1.55, 2.55),
        textcoords="data",
        ha="center",
        va="top",
        color=COL["muted"],
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COL["muted"]),
    )
    axs[2].set_title("Task cost")
    label(axs[2], "(c)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Berea support-density response under the locked operator", y=1.04, fontweight="bold")
    save(fig, "Figure_12_support_density_response", png=True)
    save(fig_support_density_si(df), "Figure_S03_berea_support_ladder")


def fig_support_density_si(df: pd.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(8.8, 5.2))
    axs = axs.ravel()
    x = df["support"]
    for col, color, lab in [("e_K_percent", COL["blue"], "$e_K$"), ("e_phi_percent", COL["orange"], "$e_\\phi$"), ("e_u_percent", COL["teal"], "$e_u$")]:
        axs[0].plot(x, df[col], "o-", color=color, label=lab)
    axs[0].set_xscale("log"); axs[0].set_ylabel("Error (%)"); axs[0].legend(frameon=False); label(axs[0], "(a)")
    axs[1].plot(x, df["coverage_percent"], "o-", color=COL["green"]); axs[1].set_xscale("log"); axs[1].set_ylabel("State coverage (%)"); label(axs[1], "(b)")
    axs[2].plot(x, df["N_c"], "o-", color=COL["purple"]); axs[2].set_xscale("log"); axs[2].set_ylabel("$N_c$"); axs[2].set_xlabel("Support windows"); label(axs[2], "(c)")
    axs[3].plot(x, df["t_total_s"], "o-", color=COL["gold"]); axs[3].set_xscale("log"); axs[3].set_ylabel("Task time (s)"); axs[3].set_xlabel("Support windows"); label(axs[3], "(d)")
    for ax in axs:
        ax.axvline(1000, color=COL["ink"], ls="--", lw=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Public Berea sampled-state support ladder", y=1.02, fontweight="bold")
    return fig


def fig_mainline_transfer() -> None:
    df = pd.read_csv(DATA / "sampled_state_mainline_transfer_rows.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.1), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    names = ["Bentheimer", "Berea"]
    colors = [COL["blue"], COL["orange"]]
    axs[0].bar(names, df["e_K_percent"], color=colors)
    axs[0].set_ylabel("$e_K$ (%)")
    axs[0].set_title("Permeability recovery")
    label(axs[0], "(a)")
    w = 0.34
    xpos = np.arange(2)
    axs[1].bar(xpos - w / 2, df["e_phi_percent"], width=w, color=COL["purple"], label="$e_\\phi$")
    axs[1].bar(xpos + w / 2, df["e_u_percent"], width=w, color=COL["teal"], label="$e_u$")
    axs[1].set_xticks(xpos, names)
    axs[1].set_ylabel("Field error (%)")
    axs[1].legend(frameon=False)
    axs[1].set_title("State-to-flux field accuracy")
    label(axs[1], "(b)")
    clean(axs[2])
    axs[2].set_xlim(0, 1); axs[2].set_ylim(0, 1)
    for i, row in df.iterrows():
        y = 0.73 - i * 0.43
        axs[2].add_patch(patches.FancyBboxPatch((0.05, y - 0.16), 0.9, 0.30, boxstyle="round,pad=0.02", fc="#f8fafc", ec=COL["grid"]))
        spd = "--" if pd.isna(row["speedup"]) else f"{row['speedup']:.0f}x"
        axs[2].text(0.10, y + 0.08, names[i], fontweight="bold", color=colors[i])
        axs[2].text(0.10, y - 0.01, f"S_m={int(row.S_m):,}; S_a={int(row.S_a):,}; C={row.C_comp:.2f}")
        axs[2].text(0.10, y - 0.10, f"coverage={row.coverage_percent:.2f}%; task speedup={spd}")
    axs[2].set_title("Declared transfer rows")
    label(axs[2], "(c)")
    for ax in axs[:2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Real-mask sampled-state transfer mainline", y=1.04, fontweight="bold")
    save(fig, "Figure_13_mainline_transfer")


def fig_mechanism_pareto() -> None:
    df = pd.read_csv(DATA / "main_solver_table_for_table1.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    colors = [COL["blue"], COL["teal"], COL["orange"], COL["red"], COL["purple"], COL["green"]]
    axs[0].scatter(df["compression_C"], df["e_K_percent"], s=np.clip(df["speedup_vs_ref"].fillna(50) / 2, 35, 180), c=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
    for _, r in df.iterrows():
        axs[0].annotate(str(r["paper_case"]).split()[0], (r["compression_C"], r["e_K_percent"]), xytext=(3, 3), textcoords="offset points", fontsize=6.8)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("$C_{comp}$")
    axs[0].set_ylabel("$e_K$ (%)")
    axs[0].set_title("Accuracy--compression")
    label(axs[0], "(a)")
    y = np.arange(len(df))
    axs[1].barh(y - 0.18, df["e_phi_percent"], height=0.36, color=COL["purple"], label="$e_\\phi$")
    axs[1].barh(y + 0.18, df["e_u_percent"], height=0.36, color=COL["teal"], label="$e_u$")
    axs[1].set_yticks(y, [str(x).replace(" synthetic", "").replace(" segmented sandstone crop", " crop") for x in df["paper_case"]])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Field error (%)")
    axs[1].legend(frameon=False)
    axs[1].set_title("Field response")
    label(axs[1], "(b)")
    axs[2].barh(y, df["speedup_vs_ref"].fillna(0), color=COL["green"])
    axs[2].set_yticks(y, [])
    axs[2].set_xlabel("End-to-end speedup")
    axs[2].set_title("Flow solve cost")
    label(axs[2], "(c)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is not axs[0] else "y", color=COL["grid"], lw=0.6)
    fig.suptitle("Mechanism-row accuracy and cost under the final operator", y=1.04, fontweight="bold")
    save(fig, "Figure_11_synthetic_pareto", png=True)


def fig_final_summary() -> None:
    main = pd.read_csv(DATA / "sampled_state_mainline_transfer_rows.csv")
    mech = pd.read_csv(DATA / "main_solver_table_for_table1.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.05))
    names = ["Bentheimer", "Berea"]
    axs[0].bar(names, main["e_K_percent"], color=[COL["blue"], COL["orange"]])
    axs[0].set_ylabel("$e_K$ (%)")
    axs[0].set_title("Mainline permeability")
    label(axs[0], "(a)")
    vals = [9.1, 11.3, float(main.loc[main["case"].str.contains("Berea"), "speedup"].iloc[0])]
    labs = ["ownership\nlower", "ownership\nupper", "Berea task"]
    axs[1].bar(labs, vals, color=[COL["teal"], COL["teal"], COL["green"]])
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Speedup")
    axs[1].set_title("Acceleration scales")
    label(axs[1], "(b)")
    axs[2].boxplot([mech["e_K_percent"], mech["e_phi_percent"], mech["e_u_percent"]], tick_labels=["$e_K$", "$e_\\phi$", "$e_u$"], patch_artist=True, boxprops=dict(facecolor="#f8fafc", color=COL["ink"]), medianprops=dict(color=COL["red"]))
    axs[2].set_ylabel("Mechanism-row error (%)")
    axs[2].set_title("Mechanism envelope")
    label(axs[2], "(c)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("GeoVoronoi-FV evidence summary", y=1.04, fontweight="bold")
    save(fig, "Figure_15_final_evidence_summary")


def fig_si_construction_scaling() -> None:
    df = pd.read_csv(DATA / "current_large_scale_roi_jfa_table_for_tex.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for case, g in df.groupby("paper_case"):
        axs[0].plot(g["N_fl"], g["label_plus_split_ms"], "o-", label=case)
        axs[1].plot(g["N_fl"], g["roi_fraction_of_fluid_percent"], "o-", label=case)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("$N_{fl}$"); axs[0].set_ylabel("Label + split time (ms)"); label(axs[0], "(a)")
    axs[1].set_xscale("log"); axs[1].set_xlabel("$N_{fl}$"); axs[1].set_ylabel("Active ROI fraction (%)"); label(axs[1], "(b)")
    axs[1].legend(frameon=False, loc="best")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(True, color=COL["grid"], lw=0.6)
    fig.suptitle("Large-scale ROI-JFA construction scaling records", y=1.04, fontweight="bold")
    save(fig, "Figure_S05_construction_scaling")


def fig_si_agglomeration_relocation() -> None:
    agg = pd.read_csv(DATA / "same_mask_agglomeration_baseline.csv")
    rel = pd.read_csv(DATA / "state_location_preservation_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
    piv = agg.pivot_table(index="paper_case", columns="method", values="e_K_percent", aggfunc="first")
    piv = piv.loc[[x for x in piv.index if x in ["Thin-wall synthetic", "Narrow-throat synthetic", "Bentheimer segmented sandstone crop", "Fibrous filter proxy"]]]
    x = np.arange(len(piv))
    axs[0].bar(x - 0.18, piv.get("GeoVoronoi-FV"), width=0.36, color=COL["teal"], label="GeoVoronoi-FV")
    axs[0].bar(x + 0.18, piv.get("Block agglomeration"), width=0.36, color=COL["orange"], label="Block agglomeration")
    axs[0].set_yscale("log")
    axs[0].set_xticks(x, [s.replace(" synthetic", "").replace(" segmented sandstone crop", " crop").replace(" filter proxy", "") for s in piv.index], rotation=25, ha="right")
    axs[0].set_ylabel("$e_K$ (%)")
    axs[0].legend(frameon=False)
    label(axs[0], "(a)")
    axs[1].bar(rel["paper_case"].str.replace(" synthetic", "").str.replace(" segmented sandstone crop", " crop").str.replace(" filter proxy", ""), rel["block_relocation_p95_vox"], color=COL["purple"])
    axs[1].set_ylabel("Block relocation p95 (voxels)")
    axs[1].tick_params(axis="x", rotation=25)
    label(axs[1], "(b)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Same-mask agglomeration and state-location positioning", y=1.04, fontweight="bold")
    save(fig, "Figure_S06_agglomeration_relocation")


def fig_si_fibrous_wall() -> None:
    df = pd.read_csv(DATA / "fibrous_wall_beta_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(df["beta"], df["e_K_percent"], "o-", color=COL["orange"])
    ax.set_xlabel("$\\beta_w$")
    ax.set_ylabel("$e_K$ (%)")
    ax.set_title("Fibrous proxy wall-beta sensitivity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    save(fig, "Figure_S07_fibrous_wall_sensitivity")


def fig_si_wall_closure_variants() -> None:
    df = pd.read_csv(DATA / "hydraulic_closure_variant_decision_table.csv")
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    order = ["patch_throat_size_factor", "component_merged_throat_size_factor", "patch_throat_size_factor_periodic_lift", "component_merged_throat_size_factor_periodic_lift"]
    data = [df[df["closure"] == c]["e_K_percent"].to_numpy() for c in order]
    bp = ax.boxplot(data, tick_labels=["patch", "component", "patch lift", "component lift"], patch_artist=True)
    for patch, color in zip(bp["boxes"], [COL["blue"], COL["orange"], COL["purple"], COL["red"]]):
        patch.set_facecolor(color); patch.set_alpha(0.25)
    ax.set_yscale("log")
    ax.set_ylabel("$e_K$ (%)")
    ax.set_title("Wall/periodic closure variant audit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    save(fig, "Figure_S08_wall_closure_variants")


def fig_si_state_density() -> None:
    df = pd.read_csv(DATA / "sampled_state_mainline_transfer_rows.csv")
    diag = pd.read_csv(DATA / "current_bentheimer_density_table_for_tex.csv")
    bent = pd.read_csv(DATA / "bentheimer_final_no_reference_selected_row.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.1))
    labels = ["(4,8,8)", "(2,6,6)", "(2,4,4)", "selected"]
    ephi = list(diag["e_phi_percent"]) + [float(bent["e_phi_percent"].iloc[0])]
    eu = list(diag["e_u_percent"]) + [float(bent["e_u_percent"].iloc[0])]
    x = np.arange(len(labels))
    axs[0].bar(x - 0.17, ephi, width=0.34, color=COL["purple"], label="$e_\\phi$")
    axs[0].bar(x + 0.17, eu, width=0.34, color=COL["teal"], label="$e_u$")
    axs[0].set_xticks(x, labels, rotation=25)
    axs[0].set_ylabel("Field error (%)")
    axs[0].legend(frameon=False)
    label(axs[0], "(a)")
    axs[1].bar(["Bentheimer", "Berea"], df["coverage_percent"], color=[COL["blue"], COL["orange"]])
    axs[1].set_ylim(94, 100)
    axs[1].set_ylabel("State coverage (%)")
    label(axs[1], "(b)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=COL["grid"], lw=0.6)
    fig.suptitle("Sampled-state density and coverage diagnostics", y=1.04, fontweight="bold")
    save(fig, "Figure_S09_state_density_diagnostic")


def fig_si_reference_convergence() -> None:
    df = pd.read_csv(DATA / "reference_convergence_table_for_tex.csv")
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    ax.bar(df["case_label"], df["relative_K_ref_change_percent"], color=COL["blue"])
    ax.set_ylabel("Reference drift (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Same-mask reference convergence audit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    save(fig, "Figure_S10_reference_convergence")


def main() -> None:
    fig_site_to_cell_complex()
    fig_ownership_speed_audit()
    fig_geodesic_face_operator()
    fig_wall_facelet_closure()
    fig_state_projection()
    fig_case_atlas()
    fig_experimental_protocol()
    fig_support_density_response()
    fig_mainline_transfer()
    fig_mechanism_pareto()
    fig_final_summary()
    fig_si_construction_scaling()
    fig_si_agglomeration_relocation()
    fig_si_fibrous_wall()
    fig_si_wall_closure_variants()
    fig_si_state_density()
    fig_si_reference_convergence()


if __name__ == "__main__":
    main()

