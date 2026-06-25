from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "cmame_artifacts" / "figure_source_data"
FIG_DIR = ROOT / "cmame_artifacts" / "figures"

SOURCE_STEM = "figure4_real_berea_3d_geodesic_face_operator_irregular_same_slice"
SOURCE_NPZ = DATA_DIR / f"{SOURCE_STEM}.npz"
SOURCE_JSON = DATA_DIR / f"{SOURCE_STEM}.json"

OUT_STEM = "Figure_05_operator_geometry_unified"
OUT_DATA = DATA_DIR / "figure5_operator_geometry_unified.json"


PALETTE = {
    "solid": "#1f2a30",
    "pore": "#f3efe4",
    "omega_i": "#7fa3bd",
    "omega_i_dark": "#526c88",
    "omega_j": "#dfc17d",
    "omega_j_dark": "#8b5918",
    "gamma": "#c025d3",
    "wall": "#00a887",
    "wall_pick": "#e68a1f",
    "path": "#007c77",
    "chord": "#c56d08",
    "site": "#b65382",
    "ink": "#222222",
    "axis": "#c9d2d5",
    "grid": "#ffffff",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "mathtext.default": "it",
    }
)


def load_source() -> dict[str, np.ndarray]:
    if not SOURCE_NPZ.exists():
        raise FileNotFoundError(f"Missing figure source data: {SOURCE_NPZ}")
    with np.load(SOURCE_NPZ) as data:
        return {key: data[key] for key in data.files}


def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def slice_image(mask: np.ndarray, labels: np.ndarray, z: int) -> np.ndarray:
    h, w = mask.shape[1:]
    img = np.zeros((h, w, 3), dtype=float)
    img[:] = hex_to_rgb(PALETTE["solid"])
    pore = mask[z].astype(bool)
    img[pore] = hex_to_rgb(PALETTE["pore"])
    img[(labels[z] == 0) & pore] = hex_to_rgb(PALETTE["omega_i"])
    img[(labels[z] == 1) & pore] = hex_to_rgb(PALETTE["omega_j"])
    return img


def path_xy(path_zyx: np.ndarray) -> np.ndarray:
    return np.column_stack([path_zyx[:, 2] + 0.5, path_zyx[:, 1] + 0.5])


def add_grid(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float], lw: float = 0.34, alpha: float = 0.28) -> None:
    xmin, xmax = sorted(xlim)
    ymin, ymax = sorted(ylim)
    xs = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
    ys = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    ax.vlines(xs, ymin, ymax, color=PALETTE["grid"], lw=lw, alpha=alpha, zorder=5)
    ax.hlines(ys, xmin, xmax, color=PALETTE["grid"], lw=lw, alpha=alpha, zorder=5)


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.035, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, weight="bold", color="#000000")
    ax.text(0.115, 1.035, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=PALETTE["ink"])


def add_panel_label_3d(ax: plt.Axes, label: str, title: str) -> None:
    ax.text2D(0.0, 1.035, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, weight="bold", color="#000000")
    ax.text2D(0.115, 1.035, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=PALETTE["ink"])


def label_text(ax: plt.Axes, x: float, y: float, text: str, color: str = "#26333b", size: float = 7.4, ha: str = "center") -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        zorder=60,
        path_effects=[pe.withStroke(linewidth=2.7, foreground="white", alpha=0.96)],
    )


def draw_base(ax: plt.Axes, img: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.imshow(img, origin="upper", extent=(0, img.shape[1], img.shape[0], 0), interpolation="nearest", zorder=0)
    add_grid(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d4dde0")
        spine.set_linewidth(0.8)


def add_site(ax: plt.Axes, site: np.ndarray, label: str, dx: float, dy: float) -> None:
    x, y = site[2] + 0.5, site[1] + 0.5
    ax.add_patch(Circle((x, y), 0.70, facecolor=PALETTE["site"], edgecolor="white", lw=1.2, zorder=35))
    label_text(ax, x + dx, y + dy, label, color=PALETTE["site"], size=8.0)


def wall_segments(mask: np.ndarray, labels: np.ndarray, z: int, owner: int = 0) -> list[dict[str, np.ndarray]]:
    h, w = mask.shape[1:]
    out: list[dict[str, np.ndarray]] = []
    steps = [
        (0, 1, np.array([1.0, 0.0]), "right"),
        (0, -1, np.array([-1.0, 0.0]), "left"),
        (1, 0, np.array([0.0, 1.0]), "down"),
        (-1, 0, np.array([0.0, -1.0]), "up"),
    ]
    for y in range(h):
        for x in range(w):
            if labels[z, y, x] != owner:
                continue
            for dy, dx, normal, side in steps:
                yy, xx = y + dy, x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w or mask[z, yy, xx]:
                    continue
                if side == "right":
                    seg = np.array([[x + 1, y], [x + 1, y + 1]], dtype=float)
                    centre = np.array([x + 1, y + 0.5], dtype=float)
                elif side == "left":
                    seg = np.array([[x, y], [x, y + 1]], dtype=float)
                    centre = np.array([x, y + 0.5], dtype=float)
                elif side == "down":
                    seg = np.array([[x, y + 1], [x + 1, y + 1]], dtype=float)
                    centre = np.array([x + 0.5, y + 1], dtype=float)
                else:
                    seg = np.array([[x, y], [x + 1, y]], dtype=float)
                    centre = np.array([x + 0.5, y], dtype=float)
                out.append(
                    {
                        "seg": seg,
                        "centre": centre,
                        "cell_centre": np.array([x + 0.5, y + 0.5], dtype=float),
                        "normal": normal,
                        "voxel_yx": np.array([y, x], dtype=int),
                        "solid_yx": np.array([yy, xx], dtype=int),
                    }
                )
    return out


def select_wall_segment(segments: list[dict[str, np.ndarray]], site_i: np.ndarray, face_center: np.ndarray, mask: np.ndarray, labels: np.ndarray, z: int) -> dict[str, np.ndarray]:
    site_xy = np.array([site_i[2] + 0.5, site_i[1] + 0.5], dtype=float)
    face_xy = np.array([face_center[2], face_center[1]], dtype=float)
    best: dict[str, np.ndarray] | None = None
    best_score = -1.0e9
    for item in segments:
        centre = np.asarray(item["centre"], dtype=float)
        normal = np.asarray(item["normal"], dtype=float)
        wall_delta = float(np.dot(centre - site_xy, normal))
        if wall_delta < 1.8:
            continue
        x, y = centre
        edge_margin = min(x, y, mask.shape[2] - x, mask.shape[1] - y)
        x0, x1 = max(0, int(np.floor(x - 8))), min(mask.shape[2], int(np.ceil(x + 11)))
        y0, y1 = max(0, int(np.floor(y - 7))), min(mask.shape[1], int(np.ceil(y + 8)))
        local_labels = labels[z, y0:y1, x0:x1]
        local_j = int(np.sum(local_labels == 1))
        local_i = int(np.sum(local_labels == 0))
        score = (
            0.65 * edge_margin
            - 0.10 * np.linalg.norm(centre - site_xy)
            + 0.05 * min(np.linalg.norm(centre - face_xy), 24.0)
            - 0.030 * local_j
            + 0.010 * local_i
            - 0.45 * abs(y - site_xy[1])
            - 0.12 * abs(wall_delta - 3.8)
        )
        if score > best_score:
            best = item
            best_score = score
    if best is None:
        raise RuntimeError("No wall facelet found for selected ownership cell")
    return best


def draw_paths(ax: plt.Axes, site_i: np.ndarray, site_j: np.ndarray, path_i: np.ndarray, path_j: np.ndarray, face_center: np.ndarray, show_chord: bool) -> None:
    fc_xy = np.array([face_center[2], face_center[1]], dtype=float)
    for p in (path_xy(path_i), path_xy(path_j)):
        ax.plot(
            p[:, 0],
            p[:, 1],
            color=PALETTE["path"],
            lw=2.05,
            solid_capstyle="round",
            zorder=30,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white", alpha=0.95)],
        )
    if show_chord:
        for site in (site_i, site_j):
            sx, sy = site[2] + 0.5, site[1] + 0.5
            ax.plot(
                [sx, fc_xy[0]],
                [sy, fc_xy[1]],
                color=PALETTE["chord"],
                lw=1.35,
                ls=(0, (3.2, 2.6)),
                zorder=24,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white", alpha=0.75)],
            )


def component_from_seed(labels: np.ndarray, owner: int, seed_zyx: np.ndarray) -> np.ndarray:
    seed = tuple(int(v) for v in seed_zyx)
    if labels[seed] != owner:
        raise ValueError(f"Seed {seed} is not in label {owner}")
    comp = np.zeros(labels.shape, dtype=bool)
    q: deque[tuple[int, int, int]] = deque([seed])
    comp[seed] = True
    steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while q:
        z, y, x = q.popleft()
        for dz, dy, dx in steps:
            zz, yy, xx = z + dz, y + dy, x + dx
            if zz < 0 or yy < 0 or xx < 0 or zz >= labels.shape[0] or yy >= labels.shape[1] or xx >= labels.shape[2]:
                continue
            if comp[zz, yy, xx] or labels[zz, yy, xx] != owner:
                continue
            comp[zz, yy, xx] = True
            q.append((zz, yy, xx))
    return comp


def crop_from_geometry(points: list[np.ndarray], shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    stack = np.vstack([np.asarray(p).reshape(-1, 3) for p in points])
    lo = np.floor(stack.min(axis=0)).astype(int) - np.array([5, 6, 6])
    hi = np.ceil(stack.max(axis=0)).astype(int) + np.array([6, 7, 7])
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape) - 1)
    return (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1))


def cube_face_vertices(z: int, y: int, x: int, axis: int, side: int) -> list[tuple[float, float, float]]:
    x0, x1 = x, x + 1
    y0, y1 = y, y + 1
    z0, z1 = z, z + 1
    if axis == 2 and side > 0:
        return [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
    if axis == 2 and side < 0:
        return [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]
    if axis == 1 and side > 0:
        return [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]
    if axis == 1 and side < 0:
        return [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
    if axis == 0 and side > 0:
        return [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]


def exposed_faces(mask: np.ndarray, crop: tuple[slice, slice, slice]) -> list[list[tuple[float, float, float]]]:
    zsl, ysl, xsl = crop
    faces: list[list[tuple[float, float, float]]] = []
    steps = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    zzs, yys, xxs = np.where(mask[crop])
    z0, y0, x0 = zsl.start or 0, ysl.start or 0, xsl.start or 0
    for dz0, dy0, dx0 in zip(zzs, yys, xxs):
        z, y, x = int(dz0 + z0), int(dy0 + y0), int(dx0 + x0)
        for axis, side in steps:
            nb = [z, y, x]
            nb[axis] += side
            inside = 0 <= nb[0] < mask.shape[0] and 0 <= nb[1] < mask.shape[1] and 0 <= nb[2] < mask.shape[2]
            if not inside or not mask[tuple(nb)]:
                faces.append(cube_face_vertices(z, y, x, axis, side))
    return faces


def intercell_faces(labels: np.ndarray, crop: tuple[slice, slice, slice]) -> list[list[tuple[float, float, float]]]:
    faces: list[list[tuple[float, float, float]]] = []
    zsl, ysl, xsl = crop
    zrange = range(zsl.start or 0, zsl.stop or labels.shape[0])
    yrange = range(ysl.start or 0, ysl.stop or labels.shape[1])
    xrange = range(xsl.start or 0, xsl.stop or labels.shape[2])
    for z in zrange:
        for y in yrange:
            for x in xrange:
                if labels[z, y, x] not in (0, 1):
                    continue
                for axis in (0, 1, 2):
                    nb = [z, y, x]
                    nb[axis] += 1
                    if nb[0] >= labels.shape[0] or nb[1] >= labels.shape[1] or nb[2] >= labels.shape[2]:
                        continue
                    pair = {int(labels[z, y, x]), int(labels[tuple(nb)])}
                    if pair == {0, 1}:
                        faces.append(cube_face_vertices(z, y, x, axis, 1))
    return faces


def selected_face_poly(face_c_i: np.ndarray, face_c_j: np.ndarray) -> list[tuple[float, float, float]]:
    a = np.asarray(face_c_i, dtype=int)
    b = np.asarray(face_c_j, dtype=int)
    diff = b - a
    axis = int(np.flatnonzero(diff)[0])
    side = int(np.sign(diff[axis]))
    return cube_face_vertices(int(a[0]), int(a[1]), int(a[2]), axis, side)


def add_poly(ax, faces, color, alpha, edge="#ffffff", lw=0.14, zsort="average") -> None:
    if not faces:
        return
    coll = Poly3DCollection(faces, facecolor=color, edgecolor=edge, linewidth=lw, alpha=alpha, zsort=zsort)
    ax.add_collection3d(coll)


def add_site_3d(ax, site: np.ndarray, label: str, color: str, text_offset: tuple[float, float, float]) -> None:
    x, y, z = float(site[2] + 0.5), float(site[1] + 0.5), float(site[0] + 0.5)
    ax.scatter([x], [y], [z], s=54, c=color, depthshade=True, edgecolors="white", linewidths=0.65, zorder=8)
    tx, ty, tz = x + text_offset[0], y + text_offset[1], z + text_offset[2]
    ax.text(tx, ty, tz, label, color=color, fontsize=7.2, path_effects=[pe.withStroke(linewidth=2.3, foreground="white", alpha=0.95)])


def draw_panel_d(ax, labels, site_i, site_j, path_i, path_j, face_c_i, face_c_j, z: int) -> dict[str, object]:
    comp_i = component_from_seed(labels, 0, np.asarray(face_c_i, dtype=int))
    comp_j = component_from_seed(labels, 1, np.asarray(face_c_j, dtype=int))
    crop = crop_from_geometry([site_i, site_j, face_c_i, face_c_j, path_i, path_j], labels.shape)
    faces_i = exposed_faces(comp_i, crop)
    faces_j = exposed_faces(comp_j, crop)
    fij_faces = intercell_faces(labels, crop)
    selected = selected_face_poly(face_c_i, face_c_j)

    zsl, ysl, xsl = crop
    xmin, xmax = xsl.start or 0, xsl.stop or labels.shape[2]
    ymin, ymax = ysl.start or 0, ysl.stop or labels.shape[1]
    zmin, zmax = zsl.start or 0, zsl.stop or labels.shape[0]

    slice_plane = [[(xmin, ymin, z + 0.5), (xmax, ymin, z + 0.5), (xmax, ymax, z + 0.5), (xmin, ymax, z + 0.5)]]
    add_poly(ax, slice_plane, "#d5eee9", 0.16, edge="#8bbeb1", lw=0.35)
    add_poly(ax, faces_i, PALETTE["omega_i"], 0.80, edge="#e8f1f5", lw=0.025)
    add_poly(ax, faces_j, PALETTE["omega_j"], 0.80, edge="#fff4cf", lw=0.025)
    add_poly(ax, fij_faces, PALETTE["gamma"], 0.78, edge=PALETTE["gamma"], lw=0.22, zsort="min")
    add_poly(ax, [selected], PALETTE["gamma"], 0.98, edge="#ffffff", lw=0.65, zsort="min")
    add_site_3d(ax, site_i, r"$s_i$", PALETTE["site"], (-2.2, -1.2, 1.1))
    add_site_3d(ax, site_j, r"$s_j$", PALETTE["site"], (1.2, 1.2, 1.1))

    ax.text2D(0.06, 0.82, r"$\Omega_i$", transform=ax.transAxes, color=PALETTE["omega_i_dark"], fontsize=7.0, path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.95)])
    ax.text2D(0.73, 0.82, r"$\Omega_j$", transform=ax.transAxes, color=PALETTE["omega_j_dark"], fontsize=7.0, path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.95)])
    ax.text2D(0.08, 0.13, "slice from a", transform=ax.transAxes, color="#51636f", fontsize=6.1)
    ax.text2D(0.56, 0.13, r"$\mathcal{F}_{ij}$", transform=ax.transAxes, color=PALETTE["gamma"], fontsize=6.8, path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.95)])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, max(1, zmax - zmin) * 1.35), zoom=1.02)
    ax.view_init(elev=25, azim=-55)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    add_panel_label_3d(ax, "d", "3D ownership cells")
    return {
        "crop_zyx": [[int(zsl.start or 0), int(zsl.stop or labels.shape[0])], [int(ysl.start or 0), int(ysl.stop or labels.shape[1])], [int(xsl.start or 0), int(xsl.stop or labels.shape[2])]],
        "omega_i_component_voxels": int(comp_i.sum()),
        "omega_j_component_voxels": int(comp_j.sum()),
        "omega_i_rendered_voxels": int(comp_i[crop].sum()),
        "omega_j_rendered_voxels": int(comp_j[crop].sum()),
        "rendered_intercell_facelets": int(len(fij_faces)),
    }


def validate_facelet(mask: np.ndarray, labels: np.ndarray, face_c_i: np.ndarray, face_c_j: np.ndarray, wall_pick: dict[str, np.ndarray], z: int) -> None:
    ci = tuple(int(v) for v in face_c_i)
    cj = tuple(int(v) for v in face_c_j)
    if not (mask[ci] and mask[cj]):
        raise AssertionError("Selected intercell facelet is not pore-pore")
    if {int(labels[ci]), int(labels[cj])} != {0, 1}:
        raise AssertionError("Selected intercell facelet is not between Omega_i and Omega_j")
    if int(np.sum(np.abs(np.asarray(ci) - np.asarray(cj)))) != 1:
        raise AssertionError("Selected intercell facelet is not a positive-area face contact")
    vy, vx = np.asarray(wall_pick["voxel_yx"], dtype=int)
    sy, sx = np.asarray(wall_pick["solid_yx"], dtype=int)
    if labels[z, vy, vx] != 0 or mask[z, sy, sx]:
        raise AssertionError("Selected wall facelet is not Omega_i-solid")


def main() -> None:
    data = load_source()
    meta = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))

    mask = data["mask"].astype(bool)
    labels = data["labels"]
    site_i = data["site_i"]
    site_j = data["site_j"]
    path_i = data["path_i"]
    path_j = data["path_j"]
    face_center = data["face_center"]
    face_c_i = data["face_c_i"]
    face_c_j = data["face_c_j"]
    gamma_segments = data["slice_facelets_xy"]
    selected_gamma = data["selected_face_segment_xy"]
    z = int(site_i[0])
    img = slice_image(mask, labels, z)

    walls = wall_segments(mask, labels, z, owner=0)
    wall_pick = select_wall_segment(walls, site_i, face_center, mask, labels, z)
    validate_facelet(mask, labels, face_c_i, face_c_j, wall_pick, z)
    wall_all = np.asarray([w["seg"] for w in walls], dtype=float)
    wall_seg = np.asarray(wall_pick["seg"], dtype=float)
    wall_centre = np.asarray(wall_pick["centre"], dtype=float)
    wall_cell = np.asarray(wall_pick["cell_centre"], dtype=float)
    wall_normal = np.asarray(wall_pick["normal"], dtype=float)
    state_xy = np.array([site_i[2] + 0.5, site_i[1] + 0.5], dtype=float)
    wall_delta = float(np.dot(wall_centre - state_xy, wall_normal))
    projected_wall_xy = state_xy + wall_delta * wall_normal

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.08, 5.15), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.035, right=0.985, bottom=0.055, top=0.935, wspace=0.095, hspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], projection="3d")

    xlim_a = (0.0, 48.0)
    ylim_a = (57.5, 10.0)
    draw_base(ax_a, img, xlim_a, ylim_a)
    ax_a.add_collection(LineCollection(gamma_segments, colors=PALETTE["gamma"], linewidths=0.70, alpha=0.18, zorder=20, capstyle="butt"))
    ax_a.add_collection(LineCollection(wall_all, colors=PALETTE["wall"], linewidths=0.58, alpha=0.20, zorder=19, capstyle="butt"))
    ax_a.add_collection(LineCollection([selected_gamma], colors=PALETTE["gamma"], linewidths=3.15, zorder=27, capstyle="butt"))
    ax_a.add_collection(LineCollection([wall_seg], colors=PALETTE["wall_pick"], linewidths=3.05, zorder=28, capstyle="butt"))
    add_site(ax_a, site_i, r"$s_i$", 2.3, 0.8)
    add_site(ax_a, site_j, r"$s_j$", -3.6, 2.3)
    label_text(ax_a, 12.8, 25.7, r"$\Omega_i$", color=PALETTE["omega_i_dark"], size=8.5)
    label_text(ax_a, 30.3, 41.0, r"$\Omega_j$", color=PALETTE["omega_j_dark"], size=8.5)
    label_text(ax_a, 33.3, 28.9, r"$\mathcal{F}_{ij}$", color=PALETTE["gamma"], size=7.6)
    label_text(ax_a, wall_centre[0] + 3.2, wall_centre[1] - 2.5, r"$\mathcal{W}_i$", color=PALETTE["wall_pick"], size=7.4, ha="left")
    add_panel_label(ax_a, "a", "Graph-geodesic ownership slice")

    fc_xy = np.array([face_center[2], face_center[1]], dtype=float)
    xlim_b = (fc_xy[0] - 10.0, fc_xy[0] + 12.5)
    ylim_b = (fc_xy[1] + 10.5, fc_xy[1] - 8.5)
    draw_base(ax_b, img, xlim_b, ylim_b)
    ax_b.add_collection(LineCollection(gamma_segments, colors=PALETTE["gamma"], linewidths=0.85, alpha=0.22, zorder=20, capstyle="butt"))
    ax_b.add_collection(LineCollection([selected_gamma], colors=PALETTE["gamma"], linewidths=3.9, zorder=29, capstyle="butt"))
    draw_paths(ax_b, site_i, site_j, path_i, path_j, face_center, show_chord=True)
    p_xy = np.array([face_c_i[2] + 0.5, face_c_i[1] + 0.5], dtype=float)
    q_xy = np.array([face_c_j[2] + 0.5, face_c_j[1] + 0.5], dtype=float)
    ax_b.add_patch(Circle(tuple(p_xy), 0.43, facecolor="white", edgecolor=PALETTE["ink"], lw=1.0, zorder=36))
    ax_b.add_patch(Circle(tuple(q_xy), 0.43, facecolor="white", edgecolor=PALETTE["ink"], lw=1.0, zorder=36))
    label_text(ax_b, p_xy[0] - 2.2, p_xy[1] - 1.2, r"$p_\gamma$", color=PALETTE["ink"], size=6.9)
    label_text(ax_b, q_xy[0] + 1.95, q_xy[1] + 1.35, r"$q_\gamma$", color=PALETTE["ink"], size=6.9)
    label_text(ax_b, fc_xy[0] + 2.05, fc_xy[1] - 2.25, r"$\gamma\in\mathcal{F}_{ij}$", color=PALETTE["gamma"], size=6.9, ha="left")
    label_text(ax_b, fc_xy[0] - 6.3, fc_xy[1] - 4.9, r"$r_{i,\gamma}^{g}$", color=PALETTE["path"], size=6.9)
    label_text(ax_b, fc_xy[0] + 5.2, fc_xy[1] + 3.9, r"$r_{j,\gamma}^{g}$", color=PALETTE["path"], size=6.9)
    label_text(ax_b, fc_xy[0] - 1.5, fc_xy[1] + 6.9, r"$\ell_\gamma^g=r_{i,\gamma}^{g}+r_{j,\gamma}^{g}$", color=PALETTE["ink"], size=6.8)
    label_text(ax_b, fc_xy[0] - 5.9, fc_xy[1] + 5.25, "Euclidean chord", color=PALETTE["chord"], size=6.1)
    add_panel_label(ax_b, "b", "Cell-cell exchange facelet")

    wc = wall_centre
    xlim_c = (min(state_xy[0], wc[0]) - 3.0, max(state_xy[0], wc[0]) + 8.0)
    ylim_c = (max(state_xy[1], wc[1]) + 7.5, min(state_xy[1], wc[1]) - 7.5)
    draw_base(ax_c, img, xlim_c, ylim_c)
    ax_c.add_collection(LineCollection(wall_all, colors=PALETTE["wall"], linewidths=0.82, alpha=0.25, zorder=20, capstyle="butt"))
    ax_c.add_collection(LineCollection([wall_seg], colors=PALETTE["wall_pick"], linewidths=4.0, zorder=30, capstyle="butt"))
    ax_c.add_patch(Circle(tuple(state_xy), 0.56, facecolor=PALETTE["site"], edgecolor="white", lw=1.1, zorder=35))
    ax_c.add_patch(Circle(tuple(projected_wall_xy), 0.26, facecolor=PALETTE["wall_pick"], edgecolor="white", lw=0.7, zorder=36))
    ax_c.plot(
        [state_xy[0], projected_wall_xy[0]],
        [state_xy[1], projected_wall_xy[1]],
        color=PALETTE["chord"],
        lw=2.35,
        zorder=32,
        path_effects=[pe.withStroke(linewidth=3.8, foreground="white", alpha=0.92)],
    )
    label_text(ax_c, state_xy[0] - 0.6, state_xy[1] + 2.05, r"$\mathbf{x}_i^a$", color=PALETTE["site"], size=7.0)
    label_text(ax_c, (state_xy[0] + projected_wall_xy[0]) / 2, (state_xy[1] + projected_wall_xy[1]) / 2 - 1.15, r"$\delta_{i,\gamma}^{\mathrm{w}}$", color=PALETTE["chord"], size=7.0)
    label_text(ax_c, wc[0] + 1.50, wc[1] - 1.35, r"$\gamma\in\mathcal{W}_i$", color=PALETTE["wall_pick"], size=6.9, ha="left")
    label_text(ax_c, wc[0] + 1.55, wc[1] + 1.35, r"$A_\gamma$", color=PALETTE["wall_pick"], size=6.9, ha="left")
    label_text(ax_c, state_xy[0] + 7.0, state_xy[1] + 4.4, r"$\Omega_i$", color=PALETTE["omega_i_dark"], size=7.2)
    add_panel_label(ax_c, "c", "Wall closure facelet")

    panel_d_audit = draw_panel_d(ax_d, labels, site_i, site_j, path_i, path_j, face_c_i, face_c_j, z)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{OUT_STEM}.{ext}", dpi=700 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    audit = {
        "figure": OUT_STEM,
        "source_data": SOURCE_NPZ.name,
        "source_meta": meta,
        "display_slice_z_local": int(z),
        "site_i_local_zyx": np.asarray(site_i, dtype=int).tolist(),
        "site_j_local_zyx": np.asarray(site_j, dtype=int).tolist(),
        "selected_gamma": {
            "cell_i_zyx": np.asarray(face_c_i, dtype=int).tolist(),
            "cell_j_zyx": np.asarray(face_c_j, dtype=int).tolist(),
            "face_center_zyx": np.asarray(face_center, dtype=float).tolist(),
            "segment_xy": np.asarray(selected_gamma, dtype=float).tolist(),
            "is_positive_area_pore_pore_contact": True,
        },
        "intercell_facelets_on_slice": int(len(gamma_segments)),
        "intercell_facelets_3d_from_source_meta": int(meta.get("inter_label_facelets_3d", -1)),
        "selected_wall_facelet": {
            "segment_xy": wall_seg.tolist(),
            "centre_xy": wall_centre.tolist(),
            "omega_i_voxel_yx": np.asarray(wall_pick["voxel_yx"], dtype=int).tolist(),
            "solid_voxel_yx": np.asarray(wall_pick["solid_yx"], dtype=int).tolist(),
            "adjacent_cell_centre_xy": wall_cell.tolist(),
            "anchor_xy": state_xy.tolist(),
            "normal_projection_endpoint_xy": projected_wall_xy.tolist(),
            "delta_in_voxels_before_clamp": wall_delta,
            "is_positive_area_pore_solid_contact": True,
        },
        "panel_d": panel_d_audit,
        "visual_claim": "The same 3D graph-geodesic ownership record supplies cell-cell facelet exchange and wall-facelet closure.",
    }
    OUT_DATA.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(FIG_DIR / f"{OUT_STEM}.png")
    print(FIG_DIR / f"{OUT_STEM}.pdf")
    print(FIG_DIR / f"{OUT_STEM}.svg")
    print(OUT_DATA)


if __name__ == "__main__":
    main()


