from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy import ndimage
from skimage import measure, transform


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
    }
)


def _crop_mask(mask: np.ndarray, margin: int = 35) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mask.shape[0], y1 + margin)
    x1 = min(mask.shape[1], x1 + margin)
    return mask[y0:y1, x0:x1], (y0, y1, x0, x1)


def _flow_texture(mask: np.ndarray, dark: bool, seed: int = 20260625) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / max(w - 1, 1)
    y = yy / max(h - 1, 1)
    noise1 = ndimage.gaussian_filter(rng.normal(size=(h, w)), sigma=9)
    noise2 = ndimage.gaussian_filter(rng.normal(size=(h, w)), sigma=26)
    noise3 = ndimage.gaussian_filter(rng.normal(size=(h, w)), sigma=54)
    wave = np.sin(10.5 * x + 4.0 * noise1 + 1.9 * np.sin(5.2 * y))
    ribbon = np.sin(23.0 * (x - 0.31 * y) + 5.8 * noise2)
    plume = np.sin(7.5 * (x + 0.55 * y) + 6.5 * noise3)
    t = 0.5 + 0.24 * wave + 0.28 * ribbon + 0.18 * plume
    t = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    hot = np.clip((ribbon + 1) / 2, 0, 1)
    plume_w = np.clip((plume + 1) / 2, 0, 1)
    q = np.mod(0.62 * t + 0.26 * hot + 0.12 * plume_w + 0.10 * x, 1.0)
    if dark:
        palette = np.array(
            [
                [6, 10, 28],
                [28, 58, 155],
                [132, 42, 185],
                [229, 28, 175],
                [0, 192, 183],
                [98, 155, 255],
            ],
            dtype=float,
        ) / 255.0
    else:
        palette = np.array(
            [
                [219, 245, 248],
                [84, 128, 197],
                [151, 91, 188],
                [213, 62, 154],
                [30, 161, 157],
                [122, 184, 231],
            ],
            dtype=float,
        ) / 255.0
    stops = np.linspace(0, 1, len(palette))
    rgb = np.stack([np.interp(q, stops, palette[:, i]) for i in range(3)], axis=-1)
    highlight = np.clip(np.sin(38 * (x - 0.22 * y) + 4 * noise1), 0, 1) ** 8
    rgb = np.clip(rgb + highlight[..., None] * (0.20 if dark else 0.10), 0, 1)
    if dark:
        rgb = np.clip((rgb - 0.30) * 1.25 + 0.30, 0, 1)
    else:
        rgb = np.clip(0.90 * rgb + 0.10, 0, 1)
    rgb[~mask] = 1.0 if not dark else 0.0
    return rgb

def _project_points(points: np.ndarray, source_shape: tuple[int, int, int], target_shape: tuple[int, int]) -> np.ndarray:
    nz, ny, nx = source_shape
    h, w = target_shape
    if points.size == 0:
        return np.zeros((0, 2), dtype=float)
    x = points[:, 2] / max(nx - 1, 1) * w
    y = points[:, 1] / max(ny - 1, 1) * h
    return np.column_stack([x, y])


def _draw_stream_segments(
    ax,
    stream_points: np.ndarray,
    stream_offsets: np.ndarray,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int],
    dark: bool,
) -> None:
    """Draw sparse path fragments so the hero reads as fluid-in-text, not a debug trace."""
    line_pairs = list(zip(stream_offsets[:-1], stream_offsets[1:]))
    if not line_pairs:
        return
    color = "#F1FFFF" if dark else "#D7FFFF"
    edge = "#58C8D8" if dark else "#078894"
    stride = max(1, len(line_pairs) // 10)
    for idx, (a, b) in enumerate(line_pairs[::stride]):
        pts = stream_points[a:b]
        if len(pts) < 40:
            continue
        pxy = _project_points(pts, source_shape, target_shape)
        keep = (pxy[:, 0] > 70) & (pxy[:, 0] < target_shape[1] - 70)
        pxy = pxy[keep]
        if len(pxy) < 40:
            continue
        starts = [int(0.52 * len(pxy))]
        for start in starts:
            length = min(58, max(26, len(pxy) // 10))
            start = min(max(4, start + (idx % 5 - 2) * 9), max(5, len(pxy) - length - 4))
            seg = pxy[start : start + length].copy()
            if len(seg) < 12:
                continue
            win = min(11, max(5, (len(seg) // 10) * 2 + 1))
            kernel = np.ones(win) / win
            sx = np.convolve(seg[:, 0], kernel, mode="same")
            sy = np.convolve(seg[:, 1], kernel, mode="same")
            ax.plot(sx, sy, color=edge, linewidth=1.5, alpha=0.10, solid_capstyle="round", zorder=7)
            ax.plot(sx, sy, color=color, linewidth=0.62, alpha=0.28 if dark else 0.22, solid_capstyle="round", zorder=8)


def render_readme_hero(
    mask2d: np.ndarray,
    mask3d_shape: tuple[int, int, int],
    stream_points: np.ndarray,
    stream_offsets: np.ndarray,
    particles: np.ndarray,
    out_path: str | Path,
    dark: bool = False,
) -> None:
    cropped, _bbox = _crop_mask(mask2d, margin=48)
    target_w = 1900
    target_h = int(round(target_w * cropped.shape[0] / cropped.shape[1]))
    target_h = max(420, min(620, target_h))
    m = transform.resize(cropped.astype(float), (target_h, target_w), order=1, anti_aliasing=True) > 0.38
    texture = _flow_texture(m, dark=dark)

    fig_w = 13.6
    fig_h = 5.0
    bg = "#05070D" if dark else "#FBFDFE"
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(-90, target_w + 230)
    ax.set_ylim(target_h + 130, -80)
    ax.axis("off")

    dx, dy = 92, -48
    slab_edge = "#9DB8C5" if not dark else "#597287"
    slab_fill = "#E2EEF2" if not dark else "#1D2B3E"
    back_poly = Polygon(
        [(dx, dy), (target_w + dx, dy), (target_w + dx + 120, target_h + dy + 80), (120 + dx, target_h + dy + 80)],
        closed=True,
        facecolor=slab_fill,
        edgecolor=slab_edge,
        linewidth=1.2,
        alpha=0.18 if not dark else 0.30,
        zorder=1,
    )
    front = FancyBboxPatch(
        (0, 0),
        target_w,
        target_h,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor=slab_fill,
        edgecolor=slab_edge,
        linewidth=1.3,
        alpha=0.28 if not dark else 0.36,
        zorder=2,
    )
    ax.add_patch(back_poly)
    ax.add_patch(front)

    back_rgba = np.dstack([texture * (0.62 if dark else 0.74), m.astype(float) * (0.34 if dark else 0.30)])
    ax.imshow(back_rgba, extent=(dx, target_w + dx, target_h + dy, dy), zorder=3, interpolation="bilinear")
    front_rgba = np.dstack([texture, m.astype(float) * (0.99 if dark else 0.98)])
    ax.imshow(front_rgba, extent=(0, target_w, target_h, 0), zorder=5, interpolation="bilinear")

    contour_color = "#E5F8FF" if dark else "#234D5D"
    ax.contour(m.astype(float), levels=[0.5], colors=[contour_color], linewidths=1.15, alpha=0.74, zorder=6)
    for cont in measure.find_contours(m.astype(float), 0.5):
        ax.plot(
            cont[:, 1] + dx,
            cont[:, 0] + dy,
            color="#81AFC0" if not dark else "#405D79",
            linewidth=0.8,
            alpha=0.45,
            zorder=4,
        )

    _draw_stream_segments(ax, stream_points, stream_offsets, mask3d_shape, m.shape, dark)

    pp = _project_points(particles, mask3d_shape, m.shape)
    if len(pp):
        stride = max(1, len(pp) // 68)
        pp = pp[::stride]
        ax.scatter(pp[:, 0], pp[:, 1], s=20, c="#CC79A7", edgecolors="#5B244C", linewidths=0.28, alpha=0.84, zorder=9)

    ax.text(
        24,
        target_h + 74,
        "PoreVoronoi: letter-shaped pore space with pressure-driven streamlines and sampled states inside the void",
        color="#AFC5D1" if dark else "#4F6774",
        fontsize=9.2,
        ha="left",
        va="center",
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=360, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_debug_panels(
    mask2d: np.ndarray,
    pore3d: np.ndarray,
    pressure: np.ndarray,
    speed: np.ndarray,
    out_path: str | Path,
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(11.2, 5.7), dpi=180)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    axs[0, 0].imshow(mask2d, cmap="gray_r")
    axs[0, 0].set_title("2D typography pore mask", fontsize=8)
    axs[0, 1].imshow(pore3d.max(axis=0), cmap="viridis")
    axs[0, 1].set_title("3D pore projection", fontsize=8)
    mid = pore3d.shape[0] // 2
    p = np.ma.masked_where(~pore3d[mid], pressure[mid])
    axs[0, 2].imshow(p, cmap="magma", vmin=0, vmax=1)
    axs[0, 2].set_title("mid-plane pressure", fontsize=8)
    axs[1, 0].imshow(pore3d.max(axis=1), cmap="viridis")
    axs[1, 0].set_title("top projection", fontsize=8)
    sp = np.ma.masked_where(~pore3d[mid], speed[mid])
    axs[1, 1].imshow(sp, cmap="turbo")
    axs[1, 1].set_title("velocity proxy", fontsize=8)
    contours = measure.find_contours(mask2d.astype(float), 0.5)
    axs[1, 2].set_title("mask contours for audit", fontsize=8)
    axs[1, 2].set_xlim(0, mask2d.shape[1])
    axs[1, 2].set_ylim(mask2d.shape[0], 0)
    for c in contours:
        axs[1, 2].plot(c[:, 1], c[:, 0], color="#287764", linewidth=0.4)
    fig.suptitle("Stage 1 audit projections: not the README hero image", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


