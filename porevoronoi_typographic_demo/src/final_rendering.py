from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from .io_utils import out_dir, save_json


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


def _crop2(mask: np.ndarray, margin: int = 18) -> tuple[slice, slice]:
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min() - margin), min(mask.shape[0], yy.max() + margin + 1)
    x0, x1 = max(0, xx.min() - margin), min(mask.shape[1], xx.max() + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def _label_palette(n: int, alpha: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(20260625)
    base = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
    colors = np.zeros((n, 4), dtype=float)
    for i in range(n):
        c = base[i % len(base)] * (0.70 + 0.25 * rng.random())
        colors[i, :3] = np.clip(c, 0, 1)
        colors[i, 3] = alpha
    return colors


def _project_label(label3d: np.ndarray) -> np.ndarray:
    valid = label3d >= 0
    proj = np.full(label3d.shape[1:], -1, dtype=np.int32)
    nz = label3d.shape[0]
    order = list(range(nz // 2, nz)) + list(range(nz // 2 - 1, -1, -1))
    for z in order:
        fill = (proj < 0) & valid[z]
        proj[fill] = label3d[z][fill]
    return proj


def _render_cell_inset(ax, pore2: np.ndarray, cell_id: np.ndarray, facelets: dict[str, np.ndarray], graph: dict[str, np.ndarray]) -> None:
    if len(graph["area"]) == 0:
        ax.text(0.5, 0.5, "no FV edge", ha="center", va="center", transform=ax.transAxes)
        return
    pick = int(np.argmax(graph["area"]))
    a = int(graph["edge_owner"][pick])
    b = int(graph["edge_neighbor"][pick])
    match = ((facelets["owner"] == a) & (facelets["neighbor"] == b)) | ((facelets["owner"] == b) & (facelets["neighbor"] == a))
    mids = facelets["midpoint_zyx"][match]
    if len(mids) == 0:
        z = cell_id.shape[0] // 2
        cy, cx = np.array(cell_id.shape[1:]) // 2
    else:
        z = int(np.clip(np.median(mids[:, 0]), 0, cell_id.shape[0] - 1))
        cy = int(np.clip(np.median(mids[:, 1]), 0, cell_id.shape[1] - 1))
        cx = int(np.clip(np.median(mids[:, 2]), 0, cell_id.shape[2] - 1))
    y0, y1 = max(0, cy - 18), min(cell_id.shape[1], cy + 19)
    x0, x1 = max(0, cx - 35), min(cell_id.shape[2], cx + 36)
    sl = cell_id[z, y0:y1, x0:x1]
    n = int(cell_id.max()) + 1
    colors = _label_palette(max(n, 1), alpha=0.78)
    img = np.ones(sl.shape + (4,), dtype=float)
    inside = sl >= 0
    img[inside] = colors[sl[inside] % len(colors)]
    ax.imshow(img, interpolation="nearest")
    ax.contour(inside.astype(float), levels=[0.5], colors=["#24323A"], linewidths=0.35)
    if len(mids):
        local = mids[(mids[:, 0] >= z - 1.5) & (mids[:, 0] <= z + 1.5)]
        if len(local) == 0:
            local = mids
        ax.scatter(local[:, 2] - x0, local[:, 1] - y0, s=9, c="#C025D3", alpha=0.92, linewidths=0)
    ax.set_title(r"real $\Gamma_{ij}$ facelets", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def render_four_panel(out_path: str | Path) -> dict[str, object]:
    mask2d = np.load(out_dir("masks") / "word_mask_2d.npz")["mask"]
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    particles = np.load(out_dir("particles") / "particles.npz")["particles"]
    sites = np.load(out_dir("sites") / "prescribed_sites.npz")["sites"]
    ownership = np.load(out_dir("ownership") / "graph_geodesic_ownership.npz")
    cells = np.load(out_dir("fv") / "fv_cells.npz")
    facelets_npz = np.load(out_dir("fv") / "facelets.npz")
    graph_npz = np.load(out_dir("fv") / "fv_graph.npz")
    flux = np.load(out_dir("flux") / "conservative_flux_projection.npz")
    facelets = {k: facelets_npz[k] for k in facelets_npz.files}
    graph = {k: graph_npz[k] for k in graph_npz.files}

    fig, axs = plt.subplots(2, 2, figsize=(11.4, 6.2), dpi=220)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    ys, xs = _crop2(mask2d, margin=26)
    axs[0, 0].imshow(mask2d[ys, xs], cmap=ListedColormap(["#FFFFFF", "#DCEFF2"]), interpolation="nearest")
    p2 = np.column_stack([
        particles[:, 2] / (pore.shape[2] - 1) * mask2d.shape[1],
        particles[:, 1] / (pore.shape[1] - 1) * mask2d.shape[0],
    ])
    s2 = np.column_stack([
        sites[:, 2] / (pore.shape[2] - 1) * mask2d.shape[1],
        sites[:, 1] / (pore.shape[1] - 1) * mask2d.shape[0],
    ])
    axs[0, 0].scatter(p2[::3, 0] - xs.start, p2[::3, 1] - ys.start, s=5, c="#CC79A7", alpha=0.32, linewidths=0)
    axs[0, 0].scatter(s2[:, 0] - xs.start, s2[:, 1] - ys.start, s=8, c="#B65382", alpha=0.82, linewidths=0)
    axs[0, 0].set_title(r"sampled states become fixed sites $S_m$", fontsize=8)

    lab2 = _project_label(ownership["labels"])
    nlab = int(max(lab2.max() + 1, 1))
    palette = _label_palette(nlab, alpha=0.92)
    lab_img = np.ones(lab2.shape + (4,), dtype=float)
    inside = lab2 >= 0
    lab_img[inside] = palette[lab2[inside] % nlab]
    ys_l, xs_l = _crop2(lab2 >= 0, margin=3)
    axs[0, 1].imshow(lab_img[ys_l, xs_l], interpolation="nearest")
    axs[0, 1].contour((lab2[ys_l, xs_l] >= 0).astype(float), levels=[0.5], colors=["#23323A"], linewidths=0.28)
    axs[0, 1].set_title(r"graph-geodesic ownership $L(x)$", fontsize=8)

    _render_cell_inset(axs[1, 0], mask2d, cells["cell_id"], facelets, graph)

    ax = axs[1, 1]
    kinds = flux["edge_kind"]
    internal = kinds == "internal"
    q0 = flux["q_target"][internal]
    q1 = flux["q_conservative"][internal]
    if len(q0):
        keep = np.argsort(np.abs(q0))[-min(70, len(q0)):]
        x = np.arange(len(keep))
        ax.plot(x, q0[keep], color="#8AA0AE", linewidth=0.7, alpha=0.65, label="target")
        ax.plot(x, q1[keep], color="#009E73", linewidth=1.1, alpha=0.90, label="conservative")
        ax.axhline(0, color="#D5DEE3", linewidth=0.7)
        ax.legend(loc="upper left", fontsize=6, frameon=False)
    ax.set_frame_on(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6, length=2)
    ax.set_title("edge flux before and after cell-balance projection", fontsize=8)
    ax.set_xlabel("selected FV edges", fontsize=7)
    ax.set_ylabel("signed flux proxy", fontsize=7)

    fig.suptitle("PoreVoronoi-FV construction on the typographic pore-space demo", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=360, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"four_panel": str(out), "four_panel_svg": str(out.with_suffix(".svg"))}


def render_landing_hero(base_image: str | Path, out_path: str | Path, dark: bool) -> dict[str, object]:
    base = plt.imread(base_image)
    facelets_npz = np.load(out_dir("fv") / "facelets.npz")
    graph_npz = np.load(out_dir("fv") / "fv_graph.npz")
    cells = np.load(out_dir("fv") / "fv_cells.npz")
    facelets = {k: facelets_npz[k] for k in facelets_npz.files}
    graph = {k: graph_npz[k] for k in graph_npz.files}

    fig, ax = plt.subplots(figsize=(13.6, 5.0), dpi=220)
    bg = "#05070D" if dark else "#FBFDFE"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.imshow(base)
    ax.axis("off")

    inset = ax.inset_axes([0.65, 0.10, 0.30, 0.34])
    inset.set_facecolor("#FFFFFF" if not dark else "#071019")
    _render_cell_inset(inset, np.zeros((1, 1), dtype=bool), cells["cell_id"], facelets, graph)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#D9E4EA" if not dark else "#5B7184")
        spine.set_linewidth(0.8)
    ax.text(
        0.65,
        0.47,
        r"local PoreVoronoi-FV cells and $\Gamma_{ij}$ facelets",
        transform=ax.transAxes,
        color="#DDE9EF" if dark else "#405866",
        fontsize=7.5,
        ha="left",
        va="bottom",
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=360, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {"hero": str(out), "hero_svg": str(out.with_suffix(".svg"))}


def render_final_outputs(root: str | Path | None = None) -> dict[str, object]:
    out = out_dir("final_figures")
    result: dict[str, object] = {}
    result.update(render_four_panel(out / "Figure_PoreVoronoi_FV_four_panel.png"))
    figures = out_dir("figures")
    dark = out / "Figure_PoreVoronoi_README_landing_hero_dark.png"
    light = out / "Figure_PoreVoronoi_README_landing_hero_light.png"
    result.update(render_landing_hero(figures / "Figure_PoreVoronoi_stage1_README_hero_dark.png", dark, dark=True))
    result.update({"hero_dark": str(dark), "hero_dark_svg": str(dark.with_suffix(".svg"))})
    result.update(render_landing_hero(figures / "Figure_PoreVoronoi_stage1_README_hero_light.png", light, dark=False))
    result.update({"hero_light": str(light), "hero_light_svg": str(light.with_suffix(".svg"))})
    # The unqualified landing hero follows the dark README variant.
    img = plt.imread(dark)
    plain = out / "Figure_PoreVoronoi_README_landing_hero.png"
    plt.imsave(plain, img)
    result["hero_default"] = str(plain)
    save_json(out / "final_rendering_audit.json", result)
    return result

