from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


AXIS_UNIT = np.eye(3, dtype=np.float64)


def _owner_to_neighbor_normals(cell_id: np.ndarray, facelets: dict[str, np.ndarray]) -> np.ndarray:
    """Return local face normals in z-y-x order, oriented owner -> neighbor."""
    normals = np.zeros((len(facelets["owner"]), 3), dtype=np.float64)
    shape = np.asarray(cell_id.shape)
    for fid, (owner, neighbor, axis, midpoint) in enumerate(
        zip(facelets["owner"], facelets["neighbor"], facelets["axis"], facelets["midpoint_zyx"])
    ):
        axis_i = int(axis)
        lo = np.floor(midpoint).astype(int)
        hi = lo.copy()
        hi[axis_i] += 1
        if np.any(lo < 0) or np.any(hi < 0) or np.any(lo >= shape) or np.any(hi >= shape):
            continue
        c_lo = int(cell_id[tuple(lo)])
        c_hi = int(cell_id[tuple(hi)])
        sign = 0.0
        if c_lo == int(owner) and c_hi == int(neighbor):
            sign = 1.0
        elif c_lo == int(neighbor) and c_hi == int(owner):
            sign = -1.0
        normals[fid, axis_i] = sign
    return normals


def reconstruct_cell_velocity(
    cell_id: np.ndarray,
    cell_volume: np.ndarray,
    facelets: dict[str, np.ndarray],
    graph: dict[str, np.ndarray],
    flux: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Reconstruct a cell-level velocity proxy from conservative FV edge fluxes.

    The stored FV unknowns carry conservative fluxes on graph edges.  This
    routine distributes each internal edge flux over its voxel facelets and
    reports the area-weighted physical flux vector associated with every cell.
    Boundary reservoir fluxes are added in the global +x flow direction.
    """
    n_cells = int(len(cell_volume))
    vector_sum = np.zeros((n_cells, 3), dtype=np.float64)
    area_sum = np.zeros(n_cells, dtype=np.float64)
    internal_area_sum = np.zeros(n_cells, dtype=np.float64)

    q_by_pair: dict[tuple[int, int], float] = {}
    for a, b, kind, q in zip(
        flux["edge_owner"], flux["edge_neighbor"], flux["edge_kind"], flux["q_conservative"]
    ):
        if str(kind) == "internal" and int(b) >= 0:
            aa, bb = sorted((int(a), int(b)))
            q_by_pair[(aa, bb)] = float(q)

    normals = _owner_to_neighbor_normals(cell_id, facelets)
    if len(facelets["owner"]):
        pairs = np.column_stack([facelets["owner"], facelets["neighbor"]]).astype(np.int32)
        unique_pair, inverse = np.unique(pairs, axis=0, return_inverse=True)
        face_area_by_edge = np.bincount(inverse, minlength=len(unique_pair)).astype(np.float64)
        pair_to_area = {
            (int(a), int(b)): float(area)
            for (a, b), area in zip(unique_pair, face_area_by_edge)
        }
        for fid, (owner, neighbor) in enumerate(pairs):
            pair = (int(owner), int(neighbor))
            q = q_by_pair.get(pair)
            area = pair_to_area.get(pair, 0.0)
            if q is None or area <= 0.0:
                continue
            v_face = (q / area) * normals[fid]
            for cid in pair:
                if 0 <= cid < n_cells:
                    vector_sum[cid] += v_face
                    area_sum[cid] += 1.0
                    internal_area_sum[cid] += 1.0

    # Boundary reservoir edges are axis-aligned in this typographic demo: inlet
    # and outlet are the minimum and maximum x faces of the pore domain.
    x_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for a, b, kind, q, conductance in zip(
        flux["edge_owner"],
        flux["edge_neighbor"],
        flux["edge_kind"],
        flux["q_conservative"],
        flux["conductance"],
    ):
        cid = int(a)
        if cid < 0 or cid >= n_cells:
            continue
        area = float(conductance)
        if area <= 0.0:
            continue
        if str(kind) == "inlet":
            v_boundary = (-float(q) / area) * x_dir
        elif str(kind) == "outlet":
            v_boundary = (float(q) / area) * x_dir
        else:
            continue
        vector_sum[cid] += v_boundary * area
        area_sum[cid] += area

    cell_velocity = np.zeros_like(vector_sum)
    active = area_sum > 0.0
    cell_velocity[active] = vector_sum[active] / area_sum[active, None]
    cell_speed = np.linalg.norm(cell_velocity, axis=1)

    voxel_velocity = np.zeros(cell_id.shape + (3,), dtype=np.float32)
    valid = cell_id >= 0
    voxel_velocity[valid] = cell_velocity[cell_id[valid]].astype(np.float32)

    result = {
        "cell_velocity_zyx": cell_velocity.astype(np.float32),
        "cell_speed": cell_speed.astype(np.float32),
        "cell_area_weight": area_sum.astype(np.float32),
        "cell_internal_facelet_area": internal_area_sum.astype(np.float32),
        "voxel_velocity_zyx": voxel_velocity,
    }
    audit = {
        "cells": int(n_cells),
        "velocity_cells_with_faces": int(np.sum(active)),
        "nonzero_velocity_cells": int(np.sum(cell_speed > 0.0)),
        "mean_cell_speed": float(cell_speed[active].mean()) if np.any(active) else 0.0,
        "max_cell_speed": float(cell_speed.max()) if len(cell_speed) else 0.0,
        "velocity_definition": "area-weighted cell velocity reconstructed from conservative FV graph fluxes distributed over positive-area facelets and inlet/outlet boundary faces",
        "component_order": "zyx",
    }
    return result, audit


def _project_label(label3d: np.ndarray) -> np.ndarray:
    valid = label3d >= 0
    proj = np.full(label3d.shape[1:], -1, dtype=np.int32)
    nz = label3d.shape[0]
    order = list(range(nz // 2, nz)) + list(range(nz // 2 - 1, -1, -1))
    for z in order:
        fill = (proj < 0) & valid[z]
        proj[fill] = label3d[z][fill]
    return proj


def _crop2(mask: np.ndarray, margin: int = 12) -> tuple[slice, slice]:
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min() - margin), min(mask.shape[0], yy.max() + margin + 1)
    x0, x1 = max(0, xx.min() - margin), min(mask.shape[1], xx.max() + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def render_cell_velocity_projection(
    cell_id: np.ndarray,
    cell_velocity: np.ndarray,
    out_path: str | Path,
) -> dict[str, object]:
    lab2 = _project_label(cell_id)
    inside = lab2 >= 0
    ys, xs = _crop2(inside, margin=8)
    cropped = lab2[ys, xs]
    speed = np.zeros_like(cropped, dtype=np.float32)
    valid = cropped >= 0
    cell_speed = np.linalg.norm(cell_velocity, axis=1)
    speed[valid] = cell_speed[cropped[valid]]

    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=240)
    ax.imshow(speed, cmap="mako" if "mako" in plt.colormaps() else "viridis", interpolation="nearest")
    ax.contour(valid.astype(float), levels=[0.5], colors=["#263640"], linewidths=0.25)

    labels = np.unique(cropped[valid])
    labels = labels[labels >= 0]
    if len(labels):
        mags = cell_speed[labels]
        keep = labels[np.argsort(mags)[-min(55, len(labels)):]]
        points = []
        vectors = []
        for cid in keep:
            yy, xx = np.where(cropped == cid)
            if len(xx) == 0:
                continue
            v = cell_velocity[cid]
            if np.linalg.norm(v) <= 0:
                continue
            points.append((float(xx.mean()), float(yy.mean())))
            vectors.append((float(v[2]), float(-v[1])))
        if points:
            pts = np.asarray(points)
            vec = np.asarray(vectors)
            norm = np.linalg.norm(vec, axis=1)
            vec = vec / np.maximum(norm[:, None], 1.0e-12)
            ax.quiver(
                pts[:, 0],
                pts[:, 1],
                vec[:, 0],
                vec[:, 1],
                color="#F04B7F",
                angles="xy",
                scale_units="xy",
                scale=0.23,
                width=0.006,
                headwidth=3.8,
                headlength=4.6,
                headaxislength=4.0,
                alpha=0.9,
            )
    ax.set_title("Voronoi-FV cell-level velocity reconstructed from conservative face fluxes", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout(pad=0.1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=360, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {"velocity_figure": str(out), "velocity_figure_svg": str(out.with_suffix(".svg"))}


def write_cell_velocity(root: str | Path | None = None) -> dict[str, object]:
    cells = np.load(out_dir("fv") / "fv_cells.npz")
    facelets_npz = np.load(out_dir("fv") / "facelets.npz")
    graph_npz = np.load(out_dir("fv") / "fv_graph.npz")
    flux_npz = np.load(out_dir("flux") / "conservative_flux_projection.npz")
    facelets = {k: facelets_npz[k] for k in facelets_npz.files}
    graph = {k: graph_npz[k] for k in graph_npz.files}
    flux = {k: flux_npz[k] for k in flux_npz.files}
    velocity, audit = reconstruct_cell_velocity(
        cells["cell_id"], cells["cell_volume"], facelets, graph, flux
    )
    out = out_dir("velocity")
    np.savez_compressed(out / "cell_velocity.npz", **velocity)
    audit.update(render_cell_velocity_projection(cells["cell_id"], velocity["cell_velocity_zyx"], out / "Figure_PoreVoronoi_FV_cell_velocity.png"))
    save_json(out / "cell_velocity_audit.json", audit)
    return audit
