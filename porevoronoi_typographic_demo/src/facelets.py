from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_utils import out_dir, save_json


AXES = ((0, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1)))


def _adjacent_views(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    s0 = [slice(None)] * arr.ndim
    s1 = [slice(None)] * arr.ndim
    s0[axis] = slice(0, -1)
    s1[axis] = slice(1, None)
    return arr[tuple(s0)], arr[tuple(s1)]


def extract_facelets(
    pore: np.ndarray,
    cell_id: np.ndarray,
    distance: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Extract positive-area cell-cell facelets and aggregate FV graph edges."""
    face_owner: list[np.ndarray] = []
    face_neighbor: list[np.ndarray] = []
    face_axis: list[np.ndarray] = []
    face_mid: list[np.ndarray] = []
    face_ri: list[np.ndarray] = []
    face_rj: list[np.ndarray] = []

    for axis, _direction in AXES:
        c0, c1 = _adjacent_views(cell_id, axis)
        d0, d1 = _adjacent_views(distance, axis)
        valid = (c0 >= 0) & (c1 >= 0) & (c0 != c1)
        coords = np.argwhere(valid)
        if coords.size == 0:
            continue
        owner = c0[valid].astype(np.int32)
        neighbor = c1[valid].astype(np.int32)
        mids = coords.astype(np.float32)
        mids[:, axis] += 0.5
        ri = d0[valid].astype(np.float32) + 0.5
        rj = d1[valid].astype(np.float32) + 0.5
        swap = owner > neighbor
        if np.any(swap):
            owner_s = owner.copy()
            owner[swap] = neighbor[swap]
            neighbor[swap] = owner_s[swap]
            ri_s = ri.copy()
            ri[swap] = rj[swap]
            rj[swap] = ri_s[swap]
        face_owner.append(owner)
        face_neighbor.append(neighbor)
        face_axis.append(np.full(len(owner), axis, dtype=np.int8))
        face_mid.append(mids)
        face_ri.append(ri)
        face_rj.append(rj)

    if face_owner:
        owner_arr = np.concatenate(face_owner)
        neigh_arr = np.concatenate(face_neighbor)
        axis_arr = np.concatenate(face_axis)
        mid_arr = np.vstack(face_mid).astype(np.float32)
        ri_arr = np.concatenate(face_ri).astype(np.float32)
        rj_arr = np.concatenate(face_rj).astype(np.float32)
    else:
        owner_arr = np.zeros(0, dtype=np.int32)
        neigh_arr = np.zeros(0, dtype=np.int32)
        axis_arr = np.zeros(0, dtype=np.int8)
        mid_arr = np.zeros((0, 3), dtype=np.float32)
        ri_arr = np.zeros(0, dtype=np.float32)
        rj_arr = np.zeros(0, dtype=np.float32)

    if len(owner_arr):
        pair = np.column_stack([owner_arr, neigh_arr])
        unique_pair, inverse = np.unique(pair, axis=0, return_inverse=True)
        area = np.bincount(inverse, minlength=len(unique_pair)).astype(np.float32)
        conductance = np.bincount(
            inverse,
            weights=(1.0 / np.maximum(ri_arr + rj_arr, 1.0e-6)).astype(np.float64),
            minlength=len(unique_pair),
        ).astype(np.float32)
        mean_length = area / np.maximum(conductance, 1.0e-9)
    else:
        unique_pair = np.zeros((0, 2), dtype=np.int32)
        area = np.zeros(0, dtype=np.float32)
        conductance = np.zeros(0, dtype=np.float32)
        mean_length = np.zeros(0, dtype=np.float32)

    n_cells = int(cell_id.max()) + 1 if np.any(cell_id >= 0) else 0
    wall_area = np.zeros(n_cells, dtype=np.int32)
    for dz, dy, dx in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        shifted = np.roll(pore, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
        valid = pore & ~shifted
        if dz == 1:
            valid[-1, :, :] = pore[-1, :, :]
        if dz == -1:
            valid[0, :, :] = pore[0, :, :]
        if dy == 1:
            valid[:, -1, :] = pore[:, -1, :]
        if dy == -1:
            valid[:, 0, :] = pore[:, 0, :]
        if dx == 1:
            valid[:, :, -1] = pore[:, :, -1]
        if dx == -1:
            valid[:, :, 0] = pore[:, :, 0]
        ids = cell_id[valid]
        ids = ids[ids >= 0]
        if len(ids):
            wall_area += np.bincount(ids, minlength=n_cells).astype(np.int32)

    coords = np.argwhere(pore)
    xmin = int(coords[:, 2].min())
    xmax = int(coords[:, 2].max())
    inlet_ids = cell_id[pore & (np.arange(pore.shape[2])[None, None, :] == xmin)]
    outlet_ids = cell_id[pore & (np.arange(pore.shape[2])[None, None, :] == xmax)]
    inlet_area = np.bincount(inlet_ids[inlet_ids >= 0], minlength=n_cells).astype(np.float32)
    outlet_area = np.bincount(outlet_ids[outlet_ids >= 0], minlength=n_cells).astype(np.float32)

    facelets = {
        "owner": owner_arr,
        "neighbor": neigh_arr,
        "axis": axis_arr,
        "midpoint_zyx": mid_arr,
        "r_owner_g": ri_arr,
        "r_neighbor_g": rj_arr,
    }
    graph = {
        "edge_owner": unique_pair[:, 0].astype(np.int32),
        "edge_neighbor": unique_pair[:, 1].astype(np.int32),
        "area": area,
        "conductance": conductance,
        "mean_geodesic_length": mean_length,
        "wall_area": wall_area.astype(np.float32),
        "inlet_area": inlet_area,
        "outlet_area": outlet_area,
    }
    audit = {
        "cell_cell_facelets": int(len(owner_arr)),
        "fv_graph_edges": int(len(area)),
        "wall_facelets": int(wall_area.sum()),
        "inlet_boundary_faces": int(inlet_area.sum()),
        "outlet_boundary_faces": int(outlet_area.sum()),
        "mean_edge_area": float(area.mean()) if len(area) else 0.0,
        "max_edge_area": float(area.max()) if len(area) else 0.0,
    }
    return facelets, graph, audit


def write_facelets(root: str | Path | None = None) -> dict[str, object]:
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    cells = np.load(out_dir("fv") / "fv_cells.npz")
    ownership = np.load(out_dir("ownership") / "graph_geodesic_ownership.npz")
    facelets, graph, audit = extract_facelets(pore, cells["cell_id"], ownership["distance"])
    out = out_dir("fv")
    np.savez_compressed(out / "facelets.npz", **facelets)
    np.savez_compressed(out / "fv_graph.npz", **graph)
    save_json(out / "facelets_audit.json", audit)
    return audit
