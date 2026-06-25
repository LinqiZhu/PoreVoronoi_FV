from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .io_utils import out_dir, save_json


STRUCT6 = ndimage.generate_binary_structure(3, 1)


def split_face_connected_cells(
    pore: np.ndarray,
    labels: np.ndarray,
    n_sites: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Split each ownership label into 6-face-connected finite-volume cells."""
    comp = np.full(pore.shape, -1, dtype=np.int32)
    comp_site: list[int] = []
    comp_volume: list[int] = []
    split_labels = 0
    disconnected_labels = 0

    cid = 0
    for sid in range(n_sites):
        region = pore & (labels == sid)
        if not region.any():
            continue
        local, nlab = ndimage.label(region, structure=STRUCT6)
        if nlab > 1:
            disconnected_labels += 1
            split_labels += int(nlab - 1)
        for local_id in range(1, nlab + 1):
            vox = local == local_id
            vol = int(vox.sum())
            if vol == 0:
                continue
            comp[vox] = cid
            comp_site.append(sid)
            comp_volume.append(vol)
            cid += 1

    comp_site_arr = np.asarray(comp_site, dtype=np.int32)
    comp_volume_arr = np.asarray(comp_volume, dtype=np.int32)
    audit = {
        "sites_with_cells": int(len(np.unique(comp_site_arr))) if len(comp_site_arr) else 0,
        "finite_volume_cells": int(len(comp_site_arr)),
        "face_connected_splits": int(split_labels),
        "disconnected_site_labels": int(disconnected_labels),
        "min_cell_voxels": int(comp_volume_arr.min()) if len(comp_volume_arr) else 0,
        "median_cell_voxels": float(np.median(comp_volume_arr)) if len(comp_volume_arr) else 0.0,
        "max_cell_voxels": int(comp_volume_arr.max()) if len(comp_volume_arr) else 0,
    }
    return comp, comp_site_arr, comp_volume_arr, audit


def write_fv_cells(root: str | Path | None = None) -> dict[str, object]:
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    site_data = np.load(out_dir("sites") / "prescribed_sites.npz")
    ownership = np.load(out_dir("ownership") / "graph_geodesic_ownership.npz")
    comp, comp_site, comp_volume, audit = split_face_connected_cells(
        pore, ownership["labels"], len(site_data["sites"])
    )
    out = out_dir("fv")
    np.savez_compressed(
        out / "fv_cells.npz",
        cell_id=comp,
        cell_site=comp_site,
        cell_volume=comp_volume,
    )
    save_json(out / "fv_cells_audit.json", audit)
    return audit
