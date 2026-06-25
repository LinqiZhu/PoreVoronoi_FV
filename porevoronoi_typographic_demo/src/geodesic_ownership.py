from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from .io_utils import out_dir, save_json


NEIGHBORS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int16
)


def multi_source_geodesic_ownership(
    pore: np.ndarray,
    sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compute graph-geodesic nearest-site ownership on the 6-neighbour pore graph."""
    pore = np.asarray(pore, dtype=bool)
    nz, ny, nx = pore.shape
    labels = np.full(pore.shape, -1, dtype=np.int32)
    distance = np.full(pore.shape, -1, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()

    for sid, (z, y, x) in enumerate(sites.astype(np.int64)):
        if not pore[z, y, x]:
            raise ValueError(f"Site {sid} is not inside the pore graph: {(int(z), int(y), int(x))}")
        if distance[z, y, x] == 0:
            continue
        labels[z, y, x] = sid
        distance[z, y, x] = 0
        queue.append((int(z), int(y), int(x)))

    while queue:
        z, y, x = queue.popleft()
        src_label = int(labels[z, y, x])
        src_dist = int(distance[z, y, x])
        nd = src_dist + 1
        for dz, dy, dx in NEIGHBORS:
            zz, yy, xx = z + int(dz), y + int(dy), x + int(dx)
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            if not pore[zz, yy, xx]:
                continue
            old_d = int(distance[zz, yy, xx])
            old_l = int(labels[zz, yy, xx])
            if old_d < 0 or nd < old_d or (nd == old_d and src_label < old_l):
                distance[zz, yy, xx] = nd
                labels[zz, yy, xx] = src_label
                queue.append((zz, yy, xx))

    missing = int((pore & (labels < 0)).sum())
    rho = 0
    edge_count = 0
    for dz, dy, dx in NEIGHBORS:
        src = pore.copy()
        dst = np.roll(pore, shift=(-int(dz), -int(dy), -int(dx)), axis=(0, 1, 2))
        valid = src & dst
        if dz == 1:
            valid[-1, :, :] = False
        if dz == -1:
            valid[0, :, :] = False
        if dy == 1:
            valid[:, -1, :] = False
        if dy == -1:
            valid[:, 0, :] = False
        if dx == 1:
            valid[:, :, -1] = False
        if dx == -1:
            valid[:, :, 0] = False
        du = distance[valid]
        dv = np.roll(distance, shift=(-int(dz), -int(dy), -int(dx)), axis=(0, 1, 2))[valid]
        if du.size:
            rho = max(rho, int(np.max(dv - du - 1)))
            edge_count += int(du.size)
    rho = max(0, rho)
    audit = {
        "sites": int(len(sites)),
        "assigned_voxels": int((pore & (labels >= 0)).sum()),
        "unassigned_voxels": missing,
        "max_graph_distance": int(distance[pore].max()),
        "rho_D": int(rho),
        "directed_edge_tests": int(edge_count),
        "ownership_rule": "multi-source 6-neighbour graph-geodesic BFS with lexicographic site-id tie break",
    }
    return labels, distance, audit


def write_ownership(root: str | Path | None = None) -> dict[str, object]:
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    site_data = np.load(out_dir("sites") / "prescribed_sites.npz")
    labels, distance, audit = multi_source_geodesic_ownership(pore, site_data["sites"])
    out = out_dir("ownership")
    np.savez_compressed(out / "graph_geodesic_ownership.npz", labels=labels, distance=distance)
    save_json(out / "graph_geodesic_ownership_audit.json", audit)
    return audit
