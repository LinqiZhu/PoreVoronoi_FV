from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_dns_metrics(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    row = pd.read_csv(source).iloc[0].to_dict()
    result: dict[str, Any] = {}
    for key, value in row.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            result[str(key)] = value
    return result


def aggregate_dns_cell_state_to_ownership(
    labels: np.ndarray,
    dns_cell_path: Path,
    *,
    n_cells: int,
    n_dns_cells: int,
    chunk_rows: int = 300_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Aggregate the one-voxel DNS state onto a supplied ownership partition."""
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim != 3:
        raise ValueError(f"Expected three-dimensional ownership labels, got {label_array.shape}")
    if int(n_cells) <= 0 or int(n_dns_cells) <= 0 or int(chunk_rows) <= 0:
        raise ValueError("Cell counts and chunk_rows must be positive")

    velocity_sum = np.zeros((n_cells, 3), dtype=np.float64)
    pressure_sum = np.zeros(n_cells, dtype=np.float64)
    ownership_count = np.zeros(n_cells, dtype=np.int64)
    dns_cell_to_owner = np.full(n_dns_cells, -1, dtype=np.int32)
    rows = 0
    usecols = ["cell_id", "x", "y", "z", "Ux", "Uy", "Uz", "p"]
    for chunk in pd.read_csv(Path(dns_cell_path), usecols=usecols, chunksize=int(chunk_rows)):
        cell_id = chunk["cell_id"].to_numpy(dtype=np.int64)
        if np.any(cell_id < 0) or np.any(cell_id >= n_dns_cells):
            raise ValueError("DNS cell_id lies outside the declared DNS-cell range")
        x = np.rint(chunk["x"].to_numpy(dtype=np.float64) - 0.5).astype(np.int64)
        y = np.rint(chunk["y"].to_numpy(dtype=np.float64) - 0.5).astype(np.int64)
        z = np.rint(chunk["z"].to_numpy(dtype=np.float64) - 0.5).astype(np.int64)
        if (
            np.any(z < 0)
            or np.any(z >= label_array.shape[0])
            or np.any(y < 0)
            or np.any(y >= label_array.shape[1])
            or np.any(x < 0)
            or np.any(x >= label_array.shape[2])
        ):
            raise ValueError("DNS cell coordinate lies outside the ownership array")
        owner_label = label_array[z, y, x].astype(np.int64, copy=False)
        if np.any(owner_label < 0) or np.any(owner_label >= n_cells):
            raise ValueError("At least one DNS pore cell has no valid ownership label")
        previous = dns_cell_to_owner[cell_id]
        if np.any((previous >= 0) & (previous != owner_label)):
            raise ValueError("A repeated DNS cell_id maps to inconsistent ownership cells")
        dns_cell_to_owner[cell_id] = owner_label.astype(np.int32, copy=False)

        velocity = chunk[["Ux", "Uy", "Uz"]].to_numpy(dtype=np.float64)
        pressure = chunk["p"].to_numpy(dtype=np.float64)
        np.add.at(ownership_count, owner_label, 1)
        for component in range(3):
            np.add.at(velocity_sum[:, component], owner_label, velocity[:, component])
        np.add.at(pressure_sum, owner_label, pressure)
        rows += int(chunk.shape[0])

    if rows != n_dns_cells or np.any(dns_cell_to_owner < 0):
        raise ValueError(
            "DNS cell-state coverage is incomplete: "
            f"rows={rows}, expected={n_dns_cells}, unmapped={int(np.count_nonzero(dns_cell_to_owner < 0))}"
        )
    if np.any(ownership_count == 0):
        raise ValueError(
            f"Ownership cells without a DNS reference state: {int(np.count_nonzero(ownership_count == 0))}"
        )
    velocity = velocity_sum / ownership_count[:, None]
    pressure = pressure_sum / ownership_count
    metadata = {
        "dns_cell_state_path": str(Path(dns_cell_path)),
        "dns_cell_state_rows": int(rows),
        "dns_reference_measured_cells": int(np.count_nonzero(ownership_count)),
        "dns_reference_min_voxels_per_cell": int(np.min(ownership_count)),
        "dns_reference_max_voxels_per_cell": int(np.max(ownership_count)),
        "dns_cell_chunk_rows": int(chunk_rows),
    }
    return velocity, pressure, dns_cell_to_owner, metadata


def aggregate_dns_face_flux_to_ownership(
    dns_cell_to_owner: np.ndarray,
    owner: np.ndarray,
    neigh: np.ndarray,
    dns_face_path: Path,
    *,
    n_cells: int,
    chunk_rows: int = 750_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate oriented voxel-DNS face fluxes onto ownership-cell edges."""
    coarse_owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    coarse_neigh = np.asarray(neigh, dtype=np.int64).reshape(-1)
    if coarse_owner.size != coarse_neigh.size:
        raise ValueError("Coarse owner and neighbour arrays must have equal length")
    low = np.minimum(coarse_owner, coarse_neigh)
    high = np.maximum(coarse_owner, coarse_neigh)
    stored_sign = np.where(coarse_owner == low, 1.0, -1.0)
    coarse_key = low * np.int64(n_cells) + high
    order = np.argsort(coarse_key)
    sorted_key = coarse_key[order]
    if np.any(sorted_key[1:] == sorted_key[:-1]):
        raise ValueError("Coarse ownership graph contains duplicate cell-pair edges")
    canonical_flux_sorted = np.zeros(sorted_key.size, dtype=np.float64)

    mapping = np.asarray(dns_cell_to_owner, dtype=np.int32).reshape(-1)
    rows = 0
    crossing_rows = 0
    unmatched_pairs = 0
    usecols = ["owner", "neigh", "phi"]
    for chunk in pd.read_csv(Path(dns_face_path), usecols=usecols, chunksize=int(chunk_rows)):
        fine_owner = chunk["owner"].to_numpy(dtype=np.int64)
        fine_neigh = chunk["neigh"].to_numpy(dtype=np.int64)
        if (
            np.any(fine_owner < 0)
            or np.any(fine_owner >= mapping.size)
            or np.any(fine_neigh < 0)
            or np.any(fine_neigh >= mapping.size)
        ):
            raise ValueError("DNS face owner/neighbour lies outside the DNS-cell map")
        mapped_owner = mapping[fine_owner].astype(np.int64, copy=False)
        mapped_neigh = mapping[fine_neigh].astype(np.int64, copy=False)
        valid = mapped_owner != mapped_neigh
        rows += int(chunk.shape[0])
        crossing_rows += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue
        mapped_owner = mapped_owner[valid]
        mapped_neigh = mapped_neigh[valid]
        fine_flux = chunk["phi"].to_numpy(dtype=np.float64)[valid]
        pair_low = np.minimum(mapped_owner, mapped_neigh)
        pair_high = np.maximum(mapped_owner, mapped_neigh)
        canonical_sign = np.where(mapped_owner == pair_low, 1.0, -1.0)
        key = pair_low * np.int64(n_cells) + pair_high
        unique_key, inverse = np.unique(key, return_inverse=True)
        summed_flux = np.bincount(
            inverse,
            weights=canonical_sign * fine_flux,
            minlength=unique_key.size,
        )
        position = np.searchsorted(sorted_key, unique_key)
        matched = (position < sorted_key.size) & (
            sorted_key[np.minimum(position, sorted_key.size - 1)] == unique_key
        )
        unmatched_pairs += int(np.count_nonzero(~matched))
        if np.any(matched):
            np.add.at(canonical_flux_sorted, position[matched], summed_flux[matched])

    if unmatched_pairs:
        raise ValueError(f"DNS flux aggregation found {unmatched_pairs} ownership pairs absent from the graph")
    canonical_flux = np.empty_like(canonical_flux_sorted)
    canonical_flux[order] = canonical_flux_sorted
    stored_flux = canonical_flux * stored_sign
    metadata = {
        "dns_face_flux_path": str(Path(dns_face_path)),
        "dns_face_flux_rows": int(rows),
        "dns_face_crossing_rows": int(crossing_rows),
        "dns_coarse_flux_edges": int(stored_flux.size),
        "dns_flux_unmatched_pair_count": int(unmatched_pairs),
        "dns_face_chunk_rows": int(chunk_rows),
    }
    return stored_flux, metadata
