from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


ParticleXBoundaryMode = Literal["periodic_wrap", "clip"]


def _map_rounded_particle_coordinates(
    rounded: np.ndarray,
    shape: tuple[int, int, int],
    x_boundary_mode: ParticleXBoundaryMode,
) -> np.ndarray:
    mapped = np.asarray(rounded, dtype=np.int64).copy()
    mapped[..., 0] = np.clip(mapped[..., 0], 0, shape[0] - 1)
    mapped[..., 1] = np.clip(mapped[..., 1], 0, shape[1] - 1)
    if x_boundary_mode == "periodic_wrap":
        mapped[..., 2] = np.mod(mapped[..., 2], shape[2])
    elif x_boundary_mode == "clip":
        mapped[..., 2] = np.clip(mapped[..., 2], 0, shape[2] - 1)
    else:
        raise ValueError(f"Unknown particle x-boundary mode: {x_boundary_mode!r}")
    return mapped


def parse_integer_selection(spec: str, *, label: str = "integer") -> set[int] | None:
    """Parse comma-separated integer ids and inclusive ranges."""
    text = str(spec or "").strip()
    if not text:
        return None

    frames: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            frames.add(int(item))
            continue

        parts = [int(value) for value in item.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1 if stop >= start else -1
        elif len(parts) == 3:
            start, stop, step = parts
            if step == 0:
                raise ValueError(f"{label.capitalize()}-range step cannot be zero")
        else:
            raise ValueError(f"Bad {label} range: {item!r}")
        if (stop - start) * step < 0:
            raise ValueError(f"{label.capitalize()}-range step has the wrong sign: {item!r}")
        frames.update(range(start, stop + (1 if step > 0 else -1), step))
    if not frames:
        raise ValueError(f"{label.capitalize()} selection is empty")
    return frames


def parse_frame_selection(spec: str) -> set[int] | None:
    """Parse comma-separated frame ids and inclusive ranges."""
    return parse_integer_selection(spec, label="frame")


def _open_particle_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def validate_seed_flat(mask: np.ndarray, seed_flat: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"Expected a three-dimensional mask, got {mask_bool.shape}")
    seeds = np.unique(np.asarray(seed_flat, dtype=np.int64).reshape(-1))
    if seeds.size == 0:
        raise ValueError("The prescribed site set is empty")
    if int(seeds[0]) < 0 or int(seeds[-1]) >= mask_bool.size:
        raise ValueError("At least one prescribed site lies outside the mask array")
    if not bool(np.all(mask_bool.reshape(-1)[seeds])):
        raise ValueError("Every prescribed site must lie on a pore voxel")
    return seeds


def _snap_unique_rounded_sites(mask: np.ndarray, rounded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    snapped = rounded.copy()
    is_pore = mask[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
    displacement = np.zeros(rounded.shape[0], dtype=np.float64)
    bad = np.flatnonzero(~is_pore)
    if bad.size == 0:
        return snapped, displacement

    pore_zyx = np.argwhere(mask).astype(np.int64, copy=False)
    if pore_zyx.size == 0:
        raise ValueError("Cannot snap particle sites because the mask contains no pore voxels")
    tree = cKDTree(pore_zyx.astype(np.float64))
    nearest_distance, _nearest_index = tree.query(rounded[bad].astype(np.float64), k=1)
    for local_index, row_index in enumerate(bad):
        point = rounded[row_index]
        radius = float(nearest_distance[local_index]) + 1.0e-12
        candidate_index = np.asarray(tree.query_ball_point(point.astype(np.float64), radius), dtype=np.int64)
        candidates = pore_zyx[candidate_index]
        distance2 = np.sum((candidates - point[None, :]) ** 2, axis=1)
        minimum = int(np.min(distance2))
        tied = candidates[distance2 == minimum]
        order = np.lexsort((tied[:, 2], tied[:, 1], tied[:, 0]))
        chosen = tied[int(order[0])]
        snapped[row_index] = chosen
        displacement[row_index] = float(np.sqrt(minimum))
    return snapped, displacement


def load_particle_window_seed_flat(
    mask: np.ndarray,
    path: Path,
    *,
    frame_selection: str = "",
    particle_id_selection: str = "",
    max_snap_displacement_vox: float | None = None,
    x_boundary_mode: ParticleXBoundaryMode = "periodic_wrap",
    coordinate_center_offset_vox: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load continuous particle coordinates, snap them to pore voxels, and deduplicate sites."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"Expected a three-dimensional mask, got {mask_bool.shape}")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    selected_frames = parse_frame_selection(frame_selection)
    selected_particle_ids = parse_integer_selection(particle_id_selection, label="particle id")
    coordinate_offset = float(coordinate_center_offset_vox)
    if not np.isfinite(coordinate_offset):
        raise ValueError("coordinate_center_offset_vox must be finite")
    rounded_sites: set[tuple[int, int, int]] = set()
    rows_scanned = 0
    rows_selected = 0
    clipped_rows = 0
    wrapped_x_rows = 0
    solid_rows = 0
    d, h, w = mask_bool.shape

    with _open_particle_text(source) as handle:
        reader = csv.DictReader(handle)
        required = {"z_voxel", "y_voxel", "x_mod_voxel"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Particle window is missing coordinate columns: {sorted(missing)}")
        if selected_frames is not None and "frame_id" not in (reader.fieldnames or []):
            raise ValueError("A frame selection was requested, but the particle window has no frame_id column")
        if selected_particle_ids is not None and "particle_id" not in (reader.fieldnames or []):
            raise ValueError(
                "A particle-id selection was requested, but the particle window has no particle_id column"
            )

        for row_number, row in enumerate(reader, start=2):
            rows_scanned += 1
            if selected_frames is not None and int(row["frame_id"]) not in selected_frames:
                continue
            if selected_particle_ids is not None and int(row["particle_id"]) not in selected_particle_ids:
                continue
            rows_selected += 1
            try:
                coordinate = np.asarray(
                    [float(row["z_voxel"]), float(row["y_voxel"]), float(row["x_mod_voxel"])],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid particle coordinate at CSV row {row_number}") from exc
            if not bool(np.all(np.isfinite(coordinate))):
                raise ValueError(f"Non-finite particle coordinate at CSV row {row_number}")

            rounded = np.rint(coordinate - coordinate_offset).astype(np.int64)
            mapped = _map_rounded_particle_coordinates(rounded, mask_bool.shape, x_boundary_mode)
            if rounded[0] != mapped[0] or rounded[1] != mapped[1]:
                clipped_rows += 1
            if rounded[2] != mapped[2]:
                wrapped_x_rows += 1
            if not bool(mask_bool[tuple(mapped)]):
                solid_rows += 1
            rounded_sites.add((int(mapped[0]), int(mapped[1]), int(mapped[2])))

    if rows_selected == 0:
        raise ValueError("No particle rows matched the requested frame selection")

    rounded_unique = np.asarray(sorted(rounded_sites), dtype=np.int64)
    snapped, displacement = _snap_unique_rounded_sites(mask_bool, rounded_unique)
    if max_snap_displacement_vox is not None:
        limit = float(max_snap_displacement_vox)
        if limit < 0.0 or not np.isfinite(limit):
            raise ValueError("max_snap_displacement_vox must be finite and non-negative")
        if float(np.max(displacement)) > limit + 1.0e-12:
            raise ValueError(
                "Particle-site snapping exceeded the declared displacement limit: "
                f"maximum={float(np.max(displacement)):.12g}, limit={limit:.12g} voxels"
            )
    seed_flat = np.ravel_multi_index(snapped.T, mask_bool.shape).astype(np.int64, copy=False)
    seed_flat = validate_seed_flat(mask_bool, seed_flat)
    metadata: dict[str, object] = {
        "particle_window_path": str(source),
        "particle_frames": "all" if selected_frames is None else ",".join(map(str, sorted(selected_frames))),
        "particle_ids": "all" if selected_particle_ids is None else str(particle_id_selection),
        "particle_rows_scanned": int(rows_scanned),
        "particle_rows_selected": int(rows_selected),
        "particle_unique_rounded_sites": int(rounded_unique.shape[0]),
        "particle_unique_snapped_sites": int(seed_flat.size),
        "particle_duplicate_snapped_rows": int(rows_selected - seed_flat.size),
        "particle_clipped_coordinate_rows": int(clipped_rows),
        "particle_x_boundary_adjusted_rows": int(wrapped_x_rows),
        "particle_periodic_x_wrapped_rows": (
            int(wrapped_x_rows) if x_boundary_mode == "periodic_wrap" else 0
        ),
        "particle_clipped_x_rows": int(wrapped_x_rows) if x_boundary_mode == "clip" else 0,
        "particle_x_boundary_mode": str(x_boundary_mode),
        "particle_coordinate_center_offset_vox": coordinate_offset,
        "particle_solid_after_round_rows": int(solid_rows),
        "particle_snap_nonzero_unique_rounded": int(np.count_nonzero(displacement)),
        "particle_snap_displacement_median_vox": float(np.median(displacement)),
        "particle_snap_displacement_p95_vox": float(np.percentile(displacement, 95)),
        "particle_snap_displacement_max_vox": float(np.max(displacement)),
        "particle_snap_displacement_limit_vox": (
            "unbounded" if max_snap_displacement_vox is None else float(max_snap_displacement_vox)
        ),
    }
    return seed_flat, metadata


def load_particle_window_cell_states(
    mask: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    frame_selection: str = "",
    particle_id_selection: str = "",
    chunk_rows: int = 250_000,
    max_snap_displacement_vox: float | None = None,
    x_boundary_mode: ParticleXBoundaryMode = "periodic_wrap",
    coordinate_center_offset_vox: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Stream particle velocities into one arithmetic-mean state per ownership cell."""
    mask_bool = np.asarray(mask, dtype=bool)
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.shape != mask_bool.shape:
        raise ValueError(f"Label shape {label_array.shape} does not match mask shape {mask_bool.shape}")
    n_cells = int(label_array.max()) + 1
    if n_cells <= 0:
        raise ValueError("Ownership labels contain no cells")
    selected_frames = parse_frame_selection(frame_selection)
    selected_particle_ids = parse_integer_selection(particle_id_selection, label="particle id")
    coordinate_offset = float(coordinate_center_offset_vox)
    if not np.isfinite(coordinate_offset):
        raise ValueError("coordinate_center_offset_vox must be finite")
    source = Path(path)
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    if not source.is_file():
        raise FileNotFoundError(source)

    counts = np.zeros(n_cells, dtype=np.int64)
    sums = np.zeros((n_cells, 3), dtype=np.float64)
    coordinate_chunk: list[tuple[float, float, float]] = []
    velocity_chunk: list[tuple[float, float, float]] = []
    rounded_unique_flats: set[int] = set()
    snapped_unique_flats: set[int] = set()
    snap_displacement_by_rounded_flat: dict[int, float] = {}
    rows_scanned = 0
    rows_selected = 0
    clipped_rows = 0
    wrapped_x_rows = 0
    solid_rows = 0
    velocity_columns: tuple[str, str, str] | None = None

    def flush_chunk() -> None:
        nonlocal clipped_rows, wrapped_x_rows, solid_rows
        if not coordinate_chunk:
            return
        coordinate_array = np.asarray(coordinate_chunk, dtype=np.float64)
        velocity_array = np.asarray(velocity_chunk, dtype=np.float64)
        raw_rounded = np.rint(coordinate_array - coordinate_offset).astype(np.int64)
        rounded = _map_rounded_particle_coordinates(raw_rounded, mask_bool.shape, x_boundary_mode)
        clipped_rows += int(
            np.count_nonzero(np.any(raw_rounded[:, :2] != rounded[:, :2], axis=1))
        )
        wrapped_x_rows += int(np.count_nonzero(raw_rounded[:, 2] != rounded[:, 2]))
        row_is_pore = mask_bool[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
        solid_rows += int(np.count_nonzero(~row_is_pore))

        rounded_unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
        snapped_unique, displacement = _snap_unique_rounded_sites(mask_bool, rounded_unique)
        if max_snap_displacement_vox is not None:
            limit = float(max_snap_displacement_vox)
            if limit < 0.0 or not np.isfinite(limit):
                raise ValueError("max_snap_displacement_vox must be finite and non-negative")
            if float(np.max(displacement)) > limit + 1.0e-12:
                raise ValueError(
                    "Particle-state snapping exceeded the declared displacement limit: "
                    f"maximum={float(np.max(displacement)):.12g}, limit={limit:.12g} voxels"
                )
        rounded_flat = np.ravel_multi_index(rounded_unique.T, mask_bool.shape).astype(np.int64)
        snapped_flat = np.ravel_multi_index(snapped_unique.T, mask_bool.shape).astype(np.int64)
        rounded_unique_flats.update(int(value) for value in rounded_flat)
        snapped_unique_flats.update(int(value) for value in snapped_flat)
        for flat, value in zip(rounded_flat, displacement, strict=True):
            if value > 0.0:
                snap_displacement_by_rounded_flat[int(flat)] = float(value)

        snapped = snapped_unique[inverse]
        cell = label_array[snapped[:, 0], snapped[:, 1], snapped[:, 2]].astype(np.int64)
        if np.any(cell < 0):
            raise RuntimeError("At least one snapped particle state has no ownership cell")
        counts[:] += np.bincount(cell, minlength=n_cells).astype(np.int64)
        for component_index in range(3):
            sums[:, component_index] += np.bincount(
                cell,
                weights=velocity_array[:, component_index],
                minlength=n_cells,
            )
        coordinate_chunk.clear()
        velocity_chunk.clear()

    with _open_particle_text(source) as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for candidate in (
            ("ux_dns_frame_interp", "uy_dns_frame_interp", "uz_dns_frame_interp"),
            ("ux_dns_cell", "uy_dns_cell", "uz_dns_cell"),
            ("Ux", "Uy", "Uz"),
        ):
            if all(column in fields for column in candidate):
                velocity_columns = candidate
                break
        if velocity_columns is None:
            raise ValueError("Particle window contains no supported three-component velocity columns")
        required = {"z_voxel", "y_voxel", "x_mod_voxel"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Particle window is missing coordinate columns: {sorted(missing)}")
        if selected_frames is not None and "frame_id" not in fields:
            raise ValueError("A frame selection was requested, but the particle window has no frame_id column")
        if selected_particle_ids is not None and "particle_id" not in fields:
            raise ValueError(
                "A particle-id selection was requested, but the particle window has no particle_id column"
            )

        for row_number, row in enumerate(reader, start=2):
            rows_scanned += 1
            if selected_frames is not None and int(row["frame_id"]) not in selected_frames:
                continue
            if selected_particle_ids is not None and int(row["particle_id"]) not in selected_particle_ids:
                continue
            try:
                coordinate = tuple(float(row[column]) for column in ("z_voxel", "y_voxel", "x_mod_voxel"))
                velocity = tuple(float(row[column]) for column in velocity_columns)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid particle state at CSV row {row_number}") from exc
            if not all(np.isfinite(value) for value in coordinate + velocity):
                raise ValueError(f"Non-finite particle state at CSV row {row_number}")
            rows_selected += 1
            coordinate_chunk.append(coordinate)
            velocity_chunk.append(velocity)
            if len(coordinate_chunk) >= int(chunk_rows):
                flush_chunk()
        flush_chunk()

    if rows_selected == 0:
        raise ValueError("No particle states matched the requested frame selection")
    measured = counts > 0
    state = np.zeros((n_cells, 3), dtype=np.float64)
    state[measured] = sums[measured] / counts[measured, None]
    nonzero_displacement = np.asarray(list(snap_displacement_by_rounded_flat.values()), dtype=np.float64)
    metadata: dict[str, object] = {
        "state_particle_window_path": str(source),
        "state_particle_frames": "all" if selected_frames is None else ",".join(map(str, sorted(selected_frames))),
        "state_particle_ids": "all" if selected_particle_ids is None else str(particle_id_selection),
        "state_particle_rows_scanned": int(rows_scanned),
        "state_particle_rows": int(rows_selected),
        "state_velocity_columns": ",".join(velocity_columns),
        "state_measured_cells": int(np.count_nonzero(measured)),
        "state_measured_cell_fraction": float(np.mean(measured)),
        "state_unmeasured_cells": int(np.count_nonzero(~measured)),
        "state_unique_rounded_sites": int(len(rounded_unique_flats)),
        "state_unique_snapped_sites": int(len(snapped_unique_flats)),
        "state_clipped_coordinate_rows": int(clipped_rows),
        "state_x_boundary_adjusted_rows": int(wrapped_x_rows),
        "state_periodic_x_wrapped_rows": (
            int(wrapped_x_rows) if x_boundary_mode == "periodic_wrap" else 0
        ),
        "state_clipped_x_rows": int(wrapped_x_rows) if x_boundary_mode == "clip" else 0,
        "state_x_boundary_mode": str(x_boundary_mode),
        "state_coordinate_center_offset_vox": coordinate_offset,
        "state_solid_after_round_rows": int(solid_rows),
        "state_snap_nonzero_unique_rounded": int(nonzero_displacement.size),
        "state_snap_displacement_median_vox": (
            float(np.median(nonzero_displacement)) if nonzero_displacement.size else 0.0
        ),
        "state_snap_displacement_p95_vox": (
            float(np.percentile(nonzero_displacement, 95)) if nonzero_displacement.size else 0.0
        ),
        "state_snap_displacement_max_vox": (
            float(np.max(nonzero_displacement)) if nonzero_displacement.size else 0.0
        ),
        "state_snap_displacement_limit_vox": (
            "unbounded" if max_snap_displacement_vox is None else float(max_snap_displacement_vox)
        ),
        "state_stream_chunk_rows": int(chunk_rows),
        "state_representation": "one_arithmetic_mean_velocity_per_cell",
        "state_interior_gradient_reconstruction": "none",
    }
    return measured, state, counts, metadata


def harmonic_complete_constant_cell_states(
    owner: np.ndarray,
    neigh: np.ndarray,
    conductance: np.ndarray,
    measured: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Complete unmeasured constant cell states by an exact weighted graph-harmonic solve."""
    owner_array = np.asarray(owner, dtype=np.int64).reshape(-1)
    neigh_array = np.asarray(neigh, dtype=np.int64).reshape(-1)
    weight = np.asarray(conductance, dtype=np.float64).reshape(-1)
    measured_array = np.asarray(measured, dtype=bool).reshape(-1)
    state_array = np.asarray(state, dtype=np.float64)
    n_cells = measured_array.size
    if state_array.shape != (n_cells, 3):
        raise ValueError(f"Expected state shape {(n_cells, 3)}, got {state_array.shape}")
    if not (owner_array.size == neigh_array.size == weight.size):
        raise ValueError("Owner, neighbour, and conductance arrays must have equal length")
    if np.any(owner_array < 0) or np.any(neigh_array < 0):
        raise ValueError("Cell graph contains a negative cell index")
    if np.any(owner_array >= n_cells) or np.any(neigh_array >= n_cells):
        raise ValueError("Cell graph contains an out-of-range cell index")
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("Graph-harmonic conductances must be finite and strictly positive")
    if not np.any(measured_array):
        raise ValueError("Graph-harmonic completion requires at least one measured cell")

    unmeasured = ~measured_array
    if not np.any(unmeasured):
        return state_array.copy(), {
            "state_harmonic_completion": "not_needed",
            "state_harmonic_completed_cells": 0,
            "state_harmonic_residual_inf": 0.0,
        }

    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * weight.size, dtype=np.float64),
            (
                np.concatenate([owner_array, neigh_array]),
                np.concatenate([neigh_array, owner_array]),
            ),
        ),
        shape=(n_cells, n_cells),
    ).tocsr()
    component_count, labels = connected_components(adjacency, directed=False, return_labels=True)
    measured_components = np.bincount(
        labels[measured_array],
        minlength=component_count,
    )
    missing_component = np.flatnonzero(measured_components == 0)
    if missing_component.size:
        raise ValueError(
            "Every connected cell-graph component needs at least one measured particle state; "
            f"missing components={missing_component.tolist()}"
        )

    rows = np.concatenate([owner_array, neigh_array, owner_array, neigh_array])
    cols = np.concatenate([owner_array, neigh_array, neigh_array, owner_array])
    values = np.concatenate([weight, weight, -weight, -weight])
    laplacian = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(n_cells, n_cells),
    ).tocsr()
    unknown_index = np.flatnonzero(unmeasured)
    measured_index = np.flatnonzero(measured_array)
    matrix = laplacian[unknown_index][:, unknown_index].tocsc()
    coupling = laplacian[unknown_index][:, measured_index].tocsr()
    rhs = -(coupling @ state_array[measured_index])
    completed = state_array.copy()
    completed_unknown = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
    if completed_unknown.ndim == 1:
        completed_unknown = completed_unknown[:, None]
    if completed_unknown.shape != (unknown_index.size, 3):
        raise RuntimeError(
            f"Graph-harmonic solve returned shape {completed_unknown.shape}, expected {(unknown_index.size, 3)}"
        )
    if not np.all(np.isfinite(completed_unknown)):
        raise RuntimeError("Graph-harmonic completion returned non-finite cell states")
    completed[unknown_index] = completed_unknown
    residual = np.asarray(matrix @ completed_unknown - rhs)
    metadata: dict[str, object] = {
        "state_harmonic_completion": "exact_dirichlet_weighted_graph_laplacian",
        "state_harmonic_conductance": "facelet_sum_area_over_graph_geodesic_exchange_length",
        "state_harmonic_completed_cells": int(unknown_index.size),
        "state_harmonic_residual_inf": float(np.max(np.abs(residual))),
        "state_harmonic_component_count": int(component_count),
    }
    return completed, metadata


def auxiliary_grid_seed_flat(
    mask: np.ndarray,
    stride: tuple[int, int, int] | None,
    offset: tuple[int, int, int] | None = None,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if stride is None:
        return np.zeros(0, dtype=np.int64)
    stride_tuple = tuple(max(int(value), 1) for value in stride)
    if offset is None:
        offset_tuple = tuple(value // 2 for value in stride_tuple)
    else:
        offset_tuple = tuple(int(value) for value in offset)
    if any(value < 0 or value >= stride_tuple[index] for index, value in enumerate(offset_tuple)):
        raise ValueError(f"Auxiliary offset {offset_tuple} is outside stride {stride_tuple}")

    axes = [
        np.arange(offset_tuple[index], mask_bool.shape[index], stride_tuple[index], dtype=np.int64)
        for index in range(3)
    ]
    if any(axis.size == 0 for axis in axes):
        return np.zeros(0, dtype=np.int64)
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], axis=1)
    candidates = candidates[mask_bool[candidates[:, 0], candidates[:, 1], candidates[:, 2]]]
    if candidates.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.ravel_multi_index(candidates.T, mask_bool.shape).astype(np.int64))


def combine_particle_and_auxiliary_sites(
    mask: np.ndarray,
    particle_seed_flat: np.ndarray,
    *,
    auxiliary_stride: tuple[int, int, int] | None = None,
    auxiliary_offset: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    particles = validate_seed_flat(mask, particle_seed_flat)
    auxiliary_candidates = auxiliary_grid_seed_flat(mask, auxiliary_stride, auxiliary_offset)
    auxiliary_retained = np.setdiff1d(auxiliary_candidates, particles, assume_unique=True)
    combined = validate_seed_flat(mask, np.concatenate([particles, auxiliary_retained]))
    metadata: dict[str, object] = {
        "particle_site_count": int(particles.size),
        "auxiliary_candidate_count": int(auxiliary_candidates.size),
        "auxiliary_retained_count": int(auxiliary_retained.size),
        "total_site_count": int(combined.size),
        "auxiliary_site_rule": "none"
        if auxiliary_stride is None
        else f"explicit_regular_coverage_stride={tuple(auxiliary_stride)},offset={auxiliary_offset or 'half'}",
    }
    return combined, metadata


def load_seed_npz(mask: np.ndarray, path: Path, *, key: str = "seed_flat") -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{source} does not contain array {key!r}; available={data.files}")
        seeds = np.asarray(data[key], dtype=np.int64)
    return validate_seed_flat(mask, seeds)
