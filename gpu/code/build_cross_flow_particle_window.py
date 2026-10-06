from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AXES_ZYX = ("z", "y", "x")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask_record(path: Path) -> tuple[np.ndarray, tuple[int, int, int], str]:
    with np.load(path, allow_pickle=True) as data:
        mask = np.asarray(data["mask"], dtype=bool)
        permutation = tuple(int(value) for value in data["solver_axis_permutation_zyx"])
        drive_axis = str(data["original_drive_axis"].item())
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError(f"Invalid solver-axis permutation in {path}: {permutation}")
    return mask, permutation, drive_axis


def solver_to_physical_zyx(
    values: np.ndarray, permutation: tuple[int, int, int]
) -> np.ndarray:
    physical = np.empty_like(values)
    for solver_axis, physical_axis in enumerate(permutation):
        physical[..., physical_axis] = values[..., solver_axis]
    return physical


def physical_to_solver_zyx(
    values: np.ndarray, permutation: tuple[int, int, int]
) -> np.ndarray:
    return values[..., list(permutation)]


def solver_mask_to_physical(
    mask: np.ndarray, permutation: tuple[int, int, int]
) -> np.ndarray:
    return mask.transpose(tuple(int(value) for value in np.argsort(permutation)))


def transform_zyx(
    values: np.ndarray,
    source_permutation: tuple[int, int, int],
    target_permutation: tuple[int, int, int],
) -> np.ndarray:
    return physical_to_solver_zyx(
        solver_to_physical_zyx(values, source_permutation), target_permutation
    )


def transform_row(
    row: dict[str, str],
    source_permutation: tuple[int, int, int],
    target_permutation: tuple[int, int, int],
    source_shape: tuple[int, int, int],
) -> dict[str, str]:
    transformed = dict(row)
    source_position_zyx = np.asarray(
        [float(row["z_voxel"]), float(row["y_voxel"]), float(row["x_mod_voxel"])],
        dtype=np.float64,
    )
    source_site_zyx = map_rounded_site(source_position_zyx, source_shape)
    target_position_zyx = transform_zyx(
        source_site_zyx, source_permutation, target_permutation
    )
    transformed["z_voxel"] = repr(float(target_position_zyx[0]))
    transformed["y_voxel"] = repr(float(target_position_zyx[1]))
    transformed["x_mod_voxel"] = repr(float(target_position_zyx[2]))
    transformed["x_unwrapped_voxel"] = transformed["x_mod_voxel"]

    velocity_columns = ("uz_dns_frame_interp", "uy_dns_frame_interp", "ux_dns_frame_interp")
    if all(column in row and row[column] != "" for column in velocity_columns):
        source_velocity_zyx = np.asarray(
            [float(row[column]) for column in velocity_columns], dtype=np.float64
        )
        target_velocity_zyx = transform_zyx(
            source_velocity_zyx, source_permutation, target_permutation
        )
        for column, value in zip(velocity_columns, target_velocity_zyx, strict=True):
            transformed[column] = repr(float(value))
    return transformed


def map_rounded_site(
    coordinate_zyx: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    rounded = np.rint(coordinate_zyx).astype(np.int64)
    rounded[0] = np.clip(rounded[0], 0, shape[0] - 1)
    rounded[1] = np.clip(rounded[1], 0, shape[1] - 1)
    rounded[2] %= shape[2]
    return rounded


def rounded_site_set(
    path: Path, shape: tuple[int, int, int]
) -> set[tuple[int, int, int]]:
    sites: set[tuple[int, int, int]] = set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            coordinate = np.asarray(
                [float(row["z_voxel"]), float(row["y_voxel"]), float(row["x_mod_voxel"])],
                dtype=np.float64,
            )
            rounded = map_rounded_site(coordinate, shape)
            sites.add(tuple(int(value) for value in rounded))
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a frozen PTV window between solver-axis permutations."
    )
    parser.add_argument("--source-window", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True)
    parser.add_argument("--source-axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--target-axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--out-window", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.source_axis == args.target_axis:
        raise ValueError("Cross-flow conversion requires distinct source and target axes")
    source_mask, source_permutation, source_drive_axis = load_mask_record(args.source_mask)
    target_mask, target_permutation, target_drive_axis = load_mask_record(args.target_mask)
    if source_drive_axis != args.source_axis or target_drive_axis != args.target_axis:
        raise ValueError("Declared source/target axes do not match mask metadata")

    source_physical = solver_mask_to_physical(source_mask, source_permutation)
    target_physical = solver_mask_to_physical(target_mask, target_permutation)
    if not np.array_equal(source_physical, target_physical):
        raise ValueError("Source and target masks are not the same physical pore geometry")

    args.out_window.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with gzip.open(args.source_window, "rt", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Source particle window has no header")
        required = {"z_voxel", "y_voxel", "x_mod_voxel", "x_unwrapped_voxel"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Source particle window is missing {sorted(required)}")
        with gzip.open(args.out_window, "wt", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(
                    transform_row(
                        row,
                        source_permutation,
                        target_permutation,
                        tuple(int(value) for value in source_mask.shape),
                    )
                )
                row_count += 1

    source_sites = rounded_site_set(
        args.source_window, tuple(int(value) for value in source_mask.shape)
    )
    target_sites = rounded_site_set(
        args.out_window, tuple(int(value) for value in target_mask.shape)
    )
    expected_target_sites = {
        tuple(
            int(value)
            for value in transform_zyx(
                np.asarray(site, dtype=np.int64), source_permutation, target_permutation
            )
        )
        for site in source_sites
    }
    target_sites_in_pore = all(target_mask[site] for site in target_sites)
    if target_sites != expected_target_sites or not target_sites_in_pore:
        raise RuntimeError("Transformed site set failed coordinate or pore-mask verification")

    component_map = {
        AXES_ZYX[solver_axis]: AXES_ZYX[physical_axis]
        for solver_axis, physical_axis in enumerate(target_permutation)
    }
    manifest: dict[str, Any] = {
        "status": "PASS_FROZEN_CROSS_FLOW_WINDOW_TRANSFORM",
        "source_axis": args.source_axis,
        "target_axis": args.target_axis,
        "source_window": str(args.source_window.resolve()),
        "output_window": str(args.out_window.resolve()),
        "source_mask": str(args.source_mask.resolve()),
        "target_mask": str(args.target_mask.resolve()),
        "source_window_sha256": file_sha256(args.source_window),
        "output_window_sha256": file_sha256(args.out_window),
        "source_mask_sha256": file_sha256(args.source_mask),
        "target_mask_sha256": file_sha256(args.target_mask),
        "source_solver_axis_permutation_zyx": list(source_permutation),
        "target_solver_axis_permutation_zyx": list(target_permutation),
        "target_solver_to_physical_component_map": component_map,
        "same_physical_mask": True,
        "row_count": row_count,
        "source_unique_rounded_sites": len(source_sites),
        "target_unique_rounded_sites": len(target_sites),
        "target_sites_in_pore": target_sites_in_pore,
        "site_selection_changed": False,
        "particle_ids_or_frames_changed": False,
        "coordinate_representation": "source_runner_mapped_rounded_site_voxels",
        "velocity_values_used_by_forward_runner": False,
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
