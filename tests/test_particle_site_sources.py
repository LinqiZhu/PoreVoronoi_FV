from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hybrid_site_sources import load_particle_window_seed_flat


def _write_particle_window(path: Path) -> None:
    fields = ["frame_id", "particle_id", "z_voxel", "y_voxel", "x_mod_voxel"]
    rows = [
        {"frame_id": 0, "particle_id": 0, "z_voxel": 0, "y_voxel": 0, "x_mod_voxel": 0},
        {"frame_id": 1, "particle_id": 0, "z_voxel": 0, "y_voxel": 0, "x_mod_voxel": 1},
        {"frame_id": 0, "particle_id": 1, "z_voxel": 0, "y_voxel": 0, "x_mod_voxel": 2},
        {"frame_id": 1, "particle_id": 1, "z_voxel": 0, "y_voxel": 0, "x_mod_voxel": 3},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_particle_id_selection_keeps_nested_track_prefix() -> None:
    source = Path(__file__).with_name("_particle_window_test.csv")
    try:
        _write_particle_window(source)
        mask = np.ones((1, 1, 4), dtype=bool)

        all_sites, all_metadata = load_particle_window_seed_flat(mask, source)
        prefix_sites, prefix_metadata = load_particle_window_seed_flat(
            mask,
            source,
            particle_id_selection="0",
        )

        assert all_sites.tolist() == [0, 1, 2, 3]
        assert prefix_sites.tolist() == [0, 1]
        assert all_metadata["particle_ids"] == "all"
        assert prefix_metadata["particle_ids"] == "0"
        assert prefix_metadata["particle_rows_selected"] == 2
    finally:
        source.unlink(missing_ok=True)
