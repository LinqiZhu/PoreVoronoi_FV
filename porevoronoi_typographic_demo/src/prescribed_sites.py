from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .io_utils import out_dir, save_json


def particles_to_sites(
    pore: np.ndarray,
    particles: np.ndarray,
    pressure: np.ndarray,
    speed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Snap sampled particles to pore voxels and return fixed site states."""
    pore = np.asarray(pore, dtype=bool)
    nz, ny, nx = pore.shape
    if particles.size == 0:
        raise ValueError("No particles were provided for prescribed-site construction.")

    nearest = ndimage.distance_transform_edt(~pore, return_distances=False, return_indices=True)
    rounded = np.rint(particles).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, nz - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, ny - 1)
    rounded[:, 2] = np.clip(rounded[:, 2], 0, nx - 1)
    snapped = nearest[:, rounded[:, 0], rounded[:, 1], rounded[:, 2]].T.astype(np.int32)

    seen: set[tuple[int, int, int]] = set()
    unique: list[np.ndarray] = []
    for coord in snapped:
        key = tuple(int(v) for v in coord)
        if key in seen:
            continue
        seen.add(key)
        unique.append(coord)
    sites = np.vstack(unique).astype(np.int32)

    state = np.zeros(
        len(sites),
        dtype=[("pressure", "f4"), ("speed", "f4"), ("kind", "U8")],
    )
    state["pressure"] = pressure[sites[:, 0], sites[:, 1], sites[:, 2]].astype(np.float32)
    state["speed"] = speed[sites[:, 0], sites[:, 1], sites[:, 2]].astype(np.float32)
    state["kind"] = "sample"

    audit = {
        "input_particles": int(len(particles)),
        "sample_sites": int(len(sites)),
        "duplicates_removed": int(len(particles) - len(sites)),
        "site_rule": "sampled particles snapped to nearest pore voxel",
        "pressure_min": float(state["pressure"].min()),
        "pressure_max": float(state["pressure"].max()),
        "speed_mean": float(state["speed"].mean()),
    }
    return sites, state, audit


def write_prescribed_sites(root: str | Path | None = None) -> dict[str, object]:
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    flow = np.load(out_dir("flow") / "pressure_flow.npz")
    particles = np.load(out_dir("particles") / "particles.npz")["particles"]
    sites, state, audit = particles_to_sites(pore, particles, flow["pressure"], flow["speed"])
    out = out_dir("sites")
    np.savez_compressed(out / "prescribed_sites.npz", sites=sites, state=state)
    save_json(out / "prescribed_sites_audit.json", audit)
    return audit
