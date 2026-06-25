from __future__ import annotations

import numpy as np


NEIGHBORS = np.array(
    [[0, 0, 1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, -1]], dtype=int
)


def trace_streamlines(
    pore: np.ndarray,
    pressure: np.ndarray,
    n_lines: int = 220,
    max_steps: int = 1600,
    seed: int = 20260625,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Trace graph-constrained streamlines downhill in pressure."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = pore.shape
    finite = pore & np.isfinite(pressure)
    candidates = np.argwhere(finite & (pressure > 0.86))
    if len(candidates) == 0:
        candidates = np.argwhere(finite)
    starts = candidates[rng.choice(len(candidates), size=min(n_lines, len(candidates)), replace=False)]

    all_points: list[np.ndarray] = []
    offsets = [0]
    reached = 0
    for start in starts:
        current = start.astype(int)
        pts = [current.astype(float)]
        visited = {tuple(current)}
        for _ in range(max_steps):
            z, y, x = current
            if x >= nx - 2 or pressure[z, y, x] <= 0.02:
                reached += 1
                break
            neigh = []
            for dz, dy, dx in NEIGHBORS:
                zz, yy, xx = z + dz, y + dy, x + dx
                if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                if not finite[zz, yy, xx] or (zz, yy, xx) in visited:
                    continue
                score = pressure[zz, yy, xx] - 0.015 * dx + rng.normal(0.0, 0.003)
                neigh.append((score, np.array([zz, yy, xx], dtype=int)))
            if not neigh:
                break
            neigh.sort(key=lambda item: item[0])
            next_cell = neigh[0][1]
            if pressure[tuple(next_cell)] > pressure[z, y, x] + 0.02:
                break
            current = next_cell
            visited.add(tuple(current))
            pts.append(current.astype(float))
        arr = np.vstack(pts)
        if len(arr) >= 5:
            all_points.append(arr)
            offsets.append(offsets[-1] + len(arr))

    if all_points:
        points = np.vstack(all_points).astype(np.float32)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    audit = {
        "requested_lines": int(n_lines),
        "saved_lines": int(len(offsets_arr) - 1),
        "points": int(points.shape[0]),
        "reached_outlet_like": int(reached),
        "seed": int(seed),
    }
    return points, offsets_arr, audit


def sample_particles_from_lines(
    points: np.ndarray,
    offsets: np.ndarray,
    n_particles: int = 360,
    seed: int = 20260625,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed + 17)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), {"particles": 0, "seed": int(seed + 17)}
    idx = rng.choice(len(points), size=min(n_particles, len(points)), replace=False)
    particles = points[idx].copy()
    particles += rng.normal(0.0, 0.18, size=particles.shape).astype(np.float32)
    return particles.astype(np.float32), {"particles": int(len(particles)), "seed": int(seed + 17), "rule": "from_streamline_points"}


def sample_particles_from_pore(
    pore: np.ndarray,
    pressure: np.ndarray | None = None,
    n_particles: int = 360,
    seed: int = 20260625,
    x_bins: int = 42,
) -> tuple[np.ndarray, dict[str, object]]:
    """Sample visual state particles throughout the connected typographic pore."""
    rng = np.random.default_rng(seed + 31)
    finite = pore.copy()
    if pressure is not None:
        finite &= np.isfinite(pressure)
    coords = np.argwhere(finite)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), {"particles": 0, "seed": int(seed + 31), "rule": "from_pore_voxels"}

    nz, ny, nx = pore.shape
    interior = coords[(coords[:, 2] > 1) & (coords[:, 2] < nx - 2)]
    if len(interior) == 0:
        interior = coords
    bins = np.linspace(interior[:, 2].min(), interior[:, 2].max() + 1, x_bins + 1)
    per_bin = max(2, int(np.ceil(n_particles / x_bins)))
    chosen: list[np.ndarray] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = interior[(interior[:, 2] >= lo) & (interior[:, 2] < hi)]
        if len(bucket) == 0:
            continue
        take = min(per_bin, len(bucket))
        idx = rng.choice(len(bucket), size=take, replace=False)
        chosen.append(bucket[idx])
    if chosen:
        particles = np.vstack(chosen)
    else:
        particles = interior
    if len(particles) > n_particles:
        idx = rng.choice(len(particles), size=n_particles, replace=False)
        particles = particles[idx]
    particles = particles.astype(np.float32)
    particles += rng.normal(0.0, 0.22, size=particles.shape).astype(np.float32)
    particles[:, 0] = np.clip(particles[:, 0], 0, nz - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, ny - 1)
    particles[:, 2] = np.clip(particles[:, 2], 0, nx - 1)
    return particles.astype(np.float32), {
        "particles": int(len(particles)),
        "seed": int(seed + 31),
        "rule": "stratified_from_connected_pore_voxels",
        "x_bins": int(x_bins),
    }
