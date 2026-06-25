from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


NEIGHBORS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=int
)


def solve_pressure(mask: np.ndarray, rtol: float = 1e-8, maxiter: int = 4000) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve a simple resistor-network pressure field in the pore voxels."""
    pore = np.asarray(mask, dtype=bool)
    nz, ny, nx = pore.shape
    pore_coords = np.argwhere(pore)
    xmin = int(pore_coords[:, 2].min())
    xmax = int(pore_coords[:, 2].max())
    inlet = pore & (np.arange(nx)[None, None, :] == xmin)
    outlet = pore & (np.arange(nx)[None, None, :] == xmax)
    if not inlet.any() or not outlet.any():
        raise ValueError("Pore mask must expose both left and right pore sections.")

    fixed = inlet | outlet
    unknown = pore & ~fixed
    unknown_id = -np.ones(pore.shape, dtype=np.int64)
    coords = np.argwhere(unknown)
    unknown_id[unknown] = np.arange(len(coords))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(len(coords), dtype=float)

    for row, (z, y, x) in enumerate(coords):
        diag = 0.0
        for dz, dy, dx in NEIGHBORS:
            zz, yy, xx = z + dz, y + dy, x + dx
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            if not pore[zz, yy, xx]:
                continue
            diag += 1.0
            if inlet[zz, yy, xx]:
                rhs[row] += 1.0
            elif outlet[zz, yy, xx]:
                rhs[row] += 0.0
            else:
                col = unknown_id[zz, yy, xx]
                if col >= 0:
                    rows.append(row)
                    cols.append(int(col))
                    data.append(-1.0)
        rows.append(row)
        cols.append(row)
        data.append(diag)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(coords), len(coords)))
    sol, info = spla.cg(mat, rhs, rtol=rtol, maxiter=maxiter)
    method = "cg"
    if info != 0:
        sol = spla.spsolve(mat, rhs)
        method = "spsolve_after_cg"

    pressure = np.full(pore.shape, np.nan, dtype=np.float32)
    pressure[inlet] = 1.0
    pressure[outlet] = 0.0
    pressure[unknown] = sol.astype(np.float32)

    velocity = np.zeros(pore.shape + (3,), dtype=np.float32)
    for dz, dy, dx in NEIGHBORS:
        shifted = np.roll(pressure, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
        valid = pore & np.roll(pore, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
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
        flux = np.zeros_like(pressure, dtype=np.float32)
        flux[valid] = np.maximum(pressure[valid] - shifted[valid], 0)
        velocity[..., 0] += flux * dx
        velocity[..., 1] += flux * dy
        velocity[..., 2] += flux * dz

    speed = np.linalg.norm(velocity, axis=-1)
    finite = np.isfinite(pressure) & pore
    audit = {
        "solver": method,
        "cg_info": int(info),
        "unknowns": int(len(coords)),
        "inlet_voxels": int(inlet.sum()),
        "outlet_voxels": int(outlet.sum()),
        "pressure_min": float(np.nanmin(pressure[finite])),
        "pressure_max": float(np.nanmax(pressure[finite])),
        "speed_mean": float(speed[pore].mean()),
        "speed_p95": float(np.percentile(speed[pore], 95)),
    }
    return pressure, speed.astype(np.float32), audit


