from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import measure, transform

from .connectivity import largest_component


def make_3d_pore(
    mask2d: np.ndarray,
    nx: int = 260,
    ny: int = 82,
    nz: int = 30,
    shear_voxels: int = 15,
) -> tuple[np.ndarray, dict[str, object]]:
    """Extrude a 2D text pore into a slanted 3D voxel pore."""
    resized = transform.resize(
        mask2d.astype(float),
        (ny, nx),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ) > 0.35
    pore = np.zeros((nz, ny, nx), dtype=bool)
    for z in range(nz):
        shift = int(round((z / max(nz - 1, 1) - 0.5) * shear_voxels))
        pore[z] = np.roll(resized, shift=shift, axis=1)
        if shift > 0:
            pore[z, :, :shift] = False
        elif shift < 0:
            pore[z, :, shift:] = False

    pore = ndimage.binary_closing(pore, iterations=1)
    pore, stats = largest_component(pore, connectivity=1)
    spacing = (1.0 / max(nz - 1, 1), 1.0 / max(ny - 1, 1), 4.2 / max(nx - 1, 1))
    audit = {
        "shape_zyx": [int(v) for v in pore.shape],
        "spacing_zyx": [float(v) for v in spacing],
        "shear_voxels": int(shear_voxels),
        "porosity_3d": float(pore.mean()),
        "pore_voxels": int(pore.sum()),
        **stats,
    }
    return pore.astype(bool), audit


def write_obj_from_mask(mask: np.ndarray, path: str | Path, spacing=(1.0, 1.0, 1.0)) -> dict[str, int]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    vol = mask.astype(float)
    verts, faces, _normals, _values = measure.marching_cubes(vol, level=0.5, spacing=spacing)
    with p.open("w", encoding="utf-8") as f:
        f.write("# PoreVoronoi typographic pore surface\n")
        for z, y, x in verts:
            f.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for tri in faces + 1:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    return {"vertices": int(len(verts)), "faces": int(len(faces))}

