from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io_utils import out_dir, save_json
from src.pore_geometry import make_3d_pore, write_obj_from_mask


def main() -> None:
    mask2d = np.load(out_dir("masks") / "word_mask_2d.npz")["mask"]
    pore, audit = make_3d_pore(mask2d)
    mask_dir = out_dir("masks")
    mesh_dir = out_dir("meshes")
    np.savez_compressed(mask_dir / "pore_mask_3d.npz", mask=pore)
    mesh_stats = write_obj_from_mask(pore, mesh_dir / "pore_surface.obj")
    audit.update(mesh_stats)
    save_json(mask_dir / "pore_mask_3d_audit.json", audit)
    print(f"3D pore mask saved: {mask_dir / 'pore_mask_3d.npz'}")
    print(audit)


if __name__ == "__main__":
    main()

