from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile as tiff


from _paths import REPO_ROOT, EXAMPLES_ROOT
ROOT = REPO_ROOT
DEFAULT_SOURCE = Path(os.environ.get("BENTHEIMER_SEGMENTED_TIF", "data/external/Segmented_Dry_Scan.tif"))
DEFAULT_OUT = EXAMPLES_ROOT / "segmented_masks"


def parse_zyx(text: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected three integers, e.g. 1562,59,349")
    return tuple(vals)  # type: ignore[return-value]


def face_structure_3d() -> np.ndarray:
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, :] = True
    st[1, :, 1] = True
    st[:, 1, 1] = True
    return st


def connectivity_summary(mask: np.ndarray) -> dict:
    labels, ncomp = ndimage.label(mask, structure=face_structure_3d())
    if ncomp == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "largest_component_fraction": 0.0,
            "span_z": False,
            "span_y": False,
            "span_x": False,
        }
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = labels == int(counts.argmax())
    return {
        "n_components": int(ncomp),
        "largest_component_voxels": int(largest.sum()),
        "largest_component_fraction": float(largest.sum() / max(int(mask.sum()), 1)),
        "span_z": bool(largest[0, :, :].any() and largest[-1, :, :].any()),
        "span_y": bool(largest[:, 0, :].any() and largest[:, -1, :].any()),
        "span_x": bool(largest[:, :, 0].any() and largest[:, :, -1].any()),
    }


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, ncomp = ndimage.label(mask, structure=face_structure_3d())
    if ncomp == 0:
        return mask & False
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--origin-zyx", type=parse_zyx, default=(1562, 59, 349))
    ap.add_argument("--shape-zyx", type=parse_zyx, default=(16, 96, 96))
    ap.add_argument("--pore-value", type=int, default=0)
    ap.add_argument("--name", default="bentheimer_dry_crop")
    ap.add_argument("--keep-largest-component", action="store_true", default=True)
    args = ap.parse_args()

    source = args.source.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    z0, y0, x0 = args.origin_zyx
    dz, dy, dx = args.shape_zyx

    mm = tiff.memmap(str(source))
    crop = np.asarray(mm[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx])
    if crop.shape != (dz, dy, dx):
        raise ValueError(f"Crop shape mismatch: got {crop.shape}, expected {(dz, dy, dx)}")

    raw_mask = crop == int(args.pore_value)
    mask = largest_component(raw_mask) if args.keep_largest_component else raw_mask
    summary = connectivity_summary(mask)
    meta = {
        "case_name": args.name,
        "source_path": str(source),
        "source_shape_zyx": [int(v) for v in mm.shape],
        "origin_zyx": [int(v) for v in args.origin_zyx],
        "shape_zyx": [int(v) for v in args.shape_zyx],
        "pore_value": int(args.pore_value),
        "mask_dtype": "bool",
        "raw_N_fl": int(raw_mask.sum()),
        "raw_porosity": float(raw_mask.mean()),
        "keep_largest_component": bool(args.keep_largest_component),
        "N_fl": int(mask.sum()),
        "porosity": float(mask.mean()),
        "connectivity": summary,
        "interpretation": (
            "Segmented Bentheimer dry-scan crop; voxels equal to pore_value are "
            "used as traversable pore space. For permeability production, only "
            "the largest face-connected pore component is retained. The selected "
            "component spans the x, y, and z directions."
        ),
    }
    stem = (
        f"{args.name}_{dz}x{dy}x{dx}_"
        f"origin_{z0}_{y0}_{x0}_pore{int(args.pore_value)}"
    )
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}_manifest.json"
    np.savez_compressed(
        npz_path,
        mask=mask.astype(np.bool_),
        raw_crop=crop.astype(np.uint8, copy=False),
        origin_zyx=np.asarray(args.origin_zyx, dtype=np.int32),
        shape_zyx=np.asarray(args.shape_zyx, dtype=np.int32),
        pore_value=np.asarray(int(args.pore_value), dtype=np.int32),
    )
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "manifest": str(json_path), **meta}, indent=2))


if __name__ == "__main__":
    main()



