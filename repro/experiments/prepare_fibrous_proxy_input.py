from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


from _paths import REPO_ROOT, EXAMPLES_ROOT
ROOT = REPO_ROOT
DEFAULT_SOURCE = EXAMPLES_ROOT / "review_synthetic_dataset.npz"
DEFAULT_OUT = EXAMPLES_ROOT / "segmented_masks"


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
    ap.add_argument("--size-yx", type=int, default=64)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--name", default="fibrous_filter_proxy")
    ap.add_argument("--keep-largest-component", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.source, allow_pickle=True)
    solid_like = np.asarray(data["fibrous_filter_mask"], dtype=bool)
    pore2d = ~solid_like
    size = int(args.size_yx)
    zoom = (size / pore2d.shape[0], size / pore2d.shape[1])
    pore2d_resized = ndimage.zoom(pore2d.astype(np.uint8), zoom, order=0).astype(bool)
    raw_mask = np.repeat(pore2d_resized[None, :, :], int(args.depth), axis=0)
    mask = largest_component(raw_mask) if args.keep_largest_component else raw_mask

    summary = connectivity_summary(mask)
    meta = {
        "case_name": args.name,
        "source_path": str(args.source.resolve()),
        "source_key": "fibrous_filter_mask",
        "source_interpretation": (
            "`fibrous_filter_mask` is treated as the solid/fiber pattern; its "
            "logical complement is the traversable pore phase."
        ),
        "source_shape_yx": [int(v) for v in solid_like.shape],
        "shape_zyx": [int(v) for v in mask.shape],
        "resize": "nearest-neighbor",
        "raw_N_fl": int(raw_mask.sum()),
        "raw_porosity": float(raw_mask.mean()),
        "keep_largest_component": bool(args.keep_largest_component),
        "N_fl": int(mask.sum()),
        "porosity": float(mask.mean()),
        "connectivity": summary,
        "interpretation": (
            "Fibrous-filter proxy derived from the review-dataset pattern. The "
            "largest face-connected pore component is retained for permeability "
            "production. This is a complex porous-material proxy, not a real-rock "
            "image case."
        ),
    }
    stem = f"{args.name}_{mask.shape[0]}x{mask.shape[1]}x{mask.shape[2]}_from_review_mask"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}_manifest.json"
    np.savez_compressed(
        npz_path,
        mask=mask.astype(np.bool_),
        source_mask_2d=solid_like.astype(np.bool_),
        pore_mask_2d_resized=pore2d_resized.astype(np.bool_),
    )
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "manifest": str(json_path), **meta}, indent=2))


if __name__ == "__main__":
    main()


