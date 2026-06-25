from __future__ import annotations

import numpy as np
from scipy import ndimage


def largest_component(mask: np.ndarray, connectivity: int = 1) -> tuple[np.ndarray, dict[str, int]]:
    """Return the largest connected component of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labels, nlab = ndimage.label(mask, structure=structure)
    if nlab == 0:
        return mask.copy(), {"components": 0, "kept_label": 0, "kept_voxels": 0}
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    kept = labels == keep
    return kept, {"components": int(nlab), "kept_label": keep, "kept_voxels": int(counts[keep])}


def component_count(mask: np.ndarray, connectivity: int = 1) -> int:
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    _, nlab = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return int(nlab)

