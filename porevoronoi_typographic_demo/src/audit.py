from __future__ import annotations

from pathlib import Path

import numpy as np

from .connectivity import component_count
from .io_utils import load_json, save_json


def audit_stage1(root: str | Path) -> dict[str, object]:
    root = Path(root)
    required = {
        "mask2d": root / "outputs" / "masks" / "word_mask_2d.npz",
        "mask3d": root / "outputs" / "masks" / "pore_mask_3d.npz",
        "flow": root / "outputs" / "flow" / "pressure_flow.npz",
        "streamlines": root / "outputs" / "streamlines" / "streamlines.npz",
        "particles": root / "outputs" / "particles" / "particles.npz",
        "hero_light": root / "outputs" / "figures" / "Figure_PoreVoronoi_stage1_README_hero_light.png",
        "hero_dark": root / "outputs" / "figures" / "Figure_PoreVoronoi_stage1_README_hero_dark.png",
        "debug_panels": root / "outputs" / "figures" / "Figure_PoreVoronoi_stage1_debug_projection_panels.png",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    passed = not missing
    details: dict[str, object] = {"missing": missing}
    if not missing:
        mask2d = np.load(required["mask2d"])["mask"]
        mask3d = np.load(required["mask3d"])["mask"]
        flow = np.load(required["flow"])
        points = np.load(required["streamlines"])["points"]
        particles = np.load(required["particles"])["particles"]
        details.update(
            {
                "mask2d_components_4conn": component_count(mask2d, connectivity=1),
                "mask3d_components_6conn": component_count(mask3d, connectivity=1),
                "mask3d_shape_zyx": [int(v) for v in mask3d.shape],
                "pore_voxels": int(mask3d.sum()),
                "pressure_finite_fraction": float(np.isfinite(flow["pressure"][mask3d]).mean()),
                "streamline_points": int(points.shape[0]),
                "particles": int(particles.shape[0]),
            }
        )
        passed = (
            details["mask2d_components_4conn"] == 1
            and details["mask3d_components_6conn"] == 1
            and details["pressure_finite_fraction"] == 1.0
            and details["streamline_points"] > 100
            and details["particles"] > 20
        )

    report = {
        "stage": "stage1_typographic_pore_demo",
        "passed": bool(passed),
        "details": details,
    }
    save_json(root / "outputs" / "reports" / "stage1_audit.json", report)
    return report


def stage1_ready(root: str | Path) -> bool:
    report_path = Path(root) / "outputs" / "reports" / "stage1_audit.json"
    if not report_path.exists():
        return False
    return bool(load_json(report_path).get("passed", False))

