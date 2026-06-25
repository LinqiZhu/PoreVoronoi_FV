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



def audit_stage2(root: str | Path) -> dict[str, object]:
    root = Path(root)
    required = {
        "sites": root / "outputs" / "sites" / "prescribed_sites.npz",
        "ownership": root / "outputs" / "ownership" / "graph_geodesic_ownership.npz",
        "cells": root / "outputs" / "fv" / "fv_cells.npz",
        "facelets": root / "outputs" / "fv" / "facelets.npz",
        "fv_graph": root / "outputs" / "fv" / "fv_graph.npz",
        "flux": root / "outputs" / "flux" / "conservative_flux_projection.npz",
        "cell_velocity": root / "outputs" / "velocity" / "cell_velocity.npz",
        "cell_velocity_figure": root / "outputs" / "velocity" / "Figure_PoreVoronoi_FV_cell_velocity.png",
        "four_panel": root / "outputs" / "final_figures" / "Figure_PoreVoronoi_FV_four_panel.png",
        "hero_dark": root / "outputs" / "final_figures" / "Figure_PoreVoronoi_README_landing_hero_dark.png",
        "hero_light": root / "outputs" / "final_figures" / "Figure_PoreVoronoi_README_landing_hero_light.png",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    details: dict[str, object] = {"missing": missing}
    passed = not missing
    if not missing:
        sites = np.load(required["sites"])
        ownership = np.load(required["ownership"])
        cells = np.load(required["cells"])
        facelets = np.load(required["facelets"])
        graph = np.load(required["fv_graph"])
        flux = np.load(required["flux"])
        velocity = np.load(required["cell_velocity"])
        pore = np.load(root / "outputs" / "masks" / "pore_mask_3d.npz")["mask"]
        div = flux["divergence_conservative"]
        details.update(
            {
                "sample_sites": int(len(sites["sites"])),
                "assigned_fraction": float(np.mean(ownership["labels"][pore] >= 0)),
                "finite_volume_cells": int(len(cells["cell_site"])),
                "cell_cell_facelets": int(len(facelets["owner"])),
                "fv_graph_edges": int(len(graph["area"])),
                "projection_edges": int(len(flux["q_target"])),
                "velocity_cells_with_faces": int(np.sum(velocity["cell_area_weight"] > 0)),
                "nonzero_velocity_cells": int(np.sum(velocity["cell_speed"] > 0)),
                "max_cell_speed": float(np.max(velocity["cell_speed"])) if len(velocity["cell_speed"]) else 0.0,
                "max_abs_cell_balance_after_projection": float(np.max(np.abs(div))) if len(div) else 0.0,
                "l2_cell_balance_after_projection": float(np.linalg.norm(div)) if len(div) else 0.0,
            }
        )
        passed = (
            details["sample_sites"] > 20
            and details["assigned_fraction"] == 1.0
            and details["finite_volume_cells"] >= details["sample_sites"]
            and details["cell_cell_facelets"] > 0
            and details["fv_graph_edges"] > 0
            and details["projection_edges"] >= details["fv_graph_edges"]
            and details["velocity_cells_with_faces"] > 0
            and details["nonzero_velocity_cells"] > 0
            and details["max_cell_speed"] > 0.0
            and details["max_abs_cell_balance_after_projection"] < 1.0e-4
        )
    report = {
        "stage": "stage2_porevoronoi_fv_typographic_demo",
        "passed": bool(passed),
        "details": details,
    }
    save_json(root / "outputs" / "reports" / "stage2_audit.json", report)
    return report
