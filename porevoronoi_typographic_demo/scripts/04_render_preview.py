from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io_utils import out_dir
from src.rendering import render_debug_panels, render_readme_hero


def main() -> None:
    mask2d = np.load(out_dir("masks") / "word_mask_2d.npz")["mask"]
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    flow = np.load(out_dir("flow") / "pressure_flow.npz")
    streams = np.load(out_dir("streamlines") / "streamlines.npz")
    particles = np.load(out_dir("particles") / "particles.npz")["particles"]
    fig_dir = out_dir("figures")
    render_readme_hero(mask2d, pore.shape, streams["points"], streams["offsets"], particles, fig_dir / "Figure_PoreVoronoi_stage1_README_hero_light.png", dark=False)
    render_readme_hero(mask2d, pore.shape, streams["points"], streams["offsets"], particles, fig_dir / "Figure_PoreVoronoi_stage1_README_hero_dark.png", dark=True)
    render_debug_panels(mask2d, pore, flow["pressure"], flow["speed"], fig_dir / "Figure_PoreVoronoi_stage1_debug_projection_panels.png")
    print(f"figures saved under: {fig_dir}")


if __name__ == "__main__":
    main()

