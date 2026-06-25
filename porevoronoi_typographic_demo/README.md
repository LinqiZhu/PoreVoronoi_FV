# PoreVoronoi typographic pore demo

This directory builds a compact, reproducible visual demo for the PoreVoronoi
repository.  The demo deliberately separates an auditable numerical pipeline
from a memorable repository hero image.

The geometry is a translucent slab in which the readable word `PoreVoronoi`
is the connected pore/void space.  Flow is solved inside the letters, sampled
particles live inside the same pore space, and the main image is intended as a
GitHub README landing figure rather than a diagnostic projection plot.

## Stage 1

Stage 1 generates:

- a connected 2D typography mask for `PoreVoronoi`;
- a slanted 3D extruded pore mask;
- a pressure-driven reference flow on the pore voxel graph;
- streamlines and sampled particles inside the letter-shaped pore;
- two README-style hero renders and one debug/audit projection figure.

Run from this directory:

```bash
python scripts/00_make_word_mask_2d.py
python scripts/01_make_typographic_pore_3d.py
python scripts/02_solve_pressure_flow.py
python scripts/03_trace_streamlines_and_particles.py
python scripts/04_render_preview.py
python scripts/05_audit_outputs.py
```

Primary visual outputs:

```text
outputs/figures/Figure_PoreVoronoi_stage1_README_hero_light.png
outputs/figures/Figure_PoreVoronoi_stage1_README_hero_dark.png
outputs/figures/Figure_PoreVoronoi_stage1_debug_projection_panels.png
```

The debug panels are required for audit, but they are not the main figure.

## Stage 2

Stage 2 should be executed only after Stage 1 outputs exist and pass audit.
It will add prescribed sites, graph-geodesic ownership, PoreVoronoi-FV cells,
positive-area facelets, conservative flux projection, a paper figure, and a
single README landing hero image. Run after Stage 1:

```bash
python scripts/06_select_prescribed_sites.py
python scripts/07_compute_graph_geodesic_ownership.py
python scripts/08_extract_fv_cells_and_facelets.py
python scripts/09_project_conservative_flux.py
python scripts/09b_reconstruct_cell_velocity.py
python scripts/10_render_final_figure.py
python scripts/11_audit_porevoronoi_fv.py
```

Stage 2 outputs:

```text
outputs/final_figures/Figure_PoreVoronoi_README_landing_hero.png
outputs/final_figures/Figure_PoreVoronoi_README_landing_hero_dark.png
outputs/final_figures/Figure_PoreVoronoi_README_landing_hero_light.png
outputs/final_figures/Figure_PoreVoronoi_FV_four_panel.png
outputs/velocity/cell_velocity.npz
outputs/velocity/Figure_PoreVoronoi_FV_cell_velocity.png
outputs/reports/stage2_audit.json
```

The Stage 2 scripts now execute the PoreVoronoi-FV demo chain on the same typographic pore space: sampled particles are snapped to fixed sites, graph-geodesic ownership assigns all pore voxels, face-connected cells and positive-area facelets are extracted, sampled site states are projected onto a conservative FV-graph flux field, and a cell-level velocity field is reconstructed from the conservative face fluxes.

## Visual target

The main image should feel like a polished scientific simulation repository
landing image:

- the word is readable;
- the letters are voids, not obstacles;
- flow travels inside the letters;
- sampled particles are inside the same pore;
- the background is clean and suitable for a scientific GitHub README.

Projection images, slices, maximum-intensity projections, and mask overlays
are kept as audit/debug artifacts only.
