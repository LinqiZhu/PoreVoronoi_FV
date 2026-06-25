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
single README landing hero image.

The Stage 2 scripts in this folder currently act as guards.  They check that
Stage 1 has been completed before any later method-specific visual products
are generated.

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

