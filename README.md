# PoreVoronoi

PoreVoronoi is the reproducibility and reference-code package for the CMAME manuscript on **GeoVoronoi-FV**, a prescribed-site conservative finite-volume construction for obstructed voxel domains.

The repository is intentionally organised around the manuscript evidence chain: graph-geodesic ownership, positive-area facelet exchange, finite-volume operator assembly, conservative state-to-flux projection, and the compact synthetic and segmented-mask examples used in the paper.

## What this repository contains

```text
src/geovoronoi_fv/          Core implementation files used by the manuscript runs
repro/experiments/          Experiment, audit, and source-data construction scripts
repro/figures/              Figure regeneration scripts for selected final figures
data/final_source_data/     CSV, JSON, and NPZ source data used by final manuscript figures
data/paper_ready_tables/    Paper-ready CSV tables and TeX snippets
examples/                   Compact redistributable example inputs
manuscript/figures/         Final figure PDFs and available editable SVG sources
docs/                       Data, reproducibility, and package-manifest notes
```

The package includes small reference datasets and final table/figure source data so that readers can inspect the numerical evidence without downloading the full raw tomography volumes.

## What PoreVoronoi does

PoreVoronoi implements and documents the code path behind GeoVoronoi-FV:

1. prescribed sample sites are kept fixed in the pore space;
2. graph-geodesic ownership assigns traversable voxels without crossing solid obstructions;
3. positive-area voxel facelets define conservative exchange between neighbouring cells;
4. face-connected cell assembly turns labels into finite-volume unknowns;
5. geodesic face distances are used in the cell-cell exchange operator;
6. sampled states are projected onto a conservative face-flux field.

This is a research code release rather than a polished application library. The priority is traceability to the manuscript figures, tables, and audits.

## Installation

Create a Python environment with Python 3.11 or newer, then install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU production paths require a compatible NVIDIA GPU and CuPy. If your CUDA version differs from the default requirement, install the matching CuPy wheel manually.

## Quick verification

From the repository root:

```bash
python -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in pathlib.Path('.').rglob('*.py')]; print('AST_OK')"
```

To regenerate selected figure products from packaged source data:

```bash
python repro/figures/make_figure_04_ownership_speed_audit.py
python repro/figures/make_figure_05_operator_geometry_unified.py
python repro/figures/Figure_07_state_projection.py
python repro/figures/make_figure_13_mainline_transfer.py
```

Some figure scripts call Inkscape for SVG/PDF export. Set `INKSCAPE_BIN` if the `inkscape` executable is not on your `PATH`.

## Example data

The repository includes compact examples under `examples/`:

- `synthetic_reference_data/`: duct, thin-wall, narrow-throat, and maze reference cases;
- `segmented_masks/`: compact segmented-mask examples and manifests.

Large raw tomography volumes and workstation-scale output directories are not redistributed. Scripts that need those inputs read repository-relative paths by default and can be pointed to external data with environment variables.

## Environment variables

Optional variables for larger local reruns:

- `GEOVORONOI_FV_ROOT`: repository root when scripts are launched externally;
- `GEOVORONOI_FV_WORK_ROOT`: external work directory containing large local inputs;
- `GEOVORONOI_FV_GPU_PYTHON`: Python executable with CUDA/CuPy support;
- `INKSCAPE_BIN`: Inkscape executable for vector export;
- `BENTHEIMER_SEGMENTED_TIF`: local path to the full Bentheimer segmented image if rebuilding the compact crop.

## Reproducibility scope

Included and directly inspectable:

- final manuscript figure PDFs;
- available editable SVG sources;
- source CSV/JSON/NPZ files for final figures;
- paper-ready tables and TeX snippets;
- compact synthetic and segmented-mask examples;
- scripts used for figure and experiment source-data generation.

Not included:

- full raw tomography images;
- every intermediate GPU production output;
- private workstation scratch directories;
- exploratory tuning runs that were not used as manuscript evidence.

## Manifest

`MANIFEST.csv` records file paths, byte counts, and SHA256 hashes for the staged package.

## Citation

Citation metadata will be added after the associated manuscript record is finalised. Until then, cite the CMAME manuscript describing GeoVoronoi-FV and refer to this repository as the PoreVoronoi reproducibility package.

## License

No public reuse license has been selected yet. Add the final author-approved license before making the GitHub repository public.
