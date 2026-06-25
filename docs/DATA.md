# Data and reproducibility notes

PoreVoronoi stores compact data products that support the GeoVoronoi-FV manuscript. The data layout separates final figure source data, paper-ready table data, and compact example inputs.

## Included data

### `data/final_source_data/`

Figure-level CSV, JSON, and NPZ source files used by the final manuscript figures. These are the preferred files for checking plotted values and scene-level figure inputs.

### `data/paper_ready_tables/`

CSV summaries, Markdown audit notes, and TeX snippets used in the manuscript and Supporting Information tables.

### `examples/synthetic_reference_data/`

Compact `.npz` reference data for the controlled synthetic cases: orthogonal duct, skewed duct, thin wall, narrow throat, and maze.

### `examples/segmented_masks/`

Compact segmented-mask examples and manifests. These are derived examples, not a redistribution of the large raw tomography volumes.

## Data not included

The package does not include large raw tomography volumes, private scratch directories, or every intermediate production output. Rebuilding those larger cases requires local copies of the original input data and a compatible CUDA environment.

## Public-release check

For public release, the included compact data products are covered by `DATA_LICENSE.md`. Raw tomography images and external source datasets are not redistributed here; readers should use the original data sources cited in the manuscript.
