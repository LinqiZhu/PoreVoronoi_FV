# State-location Preservation Audit

Date: 2026-06-07

Purpose: quantify the location error introduced when the same prescribed sites used by GeoVoronoi-FV are represented by an axis-aligned block-agglomeration centroid. This is a no-flow twin audit: GeoVoronoi-FV keeps the supplied state coordinate as the unknown location, whereas the block baseline relocates that state to the centroid of the containing block component.

| Case | S | block cells | split | relocation p50 (vox) | relocation p95 (vox) | max (vox) | smooth-state rel. RMSE | wall-state rel. RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thin-wall synthetic | 576 | 576 | 0 | 0.87 | 0.87 | 0.87 | 0.085 | 0.333 |
| Narrow-throat synthetic | 576 | 576 | 0 | 0.87 | 0.87 | 0.87 | 0.088 | 0.335 |
| Bentheimer segmented sandstone crop | 1319 | 1022 | 6 | 2.18 | 2.61 | 2.87 | 0.092 | 0.594 |
| Fibrous filter proxy | 1256 | 1024 | 92 | 1.82 | 2.60 | 2.87 | 0.126 | 0.592 |

Recommended use:

- Safe main-text use: one sentence or a small SI table showing that block-centred agglomeration is not a prescribed-state method because it relocates supplied sites by O(1) voxels under the same mask and cell scale.
- Unsafe use: claiming that this no-flow audit proves permeability superiority. Permeability evidence must remain in the same-mask agglomeration table.
- Best story fit: pair this audit with the same-mask agglomeration result. Regular synthetic masks show block agglomeration can be numerically competitive; segmented morphologies show why retaining prescribed state locations matters.

Elapsed: 10.67 s
