# Same-mask Block-agglomeration Baseline

Date: 2026-06-06

This audit compares the current GeoVoronoi-FV construction with a conservative block-voxel agglomeration baseline on the same masks, the same locked voxel-reference permeabilities, and the same monolithic coarse-Stokes solve. The block baseline is a current-operator construction comparison, not a claim about a fully optimized implementation of the cited voxel-agglomeration literature.

| Case | Method | Rule | N_cv | C | eK (%) | Speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Thin-wall synthetic | GeoVoronoi-FV | regular half-offset (4,4,4) | 576 | 61.4 | 5.05 | 170.5 |
| Thin-wall synthetic | Block agglomeration | axis-aligned block (4, 4, 4) | 576 | 61.4 | 2.11 | 1354.0 |
| Narrow-throat synthetic | GeoVoronoi-FV | regular half-offset (4,4,4) | 576 | 63.1 | 0.47 | 296.4 |
| Narrow-throat synthetic | Block agglomeration | axis-aligned block (4, 4, 4) | 576 | 63.1 | 0.16 | 1074.2 |
| Bentheimer segmented sandstone crop | GeoVoronoi-FV | mask-selector origin-offset (2,4,4) | 1325 | 32.3 | 3.95 | 323.1 |
| Bentheimer segmented sandstone crop | Block agglomeration | axis-aligned block (4, 4, 4) | 1022 | 41.8 | 56.93 | 1438.6 |
| Fibrous filter proxy | GeoVoronoi-FV | mask-selector wall-biased fast (2,4,4) | 1337 | 29.1 | 6.48 | 452.2 |
| Fibrous filter proxy | Block agglomeration | axis-aligned block (4, 4, 4) | 1024 | 38.0 | 1649.10 | 1582.2 |

## Use boundary

- Safe use: same-mask construction comparison under the current coarse operator.
- Unsafe use: claiming a definitive speed benchmark against all voxel-agglomeration implementations.
- Interpretation should focus on whether prescribed geodesic/state sites improve accuracy at comparable compression on the retained masks.
- The retained result is nuanced: block agglomeration is competitive on regular synthetic masks but fails badly on the two segmented complex morphologies.
