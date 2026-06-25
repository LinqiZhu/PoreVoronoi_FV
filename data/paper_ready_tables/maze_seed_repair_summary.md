# C_maze Seed-refinement Audit

Date: 2026-06-06

This audit keeps the wall factor fixed at beta=1.25 and changes only the GeoVoronoi-FV seed placement/density for the maze stress case. It is therefore a method-internal seed-refinement result rather than a post-hoc wall-calibration result.

- Baseline regular half-offset `(4,4,4)`: eK=27.19%, speedup=62.7x.
- Fast-scan balanced candidate `(2,3,4)` half-offset: eK=5.74%, speedup=182.1x.
- Full-field exported balanced refinement `(2,3,4)`: eK=5.32%, e_u=27.96%, e_phi=25.15%, speedup=163.0x.
- Full-field exported accuracy refinement `(2,2,4)`: eK=2.63%, e_u=24.90%, e_phi=23.65%, speedup=143.7x.
- Best fast-scan row `stride_2x2x4_half`: eK=3.19%, speedup=80.5x; supersede this scan number with full-field export when available.

| Role | Seed family | Seed id | Stride | S | N_cv | eK (%) | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline_61x | stride_half_admissible | `stride_4x4x4_half` | (4, 4, 4) | 576 | 576 | 27.19 | 62.7 |
| gfps_balanced_candidate | gfps_admissible | `gfps_target_stride_3x4x4` | (3, 4, 4) | 768 | 813 | 7.32 | 271.7 |
| balanced_refinement | stride_half_admissible | `stride_2x3x4_half` | (2, 3, 4) | 1536 | 1536 | 5.32 | 163.0 |
| accuracy_refinement | stride_half_admissible | `stride_2x2x4_half` | (2, 2, 4) | 2304 | 2304 | 2.63 | 143.7 |
| screened_candidate | gfps_admissible | `maze_gfps_stride_2x4x4_r0` | (2, 4, 4) | 1152 | 1186 | 7.73 | 255.7 |

## Manuscript Use

- Safe: state that the unrefined maze row is repairable by seed refinement without changing wall beta.
- Safe: use the full-field exported `(2,2,4)` row as the accuracy-focused repaired row in the main accuracy-cost table.
- Safe: retain the full-field exported `(2,3,4)` row as a faster balanced trade-off.
- Do not present the scan-only 3.19% number as the final accuracy result; the full-field exported `(2,2,4)` row supersedes it with 2.63%.
