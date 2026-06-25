# Mask-only Seed Selector Audit

Date: 2026-06-06

This audit fixes a seed-family selector from mask morphology only, then reports the already-computed validation rows. The selector does not use permeability or velocity error to choose a seed family; locked references are used only after selection to quantify the result.

## Selector rule

- `fibrous_high_wall_contact`: if near-wall fluid fraction is at least 0.35 and porosity is at least 0.45, use wall-biased sites for a fast policy, or denser GFPS `(1,4,4)` for an accurate fibrous policy.
- `granular_low_porosity`: if porosity is below 0.45 and the mean solid-neighbour count is at least 0.25, use origin-offset structured sites.
- `structured_or_open_channel`: otherwise use regular half-offset structured sites.

## Validation rows

| Case | Policy | Porosity | Near-wall fraction | Selector | Seed family | eK (%) | Speedup | Evidence |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| Orthogonal duct | fast | 1.000 | 0.000 | structured_or_open_channel | stride_half_admissible | 1.10 | 3.8 | main_solver_table_for_table1.csv |
| Skewed duct | fast | 0.987 | 0.028 | structured_or_open_channel | stride_half_admissible | 1.15 | 17.3 | main_solver_table_for_table1.csv |
| Thin-wall synthetic | fast | 0.960 | 0.084 | structured_or_open_channel | stride_half_admissible | 0.93 | 380.5 | main_solver_table_for_table1.csv |
| Narrow-throat synthetic | fast | 0.986 | 0.030 | structured_or_open_channel | stride_half_admissible | 2.45 | 328.1 | main_solver_table_for_table1.csv |
| Maze synthetic | fast | 0.958 | 0.089 | structured_or_open_channel | stride_half_admissible | 1.89 | 15.6 | main_solver_table_for_table1.csv |
| Bentheimer segmented sandstone crop | fast | 0.290 | 0.187 | granular_low_porosity | stride_offset | 0.62 | 365.8 | bentheimer_final_no_reference_selected_row.csv |
| Fibrous filter proxy | fast | 0.594 | 0.402 | fibrous_high_wall_contact | wall_biased | 6.48 | 497.6 | morphology_aware_seed_protocol_candidate.csv |
| Fibrous filter proxy | accurate | 0.594 | 0.402 | fibrous_high_wall_contact | gfps_admissible | 1.51 | 147.2 | morphology_aware_seed_protocol_candidate.csv |

## Manuscript use

- Safe stronger use: seed placement can be framed as a mask-conditioned protocol constraint rather than a reference-tuned post hoc choice.
- Boundary: this is not yet a universal automatic selector for arbitrary unseen rocks; it is an audit over the retained manuscript masks and the already validated candidate rows.
- Fibrous interpretation: the mask-only high-wall-contact class explains why regular half-offset sites fail on fibrous media and why wall-biased/GFPS candidates are retained as fast/accurate policies.
