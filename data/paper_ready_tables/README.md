# Paper-Ready Data Package

Date: 2026-06-07

This folder contains the numerical records behind the current CMAME manuscript.
For the visible paper, use only the final protocol rows and the final
claim-to-evidence matrix. Development scans and rejected closures may remain in
the folder as internal provenance, but they are not manuscript evidence.

## Manuscript-Facing Files

- `main_solver_table_for_table1.csv`: synchronized end-to-end accuracy-cost
  rows for the active manuscript-facing Route-B locked wall-resolution
  closure. The old cancellation-prone Bentheimer headline is not manuscript
  evidence.
- `sampled_state_mainline_transfer_rows.csv`: current two-row real-mask
  mainline transfer record. It reports the Bentheimer \(225^3\) PTV/DNS
  sampled-state row and the public Berea \(128^3\) reference-field rollout row
  under the same \(S_m+S_a\) protocol. The corresponding TeX rows are in
  `tex_snippets/table_sampled_state_mainline_transfer_rows.tex`.
- `protocol_refinement_berea_rerun_rows.csv`: fresh public Berea support ladder
  regenerated from sampled-state point-window creation through final
  conservative projection. Use it for the protocol-refinement audit and
  Figure 12.
- `protocol_refinement_mechanism_rows.csv`: locked mechanism-side
  resolution-sensitivity rows used in Figure 12.
- `bentheimer_final_no_reference_selected_row.csv`: compact audit record for
  the final Bentheimer selected row. Use this for checking the selector and the
  SI selected-row table.
- `same_mask_agglomeration_baseline.csv`: scoped same-mask comparison with
  voxel agglomeration. The Bentheimer GeoVoronoi-FV row is synchronized to the
  final selected offset.
- `site_sensitivity_table_for_tex.csv`: declared site-protocol table for the
  visible manuscript.
- `reference_convergence_table_for_tex.csv`: voxel-reference provenance and
  reference-stability records.
- `current_large_scale_roi_jfa_table_for_tex.csv`: construction-only ROI-JFA
  scalability rows. These do not support enlarged-grid flow accuracy claims.
- `current_partition_topology_table_for_tex.csv`: procedural construction
  diagnostics. In the current manuscript, segmented-rock response evidence is
  reported through the final site protocols and end-to-end rows instead.
- `claim_to_evidence_matrix.csv`: final claim-to-evidence guide for the active
  paper.
- `sampled_state_site_audit/`: PTV-rollout sampled-state audit used to support
  the state-location story without adding a new permeability headline.

## Visible Manuscript Rule

The active paper should expose one retained GeoVoronoi-FV operator and one
final protocol per reported case. Do not add rejected closures, old Bentheimer
rows, old maze baselines, process-only repair scans, or local velocity/face-flux
diagnostic errors to visible manuscript tables or figures.

The headline numerical claims are:

- Procedural rows: `e_K=0.93--2.45%` at `8.0--27.0x` compression.
- Maze stress row: `e_K=1.89%` at `7.7x` compression under the fixed final
  site rule.
- Bentheimer segmented crop: `e_K=0.62%` and `e_phi=19.78%` at `32.2x`
  compression under the no-reference internally selected offset and the locked
  wall-resolution closure.
- Reported mechanism and segmented single-solve speedups: `3.8--380.5x` against the corresponding
  voxel-reference runs in the same production runner.
- The mechanism and segmented validation rows keep mass residuals below `1.6e-15` per
  traversable voxel.
- Sampled-state site audit: frame-69 PTV rollout on the \(225^3\)
  Bentheimer mask gives 3471 unique graph sites and 15426 face-connected FV
  cells in 0.242 s; block-centroid relocation moves the PTV-derived states by
  3.84 voxels at the median and changes the local velocity sample by 0.635 at
  the median relative level.
- Sampled-state real-mask mainline: the Bentheimer \(225^3\) PTV/DNS row gives
  \(S_m=225423\), \(S_a=20111\), \(N_c=243673\), `e_K=2.30%`,
  `e_phi=19.72%`, `e_u=19.14%`, and `t_total=3.024 s`; no authoritative
  same-configuration DNS wall-clock is available for that row. The public Berea
  \(128^3\) row gives \(S_m=22571\), \(S_a=3451\), \(N_c=25831\),
  `e_K=0.39%`, `e_phi=13.32%`, `e_u=17.24%`, and `t_total=2.059 s`;  
  the same-hardware 900-step voxel reference is `818.51 s`, giving a strict
  `398x` sampled-state transfer speedup. This ratio compares the
  construction plus state-to-flux conservative projection task with generating
  the same-mask voxel-reference field; it is not a coarse-Stokes solve speedup.
- Protocol-refinement audit: the public Berea support ladder was regenerated
  from point-window creation through final projection. The 500-support row is
  the lower-support boundary (`e_u=21.26%`), the retained 1000-support row is
  the mainline transfer row, and higher support lowers `e_phi` to `9.81%` while
  scalar `e_K` remains nonmonotone. Do not call this a formal convergence-rate
  proof.

## Internal Provenance

Files such as `maze_seed_repair_scan.csv`,
`no_reference_closure_attempts_summary.csv`, and
`hydraulic_closure_variant_decision_table.csv` are retained only as internal
records. They should not be used to build visible paper tables or figures
unless the manuscript is intentionally re-scoped and the final claim matrix is
updated first.

The legacy `current_bentheimer_density_table_for_tex.csv`,
`tex_snippets/table_current_bentheimer_density_rows.tex`, and
`Figure_10_convergence.*` files are retained as historical site-density
diagnostics from an earlier fixed-beta segmented-crop stage. They are not
referenced by the current main manuscript or Supporting Information and should
not be used as current sampled-state mainline evidence.
