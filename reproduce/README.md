# reproduce/

This folder holds, for the figures and tables printed in the paper and its Supporting Information, the result
records behind the printed numbers and the scripts that compute them. It is organised by printed item: one folder
per figure or table, one folder per Supporting Information section for numbers quoted in the text, and one folder
of shared builders. The full map, one row per printed item with its data files, scripts, commands and status, is
[`../docs/figures_and_tables.csv`](../docs/figures_and_tables.csv); the commands are explained in
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

## Folders

| Folder | Contents | Printed item |
|---|---|---|
| `figure_01/` | straight-line against through-the-pore ownership on the test images: `euclidean_vs_geodesic_ownership.csv` and the CPU script `euclidean_ownership_comparison.py` | Figure 1(e)-(g) |
| `figure_03/` | `figure_03_source.json`, the plotted values (written by `shared/make_figure_03.py`) | Figure 3 |
| `figure_03/persistent_frontier_h200/` | H200 timing records of ROI-JFA against persistent-frontier BFS (`summary.csv`, `raw_repeats.csv`, `results.json`, `run.json`); the drivers are in `../gpu/ownership/h200/` | Figure 3(d); persistent-frontier column of Table 2 |
| `figure_06/` | `run_refinement.py` (GPU), the refinement ladders on three fixed images | Figure 6 |
| `figure_06/refinement/` | site refinement on the three images: `mask_fps_refinement_from_runs.csv` and one `figure_06/refinement/runs/<case>/<level>/row.json` per case and level | Figure 6(a) |
| `figure_06/voxel_refinement/` | voxel refinement inside fixed cells of the manufactured cube: `stabilization_scaling.csv` | Figure 6(b) |
| `table_02/records/` | timing and acceptance records of the five particle prefixes, the CPU verification, the launch-configuration sweep and the timing statistics | Table 2 |
| `table_03/` | manufactured-solution results as run (`manufactured_stokes.csv`, `manufactured_checks.json`); `table_03/fixed_h/` holds the ladder on one fixed image | Table 3 |
| `table_05/` | forward rows of the controlled cases and the drift of the repeated reference (`forward_rows.csv`, `reference_repeat.csv`); the Table 5 values themselves are in `../data/controlled_cases/expected/` | Table 5 |
| `table_s3/` | site-rule comparison, stability check and sampled-projection rows read by `shared/build_tables.py` | Table S3; tables of S2.4 and S5.3 |
| `table_s8/` | scripts for the public glass-filter data set, which is not redistributed (download it to `../data/glass_filter/`) | Table S8 |
| `stabilization_weight/sweep/` | grid sweep of the stabilization weight on the controlled cases c1-c6, the two package modules with a per-cell weight, and the check that they equal the package modules at weight 1 | Table 4, Figure 8, Table S5 |
| `stabilization_weight/cross_validation/` | the weight chosen from the particle records alone, the check of its records and the text tables | Tables 4 and 6, Figure 8, Table S5 |
| `supplementary/s1_ownership/` | free-space bound, H200 geometry and scale matrix, the five prefixes on the H200, cross-flow check (`supplementary/s1_ownership/cross_flow/`) | Supporting Information S1 |
| `supplementary/s2_discretization/` | permeability readouts, kernel test, stability along refinement, local quadrature | Supporting Information S2 |
| `supplementary/s3_verification/` | reference continuation, self-convergence, velocity distributions, mass backward error, forward resolution of the two ducts, weight sweep on the fibrous proxy | Supporting Information S3 |
| `supplementary/s4_cost/` | cost and sizes, repeated-query cost, forward dimensions of the Berea block (`supplementary/s4_cost/forward_manifests/`) | Supporting Information S4 |
| `supplementary/s5_particle_reconstruction/` | sampled-state projection rows, constraint controls and ablation, Berea density ladders | Supporting Information S5 |
| `shared/` | builders and libraries that import each other and therefore share one folder: `build_tables.py` (LaTeX rows of the tables and `outputs/results_summary.json`), `build_ownership_statistics.py` (Table 2 statistics and the checksum check of the timed code), `make_figure_03.py` with `roi_jfa_fields_cpu.py` and `figure_style.py` (Figure 3), `result_figures.py` (Table 5 source rows), `manufactured_stokes.py`, `stabilization_scaling_cube.py`, `stabilization_sweep.py`, `check_complex_properties.py` | several, see the CSV |

Most scripts say in their module docstring what they compute and which files they read.

## Running the scripts

* Run from the repository root, for example `python reproduce/shared/build_tables.py`. The exception is
  `stabilization_weight/`: its scripts run from their own folder (`cd reproduce/stabilization_weight/sweep`), as
  their `run_all_cases.py` schedulers start them, and they start from the stored solutions that
  `porevoronoi_fv/assisted/run_assisted_arms.py` writes to `outputs/assisted/` (run it for c1-c6 first).
* New output goes to `outputs/` at the repository root, which git ignores. Scripts that wrote a record shipped in
  this folder write to that record's folder by default, so a rerun replaces the shipped file and `git diff` shows
  any change; most of them take `--out`, `--root` or `--out-dir` (see `--help`) to write elsewhere.
* Scripts that start the GPU forward pipeline in `../gpu/` (through `gpu/studies/study_common.py`) need an NVIDIA GPU
  and CuPy (`../environment/requirements-gpu.txt`); the others run on a CPU (`../environment/requirements.txt`).
  The module docstrings and [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) say which is which.
* The re-analyses of the refinement ladder in `supplementary/` (`kernel_exact.py`, `stability_along_refinement.py`,
  `continue_reference.py`, `mass_backward_error.py`, `self_convergence.py`, `velocity_distribution.py`,
  `cost_and_sizes.py`, `repeated_query_cost.py`) read the per-level states `state.npz` and
  `sites_and_partition.npz`. The repository ships only the `row.json` records of the levels; run
  `python reproduce/figure_06/run_refinement.py` (GPU) first, which writes the states next to them in
  `figure_06/refinement/runs/`.
* `shared/build_ownership_statistics.py` recomputes the Table 2 statistics from the per-repeat timing records, which
  are not included in this repository, so it stops when it reaches them, and so does `shared/result_figures.py`,
  which calls it first. `shared/make_figure_03.py` also reads the LaTeX table that `build_ownership_statistics.py`
  writes, so of the two Figure 3 scripts only `shared/roi_jfa_fields_cpu.py` runs to the end here; the plotted
  values are in `figure_03/figure_03_source.json` and the statistics in `table_02/records/`.

## What runs from the repository and what needs the evidence archive

The status column of [`../docs/figures_and_tables.csv`](../docs/figures_and_tables.csv) says, for every printed
item, whether its records and scripts are in this repository, partly in it, in the evidence archive supplied with
the manuscript, or rest on experimental data of other studies that are not redistributed. The records in this
folder belong to the items that are in, or partly in, the repository, and to numbers quoted in the Supporting
Information text (`supplementary/`). For items marked "evidence archive" the
scripts may be here (for example in `stabilization_weight/`) while the result records of the paper are in the
archive. Until the paper is published the evidence archive is available from the corresponding author, Linqi Zhu
(linqi.zhu@imperial.ac.uk), on request.
