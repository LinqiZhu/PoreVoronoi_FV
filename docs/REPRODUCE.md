# Reproducing the figures and tables

This page tells you, for every figure and table printed in the paper *PoreVoronoi-FV: Conservative pore-scale
flow on cells fixed by tracked particles* and in its Supporting Information, where its numbers are and how to
compute them again. The same information, one row per item, is in
[`figures_and_tables.csv`](figures_and_tables.csv).

The numbers behind an item are in one of these places:

| Status | Meaning |
|---|---|
| in repository | the data files and the scripts that produce them are in this repository |
| partly in repository | some panels or columns are in this repository; the last column of the CSV says where the rest is |
| evidence archive | the result records are in the evidence archive supplied with the manuscript (see [below](#the-evidence-archive)); the scripts that wrote them may be here |
| not redistributed: experimental data | the item rests on experimental data of other studies, which we do not redistribute; the CSV names the published source |

All 29 printed items: 3 in repository, 8 partly in repository, 12 in the evidence archive, 6 resting on
experimental data that are not redistributed.

Commands are run from the repository root, except those in `reproduce/stabilization_weight/`, which are run from
their own folder. Outputs go to `outputs/`, which git ignores. The CPU scripts need only NumPy and SciPy (plus pandas
and Matplotlib for tables and figures); scripts marked GPU need an NVIDIA GPU and CuPy. Installation:
[`../environment/ENVIRONMENT.md`](../environment/ENVIRONMENT.md). The folders of `reproduce/` are described in
[`../reproduce/README.md`](../reproduce/README.md).

## Overview

| Item | What it shows | Status | Start here |
|---|---|---|---|
| Figure 1 | cell construction; through-the-pore assignment splits no cell | partly in repository | [`reproduce/figure_01/`](../reproduce/figure_01/) |
| Figure 2 | particle tracks in the seven numerical images and the drainage experiment | partly in repository | [`data/controlled_cases/`](../data/controlled_cases/), [`data/berea64/`](../data/berea64/) |
| Figure 3 | ROI-JFA proposal, certification, closure; speed-up on one H200 GPU | in repository | [`reproduce/figure_03/`](../reproduce/figure_03/) |
| Figure 4 | one shared trace per interface patch | evidence archive | |
| Figure 5 | inf-sup constant across cell counts | evidence archive | |
| Figure 6 | refining cells against refining voxels | partly in repository | [`reproduce/figure_06/`](../reproduce/figure_06/) |
| Figure 7 | tracer transport with facelet and recovered fluxes | evidence archive | |
| Figure 8 | assisted against Stokes-only error at equal weight | evidence archive | [`reproduce/stabilization_weight/`](../reproduce/stabilization_weight/) |
| Figure 9 | drainage experiment, one block | not redistributed: experimental data | |
| Figure 10 | drainage experiment, twelve block-windows | not redistributed: experimental data | |
| Table 1 | test images, particles and cell complexes | partly in repository | [`data/`](../data/) |
| Table 2 | certified ownership and speed-up on five Berea prefixes | in repository | [`reproduce/table_02/records/`](../reproduce/table_02/records/) |
| Table 3 | manufactured-solution refinement | partly in repository | [`reproduce/table_03/`](../reproduce/table_03/) |
| Table 4 | best stabilization weight | evidence archive | [`reproduce/stabilization_weight/`](../reproduce/stabilization_weight/) |
| Table 5 | Stokes-only errors on the six controlled cases | in repository | [`porevoronoi_fv/`](../porevoronoi_fv/), [`data/controlled_cases/`](../data/controlled_cases/) |
| Table 6 | assisted calculation on the controlled cases | evidence archive | [`porevoronoi_fv/assisted/`](../porevoronoi_fv/assisted/) |
| Table 7 | drainage experiment, summary | not redistributed: experimental data | |
| Figure S1 | numerical geometries | partly in repository | [`data/controlled_cases/`](../data/controlled_cases/) |
| Figure S2 | dense simulated particle tracks | evidence archive | |
| Figure S3 | measured tracks in a glass filter | not redistributed: experimental data | |
| Table S1 | normal-flux defects of recovered velocities | evidence archive | |
| Table S2 | stabilization weight on skewed-duct partitions | evidence archive | |
| Table S3 | site and partition rules at matched size | partly in repository | [`reproduce/table_s3/`](../reproduce/table_s3/) |
| Table S4 | tracer transport | evidence archive | |
| Table S5 | weights chosen from the particle records | evidence archive | [`reproduce/stabilization_weight/`](../reproduce/stabilization_weight/) |
| Table S6 | certified floor | evidence archive | |
| Table S7 | stabilization weight, drainage block-windows | not redistributed: experimental data | |
| Table S8 | glass filter, projection onto balanced fluxes | not redistributed: experimental data | [`reproduce/table_s8/`](../reproduce/table_s8/) |
| Table S9 | this map | partly in repository | [`figures_and_tables.csv`](figures_and_tables.csv) |

Numbers quoted in the text of the Supporting Information have their records in
[`reproduce/supplementary/`](../reproduce/supplementary/), one folder per section (S1 ownership, S2 discretization,
S3 verification, S4 cost, S5 particle reconstruction).

## Quick checks (CPU, a few minutes)

```bash
python -B porevoronoi_fv/check_selectors_and_assembly.py   # no flow solve; under one minute
python -B porevoronoi_fv/check_archived_solutions.py       # every printed Table 5 value from the stored solutions; about 4 s per case
python -B examples/quickstart.py --case c1                 # one full Stokes-only solve, compared with Table 5; about 3 min
python -m pytest tests                                     # unit tests of the GPU pipeline (pytest; CuPy tests skip without a GPU)
```

`check_archived_solutions.py` recomputes, for each controlled case c1-c6, every Table 5 value from the stored solution
`data/controlled_cases/*/stokes_only_solution.npz` and compares it with the printed digits
(`data/controlled_cases/expected/printed_table_values.json`); it also rebuilds the cells and checks that they equal the
stored ones, and that every particle record carries the reference velocity of its voxel. The file of printed values
gives the largest mass residual with three significant digits (1.14 × 10⁻¹⁶ for c1); the paper prints two
(1.1 × 10⁻¹⁶).

## The controlled cases: Tables 5 and 6, Figure 2, Table 1

The six controlled cases are five synthetic 24 × 24 × 64 ducts (c1 orthogonal duct, c2 skewed duct, c3 thin wall,
c4 narrow throat, c5 maze) and the 16 × 96 × 96 Bentheimer crop (c6). Each case folder
`data/controlled_cases/<case>/` (`orthogonal_duct`, `skewed_duct`, `thin_wall`, `narrow_throat`, `maze`,
`bentheimer_crop`) holds the reference flow we computed on the voxel grid (`reference_flow.npz`), the
simulated particle window (`particle_tracks.csv.gz`) with the settings that made it
(`particle_tracks_generator.json`), the stored Stokes-only solution (`stokes_only_solution.npz`) and the run
record (`run_record.json`). The printed Table 5 values are in `data/controlled_cases/expected/`.

**Table 5.** Three ways to check it, in increasing cost:

| Command | Solve | Time |
|---|---|---|
| `python -B porevoronoi_fv/check_archived_solutions.py` | no | about 4 s per case, one CPU core |
| `python -B examples/quickstart.py --case c1` | yes, one case | about 3 min, one CPU core |
| `python -B porevoronoi_fv/run_stokes_only.py --case c1 --stage e0` | yes, one case | 1.5-3 min per case, one CPU core |

`porevoronoi_fv/slurm_stokes_only.sh` runs, as one Slurm job, the solve of all six cases, a check that the fast
and the loop assembly give the same matrices, a spectral bound and the checks of `evaluate_stokes_only.py`. Submit it
from its folder, `cd porevoronoi_fv && mkdir -p ../outputs && sbatch --partition=<your partition>
slurm_stokes_only.sh`, and set `PVFV_PYTHON` to your Python interpreter (default `python3`); the folder
`porevoronoi_fv/README.md` lists the other settings. MINRES iteration counts differ between builds of BLAS and SciPy
(by up to about 2% between our own runs); the checks therefore test the residual of the solve, not the iteration
count.

`python reproduce/shared/build_tables.py` writes the LaTeX rows of Table 5 (`outputs/generated/final_accuracy_cost_table.tex`)
from `data/controlled_cases/expected/table_source_values.csv` and the run records, together with the other tables of
the paper and the Supporting Information that rest on files of this repository, and `outputs/results_summary.json`.
Run it once without `--only`; afterwards `--only <table>` works, because it reads that summary. A table whose source
file is not in the repository (the glass-filter results) is reported as `SOURCE MISSING` and skipped. It also rewrites
`reproduce/figure_06/refinement/mask_fps_refinement_from_runs.csv` from the run records of Figure 6, so `git diff`
shows whether the regenerated numbers differ. `reproduce/shared/result_figures.py` wrote the Table 5 source table
(it writes it to `outputs/figure_source_data/table_05_source.csv`); it first draws a plot of the Table 2 statistics,
which needs the per-repeat timing records that are not included (see [Ownership](#ownership-figures-1-and-3-table-2)),
so it stops before that step here.
`reproduce/table_05/forward_rows.csv` holds the rows of the forward runs and `reproduce/table_05/reference_repeat.csv`
the drift between repeated reference solves.

**Table 6.** `python -B porevoronoi_fv/assisted/run_assisted_arms.py --case c1` solves the Stokes-only, one-record and
all-record calculations of one case on the CPU and stores each solution in `outputs/assisted/<case>/` (labels `PN`,
`AN`, `MN`); run it for c1-c6. The weights chosen from the particle records come from
`run_cross_validation.py --kind CV` in `reproduce/stabilization_weight/cross_validation/` (see
[below](#stabilization-weight-figure-8-tables-4-and-s5)). The printed values and the certified floor are in the
evidence archive (extended-data entries ED5.7, ED5.11, ED5.15).

**Figure 2 and Table 1.** The particle windows and masks are the files above and `data/berea64/`. To make a
window again, give the generator the settings recorded in its `particle_tracks_generator.json`, for example for c1:

```bash
python gpu/code/build_particle_window.py \
  --reference data/controlled_cases/orthogonal_duct/reference_flow.npz --out-dir outputs/windows \
  --n-particles 721 --sample-mode uniform --seed 20260711 --step-vox-per-frame 0.1 --frame-list 0:99:1
```

Table 1 was typeset by hand from these files; its drainage-experiment row comes from data that are not
redistributed.

## Ownership: Figures 1 and 3, Table 2

| Item | Command | Hardware |
|---|---|---|
| Figure 1, panels e-g | `python reproduce/figure_01/euclidean_ownership_comparison.py --out outputs/figure_01` | CPU |
| Figure 3, panels a-c | `python reproduce/shared/roi_jfa_fields_cpu.py`, then `python reproduce/shared/make_figure_03.py` | CPU |
| Figure 3, panel d; Table 2, persistent-frontier column | `cd gpu/ownership/h200 && sbatch jobs/job_persistent_baseline.sh` (set the account, the partition and `PYTHON` first) | GPU; the recorded run took 42 s on one NVIDIA H200 |
| Table 2, other columns | `python gpu/ownership/code/time_roi_vs_bfs.py --mask-npz data/berea64/mask.npz --particle-window data/berea64/particle_tracks.csv.gz --particle-ids 0:49 --out-dir outputs/table_02` for the prefixes `0:9`, `0:49`, `0:99`, `0:199`, `0:499` | GPU; the recorded runs used an NVIDIA GeForce RTX 5080 Laptop GPU |

Figure 1 (e-g) compares straight-line and through-the-pore assignment on the masks of the seven test images
(`data/controlled_cases/*/reference_flow.npz` and `data/berea64/reference_flow_x.npz`) with 200, 800 and 3200
farthest-point sites each; its values are in `reproduce/figure_01/euclidean_vs_geodesic_ownership.csv`, and
`python reproduce/shared/build_tables.py` formats them.

`roi_jfa_fields_cpu.py` recomputes the ROI-JFA fields of Figure 3 (a-c) on the CPU from `data/berea64/mask.npz` and
the Berea particle window and writes `roi_jfa_fields.npz` and `roi_jfa_fields_check.json` into
`reproduce/figure_03/`, where `make_figure_03.py` reads them; `make_figure_03.py` checks them against the Table 2
record, rewrites `reproduce/figure_03/figure_03_source.json` (the plotted values) and draws the figure into
`outputs/figures/`. It also reads `outputs/generated/ownership_statistics.tex`, which
`build_ownership_statistics.py` writes from the per-repeat timing records, so in this repository only the first of
the two commands runs to the end; the plotted values are in `figure_03_source.json`. The
H200 timing records of panel d are in `reproduce/figure_03/persistent_frontier_h200/`; the job runs the drivers of the
self-contained folder `gpu/ownership/h200/`, which check their inputs against recorded checksums first
([`../gpu/ownership/README.md`](../gpu/ownership/README.md)).

Table 2 is `reproduce/table_02/records/ptv_ownership_statistics_canonical.json`, collected by
`reproduce/shared/build_ownership_statistics.py` from the five timing records
`reproduce/table_02/records/ptv{10,50,100,200,500}_ownership_backend_audit_gpu6.json` and from repeated timing runs.
The repeats are summarized in `reproduce/table_02/records/ptv_ownership_timing_stability_4runs*`, but their per-repeat
records are not included in this repository, so `build_ownership_statistics.py` cannot be rerun here. Given those
records, it also checks that the ownership module, the timing driver and the site rule in this repository have the
SHA-256 listed in the timing records. The persistent-frontier column is the rows `ptv10` to
`ptv500` of `reproduce/figure_03/persistent_frontier_h200/summary.csv`. The CPU verification of the GPU module is
`gpu/ownership/code/verify_against_cpu.py` (record: `reproduce/table_02/records/ptv_ownership_gpu6_cpu_reference_verification.json`).

The GPU ownership module exists in two versions that differ only in how the number of streaming multiprocessors is
looked up; the numerical kernels are the same. `gpu/ownership/code/ptv_ownership_gpu6.py` queries it at each launch
and produced the laptop timings (Table 2, host-driven column, the launch-configuration sweep and the CPU
verification); `gpu/ownership/h200/src/ptv_ownership_gpu6.py` caches it once per device and produced the H200
timings (Figure 3d, Table 2, persistent-frontier column). On the CPU, `porevoronoi_fv/pore_ownership.py` computes the
same exact ownership by a multi-source shortest-path search; it builds the cells of every CPU calculation.

## Refinement and the manufactured solution: Figure 6, Table 3

`reproduce/figure_06/refinement/runs/` holds one record per case and refinement level (200 to 6400 sites on the
orthogonal duct, the skewed duct and the Bentheimer crop); `python reproduce/shared/build_tables.py` collects them
into `reproduce/figure_06/refinement/mask_fps_refinement_from_runs.csv`, the values of panel a.
`reproduce/figure_06/voxel_refinement/stabilization_scaling.csv` holds the voxel refinement inside 4 × 4 × 4 fixed
cells of the manufactured cube at seven stabilization weights, written by `reproduce/shared/stabilization_scaling_cube.py`.
`reproduce/table_03/manufactured_stokes.csv` holds the as-run columns of Table 3 (rows with `load_functional` =
`linear`). These studies ran on the GPU forward path: `python reproduce/figure_06/run_refinement.py --root
outputs/figure_06`, `python reproduce/shared/stabilization_scaling_cube.py --root outputs/voxel_refinement` and
`python reproduce/shared/manufactured_stokes.py --root outputs/table_03` (NVIDIA GPU with CuPy; the last two also
need SymPy). The drawing script of Figure 6, the rows of panel b as printed and the fixed-size columns of Table 3 are
in the evidence archive.

## Matched partitions: Table S3

`reproduce/table_s3/site_rule_comparison.csv` holds the rows of Table S3; `python reproduce/shared/build_tables.py`
formats them. `gpu/studies/build_site_rule_partitions.py` builds the partitions from the geometry alone and
`gpu/studies/run_site_rule_comparison.py` solves them (GPU; outputs under `outputs/forward_studies/`). The face-flux permeability
column is in the evidence archive (extended-data entry ED2.10).

## Stabilization weight: Figure 8, Tables 4 and S5

Both studies start from the stored solutions that `porevoronoi_fv/assisted/run_assisted_arms.py` writes to
`outputs/assisted/`, so run that script for c1-c6 first. The scripts run from their own folders:

```bash
cd reproduce/stabilization_weight/sweep
python -B run_sweep.py --case c1           # each of c1-c6 (run_all_cases.py starts several at once)
python -B run_sweep.py --sum

cd ../cross_validation
python -B run_cross_validation.py --case c1 --kind PN    # and --kind MN, --kind CV; each of c1-c6
python -B run_cross_validation.py --sum
python -B check_records.py                 # checks the stored records without solving; add a case (c1) to rebuild its operators
python -B cv_tables.py                     # prints the tables
```

`sweep/` tests the default weight and a fixed local weight rule by hold-out validation on c1-c6;
`cross_validation/` sweeps eleven weights for the Stokes-only (`--kind PN`) and assisted (`--kind MN`)
calculations and chooses a weight from the particle records alone (`--kind CV`). Their outputs go to
`outputs/stabilization_weight/`. The result records of the paper, and the sweep on the rock partitions, are in the
evidence archive (extended-data entries ED3.9-ED3.11 and ED5.8-ED5.11).

## Data that are not redistributed

| Data | Items | Source |
|---|---|---|
| Drainage experiment 073: segmented images and Kalman-smoothed tracks | Figures 2b-c, 9, 10; Tables 1 (one row), 7, S7 | Wang et al. (2026), arXiv:2603.12516; tomograms: Bultreys et al. (2024), PSI Public Data Repository, doi:10.16907/c0dfa6c8-25da-454e-82fa-fc5db7f7c6f2 |
| Glass-filter X-ray particle tracking | Figure S3, Table S8 | Bultreys et al. (2022), Zenodo doi:10.5281/zenodo.6010490 |
| Full Bentheimer and Berea images | Figure S1a-b | Jackson et al. (2021), Zenodo doi:10.5281/zenodo.5542624; Figshare "Berea Sandstone", doi:10.6084/m9.figshare.1153794.v2 |

The open-boundary cells used for the drainage experiment are in `porevoronoi_fv/open_boundary.py` and
`porevoronoi_fv/open_boundary_weighted.py`. The glass-filter scripts are in `reproduce/table_s8/`; download the data
set into `data/glass_filter/` and run, for example, `python reproduce/table_s8/glass_filter_projection.py --out
outputs/table_s8`.

## Settings you may need to change

* Slurm templates (`porevoronoi_fv/slurm_stokes_only.sh`, `gpu/ownership/h200/jobs/*.sh`): account, partition and
  Python interpreter of your cluster.
* Output folder: `outputs/` at the repository root; most scripts also take `--out`, `--out-dir` or `--root`.
* Folders used by the GPU scripts and by the CPU package: environment variables listed in
  [`../gpu/README.md`](../gpu/README.md) and [`../porevoronoi_fv/README.md`](../porevoronoi_fv/README.md).
* The glass-filter data set: `data/glass_filter/` (not shipped).

## The evidence archive

The evidence archive supplied with the manuscript holds the result records of the items marked "evidence archive"
above, the scripts that wrote them, and summary records of the drainage experiment that contain no particle
coordinate, velocity or image voxel. It also holds the extended-data entries (ED1.1, ED2.7, ...) that the
Supporting Information cites: per-case tables, notes and figures behind the printed ranges, one file per entry,
with an index. Large voxel and interface field arrays are listed there with size and checksum, not stored; the
scripts regenerate them. Until the paper is published the archive is available from the corresponding author,
Linqi Zhu (linqi.zhu@imperial.ac.uk), on request.

## About the code in this repository

The scripts and modules here are those that produced the results. For this release their file names, comments,
help texts and path settings were made readable and repository-relative; the computations are unchanged. The
evidence archive keeps the files exactly as they ran, with the checksums that the run records name; because names,
comments and paths changed, most code files here no longer match those checksums. Some code files are identical to
the files that ran, so their checksums still match the records: the two method modules `geodesic_face_operator.py`
and `hybrid_voronoi_trace.py` (in `porevoronoi_fv/` and, as identical copies, in `gpu/code/`), whose SHA-256 every
`run_record.json` lists, and the GPU ownership module, its timing driver and the site rule
(`gpu/ownership/code/ptv_ownership_gpu6.py`, `gpu/ownership/code/time_roi_vs_bfs.py`,
`gpu/code/hybrid_site_sources.py`), which `build_ownership_statistics.py` compares with the timing records. The
numerical data files (arrays and particle tracks) are stored byte for byte; `SHA256SUMS.txt` at the repository root
lists the checksum of every file.
