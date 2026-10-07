# gpu: the GPU (CuPy) forward pipeline

This folder holds the GPU implementation used for the forward runs of the paper: a particle window gives the sites,
the GPU ownership code builds the cells, and the Stokes system is assembled and solved and compared with the
reference flow on the voxel grid. It also holds the flow notebook with the image-resolved reference solver, the study
drivers built on the pipeline, and the GPU ownership code (ROI-JFA and exact frontier search) with its timing drivers.
The controlled-case values of Table 5 can be checked without a GPU with the CPU package in
[`../porevoronoi_fv/`](../porevoronoi_fv/).

## Requirements and how to run

* An NVIDIA GPU with a driver for CUDA 12 and CuPy for CUDA 12: `pip install -r environment/requirements-gpu.txt`
  (CuPy 14.1.1 on top of the CPU requirements). The recorded runs used an NVIDIA GeForce RTX 5080 Laptop GPU
  (CUDA 12.9 runtime) and one NVIDIA H200 of a Linux cluster (CUDA 12.8); the versions are listed in
  [`../environment/ENVIRONMENT.md`](../environment/ENVIRONMENT.md).
* Run every command from the repository root, for example `python gpu/code/run_hybrid_voronoi_forward.py --help`.
  Default output folders lie under `outputs/` (not tracked); most scripts take `--out-dir`, `--out` or
  `--delivery-root`. The ownership drivers that write elsewhere are named in
  [`ownership/README.md`](ownership/README.md).
* The scripts find the pipeline modules by folder name: each one puts `gpu/code/` (and, for the ownership drivers,
  its own folder) on the import path, so the folder names `code`, `studies` and `ownership` must stay as they are.
* Unit tests of the pipeline: `python -m pytest tests` (the CuPy tests are skipped without a GPU).

## Folders

| Folder | Contents |
|---|---|
| `code/` | pipeline modules and runners (below) |
| `notebooks/flow_solver.ipynb` | flow notebook: GPU geometry, face reindexing, reference-to-coarse mapping, flux errors; code cell 1 is the image-resolved reference solver behind the reference fields of the controlled cases (`data/controlled_cases/<case>/reference_flow.npz`) |
| `studies/` | study drivers built on the pipeline (below) |
| `ownership/` | GPU ownership code and its timing drivers: `ownership/code/` (Table 2, laptop GPU) and `ownership/h200/` (self-contained bundle run on the H200: Figure 3d); see [`ownership/README.md`](ownership/README.md) |

## code/

| File | What it does |
|---|---|
| `run_hybrid_voronoi_forward.py` | Forward runner: particle window to sites (`hybrid_site_sources.py`), GPU ownership labels to cells, assembly and MINRES solve (`hybrid_voronoi_trace.py`), errors against the reference field, one run manifest per run. Behind the forward rows in `reproduce/table_05/forward_rows.csv` and the forward dimensions of the Berea block (`reproduce/supplementary/s4_cost/forward_manifests/`); it also has the options of the cross-flow check (`--site-flow-axis`, `--particle-window-provenance`). |
| `run_segmented_selector_geodesic_rows.py` | Loads the flow namespace through `flow_runner.py`, the solver configuration and the reference fields; imported by every runner. |
| `flow_runner.py` | Executes code cells 1, 3, 5, 7 and 9 of `notebooks/flow_solver.ipynb` in one namespace and installs the ownership labels from `exact_frontier_backend.py` (the exact six-neighbour frontier search on the GPU used by the forward runs), the GPU face-connected split and the cell order; run on its own it computes the reference fields and coarse runs of the cases it defines. |
| `exact_frontier_backend.py` | The exact six-neighbour frontier search on the GPU (multi-source breadth-first search over the pore voxels, ties to the lower site index), loaded by `flow_runner.py`; it gives the cells of the forward runs. |
| `hybrid_voronoi_trace.py`, `geodesic_face_operator.py` | The two method modules (trace space, assembly and solve; face groups and face metric). |
| `hybrid_site_sources.py` | The site rule: particle window to sites (rounding to voxels, zero snap distance, periodic wrap in x); also used by the ownership drivers. |
| `build_particle_window.py` | Generator of the simulated particle windows (Figure 2, Table 1): explicit Euler steps with the voxel reference velocity, each track stopped at the first wall face. The settings of every window in `data/` are in the `particle_tracks_generator.json` beside it. |
| `run_particle_trace_transfer.py`, `bentheimer_reference_adapter.py` | Sampled-state transfer: particle cell states projected onto the conservative trace space (SI S5.2-S5.3; rows in `reproduce/supplementary/s5_particle_reconstruction/`). |
| `build_cross_flow_particle_window.py`, `audit_cross_flow_prediction.py` | Cross-flow check: a window mapped to another flow axis and the prediction with the same sites checked (`reproduce/supplementary/s1_ownership/cross_flow/`). |

`hybrid_voronoi_trace.py` and `geodesic_face_operator.py` are byte-identical copies of the modules in
`porevoronoi_fv/` (the pipeline imports them from this folder); every `data/controlled_cases/<case>/run_record.json`
lists their SHA-256. `hybrid_site_sources.py` is also copied, unchanged, into `ownership/h200/src/`.

Reference fields: `python gpu/code/flow_runner.py --out-dir outputs/reference_data` runs the reference solver (cell 1
of the notebook) and the coarse runs for the cases it defines (orthogonal and skewed duct, thin wall, narrow throat,
maze, and the two example masks in `examples/segmented_masks/`; `--case-filter` selects cases, `--dry-run` lists
them). The notebook writes the fields to `<out-dir>/reference_data/<case>__reference.npz`, which is where the forward
runners look by default (`--reference-npz`, `--reference-dir`). Another image needs its own entry in
`build_case_specs` of `flow_runner.py`.

Particle windows: give `build_particle_window.py` the settings recorded in
`data/controlled_cases/<case>/particle_tracks_generator.json` (or `data/berea64/particle_tracks_generator.json`),
for example for the orthogonal duct
`python gpu/code/build_particle_window.py --reference data/controlled_cases/orthogonal_duct/reference_flow.npz --out-dir outputs/windows --n-particles 721 --sample-mode uniform --seed 20260711 --step-vox-per-frame 0.1 --frame-list 0:99:1`.

## studies/

The study drivers share `study_common.py` (paths, hashing, environment record, forward evaluation from explicit
sites or partitions, pore-graph searches, farthest-point order) and `stability_core.py` (spectral bounds). They work
on the orthogonal duct, the skewed duct and the Bentheimer crop of `data/controlled_cases/` and write to
`outputs/forward_studies/` (`--delivery-root`). Run them in this order:

| Step | Command | Result |
|---|---|---|
| 1 | `python gpu/studies/build_nested_trajectory_site_family.py` | nested site families from the trajectory prefixes (Figure 5, the grey band of Figure 6a, SI S3.2) |
| 2 | `python gpu/studies/run_controlled_trajectory_refinement.py` | forward runs on every level of these families (study A; Figure 5, SI S3.2) |
| 3 | `python gpu/studies/build_site_rule_partitions.py` | site and partition rules at matched system size, chosen from the geometry alone (study B) |
| 4 | `python gpu/studies/run_site_rule_comparison.py` | forward runs of these rules (study B; Table S3, stored rows in `reproduce/table_s3/site_rule_comparison.csv`) |
| 5 | `python gpu/studies/audit_gauge_fixed_stability.py` | gauge-fixed stability bounds of the study rows (study C; SI S2.4, stored rows in `reproduce/table_s3/stability_audit.csv`) |

`forward_runs.py` evaluates one forward point (sites or a partition) with its admissibility check and run manifest;
`reproduce/figure_06/run_refinement.py` uses it. Log lines of the three studies start with `[phaseA]`, `[phaseB]` and
`[phaseC]`. The result records of Figure 5 are in the evidence archive supplied with the manuscript.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `PVFV_GPU_ROOT`, `PVFV_COMPUTE_ROOT` | the `gpu/` folder | `study_common.py`: root that holds `code/` |
| `PVFV_OUTPUT_ROOT` | `outputs/forward_studies` | `study_common.py`: output root of the studies |
| `PVFV_PROTOCOL_ROOT` | `outputs/protocol` | `study_common.py`: recorded in the environment record |
| `PVFV_STUDIES_DIR` | `gpu/ownership/h200/src` | set by `ownership/h200/common.py` |
| `PVFV_OWNERSHIP_CODE` | `gpu/ownership/code` | `ownership/code/check_free_space_bound.py`: folder of the ownership module |
| `PVFV_LABEL_BACKEND` | `exact_frontier_gpu` (`exact_geodesic`: the relaxation of the notebook) | ownership labels of the forward runners and of `flow_runner.py` |
| `PVFV_FLOW_PROFILE`, `PVFV_FLOW_PROGRESS`, `PVFV_FLOW_PROGRESS_EVERY_S` | `production`, `1`, `30` | profile and progress lines of `flow_runner.py` |
