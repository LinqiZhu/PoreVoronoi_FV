# porevoronoi_fv: the CPU implementation

This folder holds the CPU implementation of the method (NumPy and SciPy) behind every controlled-case result of the
paper. It is a folder of flat modules, not an installable package: the modules import each other by their file
names, so run the scripts as `python -B porevoronoi_fv/<script>.py` (Python puts the script's folder on the import
path) or put this folder on `PYTHONPATH` before importing a module. All settings and input paths come from
`config.json` in this folder; relative paths in it are resolved against the folder, inputs are read from
`data/controlled_cases/` and outputs are written to `<repo>/outputs/` (not tracked).

## Modules

| Module | What it does |
|---|---|
| `config_io.py` | Reads `config.json`; SHA-256 of files; restart-safe JSON and NPZ checkpoints; environment record; progress lines. |
| `case_loader.py` | Loads one controlled case (mask, reference flow, particle records, stored solution), checks the input SHA-256 against the run record and selects record subsets. |
| `cell_complex.py` | Sites to cells: exact six-neighbour graph-distance ownership, the paper's cell order, neighbour pairs, connected-P1 trace geometry and partition checks. |
| `readouts.py` | The error readouts of the paper (e_u,cell, e_phi, e_K stored and face-flux, r_inf^mass, eta_m, full-field voxel error) and the comparison with printed values. |
| `observations.py` | Record selectors (one record per cell, nearest the centroid), observation operator, observation weight gamma and the assisted normal equations. |
| `stokes_solve.py` | Assembly (vectorised or loop), check `G1`, the paper's CPU MINRES with progress lines, the Stokes-only and assisted solve entry, the reduced KKT matrix. |
| `geodesic_face_operator.py` | Signed-normal face groups, geodesic face metric, flux readout and flux-to-velocity reconstruction. |
| `hybrid_voronoi_trace.py` | Facelets, connected patches, trace modes, loop assembly (viscous block, balance operator, stabilization) and the MINRES saddle-point solve with its residual check. |
| `fast_assembly.py` | Vectorised assembly equal to the loop assembly of `hybrid_voronoi_trace.py` to round-off, the comparison used by check `G1`, and the direct KKT matrix. |
| `pore_ownership.py` | Graph-distance ownership (lower site index wins ties), farthest-point sites, cell geometry, pressure components, component-aware MINRES. |
| `velocity_recovery.py` | Recovery operators R (cell velocity) and G (cell gradient), point operator H, relative L2 error norm, NumPy adapter for the trace builder. |
| `point_location.py` | Physical positions to voxels and an auxiliary readout; not used by the scripts of this folder. |
| `open_boundary.py` | Open-boundary complexes (one reservoir cell around an image block) for the drainage-experiment calculations of the paper (Table 7, Figures 9-10); the experimental images are not in this repository. |
| `open_boundary_weighted.py` | `open_boundary.py` with an optional per-cell stabilization multiplier (the drainage row of Table 4 and Table S7). |

`geodesic_face_operator.py` and `hybrid_voronoi_trace.py` are byte for byte the modules whose SHA-256 every
`data/controlled_cases/<case>/run_record.json` lists.

## Scripts and what they reproduce

| Script | Solve? | What it does |
|---|---|---|
| `check_selectors_and_assembly.py` | no | Record selectors on all six cases, check `G1` on a small synthetic instance (and that a perturbed matrix fails it), identity of the Stokes-only operator. |
| `check_gate_logic.py` | no | Self-test of `evaluate_stokes_only.py`: ARPACK against dense eigenvalues on a small instance, and its checks applied to a state built from the stored solution (must pass) and to a perturbed one (must fail). |
| `check_archived_solutions.py` | no | Table 5 recomputed from the stored solutions (`L1`), the cell complexes rebuilt and compared with the stored ones (`L2`), every particle record checked against the reference flow (`L3`). |
| `run_stokes_only.py` | yes | Per case: stage `g1` (check `G1` on the case's complex) and stage `e0` (a fresh Stokes-only solve; restartable). |
| `evaluate_stokes_only.py` | no flow solve | Applies the declared checks to the fresh solve of `run_stokes_only.py`: printed Table 5 values, mass backward error, solver residual, distance to the stored solution (stages `protocol`, `eig`, `a`, `summary`). |
| `slurm_stokes_only.sh` | yes | Slurm job that runs `run_stokes_only.py` and `evaluate_stokes_only.py` for all six cases (8 cores, 6 h); submit from this folder after `mkdir -p ../outputs`. |
| `assisted/run_assisted_arms.py` | yes | The Stokes-only, one-record and all-record calculations of one case on the cells of its particle tracks; the one-record variant at τ = 1 is behind Table 6 and extended data table ED5.7 of the evidence archive supplied with the manuscript. `--case sum` collects all cases. |

`assisted/assisted_arms.py` is the library of the assisted calculations (also imported by `examples/quickstart.py`
and by `reproduce/stabilization_weight/`), and `assisted/assisted_protocol.json` declares their settings and checks.

From the repository root:

```bash
python -B porevoronoi_fv/check_selectors_and_assembly.py          # no solve
python -B porevoronoi_fv/check_gate_logic.py --case c1             # no solve
python -B porevoronoi_fv/check_archived_solutions.py               # Table 5 from the stored solutions, no solve
python -B porevoronoi_fv/run_stokes_only.py --case c1 --stage g1
python -B porevoronoi_fv/run_stokes_only.py --case c1 --stage e0   # fresh Stokes-only solve of c1
python -B porevoronoi_fv/evaluate_stokes_only.py --stage protocol
python -B porevoronoi_fv/evaluate_stokes_only.py --case c1 --stage a
python -B porevoronoi_fv/evaluate_stokes_only.py --stage summary
python -B porevoronoi_fv/assisted/run_assisted_arms.py --case c1  # Stokes-only, one-record, all-record
```

Every script takes `--cfg PATH` (default `config.json` here) except the assisted runner, which always reads
`config.json`. Each script sets one BLAS thread unless `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or `MKL_NUM_THREADS`
is already set.

## Labels used in the outputs

| Label | Meaning |
|---|---|
| `c1`-`c6` | Case ids: `orthogonal_duct`, `skewed_duct`, `thin_wall`, `narrow_throat`, `maze`, `bentheimer_crop` (the `tag` in `config.json`). |
| `pub`, `pfx:K`, `frm:F` | Record subsets: all records; records of particles 0 to K-1; records of frame F. Terms joined with `+` intersect. |
| `G1` | Check: vectorised assembly equals the loop assembly (`fast_assembly.gate`). |
| `E0` | The fresh Stokes-only solve of a case (stage `e0`). |
| `E0_solver`, `E0a_table`, `E0a_mass`, `E0a_cellU` | Checks on that solve: solver residual; printed Table 5 values; mass backward error eta_m <= 1e-14; distance of the cell velocities to the stored solution within a computed bound. |
| `L1`, `L2`, `L3` | Checks of `check_archived_solutions.py` (printed values, complex, particle records). |
| `PN`, `AN`, `MN` | Calculations of the assisted runner: Stokes-only (no record), one record per cell, every record. |
| `D_part`, `D_basis`, `D_obs`, `D_solver`, `D_mass`, `D_pure_matches_E0` | Checks of the assisted runner, declared in `assisted/assisted_protocol.json`. |

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `PVFV_CONFIG` | `config_io.py` | Configuration file when `--cfg` is not given (default `config.json` here). |
| `PVFV_ASSISTED_OUT` | `assisted/assisted_arms.py` | Output folder of the assisted runner (default `<repo>/outputs/assisted`). |
| `PVFV_QOS` | `assisted/assisted_arms.py` | Optional scheduler label, copied into the output records. |
| `PVFV_DIR` | `slurm_stokes_only.sh` | This folder (default: the submit directory). |
| `PVFV_PYTHON` | `slurm_stokes_only.sh` | Python interpreter (default `python3`). |
| `PVFV_EIG_TIMEOUT`, `PVFV_EIG_VMEM_KB` | `slurm_stokes_only.sh` | Time limit (s) and memory limit (kB) of the optional `eig` stage. |

Outputs: `run_stokes_only.py` and `evaluate_stokes_only.py` write `outputs/<case>/` and `outputs/summary.json`;
the check scripts write `outputs/check_archived_solutions/`, `outputs/check_selectors_and_assembly/` and
`outputs/dry/`; the assisted runner writes `outputs/assisted/<case>/`.
