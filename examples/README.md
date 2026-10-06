# Examples

| Path | What it is |
|---|---|
| `quickstart.py` | Rebuilds one controlled case of the paper with the CPU package `porevoronoi_fv/`, solves it and checks the printed values |
| `segmented_masks/` | Two segmented pore masks, each with a manifest: a 16 x 96 x 96 Bentheimer crop (`bentheimer_dry_crop_16x96x96_origin_1562_59_349_pore0.npz`) and a 16 x 64 x 64 fibrous-filter proxy (`fibrous_filter_proxy_16x64x64.npz`) |

## quickstart.py

### What it does

`quickstart.py` runs one of the six controlled cases end to end. It makes the same calls as the package scripts
`porevoronoi_fv/run_stokes_only.py` (stage `e0`) and `porevoronoi_fv/evaluate_stokes_only.py` (stage `a`), without
their checkpoint files, and adds no numerics of its own:

1. Load `porevoronoi_fv/config.json` (`config_io.load_cfg`) and the case inputs (`case_loader.load_case`). The
   SHA-256 of the reference field and of the particle tracks are checked against the case's run record
   `data/controlled_cases/<case>/run_record.json`.
2. Map every record of the particle tracks to its voxel (the site rule), assign every pore voxel to the site it
   reaches in the fewest six-neighbour steps through the pore (ties go to the lower site index), and build the
   facelets and the connected-P1 trace
   (`run_stokes_only.build_published`, which calls `case_loader.select_records`, `cell_complex.sites_of` and
   `cell_complex.build`). The script stops unless the numbers of cells, neighbour pairs, facelets, connected patches
   and trace modes equal those of the paper's complex.
3. Assemble the Stokes system with the vectorised assembler (`stokes_solve.assemble`, kind `fast`).
4. Solve the Stokes-only calculation (`alpha = 0`) with the paper's solver settings from `config.json`: MINRES,
   relative tolerance 1e-14, at most 50000 iterations, one refinement step (`stokes_solve.solve_entry`). The solver
   raises an error if the relative residual of the saddle-point (KKT) system exceeds max(20 rtol, 1e-12)
   (`solve_moment_constrained_hybrid_stokes` in `porevoronoi_fv/hybrid_voronoi_trace.py`).
5. Compute the readouts of Table 5 of the paper (`readouts.table_row`) and compare each with the value printed in the
   paper, stored in `data/controlled_cases/expected/printed_table_values.json`, with the literal comparison that the
   package uses (`readouts.literal_fixed`, `readouts.literal_sci`).

With `--assisted` it then runs the one-record variant of the assisted calculation: in each cell the record nearest
the cell centroid is kept (`observations.nearest_centroid_selection`), and its velocity enters the same Stokes
calculation on the same cells with weight theta = 1000 (`alpha` in `config.json`). This is the one-record calculation
(label `AN`) of `porevoronoi_fv/assisted/assisted_arms.py`, called through `assisted_arms.solve_arm`. For both
calculations the script prints the velocity error on the pore voxels that hold no record, e_u,vox(U)
(`assisted_arms.leave_out_e_ff`).

### Running it

From the repository root:

```bash
python examples/quickstart.py                  # c1, orthogonal duct
python examples/quickstart.py --case c6        # Bentheimer crop
python examples/quickstart.py --assisted       # c1, plus the one-record assisted calculation
```

| Option | Meaning |
|---|---|
| `--case c1 ... c6` | Controlled case (default `c1`) |
| `--assisted` | Also solve the one-record assisted calculation and print e_u,vox(U) for both calculations |
| `--cfg PATH` | Package configuration file (default `porevoronoi_fv/config.json`) |
| `--out DIR` | Folder for the JSON summary (default `outputs/quickstart`) |

Requirements: Python 3 with NumPy and SciPy. The script runs on the CPU only and needs no GPU. Like every script of
the package it sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless they are already
set. All inputs are in `data/controlled_cases/` and are only read.

### The six cases

| Code | Folder in `data/controlled_cases/` | Case | Voxels (z, y, x) | Pore voxels N_f | Records | Cells N_c |
|---|---|---|---|---|---|---|
| `c1` | `orthogonal_duct` | Orthogonal duct | 24 x 24 x 64 | 36864 | 72100 | 4937 |
| `c2` | `skewed_duct` | Skewed duct | 24 x 24 x 64 | 36369 | 71200 | 5268 |
| `c3` | `thin_wall` | Thin-wall case | 24 x 24 x 64 | 35392 | 69200 | 4900 |
| `c4` | `narrow_throat` | Narrow-throat case | 24 x 24 x 64 | 36337 | 71100 | 5117 |
| `c5` | `maze` | Maze case | 24 x 24 x 64 | 35328 | 69100 | 5989 |
| `c6` | `bentheimer_crop` | Bentheimer crop | 16 x 96 x 96 | 42764 | 83700 | 6739 |

All six images are periodic in x, the driving direction, with walls in y and z. Quantities are in solver units:
voxel edge h = 1, kinematic viscosity nu = 1, uniform body force f = (0.002, 0, 0), so permeability is in units
of h^2.

The reference flows and particle tracks are our own numerical data; no measured flow or particle-tracking data
are used. `c1`-`c5` are synthetic duct geometries. `c6` is the 16 x 96 x 96 Bentheimer sandstone crop, included only
so that case `c6` can be repeated (see `DATA_LICENSE.md`); it is cut from a segmented dry scan that is not public.
For each image the reference field (`reference_flow.npz`) is the Stokes flow that we computed with a separate voxel
solver on the image's own voxel grid; `data/controlled_cases/README.md` describes the files and how far each
reference flow is converged. The particle tracks (`particle_tracks.csv.gz`) hold simulated particles: one particle
starts at a random voxel for every 51.1 pore voxels and is carried for 100 frames through the reference field, one
Euler step per frame with the velocity of its voxel. Each record is a position with its voxel's reference velocity,
so the records are noise-free samples of the reference field.

### How long it takes

Estimate, not measured with this script: a few minutes per case on one CPU core, most of it in MINRES. The
recorded single-thread runs of the paper give the scale:

- Our recorded single-thread solves (Intel Xeon Platinum 8470) of the calculation this script repeats took
  87.8-160.2 s per case from sites to solution: `c1` 106.3 s (7125 MINRES iterations), `c6` 160.2 s (9222
  iterations).
- `--assisted` adds a second complex build, an assembly, the reconstruction operators and a second MINRES solve
  (4797 iterations for `c1`, 8213 for `c6`; extended data table ED5.7 of the evidence archive supplied with the
  manuscript), so allow roughly twice the time of the Stokes-only run.

Peak memory has not been measured for this script. For scale, the velocity matrix A has 10,198,933 non-zeros for
`c1` and 12,213,063 for `c6` (extended data table ED4.6 of the same archive).

### What it prints

Progress lines `[1/4]` to `[4/4]` give the case size, the number of records, particles, frames and sites, the
complex counts (cells, neighbour pairs, interface facelets, connected patches, trace modes), the number of trace
unknowns and of non-zeros in A, and the time of each step. During MINRES, `stokes_solve.minres_heartbeat` prints a
line `{'beat': <label>, 'i': <iteration>, 'elapsed_s': <seconds>}` every 500 iterations or 60 s.

Then a table with one row per quantity: the computed value, the printed value and the check. The printed values for
`c1` are:

| Row | Printed value for `c1` | Source of the printed value | Check |
|---|---|---|---|
| cells N_c | 4937 | Table 5 (`printed_table_values.json`) | string equality |
| trace unknowns N_Z | 142443 | run record `data/controlled_cases/orthogonal_duct/run_record.json` | equality |
| MINRES iterations | 7066 | archived run (run record) | not checked |
| MINRES relative KKT residual | <= 1.0e-12 | max(20 rtol, 1e-12) with rtol = 1e-14 | `info == 0` and residual <= limit |
| r_inf^mass = max_i \|(D z)_i\| / V_i | 1.14e-16 | Table 5 (`printed_table_values.json`) | reported, not checked |
| eta_m (mass backward error) | <= 1e-14 | check `E0a_mass` of `porevoronoi_fv/evaluate_stokes_only.py` | value <= 1e-14 |
| e_u,cell (%) | 3.89 | Table 5 (`printed_table_values.json`) | literal |
| e_phi (%) | 4.33 | Table 5 (`printed_table_values.json`) | literal |
| e_K,flux^s (%), signed | -4.19 | Table 5 (`printed_table_values.json`) | literal |
| e_K,stored^s (%), signed | -4.19 | Table 5 (`printed_table_values.json`) | literal |

With `--assisted` two more tables follow. The first gives the number of observations (one record per cell, so equal
to N_c), the MINRES iterations, the recomputed KKT relative residual (limit 1e-12) and eta_m (limit 1e-12), the
checks in `porevoronoi_fv/assisted/assisted_protocol.json`. The second gives e_u,vox(U):

| Row | Printed value for `c1` | Source of the printed value | Check |
|---|---|---|---|
| pore voxels holding a record (%) | 13.39 | ED5.7 | literal |
| Stokes-only | 4.53 | Table 6, tau = 1 | literal |
| assisted, one record per cell | 2.76 | ED5.7, column `U_AN` | literal |
| ratio assisted / Stokes-only | 0.61-0.89 over the six cases | text with Table 6 | not checked |

The last line gives the number of checks that pass, the total time and the path of the JSON summary
(`outputs/quickstart/<case>.json` by default). The summary holds the full-precision readouts, the solver receipt,
the timings, the check results, the Python, NumPy and SciPy versions and the SHA-256 of every package source file.
The exit status is 0 when every check passes and 1 otherwise.

### How the checks work

A literal check passes when the computed value, rounded to the number of decimals printed in the paper, equals the
printed string, sign included. This is check `E0a_table` of `porevoronoi_fv/evaluate_stokes_only.py`.

Two quantities are not compared with a printed value, because they differ between runs that are both correct.
The MINRES iteration count depends on the floating-point environment: the paper's own runs of the `c1` Stokes-only
calculation recorded 7066 iterations (run record) and 7125 (ED5.7); over the six cases the run-record counts and
those of our later CPU solves differ by 0.1-2.3%. The largest cell imbalance r_inf^mass is at round-off (5.8e-18 to
1.9e-16 in Table 5), so its digits are not reproducible. As in `porevoronoi_fv/evaluate_stokes_only.py`, the mass
balance is checked instead through the backward error eta_m <= 1e-14 (check `E0a_mass`).

The check labels (`G1`, `E0`, `E0a_*`, `L1`-`L3`, `PN`/`AN`/`MN`) are explained in `porevoronoi_fv/README.md`.

### Related scripts

- `porevoronoi_fv/run_stokes_only.py`, `porevoronoi_fv/evaluate_stokes_only.py` and
  `porevoronoi_fv/slurm_stokes_only.sh`: the batch version of the same Stokes-only calculation for all six cases,
  with checkpoints, the comparison of the vectorised and loop assemblies (check `G1`) and a spectral bound on the
  cell-velocity difference from the archived solution.
- `porevoronoi_fv/check_archived_solutions.py`: recomputes the Table 5 values from the archived solutions
  `data/controlled_cases/<case>/stokes_only_solution.npz`, without a solve.
- `porevoronoi_fv/check_selectors_and_assembly.py`: tests that run no flow solve.
- `porevoronoi_fv/assisted/run_assisted_arms.py`: the Stokes-only, one-record and all-record calculations (labels
  `PN`, `AN`, `MN`) on the trajectory partitions of the six cases.
