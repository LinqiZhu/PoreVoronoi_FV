# The six controlled cases

These are the inputs and stored results of the six controlled cases of the paper (Tables 5 and 6). The package
`porevoronoi_fv/` reads them through `porevoronoi_fv/config.json`; `docs/DATA.md` gives the full description of
every data file in the repository.

| Code | Folder | Image (z × y × x) | Pore voxels | Particles | Records | Cells N_c |
|---|---|---|---|---|---|---|
| c1 | `orthogonal_duct/` | 24 × 24 × 64 | 36,864 | 721 | 72,100 | 4937 |
| c2 | `skewed_duct/` | 24 × 24 × 64 | 36,369 | 712 | 71,200 | 5268 |
| c3 | `thin_wall/` | 24 × 24 × 64 | 35,392 | 692 | 69,200 | 4900 |
| c4 | `narrow_throat/` | 24 × 24 × 64 | 36,337 | 711 | 71,100 | 5117 |
| c5 | `maze/` | 24 × 24 × 64 | 35,328 | 691 | 69,100 | 5989 |
| c6 | `bentheimer_crop/` | 16 × 96 × 96 | 42,764 | 837 | 83,700 | 6739 |

Units and boundary conditions: solver units with voxel edge h = 1, kinematic viscosity ν = 1 and a uniform body
force f = (0.002, 0, 0); every image is periodic in x (the flow direction) with walls in y and z. Permeabilities are
in units of h². Voxel arrays are indexed (z, y, x).

## Files in each case folder

| File | Content |
|---|---|
| `reference_flow.npz` | The image and our reference Stokes flow on it, computed on the voxel grid. Keys: `mask` (bool, true = pore), `labels` (index of each pore voxel in raster order, −1 in solid), `owner`, `neigh` (the two pore voxels of each open voxel face, periodic in x), `area`, `avec`, `face_centroid`, `dvec`, `tproj`, `twall` (face geometry and transmissibilities of the voxel solver), `volume`, `centroid` (per pore voxel), `U` (velocity per pore voxel), `p` (pressure), `phi` (flux through each voxel face, owner to neigh), `div` (net outflow per pore voxel), `K_eff_x` (permeability along x), `metadata` (JSON string: steps, residuals, `steady_converged`), `case` (image name). |
| `particle_tracks.csv.gz` | Simulated particle tracks: one row per particle and frame. The package reads `frame_id`, `particle_id`, `x_mod_voxel`, `y_voxel`, `z_voxel` (positions in voxel units, voxel centres at integer coordinates, x wrapped into the image) and `ux_dns_frame_interp`, `uy_dns_frame_interp`, `uz_dns_frame_interp` (the reference velocity of the voxel that holds the record). The other columns (frame time, frame position scaled to 0-1, batch, unwrapped x, reference pressure, interpolation weight, track status) are bookkeeping of the generator. |
| `particle_tracks_generator.json` | Settings and counts of the track generator (seed, sampling, step, wall rule, numbers of particles, rows and distinct site voxels). |
| `stokes_only_solution.npz` | The stored Stokes-only solution on the cells of the particle tracks at the default stabilization weight τ = 1. Keys: `U` (cell velocities), `U_ref` (volume average of the reference velocity over each cell), `p` (balance multipliers; the cell pressure is −p), `phi` (flux between neighbouring cells), `volume` (cell volumes), `face_velocity`, `face_flux_sorted` (per facelet), `trace_coefficients` (the trace unknowns), `solver_formulation_id`, `linear_solver`, `linear_solver_relative_residual`. |
| `run_record.json` | Record of the run that produced the stored solution: SHA-256 of `reference_flow.npz` and `particle_tracks.csv.gz` (checked by `porevoronoi_fv/case_loader.py` before every use) and of the two method modules `porevoronoi_fv/geodesic_face_operator.py` and `porevoronoi_fv/hybrid_voronoi_trace.py` (byte for byte the files in this repository), record and site counts, method settings, and the Table 5 row of the run (`row`: counts, errors, solver statistics, timings). |

How the tracks were made, as in the paper: one particle per 51.1 pore voxels starts at a random pore voxel (seed
20260711) and is carried for 100 frames through the reference flow, one explicit Euler step per frame with the
reference velocity of its voxel, and stops at the first wall face it meets. Each record is a position with the
reference velocity of its voxel, so the records are noise-free samples of the reference flow. No measured data are
used.

Reference flows: the reference solve of c1 reached steady state (`steady_converged` true in its metadata). The
reference flows of c2-c6 were stored before full steady state: their solves stopped after 900 steps
(`steady_converged` false, largest momentum residual 4.3e-6 to 3.2e-5 in solver units, both in the metadata).
Section S3.1 of the Supporting Information bounds the effect of this on the reported errors.

## expected/

| File | What it holds | Compared by |
|---|---|---|
| `printed_table_values.json` | The values of Table 5 as printed, as strings; r_inf is given with three significant digits, one more than the table prints. | `porevoronoi_fv/check_archived_solutions.py` (check L1), `porevoronoi_fv/evaluate_stokes_only.py` (check `E0a_table`), `examples/quickstart.py` |
| `table_source_values.csv` | The source values of Table 5 at full precision, one row per case. | `porevoronoi_fv/check_archived_solutions.py` (at least 10 significant digits), `reproduce/shared/build_tables.py` |
| `permeability_readouts.json` | Permeability of the method, the stored reference permeability (`K_eff_x` of `reference_flow.npz`) and the reference permeability from the same face-flux readout as the method, with the signed errors in percent. | `porevoronoi_fv/check_archived_solutions.py` |

`MANIFEST.tsv` lists the size in bytes and the SHA-256 of the reference flow, particle tracks, stored solution and
run record of every case and of every file in `expected/`.

## Terms

The Bentheimer crop (the mask and label arrays of `bentheimer_crop/reference_flow.npz`, the image of c6) is
included only so that case c6 can be repeated; it is not licensed (`DATA_LICENSE.md`, Section 3). Everything else in
this folder, including the reference flow we computed on the crop, is our own data under CC BY 4.0
(`DATA_LICENSE.md`, Section 1).
