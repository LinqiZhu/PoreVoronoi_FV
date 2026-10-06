# Data in this repository

This repository releases the authors' own numerical data behind the manuscript (2026) *PoreVoronoi-FV:
Conservative pore-scale flow on cells fixed by tracked particles* and two small segmented rock images used as test
geometries. It holds the inputs of the six controlled cases (pore masks, the reference flows we computed on them,
particle windows simulated in those flows, stored Stokes-only solutions), a 64³ block of the public Berea image with
our reference flow and a simulated particle window, two segmented example masks, the masks and site sets of the GPU
ownership timings, and the result records behind the printed numbers.

**No experimental flow data are included.** There is no measured particle track, and no image, segmentation or
tomogram of the drainage experiment 073 or of the glass-filter and sand-pack data sets of other studies (Section 6).
Every particle window here is simulated in our reference flows. The only rock images are two small segmented test
geometries: a 16 × 96 × 96 crop of a segmented dry scan of Bentheimer sandstone (Section 3) and a 64³ block of the
public Berea image (Section 2). The licences are in [`../DATA_LICENSE.md`](../DATA_LICENSE.md).

## Overview

| Folder | Content | Origin | Terms |
|---|---|---|---|
| `data/controlled_cases/` | the six controlled cases: masks with reference flows, simulated particle windows with their generator records, stored Stokes-only solutions, run records; `expected/`: the printed Table 5 values | ours; the c6 mask is the Bentheimer crop | CC BY 4.0; crop: Section 3 |
| `data/berea64/` | 64³ Berea block, our reference flow along x on it, a simulated particle window (1000 particles) and its generator record | block derived from the Figshare "Berea Sandstone" record; flow and window ours | CC BY 4.0 with attribution |
| `examples/segmented_masks/` | Bentheimer 16 × 96 × 96 crop; fibrous-filter proxy 16 × 64 × 64 | crop: segmented dry scan, not public; proxy: ours | Section 3 |
| `gpu/ownership/h200/inputs/` | masks and farthest-point site sets of the H200 ownership timings | extracted from the files above | as their sources |
| `reproduce/` | result records behind the printed figures and tables | ours | CC BY 4.0 |

Conventions. Voxel arrays are indexed (z, y, x), x varying fastest, and x is the flow axis. The controlled cases are
periodic in x with walls in y and z and use solver units: voxel edge h = 1, kinematic viscosity ν = 1 and a uniform
body force f = (0.002, 0, 0) along x (`porevoronoi_fv/config.json`). Positions in the particle windows are in voxel
units with voxel centres at integer coordinates. Checksums: `SHA256SUMS.txt` at the repository root lists every file,
and `data/controlled_cases/MANIFEST.tsv` lists the case files; the repository stores every file byte for byte
(`.gitattributes`: `* -text`), so both can be recomputed on a clone.

## 1. The six controlled cases (`data/controlled_cases/`)

| Code | Folder | Case | Mask (z × y × x) | Pore voxels | Particles | Records | Sites = cells N_c |
|---|---|---|---|---|---|---|---|
| c1 | `orthogonal_duct` | orthogonal duct | 24 × 24 × 64 | 36,864 | 721 | 72,100 | 4937 |
| c2 | `skewed_duct` | skewed duct | 24 × 24 × 64 | 36,369 | 712 | 71,200 | 5268 |
| c3 | `thin_wall` | thin-wall case | 24 × 24 × 64 | 35,392 | 692 | 69,200 | 4900 |
| c4 | `narrow_throat` | narrow-throat case | 24 × 24 × 64 | 36,337 | 711 | 71,100 | 5117 |
| c5 | `maze` | maze case | 24 × 24 × 64 | 35,328 | 691 | 69,100 | 5989 |
| c6 | `bentheimer_crop` | Bentheimer crop | 16 × 96 × 96 | 42,764 | 837 | 83,700 | 6739 |

Pore voxels, particles, records and sites are those of the window generator records; the numbers of cells equal
Table 5 of the paper. The masks of c1 to c5 are synthetic; the mask of c6 is the Bentheimer crop of Section 3.
The six case folders hold about 70 MB in all.

Files in each case folder `data/controlled_cases/<case>/`:

| File | Content |
|---|---|
| `reference_flow.npz` | the mask, integer voxel labels, the voxel-face connectivity of the pore space and the reference solution computed by us with the voxel-grid Stokes solver of the flow notebook `gpu/notebooks/flow_solver.ipynb` (cell 1): velocity `U` per pore voxel, pressure `p`, face flux `phi`, divergence `div` and permeability `K_eff_x`; `metadata` (a JSON string) records the settings of the reference solve (2.2 to 3.2 MB each) |
| `particle_tracks.csv.gz` | the simulated particle window: one row per particle and frame, 100 frames (5.2 to 6.4 MB each) |
| `particle_tracks_generator.json` | the settings and counts of the window generator (below) |
| `stokes_only_solution.npz` | the stored Stokes-only solution on the cells of the particle tracks at τ = 1: cell velocities `U` and reference cell velocities `U_ref`, balance multipliers `p` (the cell pressure is −p, manuscript Eq. (19)), cell-pair fluxes `phi`, cell volumes, facelet velocities and fluxes, trace coefficients, solver identifier and attained relative residual (3.3 to 3.8 MB each) |
| `run_record.json` | the record of the forward run: SHA-256 of the two method modules and of the inputs, input counts, method settings and the result row |

Files in `data/controlled_cases/expected/`, read by `porevoronoi_fv/check_archived_solutions.py`,
`examples/quickstart.py` and `reproduce/shared/build_tables.py`:

| File | Content |
|---|---|
| `printed_table_values.json` | the Table 5 values that the checks compare against; the largest mass residual is given with three significant digits (1.14 × 10⁻¹⁶ for c1), one more than the paper prints |
| `permeability_readouts.json` | the record of the common face-flux permeability readout |
| `table_source_values.csv` | the source table of Table 5, at full precision |

`data/controlled_cases/MANIFEST.tsv` lists, for the reference flow, particle window, stored solution and run record
of every case and for every file of `expected/`, its path relative to `data/controlled_cases/`, its size in bytes,
its SHA-256 and its content. `data/controlled_cases/README.md`
describes the cases and the convergence of their reference solves.

The window columns are `frame_id`, `frame_absolute_time_s`, `frame_pv`, `particle_id`, `batch_id`,
`x_unwrapped_voxel`, `x_mod_voxel`, `y_voxel`, `z_voxel`, the reference velocity and pressure at the record
(`ux_dns_frame_interp`, `uy_dns_frame_interp`, `uz_dns_frame_interp`, `p_dns_frame_interp`), `track_interp_alpha`,
`track_status_before` and `track_status_after`.

How the windows were made (generator `gpu/code/build_particle_window.py`): one particle per 51.1 pore voxels starts
at a distinct random pore voxel (seed 20260711, uniform sampling), offset by up to 0.3 voxel per axis; each of the
100 frames is one explicit Euler step with the reference velocity of the particle's voxel, 0.1 voxel at the
75th-percentile pore speed, ending at the first impermeable voxel face; z and y are clipped and x is periodic. The
parameters are those of the generator records and of Section 2.1 of the paper and Section S3.1 of the Supporting
Information. The particles move in our computed reference flow; no measured velocity enters any window.

The generator records `particle_tracks_generator.json` hold the generator arguments and counts: `reference`,
`output_csv` (both repository paths), `frame_list`, `n_frames`, `n_particles`, `rows`, `unique_snapped_sites`,
`pore_voxels`, `unique_site_fraction_of_pores`, `sample_mode`, `seed`, `step_vox_per_frame`, `speed_scale_p75`,
`trajectory_integrator` (`explicit_euler_exact_voxel_face_stop`), `trajectory_connectivity`, `trajectory_wall_rule`
(`stop_at_first_impermeable_voxel_face`), `trajectory_face_crossings`, `trajectory_wall_collision_events`,
`trajectory_minimum_accepted_fraction`, `snap_disp_median_vox`, `snap_disp_p95_vox`, `snap_disp_max_vox`,
`sampled_speed_mean`, `sampled_speed_p50`, `sampled_speed_p95`, `K_eff_x_reference` and `elapsed_s`.
[`REPRODUCE.md`](REPRODUCE.md) shows how to make a window again from these settings.

## 2. The 64³ Berea block (`data/berea64/`)

| File | Content |
|---|---|
| `mask.npz` | the 64³ pore mask with metadata arrays: `source`, `raw_file`, `crop_origin_zyx`, `original_drive_axis`, `solver_axis_permutation_zyx`, `voxel_size_um`, `pore_value`, `preprocessing`, `selection_policy` |
| `reference_flow_x.npz` | our reference flow along x on this mask, with the same arrays as the controlled-case `reference_flow.npz` (3.8 MB) |
| `particle_tracks.csv.gz` | simulated particle window: 1000 particles, 100 frames, 100,000 records, 7779 distinct site voxels of 51,113 pore voxels; same columns as the controlled-case windows (7.6 MB, the largest file of the repository); the window of Table 2 and Figure 3 |
| `particle_tracks_generator.json` | its generator record (seed 20260711, uniform sampling, 0.1 voxel per frame), with the keys of Section 1 |

Origin. The block is cut from the file `Berea.raw` in the archive `Berea.7z` of the Figshare record "Berea
Sandstone" of the Imperial College Consortium on Pore-scale Imaging and Modelling (2014),
doi:[10.6084/m9.figshare.1153794.v2](https://doi.org/10.6084/m9.figshare.1153794.v2), licensed CC BY 4.0
(400³ one-byte voxels of 5.345 µm, pore value 0). It holds voxels 64–127, 224–287 and 288–351, counted from 0,
along the first (fastest-varying), second and third axes of that file; its x, y and z axes lie along the file's
third, second and first axes; and it keeps the 51,113 voxels of the block's one face-connected pore component that
spans all three axes (extended data ED7.4 of the Supporting Information).

`mask.npz` differs from the mask file that was timed for Table 2 and Figure 3 only in its text member `raw_file`,
which now names the source record instead of a local file path; every array is unchanged. The timing records keep
the SHA-256 of the timed file. `gpu/ownership/h200/inputs/berea_heldout_64.npz` holds the same block as a mask array
only. The 128³ public Berea block of the Supporting Information is not in this repository; ED7.4 gives its voxel
range in the same file.

## 3. Segmented example masks (`examples/segmented_masks/`)

**Bentheimer crop**, `bentheimer_dry_crop_16x96x96_origin_1562_59_349_pore0.npz` with its manifest
`…_manifest.json`. Arrays: `mask` (16 × 96 × 96, boolean), `raw_crop` (16 × 96 × 96, uint8), `origin_zyx`,
`shape_zyx`, `pore_value`. The crop was cut at origin (1562, 59, 349) from a segmented dry scan of Bentheimer
sandstone of 2714 × 1000 × 1000 voxels. Of its 147,456 voxels, 43,904 are pore (porosity 0.298); the 42,764 voxels
of the largest face-connected pore component (porosity 0.290), which spans x, y and z, form the sixth controlled
case. The parent scan is not public; it is a different image from the public Bentheimer image of Jackson et al.
(Section 6). The crop is included only so that case c6 can be repeated; it is not licensed under CC BY 4.0 or any
other licence, and no rights to the parent scan are granted (`DATA_LICENSE.md`, Section 3). The same mask is stored
three times, in three different containers: here, as the mask of
`data/controlled_cases/bentheimer_crop/reference_flow.npz`, and as a mask-only array in
`gpu/ownership/h200/inputs/bentheimer_crop.npz`.

**Fibrous-filter proxy**, `fibrous_filter_proxy_16x64x64.npz` with its manifest
`fibrous_filter_proxy_16x64x64_manifest.json`. Arrays: `mask` (16 × 64 × 64, boolean), `source_mask_2d`
(96 × 96, boolean), `pore_mask_2d_resized` (64 × 64, boolean). A synthetic pattern of ours: a two-dimensional fibre
pattern of our own (`source_mask_2d`), resized from 96 × 96 to 64 × 64 by nearest neighbour; its complement is the pore
space. Of 65,536 voxels, 41,248 are pore (porosity 0.629); the 38,912 voxels of the largest
face-connected component (porosity 0.594) are kept. It is a geometry and solver stress test, not an image of a real
material. It is the fibrous image of Table 1 and is used by the fibrous-proxy calculations of the Supporting
Information (`reproduce/supplementary/s3_verification/stabilization_sweep__fibrous_filter_proxy.csv`;
`reproduce/shared/stabilization_sweep.py` records its SHA-256).

## 4. Masks and site sets of the GPU ownership timings (`gpu/ownership/h200/inputs/`)

* Masks only (boolean, z × y × x): `thin_wall.npz`, `maze.npz` (24 × 24 × 64, taken from the controlled-case
  reference flows), `bentheimer_crop.npz` (16 × 96 × 96; Section 3) and `berea_heldout_64.npz` (64³; Section 2).
* Site sets `<image>__mask_graph_fps_n{200,800}__ownership.npz`, each with one array `sites` (int64 flat voxel
  indices, length 200 or 800): farthest-point site sets on the pore graph of each mask, taken from the comparison
  of straight-line and through-the-pore ownership (Figure 1).
* These are the inputs of the H200 timings (Figure 3d and Supporting Information Section S1.3). The drivers in
  `gpu/ownership/h200/` check every input against `inputs_manifest.json`, `input_hashes.json` and
  `paper_inputs_manifest.json` before they run; the five particle prefixes of Table 2 are read from
  `data/berea64/mask.npz` and `data/berea64/particle_tracks.csv.gz`. See
  [`../gpu/ownership/README.md`](../gpu/ownership/README.md).

## 5. Result records (`reproduce/`)

The CSV and JSON files under `reproduce/` are our results: one folder per printed figure or table, `supplementary/`
for numbers quoted in the Supporting Information text, and `shared/` for the scripts that build several tables and
figures. [`figures_and_tables.csv`](figures_and_tables.csv) names, for every printed item, the files that hold its
numbers and the scripts that wrote them; [`REPRODUCE.md`](REPRODUCE.md) gives the commands.

## 6. Not in this repository

### Experimental data (not redistributed)

| Data | Source | Where it is used |
|---|---|---|
| Drainage experiment 073: X-ray tomograms | Bultreys et al. (2024), *4D microvelocimetry reveals multiphase flow field perturbations in porous media*, PNAS 121, e2316723121, doi:[10.1073/pnas.2316723121](https://doi.org/10.1073/pnas.2316723121); tomograms in the PSI Public Data Repository, doi:[10.16907/c0dfa6c8-25da-454e-82fa-fc5db7f7c6f2](https://doi.org/10.16907/c0dfa6c8-25da-454e-82fa-fc5db7f7c6f2) | Figures 2b–c, 9 and 10, Table 7 (Section 6.4) and Table S7 (Supporting Information Section S6.3) |
| Drainage experiment 073: segmentations and Kalman-smoothed tracks | Wang et al. (2026), *Learning pore-scale multiphase flow from 4D velocimetry*, arXiv:2603.12516, doi:[10.48550/arXiv.2603.12516](https://doi.org/10.48550/arXiv.2603.12516); released only by that study | as above |
| Glass-filter X-ray particle tracking: pore mask and tracks | Bultreys et al. (2022), *X-ray tomographic micro-particle velocimetry in porous media*, Physics of Fluids 34, 042008, doi:[10.1063/5.0088000](https://doi.org/10.1063/5.0088000); data set Zenodo doi:[10.5281/zenodo.6010490](https://doi.org/10.5281/zenodo.6010490) | Figure S3, Table S8 |
| Sand-pack X-ray particle tracking | Bultreys et al. (2022), data set Zenodo doi:[10.5281/zenodo.6010425](https://doi.org/10.5281/zenodo.6010425) | solver cross-check of the Supporting Information |

From the glass-filter study no file is included: neither the tracer table and the 96³ pore-mask block cut from the
data set nor the results derived from them. The scripts that read the published data set,
`reproduce/table_s8/glass_filter_projection.py` and `reproduce/table_s8/glass_filter_registration.py`, are included;
they expect the downloaded data set in `data/glass_filter/`. For the drainage experiment the evidence archive
supplied with the manuscript holds the run and scoring scripts, logs and aggregate scorer outputs, with no particle
coordinate, particle velocity or image voxel.

### Third-party and parent images (not copied)

* The full Figshare "Berea Sandstone" record, doi:[10.6084/m9.figshare.1153794.v2](https://doi.org/10.6084/m9.figshare.1153794.v2) (CC BY 4.0); only the derived 64³ block of Section 2 is here.
* The full Bentheimer image (225³ voxels) of Jackson et al. (2021), *Multi-resolution X-Ray micro-CT images of
  Bentheimer Sandstones*, Zenodo doi:[10.5281/zenodo.5542624](https://doi.org/10.5281/zenodo.5542624) (CC BY 4.0).
* The segmented Bentheimer dry scan (2714 × 1000 × 1000 voxels) from which the crop of Section 3 was cut; not
  public.

### Large field arrays and per-run records

Voxel and trace field arrays of the solves (for example the state files of the refinement runs and the checkpoints
of the controlled-case runs) are not copied; they are listed with size and checksum in the evidence archive supplied
with the manuscript, and the scripts regenerate them. The per-repeat timing records from which
`reproduce/shared/build_ownership_statistics.py` collects the Table 2 statistics are not included in this
repository; their summaries are in `reproduce/table_02/records/`.
