# gpu/ownership: exact ownership on the GPU and its timings

`ptv_ownership_gpu6.py` assigns every pore voxel to the site it reaches in the fewest six-neighbour steps through the
pore (ties go to the lower site id) and returns the owner and the distance of every voxel. It holds three exact
methods: ROI-JFA (`certified_roi_jfa_gpu_6`, `certified_l1_roi_frontier_gpu_6`: free-space proposal, path
certificate, closure of the uncertified voxels), the host-driven frontier search (`exact_frontier_dijkstra_gpu_6`)
and the persistent-frontier search (`exact_persistent_frontier_gpu_6`). Requirements: an NVIDIA GPU with CuPy for
CUDA 12 (`environment/requirements-gpu.txt`); `h200/run_quadrature.py` needs no GPU.

## Which version is behind which result

The module exists in two versions. Their numerical kernels are identical; they differ only in how the number of
streaming multiprocessors (SM) of the GPU is looked up.

| File | SM count | Results |
|---|---|---|
| `code/ptv_ownership_gpu6.py` | queried at each launch | Table 2: certified and unresolved voxels and the host-driven column (NVIDIA GeForce RTX 5080 Laptop GPU); the launch-configuration sweep and the kernel-variant screen of SI S1.1; the CPU verification; the free-space bound check |
| `h200/src/ptv_ownership_gpu6.py` | cached once per device | every timing on the NVIDIA H200: Figure 3(d) and the persistent-frontier column of Table 2, the geometry/scale matrix and the five Table 2 prefixes of SI S1.3 |
| `h200/src/ptv_ownership_gpu6_sm_query.py` | queried at each launch | an identical copy of `code/ptv_ownership_gpu6.py`, loaded beside the cached version so that the H200 drivers time both in one process |

Both files keep the name `ptv_ownership_gpu6.py` because the Table 2 timing driver imports that name. The timing
records list the SHA-256 of the module, the timing driver and the site rule;
`reproduce/shared/build_ownership_statistics.py` compares them with the files in this repository.

## code/: Table 2 and SI S1.1 (laptop GPU)

| Script | What it does | Record |
|---|---|---|
| `time_roi_vs_bfs.py` | Table 2 timing driver: ROI-JFA and the host-driven frontier search on one particle prefix of the Berea block; the ROI-JFA owners and distances are checked voxel by voxel against the frontier search | `reproduce/table_02/records/ptv{10,50,100,200,500}_ownership_backend_audit_gpu6.json` |
| `verify_against_cpu.py` | GPU owners and distances against an independent CPU heap Dijkstra (random masks and the five prefixes) | `reproduce/table_02/records/ptv_ownership_gpu6_cpu_reference_verification.json` |
| `sweep_launch_configurations.py` | launch-configuration sweep of ROI-JFA that fixed the Table 2 configuration | `reproduce/table_02/records/ptv_ownership_launch_configuration_sweep.{csv,json}` |
| `screen_warp64_variants.py` | screen of the 64³ ROI-JFA kernel variants | `reproduce/table_02/records/ptv_ownership_warp64_variant_screen.json` |
| `check_free_space_bound.py` | exhaustive test that the free-space proposal of ROI-JFA is exact (SI S1.1-S1.2) | `reproduce/supplementary/s1_ownership/roi_freespace_exactness.json` |

Run from the repository root:

```bash
python gpu/ownership/code/time_roi_vs_bfs.py --mask-npz data/berea64/mask.npz \
  --particle-window data/berea64/particle_tracks.csv.gz --particle-ids 0:49 --out-dir outputs/table_02
                                              # one prefix; the five prefixes are 0:9, 0:49, 0:99, 0:199, 0:499
python gpu/ownership/code/verify_against_cpu.py --random-cases 100 \
  --ptv-record reproduce/table_02/records/ptv10_ownership_backend_audit_gpu6.json \
  --out outputs/table_02/cpu_reference_verification.json      # repeat --ptv-record for ptv50 ... ptv500
python gpu/ownership/code/sweep_launch_configurations.py
python gpu/ownership/code/screen_warp64_variants.py
python gpu/ownership/code/check_free_space_bound.py --out outputs/s1_ownership
```

`sweep_launch_configurations.py` and `screen_warp64_variants.py` take the mask, the particle window and the prefixes
from the five Table 2 records and write their own records into `reproduce/table_02/records/`, replacing the stored
ones (`git diff` shows any change). `check_free_space_bound.py` writes into `reproduce/supplementary/s1_ownership/`
unless `--out` is given. The scripts find `ptv_ownership_gpu6.py` in their own folder and `hybrid_site_sources.py` in
`gpu/code/`. `time_roi_vs_bfs.py`, `verify_against_cpu.py` and `ptv_ownership_gpu6.py` are byte for byte the files
that produced the Table 2 records.

## h200/: the bundle run on one NVIDIA H200

The folder is self-contained: the drivers load only `h200/src/` and read their inputs from `h200/inputs/` and
`data/berea64/`.

| File | What it does | Records |
|---|---|---|
| `time_persistent_baseline.py` | host-driven frontier search, persistent-frontier search and ROI-JFA on 21 cases (16 geometry/scale cases and the five Table 2 prefixes), checked against a CPU oracle | `reproduce/figure_03/persistent_frontier_h200/` (Figure 3(d), Table 2 persistent-frontier column) |
| `run_roi_matrix.py` | ROI-JFA on the 16 geometry/scale cases against the CPU oracle and the full GPU propagation, both module versions timed | `reproduce/supplementary/s1_ownership/roi_geometry_scale_matrix.json`, `roi_geometry_scale_table.csv` |
| `time_paper_prefixes.py` | the five Table 2 prefixes and call configuration on the H200, both module versions timed in one process | `reproduce/supplementary/s1_ownership/roi_paper_prefixes_h200.json` |
| `run_quadrature.py` | local quadrature check of the manufactured reference: orders 3 to 6 against order 7 (the tabulated rows used order 5; SI S2.2; CPU only) | `reproduce/supplementary/s2_discretization/local_quadrature.json`, `local_quadrature_table.csv` |
| `common.py` | JSON output, CPU shortest-path oracle, input check | |
| `src/` | the two module versions, the site rule (`hybrid_site_sources.py`), the study helper (`study_common.py`, identical to `gpu/studies/study_common.py`) and the manufactured-solution fields | |
| `inputs/` | masks and fixed farthest-point site sets (200 and 800 sites) of four of the seven test images: thin wall, maze, Bentheimer crop and Berea block | |
| `jobs/` | Slurm templates of the runs | |

Every driver checks its inputs before it runs: `run_roi_matrix.py` and `run_quadrature.py` against
`inputs_manifest.json`, `time_paper_prefixes.py` against `paper_inputs_manifest.json`, and
`time_persistent_baseline.py` against `input_hashes.json`. These three files list the SHA-256 of the files in this
repository. The records of the runs keep the SHA-256 of the files that ran; for this release some drivers and
`src/study_common.py` changed in names, comments and paths only, and `data/berea64/mask.npz` differs from the timed
mask only in its text member `raw_file` (see `docs/DATA.md`).

Submit from the bundle folder after setting the account, the partition and the Python interpreter (`PYTHON`) in the
templates:

```bash
cd gpu/ownership/h200
sbatch jobs/job_persistent_baseline.sh         # Figure 3(d), Table 2 persistent-frontier column
sbatch jobs/job_roi_matrix_and_prefixes.sh     # SI S1.3
sbatch jobs/job_quadrature.sh                  # SI S2.2
```

The drivers write their results to the folder `out` inside the bundle and Slurm writes the job logs
(`<job name>.<job id>.out`, `.err`) into the bundle folder; neither is part of the repository.
