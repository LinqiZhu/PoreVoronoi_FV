# Persistent-frontier timing on one H200 GPU

Timing records behind Figure 3(d) and the persistent-frontier column of Table 2. Three exact ownership
methods were timed on one NVIDIA H200 GPU: host-driven breadth-first search (BFS), persistent-frontier BFS
and ROI-JFA. All three are the methods of `gpu/ownership/h200/src/ptv_ownership_gpu6.py`, unchanged.

## Cases

- 16 rows: thin wall, maze, Bentheimer crop and Berea block, each with 200 and 800 farthest-point sites,
  on the image as given (`_s1`) and at twice the resolution (`_s2`).
- 5 rows (`ptv10` to `ptv500`): the sites of the first 10, 50, 100, 200 and 500 particles of the Berea
  window `data/berea64/particle_tracks.csv.gz`. These rows are the persistent-frontier column of Table 2.

## What is timed

One call with its inputs already on the GPU, including allocation, initialization, all stages of the method
and the output, synchronized immediately before and after the call. Input transfer, CPU checks, cell
construction and the Stokes solve are not included. All timings are taken after warm-up: 5 warm-up calls and
51 repeats per method for the 16 rows, 10 and 101 for the five particle prefixes; the order of the three
methods is rotated between repeats. For the five prefixes, the line-bit cache of the 64-cube mask used by
ROI-JFA is built once before the timed calls and is not included (its preparation times are in `run.json`).

The 95% intervals are paired bootstrap intervals of the median within this run; they do not describe the
spread between runs or devices. In every case the distances and owners of all three methods equal those of
an independent CPU reference.

## Files

| File | Content |
|---|---|
| `summary.csv` | one row per case: `case_id`, `sites`, median times `host_ms`, `persistent_ms`, `roi_ms` (ms), `persistent_over_roi` with its 95% interval `ci_low`, `ci_high`, and `host_over_persistent` |
| `raw_repeats.csv` | every timed call: case, repeat, position in the method order, method, seconds |
| `results.json` | full record per case: timings, intervals, method settings, output hashes and mismatch counts |
| `run.json` | device, CUDA and software versions, method parameters and the sha256 of the source and input files of the run |

The input digests that the driver checks before it runs are in `gpu/ownership/h200/input_hashes.json`.

## How to run

Set the Slurm account and partition in `gpu/ownership/h200/jobs/job_persistent_baseline.sh`, then run
`sbatch jobs/job_persistent_baseline.sh` from `gpu/ownership/h200/` (the job uses the Python in `$PYTHON`,
default `python`, with CuPy for CUDA 12). The driver `time_persistent_baseline.py` checks its inputs against
`input_hashes.json` and writes the same files to `gpu/ownership/h200/out/` (`metadata.json` there is
`run.json` here). The recorded run took 42 s.
