# Software environments

## Installing

```bash
pip install -r environment/requirements.txt        # CPU, Python 3.11 or later: NumPy, SciPy, matplotlib, pandas, SymPy
pip install -r environment/requirements-gpu.txt    # adds CuPy for CUDA 12 (GPU code only)
# or, with conda:
conda env create -f environment/environment.yml
```

`requirements.txt` pins the laptop versions (Python 3.12.10) and needs Python 3.11 or later: NumPy 2.4.6, SciPy
1.17.1 and pandas 3.0.5 do not install on Python 3.10. On Python 3.10 install the versions of the cluster runs
(environment 3) instead, `python -m pip install numpy==2.2.6 scipy==1.15.3`; `porevoronoi_fv/` needs nothing else.
These are also the versions of the README quick start.

The CPU package `porevoronoi_fv/` needs only NumPy and SciPy. CuPy is needed by the GPU ownership code in
`gpu/ownership/` and by the GPU forward path. The pinned versions are those of the laptop environment below
(environment 1). Run Python with `-B` so that no byte code is written next to the released files.

## Recorded environments

The results of the paper come from five environments. The versions are those recorded by the runs themselves
(environment fields of the result records and run logs) and in extended-data entry ED7.2 of the evidence
archive.

| # | Machine | Python | NumPy | SciPy | GPU stack | Used for |
|---|---|---|---|---|---|---|
| 1 | laptop: Windows 11, Intel64 Family 6 Model 198 (24 logical cores), NVIDIA GeForce RTX 5080 Laptop GPU (16,303 MiB, driver 595.79) | 3.12.10 | 2.4.6 | 1.17.1 | CuPy 14.1.1, CUDA runtime 12.9 | forward, refinement, pressure, stability, manufactured-solution and cost runs; the recomputations and cross-script checks of the Supporting Information |
| 2 | the same laptop | 3.13.5 (Anaconda) | 2.1.3 | | CuPy 14.1.1 | ownership timing and exactness runs: Table 2, ED1.1, the launch-configuration sweep, the CPU reference verification |
| 3 | Linux cluster, CPU nodes | 3.10.20 | 2.2.6 | 1.15.3 | none | Stokes-only solves and checks of the six controlled cases; assisted calculations (Table 6); single-frame timing |
| 4 | Linux cluster, NVIDIA H200 (132 streaming multiprocessors) | 3.10.20 | 2.2.6 | | CuPy 14.1.1; driver CUDA 12.8; CuPy runtime 12.9; system CUDA 12.8 runtime compiler | GPU ownership timings on the H200 (Figure 3d, Table 2 persistent column, extended data ED1.2-ED1.4) and local quadrature (ED2.4) |
| 5 | cluster of the Imperial College Research Computing Service | 3.12.11 | 2.5.3 | 1.18.1 | none | two-phase calculations on the drainage experiment; figures rendered there |

Sources: the environment fields of the result records and extended-data entry ED7.2 of the evidence archive.

### Further packages per environment

* Environment 1 (installed packages): matplotlib 3.10.9, pandas 3.0.5, SymPy 1.14.0, PyMuPDF 1.28.2, PyVista 0.49.0,
  VTK 9.7.0, Pillow 12.2.0, fontTools 4.63.0, NetworkX 3.6.1, Numba 0.66.0, h5py 3.16.0, scikit-image 0.26.0,
  tifffile 2026.7.14, imagecodecs 2026.6.26. External programs: Inkscape 1.4.4 and MiKTeX 25.12.
* Environment 5 (virtual environment made with uv, read from its package metadata): matplotlib 3.11.2, pandas 3.0.5,
  Pillow 12.3.0, pyarrow 25.0.1, tifffile 2026.9.15, imagecodecs 2026.8.16, contourpy 1.4.0, fontTools 4.65.0,
  kiwisolver 1.5.1, cycler 0.12.1, packaging 26.3, pyparsing 3.3.2, python-dateutil 2.9.0.post0, six 1.17.0. It has
  no CuPy and no SymPy.
* Environments 2, 3 and 4: only the versions in the table are recorded; there is no full package list.

## Differences between environments

The calculations use double precision throughout. Geometry and topology counts are reproducible, and reported errors
reproduce to about 1e-14 relative. MINRES iteration counts differ between builds of BLAS and SciPy by up to 69
iterations, about 1% of a typical 6500, because rounding near the relative tolerance 1e-14 changes the stopping
iterate; every solve compared across builds attains its residual limit (ED7.2 of the evidence archive).
The run records `data/controlled_cases/<case>/run_record.json` (the forward runs) and our later CPU solves
(environment 3) differ by 7-174 iterations (0.1-2.3%; c5: 7487 against 7661); all of these solves attain their
residual limit.
