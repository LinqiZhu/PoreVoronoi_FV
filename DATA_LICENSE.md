# Data licence

The source code in this repository (Python modules and scripts, shell scripts, the notebook) is licensed under the
Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). This file sets the terms for the data files.
What each folder contains is described in [`docs/DATA.md`](docs/DATA.md).

## 1. Our own data and results: CC BY 4.0

Unless Section 2 or 3 below says otherwise, the data files of this repository were produced by the authors and are
licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
<https://creativecommons.org/licenses/by/4.0/>). Please attribute them by citing the manuscript (2026) *PoreVoronoi-FV:
Conservative pore-scale flow on cells fixed by tracked particles* by Linqi Zhu, Chunyang Wang, Yuxuan Gu, Robert van der Merwe, Tom Bultreys, Martin J. Blunt and Gege Wen, or this repository (see [`CITATION.cff`](CITATION.cff)).

This covers in particular:

* the synthetic controlled cases c1 to c5: masks and the reference flows we computed on them (`reference_flow.npz`
  of the five duct folders in `data/controlled_cases/`);
* the reference flows we computed on the Bentheimer crop and on the 64³ Berea block (the velocity, pressure, flux
  and permeability arrays of `data/controlled_cases/bentheimer_crop/reference_flow.npz` and of
  `data/berea64/reference_flow_x.npz`);
* the simulated particle windows, in which particles are carried through those reference flows
  (`data/controlled_cases/<case>/particle_tracks.csv.gz`, `data/berea64/particle_tracks.csv.gz`) and the generator
  records (the `particle_tracks_generator.json` files);
* the stored solutions, run manifests and printed-value records in `data/controlled_cases/`;
* the synthetic fibrous-filter proxy (`examples/segmented_masks/fibrous_filter_proxy_16x64x64.npz` and its
  manifest);
* the result files under `reproduce/` and the site sets in `gpu/ownership/h200/inputs/`;
* `docs/figures_and_tables.csv`, the documentation under `docs/` and `assets/porevoronoi_logo.png`.

`assets/figure1_construction.png` and `assets/pipeline.png` are Figure 1 of the manuscript (2026) and its top row;
they are shown to explain the method and are not licensed under CC BY 4.0.

## 2. The 64³ Berea block: CC BY 4.0, with attribution to the source record

The 64³ Berea mask is derived from the data set "Berea Sandstone" of the Imperial College Consortium on Pore-scale
Imaging and Modelling (figshare, 2014, version 2, doi:[10.6084/m9.figshare.1153794.v2](https://doi.org/10.6084/m9.figshare.1153794.v2)),
which is licensed under CC BY 4.0. Changes made: a 64³ block was cut from the file `Berea.raw` (voxels 64–127,
224–287 and 288–351 along its first, second and third axes), its axes were permuted, and only the face-connected
pore component that spans all three axes was kept (details in [`docs/DATA.md`](docs/DATA.md), Section 2). The derived
block is released under CC BY 4.0; any use must attribute the source record as well as this repository.

Files: `data/berea64/mask.npz`, the mask and label arrays of `data/berea64/reference_flow_x.npz`, and the mask
copy `gpu/ownership/h200/inputs/berea_heldout_64.npz`.

## 3. The Bentheimer 16 × 96 × 96 crop: not licensed

The crop in `examples/segmented_masks/bentheimer_dry_crop_16x96x96_origin_1562_59_349_pore0.npz` (with its manifest)
was cut at origin (1562, 59, 349) from a segmented dry scan of Bentheimer sandstone of 2714 × 1000 × 1000 voxels,
which is not public. The crop is included only so that the sixth controlled case of the manuscript can be repeated.
It is not licensed under CC BY 4.0 or any other licence, and no rights to the parent scan are granted. The same crop
appears as the mask and label arrays of `data/controlled_cases/bentheimer_crop/reference_flow.npz` and as the
mask in `gpu/ownership/h200/inputs/bentheimer_crop.npz`.

## 4. Third-party and experimental data: not included

The following data sets are used in the manuscript but are neither redistributed nor re-licensed here. They remain
under the terms of their providers.

* Drainage experiment 073: X-ray tomograms of Bultreys et al. (2024), PNAS 121, e2316723121, in the PSI Public Data
  Repository, doi:[10.16907/c0dfa6c8-25da-454e-82fa-fc5db7f7c6f2](https://doi.org/10.16907/c0dfa6c8-25da-454e-82fa-fc5db7f7c6f2);
  segmentations and Kalman-smoothed tracks of Wang et al. (2026), arXiv:2603.12516, released only by that study.
* Glass-filter X-ray particle-tracking data of Bultreys et al. (2022), Zenodo
  doi:[10.5281/zenodo.6010490](https://doi.org/10.5281/zenodo.6010490), and the results derived from its tracks.
* Sand-pack X-ray particle-tracking data of Bultreys et al. (2022), Zenodo
  doi:[10.5281/zenodo.6010425](https://doi.org/10.5281/zenodo.6010425).
* The full Bentheimer image of Jackson et al. (2021), Zenodo doi:[10.5281/zenodo.5542624](https://doi.org/10.5281/zenodo.5542624).
* The full Figshare "Berea Sandstone" record, of which only the derived block of Section 2 is included.
