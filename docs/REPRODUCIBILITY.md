# Reproducibility workflow

This note gives a practical route through the PoreVoronoi package.

## 1. Inspect the evidence tables

Start with:

- `data/final_source_data/`
- `data/paper_ready_tables/`
- `MANIFEST.csv`

These files connect directly to the manuscript figures and tables.

## 2. Parse-check the Python sources

```bash
python -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in pathlib.Path('.').rglob('*.py')]; print('AST_OK')"
```

## 3. Regenerate selected figures

```bash
python repro/figures/make_figure_04_ownership_speed_audit.py
python repro/figures/make_figure_05_operator_geometry_unified.py
python repro/figures/Figure_07_state_projection.py
python repro/figures/make_figure_13_mainline_transfer.py
```

Set `INKSCAPE_BIN` if the figure exporter cannot find Inkscape.

## 4. Rerun compact examples

The compact synthetic references in `examples/synthetic_reference_data/` are intended for local audit and development checks. Full production reruns may require CuPy, an NVIDIA GPU, and external large-volume inputs.

## 5. Boundary of this release

This package supports traceability and reproduction of the manuscript's compact evidence surfaces. It is not a full dump of every development run or private workstation output tree.
