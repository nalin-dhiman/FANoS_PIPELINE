REPRODUCE (Results Pipeline Only)
=================================

This package builds plots/tables from raw CSV logs.

Outputs
-------
- artifacts/standardized/*.csv
- artifacts/stats/*.csv
- artifacts/plots/*.pdf + *.png
- artifacts/tables/*.tex (+ some CSV mirrors)
- latex_snippets/INCLUDE_FIGURES.tex
- latex_snippets/INCLUDE_TABLES.tex

Commands
--------
From the repository root:

  make -C reproduce all
  make -C reproduce validate
  make -C reproduce package

The build script will auto-generate:
- artifacts/raw/raw_quadratic_sweep_runs.csv
- artifacts/raw/raw_thermostat_diagnostics.csv

if they are missing.

