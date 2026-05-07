# Repository Scope

This repository is the FANoS-v2 experiment and release pipeline.

It includes:

- optimizer source code for reproducibility;
- benchmark scripts and full-study runners;
- generated paper package under `paper/fanos_v2_v02`;
- markdown reports under `reports`;
- local CSV/log/profile evidence under `results`.

It intentionally excludes:

- raw downloaded datasets;
- Python virtual environments;
- build caches;
- package build outputs.

Critical status: this repository is for audit and reproduction. The evidence
supports a v0.2 alpha research release, not a claim of production speed parity
or universal optimizer superiority.

