# Contributing and Verification Guidelines

This repository archives the frozen experimental protocols, source code, aggregate empirical summaries, and LaTeX manuscript sources for a completed scientific study submitted to *The European Physical Journal C (EPJC)*.

1. **Preserving Frozen Provenance:** Files bound by `PACKAGE_MANIFEST.json`, `protocol/fresh_catalog_20261108.json`, and `reports/**/ANALYSIS_MANIFEST.json` are cryptographically hashed. Do not reformat or alter line endings in `src/`, `protocol/`, or `reports/`.
2. **Isolated Output Directories:** Always run `python reproduce.py --output <new_dir>` into a fresh output directory outside tracked folders.
3. **Bug Reports & Questions:** Issues concerning reproduction instructions, cross-platform compatibility, or manuscript clarifications are welcome via GitHub Issues.
