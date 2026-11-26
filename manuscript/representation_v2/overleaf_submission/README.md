# Preserved Single-Column Reference & Presentation Generators

**Non-monotonic benefits of physical target centering in conditional models of pion momentum loss**

> **Note:** The active two-column *European Physical Journal C (EPJC)* submission package resides in [`../epjc_submission/`](../epjc_submission/), with compiled PDFs under [`../../output/pdf/`](../../output/pdf/). This directory retains the earlier single-column internal layout and the read-only presentation modules (`generate_information_figures.py`, `build_information_tables.py`, `editorial_figures/generate_editorial_figures.py`, and reference `information_tables/` / `seed_sensitivity/` inputs) invoked by `python reproduce.py` during aggregate verification.

## Preserved Presentation Assets

- `editorial_figures/` — All 21 relabeled PDF/PNG figure assets and `generate_editorial_figures.py`.
- `editorial_tables/` — Reference LaTeX table outputs matched cell-for-cell against `../epjc_submission/editorial_tables/`.
- `information_tables/` & `seed_sensitivity/` — Source-table inputs read by `scripts/rebuild_editorial_tables.py`.
- `../historical_archive_20261108/` — Archival record of preliminary development controls and display-ID mappings.

Display names (`BB-centered`, `No direct momentum`, `2-bin momentum`, `8-bin momentum`, `Exact momentum`, `Linear center`, `Polynomial-ridge center`) map to immutable frozen code and run identifiers without altering any empirical estimate or model state. Figure provenance in `FIGURE_MANIFEST.json` retains the Scientific Agent Skills procedural-tool acknowledgment (Kassis et al., 2026, DOI `10.48550/arXiv.2609.00065`) outside scientific Related Work.
