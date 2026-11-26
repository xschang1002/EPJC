# Manuscript Package Overview

**Title:** *Non-monotonic benefits of physical target centering in conditional models of pion momentum loss*  
**Authors:** Xuansong Chang, Yiran Xiao, and Chengtao Song  
**Target Journal:** *The European Physical Journal C (EPJC)* — *Computing, Software and Data Science*

## Directory Structure

- **[`epjc_submission/`](epjc_submission/)** — **Canonical EPJC submission source package** (`\documentclass[twocolumn,epjc3]{svjour3}`, `spphys` bibliography style, pre-built `main.bbl`, standalone Electronic Supplementary Material `supplementary.tex`, 21 figure assets, and verified table inputs).
  - Compiled main manuscript: [`../../output/pdf/physical_centering_epjc_20261118.pdf`](../../output/pdf/physical_centering_epjc_20261118.pdf)
  - Compiled Electronic Supplementary Material: [`../../output/pdf/physical_centering_epjc_supplement_20261118.pdf`](../../output/pdf/physical_centering_epjc_supplement_20261118.pdf)
- **[`overleaf_submission/`](overleaf_submission/)** — Preserved single-column layout and aggregate figure/table presentation scripts (`generate_information_figures.py`, `build_information_tables.py`, `editorial_figures/generate_editorial_figures.py`) used by `python reproduce.py` to reconstruct all 21 figures and 14 LaTeX tables from frozen aggregate records.
- **[`historical_archive_20261108/`](historical_archive_20261108/README.md)** — Compact archival record of preliminary development controls and display-name provenance ([`DISPLAY_NAME_MAP.md`](historical_archive_20261108/DISPLAY_NAME_MAP.md)), summarized in Supplementary Section S5.

## Scientific & Provenance Invariance

All empirical estimates, 80 primary fits, 160 secondary/control fits, fresh C/D catalog evaluations, 2,000-draw stratified paired bootstrap intervals, post-hoc five-seed sensitivity summaries, and support-truncation bounds are byte-preserved from the frozen experimental records. Display terminology maps deterministically to immutable run identifiers without altering any saved model, protocol, or statistical estimate.
