# EPJC Submission Source Package (`svjour3` + `epjc3` Two-Column)

**Title:** *Non-monotonic benefits of physical target centering in conditional models of pion momentum loss*  
**Authors:** Xuansong Chang (Anhui University), Yiran Xiao (Anhui University), and Chengtao Song (Xi'an Jiaotong University)  
**Corresponding Author:** Xuansong Chang (`alec.xs.chang@gmail.com`)  
**Target Journal:** *The European Physical Journal C (EPJC)* — Section: *Computing, Software and Data Science*

## Package Contents

- `main.tex` — Two-column EPJC main manuscript (`\documentclass[twocolumn,epjc3]{svjour3}`, `natbib` compressed numbered citations, `spphys` bibliography style, flat-compatible `\graphicspath{{editorial_figures/}{./}}`).
- `related_work.tex` — Related work section covering hybrid/residual modeling, transformation/flow theory, and HEP fast-simulation & particle-transport surrogates.
- `references.bib` & `main.bbl` — Curated 26-entry bibliography formatted with ISO-4 physics journal abbreviations and verified DOIs (`spphys.bst`).
- `supplementary.tex` — Companion Electronic Supplementary Material (`\documentclass[smallextended,epjc3]{svjour3}`), with `S`-prefixed numbering (`Table S1`–`Table S13`, `Fig. S1`–`Fig. S2`, `Eq. (S1)`–`Eq. (S8)`).
- `editorial_figures/` — All 21 vector PDF and PNG figure assets (unchanged from frozen empirical records).
- `editorial_tables/`, `editorial_theory.tex`, `support_audit_tables.tex` — Verified table and theory inputs (numerical cells byte-preserved).
- `svjour3.cls`, `svepjc3.clo`, `svglov3.clo`, `spphys.bst` — Bundled Springer SVJour3 class, official EPJC layout option (`epjc3`), and physics bibliography style files for standalone compilation on Overleaf, TeX Live, Tectonic, or Springer Editorial Manager.
- `SUBMISSION_CHECKLIST.md` — Final pre-upload verification checklist for Springer Editorial Manager.
- `COVER_LETTER_EPJC.md` — Ready-to-use Cover Letter tailored for the EPJC *Computing, Software and Data Science* editorial board.

## Compilation

Compile `main.tex` and `supplementary.tex` separately with `pdflatex` + `bibtex` (or `tectonic`):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

pdflatex supplementary.tex
pdflatex supplementary.tex
```

For journal portals that compile uploaded files in a single flat root directory without running BibTeX, `\graphicspath{{editorial_figures/}{./}}` and pre-generated `main.bbl` are both included so server-side PDF compilation succeeds out of the box.

## Scope & Provenance Note

All empirical estimates, 80 primary fits, 160 secondary/control fits, fresh C/D catalog evaluations, 2,000-draw stratified bootstrap intervals, post-hoc seed-sensitivity summaries, and support-truncation bounds are preserved without modification from the frozen research archive.
