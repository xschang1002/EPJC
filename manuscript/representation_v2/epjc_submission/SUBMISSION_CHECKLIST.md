# EPJC Final Pre-Submission Author Checklist

Author affiliations (`Anhui University, Hefei, China`; `Xi'an Jiaotong University, Xi'an, China`), corresponding author metadata (`Xuansong Chang`, `alec.xs.chang@gmail.com`), funding/competing-interest declarations, official `epjc3` (`svepjc3.clo`) formatting, flat-directory `\graphicspath{{editorial_figures/}{./}}` support, dual SCOAP³-compatible `LICENSE`, and the companion GitHub target (`https://github.com/xschang1002/EPJC`, release tag `v1.0.0-epjc`) are all configured and verified.

## Pre-Upload Steps

1. **Push the Companion Repository Publicly Before Portal Upload**:
   - Push the `main` branch and annotated tag `v1.0.0-epjc` to `https://github.com/xschang1002/EPJC.git` so editors and reviewers can clone and verify immediately:
     ```bash
     git push -u origin main --tags
     ```
   - *(Recommended for EPJC Computing, Software and Data Science):* Enable Zenodo integration on `xschang1002/EPJC` for tag `v1.0.0-epjc` (and optionally attach the ~610 MB binary checkpoint archive to the Zenodo record) to mint a permanent archival DOI.

2. **Springer Editorial Manager Upload Structure**:
   - **Article Type / Section:** Regular Article — *Computing, Software and Data Science*.
   - **Cover Letter:** Paste or upload `COVER_LETTER_EPJC.md`.
   - **Manuscript Source (Flat-Directory Compatible):** Upload `main.tex`, `related_work.tex`, `main.bbl`, `references.bib`, `svjour3.cls`, `svepjc3.clo`, `svglov3.clo`, `spphys.bst`, and the 4 main PDF figures (`information_gain_ladder.pdf`, `information_absolute_nll.pdf`, `information_local_C_Hidden5_f100.pdf`, `information_source_location.pdf`). Because `\graphicspath{{editorial_figures/}{./}}` is set in `main.tex`, Editorial Manager compiles these files directly from a flat upload folder without path edits.
   - **Electronic Supplementary Material (ESM):** Upload the compiled `physical_centering_epjc_supplement_20261118.pdf` as a separate Supplementary Material item.
