# EPJC Final Pre-Submission Author Checklist

The real author list (`Xuansong Chang`, `Yiran Xiao`, `Chengtao Song`), dual SCOAP³-compatible `LICENSE`, and live GitHub target (`https://github.com/xschang1002/EPJC`, release tag `v1.0.0-epjc`) have been applied across `main.tex`, `supplementary.tex`, and `epjc-github-release/`.

Only **3 final items** remain before uploading to Springer Editorial Manager (*The European Physical Journal C*):

1. **Institutional Affiliation & Corresponding Author Email** (`main.tex` lines 37–39 & `supplementary.tex` lines 42–44):
   - Replace `[Affiliation to be supplied by authors prior to portal upload]` and `\email{[corresponding.author@institution.edu]}` with your institutional postal address(es) and corresponding email. (ORCIDs can be entered directly in the Editorial Manager submission form or added beneath `\email{...}`.)
   - Confirm the **Funding** sentence (`main.tex` lines 729–731; currently set to no external funding) and uncomment `\begin{acknowledgements}...\end{acknowledgements}` (lines 706–708) if institutional/computing acknowledgements apply.
2. **Push `epjc-github-release/` to Make `https://github.com/xschang1002/EPJC` Live**:
   - Remote `origin` (`https://github.com/xschang1002/EPJC.git`) and annotated tag `v1.0.0-epjc` are already configured locally in `e:\JINST REP\epjc-github-release`. Push them publicly before portal upload so reviewers can access the link directly:
     ```powershell
     cd "e:\JINST REP\epjc-github-release"
     git push -u origin main --tags
     ```
   - *(Optional, strongly recommended for EPJC):* Connect the GitHub repository to Zenodo after pushing tag `v1.0.0-epjc` to mint a permanent archival DOI.
3. **Editorial Manager Upload Structure**:
   - **Article Type / Section:** Regular Article — *Computing, Software and Data Science*.
   - **Cover Letter:** Ready-to-paste draft provided in `COVER_LETTER_EPJC.md`.
   - **Manuscript Source:** Upload `main.tex`, `related_work.tex`, `main.bbl`, `references.bib`, `svjour3.cls`, `svglov3.clo`, `spphys.bst`, and the 4 main PDF figures from `editorial_figures/` (`information_gain_ladder.pdf`, `information_absolute_nll.pdf`, `information_local_C_Hidden5_f100.pdf`, `information_source_location.pdf`).
   - **Electronic Supplementary Material (ESM):** Upload the compiled `physical_centering_epjc_supplement_20261118.pdf` as a separate Supplementary Material item.
