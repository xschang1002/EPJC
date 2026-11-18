# EPJC Final Pre-Submission Author Checklist

Before uploading `epjc_submission/` to Springer Editorial Manager (*The European Physical Journal C*), replace the standardized placeholders below:

1. **Author List, Affiliations, Emails, and ORCIDs** (`main.tex` lines 32--45 & `supplementary.tex` lines 35--47):
   - Replace `First Author`, `Second Author`, `Third Author`, `\authorrunning{...}`, institutional postal addresses, `\email{...}`, and `0000-0000-0000-0000` ORCIDs.
2. **GitHub Repository URL & Release Tag** (`main.tex` *Data Availability Statement* & *Code Availability Statement*; `supplementary.tex` Sect. S4.3):
   - Push the standalone clean repository `epjc-github-release/` (already pre-configured with a SCOAP³-compatible dual `LICENSE`: `MIT` for software/code and `CC-BY-4.0` for aggregate data, tables, and figures) to GitHub and create release tag `v1.0.0-epjc`.
   - Replace `https://github.com/<owner>/<repository>` in `main.tex` and `supplementary.tex` with your live public GitHub repository URL.
3. **Acknowledgements & Declarations** (`main.tex` lines 699--744):
   - **Acknowledgements:** Insert computing/institutional acknowledgements, or remove the `\begin{acknowledgements}...\end{acknowledgements}` block if none.
   - **Funding:** Insert funding agency and grant number(s), or retain only the "no funds, grants, or other support were received" clause.
   - **Competing interests:** Confirm `"The authors have no relevant financial or non-financial interests to disclose."`
   - **Author contributions:** Replace `F. Author`, `S. Author`, `T. Author` with actual author initials.
   - **Use of AI-assisted tools:** Review and confirm the Springer Nature AI-tool disclosure paragraph.
4. **Editorial Manager Upload Structure**:
   - **Article Type / Section:** Regular Article — *Computing, Software and Data Science*.
   - **Manuscript Source:** Upload `main.tex`, `related_work.tex`, `main.bbl`, `references.bib`, `svjour3.cls`, `svglov3.clo`, `spphys.bst`, and the 4 main PDF figures from `editorial_figures/` (`information_gain_ladder.pdf`, `information_absolute_nll.pdf`, `information_local_C_Hidden5_f100.pdf`, `information_source_location.pdf`).
   - **Electronic Supplementary Material (ESM):** Upload the compiled `physical_centering_epjc_supplement_20261118.pdf` as a separate Supplementary Material item.
