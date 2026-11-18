# Current manuscript

**Non-monotonic benefits of physical target centering in conditional models of pion momentum loss**

The argument separates three questions: the gain from physical centering relative to Raw, the absolute performance of the complete predictor, and performance across physical regions. The two-bin condition has the largest relative gain; exact momentum gives the best absolute mean NLL; aggregate gains coexist with local losses.

The canonical sources are [main.tex](overleaf_submission/main.tex) and [supplementary.tex](overleaf_submission/supplementary.tex). This editorial revision introduces scientific display names without changing frozen identifiers or scientific artifacts. [The display map](historical_archive_20261108/DISPLAY_NAME_MAP.md) relates BB-centered, Linear center, Polynomial-ridge center and the four neural information levels to their run IDs.

The supplement is organized as S1 complete primary C/D results, S2 training variability and conditional diagnostics, S3 current fitted-center/capacity comparisons, S4 numerical implementation and reproduction, and S5 a short development note. Full earlier A/B/H evidence is preserved in the [standalone historical archive](historical_archive_20261108/README.md). None of the unrecovered historical claims supplies evidence for the current conclusions.

Original claim and evidence registries remain byte-preserved. Their canonical terminology is historical record, not the display-name source for the current article. [OUTLINE.md](OUTLINE.md) and [FIGURE_TABLE_PLAN.md](FIGURE_TABLE_PLAN.md) describe the current presentation; the editorial report records source preservation, row-level numeric checks and rendering checks.

The new delivery is `physical_centering_reviewed_20261109.pdf` and `physical_centering_reviewed_supplement_20261109.pdf` under `output/pdf/`, with `physical_centering_reproducibility_20261109.zip` as the source package. Prior delivered PDFs and ZIPs remain unchanged. The local mirror is synchronized without remote publication.

Author identities, declarations, human source review, license and public data-access arrangements remain author tasks; `submission_ready` remains false. This edit adds no training, simulation or neural inference. The mirror retains the 503 pre-existing validation prediction files and two aggregate bootstrap arrays; it receives no new information80/C/D per-event arrays.

## Final prose polish (9 November 2026)

The current pair has 15 main pages and 13 supplement pages. The Figure 1/2 connection is an arithmetic statement about five-seed mean NLL changes; detailed density accounting is in S2.2. Figure 3 precedes subsection 5.4, and the calibration paragraph is complete before Figure 4. See `reports/final_polish_20261109` for scoped checks. Earlier editorial receipts remain historical records.

## Reviewer completion (9 November 2026)

The current 15-page main text and 13-page supplement clarify the original-primary first-visit target, simulation geometry/beam/transport settings, and float64 preprocessing fit versus float32 applied standardization. Scientific estimates, uncertainty definitions, plotted values and conclusions are unchanged. The companion aggregate reproduction archive includes the current LaTeX/PDFs, complete C/D predictive and residual diagnostic summaries, saved bootstrap draws, frozen configuration/source dependencies, registered checkpoint identities and a portable reproduction entry point. See its `REPRODUCE.md` and `DIAGNOSTIC_INDEX.json`. This is complete aggregate reconstruction, not event-level replay or retraining. Author details and declarations remain pending. The review record is `reports/review_completion_20261109`. No new experiment or remote publication accompanies this revision.
