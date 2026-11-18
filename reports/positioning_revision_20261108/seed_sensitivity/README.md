# Post-hoc paired optimizer-seed sensitivity

Run `analyze_seed_sensitivity.py` on the completed C/D analysis directories and their frozen statistical plan plus independent numerical-review receipt. The script verifies the saved table hashes and complete 16-contrast × 5-seed structure before computing anything. It imports no frozen experiment module and does not train or score models.

```powershell
& 'E:\JINST REP\pion-material-study\.venv-ml\Scripts\python.exe' -B 'E:\JINST REP\_positioning_revision_20261108\seed_sensitivity\analyze_seed_sensitivity.py' `
  --analysis-c 'E:\JINST REP\low pion research\reports\information_extension_20261108\fresh_analysis\C' `
  --analysis-d 'E:\JINST REP\low pion research\reports\information_extension_20261108\fresh_analysis\D' `
  --plan 'E:\JINST REP\low pion research\protocol\information_statistical_analysis_20261108.json' `
  --numerical-review 'E:\JINST REP\_research_extension_20261108\independent_information_numerical_review.json' `
  --output 'E:\JINST REP\_positioning_revision_20261108\seed_sensitivity\results_reproduced' `
  --include-exploratory-t
```

Use a new output directory: the script refuses to overwrite existing sensitivity results. Omitting `--include-exploratory-t` skips the optional exploratory interval tables; it leaves the main seed/omission analysis unchanged. NumPy is required, and SciPy is required only for the optional t critical value.

The recorded run is under `results/`, with status `COMPLETE_POSTHOC_SEED_SENSITIVITY`. `seed_sensitivity.tex` is the default complete table; `exploratory_t_intervals.tex` is explicitly separate and not recommended as the default inference display. The two captions state n=5, common data/model dependence, separate C/D handling and uncertainty boundaries. `SCIENTIFIC_FINDINGS.md` supplies the actual interpretation; `results/SCIENTIFIC_INTERPRETATION.md` records the generated method guide.

Four synthetic arithmetic tests passed in 0.12 s. They check retention of discordant seeds, omission-induced sign reversal, the t standard error `sample_sd/sqrt(5)` and rejection of wrong-sized or nonfinite inputs. These checks are specific to this new post-hoc arithmetic; they are not tests of a new training procedure.

All earlier experiment, analysis and source manifests remain intact. The new `results/MANIFEST.json` binds inputs, script and generated outputs; the separate delivery receipt binds this explanatory guide and any standalone TeX preview without rewriting the numerical manifest. Human interpretation and manuscript inclusion remain the authors' responsibility.
