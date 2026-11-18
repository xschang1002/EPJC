# Saved fresh-catalog support-term audit

This is a post-hoc, read-only audit of the original C/D scoring artifacts. It does not train, select or run a neural density model. Checkpoints were deserialized only to verify identity and the recorded spline bound; no flow was constructed or evaluated. All outputs here are aggregates, run identifiers and file hashes, not individual event IDs or event arrays.

## Coverage and recorded values

All 240 registered runs were retained on each catalog, with 80 new primary runs and 160 earlier runs. C has 499285 eligible rows and D has 499275, out of 500000 launches each. The audit verified 5280 model-component NPY files and 12 catalog-array files against their original hashes, shapes and dtypes, plus the original run/configuration identities. The 20 corresponding new/earlier Full pairs have identical bytes in all eleven saved components on both catalogs. These are duplicate predictors, not added independent evidence.

Every saved support log-CDF is float32 **negative zero**: 119828400 model-event entries on C and 119826000 on D, totalling 239654400. Minima, maxima and all reported quantiles are zero; positive-zero, nonzero and nonfinite counts are zero. Every saved component, including log density, log CDF and log survival, is finite. These repeated model-event totals are not counts of independent events or predictors.

| Catalog | Cohort | Registered runs | Minimum b | Maximum b | Minimum saved b−z |
| --- | --- | ---: | ---: | ---: | ---: |
| C | New primary | 80 | 761.3554077148438 | 2776.529052734375 | 751.7498779296875 |
| C | Earlier | 160 | 761.3501586914062 | 2776.529052734375 | 751.7498073577881 |
| D | New primary | 80 | 761.3547973632812 | 2776.541259765625 | 757.0908308029175 |
| D | Earlier | 160 | 761.3495483398438 | 2776.5439453125 | 757.0907778739929 |

Here `b−z` means the difference of the two saved float32 values evaluated in float64. No saved `z` equals or exceeds `b`; the smallest standardized margin exceeds 8.24 million downward float32 ULPs of `b`. Full per-model distributions and all 480 identities remain in `results/support_audit.json` and `model_support_summary.csv`.

## Why zero is numerically expected

The frozen engine defines two rational-quadratic spline layers with an identity branch outside [−8,8]. All recorded checkpoint configurations use bound 8, and every saved `b` is above 761. Therefore each layer returns `b` unchanged, so `base(b)=b`; this follows from the frozen code and saved coordinates without a network evaluation. The support log-CDF is `−softplus(−b)`, with no manual epsilon floor or support-term clipping.

In real arithmetic let `t=exp(−b)>0`. The excluded logistic mass is `m=t/(1+t)` and the magnitude of the support log-CDF is `a=log(1+t)`. For either quantity,

`−b−exp(−b) ≤ log(m), log(a) ≤ −b`.

These are symbolic real-number bounds. The audit does not form `t` in float64 or claim that the lower endpoint can be distinguished from `−b` in machine arithmetic. Across both catalogs the smallest saved `b` is 761.3495483398438. Thus both excluded mass and `|log F(b)|` are below `10^(−330)`; the tight log10 upper anchor is approximately −330.6499076435272. At the largest `b`, the log10 anchor is approximately −1205.837714211103. The error of the anchor `−b/ln(10)` is bounded by `exp(−b)/ln(10)`.

Even the maximum mass/magnitude is below half the smallest float64 subnormal: the corresponding threshold is `1075 ln(2)=745.1332191019412`, below every saved `b`. A correctly rounded finite float64 value for that mass or log-CDF magnitude would therefore also be zero. The recorded float32 negative zero is compatible with ordinary underflow; it does not mean the mathematical excluded mass is exactly zero. A log-domain bound is the appropriate quantitative description.

The certificate concerns the logistic function **at the saved rounded b**, not an unrecorded infinite-precision model or a new float64 forward pass. It explains why the saved support normalization cannot numerically account for the reported C/D gains. It does not establish that support normalization is generally unnecessary, or audit its role on training data or other distributions.

## Input margins, ordering and no numerical drops

Two distinct input-margin calculations are explicitly recorded. The exact difference of the float32 inputs evaluated in float64 has minima 0.2957799299620092 (C) and 0.29788136715069413 (D) GeV/c. Actual float32 subtraction gives 0.2957799434661865 and 0.2978813648223877 GeV/c. Their small difference is rounding of subtraction, not conflicting data. The support predicate compares `y32<p32` directly; it is true for every eligible event.

The independent original-ledger audit reconstructs eligibility and matches the six ordered catalog arrays and UID digest. Its actual float32 margin distribution matches the additional subtraction check here. A second independent arithmetic replay reconstructs the 28 preprocessor/catalog combinations (14 per catalog), then compares every run's saved `z` and `b`: 960 arrays, 479308800 values, zero bitwise mismatches and zero ULP difference. It uses the frozen pure preprocessing implementation and no neural evaluation, so it verifies correspondence rather than independently deriving the physical-center formula.

Together the ledger, standardization and component checks retain every eligible row in every model output, with no observed numerical support mask or model-specific drop. The 715 C and 725 D ineligible launches were excluded by the predeclared scientific eligibility rule before scoring, not by numerical filtering. The scorer fails on a nonfinite density component instead of discarding such rows. These conclusions apply to these saved catalogs and models, not all possible boundary inputs.

## Reproduction and provenance

Run `audit_saved_support.py` with `--repo`, `--research` and a new `--output` directory. It needs the private saved scoring arrays and original checkpoints; the aggregate-only public bundle cannot perform that scan. The original executed script hash and file verification results are retained. `clarify_margins_and_reconcile.py` adds the separately defined float32 input subtraction and binds the two companion audits; it does not rehash or recompute all 5280 model arrays.

`results/MARGIN_SCOPE_CLARIFICATION.json` records the change of margin labels and extra subtraction check. The original result/manifest are preserved in the local audit workspace at `before_margin_scope_clarification`; that before snapshot is not included in the public aggregate export. `results/INDEPENDENT_RECONCILIATION.json` binds the ledger and standardization receipts. `results/MANIFEST.json` binds the current aggregate outputs and distinguishes initial execution from postprocessing. No prior scientific source, protocol, data, model or acceptance receipt was edited.
