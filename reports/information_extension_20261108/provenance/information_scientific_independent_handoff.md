# Independent information-study handoff

Numerical review PASS; no unresolved numerical finding. Source80 full validation replay reported exact equality and independent saved NLL recomputation error0. Both catalogs contain500000 launched events (C499285 eligible; D499275 eligible). All480 registered-run scores,16 primary contrasts,336 groups,41 legacy contrasts per catalog, all2000 launched/S0 bootstrap draws and16/42/336 interval families were checked.

All20 new Full5 runs exactly match their earlier Full counterparts in every saved weight tensor and every saved source/C/D log score. Their checkpoint container hashes differ20/20 because serialization/metadata differ; that is not evidence of different learned functions. Preserve all registered evaluations and describe240 runs,80 new fits+160 earlier runs, with20 exact rerun pairs. Do not claim the other220 are all distinct without a separate check.

## Eight gains, nat per eligible event

| Context/fraction | C mean | C simultaneous95% interval | D mean | D simultaneous95% interval | C/D positive optimizer seeds |
|---|---:|---|---:|---|---|
| gain:Hidden5/f010 | 0.006271646 | [0.005708700, 0.006834592] | 0.006400716 | [0.005830005, 0.006971427] | 5/5; 5/5 |
| gain:Coarse2/f010 | 0.010399011 | [0.009843831, 0.010954192] | 0.010622656 | [0.010059133, 0.011186180] | 5/5; 5/5 |
| gain:Coarse8/f010 | 0.003081988 | [0.002741471, 0.003422504] | 0.003320362 | [0.002978708, 0.003662015] | 5/5; 5/5 |
| gain:Full5/f010 | 0.001844930 | [0.001580059, 0.002109801] | 0.001875679 | [0.001609487, 0.002141872] | 4/5; 4/5 |
| gain:Hidden5/f100 | 0.006232648 | [0.005661133, 0.006804162] | 0.006542982 | [0.005961958, 0.007124006] | 5/5; 5/5 |
| gain:Coarse2/f100 | 0.010894500 | [0.010287134, 0.011501867] | 0.011196143 | [0.010576404, 0.011815882] | 5/5; 5/5 |
| gain:Coarse8/f100 | 0.002986969 | [0.002652612, 0.003321327] | 0.003332311 | [0.002988783, 0.003675838] | 5/5; 5/5 |
| gain:Full5/f100 | 0.001549507 | [0.001300009, 0.001799005] | 0.001659311 | [0.001408051, 0.001910572] | 5/5; 5/5 |

## All eight information-addition effects

These use more-informed gain minus less-informed gain. All16 primary intervals per catalog exclude0; this is conditional event uncertainty, not optimizer-seed certainty.

| Contrast | C mean | C interval | D mean | D interval |
|---|---:|---|---:|---|
| gain_change:Coarse2_minus_Hidden5/f010 | 0.004127366 | [0.003837528, 0.004417203] | 0.004221940 | [0.003921836, 0.004522044] |
| gain_change:Coarse8_minus_Coarse2/f010 | -0.007317024 | [-0.007780152, -0.006853895] | -0.007302295 | [-0.007776153, -0.006828437] |
| gain_change:Full5_minus_Coarse8/f010 | -0.001237058 | [-0.001496055, -0.000978060] | -0.001444682 | [-0.001707238, -0.001182126] |
| gain_change:Full5_minus_Hidden5/f010 | -0.004426716 | [-0.004974532, -0.003878900] | -0.004525037 | [-0.005084484, -0.003965590] |
| gain_change:Coarse2_minus_Hidden5/f100 | 0.004661853 | [0.004395181, 0.004928524] | 0.004653161 | [0.004386349, 0.004919973] |
| gain_change:Coarse8_minus_Coarse2/f100 | -0.007907531 | [-0.008446736, -0.007368326] | -0.007863832 | [-0.008417453, -0.007310212] |
| gain_change:Full5_minus_Coarse8/f100 | -0.001437462 | [-0.001655513, -0.001219412] | -0.001672999 | [-0.001900686, -0.001445313] |
| gain_change:Full5_minus_Hidden5/f100 | -0.004683141 | [-0.005204134, -0.004162148] | -0.004883671 | [-0.005415151, -0.004352190] |

Coarse2 gain exceeds Hidden5; Coarse8 gain is lower than Coarse2; Full5 gain is lower than Coarse8. The endpoint Full5−Hidden5 is negative. Thus the observed gain profile is nonmonotone. A larger correction gain does not mean lower absolute NLL: corrected Hidden5 remains worse than corrected Full5 by0.0190017–0.0195781 nat/event across C/D and fractions.

## Local heterogeneity

| Catalog | Primary | Negative group means /42 | Event share in these regions | Strictly negative336-family bands | Group mean range |
|---|---|---:|---:|---|
| C | Hidden5/f010 | 25 | 59.5363% | 23 | [-0.046805, 0.116801] |
| C | Coarse2/f010 | 18 | 42.8621% | 9 | [-0.017180, 0.095774] |
| C | Coarse8/f010 | 2 | 4.7658% | 0 | [-0.000589, 0.019390] |
| C | Full5/f010 | 3 | 7.1438% | 0 | [-0.001644, 0.006679] |
| C | Hidden5/f100 | 22 | 52.3881% | 20 | [-0.036764, 0.122192] |
| C | Coarse2/f100 | 13 | 30.9575% | 10 | [-0.015493, 0.090846] |
| C | Coarse8/f100 | 6 | 14.2988% | 1 | [-0.003170, 0.016750] |
| C | Full5/f100 | 9 | 21.4451% | 0 | [-0.001546, 0.004972] |
| D | Hidden5/f010 | 27 | 64.2962% | 22 | [-0.043954, 0.117005] |
| D | Coarse2/f010 | 17 | 40.4835% | 9 | [-0.015672, 0.096739] |
| D | Coarse8/f010 | 3 | 7.1480% | 0 | [-0.000577, 0.021450] |
| D | Full5/f010 | 1 | 2.3833% | 0 | [-0.000855, 0.007330] |
| D | Hidden5/f100 | 23 | 54.7636% | 20 | [-0.035895, 0.121775] |
| D | Coarse2/f100 | 14 | 33.3423% | 10 | [-0.012930, 0.095985] |
| D | Coarse8/f100 | 5 | 11.9149% | 1 | [-0.003928, 0.017855] |
| D | Full5/f100 | 9 | 21.4449% | 1 | [-0.002153, 0.004949] |

Hidden5 has22–27 negative group means, with20–23 negative336-family bands despite positive overall gains. The regional event shares52.3881–64.2962% are not individual-event deterioration rates. Full5/f010 preserves one negative optimizer-seed gain on both catalogs; the complete seed tables retain it.

## Runtime/provenance disclosure

The first source-analysis entry failed immediately because threadpoolctl was missing; source_analysis/execution_state.json and frozen_pipeline_stdout.log remain. analysis_runtime_dependency.json records the locally cached threadpoolctl3.7.0 addition and one-thread BLAS/OpenMP validation. The successful source_analysis_retry1 precedes union/plan/generation/scoring. The prior57 module tests are not described as full entry-point validation. No frozen experiment source or protocol was modified by this review.

Public export: include complete aggregate source_analysis_retry1 and C/D fresh_analysis bundles, including the two aggregate bootstrap_replicates.npz files (~5.49MB each), so immutable manifests and figure regeneration remain usable. Exclude all raw JSONL, cache/score NPY and predictions.npz. Publish optional80 checkpoints and run manifests at frozen relative paths, with explicit external-prediction/raw-data requirements. Preserve original manifests and add an export inventory rather than rewriting references to excluded data.
