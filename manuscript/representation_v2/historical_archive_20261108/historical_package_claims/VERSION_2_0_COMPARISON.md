# 1.0 to 2.0 scientific comparison and error register

## Version boundary

**1.0 (immutable baseline)** is Git commit `410a355f78b91992bdc67a0b9a8f086d37dcef47` in the embedded original repository [`low pion research/`](../low%20pion%20research/).  It contains the frozen paper sources, formal and exploratory manifests, and the historical `Physics` implementation.

**2.0** is the correction work rooted at this repository's `99e625d1bc82633cf9d9577c7b6118fbbf25eb7e`, plus the separately preserved, immutable stage outputs listed in `VERSION_2_0_MANIFEST.json`.  It is a correction/reproduction line, not a rewrite of 1.0 evidence.  Large datasets, checkpoints, predictions, and reports remain external immutable artifacts and are deliberately not Git-committed.

The same tracked source paths currently appear in both outer and embedded repositories.  That does **not** prove that those bytes were the bytes used for historical training: the historical formal/exploratory manifests record different source hashes (below).

## Evidence compared

| Item | 1.0 evidence | 2.0 evidence | Result |
|---|---|---|---|
| Historical anchor | `low pion research/src/models/anchor.py`, SHA-256 `b0e91c…bd760`; historical manifests identify this formula | `p0c-r2-controlled-attribution-preflight/source/centers.py` defines immutable `HistoricalIncorrectPhysics` and all-MeV `CorrectedPhysics` | Center-only intervention is explicit |
| Formal RQS engine | Manifest hash `9b7e88c…a49ce` | currently readable `src/representation_flow/engine.py`, SHA-256 `f035dd…fa8c5`; P4 uses separately pinned R3 core `fbaedf…a5ec` | Exact historical engine source unavailable |
| Long Hidden engine | Manifest hash `d621ec…925c` | current engine hash `f035dd…fa8c5`; P4 R3 core | Exact historical engine source unavailable |
| Formal driver | manifest hashes `727f0369…f186`, `c3d39dc2…4378` | current `run_a0_prototype.py` `0ff1e1…210568`; corrected-formal wrapper records its own identity | Historical byte-for-byte replay not established |
| Data split/subsample | formal catalog 4101; hidden f010 catalog 4101, hidden f100 catalog 4102 | P4 uses catalog 4101 for both arms and fractions | f100 historical interaction is not a same-order paired comparison |
| Corrected controlled evidence | not applicable | P4: 40 center-only cells; matched Raw: 20 cells; corrected formal Full: 30 cells | New evidence is separate; it cannot relabel 1.0 results |

Full hashes and paths are preserved in the listed manifests; shortened hashes here are only for readability.

## Difference table

| Area | Finding | Classification | Scientific consequence | 2.0 handling |
|---|---|---|---|---|
| Bethe--Bloch logarithm | `anchor.py:33` combines a GeV-squared numerator (`me`, `tmax`) with `I_SI_MEV**2` in MeV-squared.  No conversion precedes the division. | **CONFIRMED_BUG** | The Physics center, every historical Physics fit, and any Physics contrast do not represent the stated formula consistently. | Preserve as `HistoricalIncorrectPhysics`; use independently all-MeV `CorrectedPhysics`; never relabel historical Physics results. |
| Thickness and final energy conversion | `thickness_mm / 10` is mm->cm; division by 1000 is MeV->GeV. | **INTENTIONAL_CHANGE: none** | These portions are dimensionally correct; they are not a second unit bug. | Unchanged in the all-MeV equivalent calculation. |
| Raw representation | `c(X)=0`, train-only residual mean, shared train-target standard deviation (`engine.py:294-315, 355-390`). | **VERIFIED_NO_DISCREPANCY** | No implementation discrepancy found in readable source. | Retained; P4-matched Raw cells provide corrected-Physics comparison evidence. |
| Simple representation | Train-only OLS on `[1,u,v,charge]` (`engine.py:345-359`). | **VERIFIED_NO_DISCREPANCY** | No implementation discrepancy found in readable source. | Historical non-Physics contrast remains usable within its limits. |
| Learned representation | 20-column degree-3 polynomial plus charge interaction, train-only ridge and deterministic three-fold CV (`engine.py:231-245, 316-347`).  Folds are row-index modulo 3, not UID-hash folds. | **UNVERIFIED** | This is train-only and therefore is not evidence of label leakage.  It is, however, order-sensitive and the historical source bytes/order cannot be recovered, so historical Learned replay is not exact. | Do not call it a bug without evidence that the published protocol required UID-based folds.  Any new strict reproduction should pre-specify UID-based folds. |
| Full/Hidden conditioning | Full `[u,v,charge,log(p/0.65),0]`; Hidden `[v,charge]`; `p_in` remains in the center/support (`engine.py:246-266`; contract lines 66-70). | **INTENTIONAL_CHANGE** | Hidden is an information-ablation, not an accidental feature leak. | Preserve this interpretation; do not describe Hidden as removing `p_in` from the whole likelihood. |
| RQS forward/inverse/log determinant | Readable implementation uses inverse log-determinant sign consistent with `transform_to_base`, logistic base, identity tails, and `log q(z)+log|d base/dz|` (`engine.py:84-203`). | **VERIFIED_NO_DISCREPANCY** for readable source; **UNVERIFIED** for historical bytes | No second formula error is evidenced. Historical engine hashes differ, so historical numerical identity cannot be claimed. | P4 uses its separately SHA-pinned core; results are not pooled with historical results. |
| Upper-support likelihood | 1.0 contract explicitly fixes `Y < p_in` and implements `log q(z)-log(s)-log F(b)` (`EXPERIMENT_CONTRACT.md:20-22, 73-79`; `engine.py:211-229`). | **INTENTIONAL_CHANGE: none** | It is an explicit model-support choice shared by the original comparisons, not an implementation/source mismatch.  It must not be retroactively called a code bug. | Retain exactly for like-for-like comparisons unless a separately approved scientific study changes the estimand. |
| Checkpoint selection / NLL | Both readable source and frozen contract use minimum source-validation, original-unit mean NLL, validated every 500 updates (`run_a0_prototype.py:116-137`; contract lines 81-91). | **VERIFIED_NO_DISCREPANCY** in readable source; **UNVERIFIED** exact historical execution | No evidence of a different metric or selection rule.  Historical code bytes and RNG execution trace are missing. | Freeze corrected checkpoint manifests before confirmation; do not select after scoring. |
| f100 Full/Hidden aggregation | Formal Full is catalog 4101; historical Hidden uses catalog 4102.  At f100 the UID set is equal but its training order differs (audit `P0CF1_CATALOG_4101_VS_4102.md`). | **PROTOCOL_MISMATCH** | The published f100 interaction combines models whose seed-indexed minibatch trajectories were not matched.  This weakens causal attribution for the historical `+0.088958`; it does not prove an arithmetic or model-code error. | P4 uses 4101 in every paired arm.  Historical and P4 interactions remain separate results. |
| Historical source-hash gap | Formal and Hidden manifests contain source hashes that differ from available `engine.py`/drivers. | **UNVERIFIED** | Exact replay of historical training behavior, including initialization/random consumption/batching details, is impossible from the available tree. | State this limitation; do not infer that later code broke historical scoring. |
| 2.0 P4 and corrected-formal engines | They have explicit identities/configuration and are not byte-identical to all historical engine hashes. | **INTENTIONAL_CHANGE** / **PROTOCOL_MISMATCH** relative to strict historical replay | They support controlled new statements only within their engine/configuration; they cannot by themselves reproduce every 1.0 number. | Keep their artifacts and manifests outside the 1.0 baseline; report separate estimands. |
| Prediction-UID serialization repair | P4 converted the UID array serialization while preserving order, `log_p_y`, checkpoint and reconstructed NLL. | **INTENTIONAL_CHANGE** | No model fitting or numerical score change is indicated. | Keep before/after artifact hashes in the P4 repair manifest. |

## Conclusions

1. **The only independently confirmed implementation bug that changes the historical Physics scientific comparison is the Bethe--Bloch MeV/GeV error.**
2. **A second confirmed code/formula error was not found.**  This is not a proof that none exists: the historical engine and driver bytes identified by their manifests are unavailable.
3. The f100 Full/Hidden catalog/order discrepancy is a material **protocol mismatch**, not a confirmed implementation bug.  It prevents treating the historical interaction as a pure center/conditioner intervention.
4. The original Raw/Simple/Learned sources contain no identified defect from this comparison, but claims based on exact historical execution retain an **UNVERIFIED** provenance limitation because of source-hash gaps.
5. 2.0 must remain a new, explicitly versioned correction line: `HistoricalIncorrectPhysics` is a baseline; `CorrectedPhysics` is not the same representation and must be trained/evaluated under a documented matched protocol.

## Scope

This register is read-only with respect to 1.0 data, checkpoints, predictions, manuscript, and formal/exploratory results.  It does not access confirmation or untouched-holdout data and does not claim a new training result.
