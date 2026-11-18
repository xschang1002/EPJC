# Paper display names and immutable run identifiers

| Paper display | Internal ID | Scope |
| --- | --- | --- |
| Raw | `Raw` | Current primary and secondary models |
| BB-centered | `CorrectedPhysics` | Current primary and corrected secondary models |
| Linear center | `Simple` | Fitted-center secondary comparisons |
| Polynomial-ridge center | `Learned` | Fitted-center secondary comparisons |
| No direct momentum | `Hidden5` | Current primary80 only |
| 2-bin momentum | `Coarse2` | Current primary80 only |
| 8-bin momentum | `Coarse8` | Current primary80 only |
| Exact momentum | `Full5` | Current primary80 only |
| Exact momentum | `Full` | Earlier matched correction and historical Full runs; identify cohort |
| Earlier no direct momentum (two inputs) | `Hidden` | Earlier correction/historical Hidden runs only |
| H (erroneous preliminary center) | `HistoricalIncorrectPhysics` | Explicit historical comparator and documented controlled reruns |
| H (erroneous preliminary center) | `Physics` | Original pre-correction plots/reports only |

The full JSON/CSV adds definitions and limits. This is an editorial alias map, not a new representation, model, transformation or experiment. In particular, both Linear center and Polynomial-ridge center were fitted on training data; only their display names change. No generic global search/replace of the English word “learned” is appropriate.
