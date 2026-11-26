# Non-monotonic benefits of physical target centering in conditional models of pion momentum loss

Companion code, frozen protocols, aggregate empirical records, and EPJC LaTeX manuscript package for:

> **Non-monotonic benefits of physical target centering in conditional models of pion momentum loss**  
> *Target journal:* **The European Physical Journal C (EPJC)** — *Computing, Software and Data Science*  
> *Compiled manuscripts:* [`output/pdf/physical_centering_epjc_20261118.pdf`](output/pdf/physical_centering_epjc_20261118.pdf) (12 pp., two-column `svjour3`) · [`output/pdf/physical_centering_epjc_supplement_20261118.pdf`](output/pdf/physical_centering_epjc_supplement_20261118.pdf) (12 pp., Electronic Supplementary Material)

---

## Scientific Overview

An invertible physics-informed coordinate shift leaves an unrestricted conditional density family invariant under complete conditioning, yet reshapes the finite-capacity approximation problem when neural conditioner inputs or training budgets are restricted. Using Geant4 (`FTFP_BERT`) transport of $0.3\text{--}1.0\text{ GeV}/c$ $\pi^\pm$ mesons through thin silicon slabs ($0.1, 0.3, 1.0\text{ mm}$), we evaluate paired rational-quadratic spline normalizing flows comparing raw momentum loss ($Y = p_{\mathrm{in}} - p_{\mathrm{out}}$) against a prescribed Bethe–Bloch-centered residual across four controlled conditioner information levels (**no direct**, **2-bin**, **8-bin**, and **exact** incident momentum), two training fractions ($10\%$ and $100\%$), and five paired optimizer seeds (`80` primary fits plus `160` secondary/control fits), frozen prior to generating two independent $500{,}000$-event confirmation catalogs (`C` and `D`).

Key empirical findings reproduced from this repository:
1. **Non-monotonic relative centering gain:** Bethe–Bloch centering improves catalog-mean negative log-likelihood (NLL) at every conditioner level, peaking at **2-bin momentum** ($0.010399\text{--}0.011196\text{ nat/event}$ across catalogs `C`/`D` and training fractions) rather than under no direct momentum ($0.006233\text{--}0.006543\text{ nat/event}$), 8-bin momentum ($0.002987\text{--}0.003332\text{ nat/event}$), or exact momentum ($0.001550\text{--}0.001876\text{ nat/event}$).
2. **Relative gain vs. absolute predictive quality:** Providing **exact incident momentum** to the conditioner still achieves the lowest (best) absolute differential NLL for every target representation.
3. **Global improvement coexists with regional high-momentum losses:** Under no direct momentum, more than half of the 42 $(p_{\mathrm{in}}, t, q)$ strata (`22–27/42`, with `20–23` simultaneously below zero) have negative paired mean gains, and even at the 2-bin peak `13–18/42` strata remain negative (`9–10` simultaneously below zero). Saved original-unit residual medians are consistent with a physical mismatch above minimum ionization ($0.6\text{--}1.0\text{ GeV}/c$): the Geant4 Landau–Vavilov–Bichsel bulk median plateaus while the unscreened Bethe–Bloch mean rises ($T_{\max}$ grows from $15.7$ to $49.8\text{ MeV}$ across $0.555\text{--}1.00\text{ GeV}/c$, with silicon Sternheimer screening omitted at $+1.03\%$ to $+4.34\%$ over $0.30\text{--}1.00\text{ GeV}/c$), imparting a downward residual drift that the $0.65\text{ GeV}/c$ 2-bin split can separate with one bit of momentum information.

---

## Quick Start: One-Command Aggregate Reproduction (`~30 s`)

This repository (`https://github.com/xschang1002/EPJC`, tag `v1.0.0-epjc`) is structured as a **lightweight, Git-LFS-free release (`~86 MB` working tree; `~30 MB` compressed Git push/clone payload)** so reviewers can clone and verify every figure, table, confidence band, and diagnostic summary immediately on a standard laptop without downloading gigabytes of binary checkpoints or Geant4 event ledgers.

```bash
python -m pip install -r requirements.txt
python reproduce.py --output reproduced
```

*(For exact Windows x64 CPython 3.12 version pinning used during archival verification, see [`requirements-aggregate.txt`](requirements-aggregate.txt).)*

### What `python reproduce.py --output reproduced` Verifies and Reconstructs

1. **Cryptographic input integrity:** Verifies SHA-256 digests and byte counts of all tracked files in [`PACKAGE_MANIFEST.json`](PACKAGE_MANIFEST.json) and confirms zero input mutation upon completion.
2. **Pre-generation freeze & model-registry verification:** Verifies all source, protocol, and analysis SHA-256 bindings in [`protocol/fresh_catalog_20261108.json`](protocol/fresh_catalog_20261108.json), validates all `240` `run_manifest.json` files on disk, and cross-verifies all `240` checkpoint SHA-256 digests in [`CHECKPOINT_SHA256_LEDGER.json`](CHECKPOINT_SHA256_LEDGER.json) against the frozen study manifests (or hashes binary `.pt` checkpoints directly if `--checkpoint-root <dir>` is supplied).
3. **Primary tables & stratified bootstrap bands:** Reconstructs all `48` primary contrast rows (`source`, `C`, `D`) with byte-identical CSV verification and recomputes the `2,000`-draw stratified paired event-bootstrap intervals across the `16`-contrast global, `42`-stratum within-primary, and `336`-cell joint simultaneous families (maximum numerical discrepancy $< 10^{-11}$).
4. **All 21 vector/raster figures & LaTeX tables:** Regenerates all `21` manuscript and supplementary figures (`PDF` + `PNG`) verifying exact `PLOTTED_AGGREGATES.json` numerical identity, and rebuilds all primary, secondary, and seed-sensitivity LaTeX tables verifying exact cell-by-cell agreement against [`manuscript/representation_v2/epjc_submission/`](manuscript/representation_v2/epjc_submission/).
5. **Post-hoc five-seed sensitivity & regional diagnostics:** Recomputes all four five-seed sensitivity CSVs (including leave-one-seed-out and exploratory $t$-intervals), reaggregates all `160` predictive calibration rows and `8` residual-quantile summaries across `C` and `D`, and authenticates the saved `float32`/`float64` support-margin audit receipts.
6. **Audit receipt:** Writes a machine-readable verification report to `reproduced/REPRODUCTION_RESULT.json`.

### Running the Unit & Regression Test Suite

To run the Python unit tests covering the Bethe–Bloch center, spline-flow log-density/log-survival numerics, information-ladder encodings, and freeze contracts (requires `torch` and `pytest`):

```bash
python -m pip install torch pytest
PYTHONPATH=src pytest tests -q
```

---

## Terminology Mapping (Manuscript Display Names ↔ Frozen Code Identifiers)

Frozen protocol manifests, run IDs, and CSV schemas preserve original pre-registration identifiers, whereas the EPJC manuscript uses reader-facing physics labels:

| Manuscript Display Name | Frozen Code / Manifest Identifier | Role |
| :--- | :--- | :--- |
| **No direct momentum** ($N$) | `Hidden5` | Conditioner receives $(0, v, q, 0, 0)$; exact $p_{\mathrm{in}}$ retained in center & support |
| **2-bin momentum** ($M_2$) | `Coarse2` | Midpoint encoding on $[0.3, 0.65), [0.65, 1.0]\text{ GeV}/c$ |
| **8-bin momentum** ($M_8$) | `Coarse8` | Midpoint encoding on 8 equal-width bins across $[0.3, 1.0]\text{ GeV}/c$ |
| **Exact momentum** ($E$) | `Full5` | Exact $(u, v, q, \log(p_{\mathrm{in}}/0.65), 0)$ input |
| **Raw** ($R$) | `Raw` | Pooled training-mean center $\mu$ only (no $p_{\mathrm{in}}, t$ curve) |
| **BB-centered** ($B$) | `CorrectedPhysics` | Prescribed MeV-consistent silicon Bethe–Bloch momentum-loss center $c_{\mathrm{BB}}(p_{\mathrm{in}}, t, q)$ |
| **Linear center** ($L$) | `Simple` | Fitted linear location baseline on exact conditioning (secondary comparison) |
| **Polynomial-ridge center** ($P$) | `Learned` | Cross-validated polynomial-ridge location baseline (secondary comparison) |
| **10% / 100% training** | `f010` / `f100` | Nested source-training fractions (`59,952` / `599,526` eligible events; `20,000` updates) |

---

## Three-Tier Reproducibility Scope

| Tier | Scope | Included in This Git Repo? | Requirements |
| :--- | :--- | :---: | :--- |
| **Tier 1 — Aggregate & Presentation Replay** | Rebuild all 21 figures, all main/ESM tables, 2,000-draw simultaneous bootstrap intervals, 5-seed sensitivity CSVs, and C/D calibration & residual diagnostics | **Yes (100% self-contained)** | Python $\ge 3.11$, `numpy`, `scipy`, `matplotlib` (`python reproduce.py`) |
| **Tier 2 — Binary Checkpoint & Event-Score Replay** | Hash-verify all 240 `.pt` checkpoints (`~0.61 GiB`) or replay per-event log-likelihoods on the raw `C`/`D` Geant4 ledgers (`~62 GiB` uncompressed) | **SHA-256 manifests included; binary blobs archived locally** | Pass `--checkpoint-root <dir>` to `reproduce.py` for `.pt` byte verification; binary checkpoints and full event/score archives are available from the corresponding author upon reasonable request |
| **Tier 3 — Full Geant4 Transport & Flow Retraining** | Regenerate `500k` silicon slab catalogs and retrain 80+160 conditional RQS flows under pre-frozen random seeds | **All source code & frozen protocols included** | Python 3.12, `torch`, native Geant4 `11.4.1` (`FTFP_BERT`) bindings (`src/simulation/fresh_catalog_runner.py`, `src/representation_flow/run_information_study.py`) |

See [`REPRODUCE.md`](REPRODUCE.md), [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md), and [`docs/DATA_AND_PROVENANCE.md`](docs/DATA_AND_PROVENANCE.md) for complete details.

---

## Repository Layout

```text
epjc-github-release/
├── reproduce.py                        # One-command isolated aggregate reproduction entry point
├── verify_saved_diagnostics.py         # Reaggregation verifier for C/D predictive & residual diagnostics
├── PACKAGE_MANIFEST.json               # SHA-256 & byte-length manifest for every repository input
├── CHECKPOINT_SHA256_LEDGER.json       # Cryptographic SHA-256 & byte ledger for all 240 registered checkpoints
├── DIAGNOSTIC_INDEX.json               # Index binding C/D predictive & residual JSON summaries
├── requirements.txt                    # Cross-platform aggregate reproduction dependencies
├── requirements-aggregate.txt          # Exact Windows x64 CPython 3.12 verification lockfile
├── LICENSE                             # Dual license: MIT (code) + CC-BY-4.0 (aggregate data/figures, SCOAP3)
├── REPRODUCE.md                        # Detailed CLI & output specification for reproduce.py
├── output/pdf/                         # Compiled EPJC two-column main PDF (12 pp.) & ESM PDF (12 pp.)
├── manuscript/representation_v2/
│   ├── epjc_submission/                # Standalone EPJC LaTeX source (`svjour3`, `spphys`, `main.bbl`, figures, tables)
│   ├── overleaf_submission/            # Frozen figure/table generator modules & reference CSV/JSON outputs
│   └── historical_archive_20261108/    # Compact text/table record of preliminary development controls (ESM Sect. S5)
├── protocol/                           # Pre-frozen catalog, study, and statistical analysis JSON contracts
├── src/                                # Core Python packages (`models/`, `representation_flow/`, `simulation/`, `data_contract/`)
├── scripts/                            # Read-only aggregate gate & editorial table reconstruction adapters
├── reports/                            # Frozen C/D aggregate summaries, 2,000-draw bootstrap NPZs, 240 run manifests, support audits
├── tests/                              # Unit and numerical regression test suite (`pytest`)
└── docs/                               # Provenance and multi-tier reproduction documentation
```

---

## License & Citation

- **Software & source code:** [MIT License](LICENSE)
- **Aggregate data, bootstrap replicates, figures, tables & documentation:** [CC BY 4.0](LICENSE) (compatible with EPJC / SCOAP³ open-access publishing)
- Machine-readable citation metadata: [`CITATION.cff`](CITATION.cff)
