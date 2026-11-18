# Aggregate Reproduction Guide (`reproduce.py`)

This guide documents the one-command aggregate verification and figure/table reconstruction workflow in `reproduce.py`.

## 1. Environment Setup

Any standard Python `3.11`–`3.13` installation on Linux, macOS, or Windows can run the complete aggregate reconstruction (no GPU, PyTorch, Geant4, or network access at runtime is required):

```bash
python -m pip install -r requirements.txt
```

If you wish to match the exact Windows x64 CPython `3.12` package builds used when generating the archived receipts:

```bash
python -m pip install -r requirements-aggregate.txt
```

## 2. One-Command Execution

Run from the repository root into a **new** output directory (existing directories are never overwritten):

```bash
python reproduce.py --output reproduced
```

### Optional Flags

- `--figures all|none` (default: `all`): Set `--figures none` to skip Matplotlib figure rendering and run only table, bootstrap, seed-sensitivity, and diagnostic verification (~3 seconds).
- `--checkpoint-root <PATH>`: Optional path to a local research checkout or extracted full checkpoint archive containing the 240 binary `checkpoint.pt` files (`~610 MB`).
  - **Without `--checkpoint-root` (default lightweight Git-clone mode):** `reproduce.py` verifies all `240` `run_manifest.json` files byte-for-byte on disk and cross-verifies all `240` checkpoint SHA-256 digests across `FROZEN_FRESH_MODEL_UNION.json`, the three `FROZEN_CHECKPOINT_MANIFEST.json` rosters, and `CHECKPOINT_SHA256_LEDGER.json`.
  - **With `--checkpoint-root <PATH>`:** `reproduce.py` additionally streams and verifies the SHA-256 digest and byte length of all `240` binary `checkpoint.pt` files without deserializing or executing any neural model.

## 3. Step-by-Step Verification Pipeline

`reproduce.py` executes five isolated, deterministic checks:

1. **Input Manifest Gate (`PACKAGE_MANIFEST.json`):** Confirms every tracked file exists with its exact recorded byte length and SHA-256 hash, and enforces the aggregate-only array allowlist (only `C/bootstrap_replicates.npz` and `D/bootstrap_replicates.npz` are permitted binary arrays).
2. **Archived Lineage & Bootstrap Gate (`scripts/reproduce_information_aggregates.py`):**
   - Verifies every source file and audit snapshot bound by `protocol/fresh_catalog_20261108.json`;
   - Verifies the `240`-run model union (`80` primary + `160` secondary/control; noting that `20` exact-momentum primary fits deterministically match earlier Small exact-conditioner runs);
   - Recomputes the `48` primary contrast rows (`source`, `C`, `D`) and checks exact CSV byte identity;
   - Recomputes simultaneous and pointwise confidence bands from the `2,000` saved stratified paired bootstrap draws (`16` global contrasts, `42` within-primary strata, `336` joint group cells; tolerance `< 10^{-11}`).
3. **Editorial Figures (`generate_editorial_figures.py`):** Rebuilds all `21` vector PDF and raster PNG figures into `reproduced/editorial_figures/` and asserts exact JSON equality of `PLOTTED_AGGREGATES.json` against both `epjc_submission/editorial_figures/` and `overleaf_submission/editorial_figures/`.
4. **LaTeX Tables & Five-Seed Sensitivity (`scripts/rebuild_editorial_tables.py` & `analyze_seed_sensitivity.py`):** Rebuilds all LaTeX tables and asserts exact row-by-row agreement against both `epjc_submission/editorial_tables/` and `overleaf_submission/editorial_tables/`, then recomputes all four five-seed sensitivity CSVs and verifies exact SHA-256 byte identity.
5. **Predictive/Residual Diagnostics & Support Receipts (`verify_saved_diagnostics.py`):** Reaggregates all `160` predictive calibration rows and `8` residual-quantile summaries across `C` and `D`, authenticates the three saved support-audit summary JSONs, re-verifies `PACKAGE_MANIFEST.json` to prove zero input mutation, and writes `reproduced/REPRODUCTION_RESULT.json`.
