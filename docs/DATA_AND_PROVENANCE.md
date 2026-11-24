# Data Availability and Cryptographic Provenance

## 1. Included Aggregate & Provenance Assets (`~86 MB`)

This Git repository includes every aggregate dataset, bootstrap resample array, configuration contract, and cryptographic digest required to verify and rebuild all tables, figures, confidence bands, and diagnostic statistics in the EPJC main article and Electronic Supplementary Material:

- **Pre-generation freeze contracts:** `protocol/fresh_catalog_20261108.json`, `protocol/information_study_20261108.json`, `protocol/information_statistical_analysis_20261108.json`, and `reports/information_extension_20261108/PROTOCOL_FREEZE.json`.
- **Source & confirmation aggregate analyses:** `reports/information_extension_20261108/source_analysis/`, `fresh_analysis/C/`, and `fresh_analysis/D/`, including `bootstrap_replicates.npz` (`2,000` paired stratified resamples per catalog), `predictive_diagnostics.json`, `residual_diagnostics.json`, and `density_component_diagnostics.json`.
- **Model registry & checkpoint SHA-256 ledger:** All `240` `run_manifest.json` files, the three `FROZEN_CHECKPOINT_MANIFEST.json` rosters, `FROZEN_FRESH_MODEL_UNION.json`, and `CHECKPOINT_SHA256_LEDGER.json`.
- **Post-hoc five-seed sensitivity & numerical boundary audits:** `reports/positioning_revision_20261108/seed_sensitivity/` and `reports/support_audit_20261108/`.

## 2. Omitted Multi-Gigabyte Binary Assets

To keep this repository directly cloneable and pushable via standard Git without Git LFS quota barriers, two categories of large binary files are intentionally excluded from the Git working tree while remaining bound by SHA-256 digests in the committed manifests:

1. **Binary PyTorch Checkpoints (`240` `.pt` files, `~610 MB`):** Never deserialized during aggregate reproduction. If supplied locally (e.g., from a GitHub Release archive), pass `--checkpoint-root <dir>` to `python reproduce.py` to verify all `240` `.pt` files byte-for-byte against `CHECKPOINT_SHA256_LEDGER.json`.
2. **Raw Geant4 Event Ledgers & Per-Event Score Arrays (`~62 GiB` uncompressed):** The `1,000,000`-event source ledger, the two `500,000`-event confirmation ledgers (`C` and `D`), and the `5,280` per-event float32 component arrays are archived locally and available from the corresponding author upon reasonable request. Their SHA-256 digests (`events_sha256`, `score_manifest_sha256`, `cache_manifest.json`) are published in `reports/support_audit_20261108/` and `reports/information_extension_20261108/`.
