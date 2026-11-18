# Three-Tier Reproduction Architecture

## Tier 1: Complete Archived Aggregate Reconstruction (Default, Self-Contained)

Rebuilds all `48` primary contrast rows, `21` editorial figures, secondary and seed-sensitivity LaTeX tables, `2,000`-draw simultaneous bootstrap intervals, and `C`/`D` predictive/residual diagnostics directly from the committed aggregate records:

```bash
python -m pip install -r requirements.txt
python reproduce.py --output reproduced
```

Runtime is approximately `10–25` seconds on a single CPU core and requires no GPU, PyTorch, or Geant4 installation.

## Tier 2: Binary Checkpoint Verification & Event-Level Scoring Replay

- **Binary checkpoint SHA-256 verification:** Supply `--checkpoint-root <path>` to `python reproduce.py` to hash all `240` binary `checkpoint.pt` artifacts against `CHECKPOINT_SHA256_LEDGER.json`.
- **Event-level scoring replay:** Given the archived raw `C`/`D` Geant4 ledgers and checkpoints, `src/representation_flow/score_information_catalog.py` and `src/representation_flow/run_information_analysis.py` deterministically recompute per-event log-densities, PIT values, and stratified paired bootstrap draws.

## Tier 3: Full Geant4 Simulation & Flow Training

- **Geant4 slab transport (`11.4.1`, `FTFP_BERT`):** `src/simulation/fresh_catalog_runner.py` enforces pre-generation freeze checks (`protocol/confirmation_guard.py`) before launching `42`-stratum silicon slab transport chunks (`src/simulation/make_beam_chunks.py`, `src/simulation/run_chunks.py`, `src/data_contract/primary_visit.py`).
- **Matched spline-flow training:** `src/representation_flow/run_information_study.py` trains the `80`-run four-level information ladder (`Hidden5`, `Coarse2`, `Coarse8`, `Full5` $\times$ `Raw`, `CorrectedPhysics` $\times$ `f010`, `f100` $\times$ seeds `3101`–`3105`) using the RQS engine in `src/representation_flow/engine.py` and the MeV-consistent silicon Bethe–Bloch center in `src/models/anchor.py`.
