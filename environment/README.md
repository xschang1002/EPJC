# Environment files

`requirements-core.txt` pins NumPy, SciPy and PyTorch for flow training and
analysis. `requirements-test.txt` adds pytest; `requirements-plot.txt` adds
Matplotlib. `requirements-ml.txt` remains a compatibility entry point for core
plus tests. `requirements-sim.txt` describes the simulation-side environment.
The original machine used a local `.venv-ml` and `.venv-sim`; those environments
are intentionally not copied into this repository.

The 2026-11-07 local verification found CPython 3.12.13, NumPy 2.5.3,
SciPy 1.18.1, PyTorch 2.14.1+cpu and pytest 9.1.1 in the original ML environment.
CUDA was unavailable. Matplotlib was absent there; version 3.11.2 was separately
imported and used for a render probe from the local CPython 3.12 wheel cache.
This verifies the local runtime, not availability of every wheel on every
platform. Select the appropriate PyTorch build when provisioning another host.

Historical reports and checkpoints retain their identities and their erroneous
center labels. They are not corrected-center results. New runs use the separate
correction protocol and output directory documented in the reproduction guide.

