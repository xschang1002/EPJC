import copy
import json
from pathlib import Path

import numpy as np

from representation_flow import run_correction_study as study
from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.experiment_contract import validate_completed


def test_controlled_cells_share_samples_and_initialization_and_reload(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "protocol/correction_v2_20261107.json").read_text())
    protocol = copy.deepcopy(protocol)
    protocol["sample_fractions"] = [1.0]
    protocol["seeds"] = [3101]
    protocol["optimization"].update(max_updates=2, batch_size=4, validation_every=1, eval_batch_size=8)
    cache = tmp_path / "cache"
    cache.mkdir()
    n = 20
    arrays = {"event_uid": np.array([f"synthetic:{i}" for i in range(n)]),
              "split": np.array(["train"] * 12 + ["validation"] * 8),
              "p_in": np.linspace(.31, .95, n), "thickness": np.resize([.1, .3, 1.], n),
              "charge": np.resize([-1., 1.], n), "y": np.linspace(.00001, .0004, n)}
    for name, array in arrays.items():
        np.save(cache / (name + ".npy"), array)
    np.save(cache / "indices_f100.npy", np.arange(12))
    np.save(cache / "indices_validation.npy", np.arange(12, n))
    monkeypatch.setattr(study, "_ARRAYS", arrays)
    monkeypatch.setattr(study, "_CACHE", cache)
    output = tmp_path / "study"
    jobs = study.build_jobs(cache, protocol, output, "synthetic-test-identity", 2, 10**12)
    assert len(jobs) == 6
    rows = [study.run_cell(j) for j in jobs]
    for context in protocol["contexts"]:
        grouped = [r for r in rows if r["context_label"] == context]
        assert len({r["initialization_hash"] for r in grouped}) == 1
        assert len({r["train_event_uid_sha256"] for r in grouped}) == 1
        assert len({r["validation_event_uid_sha256"] for r in grouped}) == 1
    for row in rows:
        payload, pre = load_checkpoint(row["checkpoint"])
        assert pre.name == row["representation"]
        assert payload["experiment_identity"] == "synthetic-test-identity"
        assert payload["selected_update"] == payload["training_state"]["update"]
        assert payload["checkpoint_contract"]["eligibility"] == "forward"
    assert validate_completed(rows, output, "synthetic-test-identity", {j["run_id"] for j in jobs}) == {j["run_id"] for j in jobs}
