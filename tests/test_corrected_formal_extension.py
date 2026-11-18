"""Synthetic checks for the prespecified 20-reused/100-new formal extension."""
import copy
import json
from pathlib import Path

import numpy as np

from representation_flow import run_corrected_formal_extension as extension
from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import subset_indices
from representation_flow.experiment_contract import canonical_hash, sha256, validate_completed


def synthetic_cache(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "protocol/corrected_formal_extension_20261107.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    cache = tmp_path / "base_cache"
    cache.mkdir()
    n = 120
    arrays = {"event_uid": np.array([f"synthetic-extension:{i}" for i in range(n)]),
              "split": np.array(["train"] * 96 + ["validation"] * 24),
              "p_in": np.linspace(.31, .99, n), "thickness": np.resize([.1, .3, 1.], n),
              "charge": np.resize([-1., 1.], n), "y": .0001 + np.arange(n) * .000001}
    for name, values in arrays.items():
        np.save(cache / f"{name}.npy", values, allow_pickle=False)
    index_root = tmp_path / "indices"
    record = extension.prepare_extension_cache(cache, index_root, protocol, {"synthetic": True})
    return arrays, cache, index_root, record, protocol, path


def test_extension_has_exactly_40_small_and_60_large_new_cells(tmp_path):
    arrays, cache, index_root, record, protocol, path = synthetic_cache(tmp_path)
    before = {p.name: sha256(p) for p in cache.glob("*.npy")}
    jobs = extension.build_jobs(protocol, record, tmp_path / "output", "synthetic", 2, 10**12, path)
    assert len(jobs) == 100
    assert len({j["run_id"] for j in jobs}) == 100
    assert sum(j["capacity"] == "Small" for j in jobs) == 40
    assert sum(j["capacity"] == "Large" for j in jobs) == 60
    assert not any(extension.reused_cell(j["representation"], j["capacity"], j["fraction"]) for j in jobs)
    for tag, fraction in (("f010", .1), ("f030", .3), ("f100", 1.)):
        expected = subset_indices(arrays, fraction, protocol["subset_catalog_id"])
        actual = np.load(index_root / f"indices_{tag}.npy")
        assert np.array_equal(actual, expected)
        assert record["fractions"][tag]["ordered_uid_sha256"] == canonical_hash(arrays["event_uid"][actual].tolist())
    assert {p.name: sha256(p) for p in cache.glob("*.npy")} == before
    assert extension.prepare_extension_cache(cache, index_root, protocol, {"synthetic": True}) == record


def test_large_learned_cell_reloads_its_frozen_polynomial_and_metadata(tmp_path, monkeypatch):
    arrays, cache, index_root, record, protocol, path = synthetic_cache(tmp_path)
    protocol = copy.deepcopy(protocol)
    protocol["optimization"].update(batch_size=8, validation_every=1, eval_batch_size=16)
    jobs = extension.build_jobs(protocol, record, tmp_path / "output", "synthetic", 2, 10**12, path)
    job = next(j for j in jobs if j["capacity"] == "Large" and j["fraction"] == 1.0 and j["representation"] == "Learned")
    monkeypatch.setattr(extension, "_ARRAYS", arrays)
    monkeypatch.setattr(extension, "_INDEX_ROOT", index_root)
    row = extension.run_cell(job)
    payload, pre = load_checkpoint(row["checkpoint"])
    assert len(pre.coefficients) == 20
    assert pre.to_dict() == job["preprocessor"]
    assert payload["model_config"]["hidden"] == 128
    assert row["matrix_cell_id"] == job["run_id"]
    assert row["context_label"] == "Full"
    assert row["reused"] is False
    assert row["experiment_identity"] == "synthetic"
    assert np.isfinite(row["validation_nll"])
    assert validate_completed([row], tmp_path / "output", "synthetic", {job["run_id"]}) == {job["run_id"]}
