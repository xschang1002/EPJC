"""Synthetic regression checks for checkpoint and transport acceptance boundaries."""

from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_contract.merge_chunks import validate_event_eligibility
from representation_flow.engine import RepresentationPreprocessor
from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow import run_a0_prototype as trainer
from simulation.confirmation_runner import confirmation_guard_path
from simulation.run_chunks import validate_existing_chunk
from simulation.w0_smoke import assign_transition_targets


def transition_row(face="forward", *, aborted=False, p_out=0.4):
    return {
        "p_in_GeV_c": 0.5,
        "entry": {"momentum_GeV_c": [0.0, 0.0, 0.5]},
        "exit": {"momentum_GeV_c": [0.0, 0.0, p_out]},
        "exit_face": face,
        "aborted": aborted,
    }


@pytest.mark.parametrize("face,aborted,p_out", [
    ("backward", False, 0.4), ("lateral", False, 0.4),
    ("forward", True, 0.4), ("forward", False, 0.0),
])
def test_noneligible_exit_preserves_metadata_but_has_no_targets(face, aborted, p_out):
    row = transition_row(face, aborted=aborted, p_out=p_out)
    row["delta_p_GeV_c"] = 123.0  # stale derived data must also be cleared
    assign_transition_targets(row)
    assert row["S1"] is False
    assert row["exit"] is not None
    assert all(row[key] is None for key in ("p_out_GeV_c", "delta_p_GeV_c", "fractional_loss", "delta_theta_rad"))
    validate_event_eligibility(row)


def test_forward_target_is_derived_from_the_recorded_exit():
    row = transition_row()
    assign_transition_targets(row)
    validate_event_eligibility(row)
    assert row["S1"] is True
    assert row["delta_p_GeV_c"] == pytest.approx(0.1)
    assert row["fractional_loss"] == pytest.approx(0.2)
    # No artificial Y>=0 restriction: energy-gain events remain within Y<p_in.
    row = transition_row(p_out=0.6)
    assign_transition_targets(row)
    validate_event_eligibility(row)
    assert row["delta_p_GeV_c"] < 0


@pytest.mark.parametrize("field,value", [
    ("exit_face", "backward"), ("aborted", True), ("entry", None),
    ("p_out_GeV_c", 0.3), ("delta_p_GeV_c", 0.5),
    ("fractional_loss", 0.9), ("delta_theta_rad", 1.0),
    ("p_in_GeV_c", float("nan")), ("S1", 1),
])
def test_merge_rejects_false_eligibility_or_inconsistent_derived_targets(field, value):
    row = transition_row()
    assign_transition_targets(row)
    row[field] = value
    with pytest.raises(ValueError):
        validate_event_eligibility(row)


def chunk_fixture():
    item = {"input_sha256": "input-hash", "transport_seed": 17, "events": 5, "chunk_id": 0, "thickness_mm": 0.3}
    manifest = {
        "beam_input_sha256": "input-hash", "transport_seed": 17, "events": 5,
        "run_id": "synthetic", "chunk_id": 0, "thickness_mm": 0.3,
        "requested_cut_mm": 0.01, "step_fraction_candidate": 0.1,
        "max_slab_step_mm_requested": 0.03,
    }
    return item, manifest


def test_chunk_resume_accepts_only_matching_transport_configuration():
    item, manifest = chunk_fixture()
    validate_existing_chunk(manifest, item, "synthetic", cut_mm=0.01, step_fraction=0.1)
    with pytest.raises(ValueError, match="step_fraction_candidate"):
        validate_existing_chunk(manifest, item, "synthetic", cut_mm=0.01, step_fraction=0.2)
    with pytest.raises(ValueError, match="requested_cut_mm"):
        validate_existing_chunk(manifest, item, "synthetic", cut_mm=0.02, step_fraction=0.1)
    manifest["max_slab_step_mm_requested"] = 0.06
    with pytest.raises(ValueError, match="max_slab_step"):
        validate_existing_chunk(manifest, item, "synthetic", cut_mm=0.01, step_fraction=0.1)


def test_chunk_resume_refuses_missing_step_provenance():
    item, manifest = chunk_fixture()
    del manifest["step_fraction_candidate"]
    with pytest.raises(ValueError, match="step_fraction_candidate"):
        validate_existing_chunk(manifest, item, "synthetic", cut_mm=0.01, step_fraction=0.1)


def test_confirmation_guard_resolves_in_the_actual_handoff_layout():
    expected = Path(__file__).resolve().parents[1] / "protocol" / "confirmation_guard.py"
    assert confirmation_guard_path() == expected
    assert expected.is_file()


class ToyFlow(torch.nn.Module):
    """One-parameter toy for checkpoint timing, never a scientific flow fit."""
    bound = 8.0

    def __init__(self, **kwargs):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.1))

    def original_terms(self, y, center, mu, scale, p_in, context):
        return {"log_p_y": -(self.theta - y).square()}

    def parameter_count(self):
        return 1


def test_best_checkpoint_keeps_matching_optimizer_sampler_and_preprocessor(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "ConditionalSplineFlow", ToyFlow)
    scores = iter([1.0, 5.0, 9.0, 1.0])
    observed_states = []

    def evaluate(flow, tensors, preprocessor, device, batch_size):
        observed_states.append(deepcopy(flow.state_dict()))
        score = next(scores)
        n = len(tensors[0])
        result = {key: np.zeros(n, dtype=np.float32) for key in ("z", "b", "log_q_z", "log_f_b", "log_p_y", "log_cdf_y", "log_sf_y")}
        result["log_p_y"][:] = -score
        return result

    monkeypatch.setattr(trainer, "_evaluate", evaluate)
    arrays = {
        "event_uid": np.array([f"unit-{i}" for i in range(8)]),
        "p_in": np.full(8, 0.5), "thickness": np.full(8, 0.1), "charge": np.ones(8),
        "y": np.linspace(0.005, 0.02, 8), "split": np.array(["train"] * 6 + ["validation"] * 2),
    }
    pre = RepresentationPreprocessor("Raw", np.zeros(1), 0.0, 0.1)
    args = SimpleNamespace(run_prefix="unit_only", learning_rate=1e-3, max_updates=3, batch_size=2,
                           validation_every=1, eval_batch_size=8, q99_threshold=0.1,
                           experiment_identity="synthetic-checkpoint-timing")
    manifest, _ = trainer._train_one(arrays, np.arange(6), np.arange(6, 8), pre, "Raw", "Small", 5, args, tmp_path)
    payload = torch.load(manifest["checkpoint"], weights_only=True)
    restored_payload, frozen_preprocessor = load_checkpoint(manifest["checkpoint"])
    assert frozen_preprocessor.to_dict() == pre.to_dict()
    assert restored_payload["experiment_identity"] == "synthetic-checkpoint-timing"
    assert payload["selected_update"] == 1
    assert payload["training_state"]["update"] == 1
    assert payload["training_state"]["batch_cursor"] == 2
    assert payload["training_state"]["batch_order"].numel() == 6
    assert all(int(state["step"]) == 1 for state in payload["optimizer_state_dict"]["state"].values())
    assert torch.equal(payload["state_dict"]["theta"], observed_states[0]["theta"])
    assert not torch.equal(observed_states[0]["theta"], observed_states[2]["theta"])
    assert manifest["preprocessor"] == pre.to_dict()
    assert payload["checkpoint_contract"] == manifest["checkpoint_contract"]
    assert payload["experiment_identity"] == "synthetic-checkpoint-timing"
    assert payload["model_config"]["context_dim"] == 5
    checkpoint_bytes = Path(manifest["checkpoint"]).read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        trainer._train_one(arrays, np.arange(6), np.arange(6, 8), pre, "Raw", "Small", 5, args, tmp_path)
    assert Path(manifest["checkpoint"]).read_bytes() == checkpoint_bytes


def test_expired_training_deadline_does_not_start_a_run(tmp_path):
    args = SimpleNamespace(deadline_unix=0.0)
    with pytest.raises(TimeoutError, match="deadline"):
        trainer._train_one({}, np.array([], dtype=int), np.array([], dtype=int), None, "Raw", "Small", 5, args, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module_name", [
    "run_a0_prototype", "run_context_interaction", "run_exploratory_matrix",
    "run_calibrated_exploratory", "run_scale_compensation",
])
def test_exploratory_entrypoints_preserve_existing_summaries_before_loading_data(tmp_path, monkeypatch, module_name):
    module = importlib.import_module(f"representation_flow.{module_name}")
    output = tmp_path / "existing_results"
    output.mkdir()
    existing = output / "exploratory_scores.json"
    original = b'{"status":"FROZEN_EVIDENCE"}'
    existing.write_bytes(original)
    monkeypatch.setattr(sys, "argv", [module_name, "--output", str(output)])
    with pytest.raises(FileExistsError, match="new directory"):
        module.main()
    assert existing.read_bytes() == original
    assert list(output.iterdir()) == [existing]


def test_optimization_audit_preserves_an_existing_report(tmp_path, monkeypatch):
    module = importlib.import_module("representation_flow.optimization_audit")
    output = tmp_path / "frozen_audit.json"
    original = b'{"status":"FROZEN_EVIDENCE"}'
    output.write_bytes(original)
    monkeypatch.setattr(sys, "argv", ["optimization_audit", "--output", str(output)])
    with pytest.raises(FileExistsError, match="new file"):
        module.main()
    assert output.read_bytes() == original


@pytest.mark.parametrize("module_name", [
    "models.fit_pilot_models", "models.fit_formal_matrix", "models.export_formal_predictions",
    "analysis.score_formal_matrix", "analysis.attribute_formal_matrix",
    "analysis.review_main_batch_addendum", "analysis.score_confirmation",
])
def test_legacy_mdn_entrypoints_keep_the_original_physics_center(monkeypatch, module_name):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module(module_name)
    from models.anchor_legacy import pion_bethe_bloch_delta_p as historical
    from models.anchor import corrected_pion_bethe_bloch_delta_p
    p = np.array([0.3, 0.65, 1.0])
    thickness = np.array([0.1, 0.3, 1.0])
    charge = np.array([1, -1, 1])
    actual = module.pion_bethe_bloch_delta_p(p, thickness, charge)
    assert np.array_equal(actual, historical(p, thickness, charge))
    assert not np.allclose(actual, corrected_pion_bethe_bloch_delta_p(p, thickness, charge), atol=1e-12)
