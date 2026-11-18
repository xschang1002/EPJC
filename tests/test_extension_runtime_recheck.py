"""Runtime-only decisions must preserve measured evidence and the strict budget."""
import copy
import hashlib
import json

import pytest

from representation_flow.recheck_extension_runtime import (
    INITIAL_SHA, INITIAL_SNAPSHOT, calculate_runtime_decision, initial_measurement, persist_runtime_decision,
)


def fixture_records():
    measurement = {
        "common_identity": {"frozen": "unchanged"},
        "maximum_large_cell_seconds": 57.51648689992726,
        "large_benchmark_updates": 5000,
        "forecast_extension_training_seconds": 4942.5269209826365,
        "benchmark_wall_seconds": 69.26253199577332,
        "allowed_combined_under_three_hours": False,
        "conservative_combined_total_seconds": 10838.516737848753,
    }
    core = {"status": "COMPLETE", "execution_started_unix": 1000., "runs": [{"validation_nll": -1.0}] * 60}
    protocol = {"runtime_policy": {"max_total_seconds": 10800, "forecast_safety_factor": 1.35,
                "reserved_analysis_seconds": 1200, "prior_core_benchmark_seconds": 27.208680391311646,
                "prior_extension_benchmark_seconds": 24.347630500793457}}
    core_benchmark = {"benchmark_wall_seconds": 15.0}
    return measurement, core, protocol, core_benchmark


@pytest.mark.parametrize("status,count", [("RUNNING", 60), ("COMPLETE", 59), ("INCOMPLETE", 60)])
def test_unfinished_core_cannot_authorize_extension(status, count):
    measurement, core, protocol, benchmark = fixture_records()
    core.update(status=status, runs=[{}] * count)
    with pytest.raises(ValueError, match="core60 must be complete"):
        calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.)


def test_complete_with_retained_failures_cannot_authorize_extension():
    measurement, core, protocol, benchmark = fixture_records()
    core["failures"] = [{"run_id": "failed_cell", "error": "retained failure"}]
    with pytest.raises(ValueError, match="core60 must be complete"):
        calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.)


def test_completed_actual_wall_replaces_only_the_already_elapsed_forecast():
    measurement, core, protocol, benchmark = fixture_records()
    decision = calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.)
    expected = 1900. + 4942.5269209826365 * 1.35 + 1200 + 27.208680391311646 + 24.347630500793457 + 15 + 69.26253199577332 + 90
    assert decision["conservative_combined_total_seconds"] == pytest.approx(expected)
    assert decision["allowed_combined_under_three_hours"] is True
    assert decision["initial_forecast_allowed"] is False
    assert decision["actual_core_elapsed_seconds"] == 1900.
    assert decision["scientific_scores_used_for_decision"] is False
    core["runs"] = [{"validation_nll": 10**30}] * 60
    assert calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.) == decision


def test_elapsed_waiting_can_reject_an_otherwise_allowed_extension():
    measurement, core, protocol, benchmark = fixture_records()
    decision = calculate_runtime_decision(measurement, core, protocol, benchmark, 4000.)
    assert decision["actual_core_elapsed_seconds"] == 3000.
    assert decision["conservative_combined_total_seconds"] >= 10800
    assert decision["allowed_combined_under_three_hours"] is False
    assert decision["remaining_margin_seconds"] < 0


@pytest.mark.parametrize("key,value", [("max_total_seconds", 10801), ("forecast_safety_factor", 1.34), ("reserved_analysis_seconds", 1199)])
def test_budget_policy_cannot_be_relaxed(key, value):
    measurement, core, protocol, benchmark = fixture_records()
    protocol["runtime_policy"][key] = value
    with pytest.raises(ValueError, match="cannot be relaxed"):
        calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.)


def test_decision_updates_preserve_initial_measurement_and_can_revoke_permission(tmp_path):
    measurement, core, protocol, benchmark = fixture_records()
    raw = (json.dumps(measurement, indent=2, sort_keys=True) + "\n").encode()
    (tmp_path / "benchmark_result.json").write_bytes(raw)
    decision = calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.)
    first = persist_runtime_decision(tmp_path, measurement, raw, decision)
    assert first.is_file()
    assert (tmp_path / INITIAL_SNAPSHOT).read_bytes() == raw
    assert (tmp_path / INITIAL_SHA).read_text().split()[0] == hashlib.sha256(raw).hexdigest()
    updated = json.loads((tmp_path / "benchmark_result.json").read_text())
    for key in ("common_identity", "maximum_large_cell_seconds", "large_benchmark_updates", "forecast_extension_training_seconds", "benchmark_wall_seconds"):
        assert updated[key] == measurement[key]
    assert updated["allowed_combined_under_three_hours"] is True
    assert updated["initial_forecast_allowed"] is False
    original, recovered_bytes = initial_measurement(tmp_path)
    assert original == measurement and recovered_bytes == raw
    refused = calculate_runtime_decision(original, core, protocol, benchmark, 4000.)
    second = persist_runtime_decision(tmp_path, original, recovered_bytes, refused)
    assert second != first
    assert len(list((tmp_path / "runtime_decisions").glob("*.json"))) == 2
    assert (tmp_path / INITIAL_SNAPSHOT).read_bytes() == raw
    assert json.loads((tmp_path / "benchmark_result.json").read_text())["allowed_combined_under_three_hours"] is False


def test_modified_initial_measurement_is_rejected(tmp_path):
    measurement, core, protocol, benchmark = fixture_records()
    raw = json.dumps(measurement).encode()
    (tmp_path / "benchmark_result.json").write_bytes(raw)
    decision = calculate_runtime_decision(measurement, core, protocol, benchmark, 2900.)
    persist_runtime_decision(tmp_path, measurement, raw, decision)
    (tmp_path / INITIAL_SNAPSHOT).write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        initial_measurement(tmp_path)
