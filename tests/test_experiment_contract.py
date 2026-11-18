import json
from types import SimpleNamespace

import pytest

from representation_flow.experiment_contract import (
    artifact_record, bind_output, data_identity, validate_completed, validate_frozen_inputs,
)


def fixture_protocol(tmp_path):
    merged = tmp_path / "data"
    merged.mkdir()
    (merged / "manifest.json").write_text("{}")
    (merged / "events.jsonl").write_text('{"id": 1}\n')
    actual = data_identity(merged)
    freeze = {"status": "FORMAL_PROTOCOL_FROZEN", "optimization": {
        "learning_rate": 0.0003, "max_updates": 20000, "batch_size": 512, "validation_every_updates": 500},
        "source": {"manifest_json_sha256": actual["manifest_sha256"], "events_jsonl_sha256": actual["events_sha256"]}}
    preflight = {"status": "FORMAL_PREFLIGHT_PASS", "checks": {"a": True}, "manifest_hash_semantics": {
        "audit_A0_source_manifest_sha256": actual["manifest_sha256"], "center_diagnostics_source_manifest_sha256": actual["events_sha256"]}}
    args = SimpleNamespace(learning_rate=0.0003, max_updates=20000, batch_size=512, validation_every=500)
    return merged, freeze, preflight, args


@pytest.mark.parametrize("field,value", [("batch_size", 1024), ("validation_every", 1), ("learning_rate", .001), ("max_updates", 1000)])
def test_frozen_training_parameters_are_all_enforced(tmp_path, field, value):
    merged, freeze, preflight, args = fixture_protocol(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_frozen_inputs(freeze, preflight, args, merged)


def test_preflight_pass_does_not_authorize_changed_data(tmp_path):
    merged, freeze, preflight, args = fixture_protocol(tmp_path)
    validate_frozen_inputs(freeze, preflight, args, merged)
    (merged / "events.jsonl").write_text('{"id": 2}\n')
    with pytest.raises(ValueError, match="source hashes"):
        validate_frozen_inputs(freeze, preflight, args, merged)


def test_resume_rejects_legacy_and_changed_identity_without_modification(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    artifact = old / "formal_scores.json"
    artifact.write_text('{"runs": ["historical"]}')
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match="no experiment identity"):
        bind_output(old, {"center": "CorrectedPhysics"})
    assert artifact.read_bytes() == before
    assert not (old / "experiment_identity.json").exists()
    new = tmp_path / "new"
    bind_output(new, {"center": "HistoricalIncorrectPhysics"})
    before = (new / "experiment_identity.json").read_bytes()
    with pytest.raises(ValueError, match="resume rejected"):
        bind_output(new, {"center": "CorrectedPhysics"})
    assert (new / "experiment_identity.json").read_bytes() == before


def test_resume_rechecks_every_completed_artifact(tmp_path):
    identity = bind_output(tmp_path, {"study": "test"})
    artifacts = {}
    for name in ("checkpoint", "predictions", "run_manifest"):
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        artifacts[name] = artifact_record(path, tmp_path)
    row = {"run_id": "one", "experiment_identity": identity, "artifacts": artifacts}
    assert validate_completed([row], tmp_path, identity, {"one"}) == {"one"}
    (tmp_path / "predictions.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifact"):
        validate_completed([row], tmp_path, identity, {"one"})
