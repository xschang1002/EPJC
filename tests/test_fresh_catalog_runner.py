"""Synthetic tests: no formal events, model inference or Geant4 import."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import pytest

from simulation import fresh_catalog_runner as fresh


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path, root, key="audit"):
    return {"root_key": key, "path": path.relative_to(root).as_posix(),
            "sha256": fresh.sha256(path), "bytes": path.stat().st_size}


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_fresh_freeze")
    roots = {"repo": fresh.REPO, "audit": root, "models": root / "models"}
    roots["models"].mkdir()
    models = []
    for i in range(240):
        artifacts = {}
        for role in ("checkpoint", "predictions", "run_manifest"):
            path = roots["models"] / f"model_{i}_{role}"
            path.write_bytes(f"synthetic:{i}:{role}".encode())
            artifacts[role] = ref(path, roots["models"], "models")
        models.append({"model_id": f"model_{i}", "root_key": "models",
                       "cohort": "information80" if i < 80 else "legacy160", "artifacts": artifacts})
    union = {"schema": "fresh-model-union/v1", "status": "COMPLETE_FROZEN_FRESH_MODEL_UNION",
             "expected_models": 240, "models": models}
    information_freeze = put(roots["models"] / "information_freeze.json", {
        "schema": "information-checkpoint-freeze/v1", "status": "COMPLETE_INFORMATION_CHECKPOINT_FREEZE",
        "expected_cells": 80, "protocol_sha256": "synthetic-information-protocol",
        "models": [{"run_id": m["model_id"], "artifacts": m["artifacts"]} for m in models[:80]],
    })
    union["information_freeze"] = ref(information_freeze, roots["models"], "models")
    upath = put(root / "union.json", union)
    inventory = {"beam_seeds": [100], "transport_master_seeds": [101],
                 "transport_seeds": list(fresh.chunk_transport_seeds(101, list(range(42))).values()),
                 "run_ids": ["old"]}
    ipath = put(root / "inventory.json", inventory)
    analysis = put(root / "analysis.json", {"schema": "fresh-catalog-statistical-analysis/v1",
                                          "status": "FROZEN_FRESH_STATISTICAL_ANALYSIS"})
    source_analysis = put(root / "source_analysis.json", {
        "schema": "information-source-analysis/v1", "status": "COMPLETE_INFORMATION_SOURCE_ANALYSIS",
        "expected_models": 80, "model_count": 80, "all_validation_predictions_replayed": True,
        "study_freeze_sha256": fresh.sha256(information_freeze), "protocol_sha256": "synthetic-information-protocol",
    })
    start = time.time() - 3600
    budget = put(root / "budget.json", {"started_unix": start, "deadline_unix": start + 43200,
                                       "maximum_wall_seconds": 43200})
    approval = {"source": "user_message", "approval_text": "Synthetic authorization for gate tests only"}
    plan = {"schema": "fresh-catalog-protocol/v1", "status": "FROZEN_FRESH_CATALOG",
            "simulation": fresh.SIMULATION, "authorization": approval,
            "source_files": [ref(fresh.REPO / p, fresh.REPO, "repo") for p in sorted(fresh.REQUIRED_SOURCES)],
            "seed_inventory": ref(ipath, root), "model_union": ref(upath, root),
            "analysis_plan": ref(analysis, root), "budget_record": ref(budget, root),
            "source_validation_analysis": ref(source_analysis, root),
            "generation_forecast_seconds": 19000, "reserved_scoring_analysis_seconds": 7200,
            "expected_model_ids": [m["model_id"] for m in models],
            "engineering_smoke": {"run_id": "smoke", "beam_seed": 102, "transport_master_seed": 103},
            "catalogs": [{"block_id": "C", "run_id": "fresh-C", "events": 500000,
                          "beam_seed": 104, "transport_master_seed": 105},
                         {"block_id": "D", "run_id": "fresh-D", "events": 500000,
                          "beam_seed": 106, "transport_master_seed": 107}]}
    for block in plan["catalogs"]:
        block["launched_events_by_chunk"] = fresh.planned_stratum_counts(block["events"], block["beam_seed"])
    return root, roots, plan, union, inventory


def freeze_files(tmp_path, plan):
    pp = put(tmp_path / "protocol.json", plan)
    env = {"status": "PRODUCTION_FROZEN", "confirmation_enabled": True,
           "confirmation_protocol_path": str(pp), "confirmation_protocol_sha256": fresh.sha256(pp),
           "frozen_protocol_sha256": fresh.sha256(pp), "human_freeze_approval": {"synthetic": True},
           "confirmation_approval": plan["authorization"]}
    ep = put(tmp_path / "environment.json", env)
    return pp, ep


def test_complete_union_and_existing_guard_pass_without_generating_data(evidence, tmp_path):
    _, roots, plan, _, _ = evidence
    pp, ep = freeze_files(tmp_path, plan)
    before = set(tmp_path.iterdir())
    check = fresh.validate_freeze(pp, ep, roots)
    assert check["status"] == "PASS_FRESH_FREEZE_GATE"
    assert check["union"]["unique_models"] == 240
    assert check["union"]["artifact_files_checked"] == 720
    assert check["legacy_guard"]["allowed"] is True
    assert set(tmp_path.iterdir()) == before


def test_draft_cannot_create_formal_output_or_call_guard(evidence, tmp_path, monkeypatch):
    _, _, base, _, _ = evidence
    plan = deepcopy(base)
    plan["status"] = "DRAFT_NOT_AUTHORIZED_TO_GENERATE"
    pp, ep = freeze_files(tmp_path, plan)
    out = tmp_path / "must_not_exist"
    monkeypatch.setattr(sys, "argv", ["fresh", "--plan", str(pp), "--environment", str(ep),
                                     "--geant4-data-dir", str(tmp_path), "--output-root", str(out)])
    monkeypatch.setattr(fresh.subprocess, "run", lambda *a, **k: pytest.fail("draft called subprocess"))
    with pytest.raises(ValueError, match="draft"):
        fresh.main()
    assert not out.exists()


def test_old_guard_still_refuses_disabled_environment(evidence, tmp_path):
    _, roots, plan, _, _ = evidence
    pp, ep = freeze_files(tmp_path, plan)
    env = fresh.read_json(ep)
    env["confirmation_enabled"] = False
    put(ep, env)
    with pytest.raises(ValueError, match="guard refuses"):
        fresh.validate_freeze(pp, ep, roots)


def test_old_environment_cannot_authorize_new_protocol(evidence, tmp_path):
    _, roots, plan, _, _ = evidence
    pp, ep = freeze_files(tmp_path, plan)
    env = fresh.read_json(ep)
    env["frozen_protocol_sha256"] = "0" * 64
    put(ep, env)
    with pytest.raises(ValueError, match="this new frozen protocol"):
        fresh.validate_freeze(pp, ep, roots)


@pytest.mark.parametrize("change", ["missing", "duplicate_id", "cohort", "reused_checkpoint"])
def test_full_unique_union_is_mandatory(evidence, change):
    _, roots, plan, base, _ = evidence
    union = deepcopy(base)
    if change == "missing":
        union["models"].pop()
    elif change == "duplicate_id":
        union["models"][1]["model_id"] = union["models"][0]["model_id"]
    elif change == "cohort":
        union["models"][0]["cohort"] = "legacy160"
    else:
        union["models"][1]["artifacts"]["checkpoint"] = union["models"][0]["artifacts"]["checkpoint"]
    with pytest.raises(ValueError):
        fresh.check_model_union(union, plan["expected_model_ids"], roots)


def test_artifact_tamper_is_rejected(evidence):
    _, roots, plan, base, _ = evidence
    union = deepcopy(base)
    union["models"][0]["artifacts"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/size"):
        fresh.check_model_union(union, plan["expected_model_ids"], roots)


def test_relative_reference_is_portable_and_cannot_escape(tmp_path):
    root = tmp_path / "selected_root"
    root.mkdir()
    p = root / "weights"
    p.write_bytes(b"frozen")
    record = ref(p, root)
    assert fresh.verify_file(record, {"audit": root}) == p
    outside = tmp_path / "outside"
    outside.write_bytes(b"frozen")
    record["path"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        fresh.verify_file(record, {"audit": root})


@pytest.mark.parametrize("change", ["legacy_beam", "legacy_transport", "smoke", "other_block", "legacy_uid"])
def test_formal_seeds_and_uids_are_disjoint(evidence, change):
    _, _, base, _, inventory = evidence
    plan = deepcopy(base)
    row = plan["catalogs"][0]
    if change == "legacy_beam":
        row["beam_seed"] = 100
    elif change == "legacy_transport":
        row["transport_master_seed"] = 101
    elif change == "smoke":
        row["transport_master_seed"] = 103
    elif change == "other_block":
        plan["catalogs"][1]["beam_seed"] = row["beam_seed"]
    else:
        row["run_id"] = "old"
    for block in plan["catalogs"]:
        block["launched_events_by_chunk"] = fresh.planned_stratum_counts(block["events"], block["beam_seed"])
    with pytest.raises(ValueError, match="reuses"):
        fresh.check_seeds(plan, inventory)


@pytest.mark.parametrize("change", ["simulation", "missing_source"])
def test_contract_and_all_transitive_sources_are_frozen(evidence, tmp_path, change):
    _, roots, base, _, _ = evidence
    plan = deepcopy(base)
    if change == "simulation":
        plan["simulation"]["production_cut_mm"] = 0.1
    else:
        plan["source_files"] = plan["source_files"][:-1]
    pp, ep = freeze_files(tmp_path, plan)
    with pytest.raises(ValueError):
        fresh.validate_freeze(pp, ep, roots)


def test_deadline_is_not_reset_by_fresh_generation(evidence, tmp_path):
    _, roots, base, _, _ = evidence
    plan = deepcopy(base)
    expired = put(tmp_path / "expired.json", {"started_unix": 0, "deadline_unix": 43200, "maximum_wall_seconds": 43200})
    roots = {**roots, "budget": tmp_path}
    plan["budget_record"] = ref(expired, tmp_path, "budget")
    pp, ep = freeze_files(tmp_path, plan)
    with pytest.raises(TimeoutError, match="original global"):
        fresh.validate_freeze(pp, ep, roots)


def count_fixture(tmp_path, aborted=False):
    catalog = {"run_id": "synthetic", "cells": 42, "chunks": []}
    rows = []
    for chunk_id, (momentum_bin, pdg, thickness) in enumerate(fresh.CELLS):
        common = {"momentum_bin": momentum_bin, "pdg": pdg, "thickness_mm": thickness}
        catalog["chunks"].append({**common, "chunk_id": chunk_id, "events": 2})
        rows.extend([{**common, "S1": True, "aborted": False, "status": "exited"},
                     {**common, "S1": False, "aborted": aborted, "status": "decayed"}])
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return catalog, path


def test_s0_counts_are_retained_as_physical_outcomes(tmp_path):
    result = fresh.stratum_counts(*count_fixture(tmp_path))
    assert (result["launched"], result["eligible"], result["ineligible"]) == (84, 42, 42)
    assert result["technical_aborted"] == 0
    assert result["cells"][0]["statuses"]["decayed"] == 1
    assert sorted(r["group_id"] for r in result["cells"]) == list(range(42))
    assert result["cells"][0]["group_id"] == 1


def test_technical_abortion_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="technical event abortion"):
        fresh.stratum_counts(*count_fixture(tmp_path, aborted=True))


def test_fixed_launch_counts_are_balanced_and_must_match(evidence):
    _, _, base, _, inventory = evidence
    plan = deepcopy(base)
    counts = plan["catalogs"][0]["launched_events_by_chunk"]
    assert len(counts) == 42 and sum(counts) == 500000
    assert sorted(set(counts)) == [11904, 11905]
    counts[0] += 1
    with pytest.raises(ValueError, match="launch-stratum counts"):
        fresh.check_seeds(plan, inventory)


def test_correctly_hashed_analysis_draft_is_still_refused(evidence, tmp_path):
    _, roots, base, _, _ = evidence
    plan = deepcopy(base)
    draft = put(tmp_path / "draft_analysis.json", {"schema": "fresh-catalog-statistical-analysis/v1", "status": "DRAFT"})
    roots = {**roots, "analysis": tmp_path}
    plan["analysis_plan"] = ref(draft, tmp_path, "analysis")
    pp, ep = freeze_files(tmp_path, plan)
    with pytest.raises(ValueError, match="drafts are refused"):
        fresh.validate_freeze(pp, ep, roots)


def test_expired_deadline_does_not_create_stage_log(tmp_path):
    log = tmp_path / "not_created.log"
    with pytest.raises(TimeoutError, match="deadline"):
        fresh.run_stage(["-c", "raise AssertionError('should never execute')"], log, 0)
    assert not log.exists()


def test_failed_generation_child_retains_log_and_refuses_acceptance(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="retained partial evidence"):
        fresh.run_stage(["-c", "print('synthetic failure'); raise SystemExit(3)"], log, time.time() + 30)
    assert "synthetic failure" in log.read_text()
    with pytest.raises(FileExistsError):
        fresh.run_stage(["-c", "pass"], log, time.time() + 30)


@pytest.mark.parametrize("change", ["status", "count", "replay", "freeze", "protocol"])
def test_same_complete80_independent_validation_is_required(evidence, tmp_path, change):
    _, roots, base, _, _ = evidence
    plan = deepcopy(base)
    record = fresh.read_json(fresh.verify_file(plan["source_validation_analysis"], roots))
    if change == "status":
        record["status"] = "RUNNING"
    elif change == "count":
        record["model_count"] = 79
    elif change == "replay":
        record["all_validation_predictions_replayed"] = False
    elif change == "freeze":
        record["study_freeze_sha256"] = "0" * 64
    else:
        record["protocol_sha256"] = "different-protocol"
    source_analysis = put(tmp_path / "altered_analysis.json", record)
    roots = {**roots, "source_analysis": tmp_path}
    plan["source_validation_analysis"] = ref(source_analysis, tmp_path, "source_analysis")
    pp, ep = freeze_files(tmp_path, plan)
    with pytest.raises(ValueError, match="independently replayed source-validation"):
        fresh.validate_freeze(pp, ep, roots)
