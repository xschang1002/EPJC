"""Synthetic output-only audit, including coherent cross-file tampering."""
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics

import pytest

from representation_flow import score_correction_confirmation as scorer
from representation_flow import verify_correction_confirmation as verifier


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pooled(groups):
    count = sum(r["events"] for r in groups)
    mean = sum(r["events"] * r["mean"] for r in groups) / count
    ss = sum((r["events"] - 1) * r["event_sd_descriptive"] ** 2 + r["events"] * (r["mean"] - mean) ** 2 for r in groups)
    return {"events": count, "mean": mean, "event_sd_descriptive": math.sqrt(ss / (count - 1))}


def _seed_summary(values):
    return {"seeds": 5, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "minimum": min(values),
            "maximum": max(values), "positive_seeds": sum(v > 0 for v in values), "negative_seeds": sum(v < 0 for v in values), "zero_seeds": sum(v == 0 for v in values)}


def _sync_tables(root, summary):
    _save(root / "confirmation_summary.json", summary)
    for table in verifier.TABLE_COUNTS:
        with (root / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
            # Deliberately differs from alphabetically sorted JSON key order.
            columns = list(reversed(list(summary[table][0])))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(summary[table])
    for block, seed in itertools.product(("a", "b"), range(3101, 3106)):
        name = f"{block}__unique32__seed{seed}"
        job = {"job_id": name}
        for destination, source in (("models", "models"), ("contrasts", "seed_contrasts"), ("model_groups", "model_groups"), ("contrast_groups", "contrast_groups")):
            job[destination] = [r for r in summary[source] if r["block"] == block and r["seed"] == seed]
        _save(root / "paired_groups" / f"{name}.json", job)


@pytest.fixture
def complete_saved_outputs(tmp_path):
    root = tmp_path / "confirmation"; root.mkdir()
    repository = tmp_path / "source"
    source_hashes = {}
    for key, name in (("scoring_source_sha256", "score_correction_confirmation.py"),
                      ("analysis_validation_source_sha256", "analyze_correction_study.py"),
                      ("extension_validation_source_sha256", "analyze_corrected_formal_matrix.py")):
        path = repository / "src/representation_flow" / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# synthetic frozen source " + name)
        source_hashes[key] = verifier._digest(path)
    keys = set(itertools.product(("f010", "f030", "f100"), ("Small", "Large"), ("Full",), ("Raw", "Simple", "Learned", "CorrectedPhysics")))
    keys.update(itertools.product(("f010", "f100"), ("Small",), ("Hidden",), ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")))
    keys.update((tag, "Small", "Full", "HistoricalIncorrectPhysics") for tag in ("f010", "f100"))
    models = []; lookup = {}
    for seed, (tag, capacity, context, representation) in itertools.product(range(3101, 3106), sorted(keys)):
        core = capacity == "Small" and tag in ("f010", "f100") and representation in ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")
        name = (f"correction__{context}__{representation}__{capacity}__{tag}__sub4101__seed{seed}" if core else
                f"corrected_full__{representation}__{capacity}__{tag}__sub4101__seed{seed}")
        study_name = "correction_v2_20261107" if core else "corrected_formal_extension_20261107"
        weight = repository / "reports" / study_name / "study/runs" / name / "checkpoint.pt"
        weight.parent.mkdir(parents=True); weight.write_bytes(("synthetic non-model hash fixture " + name).encode())
        row = {"run_id": name, "context": context, "capacity": capacity, "fraction_tag": tag, "seed": seed, "representation": representation,
               "checkpoint": str(weight), "checkpoint_sha256": verifier._digest(weight), "checkpoint_contract_sha256": "c" * 64}
        models.append(row); lookup[(seed, tag, capacity, context, representation)] = row
    mapping = []
    for seed, tag, capacity, rep in itertools.product(range(3101, 3106), ("f010", "f030", "f100"), ("Small", "Large"), ("Raw", "Simple", "Learned", "CorrectedPhysics")):
        model = lookup[(seed, tag, capacity, "Full", rep)]
        mapping.append({"matrix_cell_id": f"corrected_full__{rep}__{capacity}__{tag}__sub4101__seed{seed}", "run_id": model["run_id"],
                        "reused": model["run_id"].startswith("correction__"), "checkpoint_sha256": model["checkpoint_sha256"]})
    specs = scorer.full160_contrast_specs()
    plan = {"schema": "frozen-core60-full120-confirmation/v2", "status": "ALL_MODELS_VALIDATED_BEFORE_CONFIRMATION_ACCESS", "scope": verifier.SCOPE,
            "model_count": 160, "models": models, "formal120_to_unique_models": mapping, "all160_contrast_definition_sha256": scorer.canonical_hash(specs),
            "core_experiment_identity": "d" * 64, "extension_gate": {"status": "COMPLETE_FROZEN_EXTENSION120_VERIFIED", "scored_here": True, "experiment_identity": "e" * 64},
            "workers": 4, "torch_threads_per_worker": 1, "max_seconds": 1200, "preflight_finished_unix": 1200., "scoring_deadline": 2000.,
            "absolute_total_study_deadline": 10000., "core_execution_started_unix": 1000., "total_budget_seconds": 10800,
            "blocks": scorer.BLOCKS, **source_hashes}
    _save(root / "SCORING_PLAN.json", plan)
    state = {"status": "COMPLETE", "scope": verifier.SCOPE, "failures": [], "elapsed_seconds": 60.,
             "scoring_plan_sha256": verifier._digest(root / "SCORING_PLAN.json"),
             "completed_jobs": [f"{block}__unique32__seed{seed}" for block, seed in itertools.product(("a", "b"), range(3101, 3106))]}
    _save(root / "execution_state.json", state)
    summary = {"schema": "frozen-core60-full120-confirmation-results/v2", "status": "COMPLETE_FROZEN_UNIQUE160_AB_CONFIRMATION", "scope": verifier.SCOPE,
               "model_evaluations": 320, "elapsed_seconds": 60., "scoring_plan_sha256": state["scoring_plan_sha256"], "blocks": {},
               **{key: plan[key] for key in ("core_experiment_identity", "extension_gate", "formal120_to_unique_models")},
               **{table: [] for table in verifier.TABLE_COUNTS}}
    for block, expected in scorer.BLOCKS.items():
        summary["blocks"][block] = {"block": block, "eligible_events": expected["eligible_events"], "all_eligible_events_scored": True,
            "no_selection_or_resampling": True, "ordered_uid_sha256": "f" * 64,
            "source_identity": {key: expected[key] for key in ("manifest_sha256", "events_sha256")}}
        quotient, remainder = divmod(expected["eligible_events"], 42)
        for seed in range(3101, 3106):
            local_groups = {}; local_noise = {}
            for tag, capacity, context, rep in sorted(keys):
                model = lookup[(seed, tag, capacity, context, rep)]
                rank = ("Raw", "Simple", "Learned", "CorrectedPhysics", "HistoricalIncorrectPhysics").index(rep)
                # Common underlying noise makes all means/variances jointly
                # feasible while retaining positive and negative contrasts.
                base = -9. + (context == "Hidden") * .02 + (capacity == "Large") * .001 + int(tag[1:]) * .000001
                base += rank * .0004 * (seed - 3103)
                slope = .001 * (rank + 1) + .0001 * (capacity == "Large")
                noise = .4 + .001 * rank + .0001 * (context == "Hidden")
                labels = {"block": block, "seed": seed, "run_id": model["run_id"], "fraction_tag": tag, "capacity": capacity, "context": context, "representation": rep}
                groups = [{**labels, "group_id": g, "events": quotient + (g < remainder), "mean": base + slope * (g - 20.5) / 42,
                           "event_sd_descriptive": noise} for g in range(42)]
                summary["model_groups"].extend(groups)
                summary["models"].append({**labels, **_pooled(groups), "mean_log_scale": -8., "mean_log_F_b": -.01})
                local_groups[(tag, capacity, context, rep)] = groups
                local_noise[(tag, capacity, context, rep)] = noise
            for spec in specs:
                labels = {"block": block, "seed": seed, **{key: value for key, value in spec.items() if key != "terms"}}
                groups = []
                for g in range(42):
                    mean = sum(-weight * local_groups[key][g]["mean"] for weight, key in spec["terms"])
                    sd = abs(sum(-weight * local_noise[key] for weight, key in spec["terms"]))
                    groups.append({**labels, "group_id": g, "events": quotient + (g < remainder), "mean": mean, "event_sd_descriptive": sd})
                summary["contrast_groups"].extend(groups)
                summary["seed_contrasts"].append({**labels, **_pooled(groups)})
    for block, spec in itertools.product(("a", "b"), specs):
        labels = {"block": block, **{key: value for key, value in spec.items() if key != "terms"}}
        values = [r["mean"] for r in summary["seed_contrasts"] if all(r[k] == v for k, v in labels.items())]
        summary["contrast_summary"].append({**labels, **_seed_summary(values)})
    _sync_tables(root, summary)
    return root, repository, summary


def test_independent_verifier_accepts_all_outputs_without_reading_events_or_mutating_inputs(complete_saved_outputs, tmp_path, monkeypatch):
    root, repository, summary = complete_saved_outputs
    before = {str(path.relative_to(root)): verifier._digest(path) for path in root.rglob("*") if path.is_file()}
    original_open = Path.open
    def forbid_events(path, *args, **kwargs):
        assert "confirmation_cache" not in path.parts and "events.jsonl" not in path.parts
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", forbid_events)
    result = verifier.verify(root, tmp_path / "verification", repository_root=repository)
    assert result["status"] == "VERIFIED_FROZEN_UNIQUE160_AB_CONFIRMATION_OUTPUTS"
    assert result["verified_counts"] == {"models": 320, "seed_contrasts": 410, "contrast_summary": 82, "model_groups": 13440, "contrast_groups": 17220}
    assert len(result["input_files"]) == 181
    assert before == {str(path.relative_to(root)): verifier._digest(path) for path in root.rglob("*") if path.is_file()}
    assert (tmp_path / "verification/SCIENTIFIC_VERIFICATION.txt").is_file()


def test_consistent_tampering_of_contrast_json_csv_job_and_summary_is_still_rejected(complete_saved_outputs, tmp_path):
    root, repository, summary = complete_saved_outputs
    target = summary["seed_contrasts"][0]
    labels = {k: target[k] for k in ("block", "seed", "scope", "fraction_tag", "capacity", "context", "contrast")}
    target["mean"] += .25
    for row in summary["contrast_groups"]:
        if all(row[k] == v for k, v in labels.items()):
            row["mean"] += .25
    summary_labels = {k: v for k, v in labels.items() if k != "seed"}
    values = [r["mean"] for r in summary["seed_contrasts"] if all(r[k] == v for k, v in summary_labels.items())]
    next(r for r in summary["contrast_summary"] if all(r[k] == v for k, v in summary_labels.items())).update(_seed_summary(values))
    _sync_tables(root, summary)
    with pytest.raises(ValueError, match="paired NLL contrast"):
        verifier.verify(root, tmp_path / "rejected", repository_root=repository)
    assert not (tmp_path / "rejected").exists()


def test_changed_checkpoint_is_rejected_before_any_pass_report(complete_saved_outputs, tmp_path):
    root, repository, _ = complete_saved_outputs
    plan = json.loads((root / "SCORING_PLAN.json").read_text())
    Path(plan["models"][0]["checkpoint"]).write_bytes(b"covertly replaced weight")
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        verifier.verify(root, tmp_path / "rejected", repository_root=repository)
    assert not (tmp_path / "rejected").exists()


def test_relocated_default_roots_ignore_unreachable_windows_provenance_paths(complete_saved_outputs, tmp_path):
    root, repository, summary = complete_saved_outputs
    relocated = tmp_path / "portable_release"
    shutil.copytree(repository, relocated)
    plan = json.loads((root / "SCORING_PLAN.json").read_text())
    for model in plan["models"]:
        model["checkpoint"] = "Z:\\UNAVAILABLE_SYNTHETIC_ORIGIN\\study\\runs\\" + model["run_id"] + "\\checkpoint.pt"
        assert not Path(model["checkpoint"]).exists()
    _save(root / "SCORING_PLAN.json", plan)
    state = json.loads((root / "execution_state.json").read_text())
    state["scoring_plan_sha256"] = summary["scoring_plan_sha256"] = verifier._digest(root / "SCORING_PLAN.json")
    _save(root / "execution_state.json", state)
    _sync_tables(root, summary)
    result = verifier.verify(root, tmp_path / "portable_verification", repository_root=relocated)
    weights = [r for r in result["input_files"] if r["role"].startswith("checkpoint:")]
    assert len(weights) == 160 and all(Path(r["path"]).is_relative_to(relocated) for r in weights)
    assert all(r["recorded_checkpoint_path"].startswith("Z:\\UNAVAILABLE_SYNTHETIC_ORIGIN") for r in weights)
