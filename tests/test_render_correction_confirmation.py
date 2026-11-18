"""Synthetic gate/arithmetic tests; never open A/B or render scientific figures."""
import itertools
import json
import math
from pathlib import Path
import statistics

import pytest

from representation_flow import render_correction_confirmation as renderer
from representation_flow.verify_correction_confirmation import _run_id


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+"\n", encoding="utf-8")


def write_analysis(path, value):
    write_json(path, value)
    write_json(path.with_name("analysis_manifest.json"), {
        "outputs": [{"name": path.name, "sha256": renderer.digest(path)}],
        "original_root": "Z:/unavailable/original-machine",
    })


def synthetic_nll(seed, tag, cap, context, rep, block="validation"):
    base = -6. + .02*(context == "Hidden") + .001*(cap == "Large")
    base += .0001*int(tag[1:]) + .0002*(seed-3101)
    gain = (-.003, .001, .002, .003, .004)[seed-3101]
    gain += .002*(context == "Hidden") + .0002*(cap == "Large")
    gain += {"validation": 0., "a": .00013, "b": -.00017}[block]
    return base + {"Raw": 0., "CorrectedPhysics": -gain, "Simple": .0003,
                   "Learned": -.0008, "HistoricalIncorrectPhysics": .08}[rep]


@pytest.fixture
def inputs(tmp_path):
    core_path = tmp_path/"moved_core_analysis"/"analysis.json"
    full_path = tmp_path/"moved_full_analysis"/"analysis.json"
    summary_path = tmp_path/"moved_confirmation"/"confirmation_summary.json"
    verification_path = tmp_path/"moved_verification"/"verification_manifest.json"
    core = {"status": "COMPLETE_CORRECTION_STUDY_DESCRIPTIVE_ANALYSIS",
            "experiment_identity": "synthetic-core", "failed_runs": [], "cells": []}
    for seed, tag, ctx, rep in itertools.product(renderer.SEEDS, ("f010", "f100"),
            ("Full", "Hidden"), ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")):
        core["cells"].append({"seed": seed, "fraction_tag": tag, "context": ctx,
            "representation": rep, "run_id": _run_id(seed, tag, "Small", ctx, rep),
            "maximum_updates": 20000, "validation_nll": synthetic_nll(seed, tag, "Small", ctx, rep)})
    full = {"status": "COMPLETE_CORRECTED_FORMAL_120",
            "analysis_status": "VERIFIED_CORRECTED_FORMAL_120_DESCRIPTIVE_ANALYSIS",
            "experiment_identity": "synthetic-extension", "core_experiment_identity": "synthetic-core",
            "new_cells": 100, "reused_cells": 20, "runs": []}
    for seed, tag, cap, rep in itertools.product(renderer.SEEDS, ("f010", "f030", "f100"),
            ("Small", "Large"), ("Raw", "Simple", "Learned", "CorrectedPhysics")):
        full["runs"].append({"seed": seed, "sample_fraction": int(tag[1:])/100, "capacity": cap,
            "representation": rep, "run_id": _run_id(seed, tag, cap, "Full", rep),
            "matrix_cell_id": f"cell-{len(full['runs'])}", "max_updates": 20000, "context_label": "Full",
            "reused": cap == "Small" and tag != "f030" and rep in ("Raw", "CorrectedPhysics"),
            "validation_nll": synthetic_nll(seed, tag, cap, "Full", rep)})
    summary = {"schema": "frozen-core60-full120-confirmation-results/v2",
               "status": "COMPLETE_FROZEN_UNIQUE160_AB_CONFIRMATION", "scope": renderer.SCOPE,
               "model_evaluations": 320, "scoring_plan_sha256": "synthetic-plan",
               "core_experiment_identity": "synthetic-core",
               "extension_gate": {"status": "COMPLETE_FROZEN_EXTENSION120_VERIFIED",
                    "experiment_identity": "synthetic-extension", "scored_here": True},
               "models": [], "seed_contrasts": [], "contrast_summary": [],
               # Nonplotted group tables are already covered by the bound independent
               # verification. This fixture supplies only their required cardinality.
               "model_groups": [{}]*13440, "contrast_groups": [{}]*17220}
    for block, (seed, tag, cap, ctx, rep) in itertools.product(("a", "b"), sorted(renderer._model_design())):
        summary["models"].append({"block": block, "seed": seed, "fraction_tag": tag,
            "capacity": cap, "context": ctx, "representation": rep,
            "run_id": _run_id(seed, tag, cap, ctx, rep), "events": renderer.BLOCK_EVIDENCE[block][0],
            "mean": synthetic_nll(seed, tag, cap, ctx, rep, block)})
    model_lookup = {(r["block"],r["seed"],r["fraction_tag"],r["capacity"],r["context"],r["representation"]): r
                    for r in summary["models"]}
    keys = ("scope", "fraction_tag", "capacity", "context", "contrast")
    for block, (labels, terms) in itertools.product(("a", "b"), renderer._contrast_design()):
        values = []
        for seed in renderer.SEEDS:
            value = math.fsum(weight*model_lookup[block,seed,*model_key]["mean"] for weight, model_key in terms)
            values.append(value)
            summary["seed_contrasts"].append({"block": block, "seed": seed, **dict(zip(keys, labels)),
                "events": renderer.BLOCK_EVIDENCE[block][0], "mean": value})
        summary["contrast_summary"].append({"block": block, **dict(zip(keys, labels)), "seeds": 5,
            "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "minimum": min(values), "maximum": max(values), "positive_seeds": sum(v>0 for v in values),
            "negative_seeds": sum(v<0 for v in values), "zero_seeds": sum(v==0 for v in values)})
    verification = {"schema": "correction-confirmation-output-verification/v1",
        "status": "VERIFIED_FROZEN_UNIQUE160_AB_CONFIRMATION_OUTPUTS", "scope": renderer.SCOPE,
        "unique_models": 160, "reused_formal_cells": 20, "verified_counts": renderer.TABLE_COUNTS,
        "scoring_plan_sha256": "synthetic-plan", "core_experiment_identity": "synthetic-core",
        "extension_experiment_identity": "synthetic-extension", "input_files": []}
    payload = {"core":core, "full":full, "summary":summary, "verification":verification,
               "paths":(core_path,full_path,summary_path,verification_path)}
    save(payload)
    return payload


def save(payload):
    core_path, full_path, summary_path, verification_path = payload["paths"]
    write_analysis(core_path, payload["core"])
    write_analysis(full_path, payload["full"])
    write_json(summary_path, payload["summary"])
    payload["verification"]["input_files"] = [{"role": "confirmation_summary.json",
        "path": "Z:/unavailable/original-machine/confirmation_summary.json",
        "sha256": renderer.digest(summary_path), "bytes": summary_path.stat().st_size}]
    write_json(verification_path, payload["verification"])


def test_complete_portable_inputs_preserve_all_seeds_and_negative_values(inputs):
    tables, sources = renderer.prepare_tables(*inputs["paths"])
    assert {k:len(v) for k,v in tables.items()} == {"core60":60,"formal120":90}
    assert len(sources) == 6 and all(Path(r["path"]).exists() for r in sources)
    for rows in tables.values():
        assert {r["dataset"] for r in rows} == {"Validation","A","B"}
        assert any(r["nll_difference"] < 0 for r in rows)
        lo, hi = renderer.limits(rows)
        assert lo < 0 < hi
        assert all(lo < r["nll_difference"] < hi for r in rows)
        for row in renderer.summarize(rows):
            group = [r["nll_difference"] for r in rows if all(r[k] == row[k] for k in
                     ("scope","context","capacity","fraction_tag","dataset","contrast"))]
            assert row["seeds"] == 5 and row["sample_sd"] == statistics.stdev(group)
            assert lo < row["mean"]-row["sample_sd"] <= row["mean"]+row["sample_sd"] < hi


@pytest.mark.parametrize("case,match", [
    ("partial_scope", "320 all160"), ("unverified", "verification is absent"),
    ("missing_unplotted_model", "complete confirmation table"),
    ("missing_seed", "complete confirmation table"), ("budget_core", "core budget"),
    ("budget_full", "Full120 budget"), ("reused_nll", "reused validation NLL"),
    ("nonfinite_model", "event count/NLL"), ("altered_contrast", "does not reconstruct"),
    ("wrong_experiment", "core experiment identities"), ("run_id", "model identity differs"),
    ("wrong_sd", "descriptive summary differs"),
])
def test_rejects_incomplete_or_inconsistent_evidence(inputs, case, match):
    if case == "partial_scope": inputs["summary"]["scope"] = "CORE60_ONLY"
    elif case == "unverified": inputs["verification"]["status"] = "IN_PROGRESS"
    elif case == "missing_unplotted_model":
        inputs["summary"]["models"] = [r for r in inputs["summary"]["models"]
            if not (r["block"]=="a" and r["seed"]==3101 and r["representation"]=="Learned" and r["capacity"]=="Large" and r["fraction_tag"]=="f030")]
    elif case == "missing_seed": inputs["summary"]["seed_contrasts"].pop()
    elif case == "budget_core": inputs["core"]["cells"][0]["maximum_updates"] = 1000
    elif case == "budget_full": inputs["full"]["runs"][0]["max_updates"] = 1000
    elif case == "reused_nll":
        next(r for r in inputs["full"]["runs"] if r["reused"])["validation_nll"] += .1
    elif case == "nonfinite_model": inputs["summary"]["models"][0]["mean"] = float("nan")
    elif case == "altered_contrast": inputs["summary"]["seed_contrasts"][0]["mean"] += .1
    elif case == "wrong_experiment": inputs["full"]["core_experiment_identity"] = "different"
    elif case == "run_id": inputs["summary"]["models"][0]["run_id"] = "different-model"
    elif case == "wrong_sd": inputs["summary"]["contrast_summary"][0]["sample_sd"] += .01
    save(inputs)
    with pytest.raises(ValueError, match=match): renderer.prepare_tables(*inputs["paths"])


def test_rejects_summary_changed_after_independent_verification(inputs):
    path = inputs["paths"][2]
    path.write_text(path.read_text(encoding="utf-8")+" ",encoding="utf-8")
    with pytest.raises(ValueError,match="not the independently verified file"):
        renderer.prepare_tables(*inputs["paths"])


def test_rejects_validation_analysis_changed_after_analysis_manifest(inputs):
    path = inputs["paths"][0]
    path.write_text(path.read_text(encoding="utf-8")+" ",encoding="utf-8")
    with pytest.raises(ValueError,match="does not match its manifest"):
        renderer.prepare_tables(*inputs["paths"])


def test_gate_failure_precedes_output_creation_and_matplotlib(inputs, monkeypatch):
    import sys
    inputs["summary"]["status"] = "IN_PROGRESS"
    save(inputs)
    output = inputs["paths"][0].parent.parent/"no_fake_figures"
    cp,fp,sp,vp = inputs["paths"]
    monkeypatch.setattr(sys,"argv",["render", "--core-analysis",str(cp), "--formal-analysis",str(fp),
        "--confirmation",str(sp),"--verification",str(vp),"--output",str(output)])
    with pytest.raises(ValueError,match="not complete all160"): renderer.main()
    assert not output.exists()
