"""Gate scientific figures against incomplete, benchmark or inconsistent evidence."""
from copy import deepcopy
import itertools
import json

import numpy as np
import pytest

from representation_flow.render_correction_analysis import (
    CORE_CONTRASTS, CORE_REPS, FORMAL_REPS, SEEDS, effect_limits,
    load_core_analysis, load_formal_matrix, load_location_diagnostic, prepare_output, sha256, summarize, validate_core_payload,
    validate_formal_payload,
)


def core_fixture():
    cells = []
    for context, tag, rep, seed in itertools.product(("Full", "Hidden"), ("f010", "f100"), CORE_REPS, SEEDS):
        cells.append({"context": context, "fraction_tag": tag, "representation": rep,
                      "seed": seed, "maximum_updates": 20000,
                      "validation_nll": -4 + .1 * CORE_REPS.index(rep) + .001 * seed + (context == "Hidden") * .2})
    lookup = {(r["context"], r["fraction_tag"], r["representation"], r["seed"]): r["validation_nll"] for r in cells}
    rows = []
    for context, tag, contrast, seed in itertools.product(("Full", "Hidden", "Full_minus_Hidden"), ("f010", "f100"), CORE_CONTRASTS, SEEDS):
        left, right = contrast.split("_minus_")
        def delta(label):
            return lookup[(label, tag, left, seed)] - lookup[(label, tag, right, seed)]
        rows.append({"context": context, "fraction_tag": tag, "contrast": contrast, "seed": seed,
                     "nll_difference": delta("Full") - delta("Hidden") if context == "Full_minus_Hidden" else delta(context)})
    return {"status": "COMPLETE_CORRECTION_STUDY_DESCRIPTIVE_ANALYSIS", "cells": cells, "seed_contrasts": rows, "failed_runs": []}


def formal_fixture():
    rows = []
    for rep, capacity, fraction, seed in itertools.product(FORMAL_REPS, ("Small", "Large"), (.1, .3, 1.), SEEDS):
        rows.append({"representation": rep, "capacity": capacity, "sample_fraction": fraction, "seed": seed,
                     "matrix_cell_id": str((rep, capacity, fraction, seed)), "max_updates": 20000, "validation_nll": -1.})
    return {"status": "COMPLETE_CORRECTED_FORMAL_120", "runs": rows}


def test_complete_core_retains_every_displayed_seed():
    rows = validate_core_payload(core_fixture())
    assert len(rows) == 40
    assert all(r["seeds"] == 5 for r in summarize(rows, ("context", "fraction_tag", "contrast")))


@pytest.mark.parametrize("fault", ("benchmark", "missing_cell", "duplicate_cell", "wrong_budget", "altered_contrast", "missing_seed", "nonfinite", "failed"))
def test_core_refuses_non_scientific_or_inconsistent_results(fault):
    data = deepcopy(core_fixture())
    if fault == "benchmark":
        data["status"] = "BENCHMARK_COMPLETE"
    elif fault == "missing_cell":
        data["cells"].pop()
    elif fault == "duplicate_cell":
        data["cells"][-1] = deepcopy(data["cells"][0])
    elif fault == "wrong_budget":
        data["cells"][0]["maximum_updates"] = 1000
    elif fault == "altered_contrast":
        data["seed_contrasts"][0]["nll_difference"] += .01
    elif fault == "missing_seed":
        data["seed_contrasts"].pop()
    elif fault == "nonfinite":
        data["cells"][0]["validation_nll"] = np.nan
    else:
        data["failed_runs"] = ["some failed run"]
    with pytest.raises(ValueError):
        validate_core_payload(data)


def test_analysis_manifest_catches_post_audit_change(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(core_fixture()), encoding="utf-8")
    manifest = tmp_path / "analysis_manifest.json"
    manifest.write_text(json.dumps({"outputs": [{"name": path.name, "sha256": sha256(path)}]}), encoding="utf-8")
    assert len(load_core_analysis(path)[0]) == 40
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_core_analysis(path)


@pytest.mark.parametrize("fault", ("none", "benchmark", "missing", "duplicate", "budget", "hidden"))
def test_formal_requires_full_scientific_matrix(fault):
    data = formal_fixture()
    if fault == "none":
        assert len(validate_formal_payload(data)) == 120
        return
    if fault == "benchmark":
        data["status"] = "BENCHMARK_COMPLETE"
    elif fault == "missing":
        data["runs"].pop()
    elif fault == "duplicate":
        data["runs"][-1] = deepcopy(data["runs"][0])
    elif fault == "budget":
        data["runs"][0]["max_updates"] = 1000
    else:
        data["runs"][0]["context_label"] = "Hidden"
    with pytest.raises(ValueError):
        validate_formal_payload(data)


def test_limits_include_whole_sd_bar_and_zero():
    rows = [{"group": "A", "seed": seed, "nll_difference": value} for seed, value in zip(SEEDS, (1, 1, 1, 1, 0))]
    lo, hi = effect_limits(rows, ("group",))
    summary = summarize(rows, ("group",))[0]
    assert lo < 0 and hi > summary["mean"] + summary["sample_sd"] > 1
    assert summary["sample_sd"] == pytest.approx(np.std((1, 1, 1, 1, 0), ddof=1))


def test_repeated_seed_cannot_define_uncertainty():
    rows = [{"group": "A", "seed": 3101, "nll_difference": value} for value in range(5)]
    with pytest.raises(ValueError, match="unique"):
        summarize(rows, ("group",))


def test_replacing_figures_cannot_leave_stale_empirical_evidence(tmp_path):
    old = tmp_path / "figure2_controlled_contrasts.png"
    old.write_bytes(b"old evidence")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output(tmp_path, False)
    assert old.exists()
    prepare_output(tmp_path, True)
    assert not old.exists() and unrelated.read_text(encoding="utf-8") == "keep"


def portable_formal_fixture(tmp_path):
    formal_root = tmp_path / "moved_formal" / "study"
    core_root = tmp_path / "moved_core" / "study"
    for root in (formal_root, core_root):
        root.mkdir(parents=True)
        np.savez(root / "predictions.npz", log_p_y=np.array([1., 1.]), event_uid=np.array(["a", "b"]))
    data = formal_fixture()
    data["experiment_identity"] = "fixture-experiment"
    for row in data["runs"]:
        row["reused"] = row["representation"] in ("Raw", "CorrectedPhysics") and row["capacity"] == "Small" and row["sample_fraction"] in (.1, 1.)
        row["artifact_root"] = "Z:/unavailable-original-host/" + ("core" if row["reused"] else "formal")
        row["experiment_identity"] = "fixture-experiment"
        row["run_id"] = row["matrix_cell_id"]
        pred = (core_root if row["reused"] else formal_root) / "predictions.npz"
        row["artifacts"] = {"predictions": {"path": "predictions.npz", "sha256": sha256(pred), "bytes": pred.stat().st_size}}
    path = formal_root / "formal_scores_120.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    freeze = {"experiment_identity": data["experiment_identity"], "models": deepcopy(data["runs"])}
    (formal_root / "FROZEN_CHECKPOINT_MANIFEST.json").write_text(json.dumps(freeze), encoding="utf-8")
    return path, core_root, data, freeze


def test_formal_plot_resolves_moved_artifacts_without_old_absolute_paths(tmp_path):
    path, core_root, _, _ = portable_formal_fixture(tmp_path)
    assert len(load_formal_matrix(path, core_root)[0]) == 90


def test_portable_formal_paths_still_reject_traversal(tmp_path):
    path, core_root, data, freeze = portable_formal_fixture(tmp_path)
    data["runs"][0]["artifacts"]["predictions"]["path"] = "../outside.npz"
    freeze["models"][0]["artifacts"]["predictions"]["path"] = "../outside.npz"
    path.write_text(json.dumps(data), encoding="utf-8")
    (path.parent / "FROZEN_CHECKPOINT_MANIFEST.json").write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_formal_matrix(path, core_root)


@pytest.mark.parametrize("fault", ("none", "primary", "incomplete", "pool", "experiment", "manifest"))
def test_posthoc_plot_preserves_scope_identity_and_weighting(tmp_path, fault):
    analysis = core_fixture()
    analysis["experiment_identity"] = "synthetic-core-identity"
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    (tmp_path / "analysis_manifest.json").write_text(json.dumps({"outputs": [{"name": "analysis.json", "sha256": sha256(analysis_path)}]}), encoding="utf-8")
    groups = []
    for tag, rep, thickness, charge in itertools.product(("f010", "f100"), CORE_REPS, (.1, .3, 1.), (-1., 1.)):
        groups.append({"fraction_tag": tag, "representation": rep, "thickness_mm": thickness, "charge": charge,
                       "events": 10, "populated_momentum_bins": 7, "within_W_rms_GeV_c": 1e-6 * (CORE_REPS.index(rep)+1) * (1+thickness)})
    summary = []
    for tag, rep in itertools.product(("f010", "f100"), CORE_REPS):
        summary.append({"fraction_tag": tag, "representation": rep,
                        "within_W_rms_GeV_c": float(np.sqrt(np.mean([g["within_W_rms_GeV_c"]**2 for g in groups if g["fraction_tag"]==tag and g["representation"]==rep])))})
    data = {"status": "COMPLETE_POSTHOC_WITHIN_W_LOCATION_DIAGNOSTIC", "post_hoc": True, "prespecified_primary": False,
            "source_experiment_identity": analysis["experiment_identity"], "groups": groups, "summary": summary}
    if fault == "primary": data["prespecified_primary"] = True
    if fault == "incomplete": data["groups"].pop()
    if fault == "pool": data["summary"][0]["within_W_rms_GeV_c"] *= 2
    if fault == "experiment": data["source_experiment_identity"] = "wrong-experiment"
    path = tmp_path / "within_condition_locations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest = {"input_sha256": {"Z:/unavailable-original-host/analysis/analysis.json": sha256(analysis_path)},
                "outputs": [{"name": path.name, "sha256": sha256(path)}]}
    if fault == "manifest": manifest["outputs"][0]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if fault == "none":
        checked, _ = load_location_diagnostic(path, analysis_path)
        assert len(checked["groups"]) == 36
    else:
        with pytest.raises(ValueError): load_location_diagnostic(path, analysis_path)
