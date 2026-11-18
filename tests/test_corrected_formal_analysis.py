"""Synthetic frozen 120-cell analysis; no fitting or confirmation input."""
import copy
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from test_correction_analysis import complete_matrix, _changed_json, _write
from representation_flow.analyze_corrected_formal_matrix import (
    PRIMARY_NAMES, analyze, check_new_cell, prespecified_contrasts, validate_matrix,
)
from representation_flow.checkpoint_contract import build_checkpoint_contract
from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, context_features
from representation_flow.experiment_contract import artifact_record, canonical_hash, sha256
from representation_flow.run_a0_prototype import _state_hash
from representation_flow.run_corrected_formal_extension import cell_id, prepare_extension_cache, reused_cell


@pytest.fixture(scope="module")
def full_matrix(complete_matrix, tmp_path_factory):
    core, core_study, core_cache, core_protocol, core_protocol_path = complete_matrix
    root = tmp_path_factory.mktemp("SYNTHETIC_full120")
    study = root / "study"; study.mkdir()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / "protocol/corrected_formal_extension_20261107.json").read_text())
    protocol["main_data"] = core_protocol["main_data"]
    protocol["core_protocol_sha256"] = canonical_hash(core_protocol)
    protocol_path = root / "synthetic_protocol.json"; _write(protocol_path, protocol)
    base_record = json.loads((core_cache / "cache_manifest.json").read_text())
    prep = prepare_extension_cache(core_cache, root / "cache", protocol, base_record)
    core_identity = json.loads((core_study / "experiment_identity.json").read_text())["identity"]
    definition = {"core_common": {key: value for key, value in core_identity.items() if key not in ("mode", "max_updates")},
                  "protocol_sha256": canonical_hash(protocol), "extension_source_sha256": sha256(repo / "src/representation_flow/run_corrected_formal_extension.py"),
                  "preprocessing_sha256": canonical_hash(prep), "mode": "execute", "max_updates": 20000}
    identity = canonical_hash(definition)
    _write(study / "experiment_identity.json", {"identity": definition, "sha256": identity})
    arrays = {key: np.load(core_cache / f"{key}.npy", allow_pickle=False) for key in ("event_uid", "p_in", "thickness", "charge", "y", "split")}
    indices = {f"indices_{tag}": np.load(root / "cache" / f"indices_{tag}.npy") for tag in ("f010", "f030", "f100", "validation")}
    valid = indices["indices_validation"]; p,t,q,y = (arrays[k][valid] for k in ("p_in", "thickness", "charge", "y"))
    rows = []
    for rep, cap, fraction, seed in itertools.product(protocol["representations"], protocol["capacities"], protocol["sample_fractions"], protocol["seeds"]):
        if reused_cell(rep, cap, fraction):
            continue
        tag = f"f{round(fraction * 100):03d}"; rec = prep["fractions"][tag]
        name = cell_id(rep, cap, fraction, seed, 4101)
        run = study / "runs" / name; run.mkdir(parents=True)
        pre = RepresentationPreprocessor.from_dict(rec["preprocessors"][rep])
        cfg = {"context_dim": 5, "hidden": protocol["capacities"][cap], "seed": seed, "bound": 8.}
        flow = ConditionalSplineFlow(**cfg); initial = _state_hash(flow.state_dict())
        contract = build_checkpoint_contract(pre, "full")
        with torch.no_grad():
            terms = flow.original_terms(torch.tensor(y,dtype=torch.float32), torch.tensor(pre.center(p,t,q),dtype=torch.float32),pre.mu,
                    torch.tensor(pre.scale,dtype=torch.float32),torch.tensor(p,dtype=torch.float32),torch.tensor(context_features(p,t,q,"full"),dtype=torch.float32))
        logp = terms["log_p_y"].numpy(); selected_nll = float(np.mean(-logp))
        checkpoint = run / "checkpoint.pt"
        torch.save({"state_dict": flow.state_dict(), "representation": rep, "seed": seed, "selected_update": 1, "initialization_hash": initial,
                    "training_state": {"update": 1}, "checkpoint_contract": contract, "experiment_identity": identity,
                    "model_config": cfg, "synthetic_untrained_fixture": True}, checkpoint)
        prediction = run / "predictions.npz"
        np.savez(prediction,event_uid=arrays["event_uid"][valid],y=y,p_in=p,c=pre.center(p,t,q),mu=np.full(len(valid),pre.mu),
                 s=np.full(len(valid),pre.scale,dtype=np.float32),representation=np.full(len(valid),rep),capacity=np.full(len(valid),cap),
                 seed=np.full(len(valid),seed),subset_id=np.full(len(valid),4101),split=np.full(len(valid),"validation"),
                 log_p_y=logp,log_q_z=terms["log_q_z"].numpy(),log_F_b=terms["log_f_b"].numpy())
        row = {"run_id": name, "matrix_cell_id": name, "reused": False, "representation": rep, "capacity": cap, "seed": seed,
               "selected_update": 1, "initialization_hash": initial, "preprocessor": pre.to_dict(), "checkpoint_contract": contract,
               "train_events": rec["train_events"], "validation_events": len(valid), "max_updates": 20000,
               "batch_size": 512, "validation_every": 500, "eval_batch_size": 8192, "learning_rate": .0003,
               "context_feature_mode": "full", "context_label": "Full", "fraction_tag": tag, "sample_fraction": fraction,
               "experiment_identity": identity, "validation_nll": float(np.mean(-logp.astype(np.float64))),
               "train_event_uid_sha256": rec["ordered_uid_sha256"], "validation_event_uid_sha256": prep["validation_uid_sha256"],
               "history": [{"update":u,"validation_nll":selected_nll if u==1 else selected_nll+1.} for u in [1,*range(500,20001,500)]],
               "synthetic_untrained_fixture": True}
        manifest = run / "run_manifest.json"; _write(manifest, row)
        row["artifacts"] = {"checkpoint":artifact_record(checkpoint,study),"predictions":artifact_record(prediction,study),"run_manifest":artifact_record(manifest,study)}
        rows.append(row)
    _write(study / "scores.json", {"status":"COMPLETE","experiment_identity":identity,"runs":rows,"failures":[],"synthetic_untrained_fixture":True})
    reused = []
    for row in json.loads((core_study / "scores.json").read_text())["runs"]:
        if row["context_label"] == "Full" and reused_cell(row["representation"],row["capacity"],row["sample_fraction"]):
            reused.append({**row,"reused":True,"artifact_root":str(core_study.resolve()),
                           "matrix_cell_id":cell_id(row["representation"],row["capacity"],row["sample_fraction"],row["seed"],4101)})
    combined = reused + [{**row,"artifact_root":str(study.resolve())} for row in rows]
    _write(study / "formal_scores_120.json", {"status":"COMPLETE_CORRECTED_FORMAL_120","experiment_identity":identity,"new_cells":100,"reused_cells":20,"runs":combined})
    keys = ("matrix_cell_id","run_id","representation","capacity","sample_fraction","seed","experiment_identity","reused","artifact_root","artifacts")
    _write(study / "FROZEN_CHECKPOINT_MANIFEST.json", {"schema":"corrected-formal-checkpoint-freeze-v1","experiment_identity":identity,
           "frozen_before_confirmation":True,"models":[{key:row[key] for key in keys} for row in combined]})
    return {"core":core,"root":root,"core_protocol_path":core_protocol_path,"protocol_path":protocol_path,
            "protocol":protocol,"arrays":arrays,"indices":indices,"preprocessing":prep,"identity":identity,"new_rows":rows}


def test_synthetic_complete120_replays_and_reports_every_prespecified_contrast(full_matrix):
    f = full_matrix
    result = analyze(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"],f["root"] / "analysis")
    assert result["analysis_status"] == "VERIFIED_CORRECTED_FORMAL_120_DESCRIPTIVE_ANALYSIS"
    assert len(result["runs"]) == 120 and sum(row["reused"] for row in result["runs"]) == 20
    assert len(result["seed_contrasts"]) == 90 and len(result["contrast_summary"]) == 18
    assert len(result["prespecified_seed_contrasts"]) == 115 and len(result["prespecified_contrast_summary"]) == 23
    assert set(row["contrast"] for row in result["prespecified_contrast_summary"]) == set(PRIMARY_NAMES)
    assert all(row["checkpoint_replayed_validation_events"] == 14 for row in result["runs"])
    assert result["confirmation_accessed"] is False
    assert (f["root"] / "analysis/analysis_manifest.json").exists()
    assert (f["root"] / "analysis/analysis.json").read_bytes() == (f["root"] / "analysis/formal_scores_120.json").read_bytes()
    with pytest.raises(FileExistsError):
        analyze(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"],f["root"] / "analysis")


@pytest.mark.parametrize("mutation", [
    lambda s:s.update(status="RUNNING"),
    lambda s:s["runs"].pop(),
    lambda s:s["runs"].append(copy.deepcopy(s["runs"][0])),
    lambda s:s["runs"][0].update(validation_nll=999.),
    lambda s:s["runs"][0].update(artifact_root="E:/outside"),
])
def test_incomplete_duplicate_relabelled_or_changed_combined_matrix_rejected(full_matrix, mutation):
    f = full_matrix
    with _changed_json(f["root"] / "study/formal_scores_120.json", mutation):
        with pytest.raises(ValueError):
            validate_matrix(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"])


def test_changed_new_checkpoint_and_preprocessing_identity_rejected(full_matrix):
    f = full_matrix; study = f["root"] / "study"
    path = study / f["new_rows"][0]["artifacts"]["checkpoint"]["path"]
    before = path.read_bytes(); path.write_bytes(before + b"changed")
    try:
        with pytest.raises(ValueError, match="artifact"):
            validate_matrix(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"])
    finally:
        path.write_bytes(before)
    with _changed_json(f["root"] / "cache/preprocessing_manifest.json",lambda s:s["fractions"]["f030"].update(ordered_uid_sha256="changed")):
        with pytest.raises(ValueError, match="preprocessing cache"):
            validate_matrix(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"])


def test_shuffled_validation_uid_and_changed_checkpoint_prediction_rejected(full_matrix):
    f = full_matrix; row = f["new_rows"][0]; study = f["root"] / "study"
    path = study / row["artifacts"]["predictions"]["path"]
    before = path.read_bytes()
    with np.load(path) as data:
        values = {key:data[key] for key in data.files}
    values["event_uid"] = values["event_uid"][::-1]
    try:
        np.savez(path, **values)
        with pytest.raises(ValueError, match="UID"):
            check_new_cell(row,study,f["arrays"],f["indices"],f["preprocessing"],f["protocol"],f["identity"])
    finally:
        path.write_bytes(before)


def test_coherently_rewritten_predictions_still_require_actual_checkpoint_replay(full_matrix):
    f = full_matrix; row = copy.deepcopy(f["new_rows"][0]); study = f["root"] / "study"
    path = study / row["artifacts"]["predictions"]["path"]
    before = path.read_bytes()
    with np.load(path) as data:
        values = {key:data[key] for key in data.files}
    values["log_p_y"] += .2; values["log_q_z"] += .2
    row["validation_nll"] = float(np.mean(-values["log_p_y"].astype(np.float64)))
    selected = float(np.mean(-values["log_p_y"]))
    row["history"] = [{"update":item["update"],"validation_nll":selected if item["update"]==1 else selected+1} for item in row["history"]]
    try:
        np.savez(path, **values)
        with pytest.raises(ValueError, match="checkpoint does not reproduce"):
            check_new_cell(row,study,f["arrays"],f["indices"],f["preprocessing"],f["protocol"],f["identity"])
    finally:
        path.write_bytes(before)


def test_frozen_source_paths_are_provenance_and_cli_roots_enable_relocation(full_matrix):
    f = full_matrix; study = f["root"] / "study"
    def relocate(record, key):
        for row in record[key]:
            row["artifact_root"] = "Z:/unavailable-source/core/study" if row["reused"] else "Z:/unavailable-source/extension/study"
    with _changed_json(study / "formal_scores_120.json", lambda s:relocate(s,"runs")):
        with _changed_json(study / "FROZEN_CHECKPOINT_MANIFEST.json", lambda s:relocate(s,"models")):
            checked = validate_matrix(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"])
            assert len(checked["runs"]) == 120


def test_private_external_caches_are_validated_without_reading_released_cache_arrays(full_matrix, tmp_path):
    f = full_matrix
    core_cache = tmp_path / "private_main_cache"
    extension_cache = tmp_path / "private_extension_cache"
    shutil.copytree(f["core"] / "cache", core_cache)
    shutil.copytree(f["root"] / "cache", extension_cache)
    # Broken default manifests ensure the explicit private paths are actually
    # used; neither the original path nor its arrays can silently satisfy this.
    with _changed_json(f["core"] / "cache/cache_manifest.json", lambda r:r.update(data={})):
        with _changed_json(f["root"] / "cache/preprocessing_manifest.json", lambda r:r.update(identity={})):
            checked = validate_matrix(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"],core_cache=core_cache,extension_cache=extension_cache)
            assert len(checked["runs"]) == 120
    with _changed_json(extension_cache / "preprocessing_manifest.json", lambda r:r["fractions"]["f030"].update(ordered_uid_sha256="changed")):
        with pytest.raises(ValueError, match="preprocessing cache identity"):
            validate_matrix(f["core"],f["root"],f["core_protocol_path"],f["protocol_path"],core_cache=core_cache,extension_cache=extension_cache)


def test_interactions_use_paired_seed_signs_and_retain_negatives():
    protocol = {"capacities":{"Small":32,"Large":128},"sample_fractions":[.1,.3,1.],"seeds":[3101,3102,3103,3104,3105],"primary_contrasts":PRIMARY_NAMES}
    predictions = {}; values = [-2.,-1.,0.,1.,2.]
    for cap, fraction, seed in itertools.product(protocol["capacities"],protocol["sample_fractions"],protocol["seeds"]):
        tag = f"f{round(fraction*100):03d}"
        gain = values[seed-3101] * (2 if cap=="Small" else 1) * (3 if fraction==.1 else 1)
        for rep, value in (("Raw",0.),("Simple",-1.),("Learned",1.),("CorrectedPhysics",gain)):
            predictions[(cap,tag,seed,rep)] = np.full(3,value)
    rows, summaries = prespecified_contrasts(predictions,protocol)
    cap = [row for row in rows if row["contrast"]==PRIMARY_NAMES[3] and row["fraction_tag"]=="f010"]
    fraction = [row for row in rows if row["contrast"]==PRIMARY_NAMES[4] and row["capacity"]=="Small"]
    assert [r["nll_difference"] for r in cap] == [3*x for x in values]
    assert [r["nll_difference"] for r in fraction] == [4*x for x in values]
    summary = next(r for r in summaries if r["contrast"]==PRIMARY_NAMES[3] and r["fraction_tag"]=="f010")
    assert summary["negative_seeds"]==2 and summary["positive_seeds"]==2 and summary["zero_seeds"]==1
    assert summary["sample_sd"]==pytest.approx(np.std(np.array(values)*3,ddof=1))
