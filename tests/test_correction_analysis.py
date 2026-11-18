"""Synthetic complete-matrix acceptance and fail-closed correction analysis."""
from contextlib import contextmanager
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from representation_flow.analyze_correction_study import analyze, diagnostic_locations, paired_contrasts, validate_study
from representation_flow.checkpoint_contract import build_checkpoint_contract
from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, context_features, subset_indices
from representation_flow.experiment_contract import artifact_record, canonical_hash, sha256, source_hashes
from representation_flow.run_a0_prototype import _state_hash


def _write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


@contextmanager
def _changed_json(path,change):
    before=path.read_bytes();value=json.loads(before);change(value);_write(path,value)
    try:yield
    finally:path.write_bytes(before)


@pytest.fixture(scope="module")
def complete_matrix(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp("SYNTHETIC_correction_matrix")
    study=root/"study";cache=root/"cache";study.mkdir();cache.mkdir()
    repo=Path(__file__).resolve().parents[1]
    protocol=json.loads((repo/"protocol/correction_v2_20261107.json").read_text())
    protocol["main_data"]={"manifest_sha256":"f"*64,"events_sha256":"e"*64}
    protocol_path=root/"synthetic_protocol.json";_write(protocol_path,protocol)
    diagnostic=list(itertools.product(np.linspace(.35,.95,7),(.1,.3,1.),(-1.,1.)))
    p=np.r_[np.linspace(.301,.999,114),[x[0] for x in diagnostic]]
    t=np.r_[np.resize([.1,.3,1.],114),[x[1] for x in diagnostic]]
    q=np.r_[np.resize([-1.,1.],114),[x[2] for x in diagnostic]]
    arrays={"event_uid":np.array([f"SYNTHETIC:{i}" for i in range(len(p))]),"p_in":p,"thickness":t,"charge":q,
            "y":.0001+.00005*t+1e-5*np.sin(np.arange(len(p))),"split":np.array(["train"]*100+["validation"]*14+["diagnostic"]*42)}
    for k,v in arrays.items():np.save(cache/f"{k}.npy",v,allow_pickle=False)
    fractions={}
    for fraction in (.1,1.):
        tag=f"f{round(fraction*100):03d}";idx=subset_indices(arrays,fraction,4101)
        np.save(cache/f"indices_{tag}.npy",idx,allow_pickle=False)
        fractions[tag]={"events":len(idx),"ordered_uid_sha256":canonical_hash(arrays["event_uid"][idx].tolist())}
    valid=np.arange(100,114);np.save(cache/"indices_validation.npy",valid,allow_pickle=False)
    record={"data":protocol["main_data"],"protocol_sha256":canonical_hash(protocol),"eligible_events":len(p),"validation_events":len(valid),"fractions":fractions,
            "files":[{"name":f.name,"sha256":sha256(f)} for f in sorted(cache.glob("*.npy"))]}
    _write(cache/"cache_manifest.json",record)
    code=source_hashes((repo/"src/representation_flow/run_correction_study.py",))
    definition={"mode":"execute","max_updates":20000,"protocol_sha256":canonical_hash(protocol),"data":protocol["main_data"],"code":code,"cache_sha256":canonical_hash(record),"synthetic_fixture":True}
    identity=canonical_hash(definition)
    _write(study/"experiment_identity.json",{"schema":"representation-experiment-v2","identity":definition,"sha256":identity})
    rows=[]
    for fraction,context,rep,seed in itertools.product((.1,1.),protocol["contexts"],protocol["representations"],protocol["seeds"]):
        tag=f"f{round(fraction*100):03d}";train=np.load(cache/f"indices_{tag}.npy")
        run_id=f"correction__{context}__{rep}__Small__{tag}__sub4101__seed{seed}"
        run=study/"runs"/run_id;run.mkdir(parents=True)
        pre=RepresentationPreprocessor(rep,np.zeros(1),0.,float(arrays["y"][train].std()))
        pre.mu=float((arrays["y"][train]-pre.center(p[train],t[train],q[train])).mean())
        mode=protocol["contexts"][context];dim=5 if context=="Full" else 2
        flow=ConditionalSplineFlow(context_dim=dim,hidden=32,seed=seed);initialization=_state_hash(flow.state_dict())
        contract=build_checkpoint_contract(pre,mode)
        with torch.no_grad():
            terms=flow.original_terms(torch.tensor(arrays["y"][valid],dtype=torch.float32),torch.tensor(pre.center(p[valid],t[valid],q[valid]),dtype=torch.float32),pre.mu,
                torch.tensor(pre.scale,dtype=torch.float32),torch.tensor(p[valid],dtype=torch.float32),torch.tensor(context_features(p[valid],t[valid],q[valid],mode),dtype=torch.float32))
        logp=terms["log_p_y"].numpy();selected_nll=float(np.mean(-logp))
        checkpoint=run/"checkpoint.pt"
        torch.save({"state_dict":flow.state_dict(),"representation":rep,"seed":seed,"selected_update":1,"initialization_hash":initialization,
                    "training_state":{"update":1},"checkpoint_contract":contract,"experiment_identity":identity,
                    "model_config":{"context_dim":dim,"hidden":32,"seed":seed,"bound":8.},"synthetic_untrained_fixture":True},checkpoint)
        prediction=run/"predictions.npz"
        np.savez(prediction,event_uid=arrays["event_uid"][valid],y=arrays["y"][valid],p_in=p[valid],c=pre.center(p[valid],t[valid],q[valid]),
                 mu=np.full(len(valid),pre.mu),s=np.full(len(valid),pre.scale,dtype=np.float32),representation=np.full(len(valid),rep),seed=np.full(len(valid),seed),
                 subset_id=np.full(len(valid),4101),split=np.full(len(valid),"validation"),log_p_y=logp,log_q_z=terms["log_q_z"].numpy(),log_F_b=terms["log_f_b"].numpy())
        row={"run_id":run_id,"representation":rep,"capacity":"Small","seed":seed,"selected_update":1,"initialization_hash":initialization,
             "preprocessor":pre.to_dict(),"checkpoint_contract":contract,"train_events":len(train),"validation_events":len(valid),"max_updates":20000,
             "batch_size":512,"validation_every":500,"learning_rate":.0003,"context_feature_mode":mode,"context_label":context,"fraction_tag":tag,"sample_fraction":fraction,
             "experiment_identity":identity,"validation_nll":float(np.mean(-logp.astype(np.float64))),
             "train_event_uid_sha256":fractions[tag]["ordered_uid_sha256"],"validation_event_uid_sha256":canonical_hash(arrays["event_uid"][valid].tolist()),
             "history":[{"update":u,"validation_nll":selected_nll if u==1 else selected_nll+1.} for u in [1,*range(500,20001,500)]],"synthetic_untrained_fixture":True}
        manifest=run/"run_manifest.json";_write(manifest,row)
        row["artifacts"]={"checkpoint":artifact_record(checkpoint,study),"predictions":artifact_record(prediction,study),"run_manifest":artifact_record(manifest,study)}
        rows.append(row)
    _write(study/"scores.json",{"status":"COMPLETE","experiment_identity":identity,"runs":rows,"failures":[],"execution_started_unix":time.time(),"synthetic_untrained_fixture":True})
    _write(study/"FROZEN_CHECKPOINT_MANIFEST.json",{"schema":"correction-checkpoint-freeze-v1","experiment_identity":identity,"frozen_before_confirmation":True,
        "models":[{k:r[k] for k in ("run_id","representation","context_label","sample_fraction","seed","artifacts")} for r in rows]})
    # Confirmation tests require explicit frozen failed-budget evidence even
    # when this synthetic fixture intentionally has no extension fits.
    from representation_flow.score_correction_confirmation import freeze_runtime_deferral
    extension=root/"deferred_extension";(extension/"benchmark").mkdir(parents=True)
    extension_protocol={"schema":"corrected-formal-extension-v1","core_protocol_sha256":canonical_hash(protocol),"synthetic_fixture":True}
    extension_protocol_path=root/"synthetic_extension_protocol.json";_write(extension_protocol_path,extension_protocol)
    _write(extension/"benchmark/benchmark_result.json",{"allowed_combined_under_three_hours":False,"conservative_combined_total_seconds":11000.,"common_identity":{"protocol_sha256":canonical_hash(extension_protocol)}})
    freeze_runtime_deferral(extension,protocol_path,extension_protocol_path)
    return root,study,cache,protocol,protocol_path


def test_complete_60_cell_analysis_retains_all_seeds_and_diagnostic_bins(complete_matrix):
    root,study,cache,protocol,protocol_path=complete_matrix
    result=analyze(study,cache,protocol_path,root/"analysis")
    assert len(result["cells"])==60
    assert len(result["seed_contrasts"])==90
    assert len(result["contrast_summary"])==18
    assert len(result["source_diagnostic_locations"])==6
    assert all(row["populated_bins"]==42 for row in result["source_diagnostic_locations"])
    assert all(row["checkpoint_replayed_validation_events"]==14 for row in result["cells"])
    assert "not uncertainty from independent" in result["interpretation_boundary"]
    assert (root/"analysis/SCIENTIFIC_ANALYSIS.txt").exists()
    with pytest.raises(FileExistsError):analyze(study,cache,protocol_path,root/"analysis")


@pytest.mark.parametrize("mutation",[
    lambda s:s.update(status="RUNNING"),
    lambda s:s["failures"].append({"run_id":"failed","error":"synthetic failure retained"}),
    lambda s:s["runs"].pop(),
    lambda s:s["runs"].append(copy.deepcopy(s["runs"][0])),
    lambda s:s["runs"][0].update(train_event_uid_sha256="wrong"),
    lambda s:s["runs"][0].update(validation_nll=123.),
])
def test_analysis_rejects_incomplete_failed_duplicate_or_changed_runs(complete_matrix,mutation):
    _,study,cache,protocol,_=complete_matrix
    with _changed_json(study/"scores.json",mutation):
        with pytest.raises(ValueError):validate_study(study,cache,protocol)


def test_analysis_rejects_unfrozen_or_changed_artifact(complete_matrix):
    _,study,cache,protocol,_=complete_matrix
    with _changed_json(study/"FROZEN_CHECKPOINT_MANIFEST.json",lambda f:f.update(frozen_before_confirmation=False)):
        with pytest.raises(ValueError,match="frozen"):validate_study(study,cache,protocol)
    scores=json.loads((study/"scores.json").read_text())
    path=study/scores["runs"][0]["artifacts"]["predictions"]["path"]
    before=path.read_bytes();path.write_bytes(before+b"changed")
    try:
        with pytest.raises(ValueError,match="artifact"):validate_study(study,cache,protocol)
    finally:path.write_bytes(before)


def test_negative_seed_contrasts_are_not_dropped():
    protocol={"sample_fractions":[.1,1.],"seeds":[3101,3102,3103,3104,3105]}
    predictions={}
    values=[-2.,-1.,0.,1.,2.]
    for context,tag,seed in itertools.product(("Full","Hidden"),("f010","f100"),protocol["seeds"]):
        offset=values[seed-3101]*(1 if context=="Full" else .5)
        predictions[(context,tag,seed,"Raw")]=np.zeros(3)
        predictions[(context,tag,seed,"CorrectedPhysics")]=np.full(3,offset)
        predictions[(context,tag,seed,"HistoricalIncorrectPhysics")]=np.full(3,offset/2)
    rows,summary=paired_contrasts(predictions,protocol)
    target=next(r for r in summary if r["fraction_tag"]=="f010" and r["context"]=="Full" and r["contrast"]=="Raw_minus_CorrectedPhysics")
    assert target["mean"]==0 and target["negative_seeds"]==2 and target["positive_seeds"]==2 and target["zero_seeds"]==1
    assert target["sample_sd"]==pytest.approx(np.std(values,ddof=1))
    assert len(rows)==90
