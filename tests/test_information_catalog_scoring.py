"""Fresh scoring adapters preserve old models and independently enforce counts."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from representation_flow.engine import RepresentationPreprocessor
from representation_flow.experiment_contract import artifact_record, canonical_hash, sha256
from representation_flow.run_a0_prototype import _train_one
from representation_flow.run_information_study import train_information_cell
from representation_flow.score_information_catalog import (COMPONENT_NAMES, preflight_models,
    read_catalog_arrays, score_model, safe_id, tensors_for_model, verify_source_analysis)


def synthetic():
    n=160; rng=np.random.default_rng(29)
    return {"event_uid":np.asarray([f"test:{i}" for i in range(n)]),"p_in":np.linspace(.3,1.,n),
            "thickness":np.resize([.1,.3,1.],n),"charge":np.resize([-1.,1.],n),
            "y":rng.normal(.0004,.00008,n),"split":np.asarray(["train"]*128+["validation"]*32)}


def test_legacy_and_information_full_scoring_preserve_all_components(tmp_path):
    a=synthetic();train=np.arange(128);valid=np.arange(128,160)
    pre=RepresentationPreprocessor("Raw",np.zeros(1),float(np.mean(a["y"][train])),float(np.std(a["y"][train])))
    args=SimpleNamespace(max_updates=2,batch_size=32,learning_rate=.0003,validation_every=1,eval_batch_size=32,
                         torch_threads=1,feature_mode="full",run_prefix="legacy",q99_threshold=0.,
                         deadline_unix=time.time()+90,experiment_identity="same")
    old,_=_train_one(a,train,valid,pre,"Raw","Small",3101,args,tmp_path/"old")
    old_artifacts={k:artifact_record(tmp_path/"old/runs"/old["run_id"]/file,tmp_path/"old")
                   for k,file in (("checkpoint","checkpoint.pt"),("predictions","predictions.npz"),("run_manifest","run_manifest.json"))}
    job={"run_id":"information_test","context_label":"Full5","representation":"Raw","seed":3101,
         "fraction_tag":"f100","sample_fraction":1.,"experiment_identity":"same","protocol_sha256":"synthetic",
         "max_updates":2,"deadline_unix":time.time()+90,"output":str(tmp_path/"new"),
         "optimization":{"batch_size":32,"learning_rate":.0003,"validation_every":1,"eval_batch_size":32,
                         "gradient_clip_norm":5.,"torch_threads_per_worker":1}}
    new=train_information_cell(a,train,valid,pre,job)
    results=[]
    for kind,artifacts,context,root in (("legacy",old_artifacts,"Full",tmp_path/"old"),("information",new["artifacts"],"Full5",tmp_path/"new")):
        row={"model_id":kind,"cohort":"legacy160" if kind=="legacy" else "information80","checkpoint_kind":kind,
             "root_key":kind,"experiment_identity":"same","artifacts":artifacts,
             "labels":{"context_label":context,"representation":"Raw","capacity":"Small","seed":3101}}
        result=score_model(row,{kind:root},a,tmp_path/"scores",canonical_hash(a["event_uid"].tolist()),time.time()+90,batch_size=32)
        assert result["eligible_events"]==160 and set(result["arrays"])==set(COMPONENT_NAMES)
        results.append(result)
    for name in COMPONENT_NAMES:
        left=np.load(tmp_path/"scores"/results[0]["arrays"][name]["path"],allow_pickle=False)
        right=np.load(tmp_path/"scores"/results[1]["arrays"][name]["path"],allow_pickle=False)
        np.testing.assert_array_equal(left,right)
    with pytest.raises(FileExistsError):score_model(row,{kind:root},a,tmp_path/"scores","same",time.time()+90,batch_size=32)
    with pytest.raises(TimeoutError):score_model(row,{kind:root},a,tmp_path/"expired","same",time.time()-1,batch_size=32)
    assert not (tmp_path/"expired").exists()


def catalog(folder):
    folder.mkdir();(folder/"merged").mkdir(); rows=[];cells=[]
    for group in range(42):
        p=.3+(group//6+.5)*.1;t=[.1,.3,1.][(group%6)//2];pdg=[-211,211][group%2]
        for s1 in (False,True):
            rows.append({"event_uid":f"fresh-synthetic:{len(rows)}","S1":s1,"aborted":False,
                         "p_in_GeV_c":p,"p_out_GeV_c":p-.001,"delta_p_GeV_c":.001,
                         "thickness_mm":t,"pdg":pdg,"momentum_bin":group//6,"exit_face":"forward"})
        cells.append({"group_id":group,"launched_events":2,"eligible_events":1,"ineligible":1,"aborted":0})
    path=folder/"merged/events.jsonl";path.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    counts={"schema":"fresh-catalog-stratum-counts/v1","status":"COMPLETE_42_STRATUM_COUNTS","run_id":"fresh-synthetic",
            "events_sha256":sha256(path),"launched":84,"eligible":42,"technical_aborted":0,"cells":cells}
    (folder/"fresh_catalog_manifest.json").write_text(json.dumps(counts),encoding="utf-8")
    return rows,counts


def test_launched_s0_accounting_and_float64_neural_inputs(tmp_path):
    folder=tmp_path/"catalog";catalog(folder)
    a,launched,eligible,uids=read_catalog_arrays(folder,{"run_id":"fresh-synthetic"},time.time()+90,expected_launched=84)
    assert len(uids)==84 and len(a["y"])==42
    assert sum(launched.values())==84 and sum(eligible.values())==42
    assert all(a[k].dtype==np.float64 for k in ("p_in","thickness","charge","y"))
    from representation_flow.information_context import context_spec
    pre=RepresentationPreprocessor("Raw",np.zeros(1),0.,.001)
    assert tensors_for_model(a,pre,context_spec("Coarse8"),"information")[3].shape==(42,5)
    assert tensors_for_model(a,pre,"thickness_charge","legacy")[3].shape==(42,2)


@pytest.mark.parametrize("change",["duplicate","abort","count","wrong_run"])
def test_catalog_rejects_corruption_even_with_updated_event_hash(tmp_path,change):
    folder=tmp_path/"catalog";rows,counts=catalog(folder)
    if change=="duplicate":rows[1]["event_uid"]=rows[0]["event_uid"]
    if change=="abort":rows[0]["aborted"]=True
    if change=="count":counts["cells"][0]["eligible_events"]=2
    if change=="wrong_run":rows[0]["event_uid"]="another:0"
    path=folder/"merged/events.jsonl";path.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    counts["events_sha256"]=sha256(path)
    (folder/"fresh_catalog_manifest.json").write_text(json.dumps(counts),encoding="utf-8")
    with pytest.raises(ValueError):read_catalog_arrays(folder,{"run_id":"fresh-synthetic"},time.time()+90,expected_launched=84)


def test_complete_roster_and_safe_output_ids_required():
    with pytest.raises(ValueError):preflight_models({}, {"models":[],"schema":"fresh-model-roster/v1","expected_models":2}, {})
    for value in ("../escape","C:\\outside","a/b",""):
        with pytest.raises(ValueError):safe_id(value)


def test_generation_binds_complete_replayed_source_analysis_and_exact_eighty_freeze(tmp_path):
    artifacts={role:{"path":role,"sha256":"a"*64,"bytes":1} for role in ("checkpoint","predictions","run_manifest")}
    models=[{"model_id":f"m{i}","cohort":"information80","experiment_identity":"fixed","artifacts":artifacts} for i in range(80)]
    freeze={"schema":"information-checkpoint-freeze/v1","status":"COMPLETE_INFORMATION_CHECKPOINT_FREEZE",
            "expected_cells":80,"experiment_identity":"fixed","protocol_sha256":"protocol",
            "models":[{"run_id":r["model_id"],"artifacts":artifacts} for r in models]}
    frozen_path=tmp_path/"freeze.json";frozen_path.write_text(json.dumps(freeze),encoding="utf-8")
    source={"schema":"information-source-analysis/v1","status":"COMPLETE_INFORMATION_SOURCE_ANALYSIS",
            "expected_models":80,"model_count":80,"all_validation_predictions_replayed":True,
            "study_freeze_sha256":sha256(frozen_path),"protocol_sha256":"protocol"}
    source_path=tmp_path/"source.json";source_path.write_text(json.dumps(source),encoding="utf-8")
    def record(path):return dict(artifact_record(path,tmp_path),root_key="tmp")
    plan={"source_validation_analysis":record(source_path)}
    union={"models":models,"information_freeze":record(frozen_path)}
    receipt={"information_freeze_sha256":sha256(frozen_path),"source_validation_analysis_sha256":sha256(source_path)}
    verify_source_analysis(plan,union,receipt,{"tmp":tmp_path})
    source["all_validation_predictions_replayed"]=False
    source_path.write_text(json.dumps(source),encoding="utf-8")
    plan["source_validation_analysis"]=record(source_path)
    receipt["source_validation_analysis_sha256"]=sha256(source_path)
    with pytest.raises(ValueError,match="independent complete"):
        verify_source_analysis(plan,union,receipt,{"tmp":tmp_path})
