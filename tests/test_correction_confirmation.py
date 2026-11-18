"""Synthetic-only tests: no real confirmation ledger is opened here."""
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from test_correction_analysis import complete_matrix, _changed_json
from representation_flow import score_correction_confirmation as scorer
from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import ConditionalSplineFlow, context_features
from representation_flow.experiment_contract import data_identity


def test_incomplete_training_blocks_confirmation_before_any_data_access(complete_matrix,monkeypatch):
    root,study,cache,_,protocol_path=complete_matrix
    called=[]
    def forbidden(*args,**kwargs):
        called.append(True)
        raise AssertionError("confirmation data must not be accessed")
    monkeypatch.setattr(scorer,"prepare_confirmation_cache",forbidden)
    with _changed_json(study/"scores.json",lambda s:s.update(status="RUNNING")):
        with pytest.raises(ValueError,match="incomplete"):
            scorer.score_confirmation(study,cache,protocol_path,root/"missing_data",root/"denied_confirmation",deadline_unix=time.time()+60)
    assert called==[] and not (root/"denied_confirmation").exists()


def test_existing_unfinished_extension_blocks_confirmation(complete_matrix,tmp_path):
    _,study,cache,protocol,protocol_path=complete_matrix
    extension=tmp_path/"extension";extension.mkdir();(extension/"benchmark").mkdir()
    with pytest.raises(ValueError,match="extension exists"):
        scorer.preflight_models(study,cache,protocol_path,extension_root=extension)
    with pytest.raises(ValueError,match="missing extension path"):
        scorer.extension_completion_gate(tmp_path/"absent_extension",study,protocol)


def test_streaming_all_six_models_matches_direct_frozen_scoring(complete_matrix,tmp_path):
    root,study,cache,_,protocol_path=complete_matrix
    plan=scorer.preflight_models(study,cache,protocol_path,extension_root=root/"deferred_extension")
    assert plan["scope"]=="ALL_CORE60_ONLY" and len(plan["models"])==60
    source=tmp_path/"data"/"synthetic_confirmation_a";source.mkdir(parents=True)
    rows=[]
    for i,(p,t,q) in enumerate((p,t,q) for p in np.linspace(.35,.95,7) for t in (.1,.3,1.) for q in (-1,1)):
        rows.append({"event_uid":f"SYNTHETIC_CONFIRMATION:{i}","S1":True,"exit_face":"forward","p_in_GeV_c":float(p),"thickness_mm":t,"pdg":211*q,"delta_p_GeV_c":.0001+t*.00005+i*1e-7})
    (source/"events.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (source/"manifest.json").write_text(json.dumps({"events":42,"synthetic_fixture":True}))
    expected={"directory":source.name,"eligible_events":42,**data_identity(source)}
    confirmation_cache=tmp_path/"cache"
    result=scorer.prepare_confirmation_cache(source.parent,confirmation_cache,"a",expected,time.time()+60)
    assert result["all_eligible_events_scored"] and result["no_selection_or_resampling"]
    arrays={p.stem:np.load(p,mmap_mode="r",allow_pickle=False) for p in (confirmation_cache/"a").glob("*.npy")}
    scorer._BLOCK_ARRAYS={"a":arrays}
    selected=[m for m in plan["models"] if m["fraction_tag"]=="f010" and m["seed"]==3101]
    job={"job_id":"synthetic","block":"a","fraction_tag":"f010","seed":3101,"models":selected,"batch_size":7,"deadline":time.time()+60}
    result=scorer.score_group(job)
    assert len(result["models"])==6 and len(result["contrasts"])==9
    assert len(result["model_groups"])==6*42 and len(result["contrast_groups"])==9*42
    direct={}
    p,t,q,y=(arrays[k].copy() for k in ("p_in","thickness","charge","y"))
    for model in selected:
        payload,pre=load_checkpoint(model["checkpoint"])
        flow=ConditionalSplineFlow(**payload["model_config"]);flow.load_state_dict(payload["state_dict"])
        mode=payload["checkpoint_contract"]["context_feature_mode"]
        with torch.no_grad():
            terms=flow.original_terms(torch.tensor(y,dtype=torch.float32),torch.tensor(pre.center(p,t,q),dtype=torch.float32),pre.mu,
                torch.tensor(pre.scale,dtype=torch.float32),torch.tensor(p,dtype=torch.float32),torch.tensor(context_features(p,t,q,mode),dtype=torch.float32))
        direct[(model["context"],model["representation"])]=terms["log_p_y"].numpy().astype(np.float64)
    for row in result["models"]:
        assert row["events"]==42
        assert row["mean"]==pytest.approx(-direct[(row["context"],row["representation"])].mean(),abs=2e-5)
    for row in result["contrasts"]:
        left,right=next(pair for pair in scorer.CONTRASTS if f"{pair[0]}_minus_{pair[1]}"==row["contrast"])
        if row["context"]=="Full_minus_Hidden":
            diff=(direct[("Full",right)]-direct[("Full",left)])-(direct[("Hidden",right)]-direct[("Hidden",left)])
        else:diff=direct[(row["context"],right)]-direct[(row["context"],left)]
        assert row["mean"]==pytest.approx(diff.mean(),abs=2e-5)
        assert row["event_sd_descriptive"]==pytest.approx(diff.std(ddof=1),abs=2e-5)
    with pytest.raises(ValueError,match="incomplete"):
        scorer.aggregate_confirmation([result])


def test_confirmation_cache_rejects_changed_data_identity_before_loading(tmp_path):
    source=tmp_path/"data";source.mkdir()
    (source/"manifest.json").write_text("{}")
    (source/"events.jsonl").write_text("not valid json; hash rejection must happen first")
    expected={"directory":"data","manifest_sha256":"0"*64,"events_sha256":"0"*64,"eligible_events":42}
    with pytest.raises(ValueError,match="hash mismatch"):
        scorer.prepare_confirmation_cache(tmp_path,tmp_path/"cache","a",expected,time.time()+60)


def test_post_preflight_cache_failure_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(scorer,"preflight_models",lambda *a,**k:{"scope":"SYNTHETIC_TEST","models":[],"core_execution_started_unix":time.time(),"total_budget_seconds":10800})
    (tmp_path/"cache_manifest.json").write_text("{}")
    timing=tmp_path/"timing.json"
    timing.write_text(json.dumps({"schema":"confirmation-main-data-timing/v1","scoring_source_sha256":scorer.sha256(Path(scorer.__file__)),
        "source_cache_manifest_sha256":scorer.sha256(tmp_path/"cache_manifest.json"),"torch_version":str(torch.__version__),"numpy_version":np.__version__,
        "worker_threads":1,"planned_workers":4,"forecasts":{"SYNTHETIC_TEST":{"conservative_total_seconds":1.}}}))
    def reject(*args,**kwargs):raise ValueError("synthetic cache identity failure")
    monkeypatch.setattr(scorer,"prepare_confirmation_cache",reject)
    output=tmp_path/"failed_scoring"
    with pytest.raises(ValueError,match="synthetic cache"):
        scorer.score_confirmation(tmp_path,tmp_path,tmp_path/"unused.json",tmp_path,output,deadline_unix=time.time()+60,timing_report=timing)
    state=json.loads((output/"execution_state.json").read_text())
    assert state["status"]=="INCOMPLETE" and state["failures"][0]["stage"]=="confirmation_cache"
    assert not (output/"confirmation_summary.json").exists()


def test_fixed_group_ids_and_expired_job(complete_matrix):
    root,study,cache,_,protocol_path=complete_matrix
    assert np.array_equal(scorer.fixed_group_ids(np.array([.3,1.]),np.array([.1,1.]),np.array([-1.,1.])),[0,41])
    with pytest.raises(ValueError,match="same-range"):
        scorer.fixed_group_ids(np.array([1.1]),np.array([.1]),np.array([1.]))
    plan=scorer.preflight_models(study,cache,protocol_path,extension_root=root/"deferred_extension")
    selected=[m for m in plan["models"] if m["fraction_tag"]=="f010" and m["seed"]==3101]
    scorer._BLOCK_ARRAYS={"a":{"y":np.zeros(1)}}
    with pytest.raises(TimeoutError,match="deadline"):
        scorer.score_group({"job_id":"expired","block":"a","fraction_tag":"f010","seed":3101,"models":selected,"batch_size":1,"deadline":time.time()-1})


def test_result_serialization_is_included_in_hard_scoring_deadline(tmp_path,monkeypatch):
    from concurrent.futures import Future
    clock=[time.time()]
    monkeypatch.setattr(scorer.time,"time",lambda:clock[0])
    models=[{"fraction_tag":tag,"seed":seed} for tag in ("f010","f100") for seed in range(3101,3106) for _ in range(6)]
    plan={"scope":"ALL_CORE60_ONLY","models":models,"model_count":60,"core_execution_started_unix":clock[0],"total_budget_seconds":10800,
          "core_experiment_identity":"synthetic","extension_gate":{"status":"DEFERRED_BY_RUNTIME_BUDGET"},"formal120_to_unique_models":[]}
    monkeypatch.setattr(scorer,"preflight_models",lambda *a,**k:plan)
    (tmp_path/"cache_manifest.json").write_text("{}")
    timing=tmp_path/"timing.json"
    timing.write_text(json.dumps({"schema":"confirmation-main-data-timing/v1","scoring_source_sha256":scorer.sha256(Path(scorer.__file__)),
        "source_cache_manifest_sha256":scorer.sha256(tmp_path/"cache_manifest.json"),"torch_version":str(torch.__version__),"numpy_version":np.__version__,
        "worker_threads":1,"planned_workers":4,"forecasts":{"ALL_CORE60_ONLY":{"conservative_total_seconds":1.}}}))
    monkeypatch.setattr(scorer,"prepare_confirmation_cache",lambda *a,**k:{"synthetic":True})
    class SyntheticExecutor:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,*a):
            future=Future();future.set_result({"synthetic":True});return future
    monkeypatch.setattr(scorer,"ProcessPoolExecutor",SyntheticExecutor)
    monkeypatch.setattr(scorer,"aggregate_confirmation",lambda jobs:{key:[] for key in ("models","seed_contrasts","contrast_summary","model_groups","contrast_groups")})
    monkeypatch.setattr(scorer,"write_csv",lambda *a:clock.__setitem__(0,clock[0]+61.))
    output=tmp_path/"deadline_expired_during_output"
    with pytest.raises(TimeoutError,match="output exceeded"):
        scorer.score_confirmation(tmp_path,tmp_path,tmp_path/"unused.json",tmp_path,output,deadline_unix=clock[0]+60,timing_report=timing)
    assert json.loads((output/"confirmation_summary.json").read_text())["status"]=="INCOMPLETE_DEADLINE_EXCEEDED"
    state=json.loads((output/"execution_state.json").read_text())
    assert state["status"]=="INCOMPLETE_DEADLINE_EXCEEDED" and state["elapsed_seconds"]>60
