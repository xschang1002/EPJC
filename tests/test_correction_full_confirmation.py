"""Synthetic all160 scope, portable reuse and formal budget-deferral tests."""
import copy
import itertools
import json
import shutil
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from test_correction_analysis import complete_matrix, _changed_json, _write
from test_corrected_formal_analysis import full_matrix
from representation_flow import score_correction_confirmation as scorer
from representation_flow.checkpoint_contract import build_checkpoint_contract, load_checkpoint
from representation_flow.engine import ConditionalSplineFlow, context_features, fit_preprocessors, subset_indices
from representation_flow.experiment_contract import artifact_record, canonical_hash, sha256
from representation_flow.run_a0_prototype import _state_hash


@pytest.fixture(scope="module")
def complete_extension(full_matrix):
    extension=full_matrix["root"]
    # Original machine roots remain provenance; current supplied roots resolve
    # all verified artifacts and independently reconstructed private caches.
    for name in ("formal_scores_120.json","FROZEN_CHECKPOINT_MANIFEST.json"):
        path=extension/"study"/name;record=json.loads(path.read_text())
        for row in record["runs" if "runs" in record else "models"]:
            row["artifact_root"]="Z:/old-machine/unavailable/core-study" if row["reused"] else "/unavailable/original/mac/extension-study"
        _write(path,record)
    return extension


def test_portable_all160_plan_reuses_twenty_without_duplicate_inference(complete_matrix,complete_extension):
    _,core,cache,_,protocol_path=complete_matrix
    plan=scorer.preflight_models(core,cache,protocol_path,extension_root=complete_extension,scope="all160",extension_protocol_path=complete_extension/"synthetic_protocol.json")
    assert plan["scope"]=="ALL_UNIQUE_CORE60_PLUS_EXTENSION100" and plan["model_count"]==160
    assert len({m["checkpoint"] for m in plan["models"]})==160
    assert len(plan["formal120_to_unique_models"])==120 and sum(m["reused"] for m in plan["formal120_to_unique_models"] )==20
    assert len([m for m in plan["models"] if m["capacity"]=="Large"])==60
    assert len([m for m in plan["models"] if m["seed"]==3101])==32


def test_full32_streaming_and_prespecified_interactions(complete_matrix,complete_extension):
    _,core,cache,_,protocol_path=complete_matrix
    plan=scorer.preflight_models(core,cache,protocol_path,extension_root=complete_extension,scope="all160",extension_protocol_path=complete_extension/"synthetic_protocol.json")
    arrays={name:np.load(cache/f"{name}.npy")[114:].copy() for name in ("p_in","thickness","charge","y")}
    arrays["group_id"]=scorer.fixed_group_ids(arrays["p_in"],arrays["thickness"],arrays["charge"])
    scorer._BLOCK_ARRAYS={"a":arrays}
    models=[m for m in plan["models"] if m["seed"]==3101]
    result=scorer.score_full_group({"job_id":"synthetic_full32","block":"a","seed":3101,"models":models,"batch_size":7,"deadline":time.time()+60})
    assert len(result["models"])==32 and len(result["contrasts"])==41
    assert len({r["run_id"] for r in result["models"]})==32
    direct={}
    for record in models:
        if record["context"]!="Full" or record["representation"] not in ("Raw","CorrectedPhysics"):continue
        payload,pre=load_checkpoint(record["checkpoint"]);flow=ConditionalSplineFlow(**payload["model_config"]);flow.load_state_dict(payload["state_dict"])
        p,t,q,y=(arrays[k] for k in ("p_in","thickness","charge","y"))
        with torch.no_grad():
            lp=flow.original_terms(torch.tensor(y,dtype=torch.float32),torch.tensor(pre.center(p,t,q),dtype=torch.float32),pre.mu,pre.scale,
                torch.tensor(p,dtype=torch.float32),torch.tensor(context_features(p,t,q,"full"),dtype=torch.float32))["log_p_y"].numpy().astype(float)
        direct[(record["fraction_tag"],record["capacity"],record["representation"])]=lp
    def gap(tag,cap):return direct[(tag,cap,"CorrectedPhysics")]-direct[(tag,cap,"Raw")]
    capacity_row=next(r for r in result["contrasts"] if r["scope"]=="formal120" and r["fraction_tag"]=="f030" and r["capacity"]=="Small_minus_Large")
    assert capacity_row["mean"]==pytest.approx(np.mean(gap("f030","Small")-gap("f030","Large")),abs=2e-5)
    fraction_row=next(r for r in result["contrasts"] if r["scope"]=="formal120" and r["fraction_tag"]=="f010_minus_f100" and r["capacity"]=="Large")
    assert fraction_row["mean"]==pytest.approx(np.mean(gap("f010","Large")-gap("f100","Large")),abs=2e-5)
    # Synthetic aggregation exercises all prescribed labels without creating or
    # interpreting any scientific result from this untrained fixture.
    jobs=[]
    for block,seed in itertools.product(("a","b"),range(3101,3106)):
        job=copy.deepcopy(result)
        for key in ("models","contrasts","model_groups","contrast_groups"):
            for row in job[key]:
                row.update(block=block,seed=seed)
                if "run_id" in row:row["run_id"]=row["run_id"].replace("seed3101",f"seed{seed}")
        jobs.append(job)
    aggregate=scorer.aggregate_full_confirmation(jobs)
    assert len(aggregate["models"])==320 and len(aggregate["contrast_summary"])==82
    assert all(r["seeds"]==5 for r in aggregate["contrast_summary"])
    with pytest.raises(ValueError,match="incomplete"):
        scorer.aggregate_full_confirmation(jobs[:-1])


def test_formal_runtime_deferral_requires_failed_gate_and_no_formal_fits(complete_matrix,tmp_path):
    _,core,cache,protocol,protocol_path=complete_matrix
    root=tmp_path/"deferred_extension";(root/"benchmark").mkdir(parents=True)
    ext_protocol={"schema":"corrected-formal-extension-v1","core_protocol_sha256":canonical_hash(protocol),"synthetic_fixture":True}
    ext_path=tmp_path/"extension_protocol.json";_write(ext_path,ext_protocol)
    benchmark={"allowed_combined_under_three_hours":False,"conservative_combined_total_seconds":11000.,"common_identity":{"protocol_sha256":canonical_hash(ext_protocol)}}
    _write(root/"benchmark/benchmark_result.json",benchmark)
    decision=scorer.freeze_runtime_deferral(root,protocol_path,ext_path)
    assert decision["status"]=="DEFERRED_BY_RUNTIME_BUDGET"
    gate=scorer.extension_completion_gate(root,core,protocol)
    assert gate["status"]=="DEFERRED_BY_RUNTIME_BUDGET"
    plan=scorer.preflight_models(core,cache,protocol_path,extension_root=root)
    assert plan["scope"]=="ALL_CORE60_ONLY"
    with pytest.raises(ValueError,match="all160"):
        scorer.preflight_models(core,cache,protocol_path,extension_root=root,scope="all160")
    with pytest.raises(FileExistsError):scorer.freeze_runtime_deferral(root,protocol_path,ext_path)
    (root/"study/runs/new").mkdir(parents=True);(root/"study/runs/new/checkpoint.pt").write_bytes(b"synthetic formal fit")
    with pytest.raises(ValueError,match="formal checkpoints"):
        scorer.extension_completion_gate(root,core,protocol)


def test_scope_fallback_is_budget_only_and_explicit_all160_cannot_fallback():
    plan={"scope":"ALL_UNIQUE_CORE60_PLUS_EXTENSION100","models":[{"id":i} for i in range(160)],"model_count":160,"extension_gate":{"scored_here":True}}
    timing={"forecasts":{"ALL_UNIQUE_CORE60_PLUS_EXTENSION100":{"conservative_total_seconds":1100},"ALL_CORE60_ONLY":{"conservative_total_seconds":400}}}
    selected=scorer.select_timed_scope(plan,timing,1000,"auto")
    assert selected["scope"]=="ALL_CORE60_ONLY" and len(selected["models"])==60 and not selected["extension_gate"]["scored_here"]
    with pytest.raises(TimeoutError):scorer.select_timed_scope(plan,timing,1000,"all160")
    with pytest.raises(TimeoutError):scorer.select_timed_scope(plan,timing,300,"auto")
    assert scorer.select_timed_scope(plan,timing,1200,"all160")["model_count"]==160


def test_confirmation_preflight_accepts_only_supplied_private_cache_identity(complete_matrix,complete_extension,tmp_path):
    _,core,cache,_,protocol_path=complete_matrix
    private_core=tmp_path/"private_core";private_extension=tmp_path/"private_extension"
    shutil.copytree(cache,private_core);shutil.copytree(complete_extension/"cache",private_extension)
    with _changed_json(cache/"cache_manifest.json",lambda r:r.update(data={})), _changed_json(complete_extension/"cache/preprocessing_manifest.json",lambda r:r.update(identity={})):
        plan=scorer.preflight_models(core,private_core,protocol_path,extension_root=complete_extension,scope="all160",
             extension_cache=private_extension,extension_protocol_path=complete_extension/"synthetic_protocol.json")
        assert plan["model_count"]==160 and plan["source_cache"]==str(private_core.resolve())
        assert plan["extension_cache"]==str(private_extension.resolve())
    with _changed_json(private_extension/"preprocessing_manifest.json",lambda r:r.update(identity={})):
        with pytest.raises(ValueError,match="preprocessing cache identity"):
            scorer.preflight_models(core,private_core,protocol_path,extension_root=complete_extension,scope="all160",
                 extension_cache=private_extension,extension_protocol_path=complete_extension/"synthetic_protocol.json")
