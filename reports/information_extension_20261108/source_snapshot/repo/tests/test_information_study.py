"""Equal-structure, feature identity, checkpoint and pairing regressions."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from representation_flow.engine import common_features, bottleneck_features, RepresentationPreprocessor
from representation_flow.information_context import CONTEXTS, context_spec, information_features
from representation_flow.information_checkpoint import load_information_checkpoint, make_tensors, evaluate_information
from representation_flow.run_information_study import train_information_cell, validate_pairing, budget_contract
from representation_flow.run_a0_prototype import _train_one
from representation_flow.experiment_contract import canonical_hash


def arrays(n=160):
    rng = np.random.default_rng(19)
    p = np.linspace(.3, 1., n)
    return {"event_uid": np.asarray([f"synthetic-information:{i}" for i in range(n)]), "p_in": p,
            "thickness": np.resize(np.array([.1, .3, 1.]), n), "charge": np.resize(np.array([-1., 1.]), n),
            "y": .0004 + rng.normal(0, .00008, n), "split": np.asarray(["train"]*128 + ["validation"]*(n-128))}


def test_features_explicit_padding_full_identity_and_nested_partitions():
    a = arrays(); p,t,q = (a[k] for k in ("p_in", "thickness", "charge"))
    np.testing.assert_array_equal(information_features(p,t,q,context_spec("Full5")), common_features(p,t,q))
    hidden = information_features(p,t,q,context_spec("Hidden5"))
    np.testing.assert_array_equal(hidden[:, [1,2]], bottleneck_features(p,t,q))
    assert np.all(hidden[:, [0,3,4]] == 0)
    p = np.asarray([.3,.3875,.475,.5625,.65,.7375,.825,.9125,1.])
    coarse = information_features(p,np.full(len(p),.3),np.ones(len(p)),context_spec("Coarse2"))
    fine = information_features(p,np.full(len(p),.3),np.ones(len(p)),context_spec("Coarse8"))
    np.testing.assert_array_equal(coarse[:,0] > 0, fine[:,0] > 0)
    assert coarse[3,0] < 0 and coarse[4,0] > 0
    assert context_spec("Hidden5")["columns"][0] == "zero"


@pytest.mark.parametrize("bad", [-.1, .299999, 1.000001, np.nan])
def test_context_rejects_outside_prespecified_physical_range(bad):
    with pytest.raises(ValueError):information_features([bad],[.3],[1.],context_spec("Coarse2"))


def test_changed_bin_edge_not_accepted():
    spec=context_spec("Coarse2");spec["edges_GeV_c"][1]=.6
    with pytest.raises(ValueError):information_features([.5],[.3],[1.],spec)


def job(root, context, rep="Raw"):
    return {"run_id":context+"__"+rep,"context_label":context,"representation":rep,"seed":3101,
            "fraction_tag":"f100","sample_fraction":1.,"experiment_identity":"synthetic-identity",
            "protocol_sha256":"test-protocol","max_updates":2,"deadline_unix":time.time()+90,"output":str(root),
            "optimization":{"batch_size":32,"learning_rate":.0003,"validation_every":1,"eval_batch_size":32,
                            "gradient_clip_norm":5.,"torch_threads_per_worker":1}}


def test_actual_training_save_reload_components_and_all_context_pairing(tmp_path):
    a=arrays();indices=np.arange(128);valid=np.arange(128,160)
    pre=RepresentationPreprocessor("Raw",np.zeros(1),float(np.mean(a["y"][indices])),float(np.std(a["y"][indices])))
    rows=[]
    for context in CONTEXTS:
        j=job(tmp_path,context)
        row=train_information_cell(a,indices,valid,pre,j);rows.append(row)
        path=tmp_path/row["artifacts"]["checkpoint"]["path"]
        payload,frozen,flow,spec=load_information_checkpoint(path,expected_sha256=row["artifacts"]["checkpoint"]["sha256"])
        assert payload["training_state"]["update"]==payload["selected_update"]
        assert all(float(state["step"])==payload["selected_update"] for state in payload["optimizer_state_dict"]["state"].values())
        pred=evaluate_information(flow,make_tensors(a,valid,frozen,spec),frozen,32,components=True)
        reconstructed=pred["base_logpdf"]+pred["spline_logabsdet"]-pred["log_s"]-pred["log_support_cdf"]
        np.testing.assert_allclose(reconstructed,pred["log_p_y"],rtol=2e-6,atol=2e-6)
        with pytest.raises(FileExistsError):train_information_cell(a,indices,valid,pre,j)
        payload["information_contract"]["context_spec"]["name"]="Full5" if context!="Full5" else "Hidden5"
        altered=tmp_path/(context+"_altered.pt");torch.save(payload,altered)
        with pytest.raises(ValueError):load_information_checkpoint(altered)
    validate_pairing(rows)
    assert len({r["parameter_count"] for r in rows})==1 and len({r["initialization_hash"] for r in rows})==1
    changed=deepcopy(rows);changed[0]["batch_trace_sha256"]="wrong"
    with pytest.raises(ValueError):validate_pairing(changed)


def test_deadline_stops_before_work_and_cannot_be_rebased(tmp_path):
    a=arrays();pre=RepresentationPreprocessor("Raw",np.zeros(1),0.,.001)
    j=job(tmp_path,"Full5");j["deadline_unix"]=time.time()-1
    with pytest.raises(TimeoutError):train_information_cell(a,np.arange(128),np.arange(128,160),pre,j)
    assert not (tmp_path/"runs").exists()
    p=tmp_path/"budget.json"
    p.write_text(json.dumps({"started_unix":time.time(),"deadline_unix":time.time()+90000,"maximum_wall_seconds":43200}))
    with pytest.raises(ValueError):budget_contract(p)


@pytest.mark.parametrize("representation", ["Raw", "CorrectedPhysics"])
def test_full_adapter_numerically_replays_frozen_training_path(tmp_path, representation):
    a=arrays(); indices=np.arange(128); valid=np.arange(128,160)
    pre=RepresentationPreprocessor(representation,np.zeros(1),0.,float(np.std(a["y"][indices])))
    pre.mu=float(np.mean(a["y"][indices]-pre.center(*(a[k][indices] for k in ("p_in","thickness","charge")))))
    j=job(tmp_path/"new","Full5",representation);j["max_updates"]=7
    row=train_information_cell(a,indices,valid,pre,j)
    args=SimpleNamespace(max_updates=7,batch_size=32,learning_rate=.0003,validation_every=1,eval_batch_size=32,
                         torch_threads=1,feature_mode="full",run_prefix="equivalence",q99_threshold=0.,
                         deadline_unix=time.time()+90,experiment_identity="synthetic-identity")
    old,prediction=_train_one(a,indices,valid,pre,representation,"Small",3101,args,tmp_path/"old")
    assert row["initialization_hash"]==old["initialization_hash"]
    assert row["parameter_count"]==old["parameter_count"] and row["selected_update"]==old["selected_update"]
    assert row["history"]==old["history"]
    fresh=np.load(tmp_path/"new"/row["artifacts"]["predictions"]["path"],allow_pickle=False)
    np.testing.assert_array_equal(fresh["log_p_y"],prediction["log_p_y"])
    original=torch.load(old["checkpoint"],map_location="cpu",weights_only=True)
    saved,*_=load_information_checkpoint(tmp_path/"new"/row["artifacts"]["checkpoint"]["path"])
    assert all(torch.equal(saved["state_dict"][k],v) for k,v in original["state_dict"].items())
    assert torch.equal(saved["training_state"]["batch_order"],original["training_state"]["batch_order"])
    saved["information_contract"]["source_files"]["src/representation_flow/engine.py"]="0"*64
    saved["information_contract"]["sha256"]=canonical_hash({k:v for k,v in saved["information_contract"].items() if k!="sha256"})
    changed=tmp_path/"changed_source.pt";torch.save(saved,changed)
    with pytest.raises(ValueError,match="scientific sources"):load_information_checkpoint(changed)


def test_hidden_neural_features_do_not_hide_true_p_from_center_or_support():
    a={"p_in":np.array([.35,.95]),"thickness":np.array([.3,.3]),"charge":np.array([1.,1.]),"y":np.array([.001,.001])}
    pre=RepresentationPreprocessor("CorrectedPhysics",np.zeros(1),0.,.001)
    y,c,p,features,scale=make_tensors(a,np.arange(2),pre,context_spec("Hidden5"))
    assert torch.equal(features[0],features[1])
    assert c[0]!=c[1] and p[0]!=p[1]
    assert ((p-c-pre.mu)/scale)[0]!=((p-c-pre.mu)/scale)[1]
