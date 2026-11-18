"""Regression tests for the unit repair and frozen scientific identity boundary."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.anchor import corrected_pion_bethe_bloch_delta_p, historical_pion_bethe_bloch_delta_p
from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, fit_preprocessors, load_source_arrays
from representation_flow.checkpoint_contract import build_checkpoint_contract, load_checkpoint, contract_digest
from representation_flow.score_frozen_representation_confirmation import prepare_frozen_models


def _pre(name="CorrectedPhysics"):
    return RepresentationPreprocessor(name, np.zeros(1), .0001, .0002)


def test_corrected_center_matches_independent_gev_formula_and_frozen_legacy():
    p = np.linspace(.3, 1., 43)
    thickness = np.resize([.1, .3, 1.], len(p))
    mass, electron, excitation = .13957039, .00051099895069, 173e-9
    energy = np.hypot(p, mass)
    beta, gamma = p/energy, energy/mass
    tmax = 2*electron*beta**2*gamma**2/(1+2*gamma*electron/mass+(electron/mass)**2)
    dE = (.307075/1000)*(14/28.0855)*2.329*(thickness/10)/beta**2*(.5*np.log(2*electron*beta**2*gamma**2*tmax/excitation**2)-beta**2)
    expected = p-np.sqrt(np.maximum(np.maximum(energy-dE, mass)**2-mass**2, 0))
    np.testing.assert_allclose(corrected_pion_bethe_bloch_delta_p(p, thickness), expected, rtol=2e-11, atol=2e-16)
    assert historical_pion_bethe_bloch_delta_p(.65,.3) == pytest.approx(4.439889241203421e-5, abs=1e-16)
    legacy_file = Path(__file__).resolve().parents[1]/"src/models/anchor_legacy.py"
    assert hashlib.sha256(legacy_file.read_bytes()).hexdigest() == "b0e91c465e4e9b2c3bd5b979f28e3b99954c53a2e1e8519d37bf541c1ffbd760"


def test_center_names_preserve_legacy_and_corrected_semantics():
    p, t, q = np.array([.3,.65,1.]), np.array([.1,.3,1.]), np.ones(3)
    np.testing.assert_array_equal(_pre("Physics").center(p,t,q), historical_pion_bethe_bloch_delta_p(p,t,q))
    np.testing.assert_array_equal(_pre("HistoricalIncorrectPhysics").center(p,t,q), _pre("Physics").center(p,t,q))
    np.testing.assert_array_equal(_pre().center(p,t,q), corrected_pion_bethe_bloch_delta_p(p,t,q))
    assert not np.allclose(_pre().center(p,t,q), _pre("Physics").center(p,t,q))


def test_full_preprocessor_roundtrip_and_identity_rejection():
    pre = RepresentationPreprocessor("CorrectedPhysicsScaled", np.array([0.]), .000123, .00042,
        1., np.array([-8., .1, .2, -.1]), -10., -6.)
    restored = RepresentationPreprocessor.from_dict(json.loads(json.dumps(pre.to_dict())))
    p,t,q=np.array([.3,.8]),np.array([.1,1.]),np.array([-1,1])
    np.testing.assert_array_equal(pre.transform(p*.001,p,t,q),restored.transform(p*.001,p,t,q))
    bad=pre.to_dict();bad["center_identity"]["center_id"]="HistoricalIncorrectPhysics"
    with pytest.raises(ValueError,match="identity"):
        RepresentationPreprocessor.from_dict(bad)
    bad=pre.to_dict();bad["scale"]=0
    with pytest.raises(ValueError,match="scale"):
        RepresentationPreprocessor.from_dict(bad)


def test_checkpoint_rejects_legacy_and_source_mutation(tmp_path):
    flow=ConditionalSplineFlow(seed=31)
    path=tmp_path/"model.pt"
    payload={"state_dict":flow.state_dict(),"representation":"Physics"}
    torch.save(payload,path)
    with pytest.raises(ValueError,match="legacy checkpoint rejected"):
        load_checkpoint(path)
    with pytest.raises(ValueError,match="corrected center"):
        load_checkpoint(path,allow_legacy=True,legacy_preprocessor=_pre(),legacy_context_feature_mode="full")
    _, legacy=load_checkpoint(path,allow_legacy=True,legacy_preprocessor=_pre("Physics"),legacy_context_feature_mode="full")
    assert legacy.center_identity["center_id"] == "HistoricalIncorrectPhysics"
    payload={"state_dict":flow.state_dict(),"representation":"CorrectedPhysics",
        "model_config":{"context_dim":5,"hidden":32,"seed":31,"bound":8.},
        "checkpoint_contract":build_checkpoint_contract(_pre(),"full")}
    torch.save(payload,path)
    loaded, pre=load_checkpoint(path)
    assert pre.to_dict()==_pre().to_dict()
    with pytest.raises(ValueError,match="currently imported implementation"):
        load_checkpoint(path,repository_root=tmp_path)
    loaded["checkpoint_contract"]["source_files"]["engine.py"]["sha256"]="0"*64
    loaded["checkpoint_contract"]["sha256"]=contract_digest(loaded["checkpoint_contract"])
    torch.save(loaded,path)
    with pytest.raises(ValueError,match="source identity mismatch"):
        load_checkpoint(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_terms_matches_previous_density_cdf_and_gradients_inside_support(dtype):
    torch.manual_seed(18)
    flow=ConditionalSplineFlow(context_dim=2,hidden=12,seed=71).to(dtype=dtype)
    with torch.no_grad():
        for layer in flow.layers:
            layer.net[-1].weight.normal_(std=.08)
    n=63
    context=torch.randn(n,2,dtype=dtype)
    y=torch.linspace(-1,1,n,dtype=dtype,requires_grad=True)
    center=torch.linspace(-.2,.2,n,dtype=dtype)
    upper=torch.full_like(y,1.5)
    scale=.3; mu=.02
    z=(y-center-mu)/scale; b=(upper-center-mu)/scale
    old_logp=flow.log_q(z,context)-np.log(scale)-flow.log_cdf(b,context)
    old_logcdf=flow.log_cdf(z,context)-flow.log_cdf(b,context)
    # Independent probability subtraction is reliable for these moderate logits.
    old_sf=(torch.exp(flow.log_cdf(b,context))-torch.exp(flow.log_cdf(z,context)))/torch.exp(flow.log_cdf(b,context))
    actual=flow.original_terms(y,center,mu,scale,upper,context)
    tol=2e-5 if dtype==torch.float32 else 1e-11
    torch.testing.assert_close(actual["log_p_y"],old_logp,rtol=tol,atol=tol)
    torch.testing.assert_close(actual["log_cdf_y"],old_logcdf,rtol=tol,atol=tol)
    torch.testing.assert_close(torch.exp(actual["log_sf_y"]),old_sf,rtol=tol,atol=tol)
    params=[y,*flow.parameters()]
    old_grad=torch.autograd.grad(old_logp.sum(),params,retain_graph=True)
    new_grad=torch.autograd.grad(actual["log_p_y"].sum(),params)
    for a,b in zip(old_grad,new_grad):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a,b,rtol=tol*10,atol=tol)


def test_support_boundary_and_extreme_tail_probabilities():
    flow=ConditionalSplineFlow(seed=29).double()
    y=torch.tensor([-torch.inf,-1000.,0.,799.999999,800.,801.,torch.inf],dtype=torch.float64)
    upper=torch.full_like(y,800.)
    terms=flow.original_terms(y,torch.zeros_like(y),0.,1.,upper,torch.zeros((len(y),5),dtype=torch.float64))
    assert torch.isneginf(terms["log_p_y"][-3:]).all()
    assert torch.equal(terms["log_cdf_y"][-3:],torch.zeros(3,dtype=torch.float64))
    assert torch.isneginf(terms["log_sf_y"][-3:]).all()
    assert torch.isneginf(terms["log_cdf_y"][0]) and terms["log_sf_y"][0]==0
    torch.testing.assert_close(torch.exp(terms["log_cdf_y"])+torch.exp(terms["log_sf_y"]),torch.ones_like(y),rtol=1e-12,atol=1e-12)
    # Identity tails give a direct stable log-SF reference even when b>800.
    yy=torch.tensor([899.,899.9999],dtype=torch.float64,requires_grad=True)
    bb=torch.full_like(yy,900.)
    tail=flow.original_terms(yy,torch.zeros_like(yy),0.,1.,bb,torch.zeros((2,5),dtype=torch.float64))
    expected=-torch.nn.functional.softplus(yy)+torch.log(-torch.expm1(yy-bb))
    torch.testing.assert_close(tail["log_sf_y"],expected,rtol=1e-12,atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(tail["log_sf_y"].sum(),yy)[0]).all()
    # Negative-tail truncation previously subtracted indistinguishable SFs.
    yy=torch.tensor([-1001.,-1000.0001],dtype=torch.float64)
    bb=torch.full_like(yy,-1000.)
    lower=flow.original_terms(yy,torch.zeros_like(yy),0.,1.,bb,torch.zeros((2,5),dtype=torch.float64))
    torch.testing.assert_close(torch.exp(lower["log_cdf_y"])+torch.exp(lower["log_sf_y"]),torch.ones_like(yy),rtol=1e-12,atol=1e-12)


def test_loader_rejects_nonforward_support_and_duplicate_uids(tmp_path):
    row={"S1":True,"event_uid":"test:1","p_in_GeV_c":.65,"thickness_mm":.3,"pdg":211,"delta_p_GeV_c":.0001,"exit_face":"forward"}
    (tmp_path/"manifest.json").write_text("{}")
    def write(rows):
        (tmp_path/"events.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    write([row]); arrays,meta=load_source_arrays(tmp_path)
    assert len(arrays["y"])==1 and meta["loader_contract"]["eligibility"]=="forward"
    write([{**row,"exit_face":"backward"}])
    with pytest.raises(ValueError,match="forward-exit"):
        load_source_arrays(tmp_path)
    assert len(load_source_arrays(tmp_path,eligibility="legacy_s1")[0]["y"])==1
    write([{**row,"delta_p_GeV_c":.65}])
    with pytest.raises(ValueError,match="Y < p_in"):
        load_source_arrays(tmp_path)
    write([row,row])
    with pytest.raises(ValueError,match="duplicate"):
        load_source_arrays(tmp_path)


def test_confirmation_preflight_rejects_legacy_without_reading_data(tmp_path):
    formal=tmp_path/"formal";hidden=tmp_path/"hidden"
    formal.mkdir();hidden.mkdir()
    checkpoint=tmp_path/"historical.pt"
    torch.save({"state_dict":ConditionalSplineFlow(seed=3101).state_dict(),
                "representation":"Physics","seed":3101,"selected_update":500},checkpoint)
    row={"run_id":"historical_physics","capacity":"Small","representation":"Physics",
         "sample_fraction":.1,"seed":3101,"checkpoint":str(checkpoint)}
    (formal/"formal_model_manifest.json").write_text(json.dumps({"models":[row]}))
    (hidden/"exploratory_model_manifest.json").write_text(json.dumps({"models":[]}))
    with pytest.raises(ValueError,match="legacy checkpoint rejected"):
        prepare_frozen_models(formal,hidden,"Physics")
    # Even explicit legacy mode cannot use a sidecar for a different artifact.
    record={"historical_physics":{"checkpoint_sha256":"0"*64,"preprocessor":_pre("Physics").to_dict(),"context_feature_mode":"full"}}
    with pytest.raises(ValueError,match="sidecar"):
        prepare_frozen_models(formal,hidden,"Physics",allow_legacy=True,legacy_records=record)
