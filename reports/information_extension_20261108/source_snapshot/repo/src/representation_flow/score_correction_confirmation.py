"""Score frozen core60 or all unique160 models on A/B within hard time gates.

This module never fits, selects models, changes checkpoints, or reads either
confirmation ledger until the complete model/integrity preflight has passed.
Scope is fixed before access using completed experiment identities and a
main-data-only timing forecast. A present extension must be complete or have
an explicit failing-budget deferral. No scientific score chooses the scope.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from representation_flow.analyze_correction_study import CONTRASTS, REPRESENTATIONS, read_json, summarize_values, validate_study, write_csv
from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import ConditionalSplineFlow, context_features, load_source_arrays
from representation_flow.experiment_contract import canonical_hash, data_identity, save_json, sha256, validate_completed

BLOCKS = {
    "a": {"directory":"confirmation_500k_a_full_merged", "eligible_events":499268,
          "manifest_sha256":"b72194770190b646ede8e42782d0df3e758eb606e7cd6d4b36c6068be4fea342",
          "events_sha256":"6684320deffa8157459af51229f5eb7ecd96d5177798fd8cb1562877b90e9d72"},
    "b": {"directory":"confirmation_500k_b_full_merged", "eligible_events":499283,
          "manifest_sha256":"a948d095cf6c73822e15f88fbc3b81de84397906e9c1bf19890928d0b1675db5",
          "events_sha256":"cb1eef3c9d9e1cf39a7b8c86d491d8543910e8f159c5534a6da699014a612189"},
}
PROVENANCE = ("A/B are same-range blocks with disclosed legacy W6/MDN use. They are not globally untouched and not OOD. "
              "All scored checkpoints were fixed before access; no confirmation-based selection, refitting, training, exclusions or budget changes are performed. "
              "All five seed values are retained; shared-data seed SD and paired event SD are descriptive, not independent-replication confidence intervals.")
_BLOCK_ARRAYS = {}


def _no_formal_extension_started(extension_root):
    study=extension_root/"study"
    if (study/"scores.json").exists():
        raise ValueError("cannot defer an extension after a formal execution record has been created")
    if study.exists() and any(study.rglob("checkpoint.pt")):
        raise ValueError("cannot defer an extension containing formal checkpoints")


def freeze_runtime_deferral(extension_root,core_protocol_path,extension_protocol_path):
    """Explicitly close an unstarted extension on a failed measured time gate."""
    _no_formal_extension_started(extension_root)
    destination=extension_root/"EXTENSION_SCOPE_DECISION.json"
    snapshot=extension_root/"extension_protocol.deferred_snapshot.json"
    if destination.exists() or snapshot.exists():
        raise FileExistsError("extension deferral is already frozen; do not overwrite it")
    core=read_json(core_protocol_path);extension=read_json(extension_protocol_path)
    benchmark_path=extension_root/"benchmark/benchmark_result.json";benchmark=read_json(benchmark_path)
    if extension["core_protocol_sha256"]!=canonical_hash(core):
        raise ValueError("extension/core protocols differ")
    if benchmark.get("allowed_combined_under_three_hours") is not False or benchmark["conservative_combined_total_seconds"]<10800:
        raise ValueError("runtime deferral requires a measured failing three-hour gate")
    if benchmark["common_identity"]["protocol_sha256"]!=canonical_hash(extension):
        raise ValueError("runtime benchmark is not bound to the current extension protocol")
    snapshot.write_bytes(Path(extension_protocol_path).read_bytes())
    decision={"schema":"corrected-extension-scope-decision/v1","status":"DEFERRED_BY_RUNTIME_BUDGET",
        "reason_code":"measured_conservative_total_exceeds_10800_seconds","formal_training_started":False,
        "frozen_before_confirmation":True,"core_protocol_sha256":canonical_hash(core),
        "extension_protocol_sha256":canonical_hash(extension),"created_unix":time.time(),
        "scoring_scope_after_deferral":"ALL_CORE60_ONLY","benchmark":{"path":"benchmark/benchmark_result.json","sha256":sha256(benchmark_path)},
        "extension_protocol_snapshot":{"path":snapshot.name,"sha256":sha256(snapshot)}}
    with destination.open("x",encoding="utf-8") as stream:json.dump(decision,stream,indent=2,sort_keys=True)
    return decision


def extension_completion_gate(extension_root, core_study, core_protocol):
    if extension_root is None or not extension_root.exists():
        raise ValueError("explicit complete or runtime-deferred extension evidence is required; missing extension path cannot unlock confirmation")
    decision_path=extension_root/"EXTENSION_SCOPE_DECISION.json"
    if decision_path.exists():
        decision=read_json(decision_path)
        if (decision.get("schema")!="corrected-extension-scope-decision/v1" or decision.get("status")!="DEFERRED_BY_RUNTIME_BUDGET"
                or decision.get("formal_training_started") is not False or decision.get("frozen_before_confirmation") is not True
                or decision.get("core_protocol_sha256")!=canonical_hash(core_protocol)
                or decision.get("scoring_scope_after_deferral")!="ALL_CORE60_ONLY"):
            raise ValueError("invalid extension deferral decision")
        _no_formal_extension_started(extension_root)
        for key in ("benchmark","extension_protocol_snapshot"):
            path=(extension_root/decision[key]["path"]).resolve()
            if not path.is_relative_to(extension_root.resolve()) or sha256(path)!=decision[key]["sha256"]:
                raise ValueError("extension deferral evidence changed")
        benchmark=read_json(extension_root/decision["benchmark"]["path"])
        extension=read_json(extension_root/decision["extension_protocol_snapshot"]["path"])
        if canonical_hash(extension)!=decision["extension_protocol_sha256"] or extension["core_protocol_sha256"]!=canonical_hash(core_protocol):
            raise ValueError("deferred extension protocol identity mismatch")
        if benchmark.get("allowed_combined_under_three_hours") is not False or benchmark["conservative_combined_total_seconds"]<10800:
            raise ValueError("deferral is not supported by a failing measured budget gate")
        if benchmark["common_identity"]["protocol_sha256"]!=canonical_hash(extension):
            raise ValueError("deferral benchmark protocol mismatch")
        return {"status":"DEFERRED_BY_RUNTIME_BUDGET","decision_sha256":sha256(decision_path),"decision":decision,"scored_here":False}
    study=extension_root/"study"
    required=("scores.json","formal_scores_120.json","FROZEN_CHECKPOINT_MANIFEST.json","experiment_identity.json")
    if not all((study/name).is_file() for name in required):
        raise ValueError("planned extension exists but is not complete and frozen; confirmation access denied")
    scores=read_json(study/"scores.json");formal=read_json(study/"formal_scores_120.json");freeze=read_json(study/"FROZEN_CHECKPOINT_MANIFEST.json")
    identity=read_json(study/"experiment_identity.json");definition=identity["identity"]
    if identity["sha256"]!=canonical_hash(definition) or scores["experiment_identity"]!=identity["sha256"]:
        raise ValueError("extension experiment identity mismatch")
    if definition.get("mode")!="execute" or definition.get("max_updates")!=20000:
        raise ValueError("extension benchmark is not a frozen training result")
    core_identity=read_json(core_study/"experiment_identity.json")["identity"]
    common=definition.get("core_common",{})
    for key in ("protocol_sha256","code","data","cache_sha256","environment"):
        if common.get(key)!=core_identity.get(key):
            raise ValueError("extension and core source/data/budget identities differ")
    if scores.get("status")!="COMPLETE" or scores.get("failures") or len(scores.get("runs",[]))!=100:
        raise ValueError("extension has missing or failed cells; confirmation access denied")
    if formal.get("status")!="COMPLETE_CORRECTED_FORMAL_120" or (formal.get("new_cells"),formal.get("reused_cells"))!=(100,20):
        raise ValueError("corrected formal120 is incomplete")
    if freeze.get("schema")!="corrected-formal-checkpoint-freeze-v1" or freeze.get("frozen_before_confirmation") is not True or freeze.get("experiment_identity")!=identity["sha256"]:
        raise ValueError("extension checkpoint freeze is invalid")
    expected={f"corrected_full__{rep}__{capacity}__f{round(frac*100):03d}__sub4101__seed{seed}"
        for rep,capacity,frac,seed in itertools.product(("Raw","Simple","Learned","CorrectedPhysics"),("Small","Large"),(.1,.3,1.),core_protocol["seeds"])}
    rows={row["matrix_cell_id"]:row for row in formal["runs"]};frozen={row["matrix_cell_id"]:row for row in freeze["models"]}
    if len(formal["runs"])!=120 or len(freeze["models"])!=120 or set(rows)!=expected or set(frozen)!=expected:
        raise ValueError("extension matrix is incomplete or duplicated")
    if formal.get("experiment_identity")!=identity["sha256"]:
        raise ValueError("combined extension identity differs")
    new_rows={r["run_id"]:r for r in scores["runs"]}
    if len(new_rows)!=100 or set(new_rows)!={r["run_id"] for r in rows.values() if not r["reused"]}:
        raise ValueError("extension completed new-cell roster differs from the combined matrix")
    if sum(bool(r["reused"]) for r in rows.values())!=20:
        raise ValueError("extension reuse count differs from the fixed plan")
    core_identity_hash=read_json(core_study/"experiment_identity.json")["sha256"]
    for matrix_id,row in rows.items():
        expected_id=f"corrected_full__{row['representation']}__{row['capacity']}__f{round(row['sample_fraction']*100):03d}__sub4101__seed{row['seed']}"
        if matrix_id!=expected_id:
            raise ValueError("extension matrix cell ID disagrees with its scientific labels")
        if frozen[matrix_id]!={key:row[key] for key in frozen[matrix_id]}:
            raise ValueError("extension frozen manifest differs from combined matrix")
        should_reuse=row["capacity"]=="Small" and row["representation"] in ("Raw","CorrectedPhysics") and row["sample_fraction"] in (.1,1.)
        expected_root=core_study.resolve() if should_reuse else study.resolve()
        expected_identity=core_identity_hash if should_reuse else identity["sha256"]
        if row["reused"]!=should_reuse or row["experiment_identity"]!=expected_identity:
            raise ValueError("extension artifact root, reuse or identity mismatch")
        # artifact_root is preserved historical provenance. Actual files resolve
        # relative to the explicitly supplied checkout, including public mirrors.
        validate_completed([row],expected_root,expected_identity,{row["run_id"]})
        checkpoint=expected_root/row["artifacts"]["checkpoint"]["path"]
        payload,pre=load_checkpoint(checkpoint,expected_sha256=row["artifacts"]["checkpoint"]["sha256"])
        if pre.to_dict()!=row["preprocessor"] or payload["experiment_identity"]!=expected_identity:
            raise ValueError("extension checkpoint scientific identity mismatch")
    return {"status":"COMPLETE_FROZEN_EXTENSION120_VERIFIED","experiment_identity":identity["sha256"],"freeze_sha256":sha256(study/"FROZEN_CHECKPOINT_MANIFEST.json"),"scored_here":False}


def preflight_models(study,source_cache,protocol_path,*,merged=None,extension_root=None,scope="auto",
                     extension_cache=None,extension_protocol_path=None):
    protocol=read_json(protocol_path)
    checked=validate_study(study,source_cache,protocol,merged=merged)
    extension=extension_completion_gate(extension_root,study,protocol)
    if extension["status"]=="COMPLETE_FROZEN_EXTENSION120_VERIFIED":
        # A released artifact directory need not contain private source arrays.
        # Validate both caches through the supplied local paths, and replay every
        # extension checkpoint before any confirmation-ledger access.
        from representation_flow.analyze_corrected_formal_matrix import validate_matrix
        if extension_protocol_path is None:
            extension_protocol_path=Path(__file__).resolve().parents[2]/"protocol/corrected_formal_extension_20261107.json"
        validated=validate_matrix(study.parent,extension_root,protocol_path,extension_protocol_path,
                                  merged=merged,core_cache=source_cache,extension_cache=extension_cache)
        if validated["experiment_identity"]!=extension["experiment_identity"]:
            raise ValueError("extension validation identity differs from its frozen roster")
        resolved_extension_cache=Path(extension_cache) if extension_cache is not None else extension_root/"cache"
        extension.update(validation_status="ALL_120_CELLS_REPLAYED_AND_PREPROCESSING_VERIFIED",
                         preprocessing_sha256=canonical_hash(read_json(resolved_extension_cache/"preprocessing_manifest.json")),
                         extension_protocol_sha256=canonical_hash(read_json(extension_protocol_path)))
    rows={row["run_id"]:row for row in read_json(study/"scores.json")["runs"]}
    models=[]
    for cell in checked["cells"]:
        row=rows[cell["run_id"]]
        models.append({"run_id":cell["run_id"],"context":cell["context"],"capacity":"Small","fraction_tag":cell["fraction_tag"],"seed":cell["seed"],"representation":cell["representation"],
            "checkpoint":str((study/row["artifacts"]["checkpoint"]["path"]).resolve()),"checkpoint_sha256":row["artifacts"]["checkpoint"]["sha256"],
            "checkpoint_contract_sha256":row["checkpoint_contract"]["sha256"]})
    full=scope=="all160" or (scope=="auto" and extension["status"]=="COMPLETE_FROZEN_EXTENSION120_VERIFIED")
    if scope not in ("auto","core60","all160"):raise ValueError("unknown prespecified scoring scope")
    if full and extension["status"]!="COMPLETE_FROZEN_EXTENSION120_VERIFIED":
        raise ValueError("all160 scoring requires the complete frozen extension120")
    mapping=[]
    if full:
        ext_study=extension_root/"study"
        for row in read_json(ext_study/"formal_scores_120.json")["runs"]:
            mapping.append({"matrix_cell_id":row["matrix_cell_id"],"run_id":row["run_id"],"reused":row["reused"],"checkpoint_sha256":row["artifacts"]["checkpoint"]["sha256"]})
            if row["reused"]:continue
            models.append({"run_id":row["run_id"],"context":"Full","capacity":row["capacity"],"fraction_tag":f"f{round(row['sample_fraction']*100):03d}",
                "seed":row["seed"],"representation":row["representation"],"checkpoint":str((ext_study/row["artifacts"]["checkpoint"]["path"]).resolve()),
                "checkpoint_sha256":row["artifacts"]["checkpoint"]["sha256"],"checkpoint_contract_sha256":row["checkpoint_contract"]["sha256"]})
        if len(models)!=160 or len({m["run_id"] for m in models})!=160 or len(mapping)!=120 or sum(m["reused"] for m in mapping)!=20:
            raise ValueError("unique160 scoring roster/reuse mapping is invalid")
    scores_record=read_json(study/"scores.json")
    if not isinstance(scores_record.get("execution_started_unix"),(float,int)):
        raise ValueError("core execution start is required to enforce the total three-hour deadline")
    return {"schema":"frozen-core60-full120-confirmation/v2","status":"ALL_MODELS_VALIDATED_BEFORE_CONFIRMATION_ACCESS","scope":"ALL_UNIQUE_CORE60_PLUS_EXTENSION100" if full else "ALL_CORE60_ONLY","models":models,"model_count":len(models),
            "all160_contrast_definition_sha256":canonical_hash(full160_contrast_specs()),
            "source_cache":str(source_cache.resolve()),
            "extension_cache":str((Path(extension_cache) if extension_cache is not None else extension_root/"cache").resolve()),
            "formal120_to_unique_models":mapping,
            "core_execution_started_unix":scores_record["execution_started_unix"],"total_budget_seconds":protocol["runtime_policy"]["max_total_seconds"],
            "core_experiment_identity":checked["experiment_identity"],"core_freeze_sha256":sha256(study/"FROZEN_CHECKPOINT_MANIFEST.json"),
            "extension_gate":{**extension,"scored_here":full},"blocks":BLOCKS,"protocol_sha256":canonical_hash(protocol),"provenance":PROVENANCE}


def fixed_group_ids(p,t,q):
    edges=np.linspace(.3,1.,8)
    if not np.isfinite(p).all() or np.any((p<.3)|(p>1.)) or not np.isin(t,[.1,.3,1.]).all() or not np.isin(q,[-1.,1.]).all():
        raise ValueError("confirmation inputs violate the fixed same-range 42-stratum design")
    pb=np.minimum(np.searchsorted(edges,p,side="right")-1,6)
    tb=np.searchsorted(np.array([.1,.3,1.]),t)
    qb=(q==1.).astype(int)
    return (pb*6+tb*2+qb).astype(np.int64)


def prepare_confirmation_cache(data_root,cache_root,block,expected,deadline):
    if time.time()>=deadline:raise TimeoutError("confirmation deadline reached before data access")
    source=data_root/expected["directory"]
    actual=data_identity(source)
    if actual!={k:expected[k] for k in ("manifest_sha256","events_sha256")}:
        raise ValueError(f"confirmation {block} data hash mismatch")
    arrays,manifest=load_source_arrays(source,validation_fraction=.1,diagnostic_fraction=.3,eligibility="forward")
    if len(arrays["y"])!=expected["eligible_events"]:
        raise ValueError("confirmation eligible-event count differs from the preserved data audit")
    root=cache_root/block;root.mkdir(parents=True,exist_ok=False)
    for name in ("p_in","thickness","charge","y"):
        np.save(root/f"{name}.npy",arrays[name],allow_pickle=False)
    np.save(root/"group_id.npy",fixed_group_ids(arrays["p_in"],arrays["thickness"],arrays["charge"]),allow_pickle=False)
    record={"block":block,"source":str(source.resolve()),"source_identity":actual,"eligible_events":len(arrays["y"]),
            "ordered_uid_sha256":canonical_hash(arrays["event_uid"].tolist()),"all_eligible_events_scored":True,"no_selection_or_resampling":True,
            "files":[{"name":p.name,"sha256":sha256(p)} for p in sorted(root.glob("*.npy"))]}
    save_json(root/"cache_manifest.json",record)
    return record


def initialize_worker(cache_root):
    global _BLOCK_ARRAYS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _BLOCK_ARRAYS={}
    for block in ("a","b"):
        root=Path(cache_root)/block;record=read_json(root/"cache_manifest.json")
        arrays={}
        for file in record["files"]:
            path=root/file["name"]
            if sha256(path)!=file["sha256"]:raise ValueError("read-only confirmation cache changed")
            arrays[path.stem]=np.load(path,mmap_mode="r",allow_pickle=False)
        _BLOCK_ARRAYS[block]=arrays


def _moments_finish(count,total,squares):
    if count==0:return {"events":0,"mean":None,"event_sd_descriptive":None}
    mean=float(total/count)
    sd=float(np.sqrt(max(0.,(squares-total*total/count)/(count-1)))) if count>1 else 0.
    return {"events":int(count),"mean":mean,"event_sd_descriptive":sd}


def score_group(job):
    """Score all six Full/Hidden x center models together for one fraction/seed."""
    arrays=_BLOCK_ARRAYS[job["block"]];n=len(arrays["y"])
    if len(job["models"])!=6:raise ValueError("confirmation group must contain all six paired models")
    loaded={}
    for record in job["models"]:
        payload,pre=load_checkpoint(record["checkpoint"],expected_sha256=record["checkpoint_sha256"])
        if payload["checkpoint_contract"]["sha256"]!=record["checkpoint_contract_sha256"] or pre.name!=record["representation"]:
            raise ValueError("checkpoint changed after confirmation preflight")
        flow=ConditionalSplineFlow(**payload["model_config"]);flow.load_state_dict(payload["state_dict"]);flow.eval()
        loaded[(record["context"],record["representation"])]=(flow,pre,payload["checkpoint_contract"]["context_feature_mode"])
    expected=set(itertools.product(("Full","Hidden"),REPRESENTATIONS))
    if set(loaded)!=expected:raise ValueError("paired confirmation models are duplicated or incomplete")
    model_sums={key:np.zeros((4,42)) for key in loaded}
    contrast_sums={(context,f"{left}_minus_{right}"):np.zeros((2,42)) for context in ("Full","Hidden","Full_minus_Hidden") for left,right in CONTRASTS}
    counts=np.zeros(42,dtype=np.int64)
    with torch.no_grad():
        for start in range(0,n,job["batch_size"]):
            if time.time()>=job["deadline"]:raise TimeoutError("confirmation scoring deadline reached; partial results remain labelled incomplete")
            sl=slice(start,min(n,start+job["batch_size"]))
            p,t,q,y=(np.asarray(arrays[k][sl]) for k in ("p_in","thickness","charge","y"))
            group=np.asarray(arrays["group_id"][sl]);counts+=np.bincount(group,minlength=42)
            logp={}
            contexts={mode:torch.as_tensor(context_features(p,t,q,mode),dtype=torch.float32) for mode in ("full","thickness_charge")}
            yy=torch.as_tensor(y.copy(),dtype=torch.float32);pp=torch.as_tensor(p.copy(),dtype=torch.float32)
            for key,(flow,pre,mode) in loaded.items():
                scale=torch.as_tensor(pre.scale_for(p,t,q),dtype=torch.float32)
                terms=flow.original_terms(yy,torch.as_tensor(pre.center(p,t,q),dtype=torch.float32),pre.mu,scale,pp,contexts[mode])
                lp=terms["log_p_y"].numpy().astype(np.float64)
                if not np.isfinite(lp).all():raise FloatingPointError("nonfinite frozen confirmation log likelihood")
                logp[key]=lp
                values=(-lp,(-lp)**2,torch.log(scale).numpy().astype(np.float64),terms["log_f_b"].numpy().astype(np.float64))
                for i,value in enumerate(values):model_sums[key][i]+=np.bincount(group,weights=value,minlength=42)
            for left,right in CONTRASTS:
                label=f"{left}_minus_{right}"
                full=logp[("Full",right)]-logp[("Full",left)]
                hidden=logp[("Hidden",right)]-logp[("Hidden",left)]
                for context,diff in (("Full",full),("Hidden",hidden),("Full_minus_Hidden",full-hidden)):
                    contrast_sums[(context,label)][0]+=np.bincount(group,weights=diff,minlength=42)
                    contrast_sums[(context,label)][1]+=np.bincount(group,weights=diff**2,minlength=42)
    prefix={"block":job["block"],"fraction_tag":job["fraction_tag"],"seed":job["seed"]}
    models=[];model_groups=[];contrasts=[];contrast_groups=[]
    for (context,rep),sums in model_sums.items():
        row={**prefix,"context":context,"representation":rep,**_moments_finish(n,sums[0].sum(),sums[1].sum()),
             "mean_log_scale":float(sums[2].sum()/n),"mean_log_F_b":float(sums[3].sum()/n)}
        models.append(row)
        for group in range(42):model_groups.append({**prefix,"context":context,"representation":rep,"group_id":group,**_moments_finish(counts[group],sums[0,group],sums[1,group])})
    for (context,label),sums in contrast_sums.items():
        contrasts.append({**prefix,"context":context,"contrast":label,**_moments_finish(n,sums[0].sum(),sums[1].sum())})
        for group in range(42):contrast_groups.append({**prefix,"context":context,"contrast":label,"group_id":group,**_moments_finish(counts[group],sums[0,group],sums[1,group])})
    return {"job_id":job["job_id"],"models":models,"contrasts":contrasts,"model_groups":model_groups,"contrast_groups":contrast_groups}


def full160_contrast_specs():
    """Predeclared linear log-density contrasts; no scores influence this list."""
    specs=[]
    def terms(tag,capacity,context,left,right):
        return [(1.,(tag,capacity,context,right)),(-1.,(tag,capacity,context,left))]
    def add(scope,tag,capacity,context,label,weights):
        specs.append({"scope":scope,"fraction_tag":tag,"capacity":capacity,"context":context,"contrast":label,"terms":weights})
    for tag,(left,right) in itertools.product(("f010","f100"),CONTRASTS):
        label=f"{left}_minus_{right}"
        full=terms(tag,"Small","Full",left,right);hidden=terms(tag,"Small","Hidden",left,right)
        add("core60",tag,"Small","Full",label,full);add("core60",tag,"Small","Hidden",label,hidden)
        add("core60",tag,"Small","Full_minus_Hidden",label,full+[(-w,k) for w,k in hidden])
    formal_pairs=(("Raw","CorrectedPhysics"),("Learned","CorrectedPhysics"),("Simple","Raw"))
    for tag,capacity,(left,right) in itertools.product(("f010","f030","f100"),("Small","Large"),formal_pairs):
        add("formal120",tag,capacity,"Full",f"{left}_minus_{right}",terms(tag,capacity,"Full",left,right))
    for tag in ("f010","f030","f100"):
        small=terms(tag,"Small","Full","Raw","CorrectedPhysics");large=terms(tag,"Large","Full","Raw","CorrectedPhysics")
        add("formal120",tag,"Small_minus_Large","Full","Raw_minus_CorrectedPhysics",small+[(-w,k) for w,k in large])
    for capacity in ("Small","Large"):
        low=terms("f010",capacity,"Full","Raw","CorrectedPhysics");high=terms("f100",capacity,"Full","Raw","CorrectedPhysics")
        add("formal120","f010_minus_f100",capacity,"Full","Raw_minus_CorrectedPhysics",low+[(-w,k) for w,k in high])
    return specs


def score_full_group(job):
    """One block/seed, 32 unique models; reused core/formal models run once."""
    arrays=_BLOCK_ARRAYS[job["block"]];n=len(arrays["y"])
    loaded={}
    for record in job["models"]:
        key=(record["fraction_tag"],record["capacity"],record["context"],record["representation"])
        if key in loaded:raise ValueError("duplicate unique-model key in full confirmation group")
        payload,pre=load_checkpoint(record["checkpoint"],expected_sha256=record["checkpoint_sha256"])
        if payload["checkpoint_contract"]["sha256"]!=record["checkpoint_contract_sha256"] or pre.name!=record["representation"]:
            raise ValueError("checkpoint changed after full confirmation preflight")
        flow=ConditionalSplineFlow(**payload["model_config"]);flow.load_state_dict(payload["state_dict"]);flow.eval()
        loaded[key]=(flow,pre,payload["checkpoint_contract"]["context_feature_mode"],record)
    expected={(tag,capacity,"Full",rep) for tag,capacity,rep in itertools.product(("f010","f030","f100"),("Small","Large"),("Raw","Simple","Learned","CorrectedPhysics"))}
    expected|={(tag,"Small","Full","HistoricalIncorrectPhysics") for tag in ("f010","f100")}
    expected|={(tag,"Small","Hidden",rep) for tag,rep in itertools.product(("f010","f100"),REPRESENTATIONS)}
    if len(loaded)!=32 or set(loaded)!=expected:raise ValueError("all160 confirmation group is not the prescribed unique32 models")
    specs=full160_contrast_specs()
    model_sums={key:np.zeros((4,42)) for key in loaded};contrast_sums=[np.zeros((2,42)) for _ in specs]
    counts=np.zeros(42,dtype=np.int64)
    with torch.no_grad():
        for start in range(0,n,job["batch_size"]):
            if time.time()>=job["deadline"]:raise TimeoutError("all160 confirmation deadline reached")
            sl=slice(start,min(n,start+job["batch_size"]));p,t,q,y=(np.asarray(arrays[k][sl]) for k in ("p_in","thickness","charge","y"))
            group=np.asarray(arrays["group_id"][sl]);counts+=np.bincount(group,minlength=42)
            context_arrays={mode:torch.as_tensor(context_features(p,t,q,mode),dtype=torch.float32) for mode in ("full","thickness_charge")}
            yy=torch.as_tensor(y.copy(),dtype=torch.float32);pp=torch.as_tensor(p.copy(),dtype=torch.float32);logp={}
            for key,(flow,pre,mode,_) in loaded.items():
                scale=torch.as_tensor(pre.scale_for(p,t,q),dtype=torch.float32)
                values=flow.original_terms(yy,torch.as_tensor(pre.center(p,t,q),dtype=torch.float32),pre.mu,scale,pp,context_arrays[mode])
                lp=values["log_p_y"].numpy().astype(np.float64)
                if not np.isfinite(lp).all():raise FloatingPointError("nonfinite all160 frozen log likelihood")
                logp[key]=lp
                for i,v in enumerate((-lp,lp**2,torch.log(scale).numpy().astype(np.float64),values["log_f_b"].numpy().astype(np.float64))):
                    model_sums[key][i]+=np.bincount(group,weights=v,minlength=42)
            for spec,sums in zip(specs,contrast_sums):
                diff=sum(weight*logp[key] for weight,key in spec["terms"])
                sums[0]+=np.bincount(group,weights=diff,minlength=42);sums[1]+=np.bincount(group,weights=diff**2,minlength=42)
    prefix={"block":job["block"],"seed":job["seed"]};models=[];model_groups=[];contrasts=[];contrast_groups=[]
    for key,sums in model_sums.items():
        tag,capacity,context,rep=key;record=loaded[key][3]
        labels={**prefix,"run_id":record["run_id"],"fraction_tag":tag,"capacity":capacity,"context":context,"representation":rep}
        models.append({**labels,**_moments_finish(n,sums[0].sum(),sums[1].sum()),"mean_log_scale":float(sums[2].sum()/n),"mean_log_F_b":float(sums[3].sum()/n)})
        for group in range(42):model_groups.append({**labels,"group_id":group,**_moments_finish(counts[group],sums[0,group],sums[1,group])})
    for spec,sums in zip(specs,contrast_sums):
        labels={**prefix,**{k:v for k,v in spec.items() if k!="terms"}}
        contrasts.append({**labels,**_moments_finish(n,sums[0].sum(),sums[1].sum())})
        for group in range(42):contrast_groups.append({**labels,"group_id":group,**_moments_finish(counts[group],sums[0,group],sums[1,group])})
    return {"job_id":job["job_id"],"models":models,"contrasts":contrasts,"model_groups":model_groups,"contrast_groups":contrast_groups}


def aggregate_full_confirmation(jobs):
    expected={(b,s) for b,s in itertools.product(("a","b"),range(3101,3106))}
    keys=[];values={"models":[],"seed_contrasts":[],"model_groups":[],"contrast_groups":[]}
    for job in jobs:
        first=job["models"][0];keys.append((first["block"],first["seed"]))
        for out,source in (("models","models"),("seed_contrasts","contrasts"),("model_groups","model_groups"),("contrast_groups","contrast_groups")):values[out].extend(job[source])
    if len(keys)!=10 or set(keys)!=expected or len(values["models"])!=320 or len({(r["block"],r["run_id"]) for r in values["models"]})!=320:
        raise ValueError("all160 confirmation is incomplete or repeats a reused model")
    summary=[]
    for block,spec in itertools.product(("a","b"),full160_contrast_specs()):
        labels={k:v for k,v in spec.items() if k!="terms"}
        rows=sorted((r for r in values["seed_contrasts"] if r["block"]==block and all(r[k]==v for k,v in labels.items())),key=lambda r:r["seed"])
        if [r["seed"] for r in rows]!=list(range(3101,3106)):raise ValueError("full confirmation seed contrasts are incomplete")
        summary.append({"block":block,**labels,**summarize_values([r["mean"] for r in rows])})
    return {**values,"contrast_summary":summary}


def benchmark_confirmation_cost(source_cache,output_path,*,sample_events=32768):
    """Bounded main-data-only timing; random frozen weights, no scientific scores."""
    benchmark_started=time.perf_counter()
    if output_path.exists():raise FileExistsError("timing report already exists")
    torch.set_num_threads(1)
    arrays={name:np.load(source_cache/f"{name}.npy",mmap_mode="r",allow_pickle=False)[:sample_events] for name in ("p_in","thickness","charge","y")}
    p,t,q,y=(arrays[k] for k in ("p_in","thickness","charge","y"));n=len(y)
    from representation_flow.engine import RepresentationPreprocessor
    pre=RepresentationPreprocessor("CorrectedPhysics",np.zeros(1),0.,float(np.std(y)))
    pre.mu=float(np.mean(y-pre.center(p,t,q)))
    measurements={}
    for label,mode,width in (("SmallFull","full",32),("SmallHidden","thickness_charge",32),("LargeFull","full",128)):
        flow=ConditionalSplineFlow(context_dim=5 if mode=="full" else 2,hidden=width,seed=12345);flow.eval()
        times=[]
        for repeat in range(3):
            started=time.perf_counter()
            with torch.no_grad():
                for start in range(0,n,8192):
                    sl=slice(start,min(n,start+8192));pp,tt,qq,yy=(np.asarray(arrays[k][sl]) for k in ("p_in","thickness","charge","y"))
                    flow.original_terms(torch.as_tensor(yy.copy(),dtype=torch.float32),torch.as_tensor(pre.center(pp,tt,qq),dtype=torch.float32),pre.mu,
                        torch.as_tensor(pre.scale_for(pp,tt,qq),dtype=torch.float32),torch.as_tensor(pp.copy(),dtype=torch.float32),
                        torch.as_tensor(context_features(pp,tt,qq,mode),dtype=torch.float32))
            if repeat:times.append(time.perf_counter()-started)
        measurements[label]={"eligible_events":n,"repeat_seconds":times,"conservative_seconds_per_event":max(times)/n}
    total_events=sum(b["eligible_events"] for b in BLOCKS.values())
    forecasts={}
    for scope,counts,imbalance in (("ALL_CORE60_ONLY",{"SmallFull":30,"SmallHidden":30,"LargeFull":0},1.),
                                    ("ALL_UNIQUE_CORE60_PLUS_EXTENSION100",{"SmallFull":70,"SmallHidden":30,"LargeFull":60},1.2)):
        serial=sum(measurements[k]["conservative_seconds_per_event"]*count*total_events for k,count in counts.items())
        forecasts[scope]={"unique_models":sum(counts.values()),"model_types":counts,"forecast_parallel_inference_seconds":serial/4*imbalance,
                          "conservative_total_seconds":serial/4*imbalance*1.5+150.,"safety_factor":1.5,"fixed_preflight_loading_output_allowance_seconds":150.}
    result={"schema":"confirmation-main-data-timing/v1","source":"main source cache only; no confirmation data accessed; initialized weights only",
            "scoring_source_sha256":sha256(Path(__file__)),"source_cache_manifest_sha256":sha256(source_cache/"cache_manifest.json"),
            "torch_version":str(torch.__version__),"numpy_version":np.__version__,"worker_threads":1,"planned_workers":4,"measurements":measurements,"forecasts":forecasts,
            "elapsed_seconds":time.perf_counter()-benchmark_started}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,sort_keys=True)
    return result


def select_timed_scope(plan,timing,remaining_seconds,requested_scope):
    forecast=timing["forecasts"][plan["scope"]]["conservative_total_seconds"]
    if forecast>=remaining_seconds and plan["scope"]=="ALL_UNIQUE_CORE60_PLUS_EXTENSION100" and requested_scope=="auto":
        plan={**plan,"scope":"ALL_CORE60_ONLY","models":plan["models"][:60],"model_count":60,
              "extension_gate":{**plan["extension_gate"],"scored_here":False},
              "full160_deferred_before_data_access":{"reason":"timing forecast exceeds remaining confirmation budget","forecast_seconds":forecast,"remaining_seconds":remaining_seconds},
              "formal120_confirmation":"not scored; Full120 validation results remain descriptive"}
        forecast=timing["forecasts"][plan["scope"]]["conservative_total_seconds"]
    if forecast>=remaining_seconds:raise TimeoutError("measured conservative confirmation forecast exceeds remaining hard budget before data access")
    return {**plan,"timing_forecast_seconds":forecast}


def aggregate_confirmation(jobs):
    expected={(b,f,s) for b,f,s in itertools.product(("a","b"),("f010","f100"),range(3101,3106))}
    keys=[];models=[];contrasts=[];model_groups=[];contrast_groups=[]
    for job in jobs:
        first=job["models"][0];keys.append((first["block"],first["fraction_tag"],first["seed"]))
        models.extend(job["models"]);contrasts.extend(job["contrasts"]);model_groups.extend(job["model_groups"]);contrast_groups.extend(job["contrast_groups"])
    if len(keys)!=20 or set(keys)!=expected:raise ValueError("confirmation is incomplete or has duplicated paired groups")
    summary=[]
    for block,tag,context,(left,right) in itertools.product(("a","b"),("f010","f100"),("Full","Hidden","Full_minus_Hidden"),CONTRASTS):
        label=f"{left}_minus_{right}"
        rows=sorted((r for r in contrasts if (r["block"],r["fraction_tag"],r["context"],r["contrast"])==(block,tag,context,label)),key=lambda r:r["seed"])
        if [r["seed"] for r in rows]!=list(range(3101,3106)):raise ValueError("confirmation seed contrast is incomplete")
        summary.append({"block":block,"fraction_tag":tag,"context":context,"contrast":label,**summarize_values([r["mean"] for r in rows])})
    return {"models":models,"seed_contrasts":contrasts,"contrast_summary":summary,"model_groups":model_groups,"contrast_groups":contrast_groups}


def score_confirmation(study,source_cache,protocol_path,data_root,output,*,merged=None,extension_root=None,max_seconds=1200,batch_size=8192,
                       scope="auto",deadline_unix=None,timing_report=None,extension_cache=None,extension_protocol_path=None):
    started=time.time()
    if deadline_unix is None or not math.isfinite(deadline_unix):raise ValueError("an absolute total-study deadline including all benchmark time is required")
    deadline=min(started+max_seconds,deadline_unix)
    if time.time()>=deadline:raise TimeoutError("total study deadline already expired before confirmation preflight")
    if output.exists():raise FileExistsError("confirmation output already exists; choose a new directory")
    if not 0<max_seconds<=1200 or batch_size<=0:raise ValueError("invalid confirmation time/batch budget")
    # Deliberately before even hashing a confirmation ledger.
    plan=preflight_models(study,source_cache,protocol_path,merged=merged,extension_root=extension_root,scope=scope,
                          extension_cache=extension_cache,extension_protocol_path=extension_protocol_path)
    if deadline_unix>plan["core_execution_started_unix"]+plan["total_budget_seconds"]:
        raise ValueError("absolute confirmation deadline would reset the total study budget")
    if timing_report is None:raise ValueError("a main-data-only timing report is required before confirmation access")
    timing=read_json(timing_report)
    if (timing.get("schema")!="confirmation-main-data-timing/v1" or timing.get("scoring_source_sha256")!=sha256(Path(__file__))
            or timing.get("source_cache_manifest_sha256")!=sha256(source_cache/"cache_manifest.json")
            or timing.get("torch_version")!=str(torch.__version__) or timing.get("numpy_version")!=np.__version__
            or timing.get("worker_threads")!=1 or timing.get("planned_workers")!=4):
        raise ValueError("confirmation timing report does not match current code/cache/runtime")
    plan=select_timed_scope(plan,timing,deadline-time.time(),scope)
    plan.update(scoring_source_sha256=sha256(Path(__file__)),analysis_validation_source_sha256=sha256(Path(__file__).with_name("analyze_correction_study.py")),
                extension_validation_source_sha256=sha256(Path(__file__).with_name("analyze_corrected_formal_matrix.py")),
                workers=4,torch_threads_per_worker=1,batch_size=batch_size,preflight_finished_unix=time.time(),max_seconds=max_seconds,
                absolute_total_study_deadline=deadline_unix,scoring_deadline=deadline,timing_report_sha256=sha256(timing_report))
    output.mkdir(parents=True,exist_ok=False);save_json(output/"SCORING_PLAN.json",plan)
    state={"status":"PREPARING_CONFIRMATION_CACHE","scope":plan["scope"],"completed_jobs":[],"failures":[],"scoring_plan_sha256":sha256(output/"SCORING_PLAN.json")}
    save_json(output/"execution_state.json",state)
    cache=output/"confirmation_cache";records={}
    try:
        for block,expected in BLOCKS.items():records[block]=prepare_confirmation_cache(data_root,cache,block,expected,deadline)
    except Exception as exc:
        state.update(status="INCOMPLETE",elapsed_seconds=time.time()-started)
        state["failures"].append({"stage":"confirmation_cache","error_type":type(exc).__name__,"error":str(exc)})
        save_json(output/"execution_state.json",state)
        raise
    jobs=[];full=plan["scope"]=="ALL_UNIQUE_CORE60_PLUS_EXTENSION100"
    if full:
        for block,seed in itertools.product(("a","b"),range(3101,3106)):
            models=[m for m in plan["models"] if m["seed"]==seed]
            jobs.append({"job_id":f"{block}__unique32__seed{seed}","block":block,"seed":seed,"models":models,"batch_size":batch_size,"deadline":deadline})
    else:
        for block,tag,seed in itertools.product(("a","b"),("f010","f100"),range(3101,3106)):
            models=[m for m in plan["models"] if m["fraction_tag"]==tag and m["seed"]==seed]
            jobs.append({"job_id":f"{block}__{tag}__seed{seed}","block":block,"fraction_tag":tag,"seed":seed,"models":models,"batch_size":batch_size,"deadline":deadline})
    state["status"]="RUNNING"
    save_json(output/"execution_state.json",state);results=[]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker,initargs=(str(cache.resolve()),)) as executor:
        futures={executor.submit(score_full_group if full else score_group,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                result=future.result();save_json(output/"paired_groups"/f"{job['job_id']}.json",result)
                results.append(result);state["completed_jobs"].append(job["job_id"])
                print(json.dumps({"scored":job["job_id"],"elapsed_seconds":time.time()-started}),flush=True)
            except Exception as exc:
                state["failures"].append({"job_id":job["job_id"],"error_type":type(exc).__name__,"error":str(exc)})
                for pending in futures:pending.cancel()
            save_json(output/"execution_state.json",state)
    state["status"]="COMPLETE" if len(results)==len(jobs) and not state["failures"] else "INCOMPLETE"
    state["elapsed_seconds"]=time.time()-started;save_json(output/"execution_state.json",state)
    if state["status"]!="COMPLETE":raise RuntimeError("confirmation incomplete; all successes/failures remain recorded")
    values=aggregate_full_confirmation(results) if full else aggregate_confirmation(results)
    summary={"schema":"frozen-core60-full120-confirmation-results/v2","status":"COMPLETE_FROZEN_UNIQUE160_AB_CONFIRMATION" if full else "COMPLETE_FROZEN_CORE60_AB_CONFIRMATION","scope":plan["scope"],"model_evaluations":plan["model_count"]*2,"blocks":records,
             "provenance":PROVENANCE,"core_experiment_identity":plan["core_experiment_identity"],"extension_gate":plan["extension_gate"],
             "formal120_to_unique_models":plan["formal120_to_unique_models"],
             "scoring_plan_sha256":state["scoring_plan_sha256"],"elapsed_seconds":state["elapsed_seconds"],**values}
    for key in ("models","seed_contrasts","contrast_summary","model_groups","contrast_groups"):write_csv(output/f"{key}.csv",values[key])
    lines=[f"冻结{plan['model_count']}个唯一模型：A/B同范围确认评分", "", f"所有模型均在访问A/B数据前完成身份核验；全部{plan['model_count']*2}次模型-数据块评分保留，未按分数选择模型。",
           "A/B有历史W6/MDN使用记录，不能称全球未触碰测试集或OOD。扩展若存在必须完整冻结或有预算失败的正式DEFERRED决策；评分范围在首次读取A/B之前按完成情况及主数据计时冻结。",
           "Full120的20个复用cell只映射到原核心权重，不重复推理。" if full else "本次仅确认核心60；Full120若有验证结果，仅保持validation描述性结论。", "", "差值为NLL(left)-NLL(right)，正值表示right更低；interaction为同seed的预设差中差。SD仅描述共享数据上的seed变化。"]
    for row in values["contrast_summary"]:lines.append(f"{row['block']} {row.get('scope','core60')} {row['fraction_tag']} {row.get('capacity','Small')} {row['context']} {row['contrast']}: mean={row['mean']:+.9f}, sampleSD={row['sample_sd']:.9f}, range=[{row['minimum']:+.9f},{row['maximum']:+.9f}]")
    lines += ["",PROVENANCE,"固定分箱group_id = momentum_bin*6 + thickness_index*2 + charge_index；动量7等宽箱[0.3,1.0]，厚度[0.1,0.3,1.0]mm，电荷[-1,+1]。未保存重复逐事件预测；分块累积全部事件的模型分数、配对差值及42组统计。"]
    (output/"SCIENTIFIC_CONFIRMATION.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Output construction is part of the same entry-to-exit budget, rather than
    # reporting only worker time as a successful completion.
    state["elapsed_seconds"]=summary["elapsed_seconds"]=time.time()-started
    if time.time()>=deadline:
        state["status"]=summary["status"]="INCOMPLETE_DEADLINE_EXCEEDED"
        state["failures"].append({"stage":"result_output","error_type":"TimeoutError","error":"total scoring deadline reached during result serialization"})
        save_json(output/"execution_state.json",state);save_json(output/"confirmation_summary.json",summary)
        with (output/"SCIENTIFIC_CONFIRMATION.txt").open("a",encoding="utf-8") as stream:
            stream.write("\nINCOMPLETE_DEADLINE_EXCEEDED: outputs are preserved for audit, not accepted as completed confirmation.\n")
        raise TimeoutError("confirmation output exceeded the hard entry-to-exit budget; incomplete artifacts retained")
    save_json(output/"execution_state.json",state);save_json(output/"confirmation_summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path("reports/correction_v2_20261107/study"))
    parser.add_argument("--source-cache",type=Path,default=Path("reports/correction_v2_20261107/cache"))
    parser.add_argument("--protocol",type=Path,default=Path("protocol/correction_v2_20261107.json"))
    parser.add_argument("--extension-root",type=Path,default=Path("reports/corrected_formal_extension_20261107"))
    parser.add_argument("--extension-cache",type=Path,help="private rebuilt extension preprocessing cache; defaults to extension-root/cache")
    parser.add_argument("--main",type=Path,default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--data-root",type=Path,default=Path("data"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-seconds",type=int,default=1200)
    parser.add_argument("--batch-size",type=int,default=8192)
    parser.add_argument("--scope",choices=("auto","core60","all160"),default="auto")
    parser.add_argument("--deadline-unix",type=float,help="absolute core-start + 10800 minus all benchmark wall times; never reset")
    parser.add_argument("--timing-report",type=Path)
    parser.add_argument("--timing-only",action="store_true",help="write a main-data-only cost forecast JSON to --output; never reads A/B")
    parser.add_argument("--freeze-runtime-deferral",action="store_true",help="freeze an explicit failed-budget decision for an unstarted extension; never reads A/B")
    parser.add_argument("--extension-protocol",type=Path,default=Path("protocol/corrected_formal_extension_20261107.json"))
    args=parser.parse_args()
    if args.freeze_runtime_deferral:
        if args.output.resolve()!=(args.extension_root/"EXTENSION_SCOPE_DECISION.json").resolve():
            parser.error("deferral --output must be the extension root's EXTENSION_SCOPE_DECISION.json")
        result=freeze_runtime_deferral(args.extension_root,args.protocol,args.extension_protocol)
        print(json.dumps(result));return
    if args.timing_only:
        result=benchmark_confirmation_cost(args.source_cache,args.output)
        print(json.dumps(result));return
    if args.deadline_unix is None or args.timing_report is None:
        parser.error("scoring requires --deadline-unix and --timing-report")
    result=score_confirmation(args.study,args.source_cache,args.protocol,args.data_root,args.output,merged=args.main,
        extension_root=args.extension_root,extension_cache=args.extension_cache,extension_protocol_path=args.extension_protocol,
        max_seconds=args.max_seconds,batch_size=args.batch_size,scope=args.scope,deadline_unix=args.deadline_unix,timing_report=args.timing_report)
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"],"output":str(args.output)}))


if __name__=="__main__":main()
