"""Audit and describe the complete correction matrix; never fit or read confirmation.

Validation selected these checkpoints. Seed contrasts and source diagnostic
bin means are descriptive, not independent replications or causal estimates.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, context_features, subset_indices
from representation_flow.experiment_contract import canonical_hash, data_identity, sha256, source_hashes, validate_completed

REPRESENTATIONS = ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")
CONTRASTS = (("Raw", "CorrectedPhysics"), ("Raw", "HistoricalIncorrectPhysics"), ("HistoricalIncorrectPhysics", "CorrectedPhysics"))
DISCLAIMER = ("All five seeds share source data. Their sample SD is optimizer-seed variation, not uncertainty from independent data replications. "
              "Validation selected the checkpoints, so its paired event differences are descriptive; no confidence interval or significance claim is made. "
              "Fixed-bin residual locations describe representation coordinates and do not establish a causal optimization mechanism. "
              "No confirmation data were read, no retraining was performed, and these current-engine results do not relabel historical results.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_cells(protocol):
    if protocol.get("schema") != "corrected-center-controlled-study-v1":
        raise ValueError("unsupported correction protocol")
    if (protocol.get("representations") != list(REPRESENTATIONS)
            or protocol.get("contexts") != {"Full": "full", "Hidden": "thickness_charge"}
            or protocol.get("sample_fractions") != [.1, 1.]
            or protocol.get("seeds") != [3101,3102,3103,3104,3105]
            or protocol.get("subset_catalog_id") != 4101
            or protocol.get("eligibility") != "forward"
            or protocol.get("capacity") != {"label":"Small", "hidden_width":32}
            or protocol["optimization"]["max_updates"] != 20000):
        raise ValueError("analysis requires the prespecified 60-cell controlled matrix")
    result={}
    for fraction, context, rep, seed in itertools.product(protocol["sample_fractions"],protocol["contexts"],REPRESENTATIONS,protocol["seeds"]):
        tag=f"f{round(fraction*100):03d}"
        run_id=f"correction__{context}__{rep}__Small__{tag}__sub4101__seed{seed}"
        result[run_id]=(context,tag,seed,rep)
    return result


def _inside_file(root, relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact is missing or outside its root: {relative}")
    return path


def load_checked_cache(cache, protocol, expected_cache_hash):
    record=read_json(cache/"cache_manifest.json")
    if canonical_hash(record)!=expected_cache_hash or record["data"]!=protocol["main_data"] or record["protocol_sha256"]!=canonical_hash(protocol):
        raise ValueError("cache identity differs from the frozen experiment")
    names=[item["name"] for item in record["files"]]
    required={"event_uid.npy","p_in.npy","thickness.npy","charge.npy","y.npy","split.npy","indices_validation.npy","indices_f010.npy","indices_f100.npy"}
    if len(set(names))!=len(names) or not required.issubset(names):
        raise ValueError("cache inventory is duplicated or incomplete")
    arrays={}
    for item in record["files"]:
        path=_inside_file(cache,item["name"])
        if sha256(path)!=item["sha256"]:
            raise ValueError(f"cache checksum mismatch: {item['name']}")
        arrays[path.stem]=np.load(path,mmap_mode="r",allow_pickle=False)
    n=len(arrays["event_uid"])
    if n!=record["eligible_events"] or len(set(arrays["event_uid"].tolist()))!=n:
        raise ValueError("cache eligible-event count or UID uniqueness mismatch")
    for name in ("p_in","thickness","charge","y","split"):
        if arrays[name].shape!=(n,):
            raise ValueError(f"cache shape mismatch: {name}")
    if not all(np.isfinite(arrays[k]).all() for k in ("p_in","thickness","charge","y")) or np.any(arrays["y"]>=arrays["p_in"]):
        raise ValueError("cache has invalid finite-target/support values")
    valid=np.flatnonzero(arrays["split"]=="validation")
    if len(valid)!=record["validation_events"] or not np.array_equal(valid,arrays["indices_validation"]):
        raise ValueError("cache validation-index order mismatch")
    for fraction in protocol["sample_fractions"]:
        tag=f"f{round(fraction*100):03d}"
        expected=subset_indices(arrays,fraction,protocol["subset_catalog_id"])
        actual=arrays[f"indices_{tag}"]
        if not np.array_equal(expected,actual):
            raise ValueError(f"cache train-index order mismatch: {tag}")
        if canonical_hash(arrays["event_uid"][actual].tolist())!=record["fractions"][tag]["ordered_uid_sha256"] or len(actual)!=record["fractions"][tag]["events"]:
            raise ValueError("cache train UID digest mismatch")
    if not np.any(arrays["split"]=="diagnostic"):
        raise ValueError("source diagnostic split is empty")
    return arrays,record


def _equal(left,right,message):
    if left!=right:
        raise ValueError(message)


def validate_study(study, cache, protocol, *, merged=None):
    """Fail closed before producing any scientific summary."""
    torch.set_num_threads(1)
    cells=expected_cells(protocol)
    scores=read_json(study/"scores.json")
    if scores.get("status")!="COMPLETE" or scores.get("failures"):
        raise ValueError("study is incomplete or has recorded failures; preserve all cells and failure records")
    identity=read_json(study/"experiment_identity.json")
    definition=identity["identity"]
    if identity.get("sha256")!=canonical_hash(definition) or scores["experiment_identity"]!=identity["sha256"]:
        raise ValueError("experiment identity digest mismatch")
    if definition.get("mode")!="execute" or definition.get("max_updates")!=protocol["optimization"]["max_updates"]:
        raise ValueError("benchmark or wrong-budget output is not a scientific study")
    if definition.get("protocol_sha256")!=canonical_hash(protocol) or definition.get("data")!=protocol["main_data"]:
        raise ValueError("protocol/data identity mismatch")
    expected_code=source_hashes((Path(__file__).with_name("run_correction_study.py").resolve(),))
    if definition.get("code")!=expected_code:
        raise ValueError("current scientific sources differ from the frozen study")
    if merged is not None and data_identity(merged)!=protocol["main_data"]:
        raise ValueError("source main-data files differ from the frozen protocol")
    if validate_completed(scores["runs"],study,identity["sha256"],set(cells))!=set(cells):
        raise ValueError("study does not contain all 60 completed cells")
    frozen=read_json(study/"FROZEN_CHECKPOINT_MANIFEST.json")
    if frozen.get("schema")!="correction-checkpoint-freeze-v1" or frozen.get("experiment_identity")!=identity["sha256"] or frozen.get("frozen_before_confirmation") is not True:
        raise ValueError("invalid frozen checkpoint manifest")
    frozen_rows={r["run_id"]:r for r in frozen["models"]}
    if len(frozen["models"])!=60 or set(frozen_rows)!=set(cells):
        raise ValueError("frozen checkpoint matrix is incomplete or duplicated")
    arrays,cache_record=load_checked_cache(cache,protocol,definition["cache_sha256"])
    valid=arrays["indices_validation"]
    uid=arrays["event_uid"][valid]
    validation_digest=canonical_hash(uid.tolist())
    checked=[];predictions={};preprocessors={};initializations={}
    opt=protocol["optimization"]
    for row in sorted(scores["runs"],key=lambda r:r["run_id"]):
        run_id=row["run_id"];context,tag,seed,rep=cells[run_id]
        fixed=frozen_rows[run_id]
        for key in ("artifacts","representation","context_label","sample_fraction","seed"):
            _equal(fixed[key],row[key],f"frozen manifest disagrees on {key}: {run_id}")
        _equal((row["context_label"],row["fraction_tag"],row["seed"],row["representation"]),(context,tag,seed,rep),"cell labels disagree with run ID")
        train=arrays[f"indices_{tag}"]
        _equal(row["train_event_uid_sha256"],cache_record["fractions"][tag]["ordered_uid_sha256"],"training UID order differs")
        _equal(row["validation_event_uid_sha256"],validation_digest,"validation UID order differs")
        manifest=read_json(_inside_file(study,row["artifacts"]["run_manifest"]["path"]))
        for key in ("run_id","representation","capacity","seed","selected_update","initialization_hash","preprocessor","checkpoint_contract","train_events","validation_events","max_updates","batch_size","validation_every","learning_rate","context_feature_mode"):
            _equal(manifest[key],row[key],f"run manifest disagrees on {key}: {run_id}")
        for key in ("max_updates","batch_size","validation_every","learning_rate"):
            _equal(row[key],opt[key],f"run changed frozen {key}: {run_id}")
        _equal((row["train_events"],row["validation_events"]),(len(train),len(valid)),"run sample counts differ")
        checkpoint=_inside_file(study,row["artifacts"]["checkpoint"]["path"])
        payload,pre=load_checkpoint(checkpoint,expected_sha256=row["artifacts"]["checkpoint"]["sha256"])
        _equal(pre.to_dict(),row["preprocessor"],"checkpoint preprocessing differs from manifest")
        _equal(payload["checkpoint_contract"],row["checkpoint_contract"],"checkpoint contract differs from manifest")
        _equal(payload["experiment_identity"],identity["sha256"],"checkpoint experiment differs")
        _equal((pre.name,payload["seed"],payload["selected_update"],payload["initialization_hash"]),(rep,seed,row["selected_update"],row["initialization_hash"]),"checkpoint labels differ")
        _equal(payload["checkpoint_contract"]["context_feature_mode"],protocol["contexts"][context],"checkpoint context differs")
        _equal(payload["checkpoint_contract"]["eligibility"],protocol["eligibility"],"checkpoint eligibility differs")
        _equal(payload["training_state"]["update"],row["selected_update"],"optimizer/sampler and selected weights differ in update")
        init_key=(context,seed)
        if init_key in initializations and initializations[init_key]!=row["initialization_hash"]:
            raise ValueError("paired representations/fractions have different initialization")
        initializations[init_key]=row["initialization_hash"]
        pre_key=(tag,rep)
        if pre_key in preprocessors and preprocessors[pre_key].to_dict()!=pre.to_dict():
            raise ValueError("paired seeds/contexts have different preprocessing")
        preprocessors[pre_key]=pre
        ctrain=pre.center(arrays["p_in"][train],arrays["thickness"][train],arrays["charge"][train])
        if not np.isclose(pre.mu,np.mean(arrays["y"][train]-ctrain),rtol=0,atol=1e-14) or not np.isclose(pre.scale,np.std(arrays["y"][train]),rtol=0,atol=1e-14):
            raise ValueError("preprocessing is not the declared train-only mean/scale")
        prediction=_inside_file(study,row["artifacts"]["predictions"]["path"])
        with np.load(prediction,allow_pickle=False) as data:
            if not np.array_equal(data["event_uid"],uid):
                raise ValueError("prediction UID order differs from the frozen validation order")
            if not np.array_equal(data["y"],arrays["y"][valid]) or not np.array_equal(data["p_in"],arrays["p_in"][valid]):
                raise ValueError("prediction targets differ from source validation")
            for key,value in (("representation",rep),("seed",seed),("subset_id",4101),("split","validation")):
                if data[key].shape!=uid.shape or not np.all(data[key]==value):
                    raise ValueError(f"prediction label mismatch: {key}")
            logp=np.asarray(data["log_p_y"],dtype=np.float64)
            if logp.shape!=uid.shape or not np.isfinite(logp).all():
                raise ValueError("prediction log likelihood is nonfinite or has wrong shape")
            c=pre.center(arrays["p_in"][valid],arrays["thickness"][valid],arrays["charge"][valid])
            if not np.allclose(data["c"],c,rtol=0,atol=1e-14) or not np.all(data["mu"]==pre.mu) or not np.allclose(data["s"],pre.scale,rtol=1e-7,atol=0):
                raise ValueError("prediction coordinates differ from frozen preprocessing")
            decomposition=data["log_q_z"].astype(np.float64)-np.log(data["s"].astype(np.float64))-data["log_F_b"].astype(np.float64)
            if not np.allclose(logp,decomposition,rtol=2e-6,atol=2e-6):
                raise ValueError("prediction likelihood decomposition fails")
            nll=float(np.mean(-logp,dtype=np.float64))
            history=row["history"]
            expected_updates=sorted({1,opt["max_updates"],*range(opt["validation_every"],opt["max_updates"]+1,opt["validation_every"])})
            if [entry["update"] for entry in history]!=expected_updates or not all(np.isfinite(entry["validation_nll"]) for entry in history):
                raise ValueError("checkpoint selection history is incomplete or nonfinite")
            best=min(history,key=lambda h:h["validation_nll"])
            if row["selected_update"]!=best["update"] or not np.isclose(float(np.mean(-data["log_p_y"])),best["validation_nll"],rtol=0,atol=1e-6):
                raise ValueError("selected checkpoint does not match minimum validation history")
        if not np.isclose(nll,row["validation_nll"],rtol=0,atol=1e-12):
            raise ValueError("saved validation NLL does not reconstruct from predictions")
        sample=np.unique(np.linspace(0,len(valid)-1,min(4096,len(valid)),dtype=int))
        sample_idx=valid[sample]
        p,t,q,y=(arrays[k][sample_idx] for k in ("p_in","thickness","charge","y"))
        flow=ConditionalSplineFlow(**payload["model_config"])
        flow.load_state_dict(payload["state_dict"]);flow.eval()
        mode=protocol["contexts"][context]
        with torch.no_grad():
            replay=flow.original_terms(torch.as_tensor(y,dtype=torch.float32),torch.as_tensor(pre.center(p,t,q),dtype=torch.float32),pre.mu,
                torch.as_tensor(pre.scale_for(p,t,q),dtype=torch.float32),torch.as_tensor(p,dtype=torch.float32),
                torch.as_tensor(context_features(p,t,q,mode),dtype=torch.float32))["log_p_y"].numpy().astype(np.float64)
        if not np.allclose(replay,logp[sample],rtol=2e-6,atol=1e-4):
            raise ValueError("frozen checkpoint does not reproduce sampled exported likelihoods")
        replay_error=float(np.max(np.abs(replay-logp[sample])))
        key=(context,tag,seed,rep);predictions[key]=logp
        checked.append({"run_id":run_id,"context":context,"fraction_tag":tag,"seed":seed,"representation":rep,"validation_events":len(valid),
            "validation_nll":nll,"selected_update":row["selected_update"],"maximum_updates":row["max_updates"],"initialization_hash":row["initialization_hash"],
            "checkpoint_sha256":row["artifacts"]["checkpoint"]["sha256"],"predictions_sha256":row["artifacts"]["predictions"]["sha256"],
            "checkpoint_replayed_validation_events":len(sample),"checkpoint_replay_max_abs_logpdf_error":replay_error})
    return {"cells":checked,"predictions":predictions,"preprocessors":preprocessors,"arrays":arrays,"experiment_identity":identity["sha256"],
            "failures":scores["failures"],"cache_record":cache_record}


def summarize_values(values):
    values=np.asarray(values,dtype=float)
    if values.shape!=(5,) or not np.isfinite(values).all():
        raise ValueError("every contrast must retain exactly five finite seed values")
    return {"seeds":5,"mean":float(values.mean()),"sample_sd":float(values.std(ddof=1)),"minimum":float(values.min()),"maximum":float(values.max()),
            "positive_seeds":int((values>0).sum()),"negative_seeds":int((values<0).sum()),"zero_seeds":int((values==0).sum())}


def paired_contrasts(predictions, protocol):
    seed_rows=[];summary=[]
    for fraction in protocol["sample_fractions"]:
        tag=f"f{round(fraction*100):03d}"
        for left,right in CONTRASTS:
            contrast=f"{left}_minus_{right}";by_context={}
            for context in ("Full","Hidden"):
                events=[]
                for seed in protocol["seeds"]:
                    diff=predictions[(context,tag,seed,right)]-predictions[(context,tag,seed,left)]
                    events.append(diff)
                    seed_rows.append({"fraction_tag":tag,"context":context,"contrast":contrast,"seed":seed,"nll_difference":float(diff.mean()),
                        "validation_events":len(diff),"event_difference_sd_descriptive":float(diff.std(ddof=1))})
                by_context[context]=events
                summary.append({"fraction_tag":tag,"context":context,"contrast":contrast,**summarize_values([x.mean() for x in events])})
            interactions=[a-b for a,b in zip(by_context["Full"],by_context["Hidden"])]
            for seed,diff in zip(protocol["seeds"],interactions):
                seed_rows.append({"fraction_tag":tag,"context":"Full_minus_Hidden","contrast":contrast,"seed":seed,"nll_difference":float(diff.mean()),
                    "validation_events":len(diff),"event_difference_sd_descriptive":float(diff.std(ddof=1))})
            summary.append({"fraction_tag":tag,"context":"Full_minus_Hidden","contrast":contrast,**summarize_values([x.mean() for x in interactions])})
    return seed_rows,summary


def diagnostic_locations(arrays,preprocessors):
    idx=np.flatnonzero(arrays["split"]=="diagnostic")
    p,t,q,y=(arrays[k][idx] for k in ("p_in","thickness","charge","y"))
    edges=np.linspace(.3,1.,8)
    if np.any((p<edges[0])|(p>edges[-1])) or not np.isin(t,[.1,.3,1.]).all() or not np.isin(q,[-1.,1.]).all():
        raise ValueError("diagnostic values escape the fixed 42-stratum design")
    bins=np.minimum(np.searchsorted(edges,p,side="right")-1,6)
    rows=[];summaries=[]
    for (tag,rep),pre in sorted(preprocessors.items()):
        residual=y-pre.center(p,t,q)-pre.mu
        local=[]
        for bin_id,thickness,charge in itertools.product(range(7),(.1,.3,1.),(-1.,1.)):
            mask=(bins==bin_id)&(t==thickness)&(q==charge);n=int(mask.sum())
            mean=float(residual[mask].mean()) if n else None
            row={"fraction_tag":tag,"representation":rep,"momentum_bin":bin_id,"p_low_GeV_c":float(edges[bin_id]),"p_high_GeV_c":float(edges[bin_id+1]),
                 "thickness_mm":thickness,"charge":charge,"events":n,"mean_residual_GeV_c":mean,"mean_standardized_residual":None if mean is None else mean/pre.scale}
            rows.append(row)
            if n:local.append((n,mean))
        counts,means=np.asarray(local,dtype=float).T
        weighted_mean=float(np.average(means,weights=counts))
        summaries.append({"fraction_tag":tag,"representation":rep,"diagnostic_events":len(idx),"populated_bins":len(local),"empty_bins":42-len(local),
            "residual_mean_GeV_c":float(residual.mean()),"event_residual_rmse_GeV_c":float(np.sqrt(np.mean(residual**2))),
            "conditional_mean_rmse_GeV_c":float(np.sqrt(np.average(means**2,weights=counts))),
            "conditional_mean_std_GeV_c":float(np.sqrt(np.average((means-weighted_mean)**2,weights=counts))),
            "standardized_conditional_mean_rmse":float(np.sqrt(np.average(means**2,weights=counts))/pre.scale)})
    return rows,summaries


def write_csv(path,rows):
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(study,cache,protocol_path,output,*,merged=None):
    if output.exists():
        raise FileExistsError("analysis output already exists; choose a new directory")
    protocol=read_json(protocol_path)
    checked=validate_study(study,cache,protocol,merged=merged)
    seeds,summary=paired_contrasts(checked["predictions"],protocol)
    bins,locations=diagnostic_locations(checked["arrays"],checked["preprocessors"])
    result={"status":"COMPLETE_CORRECTION_STUDY_DESCRIPTIVE_ANALYSIS","experiment_identity":checked["experiment_identity"],"cells":checked["cells"],
            "failed_runs":checked["failures"],"seed_contrasts":seeds,"contrast_summary":summary,"source_diagnostic_locations":locations,
            "interpretation_boundary":DISCLAIMER,"contrast_sign":"NLL(left)-NLL(right): positive means right has lower validation NLL; interaction=Full contrast-Hidden contrast",
            "validation_selection_precision":"checkpoint selection uses float32 mean; exported NLL is reconstructed with float64 accumulation",
            "checkpoint_replay":"up to 4096 uniformly spaced ordered validation events per checkpoint; rtol=2e-6, atol=1e-4",
            "optimizer_detail":"Adam, weight_decay=0; gradient norm clipped to 5.0; matched ordered subsets and batch RNG seed",
            "diagnostic_definition":"30% source diagnostic split; fixed seven [0.3,1.0] GeV/c momentum bins x three thicknesses x two charges; residual=y-c(X)-training_mu; RMSE of bin means is count-weighted"}
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("cell_scores.csv",checked["cells"]),("seed_contrasts.csv",seeds),("contrast_summary.csv",summary),("diagnostic_bins.csv",bins),("diagnostic_summary.csv",locations)):
        write_csv(output/name,rows)
    (output/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    lines=["受控单位修正研究：完整60-cell描述性分析", "", "所有60个cell、冻结权重、预测文件、代码/协议/数据身份、配对UID顺序和初始化均通过校验。", "三种表示均使用当前冻结引擎；本研究不宣称重现历史引擎。", "", "差值定义：NLL(left)-NLL(right)，正值表示right的验证NLL更低。Full-minus-Hidden为同seed的差中差。", "以下均值/样本SD/范围基于全部5个seed，保留正值、负值及零值。", ""]
    for row in summary:
        lines.append(f"{row['fraction_tag']} {row['context']} {row['contrast']}: mean={row['mean']:+.9f}, sampleSD={row['sample_sd']:.9f}, range=[{row['minimum']:+.9f}, {row['maximum']:+.9f}] nat/event; positive/negative/zero={row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']}")
    lines += ["", "固定42-bin源域diagnostic残差位置（原始单位GeV/c；不使用验证或确认数据拟合）"]
    for row in locations:
        lines.append(f"{row['fraction_tag']} {row['representation']}: residual mean={row['residual_mean_GeV_c']:+.9g}, bin-mean RMSE={row['conditional_mean_rmse_GeV_c']:.9g}, bin-mean centered SD={row['conditional_mean_std_GeV_c']:.9g}; populated/empty bins={row['populated_bins']}/{row['empty_bins']}")
    lines += ["", "解释边界：五个seed共享数据，SD只描述优化随机性，不是独立样本重复的置信区间。验证集参与checkpoint选择，不能把验证差值当独立确认检验。残差分箱位置是描述性机制线索，不证明因果或物理表示普遍优越。所有结果限于Small容量、当前样本分数和相同输入范围；不替代完整120-cell容量矩阵。", "checkpoint选择按float32均值，报告NLL按float64重算；Adam梯度范数裁剪阈值5.0。未读取确认集，未在分析阶段重训。", "", DISCLAIMER]
    (output/"SCIENTIFIC_ANALYSIS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={"analysis_source_sha256":sha256(Path(__file__)),"protocol_file_sha256":sha256(protocol_path),"input_sha256":{str(path):sha256(path) for path in (study/"scores.json",study/"experiment_identity.json",study/"FROZEN_CHECKPOINT_MANIFEST.json",cache/"cache_manifest.json")},
              "outputs":[{"name":p.name,"sha256":sha256(p)} for p in sorted(output.iterdir())]}
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path("reports/correction_v2_20261107/study"))
    parser.add_argument("--cache",type=Path,default=Path("reports/correction_v2_20261107/cache"))
    parser.add_argument("--protocol",type=Path,default=Path("protocol/correction_v2_20261107.json"))
    parser.add_argument("--merged",type=Path,default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.study,args.cache,args.protocol,args.output,merged=args.merged)
    print(json.dumps({"status":result["status"],"cells":len(result["cells"]),"output":str(args.output)}))


if __name__=="__main__":
    main()
