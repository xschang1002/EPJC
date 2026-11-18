"""Score the frozen unique 240-model union on new C/D catalogs only.

No fitting or checkpoint selection occurs here. Every model sees identical
eligible event order. Full per-event arrays are private research artifacts;
only aggregate analysis and source snapshots belong in the public mirror.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import torch

from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import ConditionalSplineFlow, context_features
from representation_flow.experiment_contract import canonical_hash, save_json, sha256
from representation_flow.information_checkpoint import (
    ROOT, SOURCE_NAMES, evaluate_information, load_information_checkpoint, make_tensors,
)
from representation_flow.information_context import fixed_group_ids
from representation_flow.run_information_study import budget_contract, check_deadline
from simulation.fresh_catalog_runner import check_model_union, verify_file

ARRAY_NAMES = ("event_uid", "p_in", "thickness", "charge", "y", "group_id")
COMPONENT_NAMES = ("log_p_y", "log_q_z", "base_value", "base_logpdf", "spline_logabsdet",
                   "log_support_cdf", "log_s", "z", "b", "log_cdf_y", "log_sf_y")
SCORING_SOURCES = set(SOURCE_NAMES) | {
    "src/representation_flow/score_information_catalog.py",
    "src/representation_flow/checkpoint_contract.py", "src/simulation/fresh_catalog_runner.py",
    "src/data_contract/beam_schedule.py",
    "protocol/information_study_20261108.json",
}
_ARRAYS = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
        raise ValueError("model ID is not a safe single directory name")
    return value


def array_record(path, root, value):
    return {"path": Path(path).relative_to(root).as_posix(), "sha256": sha256(path),
            "bytes": Path(path).stat().st_size, "shape": list(value.shape), "dtype": str(value.dtype)}


def load_model(row, roots):
    """Legacy denotes the previous v2 contract, never an uncontracted weight file."""
    record = row["artifacts"]["checkpoint"]
    path = verify_file(dict(record, root_key=record.get("root_key", row["root_key"])), roots)
    labels = row["labels"]
    if row["checkpoint_kind"] == "information":
        payload, pre, flow, spec = load_information_checkpoint(
            path, expected_sha256=record["sha256"], expected_experiment_identity=row["experiment_identity"])
        context = spec
        if labels["context_label"] != spec["name"] or labels["capacity"] != "Small":
            raise ValueError("new model context/capacity labels disagree with checkpoint")
    elif row["checkpoint_kind"] == "legacy":
        payload, pre = load_checkpoint(path, expected_sha256=record["sha256"])
        contract = payload["checkpoint_contract"]
        if contract["eligibility"] != "forward":
            raise ValueError("fresh evaluation requires the forward target contract")
        context = contract["context_feature_mode"]
        if context != {"Full": "full", "Hidden": "thickness_charge"}.get(labels["context_label"]):
            raise ValueError("legacy neural input mode cannot be relabelled")
        if payload["capacity"] != labels["capacity"]:
            raise ValueError("legacy capacity label differs from checkpoint")
        flow = ConditionalSplineFlow(**payload["model_config"])
        flow.load_state_dict(payload["state_dict"], strict=True)
        flow.eval()
    else:
        raise ValueError("unknown frozen checkpoint kind")
    if (payload["experiment_identity"] != row["experiment_identity"] or pre.name != labels["representation"]
            or payload["seed"] != labels["seed"]):
        raise ValueError("model identity/representation/seed label mismatch")
    return payload, pre, flow, context


def preflight_models(union, roster, roots):
    expected = roster["models"]
    if roster.get("schema") != "fresh-model-roster/v1" or roster.get("expected_models") != 240:
        raise ValueError("the prespecified complete model roster is required")
    check_model_union(union, roster["expected_model_ids"], roots)
    expected_by_id = {r["model_id"]: r for r in expected}
    if len(expected_by_id) != 240 or set(expected_by_id) != set(roster["expected_model_ids"]):
        raise ValueError("roster labels and explicit IDs differ")
    for row in union["models"]:
        safe_id(row["model_id"])
        reference = expected_by_id[row["model_id"]]
        for field in ("model_id", "cohort", "root_key", "checkpoint_kind", "labels"):
            if row[field] != reference[field]:
                raise ValueError("union changed a prespecified model label: " + field)
        if row["labels"]["run_id"] != row["model_id"]:
            raise ValueError("model ID and run ID differ")
        manifest_record = row["artifacts"]["run_manifest"]
        manifest = read_json(verify_file(dict(manifest_record, root_key=manifest_record.get("root_key", row["root_key"])), roots))
        payload, pre, _, _ = load_model(row, roots)
        if (manifest["run_id"] != row["model_id"] or manifest["experiment_identity"] != row["experiment_identity"]
                or manifest["preprocessor"] != pre.to_dict() or manifest["max_updates"] != 20000
                or manifest["selected_update"] != payload["selected_update"]):
            raise ValueError("frozen run manifest and loaded checkpoint disagree")
    return union["models"]


def verify_source_analysis(plan, union, receipt, roots):
    freeze_path=verify_file(union["information_freeze"],roots)
    freeze=read_json(freeze_path)
    models=[r for r in union["models"] if r["cohort"]=="information80"]
    if (freeze.get("schema")!="information-checkpoint-freeze/v1"
            or freeze.get("status")!="COMPLETE_INFORMATION_CHECKPOINT_FREEZE" or freeze.get("expected_cells")!=80
            or len(freeze.get("models",[]))!=80 or len(models)!=80
            or {r["run_id"] for r in freeze["models"]}!={r["model_id"] for r in models}
            or receipt.get("information_freeze_sha256")!=sha256(freeze_path)):
        raise ValueError("generation receipt and complete new80 model freeze disagree")
    by_id={r["run_id"]:r for r in freeze["models"]}
    for model in models:
        if model["experiment_identity"]!=freeze["experiment_identity"]:
            raise ValueError("new80 union experiment identity changed")
        for role in ("checkpoint","predictions","run_manifest"):
            if any(by_id[model["model_id"]]["artifacts"][role][key]!=model["artifacts"][role][key] for key in ("path","sha256","bytes")):
                raise ValueError("new80 artifacts differ between union and training freeze")
    source_path=verify_file(plan["source_validation_analysis"],roots)
    source=read_json(source_path)
    if (source.get("schema")!="information-source-analysis/v1" or source.get("status")!="COMPLETE_INFORMATION_SOURCE_ANALYSIS"
            or source.get("expected_models")!=80 or source.get("model_count")!=80
            or source.get("all_validation_predictions_replayed") is not True
            or source.get("study_freeze_sha256")!=sha256(freeze_path)
            or source.get("protocol_sha256")!=freeze["protocol_sha256"]
            or receipt.get("source_validation_analysis_sha256")!=sha256(source_path)):
        raise ValueError("independent complete new80 source analysis changed after generation")
    return freeze_path,freeze


def validate_generation(plan_path, generation_root, union_path, roster_path, roots, budget):
    """Validate all gates and source hashes before opening any fresh events."""
    plan, receipt = read_json(plan_path), read_json(generation_root / "GENERATION_COMPLETE.json")
    if (plan.get("schema") != "fresh-catalog-protocol/v1" or plan.get("status") != "FROZEN_FRESH_CATALOG"
            or receipt.get("status") != "COMPLETE_FRESH_GEANT4_CATALOGS_NO_MODEL_SCORING"
            or receipt["plan_sha256"] != sha256(plan_path) or receipt["deadline_unix"] != budget["deadline_unix"]):
        raise ValueError("complete generation receipt does not bind the frozen plan/budget")
    sources = {r["path"] for r in plan["source_files"] if r["root_key"] == "repo"}
    if not SCORING_SOURCES.issubset(sources):
        raise ValueError("fresh pre-generation source freeze omitted a scoring dependency")
    for record in plan["source_files"]:
        verify_file(record, roots)
    if not any(verify_file(r, roots).resolve() == Path(roster_path).resolve() for r in plan["source_files"]):
        raise ValueError("fresh source freeze did not bind the prespecified roster file")
    if verify_file(plan["model_union"], roots).resolve() != Path(union_path).resolve():
        raise ValueError("scoring union differs from the pre-generation union")
    verify_source_analysis(plan,read_json(union_path),receipt,roots)
    if budget_contract(verify_file(plan["budget_record"], roots)) != budget:
        raise ValueError("scoring budget differs from the pre-generation budget")
    analysis_path = verify_file(plan["analysis_plan"], roots)
    analysis = read_json(analysis_path)
    if (analysis.get("status") != "FROZEN_FRESH_STATISTICAL_ANALYSIS"
            or receipt["analysis_sha256"] != sha256(analysis_path)):
        raise ValueError("statistical plan was not frozen before generation")
    if {b["block_id"] for b in plan["catalogs"]} != {"C", "D"} or len(plan["catalogs"]) != 2:
        raise ValueError("both fixed fresh C/D catalogs are required")
    recorded = {r["block_id"]: r for r in receipt["catalogs"]}
    if set(recorded) != {"C", "D"} or len(receipt["catalogs"]) != 2:
        raise ValueError("generation did not complete the two fixed catalogs")
    for block in plan["catalogs"]:
        row, folder = recorded[block["block_id"]], generation_root / block["block_id"]
        if block["events"] != 500000 or any(row.get(k) != v for k, v in block.items()):
            raise ValueError("generation scope differs from the frozen launch design")
        for relative, field in (("beam/catalog.json", "catalog_sha256"), ("merged/manifest.json", "merged_manifest_sha256"),
                                ("fresh_catalog_manifest.json", "stratum_counts_sha256")):
            if sha256(folder / relative) != row[field]:
                raise ValueError("fresh catalog identity changed after generation")
    check_deadline(budget["deadline_unix"])
    return plan, receipt


def read_catalog_arrays(folder, block, deadline, *, expected_launched=500000):
    """Independently count S0/S1 and reject all technical aborts/UID duplicates."""
    counts = read_json(folder / "fresh_catalog_manifest.json")
    if (counts.get("schema") != "fresh-catalog-stratum-counts/v1" or counts.get("status") != "COMPLETE_42_STRATUM_COUNTS"
            or counts["launched"] != expected_launched or counts["technical_aborted"] != 0 or counts["run_id"] != block["run_id"]):
        raise ValueError("invalid fixed-stratum counts receipt")
    path = folder / "merged/events.jsonl"
    if sha256(path) != counts["events_sha256"]:
        raise ValueError("fresh event bytes do not match generation counts")
    cells = {row["group_id"]: row for row in counts["cells"]}
    if len(counts["cells"]) != 42 or set(cells) != set(range(42)):
        raise ValueError("exactly 42 distinct physical strata required")
    launched, eligible = np.zeros(42, dtype=np.int64), np.zeros(42, dtype=np.int64)
    result = {key: [] for key in ARRAY_NAMES}
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line_number % 8192 == 0:
                check_deadline(deadline)
            row = json.loads(line)
            uid = row["event_uid"]
            if not isinstance(uid, str) or not uid.startswith(block["run_id"] + ":") or uid in seen:
                raise ValueError("fresh event UID is duplicated or belongs to another run")
            seen.add(uid)
            if type(row["S1"]) is not bool or type(row.get("aborted")) is not bool or row["aborted"]:
                raise ValueError("technical abortion or nonboolean fresh eligibility")
            p,t,q = float(row["p_in_GeV_c"]),float(row["thickness_mm"]),float(row["pdg"])/211.
            group = int(fixed_group_ids(np.array([p]),np.array([t]),np.array([q]))[0])
            if group // 6 != row["momentum_bin"]:
                raise ValueError("event momentum differs from assigned launch stratum")
            launched[group] += 1
            if not row["S1"]:
                continue
            y, pout = float(row["delta_p_GeV_c"]),float(row["p_out_GeV_c"])
            if (row["exit_face"] != "forward" or not np.isfinite(y) or not np.isfinite(pout) or pout <= 0 or y >= p
                    or not np.isclose(p-y,pout,atol=1e-12,rtol=1e-12)):
                raise ValueError("eligible fresh event violates the forward momentum target")
            eligible[group] += 1
            for key,value in zip(ARRAY_NAMES,(uid,p,t,q,y,group)):
                result[key].append(value)
    for group,row in cells.items():
        if (row["launched_events"] != int(launched[group]) or row["eligible_events"] != int(eligible[group])
                or row["ineligible"] != int(launched[group]-eligible[group]) or row["aborted"] != 0):
            raise ValueError("independent S0/S1 counts disagree with frozen catalog receipt")
    if len(seen) != expected_launched or int(eligible.sum()) != counts["eligible"] or np.any(eligible <= 0):
        raise ValueError("fresh total counts or eligible-stratum coverage invalid")
    arrays = {key: np.asarray(value, dtype=str if key=="event_uid" else np.int64 if key=="group_id" else np.float64)
              for key,value in result.items()}
    return arrays, {str(g):int(n) for g,n in enumerate(launched)}, {str(g):int(n) for g,n in enumerate(eligible)}, seen


def tensors_for_model(arrays, pre, context, kind):
    indices = np.arange(len(arrays["y"]))
    if kind == "information":
        return make_tensors(arrays, indices, pre, context)
    p,t,q,y = (np.asarray(arrays[k]) for k in ("p_in","thickness","charge","y"))
    values = (y,pre.center(p,t,q),p,context_features(p,t,q,context),pre.scale_for(p,t,q))
    return tuple(torch.as_tensor(np.array(value,copy=True),dtype=torch.float32) for value in values)


def score_model(row, roots, arrays, output, event_uid_sha256, deadline, *, batch_size=8192):
    check_deadline(deadline)
    started = time.perf_counter()
    _,pre,flow,context = load_model(row,roots)
    tensors = tensors_for_model(arrays,pre,context,row["checkpoint_kind"])
    # Bound each inference segment so the global deadline is enforced during scoring.
    pieces = {key: [] for key in COMPONENT_NAMES}
    for start in range(0,len(arrays["y"]),batch_size):
        check_deadline(deadline)
        part = evaluate_information(flow,tuple(value[start:start+batch_size] for value in tensors),pre,batch_size,components=True)
        for key in pieces:
            if key in ("log_cdf_y","log_sf_y"):
                if np.any(np.isnan(part[key])) or np.any(part[key] > 1e-6):
                    raise ValueError("invalid logarithmic PIT probability")
            elif not np.all(np.isfinite(part[key])):
                raise ValueError("nonfinite likelihood component")
            pieces[key].append(part[key])
    values = {key: np.concatenate(parts) for key,parts in pieces.items()}
    rebuilt = values["base_logpdf"]+values["spline_logabsdet"]-values["log_s"]-values["log_support_cdf"]
    if not np.allclose(rebuilt,values["log_p_y"],rtol=2e-6,atol=2e-5):
        raise ValueError("base/Jacobian/scale/support decomposition failed")
    folder = Path(output)/"models"/safe_id(row["model_id"])
    folder.mkdir(parents=True,exist_ok=False)
    files = {}
    for name,value in values.items():
        check_deadline(deadline)
        path = folder/(name+".npy"); np.save(path,value,allow_pickle=False)
        files[name] = array_record(path,Path(output),value)
    result = {key:row[key] for key in ("model_id","cohort","checkpoint_kind","labels","root_key","experiment_identity")}
    result.update(status="COMPLETE_FROZEN_MODEL_SCORE", checkpoint_sha256=row["artifacts"]["checkpoint"]["sha256"],
                  preprocessor=pre.to_dict(),event_uid_sha256=event_uid_sha256,eligible_events=len(arrays["y"]),
                  nll=float(np.mean(-values["log_p_y"],dtype=np.float64)), arrays=files,
                  elapsed_seconds=time.perf_counter()-started)
    save_json(folder/"model_manifest.json",result)
    check_deadline(deadline)
    return result


def init_worker(cache):
    global _ARRAYS
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    _ARRAYS={key:np.load(Path(cache)/(key+".npy"),mmap_mode="r",allow_pickle=False) for key in ARRAY_NAMES}


def score_job(job):
    return score_model(job["row"],{k:Path(v) for k,v in job["roots"].items()},_ARRAYS,Path(job["output"]),
                       job["event_uid_sha256"],job["deadline_unix"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","generation-root","union","roster","budget","source-cache","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--information-cache",type=Path,default=ROOT/"reports/information_extension_20261108/cache")
    parser.add_argument("--artifact-root",action="append",default=[],metavar="KEY=PATH")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if not 1 <= args.workers <= 4:parser.error("workers must be 1..4")
    if args.output.exists():raise FileExistsError("catalog scoring never overwrites or silently resumes an output")
    started=time.time();torch.set_num_threads(1)
    budget=budget_contract(args.budget)
    roots={"repo":ROOT}
    for raw in args.artifact_root:
        key,path=raw.split("=",1)
        if key in roots:parser.error("duplicate or reserved artifact root")
        roots[key]=Path(path).resolve(strict=True)
    plan,receipt=validate_generation(args.plan,args.generation_root,args.union,args.roster,roots,budget)
    union,roster=read_json(args.union),read_json(args.roster)
    if set(plan["expected_model_ids"]) != set(roster["expected_model_ids"]):
        raise ValueError("fresh generation and prespecified evaluation roster differ")
    rows=preflight_models(union,roster,roots)
    # Source UID array identity comes from the already frozen information cache lineage.
    source_manifest=read_json(args.source_cache/"cache_manifest.json")
    information_cache=read_json(args.information_cache/"cache_manifest.json")
    training_protocol=read_json(ROOT/"protocol/information_study_20261108.json")
    info_freeze_path,info_freeze=verify_source_analysis(plan,union,receipt,roots)
    info_identity=read_json(info_freeze_path.parent/"experiment_identity.json")
    if (information_cache["source_cache_manifest_sha256"] != sha256(args.source_cache/"cache_manifest.json")
            or information_cache["protocol_sha256"] != canonical_hash(training_protocol)
            or source_manifest["data"] != training_protocol["main_data"]
            or canonical_hash(info_identity["identity"]) != info_freeze["experiment_identity"]
            or info_identity["sha256"] != info_freeze["experiment_identity"]
            or info_identity["identity"]["cache_sha256"] != canonical_hash(information_cache)
            or info_freeze["protocol_sha256"] != canonical_hash(training_protocol)):
        raise ValueError("source UID cache does not match the frozen training-data lineage")
    uid_record=next(r for r in source_manifest["files"] if r["name"]=="event_uid.npy")
    if sha256(args.source_cache/"event_uid.npy") != uid_record["sha256"]:
        raise ValueError("source event UID array changed")
    train_contracts={canonical_hash(read_json(verify_file(dict(r["artifacts"]["run_manifest"],root_key=r["root_key"]),roots))["information_contract"]["source_files"])
                     for r in rows if r["checkpoint_kind"]=="information"}
    if len(train_contracts)!=1:raise ValueError("new information models do not share one scientific implementation")
    prior_uids=set(np.load(args.source_cache/"event_uid.npy",allow_pickle=False).astype(str).tolist())
    ancestor=args.output.parent
    while not ancestor.exists():ancestor=ancestor.parent
    # Eleven float32 components for every model/event plus shared arrays and headroom.
    required_bytes=int(2*500000*240*len(COMPONENT_NAMES)*4*1.2)
    if shutil.disk_usage(ancestor).free < required_bytes:
        raise ValueError("insufficient private disk space for the complete fixed scoring scope")
    args.output.mkdir(parents=True,exist_ok=False)
    summary={"schema":"information-fresh-scoring/v1","status":"RUNNING","generated_utc":utc(),"started_unix":started,
             "deadline_unix":budget["deadline_unix"],"plan_sha256":sha256(args.plan),"union_sha256":sha256(args.union),
             "roster_sha256":sha256(args.roster),"generation_sha256":sha256(args.generation_root/"GENERATION_COMPLETE.json"),
             "source_files":{name:sha256(ROOT/name) for name in sorted(SCORING_SOURCES)},"catalogs":[],"failures":[]}
    for name in SCORING_SOURCES:
        target=args.output/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save_json(args.output/"scoring_summary.json",summary)
    try:
        for block in plan["catalogs"]:
            block_root=args.output/block["block_id"]
            arrays,launched,eligible,all_uids=read_catalog_arrays(args.generation_root/block["block_id"],block,budget["deadline_unix"])
            if prior_uids & all_uids:raise ValueError("fresh catalog overlaps source or the previous fresh block UIDs")
            prior_uids.update(all_uids)
            cache=block_root/"catalog_arrays";cache.mkdir(parents=True)
            files={}
            for key,value in arrays.items():
                path=cache/(key+".npy");np.save(path,value,allow_pickle=False);files[key]=array_record(path,block_root,value)
            uid_hash=canonical_hash(arrays["event_uid"].tolist())
            manifest={"schema":"information-catalog-scores/v1","status":"RUNNING","block_id":block["block_id"],
                      "event_uid_sha256":uid_hash,"eligible_events":len(arrays["y"]),"launched_events":sum(launched.values()),
                      "launched_counts":launched,"eligible_counts":eligible,"arrays":files,"models":[],"failures":[],
                      "plan_sha256":summary["plan_sha256"],"union_sha256":summary["union_sha256"],"generation_sha256":summary["generation_sha256"],
                      "counts_sha256":sha256(args.generation_root/block["block_id"]/"fresh_catalog_manifest.json"),
                      "events_sha256":read_json(args.generation_root/block["block_id"]/"fresh_catalog_manifest.json")["events_sha256"],
                      "source_files":summary["source_files"],"model_precision":"float32; summary means float64","generated_utc":utc()}
            with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(str(cache.resolve()),)) as executor:
                jobs={executor.submit(score_job,{"row":row,"roots":{k:str(v) for k,v in roots.items()},"output":str(block_root.resolve()),
                      "event_uid_sha256":uid_hash,"deadline_unix":budget["deadline_unix"]}):row for row in rows}
                for future in as_completed(jobs):
                    try:manifest["models"].append(future.result())
                    except Exception as exc:
                        manifest["failures"].append({"model_id":jobs[future]["model_id"],"type":type(exc).__name__,"error":str(exc)})
                        for pending in jobs:pending.cancel()
                    save_json(block_root/"score_manifest.json",manifest)
                    print(json.dumps({"block":block["block_id"],"completed_models":len(manifest["models"]),"failures":len(manifest["failures"])}),flush=True)
            if manifest["failures"] or {r["model_id"] for r in manifest["models"]}!={r["model_id"] for r in rows} or len(manifest["models"])!=240:
                raise RuntimeError("all 240 fixed models must score each complete catalog")
            check_deadline(budget["deadline_unix"])
            manifest.update(status="COMPLETE_FROZEN_240_CATALOG_SCORES",completed_utc=utc())
            save_json(block_root/"score_manifest.json",manifest)
            summary["catalogs"].append({"block_id":block["block_id"],"path":block["block_id"]+"/score_manifest.json",
                                         "sha256":sha256(block_root/"score_manifest.json"),"models":240})
            save_json(args.output/"scoring_summary.json",summary)
        validate_generation(args.plan,args.generation_root,args.union,args.roster,roots,budget)
        check_model_union(union,roster["expected_model_ids"],roots)
        summary.update(status="COMPLETE_FROZEN_240_CD_SCORES",completed_utc=utc(),wall_seconds=time.time()-started)
    except Exception as exc:
        summary["failures"].append({"type":type(exc).__name__,"error":str(exc)})
        summary.update(status="INCOMPLETE_FROZEN_240_CD_SCORES",completed_utc=utc(),wall_seconds=time.time()-started)
        save_json(args.output/"scoring_summary.json",summary)
        raise
    save_json(args.output/"scoring_summary.json",summary)
    print(json.dumps({"status":summary["status"],"wall_seconds":summary["wall_seconds"]}),flush=True)


if __name__=="__main__":
    main()
