"""Generate all 16 declared contrasts from complete checked analysis summaries.

Presentation only: no raw catalog access, scoring, model selection or bootstrap.
Source validation is descriptive; fresh event intervals and optimizer-seed SD
are exported as separate quantities. Every registered contrast/seed is retained.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from generate_information_figures import ROOT, SEEDS, load_bundle, read_csv, read_json, seed_values, sha


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def write_csv(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate_pair_provenance(bundles,source,fresh_plan):
    common=("fresh_plan_sha256","union_sha256","roster_sha256","generation_sha256","scoring_summary_sha256")
    for key in common:
        if bundles["C"]["analysis"]["input_provenance"][key]!=bundles["D"]["analysis"]["input_provenance"][key]:
            raise ValueError("C/D analysis lineage differs: "+key)
    plan=read_json(fresh_plan)
    if (plan.get("status")!="FROZEN_FRESH_CATALOG" or sha(fresh_plan)!=bundles["C"]["analysis"]["input_provenance"]["fresh_plan_sha256"]
            or plan["source_validation_analysis"]["sha256"]!=sha(source["paths"]["analysis.json"])):
        raise ValueError("source/fresh tables do not belong to the same pre-generation freeze")
    if bundles["C"]["analysis"]["event_uid_sha256"]==bundles["D"]["analysis"]["event_uid_sha256"]:
        raise ValueError("C/D evaluation event identity is duplicated")


def contrast_tex(name):
    effect,fraction=name.rsplit("/",1)
    def delta(context):return r"\Delta_{\mathrm{"+context+"}}"
    if effect.startswith("gain:"):
        return "$"+delta(effect.split(":",1)[1])+"$",fraction
    if effect.startswith("gain_change:"):
        more,less=effect.split(":",1)[1].split("_minus_")
        return "$"+delta(more)+"-"+delta(less)+"$",fraction
    raise ValueError("unknown declared contrast")


def table_rows(bundle,names):
    recorded={row["contrast"]:row for row in read_csv(bundle["paths"]["primary_contrast_summary.csv"])}
    if len(recorded)!=16 or set(recorded)!=set(names):raise ValueError("incomplete primary contrast summary")
    output=[]
    for name in names:
        values=seed_values(bundle,name);mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
        row=recorded[name]
        for key,value in (("mean",mean),("sample_sd",sd),("min",float(values.min())),("max",float(values.max()))):
            if not np.isclose(float(row[key]),value,rtol=0,atol=1e-12):raise ValueError("summary does not reproduce all five seeds")
        signs={"positive_seeds":int(np.count_nonzero(values>0)),"negative_seeds":int(np.count_nonzero(values<0)),"zero_seeds":int(np.count_nonzero(values==0))}
        if any(int(row[k])!=v for k,v in signs.items()):raise ValueError("seed-direction counts disagree")
        tex,fraction=contrast_tex(name)
        result={"catalog":bundle["block"],"contrast":name,"contrast_tex":tex,"fraction_tag":fraction,
                "training_fraction":.1 if fraction=="f010" else 1.,"mean":mean,"seed_sample_sd":sd,
                "seed_min":float(values.min()),"seed_max":float(values.max()),**signs,
                **{f"seed_{seed}":float(value) for seed,value in zip(SEEDS,values)}}
        if bundle["block"]!="source":
            interval=bundle["uncertainty"][name]
            for key in ("bootstrap_standard_error","pointwise_percentile_lower","pointwise_percentile_upper","simultaneous_lower","simultaneous_upper"):
                result[key]=float(interval[key])
            if not np.isclose(interval["estimate"],mean,rtol=0,atol=1e-10):raise ValueError("bootstrap/seed-mean point estimates differ")
        output.append(result)
    return output


def number(value,*,signed=True):
    # Round only the display; full float precision remains in JSON/CSV.
    return "$"+format(float(value),"+.6f" if signed else ".6f")+"$"


def latex_table(rows,block,events):
    source=block=="source"
    caption=(f"All 16 prespecified contrasts on the source validation set ($N={events:,}$ eligible events). "
             "These selected-checkpoint scores are descriptive; no fresh-event confidence interval is assigned. "
             if source else f"All 16 prespecified contrasts for fresh catalog {block} ($N={events:,}$ eligible events from 500,000 launches). "
             "The event intervals are approximate 95\\% simultaneous max-$|t|$ intervals over all 16 contrasts within this catalog, "
             "using 2000 paired resamples within the 42 launched strata with an aggregated $S=0$ atom. ")
    caption+=("All quantities are in nats per eligible event. $\\Delta$ is Raw-minus-Corrected NLL, averaged over five paired optimizer seeds; "
              "gain differences are more-information minus less-information. Seed SD is the sample SD across those five seeds, "
              "not a confidence interval or variation across independent datasets. $n_+$ counts positive seeds out of 5; zeros are not counted. "
              +("Minimum and maximum retain adverse seed directions." if source else "Event intervals condition on the fixed fitted models; C/D families are separate and not pooled.") )
    lines=[r"\begin{table}[!htbp]",r"\centering",r"\small",r"\setlength{\tabcolsep}{3pt}",r"\caption{"+caption+"}",
           r"\label{tab:information-"+("source16" if source else block+"-primary16")+"}",r"\begin{tabular}{llrrrrr}",r"\toprule",
           (r"Contrast & Train & Mean & Seed SD & Minimum & Maximum & $n_+/5$ \\" if source else
            r"Contrast & Train & Mean & Seed SD & 95\% lower & 95\% upper & $n_+/5$ \\"),r"\midrule"]
    for index,row in enumerate(rows):
        if index in (4,8,12):lines.append(r"\midrule")
        low,high=(row["seed_min"],row["seed_max"]) if source else (row["simultaneous_lower"],row["simultaneous_upper"])
        values=[row["contrast_tex"],r"10\%" if row["fraction_tag"]=="f010" else r"100\%",number(row["mean"]),
                number(row["seed_sample_sd"],signed=False),number(low),number(high),str(row["positive_seeds"])+"/5"]
        lines.append(" & ".join(values)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}",""])
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("source","fresh"))
    parser.add_argument("--source-analysis",type=Path,required=True)
    parser.add_argument("--c-analysis",type=Path);parser.add_argument("--d-analysis",type=Path)
    parser.add_argument("--fresh-plan",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--statistical-plan",type=Path,default=ROOT/"protocol/information_statistical_analysis_20261108.json")
    parser.add_argument("--budget",type=Path,default=ROOT.parent/"_research_extension_20261108/AUTHORIZATION_AND_BUDGET.json")
    args=parser.parse_args()
    budget=read_json(args.budget);deadline=budget["deadline_unix"]
    if (not 0<budget["maximum_wall_seconds"]<=43200 or abs(deadline-budget["started_unix"]-budget["maximum_wall_seconds"])>1e-6):
        raise ValueError("budget does not preserve the original authorized start")
    def check_time():
        if time.time()>=deadline:raise TimeoutError("original research deadline exceeded during table generation")
    check_time();started=time.time()
    plan=read_json(args.statistical_plan)
    if plan.get("status")!="FROZEN_FRESH_STATISTICAL_ANALYSIS":raise ValueError("frozen statistical plan required")
    names=[row["name"] for row in plan["primary_contrasts"]]
    if len(names)!=16 or len(set(names))!=16:raise ValueError("exact16 primary family required")
    source=load_bundle(args.source_analysis,"source",args.statistical_plan)
    if args.mode=="source":bundles={"source":source}
    else:
        if any(value is None for value in (args.c_analysis,args.d_analysis,args.fresh_plan)):
            parser.error("fresh tables require both C/D analysis directories and --fresh-plan")
        bundles={block:load_bundle(path,block,args.statistical_plan) for block,path in (("C",args.c_analysis),("D",args.d_analysis))}
        validate_pair_provenance(bundles,source,args.fresh_plan)
    check_time()
    if args.output.exists():raise FileExistsError("table exports preserve every previous output")
    prepared={block:table_rows(bundle,names) for block,bundle in bundles.items()}
    args.output.mkdir(parents=True)
    allrows=[];texts=[]
    for block,rows in prepared.items():
        check_time();allrows.extend(rows)
        events=bundles[block]["analysis"]["validation_events" if block=="source" else "eligible_events"]
        tex=latex_table(rows,block,events);texts.append(tex)
        stem="source_information_primary16" if block=="source" else "fresh_"+block+"_primary16"
        (args.output/(stem+".tex")).write_text(tex,encoding="utf-8")
        write_csv(args.output/(stem+".csv"),rows)
    entry="source_information_primary16.tex" if args.mode=="source" else "fresh_information_primary16.tex"
    if args.mode=="fresh":(args.output/entry).write_text("\n".join(texts),encoding="utf-8")
    data={"schema":"information-table-data/v1","status":"COMPLETE_INFORMATION_"+args.mode.upper()+"_TABLES",
          "generated_utc":datetime.now(timezone.utc).isoformat(),"units":"nat per eligible event","seed_count":5,
          "seed_variation":"Sample SD across five paired optimizer seeds; no independent-data interpretation",
          "event_uncertainty":None if args.mode=="source" else "Approximate95% simultaneous16-contrast family separately per catalog;2000 launched-stratum paired resamples; fixed fitted models",
          "registered_run_scope":"240 registered run IDs in fresh scoring; not 240 independent or numerically distinct predictors. All declared runs retained; the 20 new Full5 runs may repeat the 20 old Full runs.",
          "rows":allrows,"row_count":len(allrows),"display_decimal_places":6,"entry_tex":entry,
          "statistical_plan_sha256":sha(args.statistical_plan),"budget_sha256":sha(args.budget),"deadline_unix":deadline,
          "analysis_inputs":{block:{"analysis_sha256":sha(bundle["paths"]["analysis.json"]),"analysis_manifest_sha256":bundle["manifest_sha256"],
                            "seed_contrasts_sha256":sha(bundle["paths"]["primary_seed_contrasts.csv"]),"summary_sha256":sha(bundle["paths"]["primary_contrast_summary.csv"])}
                            for block,bundle in bundles.items()},
          "source_analysis_sha256":sha(source["paths"]["analysis.json"]),"elapsed_seconds":time.time()-started,
          "rendering_limit":"LaTeX syntax statically checked; no TeX compiler/layout acceptance asserted"}
    save_json(args.output/"table_data.json",data)
    for block,bundle in bundles.items():
        folder=args.output/"source_tables"/block;folder.mkdir(parents=True)
        for name in ("analysis.json","primary_seed_contrasts.csv","primary_contrast_summary.csv","ANALYSIS_MANIFEST.json"):
            shutil.copyfile(bundle["root"]/name,folder/name)
        if block!="source":shutil.copyfile(bundle["paths"]["uncertainty.json"],folder/"uncertainty.json")
    snapshots=args.output/"source_snapshot";snapshots.mkdir()
    for name in ("build_information_tables.py","generate_information_figures.py"):
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    check_time()
    files=[{"path":p.relative_to(args.output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(args.output.rglob("*")) if p.is_file()]
    save_json(args.output/"TABLE_MANIFEST.json",{"schema":"information-table-manifest/v1","status":data["status"],
              "generated_utc":datetime.now(timezone.utc).isoformat(),"entry_tex":entry,"row_count":len(allrows),"files":files,
              "builder_sha256":sha(__file__),"input_checker_sha256":sha(Path(__file__).with_name("generate_information_figures.py"))})
    print(json.dumps({"status":data["status"],"rows":len(allrows),"entry":str(args.output/entry)}))


if __name__=="__main__":main()
