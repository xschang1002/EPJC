"""Post-hoc optimizer-seed sensitivity on completed C/D score summaries.

No model training, inference, event resampling or modification of prior reports.
All16 contrasts are retained separately for each catalog. A seed is the paired
optimizer-run index, not a catalog/event or an independently resampled dataset.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

SEEDS=(3101,3102,3103,3104,3105)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return {"path":str(Path(path).resolve()),"sha256":sha(path),"bytes":Path(path).stat().st_size}
def csv_rows(path):
    with Path(path).open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
def save_json(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
def save_csv(path,rows):
    with Path(path).open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def sign_label(values):
    values=np.asarray(values)
    if np.all(values>0):return "positive"
    if np.all(values<0):return "negative"
    if np.all(values==0):return "zero"
    return "mixed_or_zero"


def sensitivity(values,seed_ids=SEEDS,t_critical=None):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(5,) or len(set(seed_ids))!=5 or not np.isfinite(values).all():
        raise ValueError("exactly five finite paired optimizer-seed values required")
    loo=np.asarray([np.mean(np.delete(values,i)) for i in range(5)])
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
    summary={"n_seeds":5,"mean":mean,"sample_sd":sd,"minimum":float(values.min()),"maximum":float(values.max()),
             "positive_seeds":int(np.sum(values>0)),"negative_seeds":int(np.sum(values<0)),"zero_seeds":int(np.sum(values==0)),
             "all_seed_direction":sign_label(values),"leave_one_out_minimum_mean":float(loo.min()),
             "leave_one_out_maximum_mean":float(loo.max()),"leave_one_out_direction":sign_label(loo),
             "leave_one_out_positive_means":int(np.sum(loo>0)),"leave_one_out_negative_means":int(np.sum(loo<0)),
             "leave_one_out_zero_means":int(np.sum(loo==0)),
             "largest_absolute_mean_shift_on_seed_omission":float(np.max(np.abs(loo-mean))),
             "leave_one_out_is_confidence_interval":False}
    omitted=[{"omitted_seed":int(seed_ids[i]),"remaining_seed_count":4,"mean":float(loo[i]),
              "mean_minus_full_five_seed_mean":float(loo[i]-mean)} for i in range(5)]
    t_result=None
    if t_critical is not None:
        half=float(t_critical)*sd/math.sqrt(5)
        t_result={"n_seeds":5,"degrees_of_freedom":4,"nominal_confidence":.95,"mean":mean,"sample_sd":sd,
                  "standard_error_s_over_sqrt5":sd/math.sqrt(5),"critical_t":float(t_critical),
                  "lower":mean-half,"upper":mean+half,
                  "direction":"positive" if mean-half>0 else "negative" if mean+half<0 else "includes_zero",
                  "multiplicity_adjusted":False,"default_main_table":False}
    return summary,omitted,t_result


def checked_tables(folder,block,names,review):
    folder=Path(folder);analysis_path=folder/"analysis.json";manifest_path=folder/"ANALYSIS_MANIFEST.json"
    analysis=read(analysis_path);manifest=read(manifest_path)
    if analysis.get("status")!="COMPLETE_INFORMATION_FRESH_ANALYSIS" or analysis.get("block_id")!=block:
        raise ValueError("complete matching fresh catalog analysis required")
    if analysis["model_count"]!=240 or analysis["primary_family_size"]!=16:
        raise ValueError("original complete240-run/16-contrast scope required")
    bound={(r["sha256"],r["bytes"]) for r in review["checked_files"]}
    for path in (analysis_path,manifest_path):
        if (sha(path),path.stat().st_size) not in bound:raise ValueError("input not bound by independent numerical PASS")
    manifest_rows={r["path"]:r for r in manifest["files"]}
    files=[analysis_path,manifest_path]
    for basename in ("primary_seed_contrasts.csv","primary_contrast_summary.csv"):
        path=folder/basename;expected=manifest_rows[basename]
        if sha(path)!=expected["sha256"] or path.stat().st_size!=expected["bytes"]:
            raise ValueError("frozen seed table bytes changed")
        files.append(path)
    rows=csv_rows(folder/"primary_seed_contrasts.csv")
    expected={(name,seed) for name in names for seed in SEEDS}
    keyed={(row["contrast"],int(row["seed"])):float(row["gain"]) for row in rows}
    if len(rows)!=80 or set(keyed)!=expected:raise ValueError("all16x5 paired seed contrasts required exactly once")
    summaries=csv_rows(folder/"primary_contrast_summary.csv")
    if len(summaries)!=16 or {r["contrast"] for r in summaries}!=set(names):raise ValueError("complete old summaries required")
    summaries={r["contrast"]:r for r in summaries}
    for name in names:
        v=np.asarray([keyed[name,seed] for seed in SEEDS])
        actual={"mean":v.mean(),"sample_sd":v.std(ddof=1),"min":v.min(),"max":v.max(),
                "positive_seeds":np.sum(v>0),"negative_seeds":np.sum(v<0),"zero_seeds":np.sum(v==0)}
        for field,value in actual.items():
            if not np.isclose(float(summaries[name][field]),value,rtol=1e-12,atol=1e-14):
                raise ValueError("source seed summary does not reproduce")
    return keyed,files


def short_name(name):
    label,fraction=name.split("/");fraction="10" if fraction=="f010" else "100"
    aliases={"Hidden5":"H","Coarse2":"C2","Coarse8":"C8","Full5":"F"}
    if label.startswith("gain:"):
        return "$G_{"+aliases[label[5:]]+"}$",fraction
    more,less=label.removeprefix("gain_change:").split("_minus_")
    return "$G_{"+aliases[more]+"}-G_{"+aliases[less]+"}$",fraction


def tex_tables(rows,path,exploratory=False):
    output=["% Generated post-hoc to known C/D outcomes. Do not edit numerical cells.",
            "% Requires booktabs and longtable. Numeric effects are shown x 10^-3 nat/eligible event."]
    for block in ("C","D"):
        output += ["\\begingroup", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}"]
        if not exploratory:
            output += ["\\begin{longtable}{llrrrrl}",
                "\\caption{Post-hoc paired optimizer-seed sensitivity on catalog "+block+". All values except signs/fractions are in $10^{-3}$ nat per eligible event. $H,C2,C8,F$ denote Hidden5, Coarse2, Coarse8 and Full5. Signs list positive/negative/zero counts among five seeds. Each leave-one-out (LOO) mean averages the remaining four seeds; its range is not a confidence interval. C and D use the same five trained-seed pairs and are not pooled.}\\label{tab:seed_sensitivity_"+block+"}\\\\",
                "\\toprule Contrast & Train (\\%) & Mean (SD) & Seed range & $+/ - /0$ & LOO mean range & LOO sign\\\\ \\midrule \\endfirsthead",
                "\\toprule Contrast & Train (\\%) & Mean (SD) & Seed range & $+/ - /0$ & LOO mean range & LOO sign\\\\ \\midrule \\endhead"]
            for row in rows:
                if row["catalog"]!=block:continue
                name,fraction=short_name(row["contrast"])
                output.append(f"{name} & {fraction} & {1000*row['mean']:.3f} ({1000*row['sample_sd']:.3f}) & [{1000*row['minimum']:.3f}, {1000*row['maximum']:.3f}] & {row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']} & [{1000*row['leave_one_out_minimum_mean']:.3f}, {1000*row['leave_one_out_maximum_mean']:.3f}] & {row['leave_one_out_direction'].replace('_',' ')}\\\\")
        else:
            output += ["\\begin{longtable}{llrrl}",
                "\\caption{Optional exploratory Student-$t$ intervals on catalog "+block+"; not the default sensitivity table. With $n=5$, these nominal unadjusted 95\\% intervals assume independent paired optimizer-seed draws and an approximately normal distribution of seed-level contrasts, conditional on the fixed training data, validation rule and evaluated catalog. They exclude training-dataset and simulator uncertainty, are not event-bootstrap intervals, and are not jointly calibrated over contrasts or C/D. Values are in $10^{-3}$ nat per eligible event.}\\label{tab:seed_sensitivity_t_"+block+"}\\\\",
                "\\toprule Contrast & Train (\\%) & Mean & Exploratory $t$ interval & Direction\\\\ \\midrule \\endfirsthead",
                "\\toprule Contrast & Train (\\%) & Mean & Exploratory $t$ interval & Direction\\\\ \\midrule \\endhead"]
            for row in rows:
                if row["catalog"]!=block:continue
                name,fraction=short_name(row["contrast"])
                output.append(f"{name} & {fraction} & {1000*row['mean']:.3f} & [{1000*row['lower']:.3f}, {1000*row['upper']:.3f}] & {row['direction'].replace('_',' ')}\\\\")
        output += ["\\bottomrule", "\\end{longtable}","\\endgroup",""]
    with Path(path).open("x",encoding="utf-8") as stream:stream.write("\n".join(output))


def run(args):
    started=time.time();plan=read(args.plan);review=read(args.numerical_review)
    if plan["status"]!="FROZEN_FRESH_STATISTICAL_ANALYSIS" or review["status"]!="PASS_COMPLETE_INFORMATION_NUMERICAL_REVIEW":
        raise ValueError("original complete design and independent result check required")
    names=[r["name"] for r in plan["primary_contrasts"]]
    if len(names)!=16 or len(set(names))!=16:raise ValueError("original16 contrast scope required")
    output_names=["seed_sensitivity.json","seed_sensitivity.csv","paired_seed_values.csv","leave_one_seed_out.csv","seed_sensitivity.tex","SCIENTIFIC_INTERPRETATION.md","MANIFEST.json"]
    if args.include_exploratory_t:output_names += ["exploratory_t_intervals.csv","exploratory_t_intervals.tex"]
    if any((args.output/name).exists() for name in output_names):raise FileExistsError("choose a new output directory; prior sensitivity records are immutable")
    args.output.mkdir(parents=True,exist_ok=True)
    critical=None;scipy_version=None
    if args.include_exploratory_t:
        import scipy
        from scipy.stats import t
        scipy_version=scipy.__version__;critical=float(t.ppf(.975,df=4))
    summaries=[];seed_rows=[];loo_rows=[];t_rows=[];inputs=[args.plan,args.numerical_review]
    for block,folder in (("C",args.analysis_c),("D",args.analysis_d)):
        keyed,files=checked_tables(folder,block,names,review);inputs+=files
        for name in names:
            values=[keyed[name,seed] for seed in SEEDS]
            summary,omitted,t_result=sensitivity(values,t_critical=critical)
            summaries.append({"catalog":block,"contrast":name,**summary})
            seed_rows += [{"catalog":block,"contrast":name,"seed":seed,"gain_nat_per_eligible_event":value} for seed,value in zip(SEEDS,values)]
            loo_rows += [{"catalog":block,"contrast":name,**row} for row in omitted]
            if t_result is not None:t_rows.append({"catalog":block,"contrast":name,**t_result})
    result={"schema":"posthoc-paired-optimizer-seed-sensitivity/v1","status":"COMPLETE_POSTHOC_SEED_SENSITIVITY",
            "generated_utc":datetime.now(timezone.utc).isoformat(),"analysis_intent":"post_hoc_to_known_C_D_results",
            "catalogs_separate":["C","D"],"contrasts_each":16,"optimizer_seeds":list(SEEDS),"n_optimizer_seeds":5,
            "estimator":"Arithmetic mean of within-seed paired log-score contrasts; each catalog analyzed separately",
            "units":"nat per eligible event","mean_sign":"Positive gain means Corrected beats Raw; gain_change is more-informed gain minus less-informed gain",
            "leave_one_out_method":"Drop one optimizer-seed contrast and average the remaining four; report all five values and their extrema/signs",
            "leave_one_out_is_confidence_interval":False,
            "conditioning":"One shared source training dataset, its fixed fractions/validation selection and paired RNG schedule; same five trained-seed pairs evaluated on both catalogs. No independent training-data resampling.",
            "catalog_pooling":"None. C/D do not create ten optimizer seeds or independent retraining replicates.",
            "t_interval_recommendation":{"default_main_presentation":False,"computed_as_optional_appendix":args.include_exploratory_t,
                "reason":"With only five seeds, the normal-shape assumption cannot be assessed reliably. Prefer the full sign/range/leave-one-out display; optional Student-t numbers are an explicitly assumption-dependent exploratory sensitivity, not the headline confidence claim.",
                "assumptions":"Independent paired optimizer-seed draws from the same stochastic training procedure and approximately normal seed-level contrasts, conditional on the fixed training/validation/evaluation inputs.",
                "excludes":"Training-dataset variation, validation-set variation, simulator-setting uncertainty and total training uncertainty; no multiplicity adjustment or joint-C/D coverage.",
                "formula":"mean +/- t(0.975,4)*sample_sd/sqrt(5)"},
            "summary_rows":summaries,"optional_exploratory_t_rows":t_rows,"input_files":[record(p) for p in inputs],
            "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy_version},
            "source_sha256":sha(Path(__file__)),"elapsed_seconds":time.time()-started}
    save_json(args.output/"seed_sensitivity.json",result)
    save_csv(args.output/"seed_sensitivity.csv",summaries);save_csv(args.output/"paired_seed_values.csv",seed_rows)
    save_csv(args.output/"leave_one_seed_out.csv",loo_rows);tex_tables(summaries,args.output/"seed_sensitivity.tex")
    if t_rows:
        save_csv(args.output/"exploratory_t_intervals.csv",t_rows);tex_tables(t_rows,args.output/"exploratory_t_intervals.tex",True)
    selected=[r for r in summaries if r["contrast"].startswith("gain_change:")]
    lines=["# Post-hoc optimizer-seed sensitivity","",
        "This analysis was added after the C/D outcomes were known. It uses the existing five paired optimizer seeds for every one of the16 contrasts on each catalog. No fitting, inference, event bootstrap or result selection was repeated. Both catalogs evaluate the same trained models and remain separate; they do not supply ten optimizer seeds.","",
        "## Main reporting decision","",
        "Use the full seed mean/SD/range/sign table and all five leave-one-seed-out means as the default. The leave-one-out range describes omission sensitivity, not a confidence interval. A stable omission sign means that removing any single observed seed leaves the observed mean sign unchanged; it does not prove a population effect or robustness to different training data.","",
        "Optional Student-t intervals are supplied in a separate file only. They require n=5 independent paired optimizer-seed draws and an approximately normal contrast distribution conditional on this source dataset, validation rule and catalog. These assumptions cannot be convincingly checked with five values. The intervals are unadjusted, exploratory, exclude dataset/simulator variation, and must not replace the registered fixed-model event intervals or be described as total training uncertainty.","",
        "## Every information-addition contrast","","| Catalog | Contrast | Seed mean | Seed SD | + / - / 0 | LOO mean range | LOO direction |","|---|---|---:|---:|---|---|---|"]
    for r in selected:lines.append(f"| {r['catalog']} | {r['contrast']} | {r['mean']:+.9f} | {r['sample_sd']:.9f} | {r['positive_seeds']}/{r['negative_seeds']}/{r['zero_seeds']} | [{r['leave_one_out_minimum_mean']:+.9f}, {r['leave_one_out_maximum_mean']:+.9f}] | {r['leave_one_out_direction']} |")
    lines += ["","The main rise at Coarse2-minus-Hidden5 and subsequent fall at Coarse8-minus-Coarse2 must be distinguished from the final step. Report the signs in the table, including any final-step or direct-gain seed exceptions, rather than treating a five-seed mean as unanimous optimizer support.","",
              "All32 summary rows,160 original paired seed values and160 omission means are saved in CSV. The TeX main table includes every contrast for each catalog; the optional t table is deliberately separate. None of the earlier manifests or frozen analysis files were rewritten.",""]
    with (args.output/"SCIENTIFIC_INTERPRETATION.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines))
    outputs=[record(args.output/name) for name in output_names if name!="MANIFEST.json"]
    save_json(args.output/"MANIFEST.json",{"schema":"posthoc-seed-sensitivity-artifacts/v1","status":result["status"],
              "generated_utc":datetime.now(timezone.utc).isoformat(),"analysis_intent":result["analysis_intent"],"inputs":result["input_files"],
              "analysis_script":record(Path(__file__)),"outputs":outputs,"no_training_or_inference":True})
    print(json.dumps({"status":result["status"],"rows":len(summaries),"seed_rows":len(seed_rows),"omission_rows":len(loo_rows),"elapsed_seconds":time.time()-started}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("analysis-c","analysis-d","plan","numerical-review","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--include-exploratory-t",action="store_true")
    run(p.parse_args())


if __name__=="__main__":main()
