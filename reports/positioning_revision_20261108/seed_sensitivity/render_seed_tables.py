"""Presentation-only views of the complete post-hoc seed sensitivity result."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return {"path":str(p.resolve()),"sha256":sha(p),"bytes":p.stat().st_size}


def main():
    source=ROOT/"results/seed_sensitivity.json";d=json.loads(source.read_text(encoding="utf-8"))
    if d["status"]!="COMPLETE_POSTHOC_SEED_SENSITIVITY" or len(d["summary_rows"])!=32:raise ValueError("complete32 rows required")
    out=ROOT/"tex";out.mkdir(exist_ok=True)
    names=("seed_adjacent_summary.tex","seed_full_summary.tex","seed_exploratory_t_summary.tex","TABLE_MANIFEST.json")
    if any((out/n).exists() for n in names):raise FileExistsError("preserve earlier table export")
    selected=[r for r in d["summary_rows"] if r["contrast"].startswith("gain_change:") and "Full5_minus_Hidden5" not in r["contrast"]]
    if len(selected)!=12:raise ValueError("all three adjacent steps x two fractions x two catalogs required")
    lines=["% Post-hoc to known C/D outcomes; no fitting or catalog pooling.","\\begin{table}[!htbp]","\\centering","\\small","\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Post-hoc optimizer-seed sensitivity of the adjacent information steps. Effects and SD are in $10^{-3}$ nat per eligible event. $H,C2,C8,F$ denote Hidden5, Coarse2, Coarse8 and Full5. Signs are positive/negative counts among the same five paired seeds. Every leave-one-out (LOO) mean averages four seeds; its range is not a confidence interval. C and D remain separate evaluations of the same trained models. All16 contrasts per catalog and full seed ranges are retained in the supplementary seed tables.}\\label{tab:seed_adjacent_summary}",
        "\\begin{tabular}{lllrrr}","\\toprule Catalog & More minus less & Train (\\%) & Mean (SD) & $+/-$ & LOO mean range\\\\ \\midrule"]
    for r in selected:
        contrast,fraction=r["contrast"].split("/")
        label=contrast.removeprefix("gain_change:").replace("Coarse2","C2").replace("Coarse8","C8").replace("Full5","F").replace("Hidden5","H").replace("_minus_","--")
        lines.append(f"{r['catalog']} & {label} & {'10' if fraction=='f010' else '100'} & {1000*r['mean']:.3f} ({1000*r['sample_sd']:.3f}) & {r['positive_seeds']}/{r['negative_seeds']} & [{1000*r['leave_one_out_minimum_mean']:.3f}, {1000*r['leave_one_out_maximum_mean']:.3f}]\\\\")
    lines += ["\\bottomrule","\\end{tabular}","\\end{table}",""]
    (out/names[0]).write_text("\n".join(lines),encoding="utf-8",newline="\n")
    (out/names[1]).write_bytes((ROOT/"results/seed_sensitivity.tex").read_bytes())
    (out/names[2]).write_bytes((ROOT/"results/exploratory_t_intervals.tex").read_bytes())
    manifest={"schema":"posthoc-seed-sensitivity-table-export/v1","status":"COMPLETE_TABLE_EXPORT_REQUIRES_LAYOUT_CHECK",
        "input":record(source),"script":record(Path(__file__)),"main_adjacent_rows":12,"complete_supplement_rows":32,
        "table_labels":["tab:seed_adjacent_summary","tab:seed_sensitivity_C","tab:seed_sensitivity_D","tab:seed_sensitivity_t_C","tab:seed_sensitivity_t_D"],
        "files":[record(out/n) for n in names[:3]],"scientific_values_changed":False,"analysis_intent":"post_hoc_to_known_C_D_results"}
    (out/names[3]).write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":manifest["status"],"main_rows":12,"full_rows":32}))


if __name__=="__main__":main()
