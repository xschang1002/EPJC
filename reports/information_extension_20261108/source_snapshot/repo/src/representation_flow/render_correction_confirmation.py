"""Plot verified all160 confirmation beside verified validation analyses.

Only selected JSON/manifest files are read. Recorded original absolute paths
are provenance, never file locations to follow. No raw A/B, caches, weights,
training or inference are accessed by this tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from representation_flow.verify_correction_confirmation import (
    BLOCK_EVIDENCE, SEEDS, SCOPE, TABLE_COUNTS, _contrast_design, _model_design,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRAST = "Raw_minus_CorrectedPhysics"
DATASETS = ("Validation", "A", "B")
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "^", "s")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, message):
    require(finite(actual) and math.isclose(actual, expected, rel_tol=5e-10, abs_tol=5e-10), message)


def index(rows, keys, label):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, f"duplicate {label}: {key}")
        result[key] = row
    return result


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def load_analysis(path, kind):
    path = Path(path).resolve()
    manifest_path = path.with_name("analysis_manifest.json")
    manifest, data = read_json(manifest_path), read_json(path)
    matches = [r for r in manifest.get("outputs", []) if r["name"] == path.name]
    require(len(matches) == 1 and matches[0]["sha256"] == digest(path), f"{kind} analysis does not match its manifest")
    require(isinstance(data.get("experiment_identity"), str) and bool(data["experiment_identity"]), f"{kind} experiment identity missing")
    if kind == "core":
        require(data.get("status") == "COMPLETE_CORRECTION_STUDY_DESCRIPTIVE_ANALYSIS" and not data.get("failed_runs"), "core analysis is not complete")
        rows = data.get("cells", [])
        expected = set(itertools.product(SEEDS, ("f010", "f100"), ("Full", "Hidden"), ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")))
        keyed = index(rows, ("seed", "fraction_tag", "context", "representation"), "core cell")
        require(len(rows) == 60 and set(keyed) == expected, "core requires all 60 unique cells")
        require(all(r.get("maximum_updates") == 20000 and finite(r.get("validation_nll")) for r in rows), "core budget or finite NLL failed")
        cells = {(seed, tag, "Small", context, rep): r for (seed, tag, context, rep), r in keyed.items()}
    else:
        require(data.get("status") == "COMPLETE_CORRECTED_FORMAL_120" and data.get("analysis_status") == "VERIFIED_CORRECTED_FORMAL_120_DESCRIPTIVE_ANALYSIS", "Full120 independent analysis is incomplete")
        require((data.get("new_cells"), data.get("reused_cells")) == (100, 20), "Full120 new/reused count differs")
        rows = data.get("runs", [])
        keyed = index(rows, ("seed", "sample_fraction", "capacity", "representation"), "Full120 cell")
        expected = set(itertools.product(SEEDS, (.1, .3, 1.), ("Small", "Large"), ("Raw", "Simple", "Learned", "CorrectedPhysics")))
        require(len(rows) == 120 and set(keyed) == expected and len({r["matrix_cell_id"] for r in rows}) == 120, "Full120 requires all 120 unique cells")
        require(all(r.get("max_updates") == 20000 and finite(r.get("validation_nll")) and r.get("context_label", "Full") == "Full" for r in rows), "Full120 budget, context or finite NLL failed")
        require(all(type(r.get("reused")) is bool for r in rows) and sum(r["reused"] for r in rows) == 20, "Full120 reuse flags differ")
        cells = {(seed, f"f{round(frac*100):03d}", cap, "Full", rep): r for (seed, frac, cap, rep), r in keyed.items()}
    return data, cells, [file_record(path), file_record(manifest_path)]


def validate_confirmation(data, verification, summary_path, core, formal):
    require(data.get("schema") == "frozen-core60-full120-confirmation-results/v2" and data.get("status") == "COMPLETE_FROZEN_UNIQUE160_AB_CONFIRMATION", "confirmation is not complete all160")
    require(data.get("scope") == SCOPE and data.get("model_evaluations") == 320, "confirmation must contain 320 all160 evaluations")
    require(verification.get("schema") == "correction-confirmation-output-verification/v1" and verification.get("status") == "VERIFIED_FROZEN_UNIQUE160_AB_CONFIRMATION_OUTPUTS", "independent confirmation verification is absent or incomplete")
    require(verification.get("scope") == SCOPE and verification.get("unique_models") == 160 and verification.get("reused_formal_cells") == 20, "verification scope/reuse mismatch")
    require(verification.get("verified_counts") == TABLE_COUNTS, "verification does not cover all five complete tables")
    require(all(len(data.get(name, [])) == count for name, count in TABLE_COUNTS.items()), "a complete confirmation table is missing")
    # Role selects the bound input; its old absolute path is deliberately ignored.
    records = [r for r in verification.get("input_files", []) if r.get("role") == "confirmation_summary.json"]
    require(len(records) == 1 and records[0]["sha256"] == digest(summary_path) and records[0]["bytes"] == Path(summary_path).stat().st_size, "selected confirmation summary is not the independently verified file")
    require(data.get("scoring_plan_sha256") == verification.get("scoring_plan_sha256"), "confirmation scoring-plan identity differs")
    core_id, extension_id = core["experiment_identity"], formal["experiment_identity"]
    require(formal.get("core_experiment_identity") == core_id and data.get("core_experiment_identity") == core_id and verification.get("core_experiment_identity") == core_id, "core experiment identities differ")
    require(data.get("extension_gate", {}).get("experiment_identity") == extension_id and verification.get("extension_experiment_identity") == extension_id, "Full120 experiment identities differ")
    require(data["extension_gate"].get("status") == "COMPLETE_FROZEN_EXTENSION120_VERIFIED" and data["extension_gate"].get("scored_here") is True, "Full120 was not included in frozen confirmation")
    models = index(data["models"], ("block", "seed", "fraction_tag", "capacity", "context", "representation"), "confirmation model")
    require(set(models) == {(block, *key) for block, key in itertools.product(("a", "b"), _model_design())}, "complete unique160 model roster differs")
    require(len({(r["block"], r["run_id"]) for r in data["models"]}) == 320, "reused model was counted twice")
    for (block, *_), row in models.items():
        require(row["events"] == BLOCK_EVIDENCE[block][0] and finite(row.get("mean")), "model event count/NLL is invalid")
    definitions = _contrast_design()
    contrasts = index(data["seed_contrasts"], ("block", "seed", "scope", "fraction_tag", "capacity", "context", "contrast"), "confirmation contrast")
    require(set(contrasts) == {(b, s, *labels) for b, s, (labels, _) in itertools.product(("a", "b"), SEEDS, definitions)}, "complete 410-contrast roster differs")
    for block, seed, (labels, terms) in itertools.product(("a", "b"), SEEDS, definitions):
        row = contrasts[block, seed, *labels]
        close(row["mean"], math.fsum(weight * models[block, seed, *model_key]["mean"] for weight, model_key in terms), "confirmation contrast does not reconstruct from model NLL")
        require(row["events"] == BLOCK_EVIDENCE[block][0], "contrast event count differs")
    summaries = index(data["contrast_summary"], ("block", "scope", "fraction_tag", "capacity", "context", "contrast"), "confirmation summary")
    require(set(summaries) == {(b, *labels) for b, (labels, _) in itertools.product(("a", "b"), definitions)}, "all 82 five-seed summaries are required")
    for (block, *labels), row in summaries.items():
        values = [contrasts[block, seed, *labels]["mean"] for seed in SEEDS]
        require(row["seeds"] == 5, "wrong seed count")
        for field, value in (("mean", statistics.mean(values)), ("sample_sd", statistics.stdev(values)), ("minimum", min(values)), ("maximum", max(values))):
            close(row[field], value, "five-seed descriptive summary differs")
        for field, value in (("positive_seeds", sum(v > 0 for v in values)), ("negative_seeds", sum(v < 0 for v in values)), ("zero_seeds", sum(v == 0 for v in values))):
            require(row[field] == value, "seed sign count differs")
    return contrasts


def prepare_tables(core_path, formal_path, summary_path, verification_path):
    core, core_cells, sources = load_analysis(core_path, "core")
    formal, formal_cells, records = load_analysis(formal_path, "formal")
    sources.extend(records)
    data, verification = read_json(summary_path), read_json(verification_path)
    contrasts = validate_confirmation(data, verification, summary_path, core, formal)
    sources.extend((file_record(summary_path), file_record(verification_path)))
    for row in data["models"]:
        key = tuple(row[name] for name in ("seed", "fraction_tag", "capacity", "context", "representation"))
        source = formal_cells.get(key, core_cells.get(key))
        require(source is not None and row["run_id"] == source["run_id"], "confirmation/validation model identity differs")
    # Preserve the shared-model meaning of the 20 reused Full cells.
    for key, row in formal_cells.items():
        seed, tag, cap, _, rep = key
        reused = cap == "Small" and tag in ("f010", "f100") and rep in ("Raw", "CorrectedPhysics")
        require(row["reused"] is reused, "unexpected reused Full120 cell")
        if reused:
            require(row["run_id"] == core_cells[key]["run_id"], "reused core run identity differs")
            close(row["validation_nll"], core_cells[key]["validation_nll"], "reused validation NLL differs")
    output = {"core60": [], "formal120": []}
    designs = {"core60": list(itertools.product(("Full", "Hidden"), ("Small",), ("f010", "f100"))),
               "formal120": list(itertools.product(("Full",), ("Small", "Large"), ("f010", "f030", "f100")))}
    for scope, design in designs.items():
        cells = core_cells if scope == "core60" else formal_cells
        for context, capacity, tag in design:
            for seed in SEEDS:
                raw, corrected = (cells[seed, tag, capacity, context, rep] for rep in ("Raw", "CorrectedPhysics"))
                labels = {"scope": scope, "context": context, "capacity": capacity, "fraction_tag": tag, "seed": seed, "contrast": CONTRAST}
                output[scope].append({**labels, "dataset": "Validation", "nll_difference": raw["validation_nll"] - corrected["validation_nll"]})
                for block in ("a", "b"):
                    row = contrasts[block, seed, scope, tag, capacity, context, CONTRAST]
                    output[scope].append({**labels, "dataset": block.upper(), "nll_difference": row["mean"]})
    require(len(output["core60"]) == 60 and len(output["formal120"]) == 90, "display tables are incomplete")
    return output, sources


def summarize(rows):
    keys = ("scope", "context", "capacity", "fraction_tag", "dataset", "contrast")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        require(sorted(r["seed"] for r in group) == list(SEEDS), "all five unique seeds are required per plotted group")
        values = [r["nll_difference"] for r in group]
        require(all(finite(v) for v in values), "nonfinite plotted contrast")
        output.append({**dict(zip(keys, key)), "seeds": 5, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                       "positive_seeds": sum(v > 0 for v in values), "negative_seeds": sum(v < 0 for v in values), "zero_seeds": sum(v == 0 for v in values)})
    return output


def limits(rows):
    values = [0.] + [r["nll_difference"] for r in rows]
    for row in summarize(rows):
        values.extend((row["mean"]-row["sample_sd"], row["mean"]+row["sample_sd"]))
    pad = max((max(values)-min(values))*.13, 1e-5)
    return min(values)-pad, max(values)+pad


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def draw(rows, scope, output, plt):
    core = scope == "core60"
    tags = ("f010", "f100") if core else ("f010", "f030", "f100")
    row_labels = ("Full", "Hidden") if core else ("Small", "Large")
    fig, axes = plt.subplots(2, len(tags), figsize=(180/25.4, (133 if core else 139)/25.4), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.13, right=.985, bottom=.235, top=.84, wspace=.15, hspace=.31)
    bound = limits(rows)
    for i, label in enumerate(row_labels):
        for j, tag in enumerate(tags):
            ax = axes[i, j]
            context, capacity = (label, "Small") if core else ("Full", label)
            for x, dataset in enumerate(DATASETS):
                group = sorted((r for r in rows if r["context"]==context and r["capacity"]==capacity and r["fraction_tag"]==tag and r["dataset"]==dataset), key=lambda r:r["seed"])
                require([r["seed"] for r in group] == list(SEEDS), "cannot draw a missing or repeated seed")
                values = [r["nll_difference"] for r in group]
                ax.scatter([x+d for d in (-.10,-.05,0.,.05,.10)], values, color=COLORS[x], marker=MARKERS[x], s=22, edgecolors="white", linewidths=.35, zorder=3)
                ax.errorbar(x+.22, statistics.mean(values), yerr=statistics.stdev(values), fmt="D", color="#202020", markersize=3.8, capsize=2.5, linewidth=1., zorder=4)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",color="#dddddd",linewidth=.5);ax.set_axisbelow(True)
            ax.axhline(0,color="#555555",linewidth=.8)
            ax.ticklabel_format(axis="y",style="plain",useOffset=False)
            ax.set(title=f"{label} · {int(tag[1:])}% training", xlim=(-.35,2.48), ylim=bound, xticks=(0,1,2), xticklabels=DATASETS)
            ax.tick_params(axis="x",labelbottom=True)
    fig.supylabel("Raw NLL − Corrected NLL (nat/event)", x=.018, fontsize=9)
    fig.suptitle(f"{'Core60' if core else 'Full120'} primary contrast: validation and frozen same-range blocks", y=.975, fontsize=10)
    fig.text(.5,.92,"Colored shapes: all five seeds     Black diamonds and bars: mean ± sample SD",ha="center",fontsize=8)
    fig.text(.5,.105,"Validation selected checkpoints. A/B are same-range blocks with prior W6/MDN use; not globally untouched or OOD.\nSeed SD is conditional on shared source data, not an independent-data CI. Positive differences favor CorrectedPhysics.\nRepeated Small Full panels use the same core/extension models; they are overlapping evidence, not additional fits.",ha="center",va="center",fontsize=7.15)
    stem = "figure5_core_validation_confirmation" if core else "figure6_full_validation_confirmation"
    files=[]
    for extension in ("pdf","png"):
        path=output/f"{stem}.{extension}";fig.savefig(path,dpi=600,facecolor="white",transparent=False);files.append(path)
    plt.close(fig)
    for suffix, values in (("seed_contrasts",rows),("summary",summarize(rows))):
        path=output/f"{scope}_{suffix}.csv";write_csv(path,values);files.append(path)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-analysis",type=Path,required=True)
    parser.add_argument("--formal-analysis",type=Path,required=True)
    parser.add_argument("--confirmation",type=Path,required=True,help="Selected completed confirmation_summary.json")
    parser.add_argument("--verification",type=Path,required=True,help="Independent verification_manifest.json bound to that summary")
    parser.add_argument("--output",type=Path,required=True,help="New empty figure directory")
    args=parser.parse_args()
    tables,sources=prepare_tables(args.core_analysis,args.formal_analysis,args.confirmation,args.verification)
    output=args.output.resolve()
    require(not output.exists() or not any(output.iterdir()),"output is not empty; preserve it and choose a new directory")
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style={"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.5,"axes.labelsize":8.5,"xtick.labelsize":7.6,"ytick.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42,"text.color":"#202020","axes.labelcolor":"#202020","axes.linewidth":.7}
    files=[]
    with plt.rc_context(style):
        for scope,rows in tables.items(): files.extend(draw(rows,scope,output,plt))
    snapshot=output/"renderer_snapshot.py";snapshot.write_bytes(Path(__file__).read_bytes());files.append(snapshot)
    verifier_source=Path(__file__).with_name("verify_correction_confirmation.py")
    provenance={"schema":"verified-confirmation-figures/v1","status":"RENDERED_VERIFIED_ALL160_CONFIRMATION_COMPARISONS",
        "scope":"Prespecified Raw-minus-CorrectedPhysics only; full source tables retain all 82 confirmation and 23 Full120 validation primary groups",
        "source_files":sources,"tool_sources":[file_record(Path(__file__)),file_record(verifier_source)],
        "matplotlib":matplotlib.__version__,"uncertainty":"Sample SD (ddof=1) of all five optimizer seeds conditional on shared source data; not an independent-data confidence interval",
        "selection":"Validation selected checkpoints; A/B did not select/refit the frozen models and retain prior W6/MDN use",
        "portability":"Only explicit current file arguments and their sibling analysis manifests are read; recorded old absolute paths are provenance only",
        "transforms":"Reconstruct all 410 confirmation contrasts and 82 seed summaries; select only fixed Raw-minus-Corrected contrast; preserve each seed and sign; deterministic x separation only; shared y limits per figure include zero, all points and complete SD bars",
        "alt_text":{"core60":"Four panels for Full/Hidden and 10%/100% show all five Raw-minus-Corrected seed contrasts on validation, A and B, with black mean diamonds and sample SD, a shared vertical scale and zero reference.",
                    "formal120":"Six panels for Small/Large Full conditioning and 10%/30%/100% show all five validation/A/B Raw-minus-Corrected seed contrasts, with sample SD and one common vertical scale. Core and extension reused models remain shared evidence."},
        "export":{"width_mm":180,"height_mm":{"core60":133,"formal120":139},"png_dpi":600,"background":"opaque white","pdf_fonttype":42,"bbox_inches":None,"journal_compliance":"Not asserted; general scientific review figures"},
        "outputs":[file_record(path) for path in files]}
    provenance["procedural_reference"] = "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065"
    notes=output/"FIGURE_NOTES.txt"
    notes.write_text("Frozen all160 confirmation comparison figures\n\n"+provenance["scope"]+"\n"+provenance["uncertainty"]+"\n"+provenance["selection"]+"\n\n"+"\n".join(provenance["alt_text"].values())+"\n\nProcedural software reference: "+provenance["procedural_reference"]+"\n",encoding="utf-8")
    provenance["outputs"].append(file_record(notes))
    (output/"figure_manifest.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":provenance["status"],"output":str(output),"core_seed_rows":len(tables["core60"]),"formal_seed_rows":len(tables["formal120"])}))


if __name__=="__main__":
    main()
