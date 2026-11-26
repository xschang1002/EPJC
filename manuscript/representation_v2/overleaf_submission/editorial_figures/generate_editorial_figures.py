"""Render complete information-study summaries without raw data or inference.

This presentation program is outside frozen numerical src/. All input analysis
files are verified against their manifests before any output directory is made.
Publication destination: The European Physical Journal C (EPJC) submission.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
CONTEXTS=("Hidden5","Coarse2","Coarse8","Full5")
CONTEXT_TEXT=("No direct\nmomentum","2-bin\nmomentum","8-bin\nmomentum","Exact\nmomentum")
DISPLAY_NAMES={"CorrectedPhysics":"BB-centered","Hidden5":"No direct momentum",
 "Coarse2":"2-bin momentum","Coarse8":"8-bin momentum","Full5":"Exact momentum",
 "Simple":"Linear center","Learned":"Polynomial-ridge center"}
ORIGINAL_RENDERER_SHA256="853d2a05efc0a2b17d261fd460df694b6939dbc7f6523cb068dab5bcddd4f301"
SOURCE_COLLAPSE_CHECKS=[]
LAYOUT_CHECKS=[]
FRACTIONS=("f010","f100")
SEEDS=(3101,3102,3103,3104,3105)
COLORS=("#0072B2","#D55E00","#009E73","#CC79A7","#6B6B6B")
MARKERS=("o","s","^","D","v")
COMPONENTS=("negative_log_base_density","negative_spline_logabsdet","log_preprocessing_scale","log_support_cdf","nll_gain")
COMPONENT_TEXT=("Base density","Spline Jacobian","Preprocessing scale","Support normalization","Total gain")
DEADLINE=None
SKILL_CITATION={"authors":["Timothy Kassis","Vinayak Agarwal","Yuhuan He","Darshil Patel","Aubrey M. Brueckner"],
    "year":2026,"title":"Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
    "doi":"10.48550/arXiv.2609.00065","url":"https://arxiv.org/abs/2609.00065",
    "version_checked":"v2; 2026-11-08","role":"Visualization software/procedural assistance; not evidence for the physical or statistical findings"}


def sha(path):
    check_time()
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def check_time():
    if DEADLINE is not None and time.time()>=DEADLINE:
        raise TimeoutError("original authorized research deadline exceeded during figure preparation")


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def safe_file(root,relative):
    name=PurePosixPath(str(relative).replace("\\","/"))
    path=(Path(root)/name).resolve()
    if name.is_absolute() or ".." in name.parts or ":" in str(name) or not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("unsafe or missing analysis file")
    return path


def load_bundle(root,block,plan):
    root=Path(root).resolve();manifest=read_json(root/"ANALYSIS_MANIFEST.json")
    status="COMPLETE_INFORMATION_SOURCE_ANALYSIS" if block=="source" else "COMPLETE_INFORMATION_FRESH_ANALYSIS"
    if manifest.get("schema")!="information-analysis-output/v1" or manifest.get("status")!=status:
        raise ValueError("complete independently checked analysis is required for every figure")
    paths={}
    for row in manifest["files"]:
        path=safe_file(root,row["path"])
        if row["path"] in paths or path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]:
            raise ValueError("analysis file changed or duplicated after completion")
        paths[row["path"]]=path
    required={"analysis.json","execution_state.json","model_summary.csv","primary_seed_contrasts.csv",
              "primary_contrast_summary.csv","density_component_diagnostics.json","group_seed_variation.csv"}
    required|={"source_location_diagnostics.json"} if block=="source" else {"uncertainty.json","group_uncertainty.csv"}
    if not required.issubset(paths):raise ValueError("analysis manifest omitted a figure source")
    analysis=read_json(paths["analysis.json"])
    if (analysis.get("status")!=status or read_json(paths["execution_state.json"]).get("status")!="COMPLETE"
            or analysis["statistical_plan_sha256"]!=sha(plan)):
        raise ValueError("analysis is incomplete or belongs to another statistical plan")
    for name,digest in analysis["analysis_sources"].items():
        if sha(safe_file(ROOT,name))!=digest:raise ValueError("analysis source code changed")
    if block=="source":
        if analysis.get("model_count")!=80 or analysis.get("all_validation_predictions_replayed") is not True:
            raise ValueError("all 80 source checkpoints must have replayed")
    elif (analysis.get("block_id")!=block or analysis.get("model_count")!=240 or analysis.get("new_models")!=80
            or analysis.get("legacy_models")!=160 or analysis.get("launched_events")!=500000 or analysis.get("primary_family_size")!=16):
        raise ValueError("fresh analysis does not cover its exact registered 240-run catalog")
    bundle={"root":root,"block":block,"analysis":analysis,"paths":paths,"manifest_sha256":sha(root/"ANALYSIS_MANIFEST.json"),
            "seed_rows":read_csv(paths["primary_seed_contrasts.csv"]),"model_rows":read_csv(paths["model_summary.csv"]),
            "component_rows":read_json(paths["density_component_diagnostics.json"])["rows"],
            "group_seed_rows":read_csv(paths["group_seed_variation.csv"])}
    expected_names={r["name"] for r in read_json(plan)["primary_contrasts"]}
    key={(row["contrast"],int(row["seed"])) for row in bundle["seed_rows"]}
    if len(key)!=80 or len(bundle["seed_rows"])!=80 or key!=set(itertools.product(expected_names,SEEDS)):
        raise ValueError("all 16 contrasts and all five paired seeds must be present")
    if block!="source":
        uncertainty=read_json(paths["uncertainty.json"])
        if (uncertainty.get("status")!="COMPLETE" or uncertainty.get("n_resamples")!=2000
                or uncertainty.get("global_family_size")!=16 or uncertainty.get("joint_primary_group_family_size")!=336
                or uncertainty.get("confidence")!=.95):
            raise ValueError("uncertainty is incomplete or from another interval family")
        bundle["uncertainty"]={r["name"]:r for r in uncertainty["effects"]}
        bundle["groups"]={(r["primary"],r["group_id"]):r for r in uncertainty["primary_groups"]}
        if set(bundle["uncertainty"])!=expected_names or len(bundle["groups"])!=336:
            raise ValueError("uncertainty tables lack declared contrasts/groups")
        for name,row in bundle["uncertainty"].items():
            values=seed_values(bundle,name)
            if not np.isclose(np.mean(values),row["estimate"],rtol=0,atol=1e-10):
                raise ValueError("seed means and event-bootstrap point estimates disagree")
    return bundle


def seed_values(bundle,name):
    rows={int(r["seed"]):float(r["gain"]) for r in bundle["seed_rows"] if r["contrast"]==name}
    if set(rows)!=set(SEEDS):raise ValueError("missing paired optimizer seed")
    values=np.asarray([rows[s] for s in SEEDS])
    if not np.isfinite(values).all():raise ValueError("nonfinite plotted gain")
    return values


def common_range(values,zero=False):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError("nonfinite plotted range")
    low,high=float(np.min(values)),float(np.max(values))
    if zero:low=min(low,0);high=max(high,0)
    pad=max(high-low,1e-5)*.08
    return low-pad,high+pad


def style_axes(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#E4E4E4",linewidth=.45)
    ax.set_axisbelow(True)


def display_text(value):
    for internal,label in DISPLAY_NAMES.items():value=value.replace(internal,label)
    return value.replace("Raw-minus-Corrected", "Raw-minus-BB-centered").replace("Raw and Corrected", "Raw and BB-centered")


def export(fig,name,output,records,caption,alt):
    caption,alt=display_text(caption),display_text(alt)
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();canvas=fig.bbox
    outside=[];overlap=[]
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        if box.x0<canvas.x0-1 or box.y0<canvas.y0-1 or box.x1>canvas.x1+1 or box.y1>canvas.y1+1:
            outside.append(artist.get_text())
    for ax in fig.axes:
        ticks=[t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        for left,right in zip(ticks,ticks[1:]):
            a,b=left.get_window_extent(renderer),right.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                overlap.append([left.get_text(),right.get_text()])
    LAYOUT_CHECKS.append({"figure":name,"text_outside_canvas":outside,"adjacent_xlabels_overlapping":overlap})
    for extension in ("pdf","png"):
        check_time()
        path=output/(name+"."+extension)
        fig.savefig(path,dpi=300,facecolor="white",transparent=False)
        records.append({"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size,
                        "width_mm":float(fig.get_figwidth()*25.4),"height_mm":float(fig.get_figheight()*25.4),
                        "format":extension,"png_dpi":300 if extension=="png" else None})
    plt.close(fig)
    (output/(name+".caption.txt")).write_text(caption+"\n",encoding="utf-8")
    (output/(name+".alt.txt")).write_text(alt+"\n",encoding="utf-8")


def primary_figure(bundles,output,records,*,interactions=False):
    if interactions:return interaction_figure(bundles,output,records)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,145/25.4),layout="constrained",sharey=True)
    ranges=[]
    for ri,block in enumerate(("C","D")):
        bundle=bundles[block]
        for ci,fraction in enumerate(FRACTIONS):
            ax=axes[ri,ci]
            names=([f"gain_change:{more}_minus_{less}/{fraction}" for less,more in zip(CONTEXTS[:-1],CONTEXTS[1:])]
                   +[f"gain_change:Full5_minus_Hidden5/{fraction}"]) if interactions else [f"gain:{context}/{fraction}" for context in CONTEXTS]
            all_seeds=np.stack([seed_values(bundle,name) for name in names])
            for si,seed in enumerate(SEEDS):
                ax.plot(np.arange(4),all_seeds[:,si],color=COLORS[si],marker=MARKERS[si],markersize=3,
                        linewidth=.55,alpha=.8,label=str(seed))
            for xi,name in enumerate(names):
                row=bundle["uncertainty"][name];mean=row["estimate"]
                ax.errorbar(xi,mean,yerr=[[mean-row["simultaneous_lower"]],[row["simultaneous_upper"]-mean]],
                            color="black",marker="_",markersize=10,capsize=3,linewidth=1.5,zorder=5)
                ranges.extend([row["simultaneous_lower"],row["simultaneous_upper"],*all_seeds[xi]])
            ax.axhline(0,color="#444444",linestyle="--",linewidth=.7)
            ax.set_xticks(np.arange(4),["2-bin momentum\n− No direct\nmomentum","8-bin momentum\n− 2-bin momentum","Exact momentum\n− 8-bin momentum","Exact momentum\n− No direct\nmomentum"] if interactions else CONTEXT_TEXT)
            ax.set_title(f"({chr(97+ri*2+ci)}) Catalog {block} · {10 if fraction=='f010' else 100}% training")
            style_axes(ax)
            if ci==0:ax.set_ylabel("Raw − BB-centered NLL\n(nat / eligible event)")
    limits=common_range(ranges,zero=True)
    for ax in axes.flat:ax.set_ylim(limits)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,title="Paired optimizer seed (colored points/lines); black interval: fixed-model event uncertainty",
               loc="outside lower center",ncol=5,fontsize=7,title_fontsize=7)
    name="information_gain_changes" if interactions else "information_gain_ladder"
    caption=("All five paired optimizer-seed contrasts are retained as colored points and connecting guides. "
             "Black ticks are their arithmetic means; black bars are approximate 95% simultaneous max-|t| bootstrap intervals "
             "over the 16 prespecified contrasts within each catalog (2000 paired launched-stratum resamples; S0 null atoms retained). "
             "These event intervals are conditional on fitted models and are not optimizer-seed intervals. C and D are not pooled. "
             "Equal input dimension and parameter count do not imply equal active parameters/effective function classes. "
             +( "Differences are more-information gain minus less-information gain; positive need not mean all model NLLs improved."
                if interactions else "Positive gain favors CorrectedPhysics; the increasing-information axis is categorical, not a quantitative entropy scale. Lines only connect paired ordered categories: no interpolation or fitted monotonic trend is asserted."))
    export(fig,name,output,records,caption,"Four panels retain every paired training seed, mean, and separate fixed-model event interval for C/D and 10/100% training; a dashed zero line shows direction changes.")


def interaction_figure(bundles,output,records):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,145/25.4),layout="constrained",sharex=True)
    labels=["2-bin momentum\n− No direct momentum","8-bin momentum\n− 2-bin momentum",
            "Exact momentum\n− 8-bin momentum","Exact momentum\n− No direct momentum"]
    ranges=[]
    for ri,block in enumerate(("C","D")):
        bundle=bundles[block]
        for ci,fraction in enumerate(FRACTIONS):
            ax=axes[ri,ci]
            names=[f"gain_change:{more}_minus_{less}/{fraction}" for less,more in zip(CONTEXTS[:-1],CONTEXTS[1:])]
            names += [f"gain_change:Full5_minus_Hidden5/{fraction}"]
            seeds=np.stack([seed_values(bundle,name) for name in names])
            for si,seed in enumerate(SEEDS):
                ax.plot(seeds[:,si],np.arange(4),color=COLORS[si],marker=MARKERS[si],markersize=3,
                        linewidth=.55,alpha=.8,label=str(seed))
            for yi,name in enumerate(names):
                row=bundle["uncertainty"][name];mean=row["estimate"]
                ax.errorbar(mean,yi,xerr=[[mean-row["simultaneous_lower"]],[row["simultaneous_upper"]-mean]],
                            color="black",marker="|",markersize=10,capsize=3,linewidth=1.5,zorder=5)
                ranges.extend([row["simultaneous_lower"],row["simultaneous_upper"],*seeds[yi]])
            ax.axvline(0,color="#444444",linestyle="--",linewidth=.7)
            ax.set_yticks(np.arange(4),labels);ax.set_ylim(3.35,-.35)
            ax.set_title(f"({chr(97+ri*2+ci)}) Catalog {block} · {10 if ci==0 else 100}% training")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="x",color="#E4E4E4",linewidth=.45);ax.set_axisbelow(True)
            ax.tick_params(axis="y",labelsize=7)
            if ri==1:ax.set_xlabel("Gain difference\n(nat / eligible event)")
    for ax in axes.flat:ax.set_xlim(common_range(ranges,zero=True))
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,title="Paired optimizer seed; black interval: fixed-model event uncertainty",
               loc="outside lower center",ncol=5,fontsize=7,title_fontsize=7)
    export(fig,"information_gain_changes",output,records,
           "Differences in centering gain between the named information conditions (more information minus less information). "
           "All five paired optimizer-seed values are retained; connecting lines are guides across four categorical contrasts. "
           "Black ticks are seed means and horizontal bars are approximate 95% simultaneous max-|t| event-bootstrap intervals "
           "over the 16 prespecified contrasts within each catalog (2000 paired launched-stratum resamples, including S0 null atoms). "
           "Event intervals are conditional on fitted models and are distinct from optimizer-seed variation. C/D are not pooled. "
           "A positive gain difference need not mean all absolute model NLLs improved.",
           "Four horizontal panels show every paired seed and fixed-model event interval for three adjacent information contrasts and the exact-versus-no-direct-momentum contrast.")


def absolute_figure(bundles,output,records):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,140/25.4),layout="constrained",sharey=True)
    ranges=[]
    for ri,block in enumerate(("C","D")):
        rows=[r for r in bundles[block]["model_rows"] if r.get("cohort")=="information80"]
        mapping={(r["context_label"],r["fraction_tag"],r["representation"],int(r["seed"])):float(r["nll"]) for r in rows}
        expected=set(itertools.product(CONTEXTS,FRACTIONS,("Raw","CorrectedPhysics"),SEEDS))
        if len(rows)!=80 or set(mapping)!=expected:raise ValueError("absolute NLL plot requires complete new80")
        for ci,fraction in enumerate(FRACTIONS):
            ax=axes[ri,ci]
            for rep_index,rep in enumerate(("Raw","CorrectedPhysics")):
                color=("#D55E00","#0072B2")[rep_index];offset=(-.12,.12)[rep_index]
                for xi,context in enumerate(CONTEXTS):
                    values=np.array([mapping[context,fraction,rep,seed] for seed in SEEDS]);mean=values.mean();sd=values.std(ddof=1)
                    for si,value in enumerate(values):ax.scatter(xi+offset+(si-2)*.025,value,s=11,color=color,marker=MARKERS[si],zorder=4)
                    ax.errorbar(xi+offset,mean,yerr=sd,color=color,marker="_",markersize=9,capsize=3,linewidth=1.1,label=DISPLAY_NAMES.get(rep,rep) if xi==0 else None)
                    ranges.extend([*values,mean-sd,mean+sd])
            ax.set_xticks(range(4),CONTEXT_TEXT);ax.set_title(f"({chr(97+ri*2+ci)}) Catalog {block} · {10 if ci==0 else 100}% training")
            style_axes(ax)
            if ci==0:ax.set_ylabel("Mean NLL\n(nat; density in (GeV/c)⁻¹)")
    for ax in axes.flat:ax.set_ylim(common_range(ranges))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,title="Points: five seeds; bars: mean ± sample SD across seeds (no event CI)")
    export(fig,"information_absolute_nll",output,records,
           "Original-unit absolute mean negative log densities for all 80 newly fitted models. Smaller is better; differential-density NLL can be negative. "
           "Every seed is shown; offset positions prevent overlap without encoding another quantity. Bars show sample SD across five optimizer seeds, "
           "not uncertainty from event resampling and not an interval for five independent datasets. Circle/square/up-triangle/diamond/down-triangle identify seeds 3101/3102/3103/3104/3105. All panels use one vertical scale.",
           "Four panels show Raw and Corrected absolute NLL at four information levels. The five seed points and mean±seed SD remain distinct from event-bootstrap uncertainty.")


def components_figure(bundles,output,records):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,150/25.4),layout="constrained",sharey=True)
    ranges=[]
    for ri,block in enumerate(("C","D")):
        mapping={(r["context"],r["fraction_tag"],r["seed"]):r["physical_groups"]["overall"] for r in bundles[block]["component_rows"]}
        if set(mapping)!=set(itertools.product(CONTEXTS,FRACTIONS,SEEDS)):raise ValueError("missing per-seed density decomposition")
        for ci,fraction in enumerate(FRACTIONS):
            ax=axes[ri,ci]
            for xi,context in enumerate(CONTEXTS):
                for ki,field in enumerate(COMPONENTS):
                    values=np.array([mapping[context,fraction,seed][field] for seed in SEEDS])
                    x=xi+(ki-2)*.13
                    for si,value in enumerate(values):ax.scatter(x+(si-2)*.012,value,s=8,color=COLORS[ki],marker=MARKERS[si],alpha=.7)
                    mean,sd=values.mean(),values.std(ddof=1)
                    ax.errorbar(x,mean,yerr=sd,color=COLORS[ki],marker="_",markersize=6,capsize=2,linewidth=1,label=COMPONENT_TEXT[ki] if xi==0 else None)
                    ranges.extend([*values,mean-sd,mean+sd])
                for seed in SEEDS:
                    row=mapping[context,fraction,seed]
                    if not np.isclose(sum(row[k] for k in COMPONENTS[:-1]),row["nll_gain"],atol=2e-5,rtol=2e-6):
                        raise ValueError("plotted exact components do not sum to gain")
            ax.axhline(0,color="#444444",linestyle="--",linewidth=.7)
            ax.set_xticks(range(4),CONTEXT_TEXT);ax.set_title(f"({chr(97+ri*2+ci)}) Catalog {block} · {10 if ci==0 else 100}% training")
            style_axes(ax)
            if ci==0:ax.set_ylabel("Raw − BB-centered component\n(nat / event)")
    for ax in axes.flat:ax.set_ylim(common_range(ranges,zero=True))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,title="Points: all five seeds; bars: mean ± seed SD")
    export(fig,"information_exact_density_terms",output,records,
           "The four signed contributions exactly reconstruct Raw-minus-Corrected NLL: negative base log density, negative spline log Jacobian, "
           "log shared preprocessing scale and log support CDF. Total gain is plotted alongside, not added as a fifth component. "
           "All five seeds and their sample SD are retained; no component event intervals are asserted. Coordinate terms can cancel and are not "
           "unique physical location, scale, shape or causal contributions. Circle/square/up-triangle/diamond/down-triangle identify seeds 3101/3102/3103/3104/3105.",
           "Four panels show all density-coordinate terms and total gain, preserving signed cancellations and every paired seed on a shared vertical scale.")


def local_figures(bundles,output,records):
    import matplotlib.pyplot as plt
    values=[]
    seedmaps={}
    for block,bundle in bundles.items():
        mapping={(r["context"],r["fraction_tag"],int(r["group_id"])):r for r in bundle["group_seed_rows"]}
        if len(mapping)!=336 or set(mapping)!=set(itertools.product(CONTEXTS,FRACTIONS,range(42))):
            raise ValueError("local display requires every primary and every 42-stratum")
        seedmaps[block]=mapping
        for key,row in mapping.items():
            interval=bundle["groups"][(key[0]+"/"+key[1],key[2])]
            seeds=[float(row[f"seed_{s}"]) for s in SEEDS]
            if not np.isclose(np.mean(seeds),interval["estimate"],rtol=0,atol=1e-10):raise ValueError("local mean/seed mismatch")
            values.extend([*seeds,interval["all_primary_groups_simultaneous_lower"],interval["all_primary_groups_simultaneous_upper"]])
    limits=common_range(values,zero=True)
    labels=[]
    for group in range(42):
        lo=.3+(group//6)*.1;t=(.1,.3,1.)[(group%6)//2];charge="−" if group%2==0 else "+"
        labels.append(f"{lo:.1f}–{lo+.1:.1f} | {t:g} | {charge}")
    for block,bundle in bundles.items():
        for context,fraction in itertools.product(CONTEXTS,FRACTIONS):
            fig,ax=plt.subplots(figsize=(180/25.4,210/25.4),layout="constrained")
            for group in range(42):
                row=seedmaps[block][context,fraction,group];ci=bundle["groups"][context+"/"+fraction,group]
                mean=ci["estimate"]
                ax.errorbar(mean,group,xerr=[[mean-ci["all_primary_groups_simultaneous_lower"]],[ci["all_primary_groups_simultaneous_upper"]-mean]],
                            color="black",marker="|",markersize=5,capsize=1.5,linewidth=.7,zorder=3)
                for si,seed in enumerate(SEEDS):ax.scatter(float(row[f"seed_{seed}"]),group+(si-2)*.11,s=9,marker=MARKERS[si],color=COLORS[si],zorder=4,label=str(seed) if group==0 else None)
            for separator in np.arange(5.5,42,6):ax.axhline(separator,color="#BBBBBB",linewidth=.5)
            ax.axvline(0,color="#333333",linestyle="--",linewidth=.7)
            ax.set(yticks=np.arange(42),yticklabels=labels,xlim=limits,ylim=(41.8,-.8),
                   xlabel="Raw − BB-centered NLL (nat / eligible event)",ylabel="Momentum bin (GeV/c) | thickness (mm) | charge",
                   title=f"Catalog {block} · {DISPLAY_NAMES[context]} · {10 if fraction=='f010' else 100}% training")
            ax.tick_params(axis="y",labelsize=6.8);style_axes(ax)
            handles,legend_labels=ax.get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc="outside lower center",ncol=5,
                       title="Seed points; black bars: fixed-model 95% band over 336 estimates",fontsize=7,title_fontsize=7)
            caption=("All 42 fixed physical strata and all five paired optimizer seeds. Black ticks give seed-mean gains; black bars show "
                     "approximate 95% simultaneous intervals over all 336 primary-by-stratum estimates within this catalog. "
                     "Colored symbols preserve optimizer-seed variation and are not separate independent catalogs. Every local panel uses one "
                     "horizontal scale; neither negative effects nor wide intervals are clipped. There is no pooling of C/D and no significance-star selection.")
            export(fig,f"information_local_{block}_{context}_{fraction}",output,records,caption,
                   f"Complete 42-row forest plot for catalog {block}, context {context}, fraction {fraction}, showing all five seeds and simultaneous fixed-model event bands, including any adverse directions.")


def source_figure(bundle,output,records):
    import matplotlib.pyplot as plt
    rows=read_json(bundle["paths"]["source_location_diagnostics.json"])["rows"]
    mapping={(r["fraction_tag"],r["representation"]):r for r in rows}
    if set(mapping)!=set(itertools.product(FRACTIONS,("Raw","CorrectedPhysics"))):raise ValueError("incomplete source diagnostic")
    # The physical-unit statistic is invariant to a constant global mu.
    # Keep one original f100 series, not an average of the duplicate displays.
    for rep in ("Raw","CorrectedPhysics"):
        a=mapping["f010",rep]["original_unit_burden"]
        b=mapping["f100",rep]["original_unit_burden"]
        va=np.array([r["event_weighted_position_rms"] for r in a["contexts"]])
        vb=np.array([r["event_weighted_position_rms"] for r in b["contexts"]])
        if not np.allclose(va,vb,rtol=2e-14,atol=0):raise ValueError("physical-unit fractions cannot be collapsed")
        SOURCE_COLLAPSE_CHECKS.append({"representation":rep,"retained_series":"f100", "omitted_duplicate_series":"f010",
            "maximum_absolute_difference_GeV_c":float(np.max(np.abs(va-vb))),
            "retained_values_GeV_c":vb.tolist(),"other_values_GeV_c":va.tolist(),
            "reason":"Same diagnostic data and fixed center; global-mu subtraction cancels in within-cell variance. Remaining differences are floating-point arithmetic."})
    fig=plt.figure(figsize=(180/25.4,145/25.4),layout="constrained")
    grid=fig.add_gridspec(2,2)
    raw_ax=fig.add_subplot(grid[0,:]); standard_axes=[fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
    panels=[(raw_ax,"f100","original_unit_burden",1000.,"MeV/c","(a) Physical units · one common diagnostic")]
    panels += [(ax,fraction,"standardized_burden",1.,"dimensionless",
                f"({chr(98+i)}) Standardized · {10 if i==0 else 100}% training scale")
               for i,(ax,fraction) in enumerate(zip(standard_axes,FRACTIONS))]
    standard_ranges=[]
    for ax,fraction,field,factor,unit,title in panels:
        ranges=[0.]
        for index,rep in enumerate(("Raw","CorrectedPhysics")):
            diagnostic=mapping[fraction,rep][field]
            by_context={r["context"]:r["event_weighted_position_rms"] for r in diagnostic["contexts"]}
            if set(by_context)!=set(CONTEXTS[:3]) or diagnostic["Full5"]["exact_conditioning_unobserved_momentum_mean_variance"]!=0:
                raise ValueError("visible-cell diagnostic definition changed")
            values=np.asarray([by_context[c]*factor for c in CONTEXTS[:3]])
            color=("#D55E00","#0072B2")[index]
            ax.plot(np.arange(3),values,color=color,marker=("o","s")[index],label=DISPLAY_NAMES.get(rep,rep))
            ax.scatter(3,0,edgecolors=color,marker=("o","s")[index],facecolors="none",s=40)
            ranges.extend(values)
        ax.set_xticks(range(4),CONTEXT_TEXT);ax.set_title(title);style_axes(ax)
        ax.set_ylabel(f"Within-visible-cell location RMS\n({unit})")
        if field=="original_unit_burden":ax.set_ylim(common_range(ranges,zero=True))
        else:standard_ranges.extend(ranges)
    for ax in standard_axes:ax.set_ylim(common_range(standard_ranges,zero=True))
    handles,labels=raw_ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,
               title="Exact momentum: open symbol denotes a theoretical zero")
    export(fig,"information_source_location",output,records,
        "Source diagnostic pool only: event-weighted RMS of 56-bin residual conditional means within retained thickness, charge and coarse-momentum cells. "
        "Panel (a) shows the physical-unit result once, in MeV/c (saved GeV/c values multiplied by 1000). "
        "The saved 100% physical-unit series is retained unchanged; the 10% series differs only at floating-point roundoff because global-mu subtraction does not alter this statistic. "
        "Panels (b) and (c) retain the separate standardizations using the 10% and 100% training scales. "
        "Exact momentum has a theoretical zero, shown with an open symbol; it is not a finite-bin estimate. "
        "The information levels are ordered categories; lines are comparison guides, not interpolation. Values are descriptive, with no seed or event-bootstrap intervals.",
        "Three panels show Raw and BB-centered source location burden: one common physical-unit panel above two training-scale-standardized panels. Exact momentum is marked as theoretical zero.")

def plotted_aggregates(source,bundles):
    # Aggregate-only evidence: internal identities remain stable while display
    # names are stored separately. No event data or newly estimated statistics.
    result={}
    for block,bundle in bundles.items():
        result[block]={"primary_seed_contrasts":bundle["seed_rows"],"primary_uncertainty":bundle["uncertainty"],
                       "model_summary":bundle["model_rows"],"density_components":bundle["component_rows"],
                       "group_seed_variation":bundle["group_seed_rows"],
                       "group_intervals":[{"primary":key[0],"group_id":key[1],**row} for key,row in sorted(bundle["groups"].items())]}
    result["source_location"]=read_json(source["paths"]["source_location_diagnostics.json"])
    return result


def main():
    parser=argparse.ArgumentParser(description="Archived aggregate-only editorial figures; no fitting, scoring or deadline modification.")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();output=args.output.resolve()
    if (output/"FIGURE_MANIFEST.json").exists() or list(output.glob("information_*.pdf")):
        raise FileExistsError("editorial figures never overwrite an existing rendering; choose a new output")
    started=time.time()
    paper=ROOT/"manuscript/representation_v2/overleaf_submission"
    original=paper/"information_figures"
    original_before={p.relative_to(original).as_posix():sha(p) for p in original.rglob("*") if p.is_file()}
    if sha(paper/"generate_information_figures.py")!=ORIGINAL_RENDERER_SHA256:raise ValueError("original renderer changed")
    old_manifest=read_json(original/"FIGURE_MANIFEST.json")
    statistical=ROOT/"protocol/information_statistical_analysis_20261108.json"
    fresh=ROOT/"protocol/fresh_catalog_20261108.json"
    source=load_bundle(ROOT/"reports/information_extension_20261108/source_analysis","source",statistical)
    bundles={block:load_bundle(ROOT/"reports/information_extension_20261108/fresh_analysis"/block,block,statistical) for block in ("C","D")}
    fresh_plan=read_json(fresh)
    if fresh_plan["status"]!="FROZEN_FRESH_CATALOG" or sha(fresh)!=old_manifest["fresh_plan_sha256"]:raise ValueError("fresh plan changed")
    if fresh_plan["source_validation_analysis"]["sha256"]!=sha(source["paths"]["analysis.json"]):raise ValueError("source lineage differs")
    inputs={block:{"analysis_manifest_sha256":bundle["manifest_sha256"],
                   "files":{name:sha(path) for name,path in bundle["paths"].items()}}
            for block,bundle in {"source":source,**bundles}.items()}
    if inputs!=old_manifest["inputs"]:raise ValueError("editorial inputs differ from the previous scientific figure inputs")
    for key in ("fresh_plan_sha256","union_sha256","roster_sha256","generation_sha256","scoring_summary_sha256"):
        if bundles["C"]["analysis"]["input_provenance"][key]!=bundles["D"]["analysis"]["input_provenance"][key]:raise ValueError("C/D lineage mismatch")
    before_numeric=plotted_aggregates(source,bundles)
    before_digest=hashlib.sha256(json.dumps(before_numeric,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style={"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,
           "legend.fontsize":7,"legend.title_fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7.5,
           "axes.linewidth":.6,"lines.linewidth":.9,"pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"}
    output.mkdir(parents=True,exist_ok=True);records=[]
    with plt.rc_context(style):
        primary_figure(bundles,output,records)
        primary_figure(bundles,output,records,interactions=True)
        absolute_figure(bundles,output,records)
        components_figure(bundles,output,records)
        source_figure(source,output,records)
        local_figures(bundles,output,records)
    after_numeric=plotted_aggregates(source,bundles)
    after_digest=hashlib.sha256(json.dumps(after_numeric,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if before_digest!=after_digest:raise ValueError("plotting mutated numerical aggregate inputs")
    original_after={p.relative_to(original).as_posix():sha(p) for p in original.rglob("*") if p.is_file()}
    if original_before!=original_after:raise ValueError("original figures changed")
    write_json(output/"PLOTTED_AGGREGATES.json",before_numeric)
    write_json(output/"DISPLAY_NAME_MAP.json",DISPLAY_NAMES)
    write_json(output/"NUMERIC_INPUT_INVARIANCE.json",{
        "status":"PASS_IDENTICAL_FROZEN_NUMERIC_INPUTS", "original_manifest_sha256":sha(original/"FIGURE_MANIFEST.json"),
        "all_input_manifests_and_file_hashes_identical":True,"original_files_unchanged":len(original_before),
        "aggregate_hash_before":before_digest,"aggregate_hash_after":after_digest,
        "physical_panel_collapse":SOURCE_COLLAPSE_CHECKS,"no_estimator_or_interval_changes":True,
        "all_seed_points_and_local_groups_retained":True,"data_ids_and_filenames_unchanged":True})
    manifest={"schema":"editorial-information-figure-manifest/v1","status":"COMPLETE_RENDER_REQUIRES_VISUAL_REVIEW",
        "generated_utc":datetime.now(timezone.utc).isoformat(),"inputs":inputs,"figures":records,"display_names":DISPLAY_NAMES,
        "fresh_plan_sha256":sha(fresh),"source_numeric_invariant_sha256":sha(output/"NUMERIC_INPUT_INVARIANCE.json"),
        "software":{"python":sys.version,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"renderer_sha256":sha(__file__)},
        "phase":"EPJC submission figure asset set; numeric coordinates and intervals byte-preserved from frozen aggregates",
        "publisher_guidance":{"url":"https://link.springer.com/journal/10052/submission-guidelines","checked":"2026-11-26"},
        "skill_citation":SKILL_CITATION,"elapsed_seconds":time.time()-started,
        "scope":"Archived complete aggregates only; no model, raw-data, resampling, training or inference calls. Historical generation/scoring gates remain unchanged.",
        "transformations":["Display names updated; numerical coordinates/intervals unchanged","Information-gain contrasts use horizontal numeric axes to fit full condition names","Source location: collapse duplicate physical-unit fraction panels using saved f100 series; retain both original standardized fraction panels","GeV/c to MeV/c display multiplication retained"],
        "layout_checks":LAYOUT_CHECKS,
        "files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()]}
    write_json(output/"FIGURE_MANIFEST.json",manifest)
    print(json.dumps({"status":manifest["status"],"figure_count":len(records)//2,"output":str(output),"numeric_inputs_unchanged":True}))


if __name__=="__main__":main()
