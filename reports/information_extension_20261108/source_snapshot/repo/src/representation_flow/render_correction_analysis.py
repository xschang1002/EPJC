"""Render audited correction evidence without training or opening raw event data.

The analytic-center figure needs no fitted results. Empirical figures require
complete 20,000-update studies; benchmark or missing cells are never plotted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from models.anchor import corrected_pion_bethe_bloch_delta_p, historical_pion_bethe_bloch_delta_p

SEEDS = (3101, 3102, 3103, 3104, 3105)
CORE_REPS = ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")
FORMAL_REPS = ("Raw", "Simple", "Learned", "CorrectedPhysics")
CORE_CONTRASTS = ("Raw_minus_CorrectedPhysics", "Raw_minus_HistoricalIncorrectPhysics", "HistoricalIncorrectPhysics_minus_CorrectedPhysics")
FORMAL_CONTRASTS = (("Raw", "CorrectedPhysics"), ("Learned", "CorrectedPhysics"), ("Simple", "Raw"))
COLORS = ("#0072B2", "#D55E00", "#009E73")
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to export an empty scientific table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(output, overwrite):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not overwrite:
        raise FileExistsError("figure output is not empty; choose a new directory or explicitly request --overwrite")
    if overwrite:
        names = [f"{stem}.{ext}" for stem in ("figure1_analytic_centers", "figure2_controlled_contrasts", "figure2b_corrected_only_detail", "figure3_corrected_formal_matrix", "figure4_posthoc_within_condition") for ext in ("pdf", "png")]
        names += ["analytic_centers.csv", "core_seed_contrasts.csv", "core_contrast_summary.csv", "corrected_only_seed_contrasts.csv", "formal_seed_contrasts.csv", "formal_contrast_summary.csv", "posthoc_within_condition_groups.csv", "posthoc_within_condition_summary.csv", "figure_manifest.json", "FIGURE_NOTES.txt", "renderer_snapshot.py"]
        paths = [output / name for name in names]
        if any(p.is_symlink() or p.exists() and not p.is_file() for p in paths):
            raise ValueError("a known figure output is a symlink or directory; refusing replacement")
        # Remove only this renderer's named outputs; a skipped figure must not
        # leave an older image looking like evidence for the current manifest.
        for path in paths:
            path.unlink(missing_ok=True)


def validate_core_payload(data):
    if data.get("status") != "COMPLETE_CORRECTION_STUDY_DESCRIPTIVE_ANALYSIS" or data.get("failed_runs"):
        raise ValueError("core analysis is incomplete, failed, or not the audited scientific analysis")
    cells = data.get("cells", [])
    expected = set(itertools.product(("Full", "Hidden"), ("f010", "f100"), CORE_REPS, SEEDS))
    indexed = {(r["context"], r["fraction_tag"], r["representation"], r["seed"]): r for r in cells}
    if len(cells) != 60 or set(indexed) != expected:
        raise ValueError("core figure requires exactly the complete 60-cell matrix")
    if any(r.get("maximum_updates") != 20000 or not np.isfinite(r["validation_nll"]) for r in cells):
        raise ValueError("benchmark, wrong-budget, or nonfinite core cells are not scientific figure inputs")
    rows = data.get("seed_contrasts", [])
    expected_contrasts = set(itertools.product(("Full", "Hidden", "Full_minus_Hidden"), ("f010", "f100"), CORE_CONTRASTS, SEEDS))
    keys = [(r["context"], r["fraction_tag"], r["contrast"], r["seed"]) for r in rows]
    if len(keys) != 90 or set(keys) != expected_contrasts:
        raise ValueError("core analysis must retain all 90 unique seed contrasts")
    for row in rows:
        left, right = row["contrast"].split("_minus_")
        tag, seed = row["fraction_tag"], row["seed"]
        def delta(context):
            return indexed[(context, tag, left, seed)]["validation_nll"] - indexed[(context, tag, right, seed)]["validation_nll"]
        expected_value = delta("Full") - delta("Hidden") if row["context"] == "Full_minus_Hidden" else delta(row["context"])
        if not np.isfinite(row["nll_difference"]) or not np.isclose(row["nll_difference"], expected_value, atol=1e-10, rtol=0):
            raise ValueError("seed contrast does not reconstruct from the checked cell scores")
    return [r for r in rows if r["context"] in ("Full", "Hidden") and r["contrast"].startswith("Raw_minus_")]


def load_core_analysis(path):
    path = Path(path).resolve()
    manifest_path = path.with_name("analysis_manifest.json")
    manifest = read_json(manifest_path)
    matches = [r for r in manifest.get("outputs", []) if r["name"] == path.name]
    if len(matches) != 1 or matches[0]["sha256"] != sha256(path):
        raise ValueError("analysis JSON does not match its analysis manifest")
    return validate_core_payload(read_json(path)), [record(path), record(manifest_path)]


def validate_formal_payload(data):
    if data.get("status") != "COMPLETE_CORRECTED_FORMAL_120":
        raise ValueError("formal figure requires COMPLETE_CORRECTED_FORMAL_120")
    rows = data.get("runs", [])
    expected = set(itertools.product(FORMAL_REPS, ("Small", "Large"), (.1, .3, 1.), SEEDS))
    keys = [(r["representation"], r["capacity"], r["sample_fraction"], r["seed"]) for r in rows]
    if len(rows) != 120 or set(keys) != expected or len({r["matrix_cell_id"] for r in rows}) != 120:
        raise ValueError("formal figure requires all 120 unique capacity/fraction/representation/seed cells")
    if any(r.get("max_updates") != 20000 or not np.isfinite(r["validation_nll"]) or r.get("context_label", "Full") != "Full" for r in rows):
        raise ValueError("benchmark, wrong-budget, nonfinite, or non-Full formal cells cannot be plotted")
    return rows


def load_formal_matrix(path, core_study=None):
    path = Path(path).resolve()
    current_formal_root = path.parent
    current_core_root = Path(core_study).resolve() if core_study is not None else path.parent.parent.parent / "correction_v2_20261107" / "study"
    data = read_json(path)
    rows = validate_formal_payload(data)
    freeze_path = path.with_name("FROZEN_CHECKPOINT_MANIFEST.json")
    freeze = read_json(freeze_path)
    models = freeze.get("models", [])
    frozen = {r["matrix_cell_id"]: r for r in models}
    if len(models) != 120 or set(frozen) != {r["matrix_cell_id"] for r in rows} or freeze.get("experiment_identity") != data.get("experiment_identity"):
        raise ValueError("formal checkpoint freeze does not match the complete matrix")
    common_uids = None
    for row in rows:
        model = frozen[row["matrix_cell_id"]]
        if any(model[key] != row[key] for key in ("artifacts", "artifact_root", "reused", "experiment_identity", "run_id")):
            raise ValueError("formal row provenance differs from the frozen checkpoint manifest")
        if type(row["reused"]) is not bool:
            raise ValueError("formal reuse marker must be Boolean")
        artifact = row["artifacts"]["predictions"]
        # The frozen absolute root records origin, not a portable read path.
        # Resolve the live tree from CLI inputs while preserving that record.
        artifact_root = current_core_root if row["reused"] else current_formal_root
        pred_path = (artifact_root / artifact["path"]).resolve()
        if not pred_path.is_relative_to(artifact_root) or not pred_path.is_file():
            raise ValueError("formal predictions are missing or outside the declared artifact root")
        if pred_path.stat().st_size != artifact["bytes"] or sha256(pred_path) != artifact["sha256"]:
            raise ValueError("formal prediction bytes do not match the frozen hash")
        with np.load(pred_path, allow_pickle=False) as prediction:
            logp = prediction["log_p_y"].astype(np.float64)
            if logp.ndim != 1 or not np.isfinite(logp).all() or not np.isclose(float(-logp.mean()), row["validation_nll"], atol=1e-12, rtol=0):
                raise ValueError("formal NLL does not reconstruct from saved validation predictions")
            uids = prediction["event_uid"].astype(str)
            if common_uids is None:
                common_uids = uids.copy()
            elif not np.array_equal(common_uids, uids):
                raise ValueError("formal validation prediction UID orders are not paired")
    lookup = {(r["representation"], r["capacity"], r["sample_fraction"], r["seed"]): r for r in rows}
    contrasts = []
    for capacity, fraction, pair, seed in itertools.product(("Small", "Large"), (.1, .3, 1.), FORMAL_CONTRASTS, SEEDS):
        left, right = pair
        contrasts.append({"capacity": capacity, "sample_fraction": fraction, "contrast": f"{left}_minus_{right}", "seed": seed,
                          "nll_difference": lookup[(left, capacity, fraction, seed)]["validation_nll"] - lookup[(right, capacity, fraction, seed)]["validation_nll"]})
    return contrasts, [record(path), record(freeze_path)]


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        if sorted(r["seed"] for r in group) != list(SEEDS):
            raise ValueError("each displayed estimate requires all five unique optimizer seeds")
        values = np.array([r["nll_difference"] for r in group], dtype=np.float64)
        result.append({**dict(zip(keys, key)), "seeds": 5, "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))})
    return result


def load_location_diagnostic(path, analysis):
    if analysis is None:
        raise ValueError("post-hoc figure requires its complete core analysis input")
    load_core_analysis(analysis)
    path = Path(path).resolve()
    manifest_path = path.with_name("manifest.json")
    data, manifest = read_json(path), read_json(manifest_path)
    matches = [r for r in manifest.get("outputs", []) if r["name"] == path.name]
    hashes = [value for key, value in manifest["input_sha256"].items() if key.replace("\\", "/").split("/")[-1] == "analysis.json"]
    if len(matches) != 1 or matches[0]["sha256"] != sha256(path) or hashes != [sha256(analysis)]:
        raise ValueError("post-hoc diagnostic does not match its manifest/core source")
    if data.get("status") != "COMPLETE_POSTHOC_WITHIN_W_LOCATION_DIAGNOSTIC" or data.get("post_hoc") is not True or data.get("prespecified_primary") is not False:
        raise ValueError("diagnostic must retain its completed post-hoc, non-primary identity")
    if data.get("source_experiment_identity") != read_json(analysis).get("experiment_identity"):
        raise ValueError("post-hoc diagnostic belongs to a different core experiment")
    groups, summary = data["groups"], data["summary"]
    expected = set(itertools.product(("f010", "f100"), CORE_REPS, (.1, .3, 1.), (-1., 1.)))
    if len(groups) != 36 or {(r["fraction_tag"], r["representation"], r["thickness_mm"], r["charge"]) for r in groups} != expected:
        raise ValueError("post-hoc figure requires every fraction/representation/retained-condition group")
    if len(summary) != 6 or {(r["fraction_tag"], r["representation"]) for r in summary} != set(itertools.product(("f010", "f100"), CORE_REPS)):
        raise ValueError("post-hoc summary is incomplete")
    for row in summary:
        matching = [g for g in groups if g["fraction_tag"] == row["fraction_tag"] and g["representation"] == row["representation"]]
        if any(g["events"] <= 0 or g["populated_momentum_bins"] != 7 or not np.isfinite(g["within_W_rms_GeV_c"]) for g in matching):
            raise ValueError("missing or invalid momentum bins cannot be silently plotted")
        rms = np.sqrt(sum(g["events"] * g["within_W_rms_GeV_c"]**2 for g in matching) / sum(g["events"] for g in matching))
        if not np.isclose(rms, row["within_W_rms_GeV_c"], rtol=1e-10, atol=1e-15):
            raise ValueError("pooled post-hoc RMS fails count-weighted reconstruction")
    for rep in CORE_REPS:
        values = [r["within_W_rms_GeV_c"] for r in summary if r["representation"] == rep]
        if not np.isclose(*values, rtol=1e-10, atol=1e-15):
            raise ValueError("fraction-invariant original-unit diagnostic unexpectedly differs")
    return data, [record(path), record(manifest_path)]


def location_figure(data, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 102 / 25.4), sharey=True, gridspec_kw={"width_ratios": [1.8, 1.]})
    fig.subplots_adjust(left=.105, right=.985, bottom=.29, top=.75, wspace=.16)
    groups = [r for r in data["groups"] if r["fraction_tag"] == "f100"]
    summary = [r for r in data["summary"] if r["fraction_tag"] == "f100"]
    conditions = list(itertools.product((.1, .3, 1.), (-1., 1.)))
    palette, markers, labels = ("#333333", COLORS[1], COLORS[0]), ("s", "^", "o"), ("Raw", "Historical incorrect", "Corrected")
    for j, (rep, color, marker, label) in enumerate(zip(CORE_REPS, palette, markers, labels)):
        values = [next(r["within_W_rms_GeV_c"] for r in groups if (r["thickness_mm"], r["charge"], r["representation"]) == (t, q, rep)) * 1e6 for t, q in conditions]
        axes[0].scatter(np.arange(6) + (j-1)*.2, values, marker=marker, color=color, s=24, label=label)
        value = next(r["within_W_rms_GeV_c"] for r in summary if r["representation"] == rep) * 1e6
        axes[1].scatter(j, value, marker=marker, color=color, s=33)
    for ax in axes:
        scientific_axes(ax)
        ax.set_ylim(0, max(r["within_W_rms_GeV_c"] for r in groups)*1e6*1.13)
    axes[0].set(title="Within each retained condition", xticks=range(6), xticklabels=[f"{t:g}\n{q:+g}" for t, q in conditions], xlabel="Thickness (mm), charge", ylabel=r"Momentum-bin mean RMS within $(t,q)$ (keV/$c$)")
    axes[1].set(title="Count-weighted pooled RMS", xticks=range(3), xticklabels=("Raw", "Historical", "Corrected"), xlim=(-.5, 2.5))
    fig.suptitle("Post-hoc diagnostic: residual location along hidden momentum", y=.97, fontsize=10.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .10, "Seven momentum bins within each fixed thickness/charge group; source diagnostic split only.\nBoth fractions give the same original-unit centered metric: subtracting fitted μ cannot change it.\nNo confidence intervals; bin-mean variation excludes within-bin event variance. Post-hoc association, not causation.", ha="center", va="center", fontsize=7.3)
    paths = [output / "posthoc_within_condition_groups.csv", output / "posthoc_within_condition_summary.csv"]
    write_csv(paths[0], data["groups"]); write_csv(paths[1], data["summary"])
    return save_figure(fig, "figure4_posthoc_within_condition", output, plt) + paths, {"title": "Post-hoc within-condition location diagnostic", "uncertainty": "No confidence interval estimated; descriptive count-weighted bin-mean variation, not within-bin event variance", "alt_text": "The left panel shows momentum-bin residual-mean RMS for all six fixed thickness/charge groups and all three representations. The right panel shows their count-weighted pooled RMS. All values use a common zero-based keV/c axis. The historical incorrect center has a larger pooled hidden-momentum location metric than Raw, whereas the corrected center has a smaller one. This diagnostic was selected after viewing the likelihood results and does not identify a causal mechanism.", "transformation": "Verify all 36 groups and six summaries and reconstruct each pooled RMS using event counts. Display f100 only after verifying equality with f010 in original units; retain both fractions in CSV. Convert GeV/c to keV/c by multiplying 1e6. No smoothing or intervals."}


def scientific_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", width=.7, length=3)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def save_figure(fig, stem, output, plt):
    files = []
    for extension in ("pdf", "png"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, format=extension, dpi=600, facecolor="white", transparent=False)
        files.append(path)
    plt.close(fig)
    return files


def center_figure(output, plt):
    p = np.linspace(.3, 1., 201)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 91 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.73, wspace=.13)
    for ax, thickness in zip(axes, (.1, .3, 1.)):
        old = historical_pion_bethe_bloch_delta_p(p, thickness, 1)
        corrected = corrected_pion_bethe_bloch_delta_p(p, thickness, 1)
        ax.plot(p, 1000 * old, color=COLORS[1], linestyle="--", marker="^", markevery=40, markersize=3.6, linewidth=1.5, label="HistoricalIncorrectPhysics")
        ax.plot(p, 1000 * corrected, color=COLORS[0], linestyle="-", marker="o", markevery=40, markersize=3.4, linewidth=1.5, label="CorrectedPhysics")
        ax.set_title(f"Silicon thickness {thickness:g} mm", fontsize=9)
        ax.set(xlim=(.3, 1.), xlabel=r"Incident momentum (GeV/$c$)", xticks=[.3, .5, .7, 1.])
        scientific_axes(ax)
        rows.extend({"p_GeV_c": float(momentum), "thickness_mm": thickness, "charge": 1,
                     "historical_center_GeV_c": float(h), "corrected_center_GeV_c": float(c),
                     "historical_center_MeV_c": float(1000 * h), "corrected_center_MeV_c": float(1000 * c)}
                    for momentum, h, c in zip(p, old, corrected))
    # Set the shared limits only after every thickness has contributed data.
    maximum = max(max(r["historical_center_MeV_c"], r["corrected_center_MeV_c"]) for r in rows)
    axes[0].set_ylim(0, maximum * 1.06)
    axes[0].set_ylabel(r"Analytic center $c$ (MeV/$c$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .90), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Historical and unit-consistent analytic centers", y=.98, fontsize=11)
    fig.text(.5, .065, "Curves show c only. Models fit μ = mean_train(Y − c), which absorbs any constant center offset.\nCenter amplitude alone does not imply better likelihood. Charges +1 and −1 have identical centers.\nNominal silicon; density-effect correction omitted. Shared vertical scale across thicknesses.", ha="center", va="center", fontsize=7.4)
    csv_path = output / "analytic_centers.csv"
    write_csv(csv_path, rows)
    files = save_figure(fig, "figure1_analytic_centers", output, plt) + [csv_path]
    return files, {"title": "Analytic centers", "uncertainty": "None: deterministic formula evaluation", "alt_text": "Three panels compare historical mixed-unit and corrected all-MeV momentum-loss centers versus incident momentum from 0.3 to 1.0 GeV/c, at silicon thicknesses 0.1, 0.3 and 1.0 mm. Dashed orange triangles identify the historical center and solid blue circles the corrected center; all panels use the same MeV/c vertical scale. Curves show analytic c only, not c plus the separately fitted training residual mean. A constant center offset is absorbed by that mean, so center amplitude alone does not imply a likelihood gain.", "transformation": "Evaluate 201 equally spaced momenta per thickness; multiply GeV/c outputs by 1000 for display; no data fitting or smoothing. Fitted residual means are not included; conditional trend/shape and finite-capacity performance require the separate experiments."}


def seed_group(ax, x, values, color, marker):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("a displayed group must have five finite seed contrasts")
    ax.scatter(x + np.linspace(-.11, .11, 5), values, s=23, color=color, marker=marker, edgecolors="white", linewidths=.35, zorder=3)
    ax.errorbar(x + .19, values.mean(), yerr=values.std(ddof=1), fmt="D", markersize=4, color="#202020", ecolor="#202020", capsize=3, linewidth=1.1, zorder=4)


def effect_limits(rows, group_keys):
    values = [float(r["nll_difference"]) for r in rows]
    for summary in summarize(rows, group_keys):
        values.extend((summary["mean"] - summary["sample_sd"], summary["mean"] + summary["sample_sd"]))
    spread = max(values + [0.]) - min(values + [0.])
    pad = max(.13 * spread, 1e-5)
    return min(values + [0.]) - pad, max(values + [0.]) + pad


def core_figure(rows, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 125 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.20, top=.85, hspace=.32, wspace=.16)
    limits = effect_limits(rows, ("context", "fraction_tag", "contrast"))
    for i, context in enumerate(("Full", "Hidden")):
        for j, tag in enumerate(("f010", "f100")):
            ax = axes[i, j]
            for x, (contrast, color, marker) in enumerate((("Raw_minus_HistoricalIncorrectPhysics", COLORS[1], "^"), ("Raw_minus_CorrectedPhysics", COLORS[0], "o"))):
                selected = sorted((r for r in rows if r["context"] == context and r["fraction_tag"] == tag and r["contrast"] == contrast), key=lambda r: r["seed"])
                seed_group(ax, x, [r["nll_difference"] for r in selected], color, marker)
            scientific_axes(ax)
            ax.axhline(0, color="#555555", linewidth=.9)
            ax.set(title=f"{context} conditioner · {10 if tag == 'f010' else 100}% training", xlim=(-.35, 1.4), ylim=limits,
                   xticks=(0, 1), xticklabels=("Historical center", "Corrected center"))
            ax.tick_params(axis="x", labelbottom=True)
    fig.supylabel("Raw NLL − center NLL (nat/event)", x=.018, fontsize=9)
    fig.suptitle("Small-capacity controlled 60-cell correction study", y=.97, fontsize=11)
    fig.text(.5, .915, "Triangles / circles: individual seeds     Black diamonds and bars: mean ± sample SD", ha="center", fontsize=8)
    fig.text(.5, .07, "All five optimizer seeds share the same source data; SD is seed variation, not an independent-data CI.\nPositive differences favor the center. Validation selected checkpoints; 20,000-update training budget.", ha="center", va="center", fontsize=8)
    paths = [output / "core_seed_contrasts.csv", output / "core_contrast_summary.csv"]
    write_csv(paths[0], rows)
    write_csv(paths[1], summarize(rows, ("context", "fraction_tag", "contrast")))
    return save_figure(fig, "figure2_controlled_contrasts", output, plt) + paths, {"title": "Controlled correction contrasts", "uncertainty": "Sample SD (ddof=1) across five optimizer seeds sharing source data; not a confidence interval", "alt_text": "Four panels show Raw-minus-center validation NLL contrasts for Full and Hidden conditioners at 10% and 100% training fractions. Every panel displays all five seed values for the historical and corrected centers, with a black mean diamond and sample-SD bar. A shared vertical scale and zero reference support comparisons.", "transformation": "Select the two prespecified Raw-minus-center contrasts from the audited complete analysis; deterministic horizontal separation changes no y values; compute mean and sample SD without exclusions."}


def core_detail_figure(rows, output, plt):
    selected = [r for r in rows if r["contrast"] == "Raw_minus_CorrectedPhysics"]
    limits = effect_limits(selected, ("context", "fraction_tag", "contrast"))
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 95 / 25.4), sharey=True)
    fig.subplots_adjust(left=.13, right=.985, bottom=.27, top=.77, wspace=.16)
    for ax, context, color, marker in zip(axes, ("Full", "Hidden"), (COLORS[0], COLORS[2]), ("o", "s")):
        for x, tag in enumerate(("f010", "f100")):
            group = sorted((r for r in selected if r["context"] == context and r["fraction_tag"] == tag), key=lambda r: r["seed"])
            seed_group(ax, x, [r["nll_difference"] for r in group], color, marker)
        scientific_axes(ax)
        ax.axhline(0, color="#555555", linewidth=.9)
        ax.set(title=f"{context} conditioner", xlim=(-.35, 1.4), ylim=limits,
               xticks=(0, 1), xticklabels=("10%", "100%"), xlabel="Training fraction")
    axes[0].set_ylabel("Raw NLL − Corrected NLL (nat/event)")
    fig.suptitle("Corrected-center detail · Small capacity · validation", y=.97, fontsize=11)
    fig.text(.5, .885, "Colored points: all five seeds     Black diamonds and bars: mean ± sample SD", ha="center", fontsize=8)
    fig.text(.5, .09, "Detail of the corrected arm; companion overview retains all historical-center contrasts.\nPositive favors CorrectedPhysics. Validation selected checkpoints; seed SD is not an independent-data CI.", ha="center", va="center", fontsize=7.7)
    path = output / "corrected_only_seed_contrasts.csv"
    write_csv(path, selected)
    return save_figure(fig, "figure2b_corrected_only_detail", output, plt) + [path], {"title": "Corrected-center detail", "uncertainty": "Sample SD (ddof=1) across all five shared-data optimizer seeds; not a confidence interval", "alt_text": "Two panels show only the prespecified Raw-minus-CorrectedPhysics validation contrast for Small capacity, using a shared detailed vertical scale across Full and Hidden. Every seed at both fractions is visible, including the negative Full 10% seed. The companion overview separately retains the much larger historical-center deficit.", "transformation": "Select all 20 Raw-minus-CorrectedPhysics seed contrasts without exclusions; retain the complete historical/corrected overview alongside this explicitly labeled detail; horizontal separation changes no y value."}


def formal_figure(rows, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 127 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.21, top=.84, hspace=.33, wspace=.16)
    limits = effect_limits(rows, ("capacity", "sample_fraction", "contrast"))
    labels = ("Raw − Corrected", "Learned − Corrected", "Simple − Raw")
    for i, capacity in enumerate(("Small", "Large")):
        for j, (left, right) in enumerate(FORMAL_CONTRASTS):
            ax = axes[i, j]
            contrast = f"{left}_minus_{right}"
            for x, fraction in enumerate((.1, .3, 1.)):
                selected = sorted((r for r in rows if r["capacity"] == capacity and r["sample_fraction"] == fraction and r["contrast"] == contrast), key=lambda r: r["seed"])
                seed_group(ax, x, [r["nll_difference"] for r in selected], COLORS[j], ("o", "s", "^")[j])
            scientific_axes(ax)
            ax.axhline(0, color="#555555", linewidth=.9)
            ax.set(title=f"{capacity} · {labels[j]}", xlim=(-.35, 2.4), ylim=limits, xticks=(0, 1, 2), xticklabels=("10%", "30%", "100%"))
            ax.tick_params(axis="x", labelbottom=True)
            if i == 1:
                ax.set_xlabel("Training fraction")
    fig.supylabel("Paired validation NLL difference (nat/event)", x=.018, fontsize=9)
    fig.suptitle("Complete corrected Full-conditioner matrix: 120 cells", y=.975, fontsize=11)
    fig.text(.5, .917, "Colored points: all five optimizer seeds     Black diamonds and bars: mean ± sample SD", ha="center", fontsize=8)
    fig.text(.5, .075, "Prespecified contrasts use NLL(left) − NLL(right); positive favors the right-hand model.\nSD describes optimizer seeds on shared data, not an independent-data CI. Training budget: 20,000 updates.", ha="center", va="center", fontsize=8)
    paths = [output / "formal_seed_contrasts.csv", output / "formal_contrast_summary.csv"]
    write_csv(paths[0], rows)
    write_csv(paths[1], summarize(rows, ("capacity", "sample_fraction", "contrast")))
    return save_figure(fig, "figure3_corrected_formal_matrix", output, plt) + paths, {"title": "Corrected formal matrix", "uncertainty": "Sample SD (ddof=1) of five paired optimizer-seed contrasts, conditional on shared source data", "alt_text": "Six panels display the prespecified Raw-minus-Corrected, Learned-minus-Corrected and Simple-minus-Raw contrasts for Small and Large capacity. Each panel contains five seed points plus mean and sample SD at 10%, 30% and 100% training fractions, using a common vertical scale and a zero line.", "transformation": "Verify the full 120-cell score matrix, checkpoint freeze and saved prediction hashes/NLL/UID order; subtract paired same-seed scores; retain all seeds; compute mean and sample SD."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, help="Completed 60-cell analysis.json with its analysis_manifest.json")
    parser.add_argument("--formal", type=Path, help="Completed formal_scores_120.json and frozen checkpoint manifest")
    parser.add_argument("--core-study", type=Path, help="Current core study directory for the formal matrix's 20 reused cells; defaults to the sibling correction_v2_20261107/study")
    parser.add_argument("--location", type=Path, help="Completed post-hoc within_condition_locations.json; requires --analysis")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace previously rendered figure outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    prepare_output(output, args.overwrite)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sources = [record(Path(__file__)), record(ROOT / "src/models/anchor.py"), record(ROOT / "src/models/anchor_legacy.py")]
    files, figures, skipped = [], [], []
    style = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "text.color": "#202020", "axes.labelcolor": "#202020", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .7, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        made, description = center_figure(output, plt)
        files.extend(made); figures.append(description)
        formal_loader = lambda path: load_formal_matrix(path, args.core_study)
        for key, path, loader, plotter in (("controlled_60", args.analysis, load_core_analysis, core_figure), ("formal_120", args.formal, formal_loader, formal_figure)):
            if path is None:
                skipped.append({"figure": key, "status": "NOT_RENDERED", "reason": "No completed scientific input supplied; no values imputed."})
                continue
            try:
                rows, inputs = loader(path)
            except (ValueError, KeyError, OSError) as exc:
                skipped.append({"figure": key, "status": "NOT_RENDERED", "reason": str(exc)})
                continue
            made, description = plotter(rows, output, plt)
            files.extend(made); figures.append(description); sources.extend(inputs)
            if key == "controlled_60":
                made, description = core_detail_figure(rows, output, plt)
                files.extend(made); figures.append(description)
        if args.location:
            data, inputs = load_location_diagnostic(args.location, args.analysis)
            made, description = location_figure(data, output, plt)
            files.extend(made); figures.append(description); sources.extend(inputs)
    snapshot = output / "renderer_snapshot.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    files.append(snapshot)
    provenance = {"schema": "correction-figures/v1", "scope": "General scientific review figures; publisher/journal requirements pending verification", "sources": sources,
                  "artifact_root_resolution": "Formal cells are read relative to the input study; reused cells use --core-study or its trusted sibling path. Frozen absolute roots remain provenance records only.",
                  "command_inputs": {"analysis": str(args.analysis.resolve()) if args.analysis else None, "formal": str(args.formal.resolve()) if args.formal else None, "core_study": str(args.core_study.resolve()) if args.core_study else None, "location": str(args.location.resolve()) if args.location else None},
                  "matplotlib": matplotlib.__version__, "numpy": np.__version__, "export": {"formats": ["PDF", "PNG"], "width_mm": 180, "png_dpi": 600, "background": "opaque white", "pdf_fonttype": 42, "bbox_inches": None},
                  "figures": figures, "not_rendered": skipped, "outputs": [record(p) for p in files],
                  "procedural_reference": {"authors": "Kassis, T.; Agarwal, V.; He, Y.; Patel, D.; Brueckner, A. M.", "year": 2026, "title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents", "doi": "10.48550/arXiv.2609.00065", "url": "https://arxiv.org/abs/2609.00065", "metadata_checked": "2026-11-07"}}
    (output / "figure_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    notes = ["Correction study figure notes", "", provenance["scope"], ""]
    for item in figures:
        notes.extend([item["title"], item["alt_text"], "Uncertainty: " + item["uncertainty"], "Transformations: " + item["transformation"], ""])
    notes.extend(["Not rendered: " + json.dumps(skipped), "", "Procedural software reference: Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065"])
    (output / "FIGURE_NOTES.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps({"figures_rendered": len(figures), "not_rendered": skipped, "output": str(output)}))
    if any(args.analysis is not None and r["figure"] == "controlled_60" or args.formal is not None and r["figure"] == "formal_120" for r in skipped):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
