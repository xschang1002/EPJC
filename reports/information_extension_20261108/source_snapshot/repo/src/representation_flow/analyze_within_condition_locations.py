"""Post-hoc location variation across momentum bins at fixed thickness/charge.

Reads only an already verified diagnostic-bin CSV and its analysis provenance.
It neither reads event ledgers nor accesses confirmation data or fitted models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

REPRESENTATIONS = ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")
FRACTIONS = ("f010", "f100")
W_GROUPS = tuple(itertools.product((.1, .3, 1.), (-1., 1.)))
BOUNDARY = (
    "Added after viewing the controlled study results. This is a post-hoc descriptive location diagnostic, "
    "not a prespecified primary contrast, causal identification, significance test, or independent confirmation. "
    "Within-W variation describes momentum-bin residual means conditional on the retained thickness and charge. "
    "It does not isolate all reasons for Hidden NLL changes: finite-bin averaging, sampling noise, conditional "
    "shape/scale, support normalization, and optimization can also matter. It excludes within-bin event variance. "
    "The global training mu is constant within each representation/fraction, so adding or removing it cannot "
    "change the within-W variance or the centered global/between-W variance. No confirmation data were read."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_location_decomposition(rows, value_key="mean_residual_GeV_c"):
    """Population-weighted variance of bin means, not event residual variance."""
    grouped = {}
    for row in rows:
        if row["events"]:
            grouped.setdefault((row["thickness_mm"], row["charge"]), []).append(row)
    if not grouped:
        raise ValueError("location decomposition requires populated bins")
    groups = []
    for (thickness, charge), bins in sorted(grouped.items()):
        count = sum(row["events"] for row in bins)
        mean = math.fsum(row["events"] * row[value_key] for row in bins) / count
        variance = math.fsum(row["events"] * (row[value_key] - mean) ** 2 for row in bins) / count
        groups.append({"thickness_mm": thickness, "charge": charge, "events": count, "populated_momentum_bins": len(bins),
                       "group_mean": mean, "within_W_variance": variance, "within_W_rms": math.sqrt(variance)})
    count = sum(group["events"] for group in groups)
    mean = math.fsum(group["events"] * group["group_mean"] for group in groups) / count
    within = math.fsum(group["events"] * group["within_W_variance"] for group in groups) / count
    between = math.fsum(group["events"] * (group["group_mean"] - mean) ** 2 for group in groups) / count
    global_variance = math.fsum(row["events"] * (row[value_key] - mean) ** 2 for row in rows if row["events"]) / count
    global_second_moment = math.fsum(row["events"] * row[value_key] ** 2 for row in rows if row["events"]) / count
    if not math.isclose(global_variance, within + between, rel_tol=1e-12, abs_tol=1e-24):
        raise ValueError("within-W plus between-W variance does not reconstruct global variance")
    if not math.isclose(global_second_moment, mean ** 2 + global_variance, rel_tol=1e-12, abs_tol=1e-24):
        raise ValueError("centered global variance does not reconstruct the global second moment")
    return groups, {"events": count, "populated_W_groups": len(groups), "global_mean": mean,
                    "within_W_variance": within, "within_W_rms": math.sqrt(within),
                    "between_W_variance": between, "between_W_rms": math.sqrt(between),
                    "global_centered_bin_mean_variance": global_variance,
                    "global_centered_bin_mean_sd": math.sqrt(global_variance),
                    "global_uncentered_bin_mean_rmse": math.sqrt(global_second_moment),
                    "variance_decomposition_absolute_error": abs(global_variance - within - between)}


def load_verified_bins(analysis_root):
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {record["name"]: record["sha256"] for record in manifest["outputs"]}
    for name in ("analysis.json", "diagnostic_bins.csv"):
        if name not in expected or sha256(analysis_root / name) != expected[name]:
            raise ValueError(f"verified analysis product changed or is missing: {name}")
    analysis = json.loads((analysis_root / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("status") != "COMPLETE_CORRECTION_STUDY_DESCRIPTIVE_ANALYSIS" or analysis.get("failed_runs") or len(analysis.get("cells", [])) != 60:
        raise ValueError("post-hoc diagnostic requires the completed verified core60 analysis")
    rows = []
    with (analysis_root / "diagnostic_bins.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row = dict(row)
            for key in ("momentum_bin", "events"):
                row[key] = int(row[key])
            for key in ("p_low_GeV_c", "p_high_GeV_c", "thickness_mm", "charge"):
                row[key] = float(row[key])
            for key in ("mean_residual_GeV_c", "mean_standardized_residual"):
                row[key] = float(row[key]) if row[key] else None
                if row["events"] and (row[key] is None or not math.isfinite(row[key])):
                    raise ValueError("a populated bin lacks a finite residual mean")
                if not row["events"] and row[key] is not None:
                    raise ValueError("an empty bin must not carry an imputed residual mean")
            if row["events"] < 0 or not math.isclose(row["p_low_GeV_c"], .3 + .1 * row["momentum_bin"], abs_tol=1e-12) or not math.isclose(row["p_high_GeV_c"], .4 + .1 * row["momentum_bin"], abs_tol=1e-12):
                raise ValueError("invalid count or changed momentum-bin boundaries")
            rows.append(row)
    expected_keys = set(itertools.product(FRACTIONS, REPRESENTATIONS, range(7), (.1,.3,1.), (-1.,1.)))
    keys = [(r["fraction_tag"],r["representation"],r["momentum_bin"],r["thickness_mm"],r["charge"]) for r in rows]
    if len(keys) != 252 or set(keys) != expected_keys:
        raise ValueError("diagnostic CSV is not the unique complete 2 x 3 x 42-bin design")
    reference_counts = {(r["momentum_bin"],r["thickness_mm"],r["charge"]):r["events"] for r in rows if r["fraction_tag"]=="f010" and r["representation"]=="Raw"}
    if any(r["events"] != reference_counts[(r["momentum_bin"],r["thickness_mm"],r["charge"])] for r in rows):
        raise ValueError("representations/fractions do not use the same diagnostic-bin counts")
    return rows, analysis


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(analysis_root, output):
    if output.exists():
        raise FileExistsError("post-hoc output exists; preserve it and choose a new directory")
    rows, source_analysis = load_verified_bins(analysis_root)
    groups, summaries = [], []
    for fraction, representation in itertools.product(FRACTIONS, REPRESENTATIONS):
        selected = [row for row in rows if (row["fraction_tag"],row["representation"]) == (fraction,representation)]
        physical_groups, physical = weighted_location_decomposition(selected)
        standardized_groups, standardized = weighted_location_decomposition(selected, "mean_standardized_residual")
        if {(g["thickness_mm"],g["charge"]) for g in physical_groups} != set(W_GROUPS):
            raise ValueError("every retained-W group must contain diagnostic events")
        prefix = {"fraction_tag":fraction,"representation":representation}
        for physical_group, standardized_group in zip(physical_groups, standardized_groups):
            groups.append({**prefix, "thickness_mm":physical_group["thickness_mm"],"charge":physical_group["charge"],
                           "events":physical_group["events"],"populated_momentum_bins":physical_group["populated_momentum_bins"],
                           "group_mean_residual_GeV_c":physical_group["group_mean"],
                           "within_W_variance_GeV2_c2":physical_group["within_W_variance"],"within_W_rms_GeV_c":physical_group["within_W_rms"],
                           "within_W_standardized_variance":standardized_group["within_W_variance"],"within_W_standardized_rms":standardized_group["within_W_rms"]})
        summaries.append({**prefix, **{key:physical[key] for key in ("events","populated_W_groups")},
                          **{f"{key}_{'GeV2_c2' if 'variance' in key or 'error' in key else 'GeV_c'}":value for key,value in physical.items() if key not in ("events","populated_W_groups")},
                          "within_W_standardized_variance":standardized["within_W_variance"],"within_W_standardized_rms":standardized["within_W_rms"]})
    result = {"status":"COMPLETE_POSTHOC_WITHIN_W_LOCATION_DIAGNOSTIC", "post_hoc":True, "prespecified_primary":False,
              "source_experiment_identity":source_analysis["experiment_identity"], "confirmation_accessed":False,
              "definition":"W=(thickness,charge); m_jW=mean(Y-c(X)-training_mu) in momentum bin j; N_W=sum_j n_jW; m_W=sum_j n_jW*m_jW/N_W; v_W=sum_j n_jW*(m_jW-m_W)^2/N_W; pooled RMS=sqrt(sum_W N_W*v_W/sum_W N_W)",
              "variance_denominator":"Population count weights; no ddof correction. These are variances of bin means, not within-bin event variances.",
              "mu_invariance":"Subtracting any representation/fraction-specific constant mu leaves all centered within-W variances unchanged.",
              "interpretation_boundary":BOUNDARY, "groups":groups,"summary":summaries}
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output / "within_condition_groups.csv", groups)
    write_csv(output / "within_condition_summary.csv", summaries)
    (output / "within_condition_locations.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    lines = ["查看结果后新增：固定厚度/电荷内的动量分箱残差位置诊断", "",
             "本诊断不是预设主对照、显著性检验或因果证明。仅使用已通过完整分析清单核验的diagnostic_bins.csv，不读取A/B或重训。",
             "W=(厚度,电荷)是Hidden仍保留的信息。先在每个W内对动量bin残差均值做按事件数加权的中心化方差，再按各W事件数汇总，最后开方。",
             "结果是离散动量bin均值在固定W内的变化；不是事件级残差SD，也不包含bin内变化。有限分箱及均值估计噪声仍存在。",
             "全42-bin未中心化RMSE同时包含整体偏置、W间位置变化和W内动量变化，因此不能直接当作隐藏动量的负担。",
             "训练mu是每个表示/样本分数的常数；移除或加回mu不改变W内中心化方差。", "",
             "fraction representation pooled within-W RMS [GeV/c] | between-W RMS [GeV/c] | global uncentered bin-mean RMSE [GeV/c]"]
    for row in summaries:
        lines.append(f"{row['fraction_tag']} {row['representation']}: {row['within_W_rms_GeV_c']:.12g} | {row['between_W_rms_GeV_c']:.12g} | {row['global_uncentered_bin_mean_rmse_GeV_c']:.12g}")
    lines += ["", "较小的W内位置变化只表示这种分箱下的残差位置更平坦，不能单独解释Hidden NLL：条件分布形状/尺度、支持归一化及优化过程等仍可能影响结果。", "", BOUNDARY]
    (output / "LOCATION_DIAGNOSTICS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    inputs = [analysis_root / name for name in ("diagnostic_bins.csv","analysis.json","analysis_manifest.json")]
    manifest = {"schema":"posthoc-within-W-location-diagnostic/v1","analysis_source_sha256":sha256(Path(__file__)),
                "input_sha256":{str(path):sha256(path) for path in inputs},
                "outputs":[{"name":path.name,"sha256":sha256(path)} for path in sorted(output.iterdir())]}
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,default=Path("reports/correction_v2_20261107/analysis"))
    parser.add_argument("--output",type=Path,default=Path("reports/correction_v2_20261107/location_diagnostics"))
    args = parser.parse_args()
    result = analyze(args.analysis,args.output)
    print(json.dumps({"status":result["status"],"summary":result["summary"],"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
