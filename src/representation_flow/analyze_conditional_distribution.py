"""Audit Y and Physics residual distributions conditional on W and p_in."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, skew

from representation_flow.engine import fit_preprocessors, load_source_arrays


def _quantile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "skew": float(skew(values, bias=False)) if len(values) > 2 else 0.0,
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }


def _kde_entropy(values, folds=5):
    values = np.asarray(values, dtype=float)
    if len(values) < 40 or np.std(values) <= 0:
        return {"entropy": float("nan"), "bandwidth_multiplier": None, "cv_loglik": float("nan"), "count": int(len(values))}
    fold_ids = np.arange(len(values)) % folds
    candidates = (0.5, 0.75, 1.0, 1.35, 1.8)
    scores = []
    for multiplier in candidates:
        heldout_loglik = []
        for fold in range(folds):
            train = values[fold_ids != fold]
            valid = values[fold_ids == fold]
            kde = gaussian_kde(train)
            kde.set_bandwidth(bw_method=kde.factor * multiplier)
            heldout_loglik.extend(np.log(np.maximum(kde(valid), 1e-300)).tolist())
        scores.append(float(np.mean(heldout_loglik)))
    best_index = int(np.argmax(scores))
    return {"entropy": float(-scores[best_index]), "bandwidth_multiplier": candidates[best_index], "cv_loglik": scores[best_index], "count": int(len(values))}


def _entropy_summary(values, p_bins, label):
    pooled = _kde_entropy(values)
    conditional = {}
    weighted = 0.0
    usable = 0
    for bin_id in range(int(np.max(p_bins)) + 1):
        mask = p_bins == bin_id
        estimate = _kde_entropy(values[mask])
        conditional[str(bin_id)] = estimate
        if math.isfinite(estimate["entropy"]):
            weighted += float(np.mean(mask)) * estimate["entropy"]
            usable += 1
    return {"variable": label, "pooled": pooled, "p_bin_conditional": conditional, "weighted_conditional_entropy": weighted, "usable_bins": usable, "conditional_dependence_proxy": pooled["entropy"] - weighted if math.isfinite(pooled["entropy"]) else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/conditional_distribution_audit.json"))
    args = parser.parse_args()
    arrays, _ = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_indices = np.flatnonzero(arrays["split"] == "train")
    pre = fit_preprocessors(arrays, train_indices)["Physics"]
    p = arrays["p_in"][train_indices].astype(float)
    t = arrays["thickness"][train_indices].astype(float)
    q = arrays["charge"][train_indices].astype(float)
    y = arrays["y"][train_indices].astype(float)
    center = pre.center(p, t, q).astype(float)
    residual = y - center
    report = {
        "status": "CONDITIONAL_DISTRIBUTION_AUDIT_COMPLETE",
        "quality_failures": [],
        "quality_warnings": [
            "KDE pooled-versus-conditional entropy differences are exploratory and are not mutual-information estimates.",
            "The standard-logistic support diagnostic saturates at the exported standardized gaps; use the learned-flow log-CDF for support attribution.",
        ],
        "train_events": int(len(train_indices)),
        "w_definition": "exact thickness and charge strata",
        "p_bin_count": 8,
        "entropy_method": "5-fold cross-fitted Gaussian KDE with bandwidth multiplier selected by held-out log likelihood",
        "strata": {},
    }
    for thickness in sorted(np.unique(t)):
        for charge in sorted(np.unique(q)):
            w_mask = (t == thickness) & (q == charge)
            w_p = p[w_mask]
            p_bins, p_edges = np.histogram(w_p, bins=8)
            bin_ids = np.digitize(w_p, p_edges[1:-1], right=False)
            entry = {"count": int(np.sum(w_mask)), "p_edges": [float(x) for x in p_edges], "by_p_bin": {}, "entropy": {}, "support": {}}
            for bin_id in range(8):
                mask_local = bin_ids == bin_id
                y_local = y[w_mask][mask_local]
                r_local = residual[w_mask][mask_local]
                entry["by_p_bin"][str(bin_id)] = {"p_mean": float(np.mean(w_p[mask_local])), "Y": _quantile_summary(y_local), "R": _quantile_summary(r_local)}
            entry["entropy"]["Y"] = _entropy_summary(y[w_mask], bin_ids, "Y")
            entry["entropy"]["R"] = _entropy_summary(residual[w_mask], bin_ids, "R")
            entry["entropy"]["hY_minus_hR"] = entry["entropy"]["Y"]["pooled"]["entropy"] - entry["entropy"]["R"]["pooled"]["entropy"]
            entry["entropy"]["dependence_proxy_R_minus_Y"] = entry["entropy"]["R"]["conditional_dependence_proxy"] - entry["entropy"]["Y"]["conditional_dependence_proxy"]
            for name, c in (("Raw", np.zeros_like(center[w_mask])), ("Physics", center[w_mask])):
                gap = (p[w_mask] - c - pre.mu) / pre.scale
                logcdf = -np.logaddexp(0.0, -gap)
                entry["support"][name] = {"gap_q05": float(np.quantile(gap, 0.05)), "gap_q50": float(np.quantile(gap, 0.50)), "gap_q95": float(np.quantile(gap, 0.95)), "mean_logF_standard_logistic": float(np.mean(logcdf))}
            report["strata"][f"t{thickness:g}__q{charge:g}"] = entry
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "strata": len(report["strata"]), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
