"""Analyze eventwise representation gaps and residual dependence on p_in."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from representation_flow.engine import fit_preprocessors, load_source_arrays, subset_indices


T95_DF4 = 2.7764451051977987


def _stats(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - T95_DF4 * se,
        "ci95_high": mean + T95_DF4 * se,
        "values": [float(v) for v in values],
    }


def _quantile_bins(values, count=5):
    edges = np.quantile(values, np.linspace(0.0, 1.0, count + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(values, edges[1:-1], right=False), edges


def eventwise(root: Path) -> dict:
    scores = json.loads((root / "exploratory_scores.json").read_text(encoding="utf-8"))
    grouped = {(item["context_label"], item["fraction_tag"], item["representation"], item["capacity"], item["seed"]): item for item in scores["runs"]}
    results = {}
    for context in sorted({item["context_label"] for item in scores["runs"]}):
        for fraction in sorted({item["fraction_tag"] for item in scores["runs"]}):
            key_prefix = (context, fraction, "Raw", "Small")
            pair_rows = []
            for seed in sorted({item["seed"] for item in scores["runs"]}):
                raw = grouped.get((*key_prefix, seed))
                phys = grouped.get((context, fraction, "Physics", "Small", seed))
                if not raw or not phys:
                    continue
                raw_pred = np.load(root / "runs" / raw["run_id"] / "predictions.npz")
                phys_pred = np.load(root / "runs" / phys["run_id"] / "predictions.npz")
                uid_raw = raw_pred["event_uid"].astype(str)
                uid_phys = phys_pred["event_uid"].astype(str)
                if not np.array_equal(uid_raw, uid_phys):
                    raise ValueError(f"UID mismatch for {context}/{fraction}/seed{seed}")
                d = (-raw_pred["log_p_y"]) - (-phys_pred["log_p_y"])
                p = raw_pred["p_in"].astype(float)
                ratio = raw_pred["y"].astype(float) / np.maximum(p, 1e-12)
                t = raw_pred["c"].astype(float) * 0.0
                pair_rows.append((seed, d, p, ratio))
            if not pair_rows:
                continue
            p_all = np.concatenate([row[2] for row in pair_rows])
            r_all = np.concatenate([row[3] for row in pair_rows])
            p_bins, p_edges = _quantile_bins(p_all)
            r_bins, r_edges = _quantile_bins(r_all)
            by_bin = {}
            for label, arrays, edges in (("p_in_quantile", [row[2] for row in pair_rows], p_edges), ("loss_ratio_quantile", [row[3] for row in pair_rows], r_edges)):
                seed_grouped = defaultdict(list)
                for seed, d, p, ratio in pair_rows:
                    values = p if label.startswith("p_in") else ratio
                    bins = np.digitize(values, edges[1:-1], right=False)
                    for bin_id in range(5):
                        mask = bins == bin_id
                        if np.any(mask):
                            seed_grouped[bin_id].append(float(np.mean(d[mask])))
                by_bin[label] = {str(bin_id): _stats(values) for bin_id, values in sorted(seed_grouped.items())}
            results[f"{context}__{fraction}"] = {"seed_count": len(pair_rows), "overall": _stats([float(np.mean(row[1])) for row in pair_rows]), "by_bin": by_bin, "p_edges": [float(x) for x in p_edges], "loss_ratio_edges": [float(x) for x in r_edges]}
    return results


def residual_diagnostics(merged: Path) -> dict:
    arrays, _ = load_source_arrays(merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_indices = np.flatnonzero(arrays["split"] == "train")
    pre = fit_preprocessors(arrays, train_indices)["Physics"]
    p = arrays["p_in"][train_indices].astype(float)
    t = arrays["thickness"][train_indices].astype(float)
    q = arrays["charge"][train_indices].astype(float)
    y = arrays["y"][train_indices].astype(float)
    center = pre.center(p, t, q).astype(float)
    residual = y - center
    p_bins, p_edges = _quantile_bins(p, 10)
    global_bins = {}
    for bin_id in range(10):
        mask = p_bins == bin_id
        global_bins[str(bin_id)] = {"count": int(np.sum(mask)), "p_mean": float(np.mean(p[mask])), "residual_mean": float(np.mean(residual[mask])), "residual_sd": float(np.std(residual[mask]))}
    x = np.column_stack([np.ones_like(p), (p - 0.65) / 0.35, (t - 0.3) / 0.7, q])
    beta = np.linalg.lstsq(x, residual, rcond=None)[0]
    fitted = x @ beta
    ss_tot = float(np.sum((residual - np.mean(residual)) ** 2))
    ss_res = float(np.sum((residual - fitted) ** 2))
    centered_squared = (residual - np.mean(residual)) ** 2
    variance_beta = np.linalg.lstsq(x, centered_squared, rcond=None)[0]
    slope_rows = []
    t_bins, t_edges = _quantile_bins(t, 4)
    for t_bin in range(4):
        for charge in sorted(np.unique(q)):
            mask = (t_bins == t_bin) & (q == charge)
            if np.sum(mask) < 100:
                continue
            xx = np.column_stack([np.ones(np.sum(mask)), (p[mask] - 0.65) / 0.35])
            slope = float(np.linalg.lstsq(xx, residual[mask], rcond=None)[0][1])
            slope_rows.append({"thickness_bin": t_bin, "charge": float(charge), "count": int(np.sum(mask)), "p_slope_residual": slope, "residual_mean": float(np.mean(residual[mask]))})
    return {
        "train_events": int(len(train_indices)),
        "physics_mu": float(pre.mu),
        "physics_scale": float(pre.scale),
        "global_residual_p_regression": {"intercept": float(beta[0]), "p_slope": float(beta[1]), "thickness_slope": float(beta[2]), "charge_slope": float(beta[3]), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0},
        "global_residual_variance_proxy_regression": {"intercept": float(variance_beta[0]), "p_slope": float(variance_beta[1]), "thickness_slope": float(variance_beta[2]), "charge_slope": float(variance_beta[3])},
        "p_in_edges": [float(x) for x in p_edges],
        "residual_by_p_in_quantile": global_bins,
        "within_thickness_charge_slopes": slope_rows,
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--interaction-root", type=Path, default=Path("reports/representation_v2/exploratory/context_interaction"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/compatibility_diagnostics.json"))
    args = parser.parse_args()
    result = {"status": "COMPATIBILITY_DIAGNOSTICS_COMPLETE", "eventwise": eventwise(args.interaction_root), "physics_residual": residual_diagnostics(args.merged)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
