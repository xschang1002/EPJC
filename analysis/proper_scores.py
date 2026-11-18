"""Proper scores and PIT diagnostics from a saved development density."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.densities import truncated_normal_cdf


def clip_probability(value: float) -> float:
    return min(1.0 - 1e-15, max(1e-15, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    mixture = checkpoint["mixture"]
    standardizer = checkpoint["standardizer"]
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if json.loads(line)["split"] == "validation"]
    train_all = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if json.loads(line)["split"] == "train"]
    threshold = float(np.quantile([row["y_delta_p_GeV_c"] / row["p_in_GeV_c"] for row in train_all], 0.99, method="linear"))
    full_nll, anchor_nll, censored_full, censored_anchor = [], [], [], []
    binary_full, binary_anchor, brier_full, brier_anchor, pit_full = [], [], [], [], []
    interval_hits = {level: [0, 0] for level in (0.5, 0.68, 0.9, 0.95, 0.99)}
    interval_widths = {level: [] for level in interval_hits}
    for row in rows:
        y = row["y_delta_p_GeV_c"]
        p = row["p_in_GeV_c"]
        upper_z = (p - standardizer["mean"]) / standardizer["scale"]
        threshold_y = p * threshold
        threshold_z = (threshold_y - standardizer["mean"]) / standardizer["scale"]
        anchor_z = (row["anchor_center_GeV_c"] - standardizer["mean"]) / standardizer["scale"]
        anchor_sigma_z = row["sigma_anchor_GeV_c"] / standardizer["scale"]
        joint_z = anchor_z + mixture["offset0_z"]
        residual_z = mixture["mu1_z"]
        anchor_q = 1.0 - float(truncated_normal_cdf(threshold_z, anchor_z, anchor_sigma_z, upper_z))
        joint_q = 1.0 - float(truncated_normal_cdf(threshold_z, joint_z, mixture["sigma0_z"], upper_z))
        residual_q = 1.0 - float(truncated_normal_cdf(threshold_z, residual_z, mixture["sigma1_z"], upper_z))
        full_q = mixture["pi0"] * joint_q + (1 - mixture["pi0"]) * residual_q
        observed_tail = y / p >= threshold
        full_nll.append(-row["log_f_full"])
        anchor_nll.append(-row["log_f_anchor_ind"])
        censored_full.append(-row["log_f_full"] if observed_tail else -math.log(clip_probability(1 - full_q)))
        censored_anchor.append(-row["log_f_anchor_ind"] if observed_tail else -math.log(clip_probability(1 - anchor_q)))
        binary_full.append(-(math.log(clip_probability(full_q)) if observed_tail else math.log(clip_probability(1 - full_q))))
        binary_anchor.append(-(math.log(clip_probability(anchor_q)) if observed_tail else math.log(clip_probability(1 - anchor_q))))
        brier_full.append((full_q - observed_tail) ** 2)
        brier_anchor.append((anchor_q - observed_tail) ** 2)
        z = (y - standardizer["mean"]) / standardizer["scale"]
        cdf_anchor = float(truncated_normal_cdf(z, anchor_z, anchor_sigma_z, upper_z))
        cdf_joint = float(truncated_normal_cdf(z, joint_z, mixture["sigma0_z"], upper_z))
        cdf_residual = float(truncated_normal_cdf(z, residual_z, mixture["sigma1_z"], upper_z))
        pit_full.append(mixture["pi0"] * cdf_joint + (1 - mixture["pi0"]) * cdf_residual)
        for level in interval_hits:
            alpha = (1 - level) / 2
            lower = float(np.quantile(pit_full[-1:], alpha)) if False else alpha
            # CDF based interval coverage is reconstructed by comparing PIT.
            interval_hits[level][0] += alpha <= pit_full[-1] <= 1 - alpha
            interval_hits[level][1] += 1
            interval_widths[level].append(level)
    def avg(values):
        return statistics.fmean(values) if values else None
    report = {
        "stage": "W5_DEVELOPMENT_PROPER_SCORES_NOT_CONFIRMATION",
        "checkpoint_seed": checkpoint["seed"],
        "validation_events": len(rows),
        "train_q99_fractional_loss": threshold,
        "full_nll": avg(full_nll),
        "anchor_nll": avg(anchor_nll),
        "full_censored_log_score": avg(censored_full),
        "anchor_censored_log_score": avg(censored_anchor),
        "full_tail_binary_log_score": avg(binary_full),
        "anchor_tail_binary_log_score": avg(binary_anchor),
        "full_tail_brier": avg(brier_full),
        "anchor_tail_brier": avg(brier_anchor),
        "full_pit_mean": avg(pit_full),
        "full_pit_variance": statistics.pvariance(pit_full) if pit_full else None,
        "interval_coverage_from_pit": {str(level): hits / total for level, (hits, total) in interval_hits.items()},
        "interval_nominal_levels": list(interval_hits),
        "tail_event_count": sum(row["y_delta_p_GeV_c"] / row["p_in_GeV_c"] >= threshold for row in rows),
        "caveat": "The pilot threshold is development-only; validation scores are not confirmation scores.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
