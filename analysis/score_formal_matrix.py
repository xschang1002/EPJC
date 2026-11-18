"""Compute proper development scores for a formal-matrix checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.densities import truncated_normal_cdf, truncated_student_cdf
from models.fit_formal_matrix import design
from data_contract.event_split import split_label


def clip(value: float) -> float:
    return min(1.0 - 1e-15, max(1e-15, float(value)))


def component_cdf(kind: str, params: list[float], name: str, row: dict, checkpoint: dict, z: float, upper: float, anchor: float, x: np.ndarray) -> float:
    if name == "anchor_independent":
        return float(truncated_normal_cdf(z, anchor, math.exp(params[0]), upper))
    if name == "shared_gaussian":
        return float(truncated_normal_cdf(z, params[0], math.exp(params[1]), upper))
    if name == "conditional_gaussian":
        return float(truncated_normal_cdf(z, float(x @ np.asarray(params[:4])), math.exp(params[4]), upper))
    if name == "conditional_student_t_df3":
        return float(truncated_student_cdf(z, float(x @ np.asarray(params[:4])), math.exp(params[4]), upper, df=3.0))
    if name == "frozen_anchor_k2":
        logit, _offset, mu1, logs0, logs1 = params
        pi0 = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
        return pi0 * float(truncated_normal_cdf(z, anchor, math.exp(logs0), upper)) + (1 - pi0) * float(truncated_normal_cdf(z, mu1, math.exp(logs1), upper))
    if name == "joint_anchor_residual_k2":
        logit, offset, mu1, logs0, logs1 = params
        pi0 = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
        return pi0 * float(truncated_normal_cdf(z, anchor + offset, math.exp(logs0), upper)) + (1 - pi0) * float(truncated_normal_cdf(z, mu1, math.exp(logs1), upper))
    if name == "conditional_gate_anchor_k2":
        v = np.asarray(params)
        pi0 = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, float(x @ v[:4])))))
        return pi0 * float(truncated_normal_cdf(z, anchor + float(x @ v[4:8]), math.exp(v[12]), upper)) + (1 - pi0) * float(truncated_normal_cdf(z, float(x @ v[8:12]), math.exp(v[13]), upper))
    if name == "learned_center_k2":
        v = np.asarray(params)
        pi0 = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, float(v[0])))))
        return pi0 * float(truncated_normal_cdf(z, float(x @ v[1:5]), math.exp(v[9]), upper)) + (1 - pi0) * float(truncated_normal_cdf(z, float(x @ v[5:9]), math.exp(v[10]), upper))
    if name == "unanchored_mdn_k2":
        v = np.asarray(params)
        pi0 = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, float(x @ v[:4])))))
        return pi0 * float(truncated_normal_cdf(z, float(x @ v[4:8]), math.exp(v[12]), upper)) + (1 - pi0) * float(truncated_normal_cdf(z, float(x @ v[8:12]), math.exp(v[13]), upper))
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="W5_DEVELOPMENT_FORMAL_MATRIX_PROPER_SCORES_NOT_CONFIRMATION")
    parser.add_argument("--caveat", default="Train threshold and scores are development-only; no model was selected and no confirmation conclusion is made.")
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    mean = float(checkpoint["metadata"]["target_mean"])
    scale = float(checkpoint["metadata"]["target_scale"])
    validation_fraction = float(checkpoint["metadata"].get("validation_fraction", 0.2))
    diagnostic_fraction = float(checkpoint["metadata"].get("diagnostic_fraction", 0.0))
    validation: dict[str, dict] = {}
    train_losses: list[float] = []
    with (args.merged / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("S1"):
                continue
            label = split_label(row["event_uid"], validation_fraction, diagnostic_fraction)
            if label == "validation":
                validation[row["event_uid"]] = row
            elif label == "train":
                train_losses.append(float(row["fractional_loss"]))
    threshold = float(np.quantile(train_losses, 0.99, method="linear"))
    predictions: dict[str, list[dict]] = {name: [] for name in checkpoint["models"]}
    with args.predictions.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["split"] == "validation":
                predictions[item["model"]].append(item)
    result_models = {}
    for name, items in predictions.items():
        if len(items) != len(validation):
            raise ValueError(f"{name}: expected {len(validation)} predictions, found {len(items)}")
        model = checkpoint["models"][name]
        params = model["params"]
        full_nll, censored, binary, brier, pits = [], [], [], [], []
        for item in items:
            row = validation[item["event_uid"]]
            p = float(row["p_in_GeV_c"])
            y = float(row["delta_p_GeV_c"])
            z = (y - mean) / scale
            upper = (p - mean) / scale
            anchor = (float(pion_bethe_bloch_delta_p(p, float(row["thickness_mm"]), float(row["pdg"]) / 211.0)) - mean) / scale
            x = design([row])[0]
            q = 1.0 - component_cdf(model["kind"], params, name, row, checkpoint, (p * threshold - mean) / scale, upper, anchor, x)
            pit = component_cdf(model["kind"], params, name, row, checkpoint, z, upper, anchor, x)
            observed_tail = float(row["fractional_loss"]) >= threshold
            logp = float(item["log_density"])
            full_nll.append(-logp)
            censored.append(-logp if observed_tail else -math.log(clip(1.0 - q)))
            binary.append(-(math.log(clip(q)) if observed_tail else math.log(clip(1.0 - q))))
            brier.append((q - observed_tail) ** 2)
            pits.append(pit)
        interval_coverage = {}
        for level in (0.5, 0.68, 0.9, 0.95, 0.99):
            alpha = (1 - level) / 2
            interval_coverage[str(level)] = float(np.mean((np.asarray(pits) >= alpha) & (np.asarray(pits) <= 1 - alpha)))
        result_models[name] = {
            "validation_events": len(items),
            "full_nll": statistics.fmean(full_nll),
            "censored_log_score": statistics.fmean(censored),
            "tail_binary_log_score": statistics.fmean(binary),
            "tail_brier": statistics.fmean(brier),
            "pit_mean": statistics.fmean(pits),
            "pit_variance": statistics.pvariance(pits),
            "interval_coverage_from_pit": interval_coverage,
        }
    report = {
        "stage": args.stage,
        "seed": checkpoint["metadata"]["seed"],
        "source_manifest_sha256": checkpoint["source_manifest_sha256"],
        "train_q99_fractional_loss": threshold,
        "validation_fraction": validation_fraction,
        "diagnostic_fraction": diagnostic_fraction,
        "tail_event_count": sum(float(row["fractional_loss"]) >= threshold for row in validation.values()),
        "models": result_models,
        "caveat": args.caveat,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seed": report["seed"], "models": len(result_models), "validation_events": len(validation)}))


if __name__ == "__main__":
    main()
