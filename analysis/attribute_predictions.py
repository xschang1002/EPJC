"""Regenerate exact two-term and three-term density attribution from files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty quantile sample")
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (position - lo)


def mean(values):
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]
    train_losses = [row["y_delta_p_GeV_c"] / row["p_in_GeV_c"] for row in rows if row["split"] == "train"]
    threshold = quantile(train_losses, 0.99)
    attributed = []
    for row in rows:
        total = row["log_f_full"] - row["log_f_reference"]
        anchor_ind = row["log_f_anchor_ind"] - row["log_f_reference"]
        reorganization = row["log_f_anchor_joint"] - row["log_f_anchor_ind"]
        direct = row["log_f_full"] - row["log_f_anchor_joint"]
        direct_via_resp = row["log_pi0"] - row["log_responsibility0"]
        if not math.isclose(direct, direct_via_resp, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"direct gain identity failed for {row['event_uid']}")
        if not math.isclose(total, anchor_ind + reorganization + direct, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"three-term identity failed for {row['event_uid']}")
        attributed.append({
            **row,
            "gain_total": total,
            "gain_anchor_ind": anchor_ind,
            "gain_anchor_reorganization": reorganization,
            "gain_direct": direct,
            "pilot_train_q99_fractional_loss": threshold,
            "tail_q99_train": row["y_delta_p_GeV_c"] / row["p_in_GeV_c"] >= threshold,
        })
    tail = [row for row in attributed if row["tail_q99_train"]]
    positive = sorted((max(0.0, row["gain_total"]) for row in attributed), reverse=True)
    negative = sorted((min(0.0, row["gain_total"]) for row in attributed))
    summary = {
        "stage": "W4_DEVELOPMENT_ATTRIBUTION_NOT_CONFIRMATION",
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": checkpoint["seed"],
        "rows": len(attributed),
        "train_q99_fractional_loss": threshold,
        "tail_rows": len(tail),
        "mean_gain_total_all": mean([row["gain_total"] for row in attributed]),
        "mean_gain_direct_all": mean([row["gain_direct"] for row in attributed]),
        "mean_gain_anchor_reorganization_all": mean([row["gain_anchor_reorganization"] for row in attributed]),
        "mean_gain_total_tail": mean([row["gain_total"] for row in tail]),
        "mean_gain_direct_tail": mean([row["gain_direct"] for row in tail]),
        "positive_gain_total_sum": sum(positive),
        "negative_gain_total_sum": sum(negative),
        "largest_positive_gain_total": positive[0] if positive else None,
        "largest_negative_gain_total": negative[0] if negative else None,
        "identity_tolerance": "rel=1e-9, abs=1e-9",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "attribution.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in attributed:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
