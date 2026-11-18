"""Summarize development seed outputs without pooling densities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        checkpoint = json.loads((args.root / f"w3_dev_seed{seed}" / "checkpoint.json").read_text(encoding="utf-8"))
        summary = json.loads((args.root / f"w3_dev_seed{seed}" / "attribution" / "summary.json").read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            "pi0": checkpoint["mixture"]["pi0"],
            "sigma0_z": checkpoint["mixture"]["sigma0_z"],
            "sigma1_z": checkpoint["mixture"]["sigma1_z"],
            "train_events": checkpoint["train_events"],
            "validation_events": checkpoint["validation_events"],
            "mean_gain_total_all": summary["mean_gain_total_all"],
            "mean_gain_direct_tail": summary["mean_gain_direct_tail"],
            "largest_positive_gain_total": summary["largest_positive_gain_total"],
            "tail_rows": summary["tail_rows"],
        })
    output = {
        "stage": "W3_W4_DEVELOPMENT_SEED_DIAGNOSTIC_NOT_CONFIRMATION",
        "seeds": rows,
        "pi0_mean": statistics.fmean(row["pi0"] for row in rows),
        "pi0_stdev": statistics.stdev(row["pi0"] for row in rows) if len(rows) > 1 else 0.0,
        "no_seed_pooling": True,
        "caveat": "These are pilot development fits with a deliberately small model; no model selection or confirmation claim is made.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
