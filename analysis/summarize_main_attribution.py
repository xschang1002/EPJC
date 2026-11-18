"""Aggregate closure and attribution summaries across main-batch seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for seed in SEEDS:
        run = args.runs_root if seed == 101 else args.runs_root.parent / f"formal_matrix_main_1000000_seed{seed}"
        path = run / f"attribution_seed{seed}.summary.json"
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    means = sorted(summaries[0]["means"])
    aggregate = {}
    for key in means:
        values = [float(item["means"][key]) for item in summaries]
        aggregate[key] = {"mean_across_seeds": statistics.fmean(values), "stdev_across_seeds": statistics.stdev(values), "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)}}
    result = {
        "stage": "W6_MAIN_BATCH_ATTRIBUTION_COMPLETE_NO_CONFIRMATION",
        "confirmation": False,
        "seeds": SEEDS,
        "events_per_seed": summaries[0]["events"],
        "primary_model": summaries[0]["primary_model"],
        "aggregate_means": aggregate,
        "max_closure_abs": {
            "direct": max(item["closure_max_abs"]["direct"] for item in summaries),
            "three_term": max(item["closure_max_abs"]["three_term"] for item in summaries),
        },
        "summaries": [str((args.runs_root if seed == 101 else args.runs_root.parent / f"formal_matrix_main_1000000_seed{seed}") / f"attribution_seed{seed}.summary.json") for seed in SEEDS],
        "caveat": "Main-batch attribution only; confirmation remains locked.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seeds": SEEDS, "max_closure_abs": result["max_closure_abs"]}))


if __name__ == "__main__":
    main()
