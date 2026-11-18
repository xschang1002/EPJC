"""Aggregate main-batch proper scores across the predeclared seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
METRICS = ("full_nll", "censored_log_score", "tail_binary_log_score", "tail_brier", "pit_mean", "pit_variance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    model_names = None
    source_hashes = set()
    split_specs = set()
    for seed in SEEDS:
        path = args.reports_dir / f"main_1000000_scores_seed{seed}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
        names = sorted(report["models"])
        model_names = names if model_names is None else model_names
        if names != model_names:
            raise ValueError(f"model mismatch for seed {seed}")
        source_hashes.add(report["source_manifest_sha256"])
        first_model = next(iter(report["models"].values()))
        split_specs.add((report.get("validation_fraction"), report.get("diagnostic_fraction"), first_model["validation_events"]))
    if len(source_hashes) != 1 or len(split_specs) != 1:
        raise ValueError("source or split metadata differ across score reports")
    aggregate = {}
    for name in model_names:
        aggregate[name] = {}
        for metric in METRICS:
            values = [float(report["models"][name][metric]) for report in reports]
            aggregate[name][metric + "_mean"] = statistics.fmean(values)
            aggregate[name][metric + "_stdev"] = statistics.stdev(values)
            aggregate[name][metric + "_per_seed"] = {str(seed): value for seed, value in zip(SEEDS, values)}
    result = {
        "stage": "W6_MAIN_BATCH_PROPER_SCORES_COMPLETE_NO_SELECTION",
        "confirmation": False,
        "selection_performed": False,
        "seeds": SEEDS,
        "models": model_names,
        "source_manifest_sha256": sorted(source_hashes)[0],
        "split_spec": next(iter(split_specs)),
        "aggregate": aggregate,
        "reports": [str(args.reports_dir / f"main_1000000_scores_seed{seed}.json") for seed in SEEDS],
        "caveat": "Main-batch proper scores only; no confirmation data were opened and no model was selected here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seeds": SEEDS, "models": len(model_names)}))


if __name__ == "__main__":
    main()
