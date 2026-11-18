"""Summarize a completed main-batch matrix without selecting a model."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    model_names = None
    source_hashes = set()
    split_specs = set()
    for run in args.runs:
        checkpoints = sorted(run.glob("checkpoint_seed*.json"))
        if len(checkpoints) != 1:
            raise ValueError(f"expected one checkpoint in {run}, found {len(checkpoints)}")
        checkpoint_path = checkpoints[0]
        seed = int(checkpoint_path.stem.replace("checkpoint_seed", ""))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        scores = json.loads((run / f"scores_seed{seed}.json").read_text(encoding="utf-8"))
        predictions = run / f"predictions_seed{seed}.jsonl"
        if not predictions.exists():
            raise ValueError(f"missing predictions for seed {seed}")
        names = sorted(checkpoint["models"])
        if model_names is None:
            model_names = names
        if names != model_names or sorted(scores) != model_names:
            raise ValueError(f"model mismatch for seed {seed}")
        metadata = checkpoint["metadata"]
        split_specs.add((metadata.get("validation_fraction", 0.2), metadata.get("diagnostic_fraction", 0.0), metadata["train_events"], metadata["validation_events"], metadata.get("diagnostic_events", 0)))
        source_hashes.add(checkpoint["source_manifest_sha256"])
        rows.append({"seed": seed, "scores": scores, "checkpoint": str(checkpoint_path), "metadata": metadata})
    seeds = sorted(row["seed"] for row in rows)
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, found {seeds}")
    if len(source_hashes) != 1 or len(split_specs) != 1:
        raise ValueError("source manifest or split metadata differ across seeds")
    aggregate = {}
    for name in model_names:
        values = [row["scores"][name]["validation_nll"] for row in rows]
        train_values = [row["scores"][name]["train_nll"] for row in rows]
        aggregate[name] = {
            "validation_nll_mean": statistics.fmean(values),
            "validation_nll_stdev": statistics.stdev(values),
            "train_nll_mean": statistics.fmean(train_values),
            "train_nll_stdev": statistics.stdev(train_values),
            "per_seed_validation_nll": {str(row["seed"]): row["scores"][name]["validation_nll"] for row in rows},
        }
    result = {
        "stage": "W6_MAIN_BATCH_FORMAL_MATRIX_COMPLETE_NO_SELECTION",
        "confirmation": False,
        "selection_performed": False,
        "completed_seeds": seeds,
        "expected_seeds": EXPECTED_SEEDS,
        "models": model_names,
        "source_manifest_sha256": sorted(source_hashes)[0],
        "split_spec": next(iter(split_specs)),
        "aggregate": aggregate,
        "runs": rows,
        "caveat": "Main-batch validation matrix only; no confirmation data were opened and no model was selected here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seeds": seeds, "models": len(model_names)}))


if __name__ == "__main__":
    main()
