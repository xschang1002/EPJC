"""Summarize completed formal-matrix outputs without fitting or selecting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = sorted(args.run.glob("checkpoint_seed*.json"))
    if not checkpoints:
        raise ValueError("no checkpoints found")
    results = []
    models = None
    source_hashes = set()
    for checkpoint_path in checkpoints:
        seed = int(checkpoint_path.stem.replace("checkpoint_seed", ""))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        scores = json.loads((args.run / f"scores_seed{seed}.json").read_text(encoding="utf-8"))
        predictions = args.run / f"predictions_seed{seed}.jsonl"
        if not predictions.exists():
            raise ValueError(f"missing predictions for seed {seed}")
        current_models = sorted(checkpoint["models"])
        if models is None:
            models = current_models
        if current_models != models or sorted(scores) != models:
            raise ValueError(f"model mismatch for seed {seed}")
        source_hashes.add(checkpoint["source_manifest_sha256"])
        results.append({"seed": seed, "scores": scores, "checkpoint": str(checkpoint_path)})
    summary = {
        "stage": "DEVELOPMENT_FORMAL_MATRIX_PARTIAL",
        "completed_seeds": [item["seed"] for item in results],
        "expected_formal_seeds": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "completed_count": len(results),
        "expected_count": 10,
        "models": models,
        "source_manifest_hashes": sorted(source_hashes),
        "results": results,
        "selection_performed": False,
        "confirmation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "completed": len(results), "expected": 10, "models": len(models)}))


if __name__ == "__main__":
    main()
