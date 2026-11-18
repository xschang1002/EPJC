"""Collect formal-matrix proper scores without selecting a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = [json.loads((args.root / f"development_200k_scores_seed{seed}.json").read_text(encoding="utf-8")) for seed in args.seeds]
    model_names = sorted(scores[0]["models"])
    for score in scores[1:]:
        if sorted(score["models"]) != model_names:
            raise ValueError("model sets differ across seeds")
    result = {
        "stage": "W5_DEVELOPMENT_FORMAL_MATRIX_SCORE_SUMMARY_NOT_CONFIRMATION",
        "seeds": args.seeds,
        "models": model_names,
        "scores": scores,
        "selection_performed": False,
        "pooled_or_ensemble": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seeds": args.seeds, "models": len(model_names)}))


if __name__ == "__main__":
    main()
