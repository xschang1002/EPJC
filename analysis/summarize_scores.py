"""Collect per-seed score JSON without averaging away optimization variation."""

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
    output = {
        "stage": "W5_DEVELOPMENT_SCORE_DIAGNOSTIC_NOT_CONFIRMATION",
        "scores": [json.loads((args.root / f"w3_dev_seed{seed}" / "proper_scores.json").read_text(encoding="utf-8")) for seed in args.seeds],
        "pooled_or_ensemble": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
