"""Summarize matched process-control retraining and proper scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("controls", nargs="+", help="name=score_json pairs")
    args = parser.parse_args()
    results = {}
    for item in args.controls:
        name, path = item.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        results[name] = {
            "stage": report["stage"],
            "seed": report["seed"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "validation_events": report["models"][next(iter(report["models"]))]["validation_events"],
            "models": report["models"],
        }
    output = {
        "stage": "W5_MATCHED_PROCESS_CONTROL_RETRAINING_DEVELOPMENT",
        "same_training_seed": len({item["seed"] for item in results.values()}) == 1,
        "selection_performed": False,
        "controls": results,
        "caveat": "Separate Geant4 process-switch runs are compared with the same training code and seed; this is a descriptive control, not an event-level counterfactual or confirmation result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "controls": sorted(results), "same_training_seed": output["same_training_seed"]}))


if __name__ == "__main__":
    main()
