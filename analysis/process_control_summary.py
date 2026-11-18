"""Compare fixed-input process controls descriptively; no causal claim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controls = []
    for root in args.runs:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (root / "events.jsonl").open(encoding="utf-8")]
        losses = sorted(row["fractional_loss"] for row in rows if row["S1"])
        controls.append({
            "run": str(root),
            "physics_list": manifest["physics_list"],
            "removed_pion_processes": manifest["removed_pion_processes"],
            "beam_seed": manifest["beam_seed"],
            "transport_seed": manifest["transport_seed"],
            "events": len(rows),
            "S1": sum(row["S1"] for row in rows),
            "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
            "secondary_count_in_slab": manifest["new_secondaries_in_slab"],
            "q50_fractional_loss": losses[len(losses) // 2] if losses else None,
            "q99_fractional_loss": losses[min(len(losses) - 1, int(0.99 * (len(losses) - 1)))] if losses else None,
            "max_fractional_loss": max(losses, default=None),
            "process_table": manifest["pion_process_table"],
        })
    output = {
        "stage": "W5_PROCESS_CONTROL_EXPLORATORY_NOT_CONFIRMATION",
        "same_incident_beam_seed": len({item["beam_seed"] for item in controls}) == 1,
        "same_transport_seed": len({item["transport_seed"] for item in controls}) == 1,
        "controls": controls,
        "caveat": "These controls have matched source settings but are separate Geant4 runs; identical seeds do not make them automatic event-level counterfactuals.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
