"""Summarize predeclared cut and step exploratory runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def q(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in args.names:
        run = args.root / f"scan_{name}_1000"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (run / "events.jsonl").open(encoding="utf-8")]
        losses = sorted(row["fractional_loss"] for row in events if row["S1"])
        slab_steps = [json.loads(line) for line in (run / "primary_steps.jsonl").open(encoding="utf-8") if json.loads(line)["pre_volume"] == "Slab"]
        rows.append({
            "name": name,
            "requested_cut_mm": manifest["requested_cut_mm"],
            "step_fraction": manifest["step_fraction_candidate"],
            "events": manifest["events"],
            "S1": len(losses),
            "S1_rate": len(losses) / manifest["events"],
            "q50_fractional_loss": q(losses, 0.5),
            "q95_fractional_loss": q(losses, 0.95),
            "q99_fractional_loss": q(losses, 0.99),
            "max_fractional_loss": max(losses, default=None),
            "primary_steps_per_event": manifest["primary_step_rows"] / manifest["events"],
            "max_slab_step_mm": max((row["step_length_mm"] for row in slab_steps), default=None),
            "new_secondaries_in_slab": manifest["new_secondaries_in_slab"],
        })
    output = {"stage": "W2_NUMERICAL_SCAN_EXPLORATORY_NOT_FROZEN", "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
