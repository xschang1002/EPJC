"""Check actual primary step lengths and registered limiter in a Geant4 chunk."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.chunk / "manifest.json").read_text(encoding="utf-8"))
    limit = manifest["max_slab_step_mm_requested"]
    if limit is None or limit <= 0:
        raise ValueError("chunk has no requested slab step limit")
    for pion in ("pi+", "pi-"):
        names = {entry["name"] for entry in manifest["pion_process_table"][pion]}
        if "StepLimiter" not in names:
            raise ValueError(f"{pion} lacks registered StepLimiter")
    count = 0
    triggered = 0
    maximum = 0.0
    with (args.chunk / "primary_steps.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["pre_volume"] != "Slab":
                continue
            length = row["step_length_mm"]
            if length < 0 or length > limit * (1 + 1e-6) + 1e-9:
                raise ValueError(f"slab step exceeds requested limit: {length} > {limit}")
            count += 1
            maximum = max(maximum, length)
            triggered += row["process_name"] == "StepLimiter"
    if not count or not triggered:
        raise ValueError("no slab steps or no actual StepLimiter-triggered step")
    print(json.dumps({"passed": True, "slab_steps": count, "max_step_mm": maximum, "requested_limit_mm": limit, "limiter_triggered_steps": triggered}))


if __name__ == "__main__":
    main()
