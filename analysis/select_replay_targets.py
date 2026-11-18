"""Select a predetermined ordinary event and the pilot's largest loss event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_contract.beam_schedule import CELLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("--ordinary-event-index", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = None
    ordinary = None
    with (args.merged / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["event_index"] == args.ordinary_event_index:
                ordinary = row
            if row["S1"] and (selected is None or row["fractional_loss"] > selected["fractional_loss"]):
                selected = row
    if selected is None or ordinary is None:
        raise ValueError("required replay targets missing")
    targets = {}
    for label, row in (("ordinary", ordinary), ("maximum_fractional_loss", selected)):
        cell = (row["momentum_bin"], row["pdg"], row["thickness_mm"])
        targets[label] = {
            "event_uid": row["event_uid"],
            "event_index": row["event_index"],
            "chunk_id": CELLS.index(cell),
            "fractional_loss": row["fractional_loss"],
            "status": row["status"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(targets, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(targets, indent=2))


if __name__ == "__main__":
    main()
