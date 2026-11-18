"""Export all eligible rows for a saved formal-matrix checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.fit_formal_matrix import design, predict_model, read_rows
from data_contract.event_split import split_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    mean = float(checkpoint["metadata"]["target_mean"])
    scale = float(checkpoint["metadata"]["target_scale"])
    validation_fraction = float(checkpoint["metadata"].get("validation_fraction", 0.2))
    diagnostic_fraction = float(checkpoint["metadata"].get("diagnostic_fraction", 0.0))
    rows = read_rows(args.merged / "events.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            p = float(row["p_in_GeV_c"])
            q = float(row["pdg"]) / 211.0
            thickness = float(row["thickness_mm"])
            y = float(row["delta_p_GeV_c"])
            z = (y - mean) / scale
            upper = (p - mean) / scale
            anchor = (float(pion_bethe_bloch_delta_p(p, thickness, q)) - mean) / scale
            x = design([row])
            for name, model in checkpoint["models"].items():
                log_density = float(predict_model(name, model, np.asarray([z]), np.asarray([upper]), np.asarray([anchor]), x)[0] - math.log(scale))
                stream.write(json.dumps({
                    "event_uid": row["event_uid"],
                    "seed": checkpoint["metadata"]["seed"],
                    "model": name,
                    "split": split_label(row["event_uid"], validation_fraction, diagnostic_fraction),
                    "p_in_GeV_c": p,
                    "pdg": int(row["pdg"]),
                    "thickness_mm": thickness,
                    "y_delta_p_GeV_c": y,
                    "fractional_loss": float(row["fractional_loss"]),
                    "anchor_center_GeV_c": float(pion_bethe_bloch_delta_p(p, thickness, q)),
                    "log_density": log_density,
                }, sort_keys=True, allow_nan=False) + "\n")
                count += 1
    print(json.dumps({"output": str(args.output), "eligible_events": len(rows), "prediction_rows": count, "models": len(checkpoint["models"])}))


if __name__ == "__main__":
    main()
