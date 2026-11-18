"""Audit existing prototype curves and checkpoints without fitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def audit(base: Path):
    manifests = sorted((base / "runs").glob("prototype__*/run_manifest.json"))
    runs = []
    for path in manifests:
        item = json.loads(path.read_text(encoding="utf-8"))
        history = item.get("history", [])
        val = np.asarray([row["validation_nll"] for row in history], dtype=float)
        train = np.asarray([row["train_nll"] for row in history], dtype=float)
        grad_present = all("grad_norm_pre_clip" in row for row in history)
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        checkpoint_ok = checkpoint.exists()
        optimizer_state = False
        if checkpoint_ok:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            optimizer_state = "optimizer_state_dict" in payload and bool(payload["optimizer_state_dict"].get("state"))
        prediction_path = Path(item["predictions"])
        if not prediction_path.is_absolute():
            prediction_path = Path.cwd() / prediction_path
        prediction_ok = prediction_path.exists()
        exported_nll = None
        finite_sf = None
        if prediction_ok:
            with np.load(prediction_path) as prediction:
                exported_nll = float(np.mean(-prediction["log_p_y"]))
                finite_sf = float(np.mean(np.isfinite(prediction["logSF_y"])))
        best_index = int(np.argmin(val))
        selected_history = next((row for row in history if int(row["update"]) == int(item["selected_update"])), history[best_index])
        entry = {
            "run_id": item["run_id"],
            "representation": item["representation"],
            "capacity": item["capacity"],
            "seed": item["seed"],
            "history_points": len(history),
            "history": history,
            "gradient_field_present": grad_present,
            "gradient_audit": "MISSING_IN_LEGACY_2000_UPDATE_LOG" if not grad_present else "RECORDED",
            "selected_update": int(item["selected_update"]),
            "best_history_update": int(history[best_index]["update"]),
            "best_validation_nll": float(val[best_index]),
            "final_validation_nll": float(val[-1]),
            "final_minus_best_validation_nll": float(val[-1] - val[best_index]),
            "final_train_nll": float(train[-1]),
            "final_validation_minus_train_nll": float(val[-1] - train[-1]),
            "validation_nll_nonfinite": bool(np.any(~np.isfinite(val))),
            "train_nll_upward_steps": int(np.sum(np.diff(train) > 0)),
            "validation_nll_upward_steps": int(np.sum(np.diff(val) > 0)),
            "checkpoint_exists": checkpoint_ok,
            "checkpoint_optimizer_state_present": optimizer_state,
            "prediction_exists": prediction_ok,
            "exported_validation_nll": exported_nll,
            "exported_nll_abs_diff_from_selected_history": None if exported_nll is None else float(abs(exported_nll - float(selected_history["validation_nll"]))),
            "exported_log_sf_finite_fraction": finite_sf,
        }
        runs.append(entry)
    # The expression above deliberately avoids using exported values as a
    # selection rule; checkpoint selection remains the saved source-validation rule.
    focus = [item for item in runs if item["capacity"] == "Large" and item["seed"] == 11]
    return {
        "status": "COMPLETED_EXISTING_LOG_AUDIT",
        "scope": "legacy 24-cell prototype logs only; no refit",
        "run_count": len(runs),
        "gradient_fields_missing_count": int(sum(not item["gradient_field_present"] for item in runs)),
        "checkpoint_missing_count": int(sum(not item["checkpoint_exists"] for item in runs)),
        "optimizer_state_missing_count": int(sum(not item["checkpoint_optimizer_state_present"] for item in runs)),
        "prediction_missing_count": int(sum(not item["prediction_exists"] for item in runs)),
        "large_seed11_focus": focus,
        "runs": runs,
        "interpretation_boundary": "Legacy curves establish checkpoint and loss behavior; they do not provide gradient norms because that field was not recorded. New audit runs record gradients.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("reports/representation_v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/optimization_audit/training_curve_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"audit output already exists; choose a new file: {args.output}")
    result = audit(args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    focus = result["large_seed11_focus"]
    print(json.dumps({"status": result["status"], "runs": result["run_count"], "gradient_missing": result["gradient_fields_missing_count"], "large_seed11_runs": len(focus)}))


if __name__ == "__main__":
    main()
