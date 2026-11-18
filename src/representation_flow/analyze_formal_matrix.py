"""Quality audit and paired uncertainty summary for the formal 120-cell study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


T95_DF4 = 2.7764451051977987


def _uid_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stats(values, ci_t=T95_DF4):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - ci_t * se,
        "ci95_high": mean + ci_t * se,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": [float(value) for value in values],
    }


def _group_stats(rows, key_fields, value_field):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    result = {}
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        result["__".join(str(value) for value in key)] = {**{field: value for field, value in zip(key_fields, key)}, **_stats([row[value_field] for row in values])}
    return result


def audit(formal_dir: Path) -> dict:
    scores = json.loads((formal_dir / "formal_scores.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((formal_dir / "formal_model_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((formal_dir / "formal_dataset_manifest.json").read_text(encoding="utf-8"))
    if scores.get("status") != "COMPLETED_FORMAL_CORE_120":
        raise RuntimeError(f"formal matrix is not complete: {scores.get('status')}")

    quality_failures = []
    rows = []
    validation_uid_digest = None
    initialization_groups = defaultdict(list)
    for item in scores["runs"]:
        run_id = item["run_id"]
        run_dir = formal_dir / "runs" / run_id
        manifest_path = run_dir / "run_manifest.json"
        prediction_path = run_dir / "predictions.npz"
        checkpoint_path = run_dir / "checkpoint.pt"
        if not manifest_path.exists() or not prediction_path.exists() or not checkpoint_path.exists():
            quality_failures.append({"run_id": run_id, "issue": "missing_run_artifact"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pred = np.load(prediction_path)
        keys = set(pred.files)
        required = {"event_uid", "split", "subset_id", "n", "y", "c", "mu", "s", "p_in", "z", "b", "log_q_z", "log_F_b", "log_p_y", "logCDF_y", "logSF_y"}
        missing = sorted(required - keys)
        if missing:
            quality_failures.append({"run_id": run_id, "issue": "missing_prediction_fields", "fields": missing})
            continue
        finite_fields = {field: bool(np.isfinite(pred[field]).all()) for field in ("z", "b", "log_q_z", "log_F_b", "log_p_y", "logCDF_y", "logSF_y")}
        if not all(finite_fields.values()):
            quality_failures.append({"run_id": run_id, "issue": "nonfinite_prediction_field", "fields": [field for field, ok in finite_fields.items() if not ok]})
        z_expected = (pred["y"] - pred["c"] - pred["mu"]) / pred["s"]
        b_expected = (pred["p_in"] - pred["c"] - pred["mu"]) / pred["s"]
        density_expected = pred["log_q_z"] - np.log(pred["s"]) - pred["log_F_b"]
        consistency = {
            "z_max_abs_error": float(np.max(np.abs(pred["z"] - z_expected))),
            "b_max_abs_error": float(np.max(np.abs(pred["b"] - b_expected))),
            "log_p_y_contract_max_abs_error": float(np.max(np.abs(pred["log_p_y"] - density_expected))),
        }
        # Predictions are intentionally stored as float32; these tolerances audit
        # the algebra without mistaking serialization round-off for a likelihood bug.
        if consistency["z_max_abs_error"] > 1e-5 or consistency["b_max_abs_error"] > 1e-3 or consistency["log_p_y_contract_max_abs_error"] > 5e-6:
            quality_failures.append({"run_id": run_id, "issue": "likelihood_contract_mismatch", "details": consistency})
        uid_digest = _uid_digest(pred["event_uid"])
        validation_uid_digest = uid_digest if validation_uid_digest is None else validation_uid_digest
        if uid_digest != validation_uid_digest:
            quality_failures.append({"run_id": run_id, "issue": "validation_uid_order_mismatch"})
        history = manifest.get("history", [])
        gradient_present = all("grad_norm_pre_clip" in point for point in history)
        gradient_finite = all(math.isfinite(float(point["grad_norm_pre_clip"])) for point in history if "grad_norm_pre_clip" in point)
        selected_update = int(manifest["selected_update"])
        best_history_update = int(min(history, key=lambda point: point["validation_nll"])["update"]) if history else None
        if not history or not gradient_present or not gradient_finite or best_history_update != selected_update or len(history) != 41:
            quality_failures.append({"run_id": run_id, "issue": "history_or_checkpoint_rule", "history_points": len(history), "gradient_present": gradient_present, "gradient_finite": gradient_finite, "selected_update": selected_update, "best_history_update": best_history_update})
        initialization_groups[(item["sample_fraction"], item["capacity"], item["seed"])].append(model_manifest["models"][next(index for index, model in enumerate(model_manifest["models"]) if model["run_id"] == run_id)]["initialization_hash"])
        rows.append({
            "run_id": run_id,
            "representation": item["representation"],
            "capacity": item["capacity"],
            "sample_fraction": item["sample_fraction"],
            "seed": item["seed"],
            "train_events": item["train_events"],
            "validation_nll": float(np.mean(-pred["log_p_y"])),
            "selected_update": selected_update,
            "final_validation_nll": float(history[-1]["validation_nll"]),
            "final_minus_best_validation_nll": float(history[-1]["validation_nll"] - min(point["validation_nll"] for point in history)),
            "max_gradient_norm_pre_clip": float(max(point["grad_norm_pre_clip"] for point in history)),
            "prediction_contract": consistency,
            "log_sf_finite_fraction": float(np.mean(np.isfinite(pred["logSF_y"]))),
        })

    for key, hashes in initialization_groups.items():
        if len(set(hashes)) != 1:
            quality_failures.append({"group": key, "issue": "paired_initialization_hash_mismatch", "hashes": hashes})

    paired = []
    paired_by_key = {(row["representation"], row["capacity"], row["sample_fraction"], row["seed"]): row for row in rows}
    for fraction in (0.1, 0.3, 1.0):
        for capacity in ("Small", "Large"):
            for seed in (3101, 3102, 3103, 3104, 3105):
                for left, right, contrast in (("Raw", "Physics", "Raw_minus_Physics"), ("Learned", "Physics", "Learned_minus_Physics"), ("Simple", "Raw", "Simple_minus_Raw")):
                    a = paired_by_key.get((left, capacity, fraction, seed))
                    b = paired_by_key.get((right, capacity, fraction, seed))
                    if a and b:
                        paired.append({"sample_fraction": fraction, "capacity": capacity, "seed": seed, "contrast": contrast, "nll_difference": a["validation_nll"] - b["validation_nll"]})

    grouped_nll = _group_stats(rows, ("representation", "capacity", "sample_fraction"), "validation_nll")
    grouped_paired = _group_stats(paired, ("contrast", "capacity", "sample_fraction"), "nll_difference")
    aggregate_paired = {}
    for contrast in ("Raw_minus_Physics", "Learned_minus_Physics", "Simple_minus_Raw"):
        values = [row["nll_difference"] for row in paired if row["contrast"] == contrast]
        aggregate_paired[contrast] = _stats(values, ci_t=2.045229642132703)  # 95% t interval, df=29; descriptive across 30 paired cells.

    gains = {(row["capacity"], row["sample_fraction"], row["seed"]): row["nll_difference"] for row in paired if row["contrast"] == "Raw_minus_Physics"}
    capacity_interactions = []
    sample_interactions = []
    for fraction in (0.1, 0.3, 1.0):
        for seed in (3101, 3102, 3103, 3104, 3105):
            capacity_interactions.append({"sample_fraction": fraction, "seed": seed, "small_minus_large_gain": gains[("Small", fraction, seed)] - gains[("Large", fraction, seed)]})
    for capacity in ("Small", "Large"):
        for seed in (3101, 3102, 3103, 3104, 3105):
            sample_interactions.append({"capacity": capacity, "seed": seed, "gain_10pct_minus_gain_100pct": gains[(capacity, 0.1, seed)] - gains[(capacity, 1.0, seed)]})

    result = {
        "status": "FORMAL_QUALITY_AUDIT_PASS" if not quality_failures and len(rows) == 120 else "FORMAL_QUALITY_AUDIT_FAIL",
        "formal_matrix_status": scores["status"],
        "runs_expected": 120,
        "runs_observed": len(rows),
        "failed_runs": scores.get("failed_runs", []),
        "quality_failures": quality_failures,
        "validation_uid_sha256": validation_uid_digest,
        "dataset_manifest": dataset_manifest,
        "per_run": rows,
        "grouped_nll": grouped_nll,
        "paired_contrasts": paired,
        "grouped_paired_contrasts": grouped_paired,
        "aggregate_paired_contrasts": aggregate_paired,
        "interactions": {
            "capacity_small_minus_large_raw_minus_physics": {"per_seed": capacity_interactions, **_stats([item["small_minus_large_gain"] for item in capacity_interactions])},
            "sample_gain_10pct_minus_100pct_raw_minus_physics": {"per_seed": sample_interactions, **_stats([item["gain_10pct_minus_gain_100pct"] for item in sample_interactions])},
        },
        "interpretation_boundary": "These are formal source-domain matched representation results. Optimization diagnostics, center diagnostics, OOD, spline-bound sensitivity, and mixture mechanisms remain separate evidence layers.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, default=Path("reports/representation_v2/formal_matrix"))
    args = parser.parse_args()
    result = audit(args.formal_dir)
    path = args.formal_dir / "formal_quality_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": result["runs_observed"], "quality_failures": len(result["quality_failures"]), "output": str(path)}, sort_keys=True))
    if result["status"] != "FORMAL_QUALITY_AUDIT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
