"""Quality and paired-result audit for an exploratory matrix."""

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


def _stats(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": int(len(values)), "mean": mean, "sd": sd, "se": se, "ci95_low": mean - T95_DF4 * se, "ci95_high": mean + T95_DF4 * se, "min": float(np.min(values)), "max": float(np.max(values)), "values": [float(v) for v in values]}


def audit(root: Path) -> dict:
    scores = json.loads((root / "exploratory_scores.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "exploratory_plan.json").read_text(encoding="utf-8"))
    max_updates = int(plan.get("max_updates", 6000))
    validation_every = int(plan.get("validation_every", 500))
    expected_history_points = 1 + max_updates // validation_every + int(max_updates % validation_every != 0)
    failures = []
    rows = []
    uid_digest = None
    for item in scores["runs"]:
        run_id = item["run_id"]
        run_dir = root / "runs" / run_id
        manifest_path = run_dir / "run_manifest.json"
        pred_path = run_dir / "predictions.npz"
        checkpoint = run_dir / "checkpoint.pt"
        if not manifest_path.exists() or not pred_path.exists() or not checkpoint.exists():
            failures.append({"run_id": run_id, "issue": "missing_artifact"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pred = np.load(pred_path)
        required = {"event_uid", "y", "c", "mu", "s", "p_in", "z", "b", "log_q_z", "log_F_b", "log_p_y", "logCDF_y", "logSF_y"}
        if not required.issubset(pred.files):
            failures.append({"run_id": run_id, "issue": "missing_prediction_fields"})
            continue
        finite = all(np.isfinite(pred[field]).all() for field in ("z", "b", "log_q_z", "log_F_b", "log_p_y", "logCDF_y", "logSF_y"))
        if not finite:
            failures.append({"run_id": run_id, "issue": "nonfinite_prediction"})
        contract = float(np.max(np.abs(pred["log_p_y"] - (pred["log_q_z"] - np.log(pred["s"]) - pred["log_F_b"]))))
        if contract > 5e-6:
            failures.append({"run_id": run_id, "issue": "likelihood_contract", "max_abs_error": contract})
        digest = _uid_digest(pred["event_uid"])
        uid_digest = digest if uid_digest is None else uid_digest
        if digest != uid_digest:
            failures.append({"run_id": run_id, "issue": "validation_uid_order"})
        history = manifest.get("history", [])
        if len(history) != expected_history_points or not all("grad_norm_pre_clip" in point and math.isfinite(float(point["grad_norm_pre_clip"])) for point in history):
            failures.append({"run_id": run_id, "issue": "history_or_gradient"})
        rows.append({"run_id": run_id, "representation": item["representation"], "capacity": item["capacity"], "seed": item["seed"], "validation_nll": float(np.mean(-pred["log_p_y"])), "selected_update": item["selected_update"], "max_gradient_norm_pre_clip": float(max(point["grad_norm_pre_clip"] for point in history)), "log_sf_finite_fraction": float(np.mean(np.isfinite(pred["logSF_y"])))})

    grouped = defaultdict(list)
    for row in scores["paired_comparisons"]:
        context_mode = row.get("context_feature_mode")
        fraction = row.get("sample_fraction")
        if context_mode is None or fraction is None:
            key = (row["contrast"], row["capacity"])
        else:
            key = (row["contrast"], context_mode, float(fraction), row["capacity"])
        grouped[key].append(row["nll_difference"])
    contrast_stats = {}
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        if len(key) == 2:
            contrast, capacity = key
            label = f"{contrast}__{capacity}"
        else:
            contrast, context_mode, fraction, capacity = key
            label = f"{contrast}__{context_mode}__f{int(round(100 * fraction)):03d}__{capacity}"
        contrast_stats[label] = _stats(values)
    nll_grouped = defaultdict(list)
    for row in rows:
        nll_grouped[(row["representation"], row["capacity"])].append(row["validation_nll"])
    nll_stats = {f"{representation}__{capacity}": _stats(values) for (representation, capacity), values in sorted(nll_grouped.items())}
    result = {
        "status": "EXPLORATORY_QUALITY_AUDIT_PASS" if not failures and len(rows) == len(scores["runs"]) else "EXPLORATORY_QUALITY_AUDIT_FAIL",
        "plan": plan,
        "runs_observed": len(rows),
        "failed_runs": scores.get("failed_runs", []),
        "quality_failures": failures,
        "validation_uid_sha256": uid_digest,
        "per_run": rows,
        "nll_by_representation_capacity": nll_stats,
        "paired_contrasts": contrast_stats,
        "interpretation_boundary": "This is an exploratory architecture-stress result on the source domain. It does not change the formal protocol and does not establish OOD or universal Physics superiority.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/representation_v2/exploratory/context_bottleneck"))
    args = parser.parse_args()
    result = audit(args.root)
    path = args.root / "exploratory_quality_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": result["runs_observed"], "quality_failures": len(result["quality_failures"]), "output": str(path)}, sort_keys=True))
    if result["status"] != "EXPLORATORY_QUALITY_AUDIT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
