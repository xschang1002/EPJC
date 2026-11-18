"""Summarize the authorized optimization audit without selecting a winner."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_scores(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {(x["representation"], x["capacity"], int(x["seed"])): x for x in payload["runs"]}


def read_manifests(path: Path):
    out = {}
    for manifest_path in sorted((path / "runs").glob("*/run_manifest.json")):
        item = json.loads(manifest_path.read_text(encoding="utf-8"))
        out[(item["representation"], item["capacity"], int(item["seed"]))] = item
    return out


def grouped_contrasts(runs, left, right):
    grouped = defaultdict(list)
    for (representation, capacity, seed), item in runs.items():
        if representation == left:
            peer = runs.get((right, capacity, seed))
            if peer is not None:
                grouped[capacity].append(float(item["validation_nll"] - peer["validation_nll"]))
    return {
        capacity: {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)), "values": values}
        for capacity, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("reports/representation_v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/optimization_audit/optimization_audit_summary.json"))
    args = parser.parse_args()
    base_scores, base = read_scores(args.base / "prototype_scores.json")
    lr_scores, lr = read_scores(args.base / "optimization_audit/lr_1e-3/prototype_scores.json")
    long_scores, long = read_scores(args.base / "optimization_audit/long_budget_large_6000/prototype_scores.json")
    base_manifests = read_manifests(args.base)
    lr_manifests = read_manifests(args.base / "optimization_audit/lr_1e-3")
    long_manifests = read_manifests(args.base / "optimization_audit/long_budget_large_6000")

    lr_delta = {}
    for key, item in lr.items():
        if key in base:
            lr_delta["__".join((key[0], key[1], str(key[2])))] = {
                "base_3e-4_nll": float(base[key]["validation_nll"]),
                "lr_1e-3_nll": float(item["validation_nll"]),
                "lr_minus_base": float(item["validation_nll"] - base[key]["validation_nll"]),
                "base_selected_update": int(base[key]["selected_update"]),
                "lr_selected_update": int(item["selected_update"]),
            }
    long_delta = {}
    for key, item in long.items():
        if key in base:
            long_delta["__".join((key[0], key[1], str(key[2])))] = {
                "base_2000_nll": float(base[key]["validation_nll"]),
                "long_6000_nll": float(item["validation_nll"]),
                "long_minus_base": float(item["validation_nll"] - base[key]["validation_nll"]),
                "base_selected_update": int(base[key]["selected_update"]),
                "long_selected_update": int(item["selected_update"]),
            }
    gradient_ranges = {}
    for label, manifests in (("lr_1e-3", lr_manifests), ("long_6000", long_manifests)):
        gradient_ranges[label] = {}
        for key, item in manifests.items():
            values = [float(row["grad_norm_pre_clip"]) for row in item["history"] if "grad_norm_pre_clip" in row]
            gradient_ranges[label]["__".join((key[0], key[1], str(key[2])))] = {
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
                "last": float(values[-1]) if values else None,
                "points": len(values),
            }
    all_elapsed = [float(item["elapsed_seconds"]) for item in long_scores["runs"]]
    result = {
        "status": "COMPLETED_OPTIMIZATION_AUDIT",
        "authorization": "A0_plus_prototype_plus_optimization_audit",
        "scopes": {
            "legacy": "24 cells, 3e-4, 2000 updates; existing logs audited, gradient field absent",
            "second_learning_rate": "24 symmetric cells, 1e-3, 2000 updates",
            "long_budget": "12 symmetric cells, Large only, 3e-4, 6000 updates",
            "center_diagnostics": "train/validation source-domain diagnostics; no new models",
        },
        "legacy_large_seed11": json.loads((args.base / "optimization_audit/training_curve_audit.json").read_text(encoding="utf-8"))["large_seed11_focus"],
        "second_learning_rate": {
            "status": lr_scores["status"],
            "failed_runs": len(lr_scores["failed_runs"]),
            "raw_minus_physics": grouped_contrasts(lr, "Raw", "Physics"),
            "learned_minus_physics": grouped_contrasts(lr, "Learned", "Physics"),
            "simple_minus_raw": grouped_contrasts(lr, "Simple", "Raw"),
            "per_seed_deltas_vs_base": lr_delta,
            "gradient_ranges": gradient_ranges["lr_1e-3"],
            "mean_seconds_per_run": float(np.mean([float(item["elapsed_seconds"]) for item in lr_scores["runs"]])),
        },
        "long_budget": {
            "status": long_scores["status"],
            "failed_runs": len(long_scores["failed_runs"]),
            "per_seed_deltas_vs_2000": long_delta,
            "mean_seconds_per_run": float(np.mean(all_elapsed)),
            "total_hours": float(np.sum(all_elapsed) / 3600.0),
            "gradient_ranges": gradient_ranges["long_6000"],
        },
        "center_diagnostics_file": "center_diagnostics.json",
        "interpretation_boundary": "Optimization audit evidence is separate from Raw/Physics performance, Physics/Learned specificity, capacity/sample interaction, OOD generalization, and mechanism claims. No formal 120-fit matrix was started.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "lr_runs": len(lr_scores["runs"]), "long_runs": len(long_scores["runs"]), "long_hours": result["long_budget"]["total_hours"]}))


if __name__ == "__main__":
    main()
