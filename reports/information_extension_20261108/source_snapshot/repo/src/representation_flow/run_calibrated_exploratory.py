"""Run a scoped exploratory matrix with a train-only calibrated Physics center."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from representation_flow.engine import fit_preprocessors, load_source_arrays, sha256_file, subset_indices
from representation_flow.run_a0_prototype import _train_one, prepare_new_output


SEEDS = (3101, 3102, 3103, 3104, 3105)
FRACTION = 0.1
SUBSET_CATALOG_ID = 4101
CAPACITY_WIDTHS = {"Tiny": 8, "Small": 32}
REPRESENTATIONS = ("Raw", "Physics", "PhysicsCalibrated", "Learned")


def _uid_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/physics_calibrated"))
    parser.add_argument("--max-updates", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    prepare_new_output(args.output)

    arrays, manifest = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_indices = subset_indices(arrays, fraction=FRACTION, catalog_id=SUBSET_CATALOG_ID)
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    preprocessors = fit_preprocessors(arrays, train_indices, include_physics_calibrated=True)
    code_hashes = {
        "engine.py": sha256_file(Path(__file__).with_name("engine.py")),
        "run_a0_prototype.py": sha256_file(Path(__file__).with_name("run_a0_prototype.py")),
        "run_calibrated_exploratory.py": sha256_file(Path(__file__)),
    }
    plan = {
        "status": "EXPLORATORY_PHYSICS_CALIBRATED_AUTHORIZED",
        "question": "Does a train-only scalar calibration rescue the Physics center without changing its functional shape?",
        "representation_definition": "PhysicsCalibrated = alpha * pion_bethe_bloch_delta_p, with alpha = argmin_train ||y-alpha*Physics||^2; alpha is fit once on the 10% train subset and then frozen",
        "control_representations": ["Raw", "Physics", "Learned"],
        "representations": list(REPRESENTATIONS),
        "context_feature_mode": "full",
        "context_definition": "common_features(p_in, thickness, charge); unchanged from the formal conditional flow",
        "capacities": CAPACITY_WIDTHS,
        "sample_fraction": FRACTION,
        "train_events": int(len(train_indices)),
        "validation_events": int(len(validation_indices)),
        "seeds": list(SEEDS),
        "learning_rate": args.learning_rate,
        "max_updates": args.max_updates,
        "batch_size": args.batch_size,
        "validation_every": args.validation_every,
        "same_checkpoint_rule": "source-validation all-event original-unit NLL",
        "new_geant4": False,
        "ood": False,
        "mixture_substitution": False,
        "formal_protocol_modified": False,
        "source_manifest_sha256": sha256_file(args.merged / "manifest.json"),
        "events_file_sha256": sha256_file(args.merged / "events.jsonl"),
        "subset_uid_sha256": _uid_digest(arrays["event_uid"][train_indices]),
        "code_hashes": code_hashes,
    }
    _save(args.output / "exploratory_plan.json", plan)
    scores_path = args.output / "exploratory_scores.json"
    scores = {
        "status": "RUNNING",
        "stage": "physics_calibrated",
        "runs": [],
        "failed_runs": [],
        "paired_comparisons": [],
        "plan": "exploratory_plan.json",
    }
    if scores_path.exists():
        previous_scores = json.loads(scores_path.read_text(encoding="utf-8"))
        if previous_scores.get("status") == "RUNNING":
            scores = previous_scores
    model_manifest = {
        "status": "RUNNING",
        "stage": "physics_calibrated",
        "code_hashes": code_hashes,
        "models": [],
        "centres": {},
    }
    manifest_path = args.output / "exploratory_model_manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("status") == "RUNNING":
            model_manifest = previous_manifest
    for name in REPRESENTATIONS:
        pre = preprocessors[name]
        model_manifest["centres"][name] = {
            "mu": pre.mu,
            "scale": pre.scale,
            "alpha": pre.alpha,
            "coefficients": [float(value) for value in np.asarray(pre.coefficients).ravel()],
            "coefficient_count": int(len(pre.coefficients)),
            "fit_labels": "10% source train subset only",
        }
    _save(manifest_path, model_manifest)

    started = time.perf_counter()
    q99 = float(np.quantile(arrays["y"][train_indices] / arrays["p_in"][train_indices], 0.99, method="linear"))
    for representation in REPRESENTATIONS:
        for capacity in CAPACITY_WIDTHS:
            for seed in SEEDS:
                run_id = f"explore__physics_calibrated__{representation}__{capacity}__f010__sub{SUBSET_CATALOG_ID}__seed{seed}"
                completed_keys = {(item["representation"], item["capacity"], item["seed"]) for item in scores["runs"]}
                if (representation, capacity, seed) in completed_keys:
                    continue
                run_args = SimpleNamespace(
                    run_prefix="explore_physics_calibrated",
                    run_id_template="explore__physics_calibrated__{representation}__{capacity}__f010__sub{subset_catalog_id}__seed{seed}",
                    fraction_tag="f010",
                    subset_catalog_id=SUBSET_CATALOG_ID,
                    sample_fraction=FRACTION,
                    run_status="COMPLETED_EXPLORATORY_PHYSICS_CALIBRATED",
                    feature_mode="full",
                    capacity_widths=CAPACITY_WIDTHS,
                    q99_threshold=q99,
                    batch_size=args.batch_size,
                    validation_every=args.validation_every,
                    eval_batch_size=args.eval_batch_size,
                    learning_rate=args.learning_rate,
                    max_updates=args.max_updates,
                )
                try:
                    result, prediction = _train_one(
                        arrays,
                        train_indices,
                        validation_indices,
                        preprocessors[representation],
                        representation,
                        capacity,
                        seed,
                        run_args,
                        args.output,
                    )
                    validation_nll = float(np.mean(-prediction["log_p_y"]))
                    scores["runs"].append(
                        {
                            "run_id": result["run_id"],
                            "representation": representation,
                            "capacity": capacity,
                            "hidden_width": CAPACITY_WIDTHS[capacity],
                            "context_feature_mode": "full",
                            "sample_fraction": FRACTION,
                            "seed": seed,
                            "validation_nll": validation_nll,
                            "selected_update": result["selected_update"],
                            "max_updates": args.max_updates,
                            "epochs_at_budget": result["epochs_at_budget"],
                            "elapsed_seconds": result["elapsed_seconds"],
                            "parameter_count": result["parameter_count"],
                            "predictions": result["predictions"],
                        }
                    )
                    model_manifest["models"].append(
                        {
                            "run_id": result["run_id"],
                            "representation": representation,
                            "capacity": capacity,
                            "seed": seed,
                            "hidden_width": CAPACITY_WIDTHS[capacity],
                            "context_dimension": 5,
                            "initialization_hash": result["initialization_hash"],
                            "checkpoint": result["checkpoint"],
                            "selected_update": result["selected_update"],
                            "optimizer": "Adam",
                            "learning_rate": args.learning_rate,
                        }
                    )
                    print(json.dumps({"completed": result["run_id"], "validation_nll": validation_nll, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                except Exception as exc:
                    failure = {
                        "run_id": run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "representation": representation,
                        "capacity": capacity,
                        "seed": seed,
                    }
                    scores["failed_runs"].append(failure)
                    print(json.dumps({"failed": failure}), flush=True)
                _save(scores_path, scores)
                _save(manifest_path, model_manifest)

    by_key = {(item["representation"], item["capacity"], item["seed"]): item for item in scores["runs"]}
    for capacity in CAPACITY_WIDTHS:
        for seed in SEEDS:
            for left, right, label in (
                ("Raw", "Physics", "Raw_minus_Physics"),
                ("Raw", "PhysicsCalibrated", "Raw_minus_PhysicsCalibrated"),
                ("PhysicsCalibrated", "Physics", "PhysicsCalibrated_minus_Physics"),
                ("Learned", "Physics", "Learned_minus_Physics"),
                ("Learned", "PhysicsCalibrated", "Learned_minus_PhysicsCalibrated"),
            ):
                a = by_key.get((left, capacity, seed))
                b = by_key.get((right, capacity, seed))
                if a and b:
                    scores["paired_comparisons"].append(
                        {
                            "capacity": capacity,
                            "seed": seed,
                            "contrast": label,
                            "nll_difference": a["validation_nll"] - b["validation_nll"],
                            "left_run": a["run_id"],
                            "right_run": b["run_id"],
                        }
                    )
    scores["status"] = "COMPLETED_EXPLORATORY_PHYSICS_CALIBRATED" if len(scores["runs"]) == 40 and not scores["failed_runs"] else "COMPLETED_WITH_FAILURES_PRESERVED"
    _save(scores_path, scores)
    model_manifest["status"] = scores["status"]
    _save(manifest_path, model_manifest)
    budget = {
        "status": "MEASURED_EXPLORATORY_PHYSICS_CALIBRATED",
        "runs_completed": len(scores["runs"]),
        "runs_failed": len(scores["failed_runs"]),
        "total_hours": (time.perf_counter() - started) / 3600.0,
        "mean_seconds_per_run": float(np.mean([item["elapsed_seconds"] for item in scores["runs"]])) if scores["runs"] else None,
        "new_geant4": False,
        "ood": False,
    }
    _save(args.output / "exploratory_compute_budget.json", budget)
    print(json.dumps({"status": scores["status"], "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "output": str(args.output)}, sort_keys=True))
    if scores["status"] != "COMPLETED_EXPLORATORY_PHYSICS_CALIBRATED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
