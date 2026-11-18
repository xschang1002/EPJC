"""Run a symmetric conditional-scale compensation probe for Raw and Physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from representation_flow.engine import fit_preprocessors, load_source_arrays, sha256_file, subset_indices, RepresentationPreprocessor
from representation_flow.run_a0_prototype import _train_one, prepare_new_output


SEEDS = (3101, 3102, 3103, 3104, 3105)
CAPACITY_WIDTHS = {"Small": 32}
FRACTIONS = {"f010": (0.1, 4101), "f100": (1.0, 4102)}
REPRESENTATIONS = ("RawScaled", "PhysicsScaled")


def _uid_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _add_symmetric_scale(preprocessors, arrays, train_indices):
    p = arrays["p_in"][train_indices]
    t = arrays["thickness"][train_indices]
    q = arrays["charge"][train_indices]
    y = arrays["y"][train_indices]
    features = np.column_stack([np.ones_like(p), (p - 0.65) / 0.35, (t - 0.3) / 0.7, q])
    epsilon = 1e-6
    result = dict(preprocessors)
    for base_name, scaled_name in (("Raw", "RawScaled"), ("Physics", "PhysicsScaled")):
        base = preprocessors[base_name]
        residual = y - base.center(p, t, q) - base.mu
        target = 0.5 * np.log(residual * residual + epsilon * epsilon)
        penalty = np.diag(np.asarray([0.0, 0.1, 0.1, 0.1], dtype=np.float64))
        beta = np.linalg.solve(features.T @ features + penalty, features.T @ target)
        log_min = float(np.log(0.25 * base.scale))
        log_max = float(np.log(4.0 * base.scale))
        scaled = RepresentationPreprocessor(scaled_name, np.asarray(base.coefficients), base.mu, base.scale, base.alpha, np.asarray(beta), log_min, log_max)
        result[scaled_name] = scaled
    return result, epsilon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/scale_compensation"))
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    prepare_new_output(args.output)
    arrays, manifest = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_pool = np.flatnonzero(arrays["split"] == "train")
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    code_hashes = {"engine.py": sha256_file(Path(__file__).with_name("engine.py")), "run_a0_prototype.py": sha256_file(Path(__file__).with_name("run_a0_prototype.py")), "run_scale_compensation.py": sha256_file(Path(__file__))}
    plan = {
        "status": "EXPLORATORY_SCALE_COMPENSATION_AUTHORIZED",
        "question": "Does a symmetric train-only conditional scale function explain the hidden-condition Raw/Physics gap?",
        "representations": list(REPRESENTATIONS),
        "unscaled_reference": "reports/representation_v2/exploratory/context_interaction_long",
        "definition": "For Raw and Physics separately, fit a ridge log s(V,W)=linear_features(V,W) on 0.5*log((Y-center-mu)^2+epsilon^2), epsilon=1e-6; clip s to [0.25,4] times the train target scale, freeze s, and use the same original-unit Jacobian and support normalization.",
        "conditioner": "thickness_charge",
        "capacities": CAPACITY_WIDTHS,
        "fractions": {tag: {"fraction": fraction, "catalog_id": catalog} for tag, (fraction, catalog) in FRACTIONS.items()},
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
        "train_pool_events": int(len(train_pool)),
        "validation_events": int(len(validation_indices)),
        "train_pool_uid_sha256": _uid_digest(arrays["event_uid"][train_pool]),
        "code_hashes": code_hashes,
    }
    _save(args.output / "exploratory_plan.json", plan)
    scores = {"status": "RUNNING", "stage": "scale_compensation", "runs": [], "failed_runs": [], "paired_comparisons": [], "plan": "exploratory_plan.json"}
    model_manifest = {"status": "RUNNING", "stage": "scale_compensation", "code_hashes": code_hashes, "models": [], "centres": {}}
    started = time.perf_counter()
    q99 = float(np.quantile(arrays["y"][train_pool] / arrays["p_in"][train_pool], 0.99, method="linear"))
    for fraction_tag, (fraction, catalog_id) in FRACTIONS.items():
        train_indices = subset_indices(arrays, fraction=fraction, catalog_id=catalog_id)
        base_preprocessors = fit_preprocessors(arrays, train_indices)
        preprocessors, epsilon = _add_symmetric_scale(base_preprocessors, arrays, train_indices)
        for name in REPRESENTATIONS:
            pre = preprocessors[name]
            model_manifest["centres"][f"{fraction_tag}__{name}"] = {"mu": pre.mu, "scale": pre.scale, "scale_coefficients": None if pre.scale_coefficients is None else [float(x) for x in pre.scale_coefficients], "epsilon": epsilon, "fit_labels": f"{fraction_tag} source train subset only"}
        for representation in REPRESENTATIONS:
            for capacity in CAPACITY_WIDTHS:
                for seed in SEEDS:
                    run_args = SimpleNamespace(
                        run_prefix="explore_scale_compensation",
                        run_id_template="explore__scale_compensation__{representation}__{capacity}__{fraction_tag}__sub{subset_catalog_id}__seed{seed}",
                        fraction_tag=fraction_tag, subset_catalog_id=catalog_id, sample_fraction=fraction,
                        run_status="COMPLETED_EXPLORATORY_SCALE_COMPENSATION", feature_mode="thickness_charge",
                        capacity_widths=CAPACITY_WIDTHS, q99_threshold=q99, batch_size=args.batch_size,
                        validation_every=args.validation_every, eval_batch_size=args.eval_batch_size,
                        learning_rate=args.learning_rate, max_updates=args.max_updates,
                    )
                    run_id = f"explore__scale_compensation__{representation}__{capacity}__{fraction_tag}__sub{catalog_id}__seed{seed}"
                    try:
                        result, prediction = _train_one(arrays, train_indices, validation_indices, preprocessors[representation], representation, capacity, seed, run_args, args.output)
                        validation_nll = float(np.mean(-prediction["log_p_y"]))
                        scores["runs"].append({"run_id": result["run_id"], "representation": representation, "capacity": capacity, "context_feature_mode": "thickness_charge", "sample_fraction": fraction, "fraction_tag": fraction_tag, "seed": seed, "validation_nll": validation_nll, "selected_update": result["selected_update"], "max_updates": args.max_updates, "epochs_at_budget": result["epochs_at_budget"], "elapsed_seconds": result["elapsed_seconds"], "parameter_count": result["parameter_count"], "predictions": result["predictions"]})
                        model_manifest["models"].append({"run_id": result["run_id"], "representation": representation, "capacity": capacity, "seed": seed, "hidden_width": CAPACITY_WIDTHS[capacity], "context_dimension": 2, "checkpoint": result["checkpoint"], "selected_update": result["selected_update"], "optimizer": "Adam", "learning_rate": args.learning_rate})
                        print(json.dumps({"completed": result["run_id"], "validation_nll": validation_nll, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                    except Exception as exc:
                        failure = {"run_id": run_id, "error_type": type(exc).__name__, "error": str(exc), "representation": representation, "capacity": capacity, "fraction_tag": fraction_tag, "seed": seed}
                        scores["failed_runs"].append(failure)
                        print(json.dumps({"failed": failure}), flush=True)
                    _save(args.output / "exploratory_scores.json", scores)
                    _save(args.output / "exploratory_model_manifest.json", model_manifest)
    by_key = {(item["fraction_tag"], item["representation"], item["capacity"], item["seed"]): item for item in scores["runs"]}
    for fraction_tag in FRACTIONS:
        for seed in SEEDS:
            for left, right, label in (("RawScaled", "PhysicsScaled", "RawScaled_minus_PhysicsScaled"),):
                a = by_key.get((fraction_tag, left, "Small", seed))
                b = by_key.get((fraction_tag, right, "Small", seed))
                if a and b:
                    scores["paired_comparisons"].append({"capacity": "Small", "seed": seed, "contrast": label, "context_feature_mode": "thickness_charge", "sample_fraction": FRACTIONS[fraction_tag][0], "fraction_tag": fraction_tag, "nll_difference": a["validation_nll"] - b["validation_nll"], "left_run": a["run_id"], "right_run": b["run_id"]})
    scores["status"] = "COMPLETED_EXPLORATORY_SCALE_COMPENSATION" if len(scores["runs"]) == 20 and not scores["failed_runs"] else "COMPLETED_WITH_FAILURES_PRESERVED"
    _save(args.output / "exploratory_scores.json", scores)
    model_manifest["status"] = scores["status"]
    _save(args.output / "exploratory_model_manifest.json", model_manifest)
    budget = {"status": "MEASURED_EXPLORATORY_SCALE_COMPENSATION", "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "total_hours": (time.perf_counter() - started) / 3600.0, "mean_seconds_per_run": float(np.mean([item["elapsed_seconds"] for item in scores["runs"]])) if scores["runs"] else None, "new_geant4": False, "ood": False}
    _save(args.output / "exploratory_compute_budget.json", budget)
    print(json.dumps({"status": scores["status"], "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "output": str(args.output)}, sort_keys=True))
    if scores["status"] != "COMPLETED_EXPLORATORY_SCALE_COMPENSATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
