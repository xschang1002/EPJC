"""Run a scoped exploratory context-bottleneck representation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from representation_flow.engine import fit_preprocessors, load_source_arrays, sha256_file, subset_indices
from representation_flow.run_a0_prototype import REPRESENTATIONS, _train_one, prepare_new_output


SEEDS = (3101, 3102, 3103, 3104, 3105)
FRACTION = 0.1
SUBSET_CATALOG_ID = 4101
CAPACITY_WIDTHS = {"Tiny": 8, "Small": 32}


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
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/context_bottleneck"))
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
    preprocessors = fit_preprocessors(arrays, train_indices)
    code_hashes = {"engine.py": sha256_file(Path(__file__).with_name("engine.py")), "run_a0_prototype.py": sha256_file(Path(__file__).with_name("run_a0_prototype.py")), "run_exploratory_matrix.py": sha256_file(Path(__file__))}
    plan = {
        "status": "EXPLORATORY_CONTEXT_BOTTLENECK_AUTHORIZED",
        "question": "Can a representation replace p_in information removed from the conditioner under finite capacity?",
        "context_feature_mode": "thickness_charge",
        "context_definition": "((thickness_mm-0.3)/0.7, charge); p_in is intentionally withheld from the flow conditioner",
        "representations": list(REPRESENTATIONS),
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
        "source_manifest_sha256": sha256_file(args.merged / "manifest.json"),
        "events_file_sha256": sha256_file(args.merged / "events.jsonl"),
        "subset_uid_sha256": _uid_digest(arrays["event_uid"][train_indices]),
        "code_hashes": code_hashes,
    }
    _save(args.output / "exploratory_plan.json", plan)
    scores_path = args.output / "exploratory_scores.json"
    scores = {"status": "RUNNING", "stage": "context_bottleneck", "runs": [], "failed_runs": [], "paired_comparisons": [], "plan": "exploratory_plan.json"}
    model_manifest = {"status": "RUNNING", "stage": "context_bottleneck", "code_hashes": code_hashes, "models": [], "centres": {}}
    for name, pre in preprocessors.items():
        model_manifest["centres"][name] = {"mu": pre.mu, "scale": pre.scale, "alpha": pre.alpha, "coefficient_count": int(len(pre.coefficients)), "fit_labels": "10% source train subset only"}
    _save(args.output / "exploratory_model_manifest.json", model_manifest)
    started = time.perf_counter()
    q99 = float(np.quantile(arrays["y"][train_indices] / arrays["p_in"][train_indices], 0.99, method="linear"))
    for representation in REPRESENTATIONS:
        for capacity in CAPACITY_WIDTHS:
            for seed in SEEDS:
                run_id = f"explore__context_bottleneck__{representation}__{capacity}__f010__sub{SUBSET_CATALOG_ID}__seed{seed}"
                run_args = SimpleNamespace(
                    run_prefix="explore_context_bottleneck",
                    run_id_template="explore__context_bottleneck__{representation}__{capacity}__f010__sub{subset_catalog_id}__seed{seed}",
                    fraction_tag="f010",
                    subset_catalog_id=SUBSET_CATALOG_ID,
                    sample_fraction=FRACTION,
                    run_status="COMPLETED_EXPLORATORY_CONTEXT_BOTTLENECK",
                    feature_mode="thickness_charge",
                    capacity_widths=CAPACITY_WIDTHS,
                    q99_threshold=q99,
                    batch_size=args.batch_size,
                    validation_every=args.validation_every,
                    eval_batch_size=args.eval_batch_size,
                    learning_rate=args.learning_rate,
                    max_updates=args.max_updates,
                )
                try:
                    result, prediction = _train_one(arrays, train_indices, validation_indices, preprocessors[representation], representation, capacity, seed, run_args, args.output)
                    scores["runs"].append({"run_id": result["run_id"], "representation": representation, "capacity": capacity, "hidden_width": CAPACITY_WIDTHS[capacity], "context_feature_mode": "thickness_charge", "sample_fraction": FRACTION, "seed": seed, "validation_nll": float(np.mean(-prediction["log_p_y"])), "selected_update": result["selected_update"], "max_updates": args.max_updates, "epochs_at_budget": result["epochs_at_budget"], "elapsed_seconds": result["elapsed_seconds"], "parameter_count": result["parameter_count"], "predictions": result["predictions"]})
                    model_manifest["models"].append({"run_id": result["run_id"], "representation": representation, "capacity": capacity, "seed": seed, "hidden_width": CAPACITY_WIDTHS[capacity], "context_dimension": 2, "initialization_hash": result["initialization_hash"], "checkpoint": result["checkpoint"], "selected_update": result["selected_update"], "optimizer": "Adam", "learning_rate": args.learning_rate})
                    print(json.dumps({"completed": result["run_id"], "validation_nll": float(np.mean(-prediction["log_p_y"])), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                except Exception as exc:
                    failure = {"run_id": run_id, "error_type": type(exc).__name__, "error": str(exc), "representation": representation, "capacity": capacity, "seed": seed}
                    scores["failed_runs"].append(failure)
                    print(json.dumps({"failed": failure}), flush=True)
                _save(scores_path, scores)
                _save(args.output / "exploratory_model_manifest.json", model_manifest)
    by_key = {(item["representation"], item["capacity"], item["seed"]): item for item in scores["runs"]}
    for capacity in CAPACITY_WIDTHS:
        for seed in SEEDS:
            for left, right, label in (("Raw", "Physics", "Raw_minus_Physics"), ("Learned", "Physics", "Learned_minus_Physics"), ("Simple", "Raw", "Simple_minus_Raw")):
                a = by_key.get((left, capacity, seed))
                b = by_key.get((right, capacity, seed))
                if a and b:
                    scores["paired_comparisons"].append({"capacity": capacity, "seed": seed, "contrast": label, "nll_difference": a["validation_nll"] - b["validation_nll"], "left_run": a["run_id"], "right_run": b["run_id"]})
    scores["status"] = "COMPLETED_EXPLORATORY_CONTEXT_BOTTLENECK" if len(scores["runs"]) == 40 and not scores["failed_runs"] else "COMPLETED_WITH_FAILURES_PRESERVED"
    _save(scores_path, scores)
    model_manifest["status"] = scores["status"]
    _save(args.output / "exploratory_model_manifest.json", model_manifest)
    budget = {"status": "MEASURED_EXPLORATORY_CONTEXT_BOTTLENECK", "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "total_hours": (time.perf_counter() - started) / 3600.0, "mean_seconds_per_run": float(np.mean([item["elapsed_seconds"] for item in scores["runs"]])) if scores["runs"] else None, "new_geant4": False, "ood": False}
    _save(args.output / "exploratory_compute_budget.json", budget)
    print(json.dumps({"status": scores["status"], "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "output": str(args.output)}, sort_keys=True))
    if scores["status"] != "COMPLETED_EXPLORATORY_CONTEXT_BOTTLENECK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
