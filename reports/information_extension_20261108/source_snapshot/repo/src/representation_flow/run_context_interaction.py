"""Run a paired conditioner-information interaction probe.

The probe compares Raw and Physics under the same flow, seed, capacity, and
sample fraction while toggling whether p_in is available to the conditioner.
"""

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
REPRESENTATIONS = ("Raw", "Physics")
CAPACITY_WIDTHS = {"Small": 32}
CONTEXTS = {"full": "full", "bottleneck": "thickness_charge"}
# New runs use the same UID ordering as Full. Historical f100 artifacts used
# 4102 and remain unchanged; do not combine them with new paired runs.
FRACTIONS = {"f010": (0.1, 4101), "f100": (1.0, 4101)}


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
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/context_interaction"))
    parser.add_argument("--max-updates", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--context-labels", nargs="+", choices=tuple(CONTEXTS), default=list(CONTEXTS))
    parser.add_argument("--fraction-tags", nargs="+", choices=tuple(FRACTIONS), default=list(FRACTIONS))
    args = parser.parse_args()
    prepare_new_output(args.output)
    selected_contexts = {label: CONTEXTS[label] for label in args.context_labels}
    selected_fractions = {tag: FRACTIONS[tag] for tag in args.fraction_tags}

    arrays, manifest = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    train_pool = np.flatnonzero(arrays["split"] == "train")
    code_hashes = {
        "engine.py": sha256_file(Path(__file__).with_name("engine.py")),
        "run_a0_prototype.py": sha256_file(Path(__file__).with_name("run_a0_prototype.py")),
        "run_context_interaction.py": sha256_file(Path(__file__)),
    }
    plan = {
        "status": "EXPLORATORY_CONTEXT_INTERACTION_AUTHORIZED",
        "question": "Does the Physics representation help differently when p_in is available versus withheld from the conditioner?",
        "representations": list(REPRESENTATIONS),
        "capacities": CAPACITY_WIDTHS,
        "contexts": selected_contexts,
        "fractions": {tag: {"fraction": fraction, "catalog_id": catalog} for tag, (fraction, catalog) in selected_fractions.items()},
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
    scores_path = args.output / "exploratory_scores.json"
    scores = {"status": "RUNNING", "stage": "context_interaction", "runs": [], "failed_runs": [], "paired_comparisons": [], "plan": "exploratory_plan.json"}
    model_manifest = {"status": "RUNNING", "stage": "context_interaction", "code_hashes": code_hashes, "models": [], "centres": {}}
    _save(args.output / "exploratory_model_manifest.json", model_manifest)

    started = time.perf_counter()
    q99 = float(np.quantile(arrays["y"][train_pool] / arrays["p_in"][train_pool], 0.99, method="linear"))
    for fraction_tag, (fraction, catalog_id) in selected_fractions.items():
        train_indices = subset_indices(arrays, fraction=fraction, catalog_id=catalog_id)
        preprocessors_subset = fit_preprocessors(arrays, train_indices)
        for name in REPRESENTATIONS:
            pre = preprocessors_subset[name]
            model_manifest["centres"][f"{fraction_tag}__{name}"] = {
                "mu": pre.mu, "scale": pre.scale, "alpha": pre.alpha,
                "coefficient_count": int(len(pre.coefficients)), "fit_labels": "current D_n train subset only",
                "train_events": len(train_indices), "ordered_uid_sha256": _uid_digest(arrays["event_uid"][train_indices]),
                "preprocessor": pre.to_dict(),
            }
        _save(args.output / "exploratory_model_manifest.json", model_manifest)
        for context_label, feature_mode in selected_contexts.items():
            for representation in REPRESENTATIONS:
                for capacity in CAPACITY_WIDTHS:
                    for seed in SEEDS:
                        run_id = f"explore__context_interaction__{context_label}__{representation}__{capacity}__{fraction_tag}__sub{catalog_id}__seed{seed}"
                        try:
                            run_args = SimpleNamespace(
                                run_prefix="explore_context_interaction",
                                run_id_template="explore__context_interaction__{context_label}__{representation}__{capacity}__{fraction_tag}__sub{subset_catalog_id}__seed{seed}",
                                context_label=context_label,
                                fraction_tag=fraction_tag,
                                subset_catalog_id=catalog_id,
                                sample_fraction=fraction,
                                run_status="COMPLETED_EXPLORATORY_CONTEXT_INTERACTION",
                                feature_mode=feature_mode,
                                capacity_widths=CAPACITY_WIDTHS,
                                q99_threshold=q99,
                                batch_size=args.batch_size,
                                validation_every=args.validation_every,
                                eval_batch_size=args.eval_batch_size,
                                learning_rate=args.learning_rate,
                                max_updates=args.max_updates,
                            )
                            result, prediction = _train_one(arrays, train_indices, validation_indices, preprocessors_subset[representation], representation, capacity, seed, run_args, args.output)
                            validation_nll = float(np.mean(-prediction["log_p_y"]))
                            scores["runs"].append({
                                "run_id": result["run_id"], "representation": representation, "capacity": capacity,
                                "hidden_width": CAPACITY_WIDTHS[capacity], "context_feature_mode": feature_mode,
                                "context_label": context_label, "sample_fraction": fraction, "fraction_tag": fraction_tag,
                                "subset_catalog_id": catalog_id, "seed": seed, "validation_nll": validation_nll,
                                "selected_update": result["selected_update"], "max_updates": args.max_updates,
                                "epochs_at_budget": result["epochs_at_budget"], "elapsed_seconds": result["elapsed_seconds"],
                                "parameter_count": result["parameter_count"], "predictions": result["predictions"],
                            })
                            model_manifest["models"].append({
                                "run_id": result["run_id"], "representation": representation, "capacity": capacity, "seed": seed,
                                "hidden_width": CAPACITY_WIDTHS[capacity], "context_dimension": 5 if feature_mode == "full" else 2,
                                "context_label": context_label, "fraction_tag": fraction_tag, "initialization_hash": result["initialization_hash"],
                                "checkpoint": result["checkpoint"], "selected_update": result["selected_update"], "optimizer": "Adam",
                                "learning_rate": args.learning_rate,
                            })
                            print(json.dumps({"completed": result["run_id"], "validation_nll": validation_nll, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                        except Exception as exc:
                            failure = {"run_id": run_id, "error_type": type(exc).__name__, "error": str(exc), "context_label": context_label, "representation": representation, "capacity": capacity, "fraction_tag": fraction_tag, "seed": seed}
                            scores["failed_runs"].append(failure)
                            print(json.dumps({"failed": failure}), flush=True)
                        _save(scores_path, scores)
                        _save(args.output / "exploratory_model_manifest.json", model_manifest)

    by_key = {(item["context_label"], item["fraction_tag"], item["representation"], item["capacity"], item["seed"]): item for item in scores["runs"]}
    for context_label in selected_contexts:
        for fraction_tag in selected_fractions:
            for capacity in CAPACITY_WIDTHS:
                for seed in SEEDS:
                    left = by_key.get((context_label, fraction_tag, "Raw", capacity, seed))
                    right = by_key.get((context_label, fraction_tag, "Physics", capacity, seed))
                    if left and right:
                        scores["paired_comparisons"].append({
                            "capacity": capacity, "seed": seed, "contrast": "Raw_minus_Physics",
                            "context_feature_mode": CONTEXTS[context_label], "context_label": context_label,
                            "sample_fraction": selected_fractions[fraction_tag][0], "fraction_tag": fraction_tag,
                            "nll_difference": left["validation_nll"] - right["validation_nll"],
                            "left_run": left["run_id"], "right_run": right["run_id"],
                        })
    expected_runs = len(selected_contexts) * len(selected_fractions) * len(REPRESENTATIONS) * len(CAPACITY_WIDTHS) * len(SEEDS)
    scores["status"] = "COMPLETED_EXPLORATORY_CONTEXT_INTERACTION" if len(scores["runs"]) == expected_runs and not scores["failed_runs"] else "COMPLETED_WITH_FAILURES_PRESERVED"
    _save(scores_path, scores)
    model_manifest["status"] = scores["status"]
    _save(args.output / "exploratory_model_manifest.json", model_manifest)
    budget = {"status": "MEASURED_EXPLORATORY_CONTEXT_INTERACTION", "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "total_hours": (time.perf_counter() - started) / 3600.0, "mean_seconds_per_run": float(np.mean([item["elapsed_seconds"] for item in scores["runs"]])) if scores["runs"] else None, "new_geant4": False, "ood": False}
    _save(args.output / "exploratory_compute_budget.json", budget)
    print(json.dumps({"status": scores["status"], "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "output": str(args.output)}, sort_keys=True))
    if scores["status"] != "COMPLETED_EXPLORATORY_CONTEXT_INTERACTION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
