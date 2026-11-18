"""Execute the frozen 120-cell matched representation study with resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from representation_flow.engine import fit_preprocessors, load_source_arrays, sha256_file, subset_indices
from representation_flow.run_a0_prototype import CAPACITIES, REPRESENTATIONS, _train_one
from representation_flow.experiment_contract import (
    artifact_record, bind_output, canonical_hash, save_json, source_hashes,
    validate_completed, validate_frozen_inputs,
)


SEEDS = (3101, 3102, 3103, 3104, 3105)
FRACTIONS = (0.1, 0.3, 1.0)
SUBSET_CATALOG_ID = 4101


def _uid_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _save(path: Path, value) -> None:
    save_json(path, value)


def _load_or_default(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, default=Path("reports/reproduction/historical_formal_current_code"))
    parser.add_argument("--preflight", type=Path, default=Path("reports/representation_v2/formal_preflight_audit.json"))
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    freeze_path = Path(__file__).with_name("formal_protocol_freeze.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    preflight_path = args.preflight
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    data_hashes = validate_frozen_inputs(freeze, preflight, args, args.merged)
    code_hashes = source_hashes((Path(__file__).resolve(),))
    experiment_identity = bind_output(args.output, {
        "study": "historical_formula_current_engine_reproduction",
        "historical_byte_identical_replay": False,
        "data": data_hashes, "code": code_hashes, "protocol_sha256": canonical_hash(freeze),
        "representations": list(REPRESENTATIONS), "capacities": CAPACITIES,
        "seeds": list(SEEDS), "fractions": list(FRACTIONS), "catalog_id": SUBSET_CATALOG_ID,
        "eligibility": "forward", "eval_batch_size": args.eval_batch_size,
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
    })

    arrays, source_manifest = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    train_pool = np.flatnonzero(arrays["split"] == "train")
    scores_path = args.output / "formal_scores.json"
    scores = _load_or_default(scores_path, {"status": "RUNNING", "stage": "formal_core_120", "runs": [], "failed_runs": [], "paired_comparisons": [], "authorization": "FORMAL_MATRIX_AUTHORIZED_AFTER_PREFLIGHT"})
    expected_ids = {f"core__{rep}__{capacity}__f{int(round(fraction*100)):03d}__sub{SUBSET_CATALOG_ID}__seed{seed}"
                    for fraction in FRACTIONS for rep in REPRESENTATIONS for capacity in CAPACITIES for seed in SEEDS}
    completed_ids = validate_completed(scores.get("runs", []), args.output, experiment_identity, expected_ids)
    model_manifest = _load_or_default(args.output / "formal_model_manifest.json", {"status": "RUNNING", "stage": "formal_core_120", "code_hashes": code_hashes, "models": [], "centres": {}})
    dataset_manifest = _load_or_default(args.output / "formal_dataset_manifest.json", {"status": "RUNNING", "source_directory": str(args.merged), "source_manifest_sha256": sha256_file(args.merged / "manifest.json"), "events_file_sha256": sha256_file(args.merged / "events.jsonl"), "split_counts": {label: int(np.sum(arrays["split"] == label)) for label in ("train", "validation", "diagnostic")}, "fractions": {}})
    for record in (scores, model_manifest, dataset_manifest):
        if record.get("experiment_identity", experiment_identity) != experiment_identity:
            raise ValueError("summary manifest experiment identity mismatch")
        record["experiment_identity"] = experiment_identity
    started = time.perf_counter()

    for fraction in FRACTIONS:
        train_indices = subset_indices(arrays, fraction=fraction, catalog_id=SUBSET_CATALOG_ID)
        fraction_tag = f"f{int(round(fraction * 100)):03d}"
        preprocessors = fit_preprocessors(arrays, train_indices)
        dataset_manifest["fractions"][fraction_tag] = {"fraction": fraction, "train_events": int(len(train_indices)), "uid_sha256": _uid_digest(arrays["event_uid"][train_indices]), "epochs_at_budget": args.max_updates * args.batch_size / len(train_indices)}
        for name, pre in preprocessors.items():
            model_manifest["centres"][f"{fraction_tag}__{name}"] = {"fraction": fraction, "train_events": int(len(train_indices)), "mu": pre.mu, "scale": pre.scale, "alpha": pre.alpha, "coefficient_count": int(len(pre.coefficients)), "fit_labels": "current D_n train subset only"}
        _save(args.output / "formal_dataset_manifest.json", dataset_manifest)
        _save(args.output / "formal_model_manifest.json", model_manifest)
        q99 = float(np.quantile(arrays["y"][train_indices] / arrays["p_in"][train_indices], 0.99, method="linear"))
        for representation in REPRESENTATIONS:
            for capacity in CAPACITIES:
                for seed in SEEDS:
                    run_id = f"core__{representation}__{capacity}__{fraction_tag}__sub{SUBSET_CATALOG_ID}__seed{seed}"
                    if run_id in completed_ids:
                        continue
                    run_args = SimpleNamespace(
                        run_prefix="formal",
                        run_id_template="core__{representation}__{capacity}__{fraction_tag}__sub{subset_catalog_id}__seed{seed}",
                        fraction_tag=fraction_tag,
                        subset_catalog_id=SUBSET_CATALOG_ID,
                        sample_fraction=fraction,
                        run_status="COMPLETED_FORMAL_CORE_120",
                        q99_threshold=q99,
                        batch_size=args.batch_size,
                        validation_every=args.validation_every,
                        eval_batch_size=args.eval_batch_size,
                        learning_rate=args.learning_rate,
                        max_updates=args.max_updates,
                        experiment_identity=experiment_identity,
                    )
                    try:
                        result, prediction = _train_one(arrays, train_indices, validation_indices, preprocessors[representation], representation, capacity, seed, run_args, args.output)
                        result["source_manifest_sha256"] = dataset_manifest["source_manifest_sha256"]
                        result["events_file_sha256"] = dataset_manifest["events_file_sha256"]
                        result["target_unit"] = "GeV/c"
                        result["checkpoint_rule"] = freeze["optimization"]["checkpoint_rule"]
                        result["gradient_record"] = freeze["optimization"]["gradient_record"]
                        scores["runs"].append({"run_id": result["run_id"], "representation": representation, "capacity": capacity, "sample_fraction": fraction, "fraction_tag": fraction_tag, "seed": seed, "train_events": int(len(train_indices)), "validation_events": int(len(validation_indices)), "validation_nll": float(np.mean(-prediction["log_p_y"])), "validation_log_sf_finite_fraction": float(np.mean(np.isfinite(prediction["log_sf_y"]))), "selected_update": result["selected_update"], "max_updates": args.max_updates, "epochs_at_budget": result["epochs_at_budget"], "elapsed_seconds": result["elapsed_seconds"], "parameter_count": result["parameter_count"], "predictions": result["predictions"]})
                        scores["runs"][-1].update(experiment_identity=experiment_identity, artifacts={
                            "checkpoint": artifact_record(Path(result["checkpoint"]), args.output),
                            "predictions": artifact_record(Path(result["predictions"]), args.output),
                            "run_manifest": artifact_record(args.output / "runs" / run_id / "run_manifest.json", args.output),
                        })
                        model_manifest["models"].append({"run_id": result["run_id"], "representation": representation, "capacity": capacity, "sample_fraction": fraction, "seed": seed, "parameter_count": result["parameter_count"], "initialization_hash": result["initialization_hash"], "code_hashes": code_hashes, "optimizer": "Adam", "optimizer_seed": seed, "learning_rate": args.learning_rate, "optimizer_state_present": True, "checkpoint": result["checkpoint"], "selected_update": result["selected_update"]})
                        completed_ids.add(run_id)
                        print(json.dumps({"completed": result["run_id"], "validation_nll": float(np.mean(-prediction["log_p_y"])), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                    except Exception as exc:
                        failure = {"run_id": run_id, "error_type": type(exc).__name__, "error": str(exc), "representation": representation, "capacity": capacity, "sample_fraction": fraction, "seed": seed}
                        scores.setdefault("failed_runs", []).append(failure)
                        print(json.dumps({"failed": failure}), flush=True)
                    _save(scores_path, scores)
                    _save(args.output / "formal_model_manifest.json", model_manifest)

    by_key = {(item["representation"], item["capacity"], item["sample_fraction"], item["seed"]): item for item in scores["runs"]}
    scores["paired_comparisons"] = []
    for fraction in FRACTIONS:
        for capacity in CAPACITIES:
            for seed in SEEDS:
                for left, right, label in (("Raw", "Physics", "Raw_minus_Physics"), ("Learned", "Physics", "Learned_minus_Physics"), ("Simple", "Raw", "Simple_minus_Raw")):
                    a = by_key.get((left, capacity, fraction, seed))
                    b = by_key.get((right, capacity, fraction, seed))
                    if a and b:
                        scores["paired_comparisons"].append({"sample_fraction": fraction, "capacity": capacity, "seed": seed, "contrast": label, "nll_difference": a["validation_nll"] - b["validation_nll"], "left_run": a["run_id"], "right_run": b["run_id"]})
    scores["status"] = "COMPLETED_FORMAL_CORE_120" if completed_ids == expected_ids and not scores["failed_runs"] else ("COMPLETED_WITH_FAILURES_PRESERVED" if scores["failed_runs"] else "RUNNING")
    scores["authorization"] = "FORMAL_MATRIX_AUTHORIZED_AFTER_PREFLIGHT"
    scores["formal_protocol"] = "formal_protocol_freeze.json"
    _save(scores_path, scores)
    model_manifest["status"] = scores["status"]
    _save(args.output / "formal_model_manifest.json", model_manifest)
    dataset_manifest["status"] = scores["status"]
    _save(args.output / "formal_dataset_manifest.json", dataset_manifest)
    budget = {"status": "MEASURED_FORMAL_CORE_120", "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "total_hours": (time.perf_counter() - started) / 3600.0, "max_updates": args.max_updates, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "epochs_by_fraction": {tag: value["epochs_at_budget"] for tag, value in dataset_manifest["fractions"].items()}, "seconds_per_run": {item["run_id"]: item["elapsed_seconds"] for item in scores["runs"]}, "new_geant4": False, "ood": False, "spline_bound_extension": False}
    _save(args.output / "formal_compute_budget.json", budget)
    print(json.dumps({"status": scores["status"], "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]), "output": str(args.output)}, sort_keys=True))
    if scores["status"] != "COMPLETED_FORMAL_CORE_120":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
