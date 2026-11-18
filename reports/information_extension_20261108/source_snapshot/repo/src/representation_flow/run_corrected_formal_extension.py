"""Benchmark and complete the corrected 120-cell Full matrix without refitting 20 shared cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import RepresentationPreprocessor, fit_preprocessors, subset_indices
from representation_flow.experiment_contract import (
    artifact_record, bind_output, canonical_hash, save_json, sha256, source_hashes, validate_completed,
)
from representation_flow.run_a0_prototype import _train_one
from representation_flow.run_correction_study import prepare_cache

_ARRAYS = None
_INDEX_ROOT = None


def initialize_worker(base_cache: str, index_root: str, threads: int):
    global _ARRAYS, _INDEX_ROOT
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _ARRAYS = load_arrays(Path(base_cache))
    _INDEX_ROOT = Path(index_root)


def load_arrays(cache: Path):
    return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
            for p in cache.glob("*.npy") if not p.stem.startswith("indices_")}


def cell_id(rep, capacity, fraction, seed, catalog):
    return f"corrected_full__{rep}__{capacity}__f{round(fraction * 100):03d}__sub{catalog}__seed{seed}"


def reused_cell(rep, capacity, fraction):
    return capacity == "Small" and rep in ("Raw", "CorrectedPhysics") and fraction in (0.1, 1.0)


def prepare_extension_cache(base_cache: Path, index_root: Path, protocol: dict, base_record: dict):
    identity = {"base_cache_sha256": canonical_hash(base_record), "protocol_sha256": canonical_hash(protocol),
                "source_hashes": source_hashes((Path(__file__).resolve(),))}
    manifest_path = index_root / "preprocessing_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("extension preprocessing cache identity changed")
        for record in saved["files"]:
            if sha256(index_root / record["name"]) != record["sha256"]:
                raise ValueError("extension index checksum mismatch")
        return saved
    if index_root.exists() and any(index_root.iterdir()):
        raise ValueError("partial extension preprocessing cache must be inspected")
    arrays = load_arrays(base_cache)
    index_root.mkdir(parents=True, exist_ok=True)
    fractions = {}
    for fraction in protocol["sample_fractions"]:
        tag = f"f{round(fraction * 100):03d}"
        indices = subset_indices(arrays, fraction, protocol["subset_catalog_id"])
        np.save(index_root / f"indices_{tag}.npy", indices, allow_pickle=False)
        pre = fit_preprocessors(arrays, indices, physics_name="CorrectedPhysics")
        fractions[tag] = {
            "fraction": fraction, "train_events": len(indices),
            "ordered_uid_sha256": canonical_hash(arrays["event_uid"][indices].tolist()),
            "q99_threshold": float(np.quantile(arrays["y"][indices] / arrays["p_in"][indices], 0.99, method="linear")),
            "preprocessors": {name: value.to_dict() for name, value in pre.items()},
        }
    valid = np.flatnonzero(arrays["split"] == "validation")
    np.save(index_root / "indices_validation.npy", valid, allow_pickle=False)
    record = {"identity": identity, "fractions": fractions, "validation_events": len(valid),
              "validation_uid_sha256": canonical_hash(arrays["event_uid"][valid].tolist()),
              "files": [{"name": p.name, "sha256": sha256(p)} for p in sorted(index_root.glob("*.npy"))]}
    save_json(manifest_path, record)
    return record


def build_jobs(protocol, preprocessing, output, identity, updates, deadline, protocol_path):
    jobs = []
    opt = protocol["optimization"]
    for fraction, rep, capacity, seed in itertools.product(protocol["sample_fractions"], protocol["representations"], protocol["capacities"], protocol["seeds"]):
        if reused_cell(rep, capacity, fraction):
            continue
        tag = f"f{round(fraction * 100):03d}"
        fraction_record = preprocessing["fractions"][tag]
        run_id = cell_id(rep, capacity, fraction, seed, protocol["subset_catalog_id"])
        jobs.append({"run_id": run_id, "representation": rep, "capacity": capacity, "fraction": fraction,
                     "fraction_tag": tag, "seed": seed, "output": str(output.resolve()), "identity": identity,
                     "preprocessor": fraction_record["preprocessors"][rep],
                     "train_event_uid_sha256": fraction_record["ordered_uid_sha256"],
                     "validation_event_uid_sha256": preprocessing["validation_uid_sha256"],
                     "args": {"run_prefix": "corrected_formal", "run_id_template": run_id,
                              "run_status": "COMPLETED_CORRECTED_FORMAL_CELL", "feature_mode": "full", "context_label": "Full",
                              "capacity_widths": protocol["capacities"], "sample_fraction": fraction,
                              "fraction_tag": tag, "subset_catalog_id": protocol["subset_catalog_id"],
                              "q99_threshold": fraction_record["q99_threshold"], "max_updates": updates,
                              "batch_size": opt["batch_size"], "validation_every": opt["validation_every"],
                              "eval_batch_size": opt["eval_batch_size"], "learning_rate": opt["learning_rate"],
                              "torch_threads": opt["torch_threads_per_worker"], "eligibility": protocol["eligibility"],
                              "experiment_identity": identity, "deadline_unix": deadline,
                              "checkpoint_source_paths": {"run_corrected_formal_extension.py": str(Path(__file__).resolve()),
                                                          "extension_protocol.json": str(protocol_path.resolve())}}})
    return jobs


def run_cell(job):
    indices = np.load(_INDEX_ROOT / f"indices_{job['fraction_tag']}.npy", allow_pickle=False)
    valid = np.load(_INDEX_ROOT / "indices_validation.npy", allow_pickle=False)
    pre = RepresentationPreprocessor.from_dict(job["preprocessor"])
    result, prediction = _train_one(_ARRAYS, indices, valid, pre, job["representation"], job["capacity"], job["seed"],
                                    SimpleNamespace(**job["args"]), Path(job["output"]))
    output = Path(job["output"])
    return {**result, "sample_fraction": job["fraction"], "fraction_tag": job["fraction_tag"], "context_label": "Full",
            "matrix_cell_id": job["run_id"], "reused": False,
            "validation_nll": float(np.mean(-prediction["log_p_y"], dtype=np.float64)),
            "train_event_uid_sha256": job["train_event_uid_sha256"], "validation_event_uid_sha256": job["validation_event_uid_sha256"],
            "experiment_identity": job["identity"],
            "artifacts": {"checkpoint": artifact_record(Path(result["checkpoint"]), output),
                          "predictions": artifact_record(Path(result["predictions"]), output),
                          "run_manifest": artifact_record(output / "runs" / result["run_id"] / "run_manifest.json", output)}}


def validate_reused_rows(core_root, protocol, preprocessing, core_common):
    study = core_root / "study"
    identity_record = json.loads((study / "experiment_identity.json").read_text(encoding="utf-8"))
    expected_identity = {**core_common, "mode": "execute", "max_updates": protocol["optimization"]["max_updates"]}
    if identity_record["identity"] != expected_identity or identity_record["sha256"] != canonical_hash(expected_identity):
        raise ValueError("completed core study code/data/environment differs from extension")
    scores = json.loads((study / "scores.json").read_text(encoding="utf-8"))
    core_ids = {f"correction__{context}__{rep}__Small__f{round(fraction*100):03d}__sub{protocol['subset_catalog_id']}__seed{seed}"
                for context, rep, fraction, seed in itertools.product(("Full", "Hidden"), ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics"), (0.1, 1.0), protocol["seeds"])}
    done = validate_completed(scores["runs"], study, identity_record["sha256"], core_ids)
    if scores["status"] != "COMPLETE" or done != core_ids:
        raise ValueError("the complete 60-cell core must be frozen before formal extension execution")
    reused = []
    for row in scores["runs"]:
        if row["context_label"] != "Full" or not reused_cell(row["representation"], row["capacity"], row["sample_fraction"]):
            continue
        fraction = preprocessing["fractions"][row["fraction_tag"]]
        payload, pre = load_checkpoint(study / row["artifacts"]["checkpoint"]["path"], expected_sha256=row["artifacts"]["checkpoint"]["sha256"])
        if pre.to_dict() != fraction["preprocessors"][row["representation"]]:
            raise ValueError("shared cell preprocessing differs")
        if row["train_event_uid_sha256"] != fraction["ordered_uid_sha256"] or row["validation_event_uid_sha256"] != preprocessing["validation_uid_sha256"]:
            raise ValueError("shared cell sample ordering differs")
        opt = protocol["optimization"]
        if any(row[key] != opt[key] for key in ("max_updates", "batch_size", "validation_every", "eval_batch_size", "learning_rate")) or payload["model_config"]["hidden"] != protocol["capacities"]["Small"]:
            raise ValueError("shared cell training configuration differs")
        reused.append({**row, "reused": True, "artifact_root": str(study.resolve()),
                       "matrix_cell_id": cell_id(row["representation"], row["capacity"], row["sample_fraction"], row["seed"], protocol["subset_catalog_id"])})
    if len(reused) != 20:
        raise ValueError("expected exactly twenty reusable Full/Small cells")
    return reused, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "execute"))
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/corrected_formal_extension_20261107.json"))
    parser.add_argument("--core-protocol", type=Path, default=Path("protocol/correction_v2_20261107.json"))
    parser.add_argument("--core-root", type=Path, default=Path("reports/correction_v2_20261107"))
    parser.add_argument("--root", type=Path, default=Path("reports/corrected_formal_extension_20261107"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    started = time.time()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    core_protocol = json.loads(args.core_protocol.read_text(encoding="utf-8"))
    if canonical_hash(core_protocol) != protocol["core_protocol_sha256"]:
        raise ValueError("core protocol changed since extension prespecification")
    for key in ("main_data", "eligibility", "split", "subset_catalog_id", "seeds", "optimization"):
        if protocol[key] != core_protocol[key]:
            raise ValueError(f"extension must use the same core {key}")
    core_benchmark = json.loads((args.core_root / "benchmark/benchmark_result.json").read_text(encoding="utf-8"))
    base_cache = args.core_root / "cache"
    base_record = prepare_cache(args.merged, base_cache, core_protocol)
    core_code = source_hashes((Path(__file__).with_name("run_correction_study.py").resolve(),))
    environment = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                   "torch_threads": protocol["optimization"]["torch_threads_per_worker"], "workers": args.workers}
    core_common = {"protocol_sha256": canonical_hash(core_protocol), "code": core_code, "data": core_protocol["main_data"],
                   "cache_sha256": canonical_hash(base_record), "environment": environment}
    if core_benchmark["common_identity"] != core_common:
        raise ValueError("extension does not match the core benchmark source/data/environment")
    index_root = args.root / "cache"
    preprocessing = prepare_extension_cache(base_cache, index_root, protocol, base_record)
    common = {"core_common": core_common, "protocol_sha256": canonical_hash(protocol),
              "extension_source_sha256": sha256(Path(__file__)), "preprocessing_sha256": canonical_hash(preprocessing)}
    benchmark_path = args.root / "benchmark/benchmark_result.json"
    output = args.root / ("benchmark" if args.mode == "benchmark" else "study")
    updates = protocol["runtime_policy"]["benchmark_updates"] if args.mode == "benchmark" else protocol["optimization"]["max_updates"]
    reused, core_scores = [], None
    if args.mode == "execute":
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark["common_identity"] != common or not benchmark["allowed_combined_under_three_hours"]:
            raise ValueError("no matching measured sub-three-hour combined budget")
        reused, core_scores = validate_reused_rows(args.core_root, protocol, preprocessing, core_common)
        deadline = core_scores["execution_started_unix"] + protocol["runtime_policy"]["max_total_seconds"] - core_benchmark["benchmark_wall_seconds"] - benchmark["benchmark_wall_seconds"] - protocol["runtime_policy"]["reserved_analysis_seconds"] - protocol["runtime_policy"]["prior_core_benchmark_seconds"] - protocol["runtime_policy"]["prior_extension_benchmark_seconds"]
    else:
        deadline = started + 900
    if time.time() >= deadline:
        raise TimeoutError("combined study deadline already expired")
    identity = bind_output(output, {**common, "mode": args.mode, "max_updates": updates})
    jobs = build_jobs(protocol, preprocessing, output, identity, updates, deadline, args.protocol)
    if len(jobs) != 100:
        raise ValueError("extension must add exactly one hundred cells")
    if args.mode == "benchmark":
        jobs = [j for j in jobs if j["capacity"] == "Large" and j["fraction"] == 1.0 and j["seed"] == protocol["seeds"][0]]
    expected = {j["run_id"] for j in jobs}
    score_path = output / "scores.json"
    scores = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {"status": "RUNNING", "experiment_identity": identity, "runs": [], "failures": [], "execution_started_unix": started}
    if scores["experiment_identity"] != identity:
        raise ValueError("extension scores identity mismatch")
    completed = validate_completed(scores["runs"], output, identity, expected)
    if completed == expected and scores["status"] == "COMPLETE":
        print(json.dumps({"status": "ALREADY_COMPLETE_VERIFIED", "cells": len(completed)}), flush=True)
        return
    save_json(score_path, scores)
    train_started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(base_cache.resolve()), str(index_root.resolve()), environment["torch_threads"])) as executor:
        futures = {executor.submit(run_cell, job): job for job in jobs if job["run_id"] not in completed}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                scores["runs"].append(result)
                # Benchmark logs intentionally contain no scientific score comparison.
                print(json.dumps({"completed": result["run_id"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
            except Exception as exc:
                scores["failures"].append({"run_id": job["run_id"], "type": type(exc).__name__, "error": str(exc)})
                for other in futures:
                    other.cancel()
            save_json(score_path, scores)
    batch_seconds = time.time() - train_started
    done = validate_completed(scores["runs"], output, identity, expected)
    scores.update(status="COMPLETE" if done == expected and not scores["failures"] else "INCOMPLETE",
                  wall_seconds_this_invocation=time.time() - started,
                  wall_seconds_since_first_start=time.time() - scores["execution_started_unix"])
    save_json(score_path, scores)
    if scores["status"] != "COMPLETE":
        raise RuntimeError("formal extension incomplete; failures remain visible")
    if args.mode == "benchmark":
        large_ratio = protocol["optimization"]["max_updates"] / updates
        small_ratio = protocol["optimization"]["max_updates"] / core_protocol["runtime_policy"]["benchmark_updates"]
        large = max(row["elapsed_seconds"] for row in scores["runs"])
        small = core_benchmark["maximum_cell_seconds"]
        small_overhead = max(0., core_benchmark["parallel_batch_seconds"] - small)
        large_overhead = max(0., batch_seconds - large)
        extension_forecast = (small * small_ratio + small_overhead) * math.ceil(40 / args.workers) + (large * large_ratio + large_overhead) * math.ceil(60 / args.workers)
        combined_training = core_benchmark["forecast_training_seconds"] + extension_forecast
        prior_probe_seconds = protocol["runtime_policy"]["prior_core_benchmark_seconds"] + protocol["runtime_policy"]["prior_extension_benchmark_seconds"]
        conservative = combined_training * protocol["runtime_policy"]["forecast_safety_factor"] + protocol["runtime_policy"]["reserved_analysis_seconds"] + core_benchmark["benchmark_wall_seconds"] + (time.time() - started) + prior_probe_seconds + 90
        result = {"common_identity": common, "benchmark_wall_seconds": time.time() - started, "parallel_batch_seconds": batch_seconds,
                  "maximum_large_cell_seconds": large, "maximum_small_cell_seconds": small,
                  "large_benchmark_updates": updates, "small_benchmark_updates": core_protocol["runtime_policy"]["benchmark_updates"],
                  "prior_performance_probe_seconds": prior_probe_seconds,
                  "extension_small_cells": 40, "extension_large_cells": 60, "reused_cells": 20,
                  "forecast_extension_training_seconds": extension_forecast, "forecast_combined_training_seconds": combined_training,
                  "conservative_combined_total_seconds": conservative,
                  "allowed_combined_under_three_hours": conservative < protocol["runtime_policy"]["max_total_seconds"],
                  "benchmark_scores_are_not_scientific_results": True}
        save_json(benchmark_path, result)
        print(json.dumps(result), flush=True)
    else:
        all_rows = reused + [{**row, "artifact_root": str(output.resolve())} for row in scores["runs"]]
        expected_matrix = {cell_id(rep, capacity, fraction, seed, protocol["subset_catalog_id"])
                           for rep, capacity, fraction, seed in itertools.product(protocol["representations"], protocol["capacities"], protocol["sample_fractions"], protocol["seeds"])}
        if len(all_rows) != 120 or {row["matrix_cell_id"] for row in all_rows} != expected_matrix:
            raise ValueError("combined corrected formal matrix is not complete and unique")
        save_json(output / "formal_scores_120.json", {"status": "COMPLETE_CORRECTED_FORMAL_120", "new_cells": 100, "reused_cells": 20,
                  "experiment_identity": identity, "runs": all_rows, "confirmation_accessed": False})
        save_json(output / "FROZEN_CHECKPOINT_MANIFEST.json", {"schema": "corrected-formal-checkpoint-freeze-v1", "experiment_identity": identity,
                  "frozen_before_confirmation": True, "models": [{key: row[key] for key in ("matrix_cell_id", "run_id", "representation", "capacity", "sample_fraction", "seed", "experiment_identity", "reused", "artifact_root", "artifacts")} for row in all_rows]})
        print(json.dumps({"status": "COMPLETE_CORRECTED_FORMAL_120", "new_cells": 100, "reused_cells": 20}), flush=True)


if __name__ == "__main__":
    main()
