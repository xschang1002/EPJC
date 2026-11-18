"""Self-contained, version-bound 60-cell center correction study with a time gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from representation_flow.engine import RepresentationPreprocessor, load_source_arrays, subset_indices
from representation_flow.experiment_contract import (artifact_record, bind_output, canonical_hash, data_identity, save_json, sha256, source_hashes, validate_completed)
from representation_flow.run_a0_prototype import _train_one

_ARRAYS = None
_CACHE = None


def initialize_worker(cache: str, threads: int):
    global _ARRAYS, _CACHE
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _CACHE = Path(cache)
    _ARRAYS = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in _CACHE.glob("*.npy") if not p.stem.startswith("indices_")}


def run_cell(job: dict) -> dict:
    arrays = _ARRAYS
    indices = np.load(_CACHE / f"indices_{job['fraction_tag']}.npy", allow_pickle=False)
    valid = np.load(_CACHE / "indices_validation.npy", allow_pickle=False)
    pre = RepresentationPreprocessor(job["representation"], np.zeros(1), job["mu"], job["scale"])
    args = SimpleNamespace(**job["args"])
    result, prediction = _train_one(arrays, indices, valid, pre, job["representation"], "Small", job["seed"], args, Path(job["output"]))
    row = {**result, "sample_fraction": job["fraction"], "fraction_tag": job["fraction_tag"], "context_label": job["context"],
           "validation_nll": float(np.mean(-prediction["log_p_y"], dtype=np.float64)), "experiment_identity": job["identity"],
           "train_event_uid_sha256": canonical_hash(arrays["event_uid"][indices].tolist()),
           "validation_event_uid_sha256": canonical_hash(arrays["event_uid"][valid].tolist())}
    output = Path(job["output"])
    row["artifacts"] = {
        "checkpoint": artifact_record(Path(result["checkpoint"]), output),
        "predictions": artifact_record(Path(result["predictions"]), output),
        "run_manifest": artifact_record(output / "runs" / result["run_id"] / "run_manifest.json", output),
    }
    return row


def prepare_cache(merged: Path, cache: Path, protocol: dict) -> dict:
    actual = data_identity(merged)
    here = Path(__file__).resolve().parent
    loader_sources = {"engine.py": sha256(here / "engine.py"), "event_split.py": sha256(here.parent / "data_contract/event_split.py")}
    if actual != protocol["main_data"]:
        raise ValueError("main data hashes differ from the prespecified correction protocol")
    index = cache / "cache_manifest.json"
    if index.exists():
        record = json.loads(index.read_text(encoding="utf-8"))
        if record["data"] != actual or record["protocol_sha256"] != canonical_hash(protocol) or record.get("loader_sources") != loader_sources:
            raise ValueError("cached source data contract changed")
        for item in record["files"]:
            if sha256(cache / item["name"]) != item["sha256"]:
                raise ValueError("cached source array checksum mismatch")
        return record
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("partial cache exists; inspect it and choose a new cache path")
    arrays, _ = load_source_arrays(merged, **protocol["split"], eligibility=protocol["eligibility"])
    cache.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(cache / f"{name}.npy", array.astype(str) if array.dtype == object else array, allow_pickle=False)
    fractions = {}
    for fraction in protocol["sample_fractions"]:
        tag = f"f{round(fraction*100):03d}"
        indices = subset_indices(arrays, fraction, protocol["subset_catalog_id"])
        np.save(cache / f"indices_{tag}.npy", indices, allow_pickle=False)
        fractions[tag] = {"events": len(indices), "ordered_uid_sha256": canonical_hash(arrays["event_uid"][indices].tolist())}
    valid = np.flatnonzero(arrays["split"] == "validation")
    np.save(cache / "indices_validation.npy", valid, allow_pickle=False)
    record = {"data": actual, "protocol_sha256": canonical_hash(protocol), "loader_sources": loader_sources, "eligible_events": len(arrays["y"]), "validation_events": len(valid), "fractions": fractions,
              "files": [{"name": p.name, "sha256": sha256(p)} for p in sorted(cache.glob("*.npy"))]}
    save_json(index, record)
    return record


def build_jobs(cache: Path, protocol: dict, output: Path, identity: str, updates: int, deadline: float) -> list[dict]:
    arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in cache.glob("*.npy") if not p.stem.startswith("indices_")}
    jobs = []
    opt = protocol["optimization"]
    for fraction in protocol["sample_fractions"]:
        tag = f"f{round(fraction*100):03d}"
        idx = np.load(cache / f"indices_{tag}.npy", allow_pickle=False)
        y = arrays["y"][idx]
        scale = float(np.std(y, ddof=0))
        q99 = float(np.quantile(y / arrays["p_in"][idx], .99, method="linear"))
        for context, rep, seed in itertools.product(protocol["contexts"], protocol["representations"], protocol["seeds"]):
            pre = RepresentationPreprocessor(rep, np.zeros(1), 0., scale)
            mu = float(np.mean(y - pre.center(arrays["p_in"][idx], arrays["thickness"][idx], arrays["charge"][idx])))
            run_id = f"correction__{context}__{rep}__Small__{tag}__sub{protocol['subset_catalog_id']}__seed{seed}"
            jobs.append({"run_id": run_id, "context": context, "representation": rep, "seed": seed, "fraction": fraction, "fraction_tag": tag,
                         "mu": mu, "scale": scale, "output": str(output.resolve()), "identity": identity,
                         "args": {"run_prefix": "correction", "run_id_template": run_id, "run_status": "COMPLETED_CORRECTION_CELL", "context_label": context,
                         "feature_mode": protocol["contexts"][context], "capacity_widths": {"Small": protocol["capacity"]["hidden_width"]},
                         "fraction_tag": tag, "sample_fraction": fraction, "subset_catalog_id": protocol["subset_catalog_id"], "q99_threshold": q99,
                         "batch_size": opt["batch_size"], "validation_every": opt["validation_every"], "eval_batch_size": opt["eval_batch_size"],
                         "learning_rate": opt["learning_rate"], "max_updates": updates, "torch_threads": opt["torch_threads_per_worker"],
                         "eligibility": protocol["eligibility"], "experiment_identity": identity, "deadline_unix": deadline,
                         "checkpoint_source_paths": {"run_correction_study.py": str(Path(__file__).resolve()), "experiment_contract.py": str(Path(__file__).with_name("experiment_contract.py").resolve())}}})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "execute"))
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/correction_v2_20261107.json"))
    parser.add_argument("--root", type=Path, default=Path("reports/correction_v2_20261107"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4; the measured benchmark contains four concurrent cells")
    started = time.time()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cache = args.root / "cache"
    cache_record = prepare_cache(args.merged, cache, protocol)
    code = source_hashes((Path(__file__).resolve(),))
    environment = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torch_threads": protocol["optimization"]["torch_threads_per_worker"], "workers": args.workers}
    common = {"protocol_sha256": canonical_hash(protocol), "code": code, "data": protocol["main_data"], "cache_sha256": canonical_hash(cache_record), "environment": environment}
    benchmark_path = args.root / "benchmark" / "benchmark_result.json"
    output = args.root / ("benchmark" if args.mode == "benchmark" else "study")
    updates = protocol["runtime_policy"]["benchmark_updates"] if args.mode == "benchmark" else protocol["optimization"]["max_updates"]
    benchmark = None
    if args.mode == "execute":
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark["common_identity"] != common or not benchmark["allowed_under_three_hours"]:
            raise ValueError("current code/environment has no matching sub-three-hour benchmark")
        deadline = started + protocol["runtime_policy"]["max_total_seconds"] - benchmark["benchmark_wall_seconds"] - protocol["runtime_policy"]["reserved_analysis_seconds"]
    else:
        deadline = started + 900
    identity = bind_output(output, {**common, "mode": args.mode, "max_updates": updates})
    jobs = build_jobs(cache, protocol, output, identity, updates, deadline)
    total_cells = len(jobs)
    if args.mode == "benchmark":
        # Full-data cells cover both context dimensions and both physical centers.
        jobs = [j for j in jobs if j["fraction"] == 1 and j["seed"] == protocol["seeds"][0] and j["representation"] != "Raw"]
    expected_ids = {j["run_id"] for j in jobs}
    score_path = output / "scores.json"
    scores = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {"status": "RUNNING", "experiment_identity": identity, "runs": [], "failures": []}
    if scores["experiment_identity"] != identity:
        raise ValueError("scores identity differs")
    completed = validate_completed(scores["runs"], output, identity, expected_ids)
    if scores.get("status") == "COMPLETE" and completed == expected_ids:
        print(json.dumps({"status": "ALREADY_COMPLETE_VERIFIED", "cells": len(completed)}), flush=True)
        return
    original_start = scores.setdefault("execution_started_unix", started)
    if args.mode == "execute":
        deadline = original_start + protocol["runtime_policy"]["max_total_seconds"] - benchmark["benchmark_wall_seconds"] - protocol["runtime_policy"]["reserved_analysis_seconds"]
        for job in jobs:
            job["args"]["deadline_unix"] = deadline
        if time.time() >= deadline:
            raise TimeoutError("the original study deadline has expired; resume cannot reset the time budget")
    save_json(score_path, scores)
    pending = [j for j in jobs if j["run_id"] not in completed]
    training_started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(str(cache.resolve()), environment["torch_threads"])) as executor:
        futures = {executor.submit(run_cell, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
                scores["runs"].append(row)
                print(json.dumps({"completed": row["run_id"], "nll": row["validation_nll"], "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
            except Exception as exc:
                scores["failures"].append({"run_id": job["run_id"], "type": type(exc).__name__, "error": str(exc)})
                print(json.dumps({"failed": scores["failures"][-1]}), flush=True)
                for other in futures:
                    other.cancel()
            save_json(score_path, scores)
    elapsed = time.time() - training_started
    done = validate_completed(scores["runs"], output, identity, expected_ids)
    scores["status"] = "COMPLETE" if done == expected_ids and not scores["failures"] else "INCOMPLETE"
    scores["wall_seconds_this_invocation"] = time.time() - started
    scores["wall_seconds_since_first_start"] = time.time() - original_start
    save_json(score_path, scores)
    if scores["status"] != "COMPLETE":
        raise RuntimeError("study is incomplete; failed cells remain visible")
    if args.mode == "benchmark":
        # Estimate steady-state throughput conservatively from the actual parallel batch.
        per_cell = max(row["elapsed_seconds"] for row in scores["runs"])
        training_forecast = per_cell * (protocol["optimization"]["max_updates"] / updates) * np.ceil(total_cells / args.workers)
        conservative = float(training_forecast * protocol["runtime_policy"]["forecast_safety_factor"] + protocol["runtime_policy"]["reserved_analysis_seconds"] + (time.time() - started) + 90)
        result = {"common_identity": common, "benchmark_wall_seconds": time.time() - started, "parallel_batch_seconds": elapsed,
                  "maximum_cell_seconds": per_cell, "full_study_cells": total_cells, "forecast_training_seconds": float(training_forecast), "conservative_total_seconds": conservative,
                  "allowed_under_three_hours": conservative < protocol["runtime_policy"]["max_total_seconds"], "benchmark_scores_are_not_scientific_results": True}
        save_json(benchmark_path, result)
        print(json.dumps(result), flush=True)
    else:
        frozen = {"schema": "correction-checkpoint-freeze-v1", "experiment_identity": identity, "frozen_before_confirmation": True,
                  "models": [{"run_id": r["run_id"], "representation": r["representation"], "context_label": r["context_label"], "sample_fraction": r["sample_fraction"], "seed": r["seed"], "artifacts": r["artifacts"]} for r in scores["runs"]]}
        save_json(output / "FROZEN_CHECKPOINT_MANIFEST.json", frozen)
        print(json.dumps({"status": scores["status"], "cells": len(scores["runs"]), "wall_seconds": scores["wall_seconds_this_invocation"]}), flush=True)


if __name__ == "__main__":
    main()
