"""Reassess only runtime after the frozen 60-cell study completes; never inspect scores for selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.experiment_contract import canonical_hash, save_json, sha256, source_hashes
from representation_flow.run_correction_study import prepare_cache
from representation_flow.run_corrected_formal_extension import validate_reused_rows

INITIAL_SNAPSHOT = "benchmark_result.initial_measured.json"
INITIAL_SHA = "benchmark_result.initial_measured.sha256"


def calculate_runtime_decision(measurement: dict, core_scores: dict, protocol: dict, core_benchmark: dict, now_unix: float) -> dict:
    """Use actual elapsed core wall time, applying the safety factor only to future work."""
    if core_scores.get("status") != "COMPLETE" or len(core_scores.get("runs", [])) != 60 or core_scores.get("failures"):
        raise ValueError("core60 must be complete before runtime reassessment")
    runtime = protocol["runtime_policy"]
    if (runtime["max_total_seconds"], runtime["forecast_safety_factor"], runtime["reserved_analysis_seconds"]) != (10800, 1.35, 1200):
        raise ValueError("the frozen 10800-second limit, 1.35 factor and 1200-second reserve cannot be relaxed")
    elapsed = float(now_unix) - float(core_scores["execution_started_unix"])
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("core elapsed time must be finite and nonnegative")
    future_forecast = float(measurement["forecast_extension_training_seconds"])
    if not np.isfinite(future_forecast) or future_forecast <= 0:
        raise ValueError("future extension forecast must be positive and finite")
    probes = {
        "archived_core_seconds": float(runtime["prior_core_benchmark_seconds"]),
        "archived_extension_seconds": float(runtime["prior_extension_benchmark_seconds"]),
        "current_core_seconds": float(core_benchmark["benchmark_wall_seconds"]),
        "current_extension_seconds": float(measurement["benchmark_wall_seconds"]),
    }
    if any(not np.isfinite(value) or value < 0 for value in probes.values()):
        raise ValueError("benchmark costs must be nonnegative and finite")
    fixed_overhead = 90.0
    future_conservative = future_forecast * runtime["forecast_safety_factor"]
    total = elapsed + future_conservative + runtime["reserved_analysis_seconds"] + sum(probes.values()) + fixed_overhead
    return {
        "schema": "completed-core-runtime-reassessment-v1",
        "decision_basis": "completed_core_actual_wall",
        "checked_at_unix": float(now_unix),
        "core_execution_started_unix": float(core_scores["execution_started_unix"]),
        "actual_core_elapsed_seconds": elapsed,
        "future_extension_forecast_seconds": future_forecast,
        "future_safety_factor": runtime["forecast_safety_factor"],
        "future_extension_conservative_seconds": future_conservative,
        "benchmark_costs": probes,
        "total_benchmark_cost_seconds": sum(probes.values()),
        "reserved_analysis_seconds": runtime["reserved_analysis_seconds"],
        "fixed_overhead_seconds": fixed_overhead,
        "conservative_combined_total_seconds": total,
        "maximum_total_seconds": runtime["max_total_seconds"],
        "remaining_margin_seconds": runtime["max_total_seconds"] - total,
        "allowed_combined_under_three_hours": total < runtime["max_total_seconds"],
        "initial_forecast_allowed": measurement["allowed_combined_under_three_hours"],
        "initial_forecast_seconds": measurement["conservative_combined_total_seconds"],
        "scientific_scores_used_for_decision": False,
    }


def initial_measurement(benchmark_dir: Path):
    """Read the immutable first measurement when present; reject a broken snapshot pair."""
    snapshot = benchmark_dir / INITIAL_SNAPSHOT
    checksum = benchmark_dir / INITIAL_SHA
    if snapshot.exists() != checksum.exists():
        raise ValueError("incomplete immutable benchmark snapshot/checksum pair")
    if snapshot.exists():
        expected = checksum.read_text(encoding="ascii").split()[0]
        if sha256(snapshot) != expected:
            raise ValueError("immutable original benchmark checksum mismatch")
        raw = snapshot.read_bytes()
    else:
        raw = (benchmark_dir / "benchmark_result.json").read_bytes()
    measurement = json.loads(raw)
    if measurement.get("decision_basis") or measurement.get("runtime_decision"):
        raise ValueError("original measured benchmark is missing; refusing to use an already revised gate as the baseline")
    return measurement, raw


def persist_runtime_decision(benchmark_dir: Path, measurement: dict, original_bytes: bytes, decision: dict) -> Path:
    """Preserve the measured evidence, then update the derived permission with an audit trail."""
    digest = hashlib.sha256(original_bytes).hexdigest()
    if json.loads(original_bytes) != measurement:
        raise ValueError("original benchmark bytes and parsed measurement disagree")
    snapshot = benchmark_dir / INITIAL_SNAPSHOT
    checksum = benchmark_dir / INITIAL_SHA
    if not snapshot.exists() and not checksum.exists():
        with snapshot.open("xb") as stream:
            stream.write(original_bytes)
        with checksum.open("x", encoding="ascii") as stream:
            stream.write(f"{digest}  {INITIAL_SNAPSHOT}\n")
    else:
        initial_measurement(benchmark_dir)
        if sha256(snapshot) != digest:
            raise ValueError("original benchmark changed between verification and persistence")
    decision = {**decision, "initial_measurement_sha256": digest, "initial_measurement_path": INITIAL_SNAPSHOT}
    history = benchmark_dir / "runtime_decisions"
    history.mkdir(exist_ok=True)
    path = history / f"decision_{time.time_ns()}.json"
    # The unique immutable decision is saved before the mutable latest pointers.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    save_json(benchmark_dir / "runtime_decision.json", decision)
    gate = {**measurement,
            "allowed_combined_under_three_hours": decision["allowed_combined_under_three_hours"],
            "conservative_combined_total_seconds": decision["conservative_combined_total_seconds"],
            "initial_forecast_allowed": measurement["allowed_combined_under_three_hours"],
            "initial_conservative_combined_total_seconds": measurement["conservative_combined_total_seconds"],
            "initial_measurement_sha256": digest,
            "actual_core_elapsed_seconds": decision["actual_core_elapsed_seconds"],
            "decision_basis": decision["decision_basis"],
            "runtime_decision": {"path": path.relative_to(benchmark_dir).as_posix(), "sha256": sha256(path)}}
    save_json(benchmark_dir / "benchmark_result.json", gate)
    return path


def verified_runtime_inputs(merged, core_root, extension_root, core_protocol_path, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    core_protocol = json.loads(core_protocol_path.read_text(encoding="utf-8"))
    measurement, original_bytes = initial_measurement(extension_root / "benchmark")
    common = measurement["common_identity"]
    if canonical_hash(protocol) != common["protocol_sha256"] or canonical_hash(core_protocol) != protocol["core_protocol_sha256"]:
        raise ValueError("protocol identity changed after the measured benchmark")
    extension_script = Path(__file__).with_name("run_corrected_formal_extension.py")
    if sha256(extension_script) != common["extension_source_sha256"]:
        raise ValueError("extension runner source changed after the measured benchmark")
    core_scores_path = core_root / "study/scores.json"
    core_scores = json.loads(core_scores_path.read_text(encoding="utf-8"))
    if core_scores.get("status") != "COMPLETE" or len(core_scores.get("runs", [])) != 60 or core_scores.get("failures"):
        raise ValueError("core60 is not complete; no runtime gate may be updated")
    if not (core_root / "cache/cache_manifest.json").is_file():
        raise ValueError("validated core cache is required")
    cache = prepare_cache(merged, core_root / "cache", core_protocol)
    core_common = common["core_common"]
    current_code = source_hashes((Path(__file__).with_name("run_correction_study.py").resolve(),))
    current_environment = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": protocol["optimization"]["torch_threads_per_worker"],
                           "workers": core_common["environment"]["workers"]}
    expected = {"protocol_sha256": canonical_hash(core_protocol), "code": current_code, "data": core_protocol["main_data"],
                "cache_sha256": canonical_hash(cache), "environment": current_environment}
    if core_common != expected:
        raise ValueError("core source, cache, data or environment identity changed")
    preprocessing = json.loads((extension_root / "cache/preprocessing_manifest.json").read_text(encoding="utf-8"))
    if canonical_hash(preprocessing) != common["preprocessing_sha256"]:
        raise ValueError("extension preprocessing identity changed")
    for record in preprocessing["files"]:
        if sha256(extension_root / "cache" / record["name"]) != record["sha256"]:
            raise ValueError("extension subset indices changed")
    _, verified_scores = validate_reused_rows(core_root, protocol, preprocessing, core_common)
    # Reuse checks verify all 60 artifact hashes and all 20 shared preprocessors.
    # Check the remaining checkpoints' source contracts as well, without scoring events.
    for row in verified_scores["runs"]:
        checkpoint = core_root / "study" / row["artifacts"]["checkpoint"]["path"]
        payload, pre = load_checkpoint(checkpoint, expected_sha256=row["artifacts"]["checkpoint"]["sha256"])
        if payload["experiment_identity"] != verified_scores["experiment_identity"] or pre.name != row["representation"]:
            raise ValueError("completed core checkpoint identity differs")
    core_benchmark = json.loads((core_root / "benchmark/benchmark_result.json").read_text(encoding="utf-8"))
    if core_benchmark["common_identity"] != core_common:
        raise ValueError("core benchmark identity differs")
    evidence = {"core_scores_sha256": sha256(core_scores_path), "core_common_identity_sha256": canonical_hash(core_common),
                "preprocessing_sha256": canonical_hash(preprocessing), "runtime_recheck_source_sha256": sha256(Path(__file__)),
                "verified_core_artifacts": 60, "verified_core_checkpoint_sources": 60}
    return measurement, original_bytes, verified_scores, protocol, core_benchmark, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--core-root", type=Path, default=Path("reports/correction_v2_20261107"))
    parser.add_argument("--extension-root", type=Path, default=Path("reports/corrected_formal_extension_20261107"))
    parser.add_argument("--core-protocol", type=Path, default=Path("protocol/correction_v2_20261107.json"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/corrected_formal_extension_20261107.json"))
    parser.add_argument("--apply", action="store_true", help="record the audited decision and update only the derived runtime gate")
    args = parser.parse_args()
    measurement, raw, core_scores, protocol, core_benchmark, evidence = verified_runtime_inputs(
        args.merged, args.core_root, args.extension_root, args.core_protocol, args.protocol)
    decision = calculate_runtime_decision(measurement, core_scores, protocol, core_benchmark, time.time())
    decision.update(evidence)
    if args.apply:
        path = persist_runtime_decision(args.extension_root / "benchmark", measurement, raw, decision)
        decision["saved_decision"] = str(path)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if not decision["allowed_combined_under_three_hours"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
