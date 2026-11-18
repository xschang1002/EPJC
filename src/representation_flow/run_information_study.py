"""Equal-structure information study; new namespace and explicit scientific identity.

Uses the unchanged flow and physical preprocessing classes. It deliberately has
its own training adapter: no monkeypatch or false legacy context-mode label.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, subset_indices
from data_contract.event_split import split_label
from representation_flow.experiment_contract import artifact_record, bind_output, canonical_hash, save_json, sha256, validate_completed
from representation_flow.information_context import CONTEXTS, context_spec, fixed_group_ids
from representation_flow.information_checkpoint import (ROOT, build_information_contract, scientific_sources,
    validate_information_contract, load_information_checkpoint, make_tensors, evaluate_information)

_ARRAYS = None
_CACHE = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def budget_contract(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {key: record[key] for key in ("started_unix", "deadline_unix", "maximum_wall_seconds")}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in result.values()):
        raise ValueError("budget must have finite numeric clocks")
    if not 0 < result["maximum_wall_seconds"] <= 43200 or abs(result["deadline_unix"] - result["started_unix"] - result["maximum_wall_seconds"]) > 1e-6:
        raise ValueError("budget deadline is not anchored to its original authorized start")
    check_deadline(result["deadline_unix"])
    return result


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("the original authorized research deadline has expired")


def validate_protocol(protocol):
    if (protocol.get("schema") != "information-study/v1" or protocol.get("contexts") != [context_spec(name) for name in CONTEXTS]
            or protocol.get("representations") != ["Raw", "CorrectedPhysics"] or protocol.get("sample_fractions") != [.1, 1.]
            or protocol.get("seeds") != [3101, 3102, 3103, 3104, 3105] or protocol.get("subset_catalog_id") != 4101
            or protocol.get("capacity") != {"label": "Small", "hidden_width": 32, "context_dim": 5}
            or protocol.get("split") != {"validation_fraction": .1, "diagnostic_fraction": .3}
            or protocol.get("eligibility") != "forward" or protocol.get("expected_cells") != 80):
        raise ValueError("protocol differs from the fixed 80-cell information design")
    expected = {"max_updates": 20000, "batch_size": 512, "validation_every": 500, "eval_batch_size": 8192,
                "learning_rate": .0003, "gradient_clip_norm": 5., "torch_threads_per_worker": 1}
    if protocol.get("optimization") != expected:
        raise ValueError("optimization differs from the declared common budget")
    return protocol


def array_cache(source_cache):
    return {name: np.load(Path(source_cache) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("event_uid", "p_in", "thickness", "charge", "y", "split")}


def prepare_cache(source_cache, output, protocol):
    source_cache, output = Path(source_cache).resolve(), Path(output).resolve()
    source_manifest = json.loads((source_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    if source_manifest["data"] != protocol["main_data"]:
        raise ValueError("source arrays do not match the frozen main data")
    expected_loader = {"engine.py": sha256(ROOT / "src/representation_flow/engine.py"), "event_split.py": sha256(ROOT / "src/data_contract/event_split.py")}
    if source_manifest.get("loader_sources") != expected_loader:
        raise ValueError("source-cache loader identity changed")
    for row in source_manifest["files"]:
        path = (source_cache / row["name"]).resolve()
        if not path.is_relative_to(source_cache) or sha256(path) != row["sha256"]:
            raise ValueError("source array identity mismatch")
    common = {"protocol_sha256": canonical_hash(protocol), "source_cache_manifest_sha256": sha256(source_cache / "cache_manifest.json"),
              "source_data": protocol["main_data"], "loader_sources": expected_loader}
    manifest_path = output / "cache_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in common.items()):
            raise ValueError("information preprocessing cache identity changed")
        for row in existing["files"]:
            if sha256(output / row["name"]) != row["sha256"]:
                raise ValueError("information cache output changed")
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("partial information cache exists; preserve and use a fresh directory")
    arrays = array_cache(source_cache)
    expected_split = np.asarray([split_label(str(uid), **protocol["split"]) for uid in arrays["event_uid"]])
    if not np.array_equal(expected_split, arrays["split"]):
        raise ValueError("source-cache split does not implement the declared event-UID partition")
    output.mkdir(parents=True, exist_ok=True)
    preprocessing, fractions = {}, {}
    for fraction in protocol["sample_fractions"]:
        tag = f"f{round(fraction * 100):03d}"
        indices = subset_indices(arrays, fraction, protocol["subset_catalog_id"])
        np.save(output / f"indices_{tag}.npy", indices, allow_pickle=False)
        y = np.asarray(arrays["y"][indices])
        scale = float(np.std(y, ddof=0))
        preprocessing[tag] = {}
        for rep in protocol["representations"]:
            pre = RepresentationPreprocessor(rep, np.zeros(1), 0., scale)
            pre.mu = float(np.mean(y - pre.center(*(arrays[k][indices] for k in ("p_in", "thickness", "charge")))))
            preprocessing[tag][rep] = pre.to_dict()
        fractions[tag] = {"events": len(indices), "ordered_uid_sha256": canonical_hash(arrays["event_uid"][indices].tolist())}
    valid = np.flatnonzero(arrays["split"] == "validation")
    np.save(output / "indices_validation.npy", valid, allow_pickle=False)
    save_json(output / "preprocessing.json", preprocessing)
    record = {**common, "fractions": fractions, "validation_events": len(valid),
              "validation_uid_sha256": canonical_hash(arrays["event_uid"][valid].tolist()),
              "files": [{"name": p.name, "sha256": sha256(p)} for p in sorted(output.iterdir())]}
    save_json(manifest_path, record)
    return record


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode()); digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_information_cell(arrays, indices, valid, pre, job):
    check_deadline(job["deadline_unix"])
    if pre.name != job["representation"]:
        raise ValueError("cell label does not match frozen preprocessing")
    output = Path(job["output"])
    run_dir = output / "runs" / job["run_id"]
    if run_dir.exists():
        raise FileExistsError("refusing to overwrite any existing information cell")
    opt = job["optimization"]
    torch.set_num_threads(opt["torch_threads_per_worker"])
    contract = build_information_contract(pre, job["context_label"], job["protocol_sha256"])
    train = make_tensors(arrays, indices, pre, contract["context_spec"])
    validation = make_tensors(arrays, valid, pre, contract["context_spec"])
    config = {"context_dim": 5, "hidden": 32, "seed": job["seed"], "bound": 8.}
    flow = ConditionalSplineFlow(**config)
    initialization_hash = state_hash(flow.state_dict())
    optimizer = torch.optim.Adam(flow.parameters(), lr=opt["learning_rate"], weight_decay=0.)
    rng = np.random.default_rng(job["seed"])
    order, cursor = rng.permutation(len(indices)), 0
    if len(indices) < opt["batch_size"]:
        raise ValueError("training subset smaller than a complete batch")
    best, selected, best_state, best_optimizer, best_training = float("inf"), None, None, None, None
    history, trace = [], hashlib.sha256()
    started = time.perf_counter()
    for update in range(1, job["max_updates"] + 1):
        check_deadline(job["deadline_unix"])
        if cursor + opt["batch_size"] > len(order):
            order, cursor = rng.permutation(len(indices)), 0
        batch = order[cursor:cursor + opt["batch_size"]]
        cursor += opt["batch_size"]
        trace.update(np.asarray(batch, dtype="<i8").tobytes())
        optimizer.zero_grad(set_to_none=True)
        y, c, p, context, scale = (value[batch] for value in train)
        loss = -flow.original_terms(y, c, pre.mu, scale, p, context)["log_p_y"].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite information training loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(flow.parameters(), opt["gradient_clip_norm"]))
        if not np.isfinite(norm):
            raise FloatingPointError("nonfinite information training gradient")
        optimizer.step()
        if update == 1 or update % opt["validation_every"] == 0 or update == job["max_updates"]:
            values = evaluate_information(flow, validation, pre, opt["eval_batch_size"])
            score = float(np.mean(-values["log_p_y"]))
            if not np.isfinite(score):
                raise FloatingPointError("nonfinite information validation NLL")
            history.append({"update": update, "train_nll": float(loss.detach()), "validation_nll": score, "grad_norm_pre_clip": norm})
            if score < best:
                best, selected = score, update
                best_state, best_optimizer = copy.deepcopy(flow.state_dict()), copy.deepcopy(optimizer.state_dict())
                best_training = {"update": update, "batch_order": torch.from_numpy(order.copy()), "batch_cursor": cursor,
                                 "numpy_rng_state": copy.deepcopy(rng.bit_generator.state), "torch_rng_state": torch.get_rng_state().clone()}
    if best_state is None:
        raise RuntimeError("no finite checkpoint was selected")
    flow.load_state_dict(best_state)
    prediction = evaluate_information(flow, validation, pre, opt["eval_batch_size"], components=True)
    check_deadline(job["deadline_unix"])
    validate_information_contract(contract)
    row = {key: job[key] for key in ("run_id", "context_label", "representation", "fraction_tag", "sample_fraction", "seed", "experiment_identity", "protocol_sha256", "max_updates")}
    row.update(status="COMPLETE_INFORMATION_CELL", capacity="Small", model_config=config, parameter_count=flow.parameter_count(),
               selected_update=selected, initialization_hash=initialization_hash, batch_trace_sha256=trace.hexdigest(),
               train_events=len(indices), validation_events=len(valid), train_uid_sha256=canonical_hash(arrays["event_uid"][indices].tolist()),
               validation_uid_sha256=canonical_hash(arrays["event_uid"][valid].tolist()),
               optimization=opt, preprocessor=pre.to_dict(), information_contract=contract, history=history,
               validation_nll=float(np.mean(-prediction["log_p_y"], dtype=np.float64)), elapsed_seconds=time.perf_counter()-started)
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {**{key: row[key] for key in ("representation", "context_label", "experiment_identity", "model_config", "parameter_count", "seed", "selected_update", "initialization_hash", "batch_trace_sha256", "train_uid_sha256", "validation_uid_sha256")},
               "state_dict": best_state, "optimizer_state_dict": best_optimizer, "training_state": best_training, "information_contract": contract}
    torch.save(payload, run_dir / "checkpoint.pt")
    np.savez_compressed(run_dir / "predictions.npz", event_uid=arrays["event_uid"][valid].astype(str),
                        **{key: arrays[key][valid] for key in ("p_in", "thickness", "charge", "y")},
                        group_id=fixed_group_ids(*(arrays[key][valid] for key in ("p_in", "thickness", "charge"))), **prediction)
    save_json(run_dir / "run_manifest.json", row)
    row["artifacts"] = {key: artifact_record(run_dir / name, output) for key, name in
                        (("checkpoint", "checkpoint.pt"), ("predictions", "predictions.npz"), ("run_manifest", "run_manifest.json"))}
    # Verify the public new-contract loader against the actual saved weights.
    _, _, reloaded, _ = load_information_checkpoint(run_dir / "checkpoint.pt", expected_sha256=row["artifacts"]["checkpoint"]["sha256"], expected_experiment_identity=job["experiment_identity"])
    sample = tuple(value[:min(256, len(valid))] for value in validation)
    replay = evaluate_information(reloaded, sample, pre, opt["eval_batch_size"])["log_p_y"]
    if not np.array_equal(replay, prediction["log_p_y"][:len(replay)]):
        if not np.allclose(replay, prediction["log_p_y"][:len(replay)], rtol=2e-6, atol=1e-4):
            raise ValueError("saved information checkpoint did not replay")
    check_deadline(job["deadline_unix"])
    return row


def initialize_worker(source_cache, cache, threads):
    global _ARRAYS, _CACHE
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _ARRAYS, _CACHE = array_cache(source_cache), Path(cache)


def run_job(job):
    pre = RepresentationPreprocessor.from_dict(job["preprocessor"])
    return train_information_cell(_ARRAYS, np.load(_CACHE / f"indices_{job['fraction_tag']}.npy", allow_pickle=False),
                                  np.load(_CACHE / "indices_validation.npy", allow_pickle=False), pre, job)


def build_jobs(protocol, cache, output, identity, updates, deadline):
    preprocessing = json.loads((Path(cache) / "preprocessing.json").read_text(encoding="utf-8"))
    jobs = []
    for context, rep, fraction, seed in itertools.product(CONTEXTS, protocol["representations"], protocol["sample_fractions"], protocol["seeds"]):
        tag = f"f{round(fraction * 100):03d}"
        jobs.append({"run_id": f"information__{context}__{rep}__Small__{tag}__sub4101__seed{seed}", "context_label": context,
                     "representation": rep, "fraction_tag": tag, "sample_fraction": fraction, "seed": seed,
                     "experiment_identity": identity, "protocol_sha256": canonical_hash(protocol), "max_updates": updates,
                     "deadline_unix": deadline, "optimization": protocol["optimization"], "preprocessor": preprocessing[tag][rep], "output": str(Path(output).resolve())})
    return jobs


def validate_pairing(rows):
    for seed in sorted({r["seed"] for r in rows}):
        selected = [r for r in rows if r["seed"] == seed]
        if len({r["initialization_hash"] for r in selected}) != 1 or len({r["parameter_count"] for r in selected}) != 1:
            raise ValueError("equal-structure or initialization matching failed")
        for fraction in sorted({r["fraction_tag"] for r in selected}):
            same = [r for r in selected if r["fraction_tag"] == fraction]
            for key in ("train_uid_sha256", "validation_uid_sha256", "batch_trace_sha256"):
                if len({r[key] for r in same}) != 1:
                    raise ValueError("ordered data/minibatch matching failed: " + key)
            if len({r["preprocessor"]["scale"] for r in same}) != 1:
                raise ValueError("representations did not share raw-target scale")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "freeze", "execute"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol/information_study_20261108.json")
    parser.add_argument("--root", type=Path, default=ROOT / "reports/information_extension_20261108")
    parser.add_argument("--source-cache", type=Path, default=ROOT / "reports/correction_v2_20261107/cache")
    parser.add_argument("--budget", type=Path, default=ROOT.parent / "_research_extension_20261108/AUTHORIZATION_AND_BUDGET.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4 for the measured execution design")
    started = time.time()
    protocol = validate_protocol(json.loads(args.protocol.read_text(encoding="utf-8")))
    budget = budget_contract(args.budget)
    cache = args.root / "cache"
    cache_record = prepare_cache(args.source_cache, cache, protocol)
    common = {"protocol_sha256": canonical_hash(protocol), "source_files": scientific_sources(), "budget": budget,
              "source_data": protocol["main_data"], "cache_sha256": canonical_hash(cache_record),
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__), "workers": args.workers, "torch_threads": 1}}
    benchmark_path, freeze_path = args.root / "benchmark/benchmark_result.json", args.root / "PROTOCOL_FREEZE.json"
    if args.mode in ("freeze", "execute"):
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark["common_identity"] != common or benchmark["status"] != "COMPLETE_INFORMATION_BENCHMARK":
            raise ValueError("matching completed scientific-code benchmark required")
        check_deadline(budget["deadline_unix"] - benchmark["conservative_training_seconds"])
        if args.mode == "freeze":
            if freeze_path.exists():
                raise FileExistsError("information protocol freeze already exists")
            save_json(freeze_path, {"schema": "information-protocol-freeze/v1", "status": "FROZEN_INFORMATION_PROTOCOL", "generated_utc": utc(),
                                  "common_identity": common, "benchmark_sha256": sha256(benchmark_path), "protocol": protocol,
                                  "training_execution_approved": True})
            print(json.dumps({"status": "FROZEN_INFORMATION_PROTOCOL", "path": str(freeze_path)})); return
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("status") != "FROZEN_INFORMATION_PROTOCOL" or freeze.get("training_execution_approved") is not True or freeze["common_identity"] != common or freeze["benchmark_sha256"] != sha256(benchmark_path):
            raise ValueError("formal information training has no matching explicit protocol freeze")
    output = args.root / ("benchmark" if args.mode == "benchmark" else "study")
    updates = protocol["runtime_policy"]["benchmark_updates"] if args.mode == "benchmark" else protocol["optimization"]["max_updates"]
    identity = bind_output(output, {**common, "mode": args.mode, "max_updates": updates})
    jobs = build_jobs(protocol, cache, output, identity, updates, budget["deadline_unix"])
    if args.mode == "benchmark":
        jobs = [j for j in jobs if j["seed"] == 3101 and j["fraction_tag"] == "f100" and j["representation"] == "CorrectedPhysics"]
    expected = {j["run_id"] for j in jobs}
    scores_path = output / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {"status": "RUNNING", "experiment_identity": identity,
             "execution_started_unix": started, "runs": [], "failures": []}
    if scores["experiment_identity"] != identity or scores.get("failures"):
        raise ValueError("existing information scores have a different identity or unresolved failures")
    done = validate_completed(scores["runs"], output, identity, expected)
    completion_product = benchmark_path if args.mode == "benchmark" else output / "FROZEN_CHECKPOINT_MANIFEST.json"
    if done == expected and scores["status"] == "COMPLETE" and completion_product.exists():
        print(json.dumps({"status": "ALREADY_COMPLETE_VERIFIED", "cells": len(done)})); return
    save_json(scores_path, scores)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.source_cache.resolve()), str(cache.resolve()), 1)) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs if job["run_id"] not in done}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result(); scores["runs"].append(row)
                print(json.dumps({"completed": row["run_id"], "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
            except Exception as exc:
                scores["failures"].append({"run_id": job["run_id"], "type": type(exc).__name__, "error": str(exc)})
                for pending in futures: pending.cancel()
            save_json(scores_path, scores)
    done = validate_completed(scores["runs"], output, identity, expected)
    validate_pairing(scores["runs"])
    if time.time() >= budget["deadline_unix"]:
        scores["failures"].append({"type": "TimeoutError", "error": "global deadline exceeded before complete matrix freeze"})
    scores.update(status="COMPLETE" if done == expected and not scores["failures"] else "INCOMPLETE",
                  wall_seconds_since_first_start=time.time()-scores["execution_started_unix"], completed_utc=utc())
    save_json(scores_path, scores)
    if scores["status"] != "COMPLETE":
        raise RuntimeError("information matrix incomplete; all failures preserved")
    check_deadline(budget["deadline_unix"])
    if args.mode == "benchmark":
        forecast = max(r["elapsed_seconds"] for r in scores["runs"]) * protocol["optimization"]["max_updates"] / updates * np.ceil(80 / args.workers)
        result = {"status": "COMPLETE_INFORMATION_BENCHMARK", "common_identity": common, "benchmark_updates": updates,
                  "benchmark_wall_seconds": time.time()-started, "forecast_training_seconds": float(forecast),
                  "conservative_training_seconds": float(forecast * protocol["runtime_policy"]["forecast_safety_factor"]),
                  "benchmark_scores_are_not_scientific_results": True, "generated_utc": utc()}
        save_json(benchmark_path, result); print(json.dumps(result), flush=True)
    else:
        frozen = {"schema": "information-checkpoint-freeze/v1", "status": "COMPLETE_INFORMATION_CHECKPOINT_FREEZE", "generated_utc": utc(),
                  "experiment_identity": identity, "protocol_sha256": canonical_hash(protocol), "expected_cells": 80,
                  "models": [{key: r[key] for key in ("run_id", "context_label", "representation", "fraction_tag", "sample_fraction", "seed", "capacity", "model_config", "parameter_count", "initialization_hash", "train_uid_sha256", "batch_trace_sha256", "artifacts")} for r in scores["runs"]]}
        save_json(output / "FROZEN_CHECKPOINT_MANIFEST.json", frozen)
        print(json.dumps({"status": frozen["status"], "cells": 80}), flush=True)


if __name__ == "__main__":
    main()
