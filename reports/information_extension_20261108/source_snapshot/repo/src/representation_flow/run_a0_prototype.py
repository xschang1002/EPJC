"""Run the authorized first round: A0 plus the 24-cell prototype only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from representation_flow.checkpoint_contract import build_checkpoint_contract, validate_checkpoint_contract
from representation_flow.engine import (
    ConditionalSplineFlow,
    context_features,
    fit_preprocessors,
    load_source_arrays,
    run_a0_checks,
    sha256_file,
    subset_indices,
)


REPRESENTATIONS = ("Raw", "Simple", "Physics", "Learned")
CAPACITIES = {"Small": 32, "Large": 128}
SEEDS = (11, 12, 13)


def prepare_new_output(output: Path) -> None:
    """Exploratory runs have no identity-checked resume: preserve existing results."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output already contains results; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _hash_indices(indices: np.ndarray, event_uids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for idx in indices:
        digest.update(str(event_uids[int(idx)]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _state_hash(state_dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        digest.update(state_dict[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _as_tensors(arrays, indices, preprocessor, device, feature_mode="full"):
    p = arrays["p_in"][indices]
    t = arrays["thickness"][indices]
    q = arrays["charge"][indices]
    y = arrays["y"][indices]
    c = preprocessor.center(p, t, q)
    scale = preprocessor.scale_for(p, t, q)
    context = context_features(p, t, q, feature_mode)
    return (
        torch.as_tensor(y, dtype=torch.float32, device=device),
        torch.as_tensor(c, dtype=torch.float32, device=device),
        torch.as_tensor(p, dtype=torch.float32, device=device),
        torch.as_tensor(context, dtype=torch.float32, device=device),
        torch.as_tensor(scale, dtype=torch.float32, device=device),
    )


def _evaluate(flow, tensors, preprocessor, device, batch_size=8192):
    y, c, p, context, scale = tensors
    n = len(y)
    out = {key: [] for key in ("z", "b", "log_q_z", "log_f_b", "log_p_y", "log_cdf_y", "log_sf_y")}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            terms = flow.original_terms(y[sl], c[sl], float(preprocessor.mu), scale[sl], p[sl], context[sl])
            for key in out:
                out[key].append(terms[key].detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in out.items()}


def _train_one(arrays, train_indices, validation_indices, preprocessor, representation, capacity, seed, args, output_dir):
    deadline_unix = getattr(args, "deadline_unix", None)
    if deadline_unix is not None and time.time() >= deadline_unix:
        raise TimeoutError("training deadline reached before starting this run")
    run_id_template = getattr(args, "run_id_template", None)
    if run_id_template:
        run_id = run_id_template.format(
            representation=representation,
            capacity=capacity,
            context_label=getattr(args, "context_label", "full"),
            fraction_tag=getattr(args, "fraction_tag", "f010"),
            subset_catalog_id=getattr(args, "subset_catalog_id", 4101),
            seed=seed,
        )
    else:
        run_id = f"{args.run_prefix}__{representation}__{capacity}__seed{seed}"
    run_dir = output_dir / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing training run: {run_dir}")
    torch_threads = getattr(args, "torch_threads", None)
    if torch_threads is not None:
        if int(torch_threads) <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(int(torch_threads))
    device = torch.device("cpu")
    feature_mode = getattr(args, "feature_mode", "full")
    checkpoint_contract = build_checkpoint_contract(
        preprocessor, feature_mode,
        source_paths={"run_a0_prototype.py": Path(__file__), **getattr(args, "checkpoint_source_paths", {})},
        eligibility=getattr(args, "eligibility", "forward"),
    )
    context_dim = 5 if feature_mode == "full" else 2
    train_tensors = _as_tensors(arrays, train_indices, preprocessor, device, feature_mode)
    validation_tensors = _as_tensors(arrays, validation_indices, preprocessor, device, feature_mode)
    hidden_width = getattr(args, "capacity_widths", CAPACITIES)[capacity]
    flow = ConditionalSplineFlow(context_dim=context_dim, hidden=hidden_width, seed=seed)
    flow.to(device)
    initialization_hash = _state_hash(flow.state_dict())
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_indices))
    cursor = 0
    best_value = float("inf")
    best_state = None
    best_optimizer_state = None
    best_training_state = None
    selected_update = None
    history = []
    start_time = time.perf_counter()
    for update in range(1, args.max_updates + 1):
        if deadline_unix is not None and time.time() >= deadline_unix:
            raise TimeoutError(f"training deadline reached before update {update}; no completed checkpoint was written")
        if cursor + args.batch_size > len(order):
            order = rng.permutation(len(train_indices))
            cursor = 0
        batch = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        optimizer.zero_grad(set_to_none=True)
        y, c, p, context, scale = (tensor[batch] for tensor in train_tensors)
        terms = flow.original_terms(y, c, float(preprocessor.mu), scale, p, context)
        loss = (-terms["log_p_y"]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at update {update}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0).detach())
        optimizer.step()
        if update == 1 or update % args.validation_every == 0 or update == args.max_updates:
            val_terms = _evaluate(flow, validation_tensors, preprocessor, device, args.eval_batch_size)
            val_nll = float(np.mean(-val_terms["log_p_y"]))
            history.append({"update": update, "train_nll": float(loss.detach()), "validation_nll": val_nll, "grad_norm_pre_clip": grad_norm})
            if val_nll < best_value:
                best_value = val_nll
                best_state = copy.deepcopy(flow.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                selected_update = update
                best_training_state = {
                    "update": update,
                    "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                    "batch_order": torch.from_numpy(order.copy()),
                    "batch_cursor": cursor,
                    "torch_rng_state": torch.get_rng_state().clone(),
                }
    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    flow.load_state_dict(best_state)
    prediction = _evaluate(flow, validation_tensors, preprocessor, device, args.eval_batch_size)
    elapsed = time.perf_counter() - start_time
    subset_catalog_id = int(getattr(args, "subset_catalog_id", 4101))
    run_status = getattr(args, "run_status", "COMPLETED_OPTIMIZATION_AUDIT" if args.run_prefix != "prototype" else "COMPLETED_PROTOTYPE")
    validate_checkpoint_contract(checkpoint_contract)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save({
        "state_dict": flow.state_dict(),
        "optimizer_state_dict": best_optimizer_state,
        "training_state": best_training_state,
        "checkpoint_contract": checkpoint_contract,
        "experiment_identity": getattr(args, "experiment_identity", None),
        "model_config": {"context_dim": context_dim, "hidden": hidden_width, "seed": seed, "bound": flow.bound},
        "representation": representation,
        "capacity": capacity,
        "seed": seed,
        "selected_update": selected_update,
        "initialization_hash": initialization_hash,
    }, checkpoint_path)
    np.savez_compressed(
        run_dir / "predictions.npz",
        event_uid=arrays["event_uid"][validation_indices].astype(str),
        split=arrays["split"][validation_indices].astype(str),
        domain=np.asarray(["source_ID"] * len(validation_indices)),
        representation=np.asarray([representation] * len(validation_indices)),
        subset_id=np.asarray([subset_catalog_id] * len(validation_indices), dtype=np.int32),
        n=np.asarray([len(train_indices)] * len(validation_indices), dtype=np.int32),
        capacity=np.asarray([capacity] * len(validation_indices)),
        seed=np.asarray([seed] * len(validation_indices), dtype=np.int32),
        y=arrays["y"][validation_indices],
        c=preprocessor.center(arrays["p_in"][validation_indices], arrays["thickness"][validation_indices], arrays["charge"][validation_indices]),
        mu=np.asarray([preprocessor.mu] * len(validation_indices)),
        s=validation_tensors[4].detach().cpu().numpy(),
        p_in=arrays["p_in"][validation_indices],
        z=prediction["z"],
        b=prediction["b"],
        log_q_z=prediction["log_q_z"],
        log_F_b=prediction["log_f_b"],
        log_p_y=prediction["log_p_y"],
        logCDF_y=prediction["log_cdf_y"],
        logSF_y=prediction["log_sf_y"],
        threshold_Q=np.asarray([args.q99_threshold] * len(validation_indices)),
    )
    checkpoint = {
        "run_id": run_id,
        "status": run_status,
        "representation": representation,
        "capacity": capacity,
        "hidden_width": hidden_width,
        "context_feature_mode": feature_mode,
        "context_dimension": context_dim,
        "seed": seed,
        "subset_catalog_id": subset_catalog_id,
        "training_fraction": float(getattr(args, "sample_fraction", len(train_indices) / 599526.0)),
        "train_events": len(train_indices),
        "validation_events": len(validation_indices),
        "max_updates": args.max_updates,
        "batch_size": args.batch_size,
        "validation_every": args.validation_every,
        "eval_batch_size": args.eval_batch_size,
        "torch_threads": torch.get_num_threads(),
        "experiment_identity": getattr(args, "experiment_identity", None),
        "selected_update": selected_update,
        "learning_rate": args.learning_rate,
        "epochs_at_budget": args.max_updates * args.batch_size / len(train_indices),
        "parameter_count": flow.parameter_count(),
        "initialization_hash": initialization_hash,
        "optimizer_state_present": True,
        "optimizer_state_update": selected_update,
        "preprocessor": preprocessor.to_dict(),
        "checkpoint_contract": checkpoint_contract,
        "elapsed_seconds": elapsed,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "predictions": str(run_dir / "predictions.npz"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
    return checkpoint, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2"))
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--run-prefix", default="prototype", help="stage prefix for run ids and audit outputs")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--capacities", nargs="+", choices=tuple(CAPACITIES), default=list(CAPACITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--a0-only", action="store_true", help="refresh the production A0 report without retraining prototype cells")
    args = parser.parse_args()
    prepare_new_output(args.output)
    # Main-data contract: validation 10%, diagnostic 30%, train 60% of S=1.
    arrays, source_manifest = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_indices = subset_indices(arrays, fraction=0.1, catalog_id=4101)
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    preprocessors = fit_preprocessors(arrays, train_indices)
    args.q99_threshold = float(np.quantile(arrays["y"][train_indices] / arrays["p_in"][train_indices], 0.99, method="linear"))
    a0 = run_a0_checks()
    a0.update({
        "source_schema": "events.jsonl",
        "source_manifest_sha256": sha256_file(args.merged / "manifest.json"),
        "eligible_S1_events": int(len(arrays["y"])),
        "split_counts": {label: int(np.sum(arrays["split"] == label)) for label in ("train", "validation", "diagnostic")},
        "prototype_train_events": int(len(train_indices)),
        "prototype_validation_events": int(len(validation_indices)),
        "subset_catalog_id": 4101,
        "subset_event_uid_sha256": _hash_indices(train_indices, arrays["event_uid"]),
        "learned_center_ood_audit": "DEFERRED_TO_AUTHORIZED_OOD_STAGE; center extrapolation audit is required before any OOD claim",
        "spline_bound_sensitivity": "DEFERRED_TO_AUTHORIZED_SENSITIVITY_STAGE; prototype bound is 8.0",
    })
    (args.output / "audit_A0.json").write_text(json.dumps(a0, indent=2, sort_keys=True), encoding="utf-8")
    if args.a0_only:
        print(json.dumps({"status": a0["status"], "output": str(args.output / "audit_A0.json")}, sort_keys=True))
        return
    (args.output / "implementation_spec.md").write_text(_implementation_spec(preprocessors, args), encoding="utf-8")
    dataset_manifest = {
        "status": "A0_PLUS_PROTOTYPE_ONLY",
        "source_directory": str(args.merged),
        "source_manifest_sha256": a0["source_manifest_sha256"],
        "source_manifest_counts": source_manifest.get("counts", {}),
        "eligible_S1_events": int(len(arrays["y"])),
        "split_counts": a0["split_counts"],
        "prototype_subset": {"catalog_id": 4101, "fraction": 0.1, "events": int(len(train_indices)), "event_uid_sha256": a0["subset_event_uid_sha256"]},
        "validation_events": int(len(validation_indices)),
        "preprocessor_fit_labels": "prototype subset train only",
        "flow_fit_labels": "prototype subset train only",
        "validation_role": "source validation for checkpoint selection and paired prototype scores",
        "old_confirmation_role": "historical/reused development only; not used",
        "forbidden_features": source_manifest.get("forbidden_prediction_features", []),
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True), encoding="utf-8")
    code_hashes = {
        "engine.py": sha256_file(Path(__file__).with_name("engine.py")),
        "run_a0_prototype.py": sha256_file(Path(__file__)),
    }
    model_manifest = {"status": "PLANNED_BEFORE_RUN", "stage": args.run_prefix, "models": [], "centres": {}, "code_hashes": code_hashes}
    for name, pre in preprocessors.items():
        model_manifest["centres"][name] = {"mu": pre.mu, "scale": pre.scale, "alpha": pre.alpha, "coefficient_count": int(len(pre.coefficients)), "fit_labels": "prototype subset train only"}
    scores = {"status": "RUNNING", "stage": args.run_prefix, "runs": [], "failed_runs": [], "paired_comparisons": [], "authorization": "A0_and_prototype_only" if args.run_prefix == "prototype" else "OPTIMIZATION_AUDIT_AUTHORIZED"}
    (args.output / "prototype_scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    for representation in args.representations:
        for capacity in args.capacities:
            for seed in args.seeds:
                try:
                    result, prediction = _train_one(arrays, train_indices, validation_indices, preprocessors[representation], representation, capacity, seed, args, args.output)
                    model_manifest["models"].append({
                        "run_id": result["run_id"], "parameter_count": result["parameter_count"], "seed": seed,
                        "initialization_hash": result["initialization_hash"],
                        "code_hashes": code_hashes,
                        "optimizer": "Adam", "optimizer_seed": seed, "learning_rate": args.learning_rate,
                        "optimizer_state_present": result["optimizer_state_present"],
                        "checkpoint": result["checkpoint"], "selected_update": result["selected_update"],
                    })
                    scores["runs"].append({
                        "run_id": result["run_id"], "representation": representation, "capacity": capacity, "seed": seed,
                        "validation_nll": float(np.mean(-prediction["log_p_y"])),
                        "validation_log_sf_finite_fraction": float(np.mean(np.isfinite(prediction["log_sf_y"]))),
                        "selected_update": result["selected_update"], "max_updates": args.max_updates,
                        "epochs_at_budget": result["epochs_at_budget"], "elapsed_seconds": result["elapsed_seconds"],
                        "parameter_count": result["parameter_count"], "predictions": result["predictions"],
                    })
                    (args.output / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8")
                    (args.output / "prototype_scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
                    print(json.dumps({"completed": result["run_id"], "validation_nll": float(np.mean(-prediction["log_p_y"])), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                except Exception as exc:  # preserve failures as first-class results
                    failure = {"run_id": f"{args.run_prefix}__{representation}__{capacity}__seed{seed}", "error_type": type(exc).__name__, "error": str(exc)}
                    scores["failed_runs"].append(failure)
                    (args.output / "prototype_scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
                    print(json.dumps({"failed": failure}), flush=True)
    by_key = {(item["representation"], item["capacity"], item["seed"]): item for item in scores["runs"]}
    for capacity in CAPACITIES:
        for seed in SEEDS:
            for left, right, label in (("Raw", "Physics", "Raw_minus_Physics"), ("Learned", "Physics", "Learned_minus_Physics"), ("Simple", "Raw", "Simple_minus_Raw")):
                a = by_key.get((left, capacity, seed))
                b = by_key.get((right, capacity, seed))
                if a and b:
                    scores["paired_comparisons"].append({"capacity": capacity, "seed": seed, "contrast": label, "nll_difference": a["validation_nll"] - b["validation_nll"], "left_run": a["run_id"], "right_run": b["run_id"]})
    scores["status"] = ("COMPLETED_A0_PLUS_PROTOTYPE" if args.run_prefix == "prototype" else "COMPLETED_OPTIMIZATION_AUDIT") if not scores["failed_runs"] else "COMPLETED_WITH_FAILURES_PRESERVED"
    scores["future_formal_matrix"] = {"core_fits": 120, "additional_authorized_only": True, "not_run": True}
    (args.output / "prototype_scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    model_manifest["status"] = "COMPLETED_A0_PLUS_PROTOTYPE" if args.run_prefix == "prototype" else "COMPLETED_OPTIMIZATION_AUDIT"
    (args.output / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8")
    second_learning_rate_record = {
        "status": "EXECUTED_SYMMETRIC_24_CELL_AUDIT",
        "learning_rate": 1.0e-3,
        "symmetric_cells": len(scores["runs"]),
    } if args.run_prefix == "optimization_lr1e-3" else {
        "status": "NOT_EXECUTED_IN_THIS_RUN",
        "note": "The authorized second-learning-rate audit is stored separately; no extra cells were started here.",
    }
    compute_budget = {
        "status": "MEASURED_FROM_PROTOTYPE",
        "batch_size": args.batch_size,
        "max_updates": args.max_updates,
        "epochs_by_training_fraction": {"0.1": args.max_updates * args.batch_size / len(train_indices), "0.3": args.max_updates * args.batch_size / round(0.3 * int(np.sum(arrays["split"] == "train"))), "1.0": args.max_updates * args.batch_size / int(np.sum(arrays["split"] == "train"))},
        "runs_completed": len(scores["runs"]), "runs_failed": len(scores["failed_runs"]),
        "seconds_per_run": {item["run_id"]: item["elapsed_seconds"] for item in scores["runs"]},
        "mean_seconds_per_run": float(np.mean([item["elapsed_seconds"] for item in scores["runs"]])) if scores["runs"] else None,
        "future_core_fits": 120, "future_subset_repetitions": 80, "future_input_only_controls": 20, "future_continuations": 20,
        "matched_epoch_endpoint": "PLANNED_NOT_RUN; endpoint updates must be chosen to match the full-n epoch count to the fixed 10% updates budget",
        "second_learning_rate": second_learning_rate_record,
    }
    (args.output / "compute_budget.json").write_text(json.dumps(compute_budget, indent=2, sort_keys=True), encoding="utf-8")


def _implementation_spec(preprocessors, args):
    return f"""# A0 production implementation and prototype specification

Stage: `{args.run_prefix}`. This run uses {', '.join(args.representations)} ×
{', '.join(args.capacities)} × seeds {', '.join(str(seed) for seed in args.seeds)}.
The 120 core fits, extra subset experiments, OOD data generation, spline-bound
sensitivity and the matched-epoch endpoint are not executed by this run.

The target is `Y = p_in - p_out` in GeV/c for S=1 primary transitions, with
support `Y < p_in`. Common features are `u=(p_in-0.65)/0.35`,
`v=(thickness_mm-0.3)/0.7`, charge, `log(p_in/0.65)`, and a reserved zero
physics feature. Every representation uses the same feature map, two-layer
monotone rational-quadratic spline, 16 bins per layer, exact identity tails
outside [-8, 8], standard Logistic base, Adam, batch {args.batch_size}, and
learning rate {args.learning_rate:g}.

The standardized target is `z=(Y-c(X)-mu_c)/s`, with `s=std_train(Y)` shared
across representations and `mu_c=mean_train(Y-c(X))`. The reported original
unit density is `log p(Y|X) = log q(z|X) - log(s) - log F_q(b|X)`, where
`b=(p_in-c(X)-mu_c)/s`. CDF and survival use the same normalized flow CDF;
survival is evaluated by stable log-difference, never by `1-CDF`.

Centers are Raw=0, Simple=OLS on `[1,u,v,charge]`, Physics=the documented
unit-audited Silicon pion Bethe-Bloch nominal loss, and Learned=20-term cubic
polynomial Ridge with a fixed train-only three-fold alpha grid. The Ridge
center is retained as the current control. Its OOD extrapolation audit is a
required later-stage artifact and is explicitly not claimed by this run.

Training: fixed nested catalog 4101, 10% of source train, batch {args.batch_size},
learning rate {args.learning_rate:g}, and {args.max_updates} updates.
"""


if __name__ == "__main__":
    main()
