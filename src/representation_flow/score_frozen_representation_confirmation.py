"""Score frozen Full/Hidden Raw/Physics flows on legacy 500k blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from representation_flow.engine import ConditionalSplineFlow, context_features, load_source_arrays, sha256_file
from representation_flow.checkpoint_contract import load_checkpoint, REPOSITORY_ROOT


def _as_tensors(arrays: dict, indices: np.ndarray, preprocessor, device: torch.device, feature_mode: str):
    p = arrays["p_in"][indices]
    t = arrays["thickness"][indices]
    q = arrays["charge"][indices]
    y = arrays["y"][indices]
    c = preprocessor.center(p, t, q)
    scale = preprocessor.scale_for(p, t, q)
    return (
        torch.as_tensor(y, dtype=torch.float32, device=device),
        torch.as_tensor(c, dtype=torch.float32, device=device),
        torch.as_tensor(p, dtype=torch.float32, device=device),
        torch.as_tensor(context_features(p, t, q, feature_mode), dtype=torch.float32, device=device),
        torch.as_tensor(scale, dtype=torch.float32, device=device),
    )


def _edges(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, 9))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _score_one(flow, tensors, preprocessor, batch_size: int, arrays: dict, indices: np.ndarray, train_arrays: dict, train_idx: np.ndarray) -> dict:
    y, c, p, context, scale = tensors
    n = len(y)
    sums = {key: 0.0 for key in ("nll", "log_q_z", "log_s", "log_F_b")}
    support_nonzero = 0
    per_group = {}
    p_np = arrays["p_in"][indices].astype(float)
    t_np = arrays["thickness"][indices].astype(float)
    q_np = arrays["charge"][indices].astype(float)
    main_p = train_arrays["p_in"][train_idx].astype(float)
    main_t = train_arrays["thickness"][train_idx].astype(float)
    main_q = train_arrays["charge"][train_idx].astype(float)
    group_bins = {}
    for thickness in sorted(np.unique(t_np)):
        for charge in sorted(np.unique(q_np)):
            key = f"t{thickness:g}__q{charge:g}"
            group_bins[key] = _edges(main_p[(main_t == thickness) & (main_q == charge)])
    groups = {f"{key}__bin{bin_id}": {"n": 0, "sum_nll": 0.0, "sum_log_q_z": 0.0, "sum_log_s": 0.0, "sum_log_F_b": 0.0} for key in group_bins for bin_id in range(8)}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            sl = slice(start, stop)
            terms = flow.original_terms(y[sl], c[sl], float(preprocessor.mu), scale[sl], p[sl], context[sl])
            log_q = terms["log_q_z"].detach().cpu().numpy().astype(float)
            log_s = torch.log(scale[sl]).detach().cpu().numpy().astype(float)
            log_f = terms["log_f_b"].detach().cpu().numpy().astype(float)
            nll = (-terms["log_p_y"]).detach().cpu().numpy().astype(float)
            sums["nll"] += float(np.sum(nll))
            sums["log_q_z"] += float(np.sum(log_q))
            sums["log_s"] += float(np.sum(log_s))
            sums["log_F_b"] += float(np.sum(log_f))
            support_nonzero += int(np.count_nonzero(log_f))
            p_batch, t_batch, q_batch = p_np[start:stop], t_np[start:stop], q_np[start:stop]
            for thickness in sorted(np.unique(t_batch)):
                for charge in sorted(np.unique(q_batch)):
                    key_w = f"t{thickness:g}__q{charge:g}"
                    local_w = (t_batch == thickness) & (q_batch == charge)
                    bins = np.digitize(p_batch[local_w], group_bins[key_w][1:-1], right=False)
                    local_idx = np.flatnonzero(local_w)
                    for bin_id in range(8):
                        mask = bins == bin_id
                        if not np.any(mask):
                            continue
                        idx = local_idx[mask]
                        group = groups[f"{key_w}__bin{bin_id}"]
                        group["n"] += int(idx.size)
                        group["sum_nll"] += float(np.sum(nll[idx]))
                        group["sum_log_q_z"] += float(np.sum(log_q[idx]))
                        group["sum_log_s"] += float(np.sum(log_s[idx]))
                        group["sum_log_F_b"] += float(np.sum(log_f[idx]))
    summary = {key: {"n": value["n"], "mean_nll": value["sum_nll"] / value["n"], "mean_log_q_z": value["sum_log_q_z"] / value["n"], "mean_log_s": value["sum_log_s"] / value["n"], "mean_log_F_b": value["sum_log_F_b"] / value["n"]} for key, value in groups.items() if value["n"]}
    return {"events": n, "mean_nll": sums["nll"] / n, "mean_log_q_z": sums["log_q_z"] / n, "mean_log_s": sums["log_s"] / n, "mean_log_F_b": sums["log_F_b"] / n, "support_nonzero_count": support_nonzero, "groups": summary}


def _checkpoint_path(row, manifest_root):
    path = Path(row["checkpoint"])
    if path.is_absolute():
        return path
    candidates = [REPOSITORY_ROOT / path, manifest_root / path]
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not existing or len(set(existing)) != 1:
        raise ValueError(f"missing or ambiguous checkpoint path: {path}")
    return existing[0]


def prepare_frozen_models(formal_root, hidden_root, physics_representation, *, allow_legacy=False, legacy_records=None):
    """Validate every frozen model before any confirmation observations are read."""
    manifests = [
        (formal_root, "formal_model_manifest.json", "full"),
        (hidden_root, "exploratory_model_manifest.json", "hidden"),
    ]
    prepared = []
    for root, name, context_label in manifests:
        manifest = json.loads((root / name).read_text(encoding="utf-8"))
        for original in manifest["models"]:
            if original["capacity"] != "Small" or original["representation"] not in {"Raw", physics_representation}:
                continue
            fraction = float(original.get("sample_fraction", .1 if original.get("fraction_tag") == "f010" else 1.0))
            if fraction not in (.1, 1.):
                continue
            if context_label == "hidden" and original.get("context_label") not in ("bottleneck", "hidden"):
                continue
            row = {**original, "context_label": context_label, "fraction_tag": f"f{int(fraction*100):03d}"}
            feature_mode = "full" if context_label == "full" else "thickness_charge"
            checkpoint = _checkpoint_path(row, root)
            legacy = (legacy_records or {}).get(row["run_id"], {})
            if allow_legacy and legacy and legacy.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError("legacy preprocessing sidecar does not identify this checkpoint")
            payload, pre = load_checkpoint(checkpoint, allow_legacy=allow_legacy,
                legacy_preprocessor=legacy.get("preprocessor"),
                legacy_context_feature_mode=legacy.get("context_feature_mode"),
                expected_sha256=row.get("checkpoint_sha256"))
            contract = payload.get("checkpoint_contract")
            stored_mode = contract["context_feature_mode"] if contract else payload["legacy_context_feature_mode"]
            if stored_mode != feature_mode or pre.name != row["representation"]:
                raise ValueError("model manifest disagrees with frozen checkpoint identity")
            if contract is not None and contract["eligibility"] != "forward":
                raise ValueError("this scorer requires the frozen forward-exit contract")
            seed = int(row.get("seed", row.get("optimizer_seed", 3101)))
            if int(payload.get("seed", seed)) != seed or payload.get("capacity", row["capacity"]) != row["capacity"]:
                raise ValueError("model manifest seed/capacity differs from checkpoint")
            if "selected_update" in row and int(row["selected_update"]) != int(payload["selected_update"]):
                raise ValueError("model manifest selected update differs from checkpoint")
            config = payload.get("model_config", {"context_dim": 5 if feature_mode == "full" else 2, "hidden": 32, "seed": seed})
            if int(config.get("seed", seed)) != seed:
                raise ValueError("model config seed differs from checkpoint identity")
            flow = ConditionalSplineFlow(**config)
            flow.load_state_dict(payload["state_dict"])
            flow.eval()
            prepared.append({"row": row, "checkpoint": checkpoint, "flow": flow,
                "pre": pre, "feature_mode": feature_mode, "seed": seed,
                "selected_update": int(payload["selected_update"]),
                "contract_sha256": None if contract is None else contract["sha256"],
                "legacy_audit": bool(payload.get("legacy_audit", False))})
    expected = {(context, fraction, seed, rep) for context in ("full", "hidden")
                for fraction in ("f010", "f100") for seed in (3101, 3102, 3103, 3104, 3105)
                for rep in ("Raw", physics_representation)}
    keys = [(item["row"]["context_label"], item["row"]["fraction_tag"], item["seed"], item["pre"].name) for item in prepared]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("frozen model matrix is incomplete, duplicated or has unexpected seed identities")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-representation", choices=("Physics", "HistoricalIncorrectPhysics", "CorrectedPhysics"), default="CorrectedPhysics")
    parser.add_argument("--allow-legacy", action="store_true", help="historical audit only; requires a hash-bound preprocessing sidecar")
    parser.add_argument("--legacy-preprocessors", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; frozen evidence is never overwritten")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.allow_legacy != (args.legacy_preprocessors is not None):
        parser.error("--allow-legacy and --legacy-preprocessors must be provided together")
    legacy_records = json.loads(args.legacy_preprocessors.read_text(encoding="utf-8")) if args.legacy_preprocessors else {}
    models = prepare_frozen_models(args.formal_root, args.hidden_root, args.physics_representation,
        allow_legacy=args.allow_legacy, legacy_records=legacy_records)
    device = torch.device("cpu")
    main_arrays, _ = load_source_arrays(args.main, validation_fraction=.1, diagnostic_fraction=.3)
    main_train = np.flatnonzero(main_arrays["split"] == "train")
    scores = []
    for block_label in ("a", "b"):
        block_path = args.data_root / f"confirmation_500k_{block_label}_full_merged"
        arrays, manifest = load_source_arrays(block_path, validation_fraction=.1, diagnostic_fraction=.3)
        all_idx = np.arange(len(arrays["event_uid"]), dtype=int)
        for model in models:
            row, pre = model["row"], model["pre"]
            tensors = _as_tensors(arrays, all_idx, pre, device, model["feature_mode"])
            result = _score_one(model["flow"], tensors, pre, args.batch_size, arrays, all_idx, main_arrays, main_train)
            scores.append({"block": block_label, "block_path": str(block_path),
                "block_manifest_sha256": sha256_file(block_path / "manifest.json"),
                "representation": pre.name, "center_identity": pre.center_identity,
                "context_label": row["context_label"], "fraction_tag": row["fraction_tag"],
                "seed": model["seed"], "selected_update": model["selected_update"],
                "checkpoint": str(model["checkpoint"]), "checkpoint_sha256": sha256_file(model["checkpoint"]),
                "checkpoint_contract_sha256": model["contract_sha256"], "legacy_audit": model["legacy_audit"], **result})
            print(json.dumps({"block": block_label, "representation": pre.name, "seed": model["seed"], "mean_nll": result["mean_nll"]}), flush=True)
    paired = {}
    grouped = {(row["block"], row["context_label"], row["fraction_tag"], row["seed"], row["representation"]): row for row in scores}
    for block in ("a", "b"):
        for context in ("full", "hidden"):
            for fraction in ("f010", "f100"):
                values, components = [], []
                for seed in (3101, 3102, 3103, 3104, 3105):
                    raw = grouped[(block, context, fraction, seed, "Raw")]
                    phys = grouped[(block, context, fraction, seed, args.physics_representation)]
                    gap = raw["mean_nll"] - phys["mean_nll"]
                    values.append(gap)
                    support_gap = raw["mean_log_F_b"] - phys["mean_log_F_b"]
                    components.append({"nll_gap": gap, "base_gap": gap-support_gap, "support_gap": support_gap})
                paired[f"{block}__{context}__{fraction}"] = {"values": values, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "components": components}
    result = {"status": "LEGACY_FROZEN_CHECKPOINT_AUDIT_COMPLETE" if args.allow_legacy else "FROZEN_REPRESENTATION_CONFIRMATION_COMPLETE",
        "protocol": {"new_training": False, "new_model_selection": False, "models": len(models),
            "blocks": ["a", "b"], "source_main": str(args.main), "eligibility": "forward",
            "physics_representation": args.physics_representation,
            "provenance": "legacy W6/MDN confirmation reuse; checkpoint identities validated before loading blocks"},
        "scores": scores, "paired": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
