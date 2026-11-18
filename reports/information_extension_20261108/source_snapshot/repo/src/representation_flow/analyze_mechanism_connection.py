"""Connect residual location complexity to frozen-flow NLL components."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from representation_flow.engine import fit_preprocessors, load_source_arrays


def _edges(values: np.ndarray, count: int = 8) -> np.ndarray:
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, count + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "values": [float(v) for v in x],
    }


def _moment_audit(arrays: dict) -> tuple[dict, dict]:
    train = np.flatnonzero(arrays["split"] == "train")
    pre = fit_preprocessors(arrays, train)
    p = arrays["p_in"][train].astype(float)
    t = arrays["thickness"][train].astype(float)
    q = arrays["charge"][train].astype(float)
    y = arrays["y"][train].astype(float)
    centers = {name: pre[name].center(p, t, q).astype(float) for name in ("Raw", "Physics")}
    residuals = {name: y - centers[name] for name in centers}
    rows = []
    by_w = {}
    for thickness in sorted(np.unique(t)):
        for charge in sorted(np.unique(q)):
            w = (t == thickness) & (q == charge)
            p_w = p[w]
            p_edges = _edges(p_w, 8)
            p_bins = np.digitize(p_w, p_edges[1:-1], right=False)
            entry = {
                "count": int(np.sum(w)),
                "p_edges": [float(v) for v in p_edges],
                "representations": {},
                "bins": {},
            }
            for name, target in (("Raw", y[w]), ("Physics", residuals["Physics"][w])):
                grand = float(np.mean(target))
                total = float(np.var(target))
                within = 0.0
                between = 0.0
                summaries = {}
                for bin_id in range(8):
                    mask = p_bins == bin_id
                    values = target[mask]
                    weight = float(np.mean(mask))
                    mean = float(np.mean(values))
                    var = float(np.var(values))
                    within += weight * var
                    between += weight * (mean - grand) ** 2
                    summaries[str(bin_id)] = {
                        "count": int(np.sum(mask)),
                        "p_mean": float(np.mean(p_w[mask])),
                        "mean": mean,
                        "variance": var,
                    }
                    rows.append({
                        "stratum": f"t{thickness:g}__q{charge:g}",
                        "bin": int(bin_id),
                        "representation": name,
                        "mean": mean,
                        "within_variance": var,
                        "weight": weight,
                    })
                entry["representations"][name] = {
                    "grand_mean": grand,
                    "total_variance": total,
                    "expected_within_variance": within,
                    "between_mean_variance": between,
                    "variance_identity_residual": total - within - between,
                    "bins": summaries,
                }
            for bin_id in range(8):
                entry["bins"][str(bin_id)] = {
                    "p_mean": float(np.mean(p_w[p_bins == bin_id])),
                    "location_complexity_excess": (
                        entry["representations"]["Physics"]["bins"][str(bin_id)]["mean"]
                        - entry["representations"]["Physics"]["grand_mean"]
                    ) ** 2
                    - (
                        entry["representations"]["Raw"]["bins"][str(bin_id)]["mean"]
                        - entry["representations"]["Raw"]["grand_mean"]
                    ) ** 2,
                }
            key = f"t{thickness:g}__q{charge:g}"
            by_w[key] = entry
    return {"train_events": int(len(train)), "strata": by_w}, {"rows": rows}


def _load_validation_meta(arrays: dict) -> dict[str, tuple[float, float, float]]:
    validation = np.flatnonzero(arrays["split"] == "validation")
    return {
        str(arrays["event_uid"][idx]): (
            float(arrays["p_in"][idx]),
            float(arrays["thickness"][idx]),
            float(arrays["charge"][idx]),
        )
        for idx in validation
    }


def _flow_audit(root: Path, arrays: dict, moments: dict) -> dict:
    scores = json.loads((root / "exploratory_scores.json").read_text(encoding="utf-8"))
    meta = _load_validation_meta(arrays)
    grouped = {
        (row["context_label"], row["fraction_tag"], row["representation"], row["capacity"], row["seed"]): row
        for row in scores["runs"]
    }
    per_group = defaultdict(lambda: {"nll_gap": [], "base_gap": [], "support_gap": [], "reconstruction_error": [], "support_abs": []})
    per_fraction = defaultdict(lambda: {"nll_gap": [], "base_gap": [], "support_gap": [], "reconstruction_error": [], "support_abs": []})
    support_ranges = defaultdict(list)
    support_nonzero = defaultdict(int)
    max_reconstruction_error = defaultdict(float)
    checkpoint_count = 0
    for fraction in ("f010", "f100"):
        for seed in (3101, 3102, 3103, 3104, 3105):
            raw_row = grouped[("bottleneck", fraction, "Raw", "Small", seed)]
            phys_row = grouped[("bottleneck", fraction, "Physics", "Small", seed)]
            raw = np.load(root / "runs" / raw_row["run_id"] / "predictions.npz")
            phys = np.load(root / "runs" / phys_row["run_id"] / "predictions.npz")
            if not np.array_equal(raw["event_uid"], phys["event_uid"]):
                raise ValueError(f"UID mismatch for {fraction}/seed{seed}")
            checkpoint_count += 2
            uids = raw["event_uid"].astype(str)
            p = np.asarray([meta[uid][0] for uid in uids])
            thickness = np.asarray([meta[uid][1] for uid in uids])
            charge = np.asarray([meta[uid][2] for uid in uids])
            nll_raw = -raw["log_p_y"].astype(float)
            nll_phys = -phys["log_p_y"].astype(float)
            base_raw = -raw["log_q_z"].astype(float) + np.log(raw["s"].astype(float))
            base_phys = -phys["log_q_z"].astype(float) + np.log(phys["s"].astype(float))
            support_raw = raw["log_F_b"].astype(float)
            support_phys = phys["log_F_b"].astype(float)
            nll_gap = nll_raw - nll_phys
            base_gap = base_raw - base_phys
            support_gap = support_raw - support_phys
            reconstruction = nll_gap - (base_gap + support_gap)
            support_abs = np.maximum(np.abs(support_raw), np.abs(support_phys))
            for field, values in (("nll_gap", nll_gap), ("base_gap", base_gap), ("support_gap", support_gap), ("reconstruction_error", reconstruction), ("support_abs", support_abs)):
                per_fraction[fraction][field].append(float(np.mean(values)))
            support_ranges[fraction].extend([
                float(np.min(support_raw)), float(np.max(support_raw)),
                float(np.min(support_phys)), float(np.max(support_phys)),
            ])
            support_nonzero[fraction] += int(np.count_nonzero(support_abs))
            max_reconstruction_error[fraction] = max(max_reconstruction_error[fraction], float(np.max(np.abs(reconstruction))))
            for key_w, stratum in moments["strata"].items():
                thickness_w = float(key_w.split("__q")[0][1:])
                charge_w = float(key_w.split("__q")[1])
                mask_w = (thickness == thickness_w) & (charge == charge_w)
                bins_w = np.digitize(p[mask_w], np.asarray(stratum["p_edges"])[1:-1], right=False)
                for bin_id in range(8):
                    mask = mask_w.copy()
                    mask[mask_w] = bins_w == bin_id
                    if not np.any(mask):
                        continue
                    key = f"{fraction}__{key_w}__bin{bin_id}"
                    for field, values in (("nll_gap", nll_gap), ("base_gap", base_gap), ("support_gap", support_gap), ("reconstruction_error", reconstruction), ("support_abs", support_abs)):
                        per_group[key][field].append(float(np.mean(values[mask])))
    aggregate = {"fractions": {fraction: {field: _stats(values) for field, values in data.items()} for fraction, data in per_fraction.items()}, "groups": {key: {field: _stats(values) for field, values in data.items()} for key, data in per_group.items()}, "checkpoint_count": checkpoint_count, "support_ranges": {key: {"min": float(np.min(vals)), "max": float(np.max(vals)), "max_abs": float(np.max(np.abs(vals)))} for key, vals in support_ranges.items()}, "support_nonzero_count": dict(support_nonzero), "max_abs_reconstruction_error": dict(max_reconstruction_error)}
    return aggregate


def _connect(moments: dict, flow: dict) -> dict:
    complexity = []
    nll = []
    for key, entry in moments["strata"].items():
        for bin_id, bin_entry in entry["bins"].items():
            complexity.append(float(bin_entry["location_complexity_excess"]))
            values = []
            for fraction in ("f010", "f100"):
                group = flow["groups"].get(f"{fraction}__{key}__bin{bin_id}")
                if group:
                    values.extend(group["nll_gap"]["values"])
            nll.append(float(np.mean(values)) if values else float("nan"))
    mask = np.isfinite(complexity) & np.isfinite(nll)
    x = np.asarray(complexity)[mask]
    y = np.asarray(nll)[mask]
    return {
        "group_count": int(x.size),
        "pearson_location_complexity_vs_nll_gap": float(pearsonr(x, y).statistic) if x.size > 2 else float("nan"),
        "pearson_pvalue": float(pearsonr(x, y).pvalue) if x.size > 2 else float("nan"),
        "spearman_location_complexity_vs_nll_gap": float(spearmanr(x, y).statistic) if x.size > 2 else float("nan"),
        "spearman_pvalue": float(spearmanr(x, y).pvalue) if x.size > 2 else float("nan"),
        "interpretation": "Descriptive paired association only; it is not a causal estimate of NLL mechanism.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--root", type=Path, default=Path("reports/representation_v2/exploratory/context_interaction_long"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/mechanism_connection_audit.json"))
    args = parser.parse_args()
    arrays, _ = load_source_arrays(args.merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    moments, _ = _moment_audit(arrays)
    flow = _flow_audit(args.root, arrays, moments)
    result = {
        "status": "MECHANISM_CONNECTION_AUDIT_COMPLETE",
        "protocol": {"root": str(args.root), "context": "bottleneck", "capacity": "Small", "fractions": ["f010", "f100"], "seeds": [3101, 3102, 3103, 3104, 3105], "no_new_training": True},
        "variance_decomposition": moments,
        "flow_likelihood_decomposition": flow,
    }
    result["connection_summary"] = _connect(moments, flow)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "group_count": result["connection_summary"]["group_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
