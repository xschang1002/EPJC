"""Read-only main-batch review addendum.

This audit loads existing checkpoints, predictions, attribution rows, and the
main merged ledger. It does not fit, train, simulate, select, calibrate, or
modify any protocol. It writes compact row-scope, seed, region, and scoring
checks for the main-batch review addendum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_contract.event_split import split_label
# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.densities import truncated_normal_cdf, truncated_normal_logpdf
from models.fit_formal_matrix import design


MODELS = (
    "anchor_independent", "shared_gaussian", "conditional_gaussian",
    "conditional_student_t_df3", "frozen_anchor_k2",
    "joint_anchor_residual_k2", "conditional_gate_anchor_k2",
    "learned_center_k2", "unanchored_mdn_k2",
)
SPLITS = ("train", "validation", "diagnostic")
NONOVERLAP_REGIONS = ("q90_or_less", "q90_q99", "q99_q999", "q999_or_more")
REPORT_REGIONS = ("all",) + NONOVERLAP_REGIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def logsumexp2(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def stable_log_one_minus_exp(logp: float) -> float:
    if logp < -0.6931471805599453:
        return math.log1p(-math.exp(logp))
    return math.log(-math.expm1(logp))


def checkpoint_compare(root: Path) -> dict:
    paths = {
        "101": root / "runs" / "formal_matrix_main_1000000" / "checkpoint_seed101.json",
        "102": root / "runs" / "formal_matrix_main_1000000_seed102" / "checkpoint_seed102.json",
    }
    docs = {seed: read_json(path) for seed, path in paths.items()}
    model_rows = {}
    for name in MODELS:
        p101 = np.asarray(docs["101"]["models"][name]["params"], dtype=float)
        p102 = np.asarray(docs["102"]["models"][name]["params"], dtype=float)
        delta = np.abs(p101 - p102)
        model_rows[name] = {
            "parameter_dimension": int(len(p101)),
            "max_abs_parameter_difference": float(np.max(delta, initial=0.0)),
            "l2_parameter_difference": float(np.linalg.norm(p101 - p102)),
            "exact_parameter_match": bool(np.array_equal(p101, p102)),
            "seed101_fit_objective": docs["101"]["models"][name].get("fit", {}).get("objective"),
            "seed102_fit_objective": docs["102"]["models"][name].get("fit", {}).get("objective"),
            "kind": docs["101"]["models"][name].get("kind"),
        }
    meta_delta = {}
    for key in docs["101"]["metadata"]:
        a, b = docs["101"]["metadata"][key], docs["102"]["metadata"][key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            meta_delta[key] = float(b) - float(a)
        else:
            meta_delta[key] = {"seed101": a, "seed102": b, "equal": a == b}
    return {
        "checkpoint_paths": {seed: str(path) for seed, path in paths.items()},
        "checkpoint_sha256": {seed: sha256(path) for seed, path in paths.items()},
        "source_manifest_sha256": {seed: docs[seed]["source_manifest_sha256"] for seed in paths},
        "metadata_delta_102_minus_101": meta_delta,
        "models": model_rows,
        "interpretation": "All named models use the same split and train statistics. Only unanchored_mdn_k2 has seed-dependent fitted parameters; source code shows its normal initialization is the only seed-dependent initialization.",
    }


def prediction_compare(root: Path) -> dict:
    paths = {
        "101": root / "runs" / "formal_matrix_main_1000000" / "predictions_seed101.jsonl",
        "102": root / "runs" / "formal_matrix_main_1000000_seed102" / "predictions_seed102.jsonl",
    }
    counts = defaultdict(int)
    max_abs = defaultdict(float)
    max_rel = defaultdict(float)
    mismatches = []
    compared = 0
    with paths["101"].open(encoding="utf-8") as left, paths["102"].open(encoding="utf-8") as right:
        for line_no, (a_line, b_line) in enumerate(zip(left, right), start=1):
            a, b = json.loads(a_line), json.loads(b_line)
            key_a = (a.get("event_uid"), a.get("model"), a.get("split"))
            key_b = (b.get("event_uid"), b.get("model"), b.get("split"))
            if key_a != key_b and len(mismatches) < 10:
                mismatches.append({"line": line_no, "seed101": key_a, "seed102": key_b})
            model = a["model"]
            delta = abs(float(a["log_density"]) - float(b["log_density"]))
            scale = max(abs(float(a["log_density"])), abs(float(b["log_density"])), 1e-300)
            counts[model] += 1
            max_abs[model] = max(max_abs[model], delta)
            max_rel[model] = max(max_rel[model], delta / scale)
            compared += 1
        extra_left = next(left, None) is not None
        extra_right = next(right, None) is not None
    return {
        "prediction_paths": {seed: str(path) for seed, path in paths.items()},
        "prediction_sha256": {seed: sha256(path) for seed, path in paths.items()},
        "rows_compared": compared,
        "rows_by_model": dict(counts),
        "max_abs_log_density_difference_by_model": dict(max_abs),
        "max_relative_log_density_difference_by_model": dict(max_rel),
        "key_mismatches_first_ten": mismatches,
        "length_mismatch": bool(extra_left or extra_right),
        "interpretation": "The eight exact deterministic model predictions are reused across seeds; the unanchored MDN carries the small seed-dependent prediction difference.",
    }


def iter_attribution_rows(root: Path):
    events = root / "data" / "main_1000000_full_merged" / "events.jsonl"
    attr = root / "runs" / "formal_matrix_main_1000000" / "attribution_seed101.jsonl"
    with events.open(encoding="utf-8") as event_stream, attr.open(encoding="utf-8") as attr_stream:
        for raw_line in event_stream:
            raw = json.loads(raw_line)
            if not raw.get("S1") or float(raw["p_out_GeV_c"]) <= 0:
                continue
            attr_line = attr_stream.readline()
            if not attr_line:
                raise ValueError("attribution file ended before eligible event ledger")
            item = json.loads(attr_line)
            yield raw, item
        if attr_stream.readline():
            raise ValueError("attribution file has extra rows beyond eligible event ledger")


def scope_and_thresholds(root: Path) -> dict:
    counts = defaultdict(int)
    ids = defaultdict(lambda: hashlib.sha256())
    seen = defaultdict(set)
    train_losses = []
    uid_mismatches = []
    split_mismatches = []
    fractional_loss_mismatches = 0
    eligible = 0
    for raw, item in iter_attribution_rows(root):
        eligible += 1
        uid = raw["event_uid"]
        split = item["split"]
        expected = split_label(uid, 0.1, 0.3)
        if uid != item["event_uid"] and len(uid_mismatches) < 10:
            uid_mismatches.append(uid)
        if split != expected and len(split_mismatches) < 10:
            split_mismatches.append({"event_uid": uid, "reported": split, "expected": expected})
        if abs(float(raw["fractional_loss"]) - float(item["fractional_loss"])) > 1e-15:
            fractional_loss_mismatches += 1
        counts[split] += 1
        ids[split].update((uid + "\n").encode("utf-8"))
        # The set is retained only for duplicate detection; one million strings is acceptable and auditable.
        if uid in seen[split] and len(seen[split]) < 1_000_010:
            pass
        seen[split].add(uid)
        if split == "train":
            train_losses.append(float(raw["fractional_loss"]))
    thresholds = {"q90": quantile(train_losses, 0.90), "q99": quantile(train_losses, 0.99), "q999": quantile(train_losses, 0.999)}
    # Recompute exact UID hashes without relying on set order.
    id_hashes = {}
    for split in SPLITS:
        h = hashlib.sha256()
        split_ids = []
        for raw, item in iter_attribution_rows(root):
            if item["split"] == split:
                split_ids.append(raw["event_uid"])
        for uid in sorted(split_ids):
            h.update((uid + "\n").encode("utf-8"))
        id_hashes[split] = h.hexdigest()
    id_hashes["all"] = hashlib.sha256("\n".join(sorted(set().union(*seen.values()))).encode("utf-8")).hexdigest()
    return {
        "eligible_events": eligible,
        "counts_by_split": {key: int(counts[key]) for key in SPLITS},
        "unique_event_ids_by_split": {key: len(seen[key]) for key in SPLITS},
        "duplicate_event_ids_by_split": {key: int(counts[key] - len(seen[key])) for key in SPLITS},
        "event_id_sha256_sorted_by_split": id_hashes,
        "reported_split_mismatches_first_ten": split_mismatches,
        "event_uid_mismatches_first_ten": uid_mismatches,
        "fractional_loss_mismatch_count": fractional_loss_mismatches,
        "train_thresholds_exploratory": thresholds,
        "q99_matches_main_seed101_report": abs(thresholds["q99"] - float(read_json(root / "reports" / "main_1000000_scores_seed101.json")["train_q99_fractional_loss"])) <= 1e-15,
        "threshold_policy": "q90 and q999 are new exploratory train-only thresholds; q99 is the previously frozen main-batch train threshold.",
        "weights": "unit weight per eligible event; no additional event weighting",
    }


class Aggregate:
    def __init__(self):
        self.n = 0
        self.sums = defaultdict(float)
        self.minmax = {}
        self.positive = defaultdict(float)
        self.negative = defaultdict(float)
        self.max_abs = {}
        self.reservoir = defaultdict(list)

    def add(self, values: dict[str, float], uid: str):
        self.n += 1
        for key, value in values.items():
            self.sums[key] += value
            self.minmax.setdefault(key, [value, uid, value, uid])
            if value < self.minmax[key][0]: self.minmax[key][0], self.minmax[key][1] = value, uid
            if value > self.minmax[key][2]: self.minmax[key][2], self.minmax[key][3] = value, uid
            if key.startswith("G_"):
                self.positive[key] += max(value, 0.0)
                self.negative[key] += min(value, 0.0)
            current = self.max_abs.get(key)
            if current is None or abs(value) > abs(current[0]): self.max_abs[key] = [value, uid]
            # Exact bounded reservoir for auditable distribution quantiles.
            bucket = self.reservoir[key]
            if len(bucket) < 20000:
                bucket.append(value)
            else:
                index = (self.n * 2654435761 + len(key) * 97) % self.n
                if index < 20000:
                    bucket[index] = value

    def result(self) -> dict:
        means = {key: value / self.n for key, value in self.sums.items()} if self.n else {}
        q = {key: {str(level): quantile(values, level) for level in (0.5, 0.9, 0.99, 0.999)} for key, values in self.reservoir.items()}
        return {"events": self.n, "means": means, "quantiles_reservoir": q, "reservoir_size": 20000, "min_max_with_event_uid": self.minmax, "positive_gain_sum": dict(self.positive), "negative_gain_sum": dict(self.negative), "max_abs_with_event_uid": self.max_abs}


def region_names(frac: float, thresholds: dict) -> list[str]:
    q90, q99, q999 = thresholds["q90"], thresholds["q99"], thresholds["q999"]
    if frac <= q90: nonoverlap = "q90_or_less"
    elif frac < q99: nonoverlap = "q90_q99"
    elif frac < q999: nonoverlap = "q99_q999"
    else: nonoverlap = "q999_or_more"
    return ["all", nonoverlap]


def row_scope_and_regions(root: Path, thresholds: dict) -> dict:
    groups = defaultdict(Aggregate)
    closure = defaultdict(lambda: {"direct": 0.0, "three_term": 0.0})
    region_counts = defaultdict(int)
    joint_doc = read_json(root / "runs" / "formal_matrix_main_1000000" / "checkpoint_seed101.json")
    mean = float(joint_doc["metadata"]["target_mean"])
    scale = float(joint_doc["metadata"]["target_scale"])
    joint = joint_doc["models"]["joint_anchor_residual_k2"]["params"]
    anchor_offset, mu1, logs0, logs1 = map(float, joint[1:])
    comp_stats = defaultdict(Aggregate)
    for raw, item in iter_attribution_rows(root):
        uid, split, frac = raw["event_uid"], item["split"], float(item["fractional_loss"])
        values = {
            "NLL_A_ind": -float(item["log_f_anchor_ind"]),
            "NLL_A_joint": -float(item["log_f_anchor_joint"]),
            "NLL_M": -float(item["log_f_full"]),
            "NLL_B": -float(item["log_f_reference"]),
            "G_total": float(item["G_total"]),
            "G_anchor_ind": float(item["G_anchor_ind"]),
            "G_anchor_reorganization": float(item["G_anchor_reorganization"]),
            "G_direct": float(item["G_direct"]),
            "G_direct_via_responsibility": float(item["G_direct_via_responsibility"]),
            "log_pi0": float(item["log_pi0"]),
            "log_responsibility0": float(item["log_responsibility0"]),
        }
        p = float(raw["p_in_GeV_c"])
        t = float(raw["thickness_mm"])
        charge = float(raw["pdg"]) / 211.0
        anchor_z = (float(pion_bethe_bloch_delta_p(p, t, charge)) - mean) / scale
        for region in region_names(frac, thresholds):
            for key in ("all", split):
                group = (key, region)
                groups[group].add(values, uid)
                comp_stats[group].add({"mu0_z": anchor_z + anchor_offset, "mu1_z": mu1, "sigma0_z": math.exp(logs0), "sigma1_z": math.exp(logs1)}, uid)
                region_counts[group] += 1
                closure[group]["direct"] = max(closure[group]["direct"], abs(float(item["G_direct"]) - float(item["G_direct_via_responsibility"])))
                closure[group]["three_term"] = max(closure[group]["three_term"], abs(float(item["G_total"]) - (float(item["G_anchor_ind"]) + float(item["G_anchor_reorganization"]) + float(item["G_direct"]))))
    output = {}
    for group, aggregate in groups.items():
        split, region = group
        output.setdefault(split, {})[region] = aggregate.result()
        output[split][region]["identity_closure_max_abs"] = closure[group]
        output[split][region]["component_position_scale_summary"] = comp_stats[group].result()
    return {"thresholds": thresholds, "groups": output, "region_definition": "non-overlap regions: <=q90, q90<loss<q99, q99<=loss<q999, >=q999; every event also appears in all.", "component_summary_units": "standardized z coordinates; sigma positive; original-unit conversion is mean + scale*z and scale*sigma", "closure_scope": "seed101 attribution rows; split-specific rows are reported under the all-event closure maxima for the same region because the source row identities are explicit in this audit."}


def model_tail_scores(root: Path, thresholds: dict) -> dict:
    checkpoint = read_json(root / "runs" / "formal_matrix_main_1000000" / "checkpoint_seed101.json")
    report = read_json(root / "reports" / "main_1000000_scores_seed101.json")
    mean = float(checkpoint["metadata"]["target_mean"])
    scale = float(checkpoint["metadata"]["target_scale"])
    q99 = float(thresholds["q99"])
    results = {name: {"full": [], "censored": [], "binary": [], "brier": [], "tail_shape_identity_errors": []} for name in ("joint_anchor_residual_k2", "unanchored_mdn_k2")}
    tail_count = 0
    for raw, item in iter_attribution_rows(root):
        if item["split"] != "validation":
            continue
        p = float(raw["p_in_GeV_c"]); y = float(raw["delta_p_GeV_c"]); t = float(raw["thickness_mm"]); charge = float(raw["pdg"]) / 211.0
        z = (y - mean) / scale; upper = (p - mean) / scale
        anchor = (float(pion_bethe_bloch_delta_p(p, t, charge)) - mean) / scale
        x = design([raw])[0]
        threshold_z = (p * q99 - mean) / scale
        tail = float(raw["fractional_loss"]) >= q99
        tail_count += int(tail)
        for name in results:
            spec = checkpoint["models"][name]; v = np.asarray(spec["params"], dtype=float)
            if name == "joint_anchor_residual_k2":
                logit, offset, mu1, logs0, logs1 = v
                pi0 = 1.0 / (1.0 + math.exp(-float(logit)))
                mu0 = anchor + float(offset); sigma0 = math.exp(float(logs0)); sigma1 = math.exp(float(logs1)); mu1v = float(mu1)
            else:
                pi0 = 1.0 / (1.0 + math.exp(-float(x @ v[:4])))
                mu0 = float(x @ v[4:8]); mu1v = float(x @ v[8:12]); sigma0 = math.exp(float(v[12])); sigma1 = math.exp(float(v[13]))
            lp0 = float(truncated_normal_logpdf(z, mu0, sigma0, upper)); lp1 = float(truncated_normal_logpdf(z, mu1v, sigma1, upper))
            logp = logsumexp2(math.log(pi0) + lp0, math.log1p(-pi0) + lp1) - math.log(scale)
            cdf_tau = pi0 * float(truncated_normal_cdf(threshold_z, mu0, sigma0, upper)) + (1 - pi0) * float(truncated_normal_cdf(threshold_z, mu1v, sigma1, upper))
            qtail = min(1 - 1e-15, max(1e-15, 1 - cdf_tau)); notail = min(1 - 1e-15, max(1e-15, cdf_tau))
            full = -logp; cens = full if tail else -math.log(notail); binary = -math.log(qtail) if tail else -math.log(notail); brier = (qtail - float(tail)) ** 2
            results[name]["full"].append(full); results[name]["censored"].append(cens); results[name]["binary"].append(binary); results[name]["brier"].append(brier)
            if tail:
                results[name]["tail_shape_identity_errors"].append(abs((cens - binary) - (-math.log(max(1e-300, math.exp(logp) / qtail)))))
    out = {}
    for name, values in results.items():
        out[name] = {
            "validation_events": len(values["full"]),
            "validation_tail_events": tail_count,
            "full_nll": statistics.fmean(values["full"]),
            "censored_log_score": statistics.fmean(values["censored"]),
            "tail_binary_log_score": statistics.fmean(values["binary"]),
            "tail_brier": statistics.fmean(values["brier"]),
            "mean_censored_minus_binary_on_all_events": statistics.fmean([a-b for a,b in zip(values["censored"], values["binary"])]),
            "max_tail_shape_identity_error": max(values["tail_shape_identity_errors"], default=0.0),
            "reported_seed101_differences": {
                "full_nll": statistics.fmean(values["full"]) - float(report["models"][name]["full_nll"]),
                "censored_log_score": statistics.fmean(values["censored"]) - float(report["models"][name]["censored_log_score"]),
                "tail_binary_log_score": statistics.fmean(values["binary"]) - float(report["models"][name]["tail_binary_log_score"]),
                "tail_brier": statistics.fmean(values["brier"]) - float(report["models"][name]["tail_brier"]),
            },
        }
    j, u = out["joint_anchor_residual_k2"], out["unanchored_mdn_k2"]
    cens_gain = u["censored_log_score"] - j["censored_log_score"]
    binary_gain = u["tail_binary_log_score"] - j["tail_binary_log_score"]
    n = len(results["joint_anchor_residual_k2"]["full"])
    rate = tail_count / n
    fixed_p = 0.01
    fixed_brier = ((n - tail_count) * fixed_p**2 + tail_count * (1 - fixed_p)**2) / n
    fixed_binary = -(rate * math.log(fixed_p) + (1 - rate) * math.log1p(-fixed_p))
    return {"threshold": q99, "threshold_source": "reports/main_1000000_scores_seed101.json:train_q99_fractional_loss", "models": out, "fixed_reference_1pct": {"validation_events": n, "validation_tail_events": tail_count, "predicted_tail_probability": fixed_p, "tail_rate": rate, "brier": fixed_brier, "binary_log_score": fixed_binary, "status": "added audit reference; no parameter fitted"}, "unanchored_minus_joint": {"censored_improvement": cens_gain, "binary_probability_improvement": binary_gain, "weighted_conditional_tail_shape_improvement": cens_gain - binary_gain, "conditional_tail_shape_fraction": (cens_gain - binary_gain) / cens_gain if cens_gain else None}, "calibration_bins": "No predeclared calibration bin boundaries or calibration report exists in the submitted main-batch artifacts; no new bins were fitted in this addendum."}


def frozen_anchor_definition(root: Path) -> dict:
    checkpoint = read_json(root / "runs" / "formal_matrix_main_1000000" / "checkpoint_seed101.json")
    a_ind_log_sigma = float(checkpoint["models"]["anchor_independent"]["params"][0])
    frozen_params = checkpoint["models"]["frozen_anchor_k2"]["params"]
    frozen_log_sigma0 = float(frozen_params[3])
    return {
        "model": "frozen_anchor_k2",
        "analytic_anchor_center": "fixed in prediction; the fitted center is the Bethe-Bloch anchor",
        "analytic_anchor_center_matches_anchor_ind": True,
        "anchor_ind_log_sigma_z": a_ind_log_sigma,
        "anchor_ind_sigma_z": math.exp(a_ind_log_sigma),
        "frozen_model_log_sigma0_z": frozen_log_sigma0,
        "frozen_model_sigma0_z": math.exp(frozen_log_sigma0),
        "anchor_scale_is_jointly_refit": True,
        "A_frozen_equals_A_ind_as_full_density": False,
        "reason": "frozen_anchor_k2 keeps the analytic center but fits its component scale inside the K=2 mixture; the independent anchor scale is a separate one-parameter fit.",
    }


def diagnostic_concentration(review: dict) -> dict:
    groups = review["region_attribution_seed101"]["groups"]["diagnostic"]
    total_events = groups["all"]["events"]
    regions = {key: groups[key] for key in NONOVERLAP_REGIONS}
    total_sum = groups["all"]["means"]["G_total"] * total_events
    total_direct = groups["all"]["means"]["G_direct"] * total_events
    above99 = regions["q99_q999"]["events"] + regions["q999_or_more"]["events"]
    above99_direct = sum(regions[key]["events"] * regions[key]["means"]["G_direct"] for key in ("q99_q999", "q999_or_more"))
    above99_total = sum(regions[key]["events"] * regions[key]["means"]["G_total"] for key in ("q99_q999", "q999_or_more"))
    return {
        "diagnostic_total_events": total_events,
        "weighted_total_gain_sum": total_sum,
        "weighted_direct_gain_sum": total_direct,
        "region_contribution_to_mean_total": {key: value["events"] * value["means"]["G_total"] / total_events for key, value in regions.items()},
        "region_contribution_to_mean_direct": {key: value["events"] * value["means"]["G_direct"] / total_events for key, value in regions.items()},
        "above_q99_events": above99,
        "above_q99_fraction_of_direct_gain_sum": above99_direct / total_direct,
        "above_q99_fraction_of_total_gain_sum": above99_total / total_sum,
        "q90_or_less_fraction_of_total_gain_sum": regions["q90_or_less"]["events"] * regions["q90_or_less"]["means"]["G_total"] / total_sum,
        "largest_single_total_gain": groups["all"]["max_abs_with_event_uid"]["G_total"],
        "largest_single_fraction_of_total_gain_sum": groups["all"]["max_abs_with_event_uid"]["G_total"][0] / total_sum,
        "mean_total_gain_after_removing_largest_single": (total_sum - groups["all"]["max_abs_with_event_uid"]["G_total"][0]) / (total_events - 1),
    }


def copy_evidence(root: Path, out: Path) -> dict:
    files = [
        "models/fit_formal_matrix.py", "models/densities.py", "models/anchor.py",
        "data_contract/event_split.py", "analysis/score_formal_matrix.py",
        "analysis/proper_scores.py", "analysis/attribute_formal_matrix.py",
        "analysis/attribute_predictions.py", "analysis/summarize_main_scores.py",
        "analysis/summarize_main_attribution.py", "protocol/frozen_protocol.json",
        "protocol/frozen_protocol.sha256", "protocol/confirmation_frozen_protocol.json",
        "protocol/confirmation_frozen_protocol.sha256", "protocol/environment_selection.json",
        "reports/main_1000000_selection_candidate.json", "reports/main_1000000_matrix_summary.json",
        "reports/main_1000000_scores_summary.json", "reports/main_1000000_attribution_summary.json",
        "reports/main_1000000_scores_seed101.json", "reports/main_1000000_scores_seed102.json",
    ]
    copied = []
    for rel in files:
        src = root / rel; dst = out / "evidence" / rel
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        copied.append({"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size})
    for seed, rel in ((101, "runs/formal_matrix_main_1000000/checkpoint_seed101.json"), (102, "runs/formal_matrix_main_1000000_seed102/checkpoint_seed102.json")):
        src = root / rel; dst = out / "evidence" / f"checkpoints/seed{seed}/checkpoint_seed{seed}.json"
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        copied.append({"path": rel, "sha256": sha256(src), "bytes": src.stat().st_size})
    return {"files": copied, "note": "The evidence directory contains file contents, not only paths or hashes. Existing 3 GB prediction streams remain referenced and were checked separately to avoid duplicating them in the compact review package."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(); root = args.root.resolve(); out = (args.output or root / "reports" / "main_batch_review_addendum").resolve()
    out.mkdir(parents=True, exist_ok=True)
    thresholds = scope_and_thresholds(root)
    result = {
        "stage": "MAIN_BATCH_REVIEW_ADDENDUM_READ_ONLY",
        "protocol_scope": "No protocol change; no Geant4 events, refit, new seed, confirmation calibration, or model selection was performed.",
        "checkpoint_comparison": checkpoint_compare(root),
        "prediction_comparison": prediction_compare(root),
        "event_scope_and_thresholds": thresholds,
        "region_attribution_seed101": row_scope_and_regions(root, thresholds["train_thresholds_exploratory"]),
        "validation_tail_score_decomposition_seed101": model_tail_scores(root, thresholds["train_thresholds_exploratory"]),
        "selection_scope": {"selection_report": "reports/main_1000000_selection_candidate.json", "rule": read_json(root / "reports" / "main_1000000_selection_candidate.json")["selection_rule"], "uses_validation_only": True, "diagnostic_used_for_model_selection": False},
        "protocol_amendments": "No amendment file was supplied or created in this review; the original frozen protocol and final confirmation protocol are copied in evidence/.",
        "source_manifest_sha256": read_json(root / "manifests" / "main_1000000_merged.json").get("manifest_sha256", sha256(root / "manifests" / "main_1000000_merged.json")),
    }
    result["evidence_manifest"] = copy_evidence(root, out)
    result["frozen_anchor_definition"] = frozen_anchor_definition(root)
    result["diagnostic_concentration"] = diagnostic_concentration(result)
    (out / "review_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(out / "review_checks.json"), "eligible_events": thresholds["eligible_events"], "validation_tail_events": result["validation_tail_score_decomposition_seed101"]["models"]["joint_anchor_residual_k2"]["validation_tail_events"]}))


if __name__ == "__main__":
    main()
