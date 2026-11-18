"""Fit the pre-freeze baseline matrix on real Geant4 pilot transitions.

This is a deterministic, auditable matrix runner. It uses only pre-transition
features and train-split statistics, preserves the upper support ``y < p_in``,
and labels its outputs ``PRE_FREEZE_PILOT``. It does not open confirmation
data or change the protocol freeze state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.densities import truncated_normal_logpdf, truncated_student_logpdf
from data_contract.event_split import split_label


MODEL_NAMES = (
    "anchor_independent",
    "shared_gaussian",
    "conditional_gaussian",
    "conditional_student_t_df3",
    "frozen_anchor_k2",
    "joint_anchor_residual_k2",
    "conditional_gate_anchor_k2",
    "learned_center_k2",
    "unanchored_mdn_k2",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("S1") and float(row["p_out_GeV_c"]) > 0:
                rows.append(row)
    return rows


def design(rows: list[dict]) -> np.ndarray:
    p = np.asarray([float(row["p_in_GeV_c"]) for row in rows])
    q = np.asarray([float(row["pdg"]) / 211.0 for row in rows])
    t = np.asarray([float(row["thickness_mm"]) for row in rows])
    return np.column_stack((np.ones(len(rows)), (p - 0.65) / 0.25, q, (t - 0.3) / 0.3))


def split(rows: list[dict], validation_fraction: float, diagnostic_fraction: float = 0.0) -> tuple[list[dict], list[dict], list[dict]]:
    train, validation, diagnostic = [], [], []
    for row in rows:
        label = split_label(row["event_uid"], validation_fraction, diagnostic_fraction)
        {"train": train, "validation": validation, "diagnostic": diagnostic}[label].append(row)
    return train, validation, diagnostic


def fit_opt(fun, initial: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    result = minimize(fun, initial, method="L-BFGS-B", options={"maxiter": 1800, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40})
    if not result.success:
        raise RuntimeError(f"optimizer failed for seed {seed}: {result.message}")
    return np.asarray(result.x), {"success": bool(result.success), "nit": int(result.nit), "objective": float(result.fun)}


def normal_nll(z, upper, mu, log_sigma):
    return -float(np.sum(truncated_normal_logpdf(z, mu, np.exp(log_sigma), upper)))


def student_nll(z, upper, mu, log_sigma):
    return -float(np.sum(truncated_student_logpdf(z, mu, np.exp(log_sigma), upper, df=3.0)))


def fit_models(rows: list[dict], seed: int, validation_fraction: float, diagnostic_fraction: float = 0.0) -> tuple[dict, dict, dict]:
    train, validation, diagnostic = split(rows, validation_fraction, diagnostic_fraction)
    y_train = np.asarray([float(row["delta_p_GeV_c"]) for row in train])
    p_train = np.asarray([float(row["p_in_GeV_c"]) for row in train])
    t_train = np.asarray([float(row["thickness_mm"]) for row in train])
    q_train = np.asarray([float(row["pdg"]) / 211.0 for row in train])
    target_mean = float(np.mean(y_train))
    target_scale = max(float(np.std(y_train)), 1e-9)
    z_train = (y_train - target_mean) / target_scale
    u_train = (p_train - target_mean) / target_scale
    a_train = (pion_bethe_bloch_delta_p(p_train, t_train, q_train) - target_mean) / target_scale
    x_train = design(train)
    rng = np.random.default_rng(seed)

    fitted: dict[str, dict] = {}
    # Analytic anchor center with train-fitted scale.
    anchor_result = minimize_scalar(lambda logs: normal_nll(z_train, u_train, a_train, logs), bounds=(-7.0, 2.0), method="bounded")
    fitted["anchor_independent"] = {"kind": "fixed_anchor", "mu": "anchor", "params": [float(anchor_result.x)], "fit": {"objective": float(anchor_result.fun), "success": bool(anchor_result.success)}}

    shared, info = fit_opt(lambda v: normal_nll(z_train, u_train, v[0], v[1]), np.array([float(np.mean(z_train)), math.log(max(float(np.std(z_train)), 0.05))]), seed)
    fitted["shared_gaussian"] = {"kind": "shared_normal", "params": shared.tolist(), "fit": info}

    cond0 = np.zeros(5)
    cond0[0] = float(np.mean(z_train))
    cond0[4] = math.log(max(float(np.std(z_train)), 0.05))
    cond, info = fit_opt(lambda v: normal_nll(z_train, u_train, x_train @ v[:4], v[4]), cond0, seed)
    fitted["conditional_gaussian"] = {"kind": "conditional_normal", "params": cond.tolist(), "fit": info}
    stud, info = fit_opt(lambda v: student_nll(z_train, u_train, x_train @ v[:4], v[4]), cond0, seed)
    fitted["conditional_student_t_df3"] = {"kind": "conditional_student_t", "df": 3.0, "params": stud.tolist(), "fit": info}

    def mixture_global(v, anchor_mode: str):
        logit, offset, mu1, logs0, logs1 = v
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(logit, -35, 35)))
        mu0 = a_train if anchor_mode == "frozen" else a_train + offset
        l0 = truncated_normal_logpdf(z_train, mu0, np.exp(logs0), u_train)
        l1 = truncated_normal_logpdf(z_train, mu1, np.exp(logs1), u_train)
        return -float(np.sum(np.logaddexp(math.log(pi0) + l0, math.log1p(-pi0) + l1)))

    init = np.array([0.0, 0.0, float(np.quantile(z_train, 0.8)), math.log(0.2), math.log(0.5)])
    frozen, info = fit_opt(lambda v: mixture_global(v, "frozen"), init, seed)
    fitted["frozen_anchor_k2"] = {"kind": "global_anchor_mixture", "params": frozen.tolist(), "fit": info}
    joint, info = fit_opt(lambda v: mixture_global(v, "joint"), init, seed)
    fitted["joint_anchor_residual_k2"] = {"kind": "joint_anchor_residual_mixture", "params": joint.tolist(), "fit": info}

    def conditional_gate(v):
        gate = x_train @ v[:4]
        offset = x_train @ v[4:8]
        residual = x_train @ v[8:12]
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(gate, -35, 35)))
        l0 = truncated_normal_logpdf(z_train, a_train + offset, np.exp(v[12]), u_train)
        l1 = truncated_normal_logpdf(z_train, residual, np.exp(v[13]), u_train)
        return -float(np.sum(np.logaddexp(np.log(pi0) + l0, np.log1p(-pi0) + l1)))

    cinit = np.zeros(14)
    cinit[0] = 0.4
    cinit[8] = float(np.quantile(z_train, 0.8))
    cinit[12:] = math.log(0.3)
    cgate, info = fit_opt(conditional_gate, cinit, seed)
    fitted["conditional_gate_anchor_k2"] = {"kind": "conditional_gate_anchor_mixture", "params": cgate.tolist(), "fit": info}

    def learned_center(v):
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(v[0], -35, 35)))
        center = x_train @ v[1:5]
        residual = x_train @ v[5:9]
        l0 = truncated_normal_logpdf(z_train, center, np.exp(v[9]), u_train)
        l1 = truncated_normal_logpdf(z_train, residual, np.exp(v[10]), u_train)
        return -float(np.sum(np.logaddexp(math.log(pi0) + l0, math.log1p(-pi0) + l1)))

    linit = np.zeros(11)
    linit[0] = 0.0
    linit[1] = float(np.mean(z_train))
    linit[5] = float(np.quantile(z_train, 0.8))
    linit[9:] = math.log(0.3)
    learned, info = fit_opt(learned_center, linit, seed)
    fitted["learned_center_k2"] = {"kind": "matched_learned_center_mixture", "params": learned.tolist(), "fit": info}

    def mdn(v):
        gate = x_train @ v[:4]
        mu0 = x_train @ v[4:8]
        mu1 = x_train @ v[8:12]
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(gate, -35, 35)))
        l0 = truncated_normal_logpdf(z_train, mu0, np.exp(v[12]), u_train)
        l1 = truncated_normal_logpdf(z_train, mu1, np.exp(v[13]), u_train)
        return -float(np.sum(np.logaddexp(np.log(pi0) + l0, np.log1p(-pi0) + l1)))

    minit = np.zeros(14)
    minit[0] = float(rng.normal(0, 0.1))
    minit[4] = float(np.quantile(z_train, 0.2))
    minit[8] = float(np.quantile(z_train, 0.8))
    minit[12:] = math.log(0.3)
    mdn_fit, info = fit_opt(mdn, minit, seed)
    fitted["unanchored_mdn_k2"] = {"kind": "unanchored_mdn_k2", "params": mdn_fit.tolist(), "fit": info}

    arrays = {
        "train": (train, z_train, u_train, a_train, x_train),
        "validation": (validation, (np.asarray([float(row["delta_p_GeV_c"]) for row in validation]) - target_mean) / target_scale,
                        (np.asarray([float(row["p_in_GeV_c"]) for row in validation]) - target_mean) / target_scale,
                        (pion_bethe_bloch_delta_p(np.asarray([float(row["p_in_GeV_c"]) for row in validation]), np.asarray([float(row["thickness_mm"]) for row in validation]), np.asarray([float(row["pdg"]) / 211.0 for row in validation])) - target_mean) / target_scale,
                        design(validation)),
    }
    metadata = {"seed": seed, "train_events": len(train), "validation_events": len(validation), "diagnostic_events": len(diagnostic), "validation_fraction": validation_fraction, "diagnostic_fraction": diagnostic_fraction, "target_mean": target_mean, "target_scale": target_scale}
    return fitted, metadata, arrays


def predict_model(name: str, spec: dict, z: np.ndarray, upper: np.ndarray, anchor: np.ndarray, x: np.ndarray) -> np.ndarray:
    if name == "anchor_independent":
        return truncated_normal_logpdf(z, anchor, np.exp(spec["params"][0]), upper)
    if name == "shared_gaussian":
        return truncated_normal_logpdf(z, spec["params"][0], np.exp(spec["params"][1]), upper)
    if name == "conditional_gaussian":
        return truncated_normal_logpdf(z, x @ np.asarray(spec["params"][:4]), np.exp(spec["params"][4]), upper)
    if name == "conditional_student_t_df3":
        return truncated_student_logpdf(z, x @ np.asarray(spec["params"][:4]), np.exp(spec["params"][4]), upper, df=3.0)
    if name in ("frozen_anchor_k2", "joint_anchor_residual_k2"):
        logit, offset, mu1, logs0, logs1 = spec["params"]
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(logit, -35, 35)))
        mu0 = anchor if name == "frozen_anchor_k2" else anchor + offset
        l0 = truncated_normal_logpdf(z, mu0, np.exp(logs0), upper)
        l1 = truncated_normal_logpdf(z, mu1, np.exp(logs1), upper)
        return np.logaddexp(math.log(pi0) + l0, math.log1p(-pi0) + l1)
    if name == "conditional_gate_anchor_k2":
        v = np.asarray(spec["params"])
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(x @ v[:4], -35, 35)))
        l0 = truncated_normal_logpdf(z, anchor + x @ v[4:8], np.exp(v[12]), upper)
        l1 = truncated_normal_logpdf(z, x @ v[8:12], np.exp(v[13]), upper)
        return np.logaddexp(np.log(pi0) + l0, np.log1p(-pi0) + l1)
    if name == "learned_center_k2":
        v = np.asarray(spec["params"])
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(v[0], -35, 35)))
        l0 = truncated_normal_logpdf(z, x @ v[1:5], np.exp(v[9]), upper)
        l1 = truncated_normal_logpdf(z, x @ v[5:9], np.exp(v[10]), upper)
        return np.logaddexp(math.log(pi0) + l0, math.log1p(-pi0) + l1)
    if name == "unanchored_mdn_k2":
        v = np.asarray(spec["params"])
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(x @ v[:4], -35, 35)))
        l0 = truncated_normal_logpdf(z, x @ v[4:8], np.exp(v[12]), upper)
        l1 = truncated_normal_logpdf(z, x @ v[8:12], np.exp(v[13]), upper)
        return np.logaddexp(np.log(pi0) + l0, np.log1p(-pi0) + l1)
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--diagnostic-fraction", type=float, default=0.0)
    args = parser.parse_args()
    rows = read_rows(args.merged / "events.jsonl")
    if len(rows) < 100:
        raise ValueError("too few eligible transitions")
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        fitted, metadata, arrays = fit_models(rows, seed, args.validation_fraction, args.diagnostic_fraction)
        train, z_train, u_train, a_train, x_train = arrays["train"]
        valid, z_val, u_val, a_val, x_val = arrays["validation"]
        scores = {}
        pred_path = args.output / f"predictions_seed{seed}.jsonl"
        with pred_path.open("w", encoding="utf-8", newline="\n") as stream:
            for name in MODEL_NAMES:
                l_train = predict_model(name, fitted[name], z_train, u_train, a_train, x_train) - math.log(metadata["target_scale"])
                l_val = predict_model(name, fitted[name], z_val, u_val, a_val, x_val) - math.log(metadata["target_scale"])
                scores[name] = {"train_nll": float(-np.mean(l_train)), "validation_nll": float(-np.mean(l_val))}
                for row, logp in zip(valid, l_val):
                    stream.write(json.dumps({"event_uid": row["event_uid"], "seed": seed, "model": name, "split": "validation", "log_density": float(logp), "y_delta_p_GeV_c": row["delta_p_GeV_c"]}, sort_keys=True, allow_nan=False) + "\n")
        checkpoint = {"stage": "PRE_FREEZE_PILOT_MATRIX", "seed": seed, "models": fitted, "metadata": metadata, "source_manifest_sha256": sha256(args.merged / "manifest.json"), "feature_policy": "pre_transition_only"}
        (args.output / f"checkpoint_seed{seed}.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        (args.output / f"scores_seed{seed}.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        all_results.append({"seed": seed, "scores": scores, "optimizer": {name: fitted[name]["fit"] for name in MODEL_NAMES}})
    summary = {"stage": "PRE_FREEZE_PILOT_MATRIX", "source_manifest_sha256": sha256(args.merged / "manifest.json"), "models": list(MODEL_NAMES), "seeds": args.seeds, "results": all_results, "caveat": "Pilot matrix for protocol preparation; no model selection or confirmation claim."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "eligible_events": len(rows), "seeds": args.seeds, "models": len(MODEL_NAMES)}))


if __name__ == "__main__":
    main()
