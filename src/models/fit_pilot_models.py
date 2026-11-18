"""Fit auditable development densities on the real pilot transitions.

This is a small development fit, not a formal confirmation trainer. It keeps
features available before the transition and writes per-row log densities for
the later attribution engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize, minimize_scalar

# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.densities import TrainStandardizer, truncated_normal_logpdf


FEATURES = ("p_in_GeV_c", "pdg", "thickness_mm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["S1"]:
                rows.append(row)
    return rows


def fit_width(y, center, upper):
    scale = max(float(np.std(y)), 1e-8)
    standardizer = TrainStandardizer(float(np.mean(y)), scale)
    z = standardizer.forward(y)
    c = standardizer.forward(center)
    u = standardizer.forward(upper)
    result = minimize_scalar(lambda log_sigma: -float(np.sum(truncated_normal_logpdf(z, c, math.exp(log_sigma), u))), bounds=(-8, 3), method="bounded")
    return standardizer, math.exp(result.x), float(result.fun), bool(result.success)


def fit_reference(y, upper, standardizer):
    z = standardizer.forward(y)
    u = standardizer.forward(upper)
    result = minimize(lambda params: -float(np.sum(truncated_normal_logpdf(z, params[0], math.exp(params[1]), u))), [float(np.mean(z)), math.log(max(float(np.std(z)), 0.1))], method="Nelder-Mead")
    return {"mu_z": float(result.x[0]), "sigma_z": float(math.exp(result.x[1])), "success": bool(result.success)}


def fit_mixture(y, anchor_center, upper, standardizer, seed):
    rng = np.random.default_rng(seed)
    z = standardizer.forward(y)
    a = standardizer.forward(anchor_center)
    u = standardizer.forward(upper)
    init = np.array([0.0, rng.normal(0, 0.02), math.log(0.8), float(np.quantile(z, 0.99)), math.log(2.0)])

    def objective(params):
        logit, offset0, logs0, mu1, logs1 = params
        pi0 = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
        log_a = truncated_normal_logpdf(z, a + offset0, np.exp(logs0), u)
        log_r = truncated_normal_logpdf(z, mu1, np.exp(logs1), u)
        log_m = np.logaddexp(np.log(pi0) + log_a, np.log1p(-pi0) + log_r)
        return -float(np.sum(log_m))

    result = minimize(objective, init, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-8})
    if not result.success:
        raise RuntimeError(f"mixture optimization failed for seed {seed}: {result.message}")
    logit, offset0, logs0, mu1, logs1 = result.x
    return {
        "pi0": float(1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))),
        "offset0_z": float(offset0),
        "sigma0_z": float(np.exp(logs0)),
        "mu1_z": float(mu1),
        "sigma1_z": float(np.exp(logs1)),
        "seed": seed,
        "success": bool(result.success),
        "objective": float(result.fun),
        "nit": int(result.nit),
    }


def log_predictions(rows, anchor_center, upper, standardizer, anchor_sigma, reference, mixture):
    y = np.asarray([row["delta_p_GeV_c"] for row in rows], float)
    z = standardizer.forward(y)
    a = standardizer.forward(anchor_center)
    u = standardizer.forward(upper)
    log_anchor_ind = truncated_normal_logpdf(z, a, anchor_sigma, u) - math.log(standardizer.scale)
    log_reference = truncated_normal_logpdf(z, reference["mu_z"], reference["sigma_z"], u) - math.log(standardizer.scale)
    log_a_joint_z = truncated_normal_logpdf(z, a + mixture["offset0_z"], mixture["sigma0_z"], u)
    log_r_z = truncated_normal_logpdf(z, mixture["mu1_z"], mixture["sigma1_z"], u)
    log_pi0 = math.log(mixture["pi0"])
    log_pi1 = math.log1p(-mixture["pi0"])
    log_full_z = np.logaddexp(log_pi0 + log_a_joint_z, log_pi1 + log_r_z)
    log_a_joint = log_a_joint_z - math.log(standardizer.scale)
    log_r = log_r_z - math.log(standardizer.scale)
    log_full = log_full_z - math.log(standardizer.scale)
    log_resp0 = log_pi0 + log_a_joint_z - log_full_z
    return [
        {
            "event_uid": row["event_uid"],
            "split": row["split"],
            "y_delta_p_GeV_c": row["delta_p_GeV_c"],
            "p_in_GeV_c": row["p_in_GeV_c"],
            "log_pi0": log_pi0,
            "log_pi1": log_pi1,
            "log_f_anchor_ind": float(log_anchor_ind[i]),
            "log_f_anchor_joint": float(log_a_joint[i]),
            "log_f_residual": float(log_r[i]),
            "log_f_full": float(log_full[i]),
            "log_f_reference": float(log_reference[i]),
            "log_responsibility0": float(log_resp0[i]),
            "anchor_center_GeV_c": float(anchor_center[i]),
            "sigma_anchor_GeV_c": float(anchor_sigma * standardizer.scale),
            "standardizer_mean_GeV_c": standardizer.mean,
            "standardizer_scale_GeV_c": standardizer.scale,
            "support_upper_GeV_c": float(upper[i]),
        }
        for i, row in enumerate(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be in (0,1)")
    rows = read_rows(args.merged / "events.jsonl")
    if len(rows) < 100:
        raise ValueError("pilot has too few eligible transitions for development fit")
    for row in rows:
        row["split"] = "validation" if int(hashlib.sha256(row["event_uid"].encode()).hexdigest()[:8], 16) / 2**32 < args.validation_fraction else "train"
    train = [row for row in rows if row["split"] == "train"]
    y_train = np.asarray([row["delta_p_GeV_c"] for row in train], float)
    p_train = np.asarray([row["p_in_GeV_c"] for row in train], float)
    t_train = np.asarray([row["thickness_mm"] for row in train], float)
    q_train = np.asarray([row["pdg"] / 211 for row in train], float)
    anchor_train = pion_bethe_bloch_delta_p(p_train, t_train, q_train)
    standardizer, anchor_sigma, anchor_loss, anchor_success = fit_width(y_train, anchor_train, p_train)
    reference = fit_reference(y_train, p_train, standardizer)
    mixture = fit_mixture(y_train, anchor_train, p_train, standardizer, args.seed)
    all_p = np.asarray([row["p_in_GeV_c"] for row in rows], float)
    all_t = np.asarray([row["thickness_mm"] for row in rows], float)
    all_q = np.asarray([row["pdg"] / 211 for row in rows], float)
    all_anchor = pion_bethe_bloch_delta_p(all_p, all_t, all_q)
    predictions = log_predictions(rows, all_anchor, all_p, standardizer, anchor_sigma, reference, mixture)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": "W3_DEVELOPMENT_PILOT_NOT_CONFIRMATION",
        "seed": args.seed,
        "features": FEATURES,
        "forbidden_features": ["realized_path_length", "step_count", "deposit_GeV", "process_name", "secondary_count_in_slab", "exit"],
        "target": "delta_p_GeV_c",
        "support": "y < p_in_GeV_c",
        "split_rule": "sha256(event_uid) uniform threshold, train-only fit statistics",
        "source_manifest_sha256": sha256(args.merged / "manifest.json"),
        "eligible_events": len(rows),
        "train_events": len(train),
        "validation_events": len(rows) - len(train),
        "standardizer": {"mean": standardizer.mean, "scale": standardizer.scale},
        "anchor": {"formula": "Bethe-Bloch nominal Silicon, pion mass 0.13957039 GeV", "width_z": anchor_sigma, "objective": anchor_loss, "optimizer_success": anchor_success},
        "reference": reference,
        "mixture": mixture,
    }
    (output / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "eligible": len(rows), "train": len(train), "validation": len(rows) - len(train), "seed": args.seed, "anchor_sigma_z": anchor_sigma, "pi0": mixture["pi0"]}))


if __name__ == "__main__":
    main()
