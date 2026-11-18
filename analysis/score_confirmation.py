"""Score untouched confirmation blocks with frozen main-batch checkpoints.

No fit, split, threshold estimation, or model selection is performed here.
The threshold is read from the corresponding main-batch score report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.densities import truncated_normal_cdf, truncated_normal_logpdf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, np.ndarray | int]:
    p, y, t, q, frac, uid = [], [], [], [], [], []
    with (path / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("S1"):
                continue
            p.append(float(row["p_in_GeV_c"]))
            y.append(float(row["delta_p_GeV_c"]))
            t.append(float(row["thickness_mm"]))
            q.append(float(row["pdg"]) / 211.0)
            frac.append(float(row["fractional_loss"]))
            uid.append(row["event_uid"])
    return {"p": np.asarray(p), "y": np.asarray(y), "t": np.asarray(t), "q": np.asarray(q), "frac": np.asarray(frac), "uids": uid, "events": len(p)}


def score(data: dict, checkpoint: dict, threshold: float) -> dict:
    mean = float(checkpoint["metadata"]["target_mean"])
    scale = float(checkpoint["metadata"]["target_scale"])
    p, y, t, q, frac = data["p"], data["y"], data["t"], data["q"], data["frac"]
    z = (y - mean) / scale
    upper = (p - mean) / scale
    anchor = (pion_bethe_bloch_delta_p(p, t, q) - mean) / scale
    x = np.column_stack((np.ones(len(p)), (p - 0.65) / 0.25, q, (t - 0.3) / 0.3))
    spec = checkpoint["models"]["unanchored_mdn_k2"]
    v = np.asarray(spec["params"], dtype=float)
    pi0 = 1.0 / (1.0 + np.exp(-np.clip(x @ v[:4], -35, 35)))
    mu0, mu1 = x @ v[4:8], x @ v[8:12]
    s0, s1 = np.exp(v[12]), np.exp(v[13])
    l0 = truncated_normal_logpdf(z, mu0, s0, upper)
    l1 = truncated_normal_logpdf(z, mu1, s1, upper)
    logp = np.logaddexp(np.log(pi0) + l0, np.log1p(-pi0) + l1) - math.log(scale)
    threshold_z = (p * threshold - mean) / scale
    cdf_threshold = pi0 * truncated_normal_cdf(threshold_z, mu0, s0, upper) + (1 - pi0) * truncated_normal_cdf(threshold_z, mu1, s1, upper)
    pit = pi0 * truncated_normal_cdf(z, mu0, s0, upper) + (1 - pi0) * truncated_normal_cdf(z, mu1, s1, upper)
    tail = frac >= threshold
    qtail = np.clip(1.0 - cdf_threshold, 1e-15, 1 - 1e-15)
    cdf_threshold = np.clip(cdf_threshold, 1e-15, 1 - 1e-15)
    full = -logp
    censored = np.where(tail, full, -np.log(cdf_threshold))
    binary = np.where(tail, -np.log(qtail), -np.log(cdf_threshold))
    brier = (qtail - tail.astype(float)) ** 2
    coverage = {}
    for level in (0.5, 0.68, 0.9, 0.95, 0.99):
        alpha = (1 - level) / 2
        coverage[str(level)] = float(np.mean((pit >= alpha) & (pit <= 1 - alpha)))
    return {"events": int(len(p)), "tail_event_count": int(tail.sum()), "full_nll": float(np.mean(full)), "censored_log_score": float(np.mean(censored)), "tail_binary_log_score": float(np.mean(binary)), "tail_brier": float(np.mean(brier)), "pit_mean": float(np.mean(pit)), "pit_variance": float(np.var(pit)), "interval_coverage_from_pit": coverage, "threshold": float(threshold)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-a", type=Path, required=True)
    ap.add_argument("--block-b", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--main-score-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    blocks = {"a": load(args.block_a), "b": load(args.block_b)}
    results = {}
    for seed in range(101, 111):
        checkpoint_base = args.checkpoint_dir if seed == 101 else args.checkpoint_dir.parent / f"formal_matrix_main_1000000_seed{seed}"
        checkpoint_path = checkpoint_base / f"checkpoint_seed{seed}.json"
        score_path = args.main_score_dir / f"main_1000000_scores_seed{seed}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        main_score = json.loads(score_path.read_text(encoding="utf-8"))
        threshold = float(main_score["train_q99_fractional_loss"])
        block_scores = {key: score(data, checkpoint, threshold) for key, data in blocks.items()}
        # Recompute pooled arrays explicitly to keep event UID lists out of numeric scoring.
        pooled_data = {"p": np.concatenate([blocks["a"]["p"], blocks["b"]["p"]]), "y": np.concatenate([blocks["a"]["y"], blocks["b"]["y"]]), "t": np.concatenate([blocks["a"]["t"], blocks["b"]["t"]]), "q": np.concatenate([blocks["a"]["q"], blocks["b"]["q"]]), "frac": np.concatenate([blocks["a"]["frac"], blocks["b"]["frac"]])}
        results[str(seed)] = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path), "main_score_report": str(score_path), "main_score_report_sha256": sha256(score_path), "threshold_source": "main_1000000_scores_seed%d.json:train_q99_fractional_loss" % seed, "selected_model": "unanchored_mdn_k2", "blocks": block_scores, "pooled": score(pooled_data, checkpoint, threshold)}
    out = {"stage": "W6_CONFIRMATION_SCORES_FROZEN_MODEL_NO_REFIT", "selected_model": "unanchored_mdn_k2", "mechanism_attribution_model": "joint_anchor_residual_k2", "seeds": list(range(101, 111)), "blocks": {"a": {"merged": str(args.block_a), "manifest_sha256": sha256(args.block_a / "manifest.json"), "events": int(blocks["a"]["events"])}, "b": {"merged": str(args.block_b), "manifest_sha256": sha256(args.block_b / "manifest.json"), "events": int(blocks["b"]["events"])}}, "scores": results, "policy": "untouched confirmation data; no train/validation split, threshold fitting, selection, or refitting"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seeds": 10, "events_a": out["blocks"]["a"]["events"], "events_b": out["blocks"]["b"]["events"]}))


if __name__ == "__main__":
    main()
