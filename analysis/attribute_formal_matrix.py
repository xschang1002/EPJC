"""Export and independently verify the three-term gain decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Historical MDN artifacts require the original center; never silently correct it.
from models.anchor_legacy import pion_bethe_bloch_delta_p
from models.densities import truncated_normal_logpdf
from models.fit_formal_matrix import design, predict_model, read_rows
from data_contract.event_split import split_label


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="W4_FORMAL_MATRIX_ATTRIBUTION_DEVELOPMENT_NOT_CONFIRMATION")
    parser.add_argument("--caveat", default="Development decomposition only; no confirmation selection or mechanism claim.")
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    mean = float(checkpoint["metadata"]["target_mean"])
    scale = float(checkpoint["metadata"]["target_scale"])
    validation_fraction = float(checkpoint["metadata"].get("validation_fraction", 0.2))
    diagnostic_fraction = float(checkpoint["metadata"].get("diagnostic_fraction", 0.0))
    rows = read_rows(args.merged / "events.jsonl")
    joint = checkpoint["models"]["joint_anchor_residual_k2"]["params"]
    shared = checkpoint["models"]["shared_gaussian"]
    anchor_model = checkpoint["models"]["anchor_independent"]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_values = {key: [] for key in ("total", "anchor_ind", "anchor_reorganization", "direct", "direct_via_responsibility")}
    closure_max = {"direct": 0.0, "three_term": 0.0}
    tail_values = []
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            p = float(row["p_in_GeV_c"])
            q = float(row["pdg"]) / 211.0
            t = float(row["thickness_mm"])
            y = float(row["delta_p_GeV_c"])
            z = (y - mean) / scale
            upper = (p - mean) / scale
            anchor = (float(pion_bethe_bloch_delta_p(p, t, q)) - mean) / scale
            x = design([row])
            l_anchor_ind = float(predict_model("anchor_independent", anchor_model, np.asarray([z]), np.asarray([upper]), np.asarray([anchor]), x)[0])
            l_reference = float(predict_model("shared_gaussian", shared, np.asarray([z]), np.asarray([upper]), np.asarray([anchor]), x)[0])
            logit, offset, mu1, logs0, logs1 = joint
            log_pi0 = -float(np.logaddexp(0.0, -float(logit)))
            log_pi1 = -float(np.logaddexp(0.0, float(logit)))
            l0_z = float(truncated_normal_logpdf(z, anchor + offset, math.exp(logs0), upper))
            l1_z = float(truncated_normal_logpdf(z, mu1, math.exp(logs1), upper))
            l_full_z = float(logsumexp([log_pi0 + l0_z, log_pi1 + l1_z]))
            l_anchor_joint = l0_z
            log_resp0 = log_pi0 + l0_z - l_full_z
            total = l_full_z - l_reference
            anchor_gain = l_anchor_ind - l_reference
            reorganization = l_anchor_joint - l_anchor_ind
            direct = l_full_z - l_anchor_joint
            direct_via = log_pi0 - log_resp0
            closure_max["direct"] = max(closure_max["direct"], abs(direct - direct_via))
            closure_max["three_term"] = max(closure_max["three_term"], abs(total - (anchor_gain + reorganization + direct)))
            values = {"total": total, "anchor_ind": anchor_gain, "anchor_reorganization": reorganization, "direct": direct, "direct_via_responsibility": direct_via}
            for key, value in values.items():
                summary_values[key].append(value)
            stream.write(json.dumps({"event_uid": row["event_uid"], "split": split_label(row["event_uid"], validation_fraction, diagnostic_fraction), "fractional_loss": row["fractional_loss"], "log_pi0": log_pi0, "log_responsibility0": log_resp0, "log_f_anchor_ind": l_anchor_ind - math.log(scale), "log_f_anchor_joint": l_anchor_joint - math.log(scale), "log_f_reference": l_reference - math.log(scale), "log_f_full": l_full_z - math.log(scale), "G_total": total, "G_anchor_ind": anchor_gain, "G_anchor_reorganization": reorganization, "G_direct": direct, "G_direct_via_responsibility": direct_via}, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "stage": args.stage,
        "seed": checkpoint["metadata"]["seed"],
        "events": len(rows),
        "source_manifest_sha256": checkpoint["source_manifest_sha256"],
        "attribution_file_sha256": sha256(output),
        "closure_max_abs": closure_max,
        "means": {key: statistics.fmean(values) for key, values in summary_values.items()},
        "positive_fraction": {key: float(np.mean(np.asarray(values) > 0)) for key, values in summary_values.items()},
        "primary_model": "joint_anchor_residual_k2",
        "caveat": args.caveat,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "summary": str(summary_path), "events": len(rows), "closure_max": closure_max}))


if __name__ == "__main__":
    main()
