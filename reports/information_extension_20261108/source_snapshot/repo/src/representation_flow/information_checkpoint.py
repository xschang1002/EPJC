"""Versioned frozen preprocessing, features and source identity for new models."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, logistic_logpdf
from representation_flow.experiment_contract import canonical_hash, sha256
from representation_flow.information_context import context_spec, information_features, validate_context_spec

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "information-checkpoint/v1"
SOURCE_NAMES = ("src/representation_flow/engine.py", "src/models/anchor.py", "src/models/anchor_legacy.py",
                "src/data_contract/event_split.py", "src/representation_flow/experiment_contract.py",
                "src/representation_flow/information_context.py", "src/representation_flow/information_checkpoint.py",
                "src/representation_flow/run_information_study.py")


def scientific_sources():
    return {name: sha256(ROOT / name) for name in SOURCE_NAMES}


def build_information_contract(preprocessor, context, protocol_sha256):
    if preprocessor.name not in ("Raw", "CorrectedPhysics") or preprocessor.scale_coefficients is not None:
        raise ValueError("information study requires Raw/CorrectedPhysics and constant shared target scale")
    contract = {"schema": SCHEMA, "preprocessor": preprocessor.to_dict(), "context_spec": context_spec(context),
                "protocol_sha256": protocol_sha256, "source_files": scientific_sources(),
                "target_unit": "GeV/c", "target_support": "Y < p_in", "eligibility": "forward"}
    contract["sha256"] = canonical_hash(contract)
    return contract


def validate_information_contract(contract):
    if contract.get("schema") != SCHEMA or contract.get("sha256") != canonical_hash({k: v for k, v in contract.items() if k != "sha256"}):
        raise ValueError("information checkpoint schema/digest mismatch")
    if (contract.get("target_unit"), contract.get("target_support"), contract.get("eligibility")) != ("GeV/c", "Y < p_in", "forward"):
        raise ValueError("information checkpoint target contract mismatch")
    if contract["source_files"] != scientific_sources():
        raise ValueError("information checkpoint scientific sources changed")
    validate_context_spec(contract["context_spec"])
    pre = RepresentationPreprocessor.from_dict(contract["preprocessor"])
    if pre.name not in ("Raw", "CorrectedPhysics") or pre.scale_coefficients is not None:
        raise ValueError("invalid information preprocessing")
    return pre


def load_information_checkpoint(path, *, expected_sha256=None, expected_experiment_identity=None):
    path = Path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("information checkpoint artifact changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    pre = validate_information_contract(payload["information_contract"])
    if payload["representation"] != pre.name or payload["context_label"] != payload["information_contract"]["context_spec"]["name"]:
        raise ValueError("checkpoint labels and scientific contract disagree")
    if expected_experiment_identity is not None and payload["experiment_identity"] != expected_experiment_identity:
        raise ValueError("checkpoint experiment identity mismatch")
    config = payload["model_config"]
    if config.get("context_dim") != 5 or config.get("hidden") != 32 or config.get("bound") != 8. or config.get("seed") != payload["seed"]:
        raise ValueError("checkpoint is not the declared equal-structure Small model")
    flow = ConditionalSplineFlow(**config)
    flow.load_state_dict(payload["state_dict"], strict=True)
    if flow.parameter_count() != payload["parameter_count"]:
        raise ValueError("checkpoint parameter count mismatch")
    flow.eval()
    return payload, pre, flow, payload["information_contract"]["context_spec"]


def make_tensors(arrays, indices, pre, spec):
    p, t, q, y = (np.asarray(arrays[k][indices]) for k in ("p_in", "thickness", "charge", "y"))
    features = information_features(p, t, q, spec)
    values = (y, pre.center(p, t, q), p, features, pre.scale_for(p, t, q))
    return tuple(torch.as_tensor(np.array(v, copy=True), dtype=torch.float32) for v in values)


def evaluate_information(flow, tensors, pre, batch_size=8192, *, components=False):
    if batch_size <= 0 or not len(tensors[0]):
        raise ValueError("nonempty evaluation data and positive batch size required")
    result = {}
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            y, c, p, context, scale = (v[start:start + batch_size] for v in tensors)
            terms = flow.original_terms(y, c, pre.mu, scale, p, context)
            saved = {"log_p_y": terms["log_p_y"]}
            if components:
                base, jac = flow.transform_to_base(terms["z"], context)
                saved.update(z=terms["z"], b=terms["b"], log_q_z=terms["log_q_z"],
                             base_value=base, base_logpdf=logistic_logpdf(base), spline_logabsdet=jac,
                             log_support_cdf=terms["log_f_b"], log_s=torch.log(scale),
                             log_cdf_y=terms["log_cdf_y"], log_sf_y=terms["log_sf_y"])
            for name, value in saved.items():
                result.setdefault(name, []).append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in result.items()}
