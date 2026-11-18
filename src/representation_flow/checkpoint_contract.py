"""Fail-closed scientific identity and preprocessing for frozen checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from representation_flow.engine import RepresentationPreprocessor, sha256_file

SCHEMA = "representation-flow-checkpoint/v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCES = {
    "engine.py": Path(__file__).with_name("engine.py"),
    "anchor.py": REPOSITORY_ROOT / "src/models/anchor.py",
    "anchor_legacy.py": REPOSITORY_ROOT / "src/models/anchor_legacy.py",
    "checkpoint_contract.py": Path(__file__),
}


def contract_digest(contract: dict) -> str:
    content = {key: value for key, value in contract.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_checkpoint_contract(preprocessor, context_feature_mode: str, source_paths=None, *, eligibility="forward") -> dict:
    if context_feature_mode not in ("full", "thickness_charge"):
        raise ValueError("unknown context feature mode")
    if eligibility not in ("forward", "legacy_s1"):
        raise ValueError("unknown eligibility contract")
    preprocessing = preprocessor.to_dict()
    RepresentationPreprocessor.from_dict(preprocessing)
    sources = {}
    for name, source in {**BASE_SOURCES, **(source_paths or {})}.items():
        path = Path(source).resolve(strict=True)
        if name in BASE_SOURCES and path != BASE_SOURCES[name].resolve():
            raise ValueError(f"cannot substitute required scientific source: {name}")
        try:
            relative = path.relative_to(REPOSITORY_ROOT)
        except ValueError as exc:
            raise ValueError("checkpoint source must be inside its repository") from exc
        sources[name] = {"path": relative.as_posix(), "sha256": sha256_file(path)}
    contract = {"schema": SCHEMA, "preprocessor": preprocessing,
                "context_feature_mode": context_feature_mode,
                "target_unit": "GeV/c", "target_support": "Y < p_in",
                "eligibility": eligibility, "source_files": sources}
    contract["sha256"] = contract_digest(contract)
    return contract


def validate_checkpoint_contract(contract: dict, *, repository_root=None) -> RepresentationPreprocessor:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint contract schema")
    if contract.get("sha256") != contract_digest(contract):
        raise ValueError("checkpoint contract digest mismatch")
    if (contract.get("target_unit"), contract.get("target_support")) != ("GeV/c", "Y < p_in"):
        raise ValueError("checkpoint target unit/support mismatch")
    if contract.get("context_feature_mode") not in ("full", "thickness_charge") or contract.get("eligibility") not in ("forward", "legacy_s1"):
        raise ValueError("invalid checkpoint context/eligibility")
    sources = contract.get("source_files", {})
    if not set(BASE_SOURCES).issubset(sources):
        raise ValueError("checkpoint is missing scientific source identities")
    root = Path(repository_root or REPOSITORY_ROOT).resolve()
    if root != REPOSITORY_ROOT:
        raise ValueError("checkpoint sources must match the currently imported implementation")
    for name, record in sources.items():
        path = (root / record["path"]).resolve()
        if name in BASE_SOURCES and record["path"] != BASE_SOURCES[name].resolve().relative_to(REPOSITORY_ROOT).as_posix():
            raise ValueError(f"checkpoint scientific source path mismatch: {name}")
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"checkpoint source unavailable: {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"checkpoint source identity mismatch: {name}")
    return RepresentationPreprocessor.from_dict(contract["preprocessor"])


def load_checkpoint(path, *, map_location="cpu", allow_legacy=False,
                    legacy_preprocessor=None, legacy_context_feature_mode=None,
                    expected_sha256=None, repository_root=None):
    """Return (payload, frozen_preprocessor); never refit from current data.

    Legacy decoding is an explicit historical-audit operation. Its caller must
    supply the actual historical preprocessing from an independently checked
    artifact, not freshly fit values or a corrected center.
    """
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("checkpoint artifact hash mismatch")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint payload must contain state_dict")
    contract = payload.get("checkpoint_contract")
    if contract is not None:
        pre = validate_checkpoint_contract(contract, repository_root=repository_root)
    else:
        if not allow_legacy:
            raise ValueError("legacy checkpoint rejected; explicit historical preprocessing and allow_legacy are required")
        if legacy_preprocessor is None or legacy_context_feature_mode not in ("full", "thickness_charge"):
            raise ValueError("legacy checkpoint needs explicit historical preprocessing and context")
        pre = RepresentationPreprocessor.from_dict(legacy_preprocessor.to_dict() if isinstance(legacy_preprocessor, RepresentationPreprocessor) else legacy_preprocessor)
        if pre.name.startswith("CorrectedPhysics"):
            raise ValueError("legacy checkpoint cannot be relabelled with a corrected center")
        payload = {**payload, "legacy_audit": True, "legacy_context_feature_mode": legacy_context_feature_mode}
    if payload.get("representation") != pre.name:
        raise ValueError("checkpoint representation and preprocessing identity disagree")
    config = payload.get("model_config")
    if config is not None:
        mode = contract["context_feature_mode"] if contract is not None else legacy_context_feature_mode
        if config.get("context_dim") != (5 if mode == "full" else 2):
            raise ValueError("checkpoint model context dimension mismatch")
    return payload, pre
