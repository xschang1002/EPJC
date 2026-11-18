"""Content-bound experiment identities and fail-closed resume checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "representation-experiment-v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes(extra: tuple[Path, ...] = ()) -> dict:
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ("engine.py", "run_a0_prototype.py", "checkpoint_contract.py", "experiment_contract.py")]
    paths += list((here.parent / "models").glob("anchor*.py"))
    paths.append(here.parent / "data_contract/event_split.py")
    paths += list(extra)
    return {str(p.relative_to(here.parent)) if p.is_relative_to(here.parent) else p.name: sha256(p) for p in paths}


def data_identity(merged: Path) -> dict:
    return {"manifest_sha256": sha256(merged / "manifest.json"), "events_sha256": sha256(merged / "events.jsonl")}


def validate_frozen_inputs(freeze: dict, preflight: dict, args, merged: Path) -> dict:
    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise ValueError("protocol is not frozen")
    if preflight.get("status") != "FORMAL_PREFLIGHT_PASS" or not preflight.get("checks") or not all(preflight["checks"].values()):
        raise ValueError("preflight is missing or has failed checks")
    opt = freeze["optimization"]
    for arg, key in (("learning_rate", "learning_rate"), ("max_updates", "max_updates"), ("batch_size", "batch_size"), ("validation_every", "validation_every_updates")):
        if getattr(args, arg) != opt[key]:
            raise ValueError(f"{arg} does not match the frozen protocol")
    actual = data_identity(merged)
    required = {"manifest_sha256": freeze["source"]["manifest_json_sha256"], "events_sha256": freeze["source"]["events_jsonl_sha256"]}
    if actual != required:
        raise ValueError("current data do not match the frozen source hashes")
    semantics = preflight.get("manifest_hash_semantics", {})
    if semantics.get("audit_A0_source_manifest_sha256") != actual["manifest_sha256"] or semantics.get("center_diagnostics_source_manifest_sha256") != actual["events_sha256"]:
        raise ValueError("preflight is not bound to the current frozen data")
    return actual


def bind_output(output: Path, identity: dict) -> str:
    """Verify an existing identity before any experiment output can be changed."""
    output = Path(output)
    key = canonical_hash(identity)
    path = output / "experiment_identity.json"
    if output.exists() and any(output.iterdir()):
        if not path.exists():
            raise ValueError("existing output has no experiment identity; preserve historical output and choose a new directory")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("sha256") != key or record.get("identity") != identity:
            raise ValueError("resume rejected: experiment data, code, preprocessing or configuration changed")
    else:
        save_json(path, {"schema": SCHEMA, "sha256": key, "identity": identity})
    return key


def artifact_record(path: Path, output: Path) -> dict:
    path = Path(path).resolve()
    root = Path(output).resolve()
    if not path.is_relative_to(root):
        raise ValueError("experiment artifact escapes output directory")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_completed(rows: list[dict], output: Path, identity: str, expected_ids: set[str]) -> set[str]:
    completed = set()
    for row in rows:
        run_id = row["run_id"]
        if run_id not in expected_ids or run_id in completed:
            raise ValueError(f"unexpected or duplicate completed run: {run_id}")
        if row.get("experiment_identity") != identity:
            raise ValueError(f"completed run has a different or missing identity: {run_id}")
        artifacts = row.get("artifacts", {})
        if not {"checkpoint", "predictions", "run_manifest"}.issubset(artifacts):
            raise ValueError(f"completed run lacks integrity records: {run_id}")
        for record in artifacts.values():
            path = (output / record["path"]).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"completed artifact missing, changed or outside output: {run_id}")
        completed.add(run_id)
    return completed
