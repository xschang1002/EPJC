"""Independently validate and merge fixed-stratum Geant4 JSONL chunks.

The catalog and every raw chunk are read back. A SQLite UNIQUE index rejects
overlapping event IDs without holding an entire production run in memory.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc


def put(stream, item: dict) -> None:
    stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")


def transition(row: dict) -> dict:
    return {key: row.get(key) for key in (
        "event_uid", "event_index", "pdg", "momentum_bin", "thickness_mm",
        "p_in_GeV_c", "entered", "entry", "exit", "exit_face", "status", "S1",
        "p_out_GeV_c", "delta_p_GeV_c", "fractional_loss", "delta_theta_rad",
        "reentry_count", "subsequent_visit_steps",
    )}


def validate_event_eligibility(row: dict) -> None:
    """Reject inconsistent forward-only eligibility or derived continuous targets."""
    if not isinstance(row.get("S1"), bool):
        raise ValueError("S1 must be a boolean")
    p_in = float(row["p_in_GeV_c"])
    if not math.isfinite(p_in) or p_in <= 0:
        raise ValueError("incident momentum must be positive and finite")
    exit_record = row.get("exit")
    p_out = None
    if exit_record is not None:
        momentum = exit_record["momentum_GeV_c"]
        if len(momentum) != 3 or not all(math.isfinite(float(v)) for v in momentum):
            raise ValueError("exit momentum must have three finite components")
        p_out = math.sqrt(sum(float(v) ** 2 for v in momentum))
    expected_s1 = (
        row.get("entry") is not None and p_out is not None and p_out > 0
        and row.get("exit_face") == "forward" and not row.get("aborted", False)
    )
    if row["S1"] != expected_s1:
        raise ValueError("S1 does not match forward first-visit eligibility")
    targets = ("p_out_GeV_c", "delta_p_GeV_c", "fractional_loss", "delta_theta_rad")
    if not expected_s1:
        if any(row.get(key) is not None for key in targets):
            raise ValueError("ineligible event contains a fabricated continuous target")
        return
    px, py, pz = exit_record["momentum_GeV_c"]
    expected = (p_out, p_in - p_out, (p_in - p_out) / p_in, math.atan2(math.hypot(px, py), pz))
    for key, target in zip(targets, expected):
        value = row.get(key)
        if value is None or not math.isfinite(float(value)) or not math.isclose(float(value), target, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"eligible transition has inconsistent {key}")
    if row["delta_p_GeV_c"] >= p_in:
        raise ValueError("eligible transition violates Y < p_in")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pending = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or pending.exists():
        parser.error("output or partial directory already exists")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog["cells"] != 42 or len(catalog["chunks"]) != 42:
        raise ValueError("expected a complete 42-cell catalog")
    pending.mkdir(parents=True)
    db = sqlite3.connect(pending / "event_uids.sqlite3")
    db.execute("CREATE TABLE seen (event_uid TEXT PRIMARY KEY, event_index INTEGER UNIQUE)")
    merged_files = {name: pending / f"{name}.jsonl" for name in (
        "events", "transitions", "primary_steps", "secondary_summary", "secondary_steps"
    )}
    counts = {name: 0 for name in merged_files}
    statuses: dict[str, int] = {}
    exit_faces: dict[str, int] = {}
    input_hashes = []
    chunk_manifest_hashes = []
    with ExitStack() as stack:
        outputs = {name: stack.enter_context(path.open("w", encoding="utf-8", newline="\n")) for name, path in merged_files.items()}
        for item in catalog["chunks"]:
            chunk_id = item["chunk_id"]
            root = args.chunks / f"chunk_{chunk_id:02d}"
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            input_path = args.catalog.parent / item["input_file"]
            if sha256(input_path) != item["input_sha256"] or manifest["beam_input_sha256"] != item["input_sha256"]:
                raise ValueError(f"chunk {chunk_id}: input hash mismatch")
            if manifest["run_id"] != catalog["run_id"] or manifest["chunk_id"] != chunk_id:
                raise ValueError(f"chunk {chunk_id}: run identity mismatch")
            if manifest["events"] != item["events"] or manifest["transport_seed"] != item["transport_seed"]:
                raise ValueError(f"chunk {chunk_id}: count or seed mismatch")
            for name, expected in manifest["files_sha256"].items():
                if sha256(root / name) != expected:
                    raise ValueError(f"chunk {chunk_id}: raw file hash mismatch: {name}")
            input_hashes.append(item["input_sha256"])
            chunk_manifest_hashes.append(sha256(manifest_path))
            source_events = list(jsonl(input_path))
            events = list(jsonl(root / "events.jsonl"))
            if len(source_events) != item["events"] or len(events) != item["events"]:
                raise ValueError(f"chunk {chunk_id}: ledger count mismatch")
            local_uids = set()
            for local_index, (source, row) in enumerate(zip(source_events, events)):
                if row["event_id"] != local_index:
                    raise ValueError(f"chunk {chunk_id}: local event ID discontinuity")
                for key in ("event_uid", "event_index", "momentum_bin", "pdg", "thickness_mm", "p_in_GeV_c"):
                    if row[key] != source[key]:
                        raise ValueError(f"chunk {chunk_id}: beam field changed: {key}")
                if row["momentum_bin"] != item["momentum_bin"] or row["pdg"] != item["pdg"]:
                    raise ValueError(f"chunk {chunk_id}: stratum mismatch")
                if not 0 <= row["event_index"] < catalog["events"]:
                    raise ValueError("event index outside requested run")
                db.execute("INSERT INTO seen VALUES (?, ?)", (row["event_uid"], row["event_index"]))
                local_uids.add(row["event_uid"])
                validate_event_eligibility(row)
                put(outputs["events"], row)
                put(outputs["transitions"], transition(row))
                counts["events"] += 1
                counts["transitions"] += 1
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
                if row["exit_face"] is not None:
                    exit_faces[row["exit_face"]] = exit_faces.get(row["exit_face"], 0) + 1
            for source_name, output_name, expected_count in (
                ("primary_steps.jsonl", "primary_steps", manifest["primary_step_rows"]),
                ("secondary_summary.jsonl", "secondary_summary", None),
                ("secondary_steps.jsonl", "secondary_steps", manifest["secondary_step_rows"]),
            ):
                local_count = 0
                for row in jsonl(root / source_name):
                    if row["event_uid"] not in local_uids:
                        raise ValueError(f"chunk {chunk_id}: orphan {output_name} row")
                    put(outputs[output_name], row)
                    counts[output_name] += 1
                    local_count += 1
                if expected_count is not None and local_count != expected_count:
                    raise ValueError(f"chunk {chunk_id}: {output_name} row count mismatch")
    db.commit()
    unique_count = db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
    db.close()
    if unique_count != catalog["events"] or counts["events"] != catalog["events"]:
        raise ValueError("merged event count or uniqueness mismatch")
    manifest = {
        "stage": "W1_W2_PILOT_CANDIDATE_NOT_ACCEPTED",
        "run_id": catalog["run_id"],
        "events": counts["events"],
        "cells": catalog["cells"],
        "counts": counts,
        "statuses": statuses,
        "exit_faces": exit_faces,
        "catalog_sha256": sha256(args.catalog),
        "input_file_hashes": input_hashes,
        "chunk_manifest_hashes": chunk_manifest_hashes,
        "merged_file_sha256": {name: sha256(path) for name, path in merged_files.items()},
        "units": {"momentum": "GeV/c", "length": "mm", "angle": "rad"},
        "prediction_features": ["p_in_GeV_c", "pdg", "thickness_mm"],
        "forbidden_prediction_features": ["realized_path_length", "step_count", "deposit", "process_labels", "secondary_counts", "outgoing_direction", "loss_region"],
    }
    (pending / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    pending.rename(args.output)
    print(json.dumps({"output": str(args.output), "events": counts["events"], "statuses": statuses, "exit_faces": exit_faces}))


if __name__ == "__main__":
    main()
