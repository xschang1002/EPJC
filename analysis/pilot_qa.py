"""Descriptive QA for a real Geant4 pilot candidate; no scientific claim."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import zlib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compressed_size(path: Path) -> int:
    compressor = zlib.compressobj(level=6, wbits=31)
    total = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(compressor.compress(block))
    return total + len(compressor.flush())


def quantile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower) if lower != upper else sorted_values[lower]


def magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.merged / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["merged_file_sha256"].items():
        if sha256(args.merged / f"{name}.jsonl") != expected:
            raise ValueError(f"merged {name} file hash mismatch")
    status = collections.Counter()
    faces = collections.Counter()
    selected = 0
    negative = 0
    missing_entry = 0
    entry_discrepancies = []
    losses = []
    reentries = 0
    first_visit_secondaries = 0
    with (args.merged / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            status[row["status"]] += 1
            if row["exit_face"]:
                faces[row["exit_face"]] += 1
            if row["entry"] is None:
                missing_entry += 1
            else:
                entry_p = magnitude(row["entry"]["momentum_GeV_c"])
                entry_discrepancies.append(abs(entry_p - row["p_in_GeV_c"]))
            reentries += row.get("reentry_count", 0)
            first_visit_secondaries += row["secondary_count_in_slab"]
            if row["S1"]:
                selected += 1
                losses.append(row["fractional_loss"])
                negative += row["delta_p_GeV_c"] < 0
    losses.sort()
    if sum(status.values()) != manifest["events"] or selected > manifest["events"]:
        raise ValueError("event counts do not close")
    chunk_manifests = [json.loads((args.chunks / f"chunk_{i:02d}" / "manifest.json").read_text(encoding="utf-8")) for i in range(manifest["cells"])]
    raw_bytes = {name: (args.merged / f"{name}.jsonl").stat().st_size for name in manifest["merged_file_sha256"]}
    gzip_bytes = {name: compressed_size(args.merged / f"{name}.jsonl") for name in manifest["merged_file_sha256"]}
    secondary_step_rows = manifest["counts"].get("secondary_steps", 0)
    report = {
        "stage": "W3_DEVELOPMENT_DATA_QA" if manifest["events"] >= 200000 else "W2_DESCRIPTIVE_QA_PILOT_CANDIDATE_NOT_ACCEPTED",
        "run_id": manifest["run_id"],
        "events": manifest["events"],
        "S1_events": selected,
        "statuses": dict(status),
        "exit_faces": dict(faces),
        "negative_delta_p_events": negative,
        "reentries": reentries,
        "new_secondaries_in_first_visit": first_visit_secondaries,
        "secondary_step_rows": secondary_step_rows,
        "full_secondary_step_policy": manifest.get("full_secondary_step_policy"),
        "missing_entry_events": missing_entry,
        "max_abs_entry_minus_source_momentum_GeV_c": max(entry_discrepancies, default=None),
        "fractional_loss_quantiles_exploratory_not_frozen": {str(p): quantile(losses, p) for p in (0.5, 0.9, 0.95, 0.99, 0.999)},
        "fractional_loss_min": min(losses, default=None),
        "fractional_loss_max": max(losses, default=None),
        "raw_bytes": raw_bytes,
        "gzip_level6_bytes_estimate": gzip_bytes,
        "gzip_level6_bytes_per_launched_event": sum(gzip_bytes.values()) / manifest["events"],
        "primary_steps_per_launched_event": manifest["counts"]["primary_steps"] / manifest["events"],
        "sum_chunk_geant4_seconds_after_import": sum(item["total_seconds"] for item in chunk_manifests),
        "sum_chunk_initialization_seconds_after_import": sum(item["initialization_seconds"] for item in chunk_manifests),
        "max_chunk_peak_rss_bytes": max(item["memory_bytes"]["peak_rss"] for item in chunk_manifests),
        "source_manifest": str(args.merged / "manifest.json"),
        "caveats": [
            *([] if secondary_step_rows else ["No full secondary step trajectories were captured in this candidate run."]),
            "Runtime sum excludes interpreter startup, chunk audits and merge; not an end-to-end benchmark.",
            "Quantiles here are descriptive and are not frozen train thresholds.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "events": report["events"], "S1": selected, "max_entry_discrepancy": report["max_abs_entry_minus_source_momentum_GeV_c"]}))


if __name__ == "__main__":
    main()
