"""Independent, standard-library-only audit of a W0 Geant4 smoke output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def momentum(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        actual = sha256(root / name)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")

    events = list(rows(root / "events.jsonl"))
    if len(events) != manifest["events"]:
        raise ValueError("ledger row count does not match requested events")
    ids = [row["event_id"] for row in events]
    if ids != list(range(manifest["events"])):
        raise ValueError("event IDs are missing, duplicated, or out of order")
    uids = [row["event_uid"] for row in events]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate event_uid")

    event_by_uid = {row["event_uid"]: row for row in events}
    statuses: dict[str, int] = {}
    for row in events:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        if row["pdg"] not in (-211, 211):
            raise ValueError("non-pion beam record")
        if row["primary_track_id"] is None or row.get("pion_mass_GeV") is None:
            raise ValueError("original-primary identity or runtime mass missing")
        if row["S1"]:
            if row["status"] not in ("exited", "exited_first_visit") or row["exit"] is None:
                raise ValueError("S1 event without a valid exit")
            p_out = momentum(row["exit"]["momentum_GeV_c"])
            if not math.isclose(p_out, row["p_out_GeV_c"], abs_tol=1e-10):
                raise ValueError("outgoing momentum magnitude mismatch")
            if not math.isclose(
                row["delta_p_GeV_c"], row["p_in_GeV_c"] - p_out, abs_tol=1e-10
            ):
                raise ValueError("delta_p definition mismatch")
            if not math.isclose(
                row["fractional_loss"], row["delta_p_GeV_c"] / row["p_in_GeV_c"], abs_tol=1e-10
            ):
                raise ValueError("fractional loss definition mismatch")
            px, py, pz = row["exit"]["momentum_GeV_c"]
            if not math.isclose(
                row["delta_theta_rad"], math.atan2(math.hypot(px, py), pz), abs_tol=1e-10
            ):
                raise ValueError("delta theta definition mismatch")
        elif row["p_out_GeV_c"] is not None or row["delta_p_GeV_c"] is not None:
            raise ValueError("non-S1 event has fabricated continuous target")

    step_counts = {uid: 0 for uid in uids}
    last_step_index = {uid: 0 for uid in uids}
    for step in rows(root / "primary_steps.jsonl"):
        uid = step["event_uid"]
        if uid not in event_by_uid:
            raise ValueError("orphan primary step")
        if step["track_id"] != event_by_uid[uid]["primary_track_id"]:
            raise ValueError("step does not belong to original primary")
        if step["step_index"] <= last_step_index[uid]:
            raise ValueError("non-increasing original-primary step index")
        last_step_index[uid] = step["step_index"]
        step_counts[uid] += 1
        for point in (step["pre"], step["post"]):
            p = momentum(point["momentum_GeV_c"])
            mass = event_by_uid[uid]["pion_mass_GeV"]
            kinetic_expected = math.sqrt(p * p + mass * mass) - mass
            if not math.isclose(
                point["kinetic_GeV"], kinetic_expected, rel_tol=1e-6, abs_tol=1e-8
            ):
                raise ValueError(f"mass-shell mismatch: {uid} step {step['step_index']}")
    if sum(step_counts.values()) != manifest["primary_step_rows"]:
        raise ValueError("primary step row count mismatch")
    for uid, count in step_counts.items():
        if count == 0:
            raise ValueError(f"event lacks original-primary step record: {uid}")

    secondary_count = 0
    for item in rows(root / "secondary_summary.jsonl"):
        if item["event_uid"] not in event_by_uid:
            raise ValueError("orphan secondary summary")
        secondary_count += len(item["new_secondaries"])
    if secondary_count != manifest["new_secondaries_in_slab"]:
        raise ValueError("secondary count mismatch")

    secondary_step_count = 0
    last_secondary_step = {}
    if "secondary_steps.jsonl" in manifest["files_sha256"]:
        for item in rows(root / "secondary_steps.jsonl"):
            uid = item["event_uid"]
            if uid not in event_by_uid:
                raise ValueError("orphan secondary step")
            if item["track_id"] == event_by_uid[uid]["primary_track_id"] or item["parent_id"] <= 0:
                raise ValueError("secondary step belongs to original primary or has no parent")
            key = (uid, item["track_id"])
            if item["step_index"] <= last_secondary_step.get(key, 0):
                raise ValueError("non-increasing secondary track step index")
            last_secondary_step[key] = item["step_index"]
            secondary_step_count += 1
        if secondary_step_count != manifest["secondary_step_rows"]:
            raise ValueError("secondary step count mismatch")

    report = {
        "passed": True,
        "run": str(root),
        "events": len(events),
        "statuses": statuses,
        "original_primary_steps": sum(step_counts.values()),
        "new_secondaries_in_slab": secondary_count,
        "secondary_step_rows": secondary_step_count,
        "real_geant4_runtime": manifest["geant4_runtime_string"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
