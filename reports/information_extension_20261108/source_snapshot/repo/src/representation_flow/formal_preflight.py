"""Verify the frozen source contract before the formal matched study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from representation_flow.engine import load_source_arrays, sha256_file, subset_indices


def _uid_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _row_digest(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        fields = (
            row["event_uid"],
            row["p_in_GeV_c"],
            row["p_out_GeV_c"],
            row["thickness_mm"],
            row["pdg"],
            row["delta_p_GeV_c"],
        )
        digest.update(json.dumps(fields, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_s1_rows(events_path: Path):
    rows = []
    with events_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("S1"):
                rows.append(row)
    return rows


def audit(merged: Path, report_dir: Path) -> dict:
    arrays, manifest = load_source_arrays(merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    rows = _load_s1_rows(merged / "events.jsonl")
    audit_a0 = json.loads((report_dir / "audit_A0.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((report_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    center = json.loads((report_dir / "optimization_audit" / "center_diagnostics.json").read_text(encoding="utf-8"))

    direct_uids = np.asarray([row["event_uid"] for row in rows], dtype=object)
    direct_p = np.asarray([float(row["p_in_GeV_c"]) for row in rows], dtype=np.float64)
    direct_t = np.asarray([float(row["thickness_mm"]) for row in rows], dtype=np.float64)
    direct_q = np.asarray([float(row["pdg"]) / 211.0 for row in rows], dtype=np.float64)
    direct_y = np.asarray([float(row["delta_p_GeV_c"]) for row in rows], dtype=np.float64)
    direct_formula_y = np.asarray([float(row["p_in_GeV_c"]) - float(row["p_out_GeV_c"]) for row in rows], dtype=np.float64)

    same_arrays = {
        "event_uid": bool(np.array_equal(arrays["event_uid"], direct_uids)),
        "p_in_GeV_c": bool(np.array_equal(arrays["p_in"], direct_p)),
        "thickness_mm": bool(np.array_equal(arrays["thickness"], direct_t)),
        "charge": bool(np.array_equal(arrays["charge"], direct_q)),
        "delta_p_GeV_c": bool(np.array_equal(arrays["y"], direct_y)),
    }
    split_uid_digests = {}
    for split in ("train", "validation", "diagnostic"):
        indices = np.flatnonzero(arrays["split"] == split)
        split_uid_digests[split] = {
            "events": int(len(indices)),
            "uid_sha256": _uid_digest(arrays["event_uid"][indices]),
        }

    train_indices = subset_indices(arrays, fraction=0.1, catalog_id=4101)
    manifest_hash = sha256_file(merged / "manifest.json")
    events_hash = sha256_file(merged / "events.jsonl")
    source_counts = {label: int(np.sum(arrays["split"] == label)) for label in ("train", "validation", "diagnostic")}
    target_error = np.abs(direct_y - direct_formula_y)
    checks = {
        "same_source_directory": dataset_manifest["source_directory"].replace("/", "\\").endswith(str(merged).replace("/", "\\")),
        "manifest_file_hash_matches_audit_A0": manifest_hash == audit_a0["source_manifest_sha256"],
        "events_file_hash_matches_center": events_hash == center["source_manifest_sha256"],
        "manifest_hash_and_events_hash_are_distinct_keys": manifest_hash != events_hash,
        "eligible_S1_count_matches": len(rows) == audit_a0["eligible_S1_events"] == dataset_manifest["eligible_S1_events"],
        "split_counts_match": source_counts == audit_a0["split_counts"] == dataset_manifest["split_counts"],
        "arrays_match_direct_event_rows": all(same_arrays.values()),
        "subset_uid_hash_matches": _uid_digest(arrays["event_uid"][train_indices]) == audit_a0["subset_event_uid_sha256"],
        "target_formula_matches_p_in_minus_p_out": bool(np.max(target_error) <= 1e-12),
        "support_y_less_than_p_in": bool(np.all(direct_y < direct_p)),
        "center_train_validation_counts_match": center["train_subset_events"] == len(train_indices) and center["validation_events"] == source_counts["validation"],
    }
    result = {
        "status": "FORMAL_PREFLIGHT_PASS" if all(checks.values()) else "FORMAL_PREFLIGHT_FAIL",
        "source_directory": str(merged),
        "manifest_hash_semantics": {
            "audit_A0_source_manifest_sha256": manifest_hash,
            "audit_A0_hash_kind": "sha256(manifest.json)",
            "center_diagnostics_source_manifest_sha256": events_hash,
            "center_diagnostics_hash_kind": "sha256(events.jsonl)",
            "explanation": "The fields have the same name but intentionally refer to different provenance artifacts. The source manifest describes the merged dataset; center diagnostics loaded the event stream and recorded its content hash.",
        },
        "source_manifest_counts": manifest.get("counts", {}),
        "eligible_S1_events": len(rows),
        "split_counts": source_counts,
        "split_uid_digests": split_uid_digests,
        "all_S1_uid_sha256": _uid_digest(direct_uids),
        "direct_event_row_digest": _row_digest(rows),
        "array_field_equality": same_arrays,
        "target_definition": {
            "formula": "delta_p_GeV_c = p_in_GeV_c - p_out_GeV_c",
            "unit": "GeV/c",
            "max_abs_formula_error": float(np.max(target_error)),
            "support": "Y < p_in",
            "support_violations": int(np.sum(direct_y >= direct_p)),
        },
        "subset_catalog": {"catalog_id": 4101, "fraction": 0.1, "events": len(train_indices), "uid_sha256": _uid_digest(arrays["event_uid"][train_indices])},
        "checks": checks,
        "formal_scope_boundary": {
            "new_geant4": False,
            "ood": False,
            "spline_bound_extension": False,
            "additional_models": False,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/representation_v2"))
    args = parser.parse_args()
    result = audit(args.merged, args.report_dir)
    out = args.report_dir / "formal_preflight_audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(out)}, sort_keys=True))
    if result["status"] != "FORMAL_PREFLIGHT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
