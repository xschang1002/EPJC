"""Audit legacy 500k Geant4 blocks for frozen representation-study reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from representation_flow.engine import load_source_arrays, sha256_file


def audit_block(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    arrays, _ = load_source_arrays(path, validation_fraction=0.1, diagnostic_fraction=0.3)
    p = arrays["p_in"].astype(float)
    t = arrays["thickness"].astype(float)
    q = arrays["charge"].astype(float)
    y = arrays["y"].astype(float)
    return {
        "path": str(path),
        "run_id": manifest.get("run_id"),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "events_file_sha256": sha256_file(path / "events.jsonl"),
        "manifest_stage": manifest.get("stage"),
        "events_loaded": int(len(arrays["event_uid"])),
        "manifest_events": int(manifest.get("events", -1)),
        "p_in_range_GeV_c": [float(np.min(p)), float(np.max(p))],
        "thickness_values_mm": [float(v) for v in sorted(np.unique(t))],
        "charge_values": [float(v) for v in sorted(np.unique(q))],
        "target_range_GeV_c": [float(np.min(y)), float(np.max(y))],
        "support_violations": int(np.sum(y >= p)),
        "uid_count": int(len(np.unique(arrays["event_uid"]))),
        "contract_match": bool(
            len(arrays["event_uid"]) == int(manifest.get("exit_faces", {}).get("forward", -1))
            and np.allclose(sorted(np.unique(t)), [0.1, 0.3, 1.0])
            and np.array_equal(sorted(np.unique(q)), [-1.0, 1.0])
            and float(np.min(p)) >= 0.3
            and float(np.max(p)) <= 1.0
            and np.all(y < p)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/independent_confirmation_data_audit.json"))
    args = parser.parse_args()
    blocks = [audit_block(args.root / "confirmation_500k_a_full_merged"), audit_block(args.root / "confirmation_500k_b_full_merged")]
    representation_refs = []
    output_resolved = args.output.resolve()
    for path in (Path("reports/representation_v2"),):
        for file in path.rglob("*"):
            if file.is_file() and file.resolve() != output_resolved and file.suffix in {".md", ".json", ".py"}:
                text = file.read_text(encoding="utf-8", errors="ignore")
                if "confirmation_500k" in text:
                    representation_refs.append(str(file))
    result = {
        "status": "CANDIDATE_REUSE_WITH_PROVENANCE_BOUNDARY" if all(block["contract_match"] for block in blocks) else "CONTRACT_FAIL",
        "blocks": blocks,
        "representation_v2_report_references_to_confirmation_blocks": sorted(representation_refs),
        "provenance_boundary": {
            "independent_of_current_representation_v2": len(representation_refs) == 0,
            "globally_untouched": False,
            "reason": "The blocks were already used by the repository's legacy W6/MDN confirmation workflow; they are not globally untouched, even though current representation_v2 artifacts do not reference them.",
            "reuse_rule": "If scored, label as independent-of-representation-v2 source-domain confirmation with legacy-provenance disclosure, not as globally untouched confirmation.",
        },
        "no_training_or_model_selection": True,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "blocks": len(blocks), "representation_refs": len(representation_refs)}, sort_keys=True))


if __name__ == "__main__":
    main()
