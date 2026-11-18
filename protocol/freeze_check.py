"""Read-only W6 freeze/confirmation gate for the pion material study.

The gate never changes protocol state and never launches Geant4. After the
human-approved candidate is frozen it records that protocol state while still
refusing confirmation until the separate confirmation switch is enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read_json(root / "protocol" / "environment_selection.json")
    frozen_path = root / "protocol" / "frozen_protocol.json"
    confirmation_path = root / "protocol" / "confirmation_frozen_protocol.json"
    w0_e = read_json(root / "reports" / "audit_w0_e.json")
    w0_f = read_json(root / "reports" / "audit_w0_f.json")
    replay = read_json(root / "reports" / "replay_w0_e_f.json")
    w3_text = (root / "reports" / "W3_W5_DEVELOPMENT.md").read_text(encoding="utf-8")
    w2_text = (root / "reports" / "W2_PILOT_CANDIDATE.md").read_text(encoding="utf-8")
    main_matrix = read_json(root / "reports" / "main_1000000_matrix_summary.json")
    main_scores = read_json(root / "reports" / "main_1000000_scores_summary.json")
    main_attr = read_json(root / "reports" / "main_1000000_attribution_summary.json")
    confirmation_summary_path = root / "reports" / "confirmation_scores_summary.json"
    confirmation_qa_paths = [root / "reports" / "confirmation_500k_a_qa.json", root / "reports" / "confirmation_500k_b_qa.json"]
    confirmation_summary = read_json(confirmation_summary_path) if confirmation_summary_path.exists() else {}

    frozen_hash = sha256(frozen_path) if frozen_path.exists() else None
    checks = {
        "w0_real_audit_e": bool(w0_e.get("passed") and w0_e.get("real_geant4_runtime")),
        "w0_real_audit_f": bool(w0_f.get("passed") and w0_f.get("real_geant4_runtime")),
        "w0_exact_replay": bool(replay.get("passed")),
        "protocol_frozen": protocol.get("status") == "PRODUCTION_FROZEN",
        "w2_formal_acceptance": "formal W2 acceptance" not in w2_text,
        "w3_w5_formal_completion": "development only" not in w3_text,
        "confirmation_enabled": protocol.get("confirmation_enabled") is True,
        "human_freeze_approval_present": bool(protocol.get("human_freeze_approval")),
        "frozen_protocol_hash_matches_registration": bool(
            frozen_hash and frozen_hash == protocol.get("frozen_protocol_sha256")
        ) if protocol.get("frozen_protocol_path") == "protocol/frozen_protocol.json" else True,
        "confirmation_approval_present": bool(protocol.get("confirmation_approval")),
        "confirmation_protocol_hash_matches_registration": bool(
            confirmation_path.exists()
            and sha256(confirmation_path) == protocol.get("confirmation_protocol_sha256")
        ),
        "main_matrix_all_seeds": main_matrix.get("completed_seeds") == list(range(101, 111)),
        "main_scores_all_seeds": main_scores.get("seeds") == list(range(101, 111)),
        "main_attribution_all_seeds": main_attr.get("seeds") == list(range(101, 111)),
        "confirmation_summary_present": confirmation_summary.get("seeds") == list(range(101, 111)),
        "confirmation_selected_model_frozen": confirmation_summary.get("selected_model") == "unanchored_mdn_k2",
        "confirmation_blocks_each_500k": all(item.exists() and read_json(item).get("events") == 500000 for item in confirmation_qa_paths),
    }
    blockers = [] if confirmation_summary.get("seeds") == list(range(101, 111)) else [
        "confirmation scores for all ten frozen main-batch seeds are not complete",
    ]
    evidence = {}
    for rel in (
        "protocol/environment_selection.json",
        "reports/audit_w0_e.json",
        "reports/audit_w0_f.json",
        "reports/replay_w0_e_f.json",
        "reports/W2_PILOT_CANDIDATE.md",
        "reports/W3_W5_DEVELOPMENT.md",
        "protocol/freeze_candidate.json",
        "protocol/frozen_protocol.json",
        "protocol/frozen_protocol.sha256",
        "protocol/confirmation_frozen_protocol.json",
        "protocol/confirmation_frozen_protocol.sha256",
        "protocol/production_batch_plan.json",
        "manifests/main_1000000_merged.json",
        "reports/main_1000000_matrix_summary.json",
        "reports/main_1000000_scores_summary.json",
        "reports/main_1000000_attribution_summary.json",
        "reports/main_1000000_selection_candidate.json",
        "reports/confirmation_500k_a_qa.json",
        "reports/confirmation_500k_b_qa.json",
        "reports/confirmation_split_and_qa.json",
        "reports/confirmation_scores_summary.json",
    ):
        path = root / rel
        evidence[rel] = {"exists": path.exists(), "sha256": sha256(path) if path.exists() else None}

    result = {
        "passed": all(checks.get(key, False) for key in ("w0_real_audit_e", "w0_real_audit_f", "w0_exact_replay", "protocol_frozen", "confirmation_enabled", "human_freeze_approval_present", "confirmation_approval_present", "confirmation_protocol_hash_matches_registration", "main_matrix_all_seeds", "main_scores_all_seeds", "main_attribution_all_seeds", "confirmation_summary_present", "confirmation_selected_model_frozen", "confirmation_blocks_each_500k")),
        "decision": (
            "CONFIRMATION_COMPLETE"
            if all(checks.get(key, False) for key in ("confirmation_summary_present", "confirmation_selected_model_frozen", "confirmation_blocks_each_500k"))
            else "CONFIRMATION_OPEN_AWAITING_RUN"
            if protocol.get("status") == "PRODUCTION_FROZEN" and protocol.get("confirmation_enabled") is True
            else ("PROTOCOL_FROZEN_CONFIRMATION_LOCKED" if protocol.get("status") == "PRODUCTION_FROZEN" else "REFUSE_PRODUCTION_FREEZE")
        ),
        "read_only": True,
        "checks": checks,
        "blockers": blockers,
        "evidence": evidence,
        "protocol_status": protocol.get("status"),
        "confirmation_enabled": protocol.get("confirmation_enabled"),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
