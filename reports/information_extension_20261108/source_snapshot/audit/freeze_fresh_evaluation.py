"""Bind the predetermined complete model roster and reviewed fresh analysis.

This orchestration never trains, selects, scores or generates events. It refuses
to replace an existing union, final protocol or environment. Numerical guards
remain in the separately tested fresh_catalog_runner and scoring/analysis code.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent / "low pion research"
ROOTS = {
    "audit": AUDIT, "repo": REPO,
    "core": REPO / "reports/correction_v2_20261107/study",
    "full": REPO / "reports/corrected_formal_extension_20261107/study",
    "information": REPO / "reports/information_extension_20261108/study",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(root_key, path):
    root = ROOTS[root_key].resolve()
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("file must be inside its declared root")
    return dict(root_key=root_key, path=path.relative_to(root).as_posix(),
                sha256=digest(path), bytes=path.stat().st_size)


def write_new(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def build_union():
    roster_path = AUDIT / "EXPECTED_FRESH_MODELS.json"
    roster = read(roster_path)
    freezes = {k: read(ROOTS[k] / "FROZEN_CHECKPOINT_MANIFEST.json")
               for k in ("core", "full", "information")}
    info = freezes["information"]
    if info.get("status") != "COMPLETE_INFORMATION_CHECKPOINT_FREEZE" or len(info["models"]) != 80:
        raise ValueError("all80 fresh information fits must be completely frozen")
    lookup = {k: {m["run_id"]: m for m in d["models"] if not m.get("reused", False)}
              for k, d in freezes.items()}
    models = []
    for declared in roster["models"]:
        row = copy.deepcopy(declared)
        frozen = lookup[row["root_key"]][row["model_id"]]
        row["artifacts"] = copy.deepcopy(frozen["artifacts"])
        for role in ("checkpoint", "predictions", "run_manifest"):
            expected = row["artifacts"][role]
            actual = ref(row["root_key"], ROOTS[row["root_key"]] / expected["path"])
            if any(actual[k] != expected[k] for k in ("path", "sha256", "bytes")):
                raise ValueError(f"frozen artifact mismatch: {row['model_id']}/{role}")
        manifest = read(ROOTS[row["root_key"]] / row["artifacts"]["run_manifest"]["path"])
        row["experiment_identity"] = manifest.get("experiment_identity", frozen.get("experiment_identity"))
        models.append(row)
    if len(models) != 240 or {m["model_id"] for m in models} != set(roster["expected_model_ids"]):
        raise ValueError("predetermined240 roster mismatch")
    union = dict(schema="fresh-model-union/v1", status="COMPLETE_FROZEN_FRESH_MODEL_UNION",
                 frozen_utc=datetime.now(timezone.utc).isoformat(), expected_models=240,
                 predetermined_roster=ref("audit", roster_path), models=models,
                 information_freeze=ref("information", ROOTS["information"] / "FROZEN_CHECKPOINT_MANIFEST.json"),
                 previous_freezes={k: ref(k, ROOTS[k] / "FROZEN_CHECKPOINT_MANIFEST.json") for k in ("core", "full")},
                 selection="Complete predetermined roster; no filtering by validation or fresh outcomes.")
    sys.path.insert(0, str(REPO / "src"))
    from simulation.fresh_catalog_runner import check_model_union
    check_model_union(union, roster["expected_model_ids"], ROOTS)
    write_new(AUDIT / "FROZEN_FRESH_MODEL_UNION.json", union)


def freeze_plan(source_analysis_path, review_path):
    if (AUDIT / "fresh_C_D").exists():
        raise ValueError("formal output path already exists; cannot retrospectively freeze")
    budget = read(AUDIT / "AUTHORIZATION_AND_BUDGET.json")
    if time.time() >= budget["deadline_unix"]:
        raise ValueError("original12hour deadline reached")
    analysis_path = REPO / "protocol/information_statistical_analysis_20261108.json"
    if read(analysis_path).get("status") != "FROZEN_FRESH_STATISTICAL_ANALYSIS":
        raise ValueError("analysis must already be independently reviewed and frozen")
    review = read(review_path)
    if (review.get("status") != "PASS_COMPLETE_FRESH_IMPLEMENTATION_REVIEW"
            or review.get("unresolved_blocking_findings") != []):
        raise ValueError("independent final implementation review must pass")
    checked_files = review.get("checked_files", [])
    if not checked_files:
        raise ValueError("final review must bind the actual implementation bytes")
    for checked in checked_files:
        actual = ref(checked["root_key"], ROOTS[checked["root_key"]] / checked["path"])
        if actual != checked:
            raise ValueError(f"implementation changed after independent review: {checked['path']}")
    plan = read(REPO / "protocol/fresh_catalog_20261108.draft.json")
    plan.pop("draft_note", None)
    plan["status"] = "FROZEN_FRESH_CATALOG"
    plan["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    plan["expected_model_ids"] = read(AUDIT / "EXPECTED_FRESH_MODELS.json")["expected_model_ids"]
    plan["model_union"] = ref("audit", AUDIT / "FROZEN_FRESH_MODEL_UNION.json")
    plan["analysis_plan"] = ref("repo", analysis_path)
    plan["source_validation_analysis"] = ref("audit", source_analysis_path)
    plan["independent_implementation_review"] = ref("audit", review_path)
    plan["budget_record"] = ref("audit", AUDIT / "AUTHORIZATION_AND_BUDGET.json")
    source_paths = {r["path"] for r in plan["source_files"] if r["root_key"] == "repo"}
    # Bind all actual local Python dependency modules, not just an optimistic
    # hand-picked subset. Private caches and numerical outputs are not sources.
    source_paths.update(p.relative_to(REPO).as_posix() for p in (REPO / "src").rglob("*.py"))
    source_paths.update([
        "protocol/information_study_20261108.json",
        "reports/information_extension_20261108/PROTOCOL_FREEZE.json",
        "protocol/information_statistical_analysis_20261108.json",
    ])
    plan["source_files"] = [ref("repo", REPO / p) for p in sorted(source_paths)]
    plan["source_files"] += [ref("audit", AUDIT / p) for p in
                             ("RESEARCH_DECISIONS.json", "EXPECTED_FRESH_MODELS.json", Path(__file__).name)]
    final_path = REPO / "protocol/fresh_catalog_20261108.json"
    env_path = REPO / "protocol/fresh_environment_20261108.json"
    if final_path.exists() or env_path.exists():
        raise ValueError("refusing to replace final freeze or environment")
    write_new(final_path, plan)
    environment = dict(
        schema="fresh-research-environment/v1", status="PRODUCTION_FROZEN",
        confirmation_enabled=True, frozen_protocol_sha256=digest(final_path),
        confirmation_protocol_sha256=digest(final_path),
        confirmation_protocol_path="protocol/fresh_catalog_20261108.json",
        confirmation_approval=plan["authorization"],
        human_freeze_approval={
            "source": "user_message", "approval_text": plan["authorization"]["approval_text"],
            "scope": "User authorized the new research and fresh Geant4 workflow within8-12hours.",
            "boundary": "Workflow authorization, not a claim that the user manually reviewed these artifact bytes or implementation. Automated independent checks are recorded separately.",
        },
    )
    write_new(env_path, environment)
    print(json.dumps({"protocol": str(final_path), "sha256": digest(final_path), "environment": str(env_path)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("union", "plan"))
    p.add_argument("--source-analysis", type=Path)
    p.add_argument("--review", type=Path)
    a = p.parse_args()
    if a.action == "union":
        build_union()
    else:
        if a.source_analysis is None or a.review is None:
            p.error("plan requires --source-analysis and --review")
        freeze_plan(a.source_analysis, a.review)
