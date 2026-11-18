"""Generate a new Geant4 catalog only after a complete model/analysis freeze.

This is a new gate, layered on the unchanged historical confirmation_guard.py.
It never repurposes the old A/B authorization or changes historical protocols.
All file references are relative to explicitly supplied roots; provenance paths
inside older model manifests are not used to locate files. Check-only is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_contract.beam_schedule import CELLS, chunk_transport_seeds, derive_seed


REPO = Path(__file__).resolve().parents[2]
REQUIRED_SOURCES = {
    "protocol/confirmation_guard.py", "src/simulation/fresh_catalog_runner.py",
    "src/simulation/make_beam_chunks.py", "src/simulation/run_chunks.py",
    "src/simulation/w0_smoke.py", "src/simulation/audit_w0.py",
    "src/simulation/audit_cut_log.py", "src/simulation/audit_step_limit.py",
    "src/data_contract/beam_schedule.py", "src/data_contract/primary_visit.py",
    "src/data_contract/merge_chunks.py",
}
SIMULATION = {
    "physics_list": "FTFP_BERT", "material": "G4_Si",
    "world_material": "G4_Galactic", "production_cut_mm": 0.01,
    "max_step_over_thickness": 0.1, "record_secondary_steps": True,
    "momentum_range_GeV_c": [0.3, 1.0], "momentum_bins": 7,
    "thickness_mm": [0.1, 0.3, 1.0], "pdg": [211, -211], "cells": 42,
    "eligibility": "forward_first_visit_nonaborted_positive_exit_momentum",
    "target": "delta_p_GeV_c=p_in_GeV_c-p_out_GeV_c; Y<p_in; negative_Y_allowed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(record: dict, roots: dict[str, Path]) -> Path:
    root = roots[record["root_key"]].resolve(strict=True)
    relative = Path(record["path"])
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("frozen file reference escapes its declared root")
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"frozen artifact hash/size mismatch: {record['path']}")
    return path


def check_model_union(union: dict, expected_ids: list[str], roots: dict[str, Path]) -> dict:
    if union.get("schema") != "fresh-model-union/v1" or union.get("status") != "COMPLETE_FROZEN_FRESH_MODEL_UNION":
        raise ValueError("complete fresh model union is required")
    models = union["models"]
    ids = [row["model_id"] for row in models]
    if (union.get("expected_models") != 240 or len(models) != 240
            or len(set(ids)) != 240 or len(expected_ids) != 240
            or len(set(expected_ids)) != 240 or set(ids) != set(expected_ids)):
        raise ValueError("the full prespecified union must contain exactly 240 unique models")
    cohorts = {name: sum(row.get("cohort") == name for row in models)
               for name in ("information80", "legacy160")}
    if cohorts != {"information80": 80, "legacy160": 160}:
        raise ValueError("union must include all information80 and legacy160 models")
    checkpoint_paths = set()
    checked = 0
    for row in models:
        artifacts = row["artifacts"]
        if not {"checkpoint", "predictions", "run_manifest"}.issubset(artifacts):
            raise ValueError("checkpoint, validation predictions and run manifest are required")
        for role, record in artifacts.items():
            bound = dict(record, root_key=record.get("root_key", row["root_key"]))
            path = verify_file(bound, roots)
            if role == "checkpoint":
                if path in checkpoint_paths:
                    raise ValueError("reused checkpoint counted twice in the unique model union")
                checkpoint_paths.add(path)
            checked += 1
    return {"unique_models": len(models), "artifact_files_checked": checked, "cohorts": cohorts}


def planned_stratum_counts(events: int, beam_seed: int) -> list[int]:
    """Predict allocation counts only, without sampling any incident events."""
    complete, remaining = divmod(events, 42)
    final_permutation = list(CELLS)
    random.Random(derive_seed(beam_seed, "cell-block", complete)).shuffle(final_permutation)
    extra = set(final_permutation[:remaining])
    return [complete + int(cell in extra) for cell in CELLS]


def check_seeds(plan: dict, inventory: dict) -> None:
    old_beam = set(inventory["beam_seeds"])
    old_master = set(inventory["transport_master_seeds"])
    old_transport = set(inventory["transport_seeds"])
    old_runs = set(inventory["run_ids"])
    smoke = plan["engineering_smoke"]
    old_beam.add(smoke["beam_seed"])
    old_master.add(smoke["transport_master_seed"])
    old_transport.update(chunk_transport_seeds(smoke["transport_master_seed"], list(range(42))).values())
    old_runs.add(smoke["run_id"])
    blocks = plan["catalogs"]
    if not blocks or len(blocks) > 2:
        raise ValueError("freeze one or two fresh 500k catalogs before generation")
    names = set()
    for block in blocks:
        if block["block_id"] not in ("C", "D") or block["block_id"] in names:
            raise ValueError("fresh block identifiers must be unique C/D")
        names.add(block["block_id"])
        if block["events"] != 500_000:
            raise ValueError("each formal catalog must have the frozen 500000 events")
        if not block["run_id"] or any(c in block["run_id"] for c in (":", "/", "\\")):
            raise ValueError("invalid fresh run ID")
        for key in ("beam_seed", "transport_master_seed"):
            if type(block[key]) is not int or not 0 < block[key] < 2**31:
                raise ValueError("seeds must be positive 31-bit integers")
        if block.get("launched_events_by_chunk") != planned_stratum_counts(block["events"], block["beam_seed"]):
            raise ValueError("the 42 fixed launch-stratum counts must be frozen before generation")
        transport = set(chunk_transport_seeds(block["transport_master_seed"], list(range(42))).values())
        if (block["beam_seed"] in old_beam or block["transport_master_seed"] in old_master
                or transport & old_transport or block["run_id"] in old_runs):
            raise ValueError("fresh catalog reuses a historical, smoke or other fresh seed/run ID")
        old_beam.add(block["beam_seed"])
        old_master.add(block["transport_master_seed"])
        old_transport.update(transport)
        old_runs.add(block["run_id"])


def validate_freeze(plan_path: Path, environment_path: Path, roots: dict[str, Path]) -> dict:
    """Read-only freeze validation, performed before any formal path is created."""
    plan = read_json(plan_path)
    if plan.get("schema") != "fresh-catalog-protocol/v1" or plan.get("status") != "FROZEN_FRESH_CATALOG":
        raise ValueError("draft or incomplete fresh protocol cannot authorize generation")
    if plan.get("simulation") != SIMULATION:
        raise ValueError("fresh transport/eligible-event contract differs from the fixed original design")
    for key in ("generation_forecast_seconds", "reserved_scoring_analysis_seconds"):
        value = plan.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"a finite positive frozen runtime allowance is required: {key}")
    approval = plan.get("authorization", {})
    if approval.get("source") != "user_message" or not approval.get("approval_text"):
        raise ValueError("the new research authorization must be recorded explicitly")
    source_paths = {r["path"] for r in plan["source_files"] if r["root_key"] == "repo"}
    if not REQUIRED_SOURCES.issubset(source_paths):
        raise ValueError("source freeze omits a generator, guard or validator dependency")
    for record in plan["source_files"]:
        verify_file(record, roots)
    config = read_json(environment_path)
    actual = sha256(plan_path)
    reference = Path(config.get("confirmation_protocol_path", ""))
    if not reference.is_absolute():
        reference = environment_path.resolve().parent.parent / reference
    if (reference.resolve() != plan_path.resolve()
            or config.get("confirmation_protocol_sha256") != actual
            or config.get("frozen_protocol_sha256") != actual
            or config.get("confirmation_approval") != approval):
        raise ValueError("environment must bind this new frozen protocol and its authorization")
    guard = subprocess.run([sys.executable, "-B", str(REPO / "protocol/confirmation_guard.py"),
                            "--environment", str(environment_path)], capture_output=True, text=True, check=False)
    if guard.returncode != 0 or read_json_text(guard.stdout).get("allowed") is not True:
        raise ValueError(f"historical confirmation guard refuses new environment: {guard.stdout} {guard.stderr}")
    inventory = read_json(verify_file(plan["seed_inventory"], roots))
    check_seeds(plan, inventory)
    analysis = verify_file(plan["analysis_plan"], roots)
    analysis_record = read_json(analysis)
    if (analysis_record.get("schema") != "fresh-catalog-statistical-analysis/v1"
            or analysis_record.get("status") != "FROZEN_FRESH_STATISTICAL_ANALYSIS"):
        raise ValueError("a semantically frozen statistical analysis plan is required; drafts are refused")
    union_path = verify_file(plan["model_union"], roots)
    union = read_json(union_path)
    union_check = check_model_union(union, plan["expected_model_ids"], roots)
    information_freeze_path = verify_file(union["information_freeze"], roots)
    information_freeze = read_json(information_freeze_path)
    information_models = [r for r in union["models"] if r["cohort"] == "information80"]
    if (information_freeze.get("schema") != "information-checkpoint-freeze/v1"
            or information_freeze.get("status") != "COMPLETE_INFORMATION_CHECKPOINT_FREEZE"
            or information_freeze.get("expected_cells") != 80
            or len(information_freeze.get("models", [])) != 80
            or {r["run_id"] for r in information_freeze["models"]} != {r["model_id"] for r in information_models}):
        raise ValueError("the new80 freeze does not match the full declared union")
    frozen_by_id = {r["run_id"]: r for r in information_freeze["models"]}
    for model in information_models:
        for role in ("checkpoint", "predictions", "run_manifest"):
            frozen = frozen_by_id[model["model_id"]]["artifacts"][role]
            selected = model["artifacts"][role]
            if any(frozen[k] != selected[k] for k in ("path", "sha256", "bytes")):
                raise ValueError("new80 artifact differs between model union and information freeze")
    source_validation_path = verify_file(plan["source_validation_analysis"], roots)
    source_validation = read_json(source_validation_path)
    if (source_validation.get("schema") != "information-source-analysis/v1"
            or source_validation.get("status") != "COMPLETE_INFORMATION_SOURCE_ANALYSIS"
            or source_validation.get("expected_models") != 80
            or source_validation.get("model_count") != 80
            or source_validation.get("all_validation_predictions_replayed") is not True
            or source_validation.get("study_freeze_sha256") != sha256(information_freeze_path)
            or source_validation.get("protocol_sha256") != information_freeze.get("protocol_sha256")):
        raise ValueError("complete independently replayed source-validation analysis of the same new80 freeze is required")
    budget = read_json(verify_file(plan["budget_record"], roots))
    if (budget["maximum_wall_seconds"] != 43200
            or budget["deadline_unix"] != budget["started_unix"] + budget["maximum_wall_seconds"]):
        raise ValueError("global 12-hour budget record is inconsistent")
    if time.time() >= budget["deadline_unix"]:
        raise TimeoutError("the original global research deadline has expired")
    return {"status": "PASS_FRESH_FREEZE_GATE", "plan_sha256": actual,
            "environment_sha256": sha256(environment_path), "union": union_check,
            "analysis_sha256": sha256(analysis), "legacy_guard": read_json_text(guard.stdout),
            "source_validation_analysis_sha256": sha256(source_validation_path),
            "information_freeze_sha256": sha256(information_freeze_path),
            "deadline_unix": budget["deadline_unix"]}


def read_json_text(text: str):
    return json.loads(text)


def verify_runtime(plan: dict, data_root: Path) -> dict:
    """Rehash installed physics data, then check the actual native runtime."""
    expected = plan["runtime"]
    names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if names != sorted(expected["dataset_names"]):
        raise ValueError("installed Geant4 dataset directory set differs from the freeze")
    digest = hashlib.sha256()
    files = total = 0
    for name in names:
        for path in sorted(p for p in (data_root / name).rglob("*") if p.is_file()):
            if path.is_symlink():
                raise ValueError("physics dataset symlinks are not allowed")
            digest.update(f"{path.relative_to(data_root).as_posix()}\0{sha256(path)}\n".encode())
            files += 1
            total += path.stat().st_size
    if digest.hexdigest() != expected["dataset_tree_sha256"]:
        raise ValueError("Geant4 physics dataset tree differs from the freeze")
    code = ("import json,sys,geant4_pybind as g; "
            "print(json.dumps({'runtime_number':int(g.G4VERSION_NUMBER),"
            "'runtime_string':str(g.G4Version),'binding_path':g.__file__,"
            "'python':'.'.join(map(str,sys.version_info[:3]))}))")
    env = dict(os.environ, GEANT4_DATA_DIR=str(data_root.resolve()))
    proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, input="n\n",
                          capture_output=True, text=True, timeout=120, check=True)
    actual = json.loads(proc.stdout.splitlines()[-1])
    actual["binding_sha256"] = sha256(Path(actual["binding_path"]))
    for key in ("runtime_number", "runtime_string", "binding_sha256", "python"):
        if actual[key] != expected[key]:
            raise ValueError(f"native Geant4 runtime mismatch: {key}")
    return {**actual, "dataset_tree_sha256": digest.hexdigest(), "dataset_files": files, "dataset_bytes": total}


def run_stage(command: list[str], log: Path, deadline_unix: float) -> dict:
    started = time.monotonic()
    remaining = deadline_unix - time.time()
    if remaining <= 0:
        raise TimeoutError("original global research deadline reached before generation stage")
    with log.open("xb") as stream:
        kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                  else {"start_new_session": True})
        process = subprocess.Popen([sys.executable, "-B", *command], stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError("global deadline reached; generation process tree stopped, partial outputs retained")
    record = {"command": command, "wall_seconds": time.monotonic() - started, "returncode": returncode}
    if returncode:
        raise RuntimeError(f"fresh generation failed; retained partial evidence, inspect {log}")
    return record


def stratum_counts(catalog: dict, event_path: Path) -> dict:
    """Retain launched and S0 counts for unconditional stratified resampling."""
    if catalog.get("cells") != 42 or len(catalog["chunks"]) != 42:
        raise ValueError("counts require the complete 42-cell catalog")
    rows = [{"chunk_id": c["chunk_id"], "momentum_bin": c["momentum_bin"],
             "group_id": c["momentum_bin"] * 6 + [0.1, 0.3, 1.0].index(c["thickness_mm"]) * 2 + int(c["pdg"] == 211),
             "pdg": c["pdg"], "thickness_mm": c["thickness_mm"],
             "launched": 0, "eligible": 0, "ineligible": 0, "aborted": 0, "statuses": {}}
            for c in catalog["chunks"]]
    lookup = {(r["momentum_bin"], r["pdg"], r["thickness_mm"]): r for r in rows}
    with event_path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            row = lookup[(event["momentum_bin"], event["pdg"], event["thickness_mm"])]
            if type(event["S1"]) is not bool or type(event.get("aborted")) is not bool:
                raise ValueError("event ledger lacks boolean eligibility/abort contract")
            row["launched"] += 1
            row["eligible" if event["S1"] else "ineligible"] += 1
            row["aborted"] += int(event["aborted"])
            status = event["status"]
            row["statuses"][status] = row["statuses"].get(status, 0) + 1
    for row, chunk in zip(rows, catalog["chunks"]):
        if row["launched"] != chunk["events"]:
            raise ValueError("stratum launched-count mismatch")
        if row["aborted"]:
            raise ValueError("technical event abortion: refuse formal catalog, retain ledger for investigation")
        row["launched_events"] = row["launched"]
        row["eligible_events"] = row["eligible"]
    return {"schema": "fresh-catalog-stratum-counts/v1", "status": "COMPLETE_42_STRATUM_COUNTS",
            "run_id": catalog["run_id"], "events_sha256": sha256(event_path), "cells": rows,
            "group_id_rule": "momentum_bin*6 + thickness_index*2 + charge_index; thickness=[0.1,0.3,1.0]; charge=[-1,+1]",
            "launched": sum(r["launched"] for r in rows), "eligible": sum(r["eligible"] for r in rows),
            "ineligible": sum(r["ineligible"] for r in rows), "technical_aborted": 0,
            "policy": "S0 is retained as a null-score atom for paired launched-event bootstrap; not a technical failure"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--artifact-root", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--geant4-data-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    roots = {"repo": REPO}
    for raw in args.artifact_root:
        key, value = raw.split("=", 1)
        if key in roots:
            parser.error("duplicate or reserved artifact root")
        roots[key] = Path(value)
    gate = validate_freeze(args.plan, args.environment, roots)
    plan = read_json(args.plan)
    runtime = verify_runtime(plan, args.geant4_data_dir.resolve(strict=True))
    if time.time() >= gate["deadline_unix"]:
        raise TimeoutError("original global deadline expired during environment verification")
    if args.check_only:
        print(json.dumps({**gate, "runtime": runtime, "no_event_files_created": True}, indent=2))
        return 0
    if args.output_root is None:
        parser.error("generation requires --output-root")
    output = args.output_root.resolve()
    if output.is_relative_to(REPO) or any(output.is_relative_to(p.resolve()) for k, p in roots.items() if k != "audit"):
        raise ValueError("fresh raw output must be outside source and frozen artifact roots")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError("fresh generation never overwrites or silently resumes an existing output")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    reserve = int(plan["minimum_free_bytes"])
    if reserve < len(plan["catalogs"]) * 20 * 2**30 or shutil.disk_usage(ancestor).free < reserve:
        raise ValueError("insufficient frozen disk allowance (at least 20 GiB per 500k block)")
    if time.time() + plan["generation_forecast_seconds"] + plan["reserved_scoring_analysis_seconds"] >= gate["deadline_unix"]:
        raise TimeoutError("frozen generation forecast and remaining scoring reserve exceed original global deadline")
    # The complete, successful gate precedes every formal input/output creation.
    output.mkdir(parents=True)
    receipt = {**gate, "runtime": runtime, "generated_utc": datetime.now(timezone.utc).isoformat(),
               "status": "FRESH_GENERATION_STARTED_AFTER_COMPLETE_FREEZE", "stages": [], "catalogs": []}
    (output / "GENERATION_GATE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    for block in plan["catalogs"]:
        folder = output / block["block_id"]
        folder.mkdir()
        commands = [
            [str(REPO / "src/simulation/make_beam_chunks.py"), "--run-id", block["run_id"],
             "--events", str(block["events"]), "--beam-seed", str(block["beam_seed"]),
             "--transport-master-seed", str(block["transport_master_seed"]), "--output", str(folder / "beam")],
            [str(REPO / "src/simulation/run_chunks.py"), str(folder / "beam/catalog.json"),
             "--output", str(folder / "runs"), "--logs", str(folder / "logs"),
             "--geant4-data-dir", str(args.geant4_data_dir), "--cut-mm", "0.01", "--step-fraction", "0.1", "--record-secondary-steps"],
            [str(REPO / "src/data_contract/merge_chunks.py"), str(folder / "beam/catalog.json"),
             str(folder / "runs"), "--output", str(folder / "merged")],
        ]
        for index, command in enumerate(commands):
            receipt["stages"].append({"block_id": block["block_id"], **run_stage(command, folder / f"stage_{index}.log", gate["deadline_unix"])})
        merged = read_json(folder / "merged/manifest.json")
        if merged["events"] != block["events"] or merged["cells"] != 42:
            raise ValueError("merged fresh event/cell count mismatch")
        counts = stratum_counts(read_json(folder / "beam/catalog.json"), folder / "merged/events.jsonl")
        (folder / "fresh_catalog_manifest.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        receipt["catalogs"].append({**block, "catalog_sha256": sha256(folder / "beam/catalog.json"),
                                    "merged_manifest_sha256": sha256(folder / "merged/manifest.json"),
                                    "stratum_counts_sha256": sha256(folder / "fresh_catalog_manifest.json")})
    # A post-run check detects modifications while generation was running.
    validate_freeze(args.plan, args.environment, roots)
    receipt.update(status="COMPLETE_FRESH_GEANT4_CATALOGS_NO_MODEL_SCORING", wall_seconds=time.monotonic() - started,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "GENERATION_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "catalogs": len(receipt["catalogs"]), "wall_seconds": receipt["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
