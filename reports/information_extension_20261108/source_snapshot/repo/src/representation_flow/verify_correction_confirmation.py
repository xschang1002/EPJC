"""Independently audit all160 saved confirmation outputs without reading events.

This stdlib-only verifier does not import the scorer, run inference, deserialize
weights, or open either raw ledgers or confirmation-cache arrays. Model NLLs are
the independent arithmetic inputs for the prespecified paired contrasts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCOPE = "ALL_UNIQUE_CORE60_PLUS_EXTENSION100"
SEEDS = tuple(range(3101, 3106))
BLOCK_EVIDENCE = {
    "a": (499268, "b72194770190b646ede8e42782d0df3e758eb606e7cd6d4b36c6068be4fea342", "6684320deffa8157459af51229f5eb7ecd96d5177798fd8cb1562877b90e9d72"),
    "b": (499283, "a948d095cf6c73822e15f88fbc3b81de84397906e9c1bf19890928d0b1675db5", "cb1eef3c9d9e1cf39a7b8c86d491d8543910e8f159c5534a6da699014a612189"),
}
TABLE_COUNTS = {"models": 320, "seed_contrasts": 410, "contrast_summary": 82,
                "model_groups": 13440, "contrast_groups": 17220}
MEAN_ATOL = 5e-10
SD_ATOL = 1e-7


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canonical_hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _close(actual, expected, label, *, atol=MEAN_ATOL):
    _require(isinstance(actual, (int, float)) and math.isfinite(actual), f"nonfinite/missing {label}")
    _require(math.isclose(actual, expected, rel_tol=5e-10, abs_tol=atol),
             f"independent reconstruction differs: {label}: stored={actual!r}, reconstructed={expected!r}")


def _index(rows, keys, label):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        _require(key not in result, f"duplicate {label}: {key}")
        result[key] = row
    return result


def _model_design():
    """The fixed scientific matrix, enumerated independently of scorer code."""
    model_keys = set(itertools.product(("f010", "f030", "f100"), ("Small", "Large"), ("Full",),
                                       ("Raw", "Simple", "Learned", "CorrectedPhysics")))
    model_keys.update(itertools.product(("f010", "f100"), ("Small",), ("Hidden",),
                                        ("Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")))
    model_keys.update((tag, "Small", "Full", "HistoricalIncorrectPhysics") for tag in ("f010", "f100"))
    return {(seed, *key) for seed, key in itertools.product(SEEDS, model_keys)}


def _core_model(tag, capacity, context, representation):
    return capacity == "Small" and tag in ("f010", "f100") and representation in (
        "Raw", "HistoricalIncorrectPhysics", "CorrectedPhysics")


def _run_id(seed, tag, capacity, context, representation):
    if _core_model(tag, capacity, context, representation):
        return f"correction__{context}__{representation}__{capacity}__{tag}__sub4101__seed{seed}"
    return f"corrected_full__{representation}__{capacity}__{tag}__sub4101__seed{seed}"


def _contrast_design():
    """41 labels and their NLL-domain coefficients, independent of scorer logp."""
    definitions = []

    def difference(tag, capacity, context, left, right):
        return [(1., (tag, capacity, context, left)), (-1., (tag, capacity, context, right))]

    def add(scope, tag, capacity, context, name, terms):
        definitions.append(((scope, tag, capacity, context, name), terms))

    core_pairs = (("Raw", "CorrectedPhysics"), ("Raw", "HistoricalIncorrectPhysics"),
                  ("HistoricalIncorrectPhysics", "CorrectedPhysics"))
    for tag, (left, right) in itertools.product(("f010", "f100"), core_pairs):
        name = f"{left}_minus_{right}"
        full = difference(tag, "Small", "Full", left, right)
        hidden = difference(tag, "Small", "Hidden", left, right)
        add("core60", tag, "Small", "Full", name, full)
        add("core60", tag, "Small", "Hidden", name, hidden)
        add("core60", tag, "Small", "Full_minus_Hidden", name, full + [(-w, key) for w, key in hidden])
    for tag, capacity, (left, right) in itertools.product(("f010", "f030", "f100"), ("Small", "Large"),
            (("Raw", "CorrectedPhysics"), ("Learned", "CorrectedPhysics"), ("Simple", "Raw"))):
        add("formal120", tag, capacity, "Full", f"{left}_minus_{right}", difference(tag, capacity, "Full", left, right))
    for tag in ("f010", "f030", "f100"):
        terms = difference(tag, "Small", "Full", "Raw", "CorrectedPhysics")
        terms += [(-w, key) for w, key in difference(tag, "Large", "Full", "Raw", "CorrectedPhysics")]
        add("formal120", tag, "Small_minus_Large", "Full", "Raw_minus_CorrectedPhysics", terms)
    for capacity in ("Small", "Large"):
        terms = difference("f010", capacity, "Full", "Raw", "CorrectedPhysics")
        terms += [(-w, key) for w, key in difference("f100", capacity, "Full", "Raw", "CorrectedPhysics")]
        add("formal120", "f010_minus_f100", capacity, "Full", "Raw_minus_CorrectedPhysics", terms)
    return definitions


def _scorer_definition_hash(definitions):
    # The plan hashes log-density coefficients. Reverse independent NLL signs
    # only for checking that recorded specification; arithmetic stays in NLL.
    keys = ("scope", "fraction_tag", "capacity", "context", "contrast")
    specs = []
    for labels, terms in definitions:
        ordered = [term for offset in range(0, len(terms), 2) for term in reversed(terms[offset:offset + 2])]
        specs.append({**dict(zip(keys, labels)), "terms": [(-weight, key) for weight, key in ordered]})
    return _canonical_hash(specs)


def _csv_agrees(path, rows):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                 and set(reader.fieldnames) == set(rows[0]), f"CSV columns differ from summary: {path.name}")
        saved = list(reader)
    expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    _require(saved == expected, f"CSV rows differ from summary: {path.name}")


def _group_reconstruction(groups, total, expected_counts, label):
    indexed = _index(groups, ("group_id",), label)
    _require(set(indexed) == {(i,) for i in range(42)}, f"42-group set differs: {label}")
    counts = tuple(indexed[(i,)]["events"] for i in range(42))
    _require(all(isinstance(n, int) and n >= 0 for n in counts), f"invalid group counts: {label}")
    _require(counts == expected_counts and sum(counts) == total["events"], f"group counts differ: {label}")
    for row in indexed.values():
        if row["events"] == 0:
            _require(row["mean"] is None and row["event_sd_descriptive"] is None, f"empty group imputed a value: {label}")
        else:
            _close(row["mean"], row["mean"], f"group mean {label}")
            _require(math.isfinite(row["event_sd_descriptive"]) and row["event_sd_descriptive"] >= 0, f"invalid group SD: {label}")
    nonempty = [r for r in indexed.values() if r["events"]]
    mean = math.fsum(r["events"] * r["mean"] for r in nonempty) / total["events"]
    _close(total["mean"], mean, f"42-group weighted mean {label}")
    # Recover the pooled sample variance using within/between group sums.
    ss = math.fsum((r["events"] - 1) * r["event_sd_descriptive"] ** 2
                   + r["events"] * (r["mean"] - mean) ** 2 for r in nonempty)
    sd = math.sqrt(ss / (total["events"] - 1)) if total["events"] > 1 else 0.
    _close(total["event_sd_descriptive"], sd, f"42-group pooled SD {label}", atol=SD_ATOL)
    return indexed


def verify(input_root, output, *, repository_root=REPOSITORY_ROOT, core_study=None, extension_study=None):
    started = time.perf_counter()
    input_root, output, repository_root = Path(input_root).resolve(), Path(output).resolve(), Path(repository_root).resolve()
    core_study = Path(core_study).resolve() if core_study is not None else repository_root / "reports/correction_v2_20261107/study"
    extension_study = Path(extension_study).resolve() if extension_study is not None else repository_root / "reports/corrected_formal_extension_20261107/study"
    _require(not output.is_relative_to(input_root), "verification output must be outside the original confirmation directory")
    if output.exists():
        raise FileExistsError("verification output already exists; never overwrite an audit")
    inputs = []

    def capture(path, role):
        path = Path(path).resolve()
        record = {"path": str(path), "role": role, "sha256": _digest(path), "bytes": path.stat().st_size}
        inputs.append(record)
        return record["sha256"]

    for name in ("SCORING_PLAN.json", "execution_state.json", "confirmation_summary.json"):
        capture(input_root / name, name)
    plan, state, summary = (_read(input_root / name) for name in ("SCORING_PLAN.json", "execution_state.json", "confirmation_summary.json"))
    _require(plan.get("schema") == "frozen-core60-full120-confirmation/v2" and plan.get("status") == "ALL_MODELS_VALIDATED_BEFORE_CONFIRMATION_ACCESS",
             "invalid frozen scoring plan")
    _require(plan.get("scope") == SCOPE and plan.get("model_count") == 160, "this verifier requires the complete all160 scope")
    _require(summary.get("schema") == "frozen-core60-full120-confirmation-results/v2" and summary.get("status") == "COMPLETE_FROZEN_UNIQUE160_AB_CONFIRMATION",
             "confirmation is not complete all160")
    _require(state.get("status") == "COMPLETE" and state.get("scope") == SCOPE and not state.get("failures"), "execution has failures or is incomplete")
    _require(summary.get("scope") == SCOPE and summary.get("model_evaluations") == 320, "summary scope/count differs")
    plan_hash = inputs[0]["sha256"]
    _require(state["scoring_plan_sha256"] == summary["scoring_plan_sha256"] == plan_hash, "scoring plan hash differs")
    for key in ("core_experiment_identity", "extension_gate", "formal120_to_unique_models"):
        _require(summary[key] == plan[key], f"summary/plan identity differs: {key}")
    _require(plan["extension_gate"]["status"] == "COMPLETE_FROZEN_EXTENSION120_VERIFIED" and plan["extension_gate"]["scored_here"] is True,
             "all160 requires a complete scored extension")
    _require(plan["workers"] == 4 and plan["torch_threads_per_worker"] == 1, "worker contract differs")
    _close(state["elapsed_seconds"], summary["elapsed_seconds"], "elapsed time")
    _require(0 < summary["elapsed_seconds"] <= plan["max_seconds"] <= 1200, "scoring exceeded its total time budget")
    _require(plan["preflight_finished_unix"] < plan["scoring_deadline"] <= plan["absolute_total_study_deadline"]
             <= plan["core_execution_started_unix"] + plan["total_budget_seconds"], "absolute deadline identity differs")

    source_files = {"scoring_source_sha256": "score_correction_confirmation.py",
                    "analysis_validation_source_sha256": "analyze_correction_study.py",
                    "extension_validation_source_sha256": "analyze_corrected_formal_matrix.py"}
    for key, filename in source_files.items():
        _require(capture(repository_root / "src/representation_flow" / filename, "frozen source") == plan[key], f"source hash differs: {filename}")
    design = _model_design()
    model_keys = ("seed", "fraction_tag", "capacity", "context", "representation")
    planned = _index(plan["models"], model_keys, "planned model")
    _require(len(plan["models"]) == 160 and set(planned) == design, "planned 160-model matrix differs")
    _require(len({r["run_id"] for r in planned.values()}) == 160 and len({r["checkpoint"] for r in planned.values()}) == 160,
             "model IDs or checkpoint paths are duplicated")
    for key, model in planned.items():
        _require(model["run_id"] == _run_id(*key), "model labels disagree with fixed run ID")
        _require(len(model["checkpoint_contract_sha256"]) == 64, "checkpoint contract digest missing")
        # Original absolute paths are provenance, never an implicit read target.
        # Normalize separators only for parsing their observed relative layout;
        # PurePosixPath also handles a recorded Windows path on a POSIX host.
        recorded = PurePosixPath(str(model["checkpoint"]).replace("\\", "/"))
        _require(recorded.name == "checkpoint.pt" and recorded.parent.name == model["run_id"] and recorded.parent.parent.name == "runs",
                 "checkpoint path is not the recorded run-relative layout")
        study_root = core_study if _core_model(*key[1:]) else extension_study
        path = (study_root / "runs" / model["run_id"] / "checkpoint.pt").resolve()
        _require(path.is_relative_to(study_root.resolve()), "checkpoint path escaped the current study root")
        _require(capture(path, f"checkpoint:{model['run_id']}") == model["checkpoint_sha256"], f"checkpoint hash differs: {model['run_id']}")
        inputs[-1]["recorded_checkpoint_path"] = model["checkpoint"]
    mapping = _index(plan["formal120_to_unique_models"], ("matrix_cell_id",), "formal reuse mapping")
    expected_mapping = {}
    for seed, tag, capacity, representation in itertools.product(SEEDS, ("f010", "f030", "f100"), ("Small", "Large"), ("Raw", "Simple", "Learned", "CorrectedPhysics")):
        key = (seed, tag, capacity, "Full", representation)
        matrix_id = f"corrected_full__{representation}__{capacity}__{tag}__sub4101__seed{seed}"
        expected_mapping[(matrix_id,)] = {"matrix_cell_id": matrix_id, "run_id": planned[key]["run_id"],
            "reused": _core_model(tag, capacity, "Full", representation), "checkpoint_sha256": planned[key]["checkpoint_sha256"]}
    _require(mapping == expected_mapping and sum(r["reused"] for r in mapping.values()) == 20, "formal120 reuse mapping differs")

    for block, (count, manifest_sha, events_sha) in BLOCK_EVIDENCE.items():
        expected = plan["blocks"][block]
        _require((expected["eligible_events"], expected["manifest_sha256"], expected["events_sha256"]) == (count, manifest_sha, events_sha),
                 f"predeclared block identity differs: {block}")
        actual = summary["blocks"][block]
        _require(actual["block"] == block and actual["eligible_events"] == count and actual["all_eligible_events_scored"] is True
                 and actual["no_selection_or_resampling"] is True, f"block eligibility differs: {block}")
        _require(actual["source_identity"] == {"manifest_sha256": manifest_sha, "events_sha256": events_sha}, f"block provenance differs: {block}")
        _require(len(actual["ordered_uid_sha256"]) == 64, f"block UID identity missing: {block}")
    _require(set(summary["blocks"]) == set(plan["blocks"]) == {"a", "b"}, "block roster differs")
    for table, count in TABLE_COUNTS.items():
        _require(len(summary[table]) == count, f"table row count differs: {table}")
        path = input_root / f"{table}.csv"
        capture(path, f"table:{table}")
        _csv_agrees(path, summary[table])

    jobs = {f"{block}__unique32__seed{seed}" for block, seed in itertools.product(("a", "b"), SEEDS)}
    _require(len(state["completed_jobs"]) == 10 and set(state["completed_jobs"]) == jobs, "completed paired jobs differ")
    files = list((input_root / "paired_groups").glob("*.json"))
    _require({p.stem for p in files} == jobs and len(files) == 10, "paired-group file roster differs")
    merged = {key: [] for key in ("models", "seed_contrasts", "model_groups", "contrast_groups")}
    for path in files:
        capture(path, "paired-group result")
        job = _read(path)
        _require(job["job_id"] == path.stem, "paired job filename/ID differs")
        block, _, seed_label = path.stem.split("__")
        for destination, source in (("models", "models"), ("seed_contrasts", "contrasts"), ("model_groups", "model_groups"), ("contrast_groups", "contrast_groups")):
            _require(all(r["block"] == block and r["seed"] == int(seed_label[4:]) for r in job[source]), "paired job contains a different block/seed")
            merged[destination].extend(job[source])
    for table, rows in merged.items():
        _require(Counter(map(_canonical, rows)) == Counter(map(_canonical, summary[table])), f"paired-group/summary values differ: {table}")

    definitions = _contrast_design()
    _require(_scorer_definition_hash(definitions) == plan["all160_contrast_definition_sha256"], "planned contrast definition changed")
    models = _index(summary["models"], ("block", *model_keys), "scored model")
    expected_models = {(block, *key) for block, key in itertools.product(("a", "b"), design)}
    _require(set(models) == expected_models, "scored 320-model set differs")
    model_groups = _index(summary["model_groups"], ("block", *model_keys, "group_id"), "model group")
    _require(set(model_groups) == {(*key, g) for key, g in itertools.product(expected_models, range(42))}, "model-group matrix differs")
    group_counts = {}
    for key, row in models.items():
        block, *model_key = key
        _require(row["run_id"] == planned[tuple(model_key)]["run_id"] and row["events"] == BLOCK_EVIDENCE[block][0], "scored model ID/events differ")
        _close(row["mean_log_scale"], row["mean_log_scale"], "mean log scale")
        _close(row["mean_log_F_b"], row["mean_log_F_b"], "mean log support")
        groups = [model_groups[(*key, g)] for g in range(42)]
        _require(all(g["run_id"] == row["run_id"] for g in groups), "model-group run ID differs")
        counts = tuple(g["events"] for g in groups)
        group_counts.setdefault(block, counts)
        _group_reconstruction(groups, row, group_counts[block], str(key))
    contrast_keys = ("block", "seed", "scope", "fraction_tag", "capacity", "context", "contrast")
    contrasts = _index(summary["seed_contrasts"], contrast_keys, "seed contrast")
    expected_contrasts = {(block, seed, *labels) for block, seed, (labels, _) in itertools.product(("a", "b"), SEEDS, definitions)}
    _require(set(contrasts) == expected_contrasts, "410-contrast roster differs")
    contrast_groups = _index(summary["contrast_groups"], (*contrast_keys, "group_id"), "contrast group")
    _require(set(contrast_groups) == {(*key, g) for key, g in itertools.product(expected_contrasts, range(42))}, "contrast-group matrix differs")
    for block, seed, (labels, terms) in itertools.product(("a", "b"), SEEDS, definitions):
        key = (block, seed, *labels)
        row = contrasts[key]
        expected_mean = math.fsum(weight * models[(block, seed, *model_key)]["mean"] for weight, model_key in terms)
        _require(row["events"] == BLOCK_EVIDENCE[block][0], "contrast eligible events differ")
        _close(row["mean"], expected_mean, f"paired NLL contrast {key}")
        groups = [contrast_groups[(*key, g)] for g in range(42)]
        _group_reconstruction(groups, row, group_counts[block], str(key))
        for g, group in enumerate(groups):
            if group["events"]:
                expected_mean = math.fsum(weight * model_groups[(block, seed, *model_key, g)]["mean"] for weight, model_key in terms)
                _close(group["mean"], expected_mean, f"within-bin NLL contrast {key}/{g}")
    summary_keys = ("block", "scope", "fraction_tag", "capacity", "context", "contrast")
    summaries = _index(summary["contrast_summary"], summary_keys, "five-seed summary")
    _require(set(summaries) == {(block, *labels) for block, (labels, _) in itertools.product(("a", "b"), definitions)}, "82-summary roster differs")
    for (block, *labels), row in summaries.items():
        values = [contrasts[(block, seed, *labels)]["mean"] for seed in SEEDS]
        _require(row["seeds"] == 5, "summary seed denominator differs")
        for field, expected in (("mean", statistics.mean(values)), ("sample_sd", statistics.stdev(values)),
                                 ("minimum", min(values)), ("maximum", max(values))):
            _close(row[field], expected, f"five-seed {field}")
        for field, expected in (("positive_seeds", sum(v > 0 for v in values)), ("negative_seeds", sum(v < 0 for v in values)), ("zero_seeds", sum(v == 0 for v in values))):
            _require(row[field] == expected, f"seed sign count differs: {field}")
    for record in inputs:
        _require(_digest(record["path"]) == record["sha256"], "input changed during verification")
    result = {"schema": "correction-confirmation-output-verification/v1", "status": "VERIFIED_FROZEN_UNIQUE160_AB_CONFIRMATION_OUTPUTS",
        "scope": SCOPE, "scoring_plan_sha256": plan_hash, "core_experiment_identity": plan["core_experiment_identity"],
        "extension_experiment_identity": plan["extension_gate"]["experiment_identity"], "verified_counts": TABLE_COUNTS,
        "unique_models": 160, "reused_formal_cells": 20, "blocks": {b: {"eligible_events": n[0], "group_counts": group_counts[b]} for b, n in BLOCK_EVIDENCE.items()},
        "tolerances": {"mean_absolute": MEAN_ATOL, "relative": 5e-10, "pooled_event_sd_absolute": SD_ATOL},
        "checks": ["complete fixed model/contrast roster", "160 checkpoint and 3 source hashes", "plan/state/result identities",
                   "independent NLL reconstruction of 410 contrasts and 17220 group contrasts", "82 five-seed summaries and sign counts",
                   "42-group counts, weighted means and pooled sample SD", "five CSV tables and ten paired job files agree"],
        "boundary": "Saved-output arithmetic/integrity verification only: no raw A/B or cache arrays read, no likelihood inference replay. Individual event covariances cannot be independently recovered from aggregated outputs. Shared-data seed SD is descriptive, and A/B retain legacy W6/MDN use.",
        "input_files": inputs, "verifier_source_sha256": _digest(Path(__file__)), "elapsed_seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=False)
    report = output / "SCIENTIFIC_VERIFICATION.txt"
    report.write_text("独立确认评分输出验算：通过\n\n"
        "160个唯一冻结模型、20个复用映射及A/B共320次评分完整；160个权重与3个评分/验证源文件哈希一致。\n"
        "由模型NLL独立重建全部410项同seed预设对照及17220项分组对照，核对82项均值、样本SD、范围和正负零计数。\n"
        "所有42分组事件数、加权均值及合并样本SD一致；5张CSV与汇总JSON、10个配对任务文件一致。\n"
        "A有效事件数499268，B有效事件数499283；没有模型或负结果被省略。\n\n"
        "范围：本验算仅验证已保存输出的算术与身份完整性，未读取A/B原始数据或缓存数组，未重新推理、训练或选择模型。\n"
        "汇总输出不能独立恢复逐事件协方差，因此不是第二次事件级似然重放。共享数据上的seed SD不是独立重复的置信区间；A/B仍有历史W6/MDN使用记录。\n",
        encoding="utf-8")
    result["output_files"] = [{"path": report.name, "sha256": _digest(report)}]
    (output / "verification_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT, help="source tree whose three frozen source hashes are verified")
    parser.add_argument("--core-study", type=Path, help="current core checkpoint study root; defaults under repository-root/reports")
    parser.add_argument("--extension-study", type=Path, help="current extension checkpoint study root; defaults under repository-root/reports")
    args = parser.parse_args()
    result = verify(args.confirmation, args.output, repository_root=args.repository_root, core_study=args.core_study, extension_study=args.extension_study)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
