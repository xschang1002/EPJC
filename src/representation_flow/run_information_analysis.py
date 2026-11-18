"""Read-only source validation and predeclared fresh-catalog analysis.

Outputs always go to a new directory. This entry point never trains, selects a
model, generates data, or edits its input artifacts. The fresh mode only reads
already completed, hash-bound score arrays; it performs no model inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path, PurePosixPath
import time

import numpy as np

from representation_flow.experiment_contract import canonical_hash, save_json, sha256
from representation_flow.information_analysis import (
    flow_nll_components, location_scale_shape, paired_component_summary,
    pit_diagnostics, source_location_burden,
)
from representation_flow.information_uncertainty import (
    CONTEXT_ORDER, FRACTION_ORDER, paired_seed_gains, planned_contrasts,
    primary_labels, stratified_paired_bootstrap,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3101, 3102, 3103, 3104, 3105)
SCORE_FIELDS = ("log_p_y", "log_q_z", "base_value", "base_logpdf", "spline_logabsdet",
                "log_support_cdf", "log_s", "z", "b", "log_cdf_y", "log_sf_y")
COMMON_FIELDS = ("event_uid", "p_in", "thickness", "charge", "y", "group_id")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


class Deadline:
    def __init__(self, budget, plan, max_seconds):
        self.started = time.time()
        if not 0 < max_seconds <= 3600:
            raise ValueError("analysis local wall budget must be in (0,3600] seconds")
        if (not 0 < budget["maximum_wall_seconds"] <= 43200
                or abs(budget["deadline_unix"] - budget["started_unix"] - budget["maximum_wall_seconds"]) > 1e-6
                or budget["deadline_unix"] != plan["runtime"]["absolute_deadline_unix"]):
            raise ValueError("analysis budget does not match original anchored global deadline")
        self.deadline = min(budget["deadline_unix"], self.started + max_seconds)
        self()

    def __call__(self):
        if time.time() >= self.deadline:
            raise TimeoutError("fixed local/global analysis deadline exceeded")


def validate_plan(path):
    plan = read_json(path)
    if (plan.get("schema") != "fresh-catalog-statistical-analysis/v1"
            or plan.get("status") != "FROZEN_FRESH_STATISTICAL_ANALYSIS"):
        raise ValueError("a final frozen statistical plan is required")
    if plan["primary_labels"] != primary_labels() or plan["primary_contrasts"] != [
        {"name": name, "weights": weights.tolist()} for name, weights in planned_contrasts().items()
    ]:
        raise ValueError("analysis contrast definitions disagree with frozen plan")
    expected = {"n_resamples": 2000, "seed": 20261108, "batch_size": 32, "blas_threads": 1,
                "global_family": 16}
    if any(plan["bootstrap"].get(key) != value for key, value in expected.items()):
        raise ValueError("bootstrap settings differ from the fixed plan")
    for relative, digest in plan["implementation_sha256"].items():
        path_on_disk = safe_path(ROOT, relative)
        if sha256(path_on_disk) != digest:
            raise ValueError("frozen analysis implementation changed: " + relative)
    required = {"src/representation_flow/information_analysis.py",
                "src/representation_flow/information_uncertainty.py",
                "src/representation_flow/run_information_analysis.py"}
    if not required.issubset(plan["implementation_sha256"]):
        raise ValueError("plan does not bind all analysis implementations")
    return plan


def safe_path(root, relative):
    normalized = str(relative).replace("\\", "/")
    part = PurePosixPath(normalized)
    if part.is_absolute() or ".." in part.parts or ":" in normalized:
        raise ValueError("artifact path must be a safe root-relative path")
    path = (Path(root) / part).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("artifact is missing or outside its supplied root")
    return path


def checked_record(root, record, timer):
    timer()
    path = safe_path(root, record["path"])
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError("artifact size/hash mismatch: " + record["path"])
    timer()
    return path


def checked_array(root, record, timer):
    path = checked_record(root, record, timer)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != record["shape"] or np.dtype(record["dtype"]) != array.dtype:
        raise ValueError("array shape/dtype differs from manifest")
    return array


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot export an empty scientific table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "positive_seeds": int(np.count_nonzero(values > 0)),
            "negative_seeds": int(np.count_nonzero(values < 0)),
            "zero_seeds": int(np.count_nonzero(values == 0))}


def validate_new_roster(models):
    expected = set(itertools.product(CONTEXT_ORDER, FRACTION_ORDER, ("Raw", "CorrectedPhysics"), SEEDS))
    mapping = {}
    for model in models:
        label = model["labels"]
        key = (label["context_label"], label["fraction_tag"], label["representation"], label["seed"])
        if key in mapping or key not in expected or label["capacity"] != "Small":
            raise ValueError("new model roster contains duplicate/unplanned labels")
        mapping[key] = model
    if set(mapping) != expected:
        raise ValueError("all 80 planned new models are required")
    return mapping


def new_model_diagnostics(models, load_scores, common, timer):
    from representation_flow.engine import RepresentationPreprocessor
    mapping = validate_new_roster(models)
    n = len(common["y"])
    groups = np.asarray(common["group_id"])
    if set(np.unique(groups).tolist()) != set(range(42)):
        raise ValueError("all42 predeclared groups must be present")
    event_gains = np.empty((n, 8), dtype=np.float64)
    seed_gains = np.empty((5, 8), dtype=np.float64)
    component_rows, predictive_rows, residual_rows = [], [], []
    preprocessing = {}
    for column, label in enumerate(primary_labels()):
        context, fraction = label.split("/")
        raw_seed = np.empty((n, 5), dtype=np.float64)
        corrected_seed = np.empty((n, 5), dtype=np.float64)
        for si, seed in enumerate(SEEDS):
            timer()
            pair = []
            for representation, destination in (("Raw", raw_seed), ("CorrectedPhysics", corrected_seed)):
                model = mapping[(context, fraction, representation, seed)]
                score = load_scores(model)
                if np.asarray(score["log_p_y"]).shape != (n,):
                    raise ValueError("new model scores do not align with eligible events")
                flow_nll_components(score)
                destination[:, si] = score["log_p_y"]
                key = (fraction, representation)
                if key in preprocessing and preprocessing[key] != model["preprocessor"]:
                    raise ValueError("same representation/fraction changed preprocessing across context or seed")
                preprocessing[key] = model["preprocessor"]
                predictive_rows.append({"model_id": model["model_id"], "labels": model["labels"],
                    "physical_groups": pit_diagnostics(score["log_cdf_y"], score["log_sf_y"], groups),
                    "momentum_groups": pit_diagnostics(score["log_cdf_y"], score["log_sf_y"], groups // 6)})
                pair.append(score)
            component_rows.append({"context": context, "fraction_tag": fraction, "seed": seed,
                "physical_groups": paired_component_summary(pair[0], pair[1], groups),
                "momentum_groups": paired_component_summary(pair[0], pair[1], groups // 6)})
        event_gains[:, column], seed_summary = paired_seed_gains(raw_seed, corrected_seed)
        seed_gains[:, column] = seed_summary["seed_means"]
    for (fraction, representation), serialized in preprocessing.items():
        timer()
        pre = RepresentationPreprocessor.from_dict(serialized)
        center = pre.center(common["p_in"], common["thickness"], common["charge"])
        residual = np.asarray(common["y"], dtype=np.float64) - center - pre.mu
        scale = pre.scale_for(common["p_in"], common["thickness"], common["charge"])
        residual_rows.append({"fraction_tag": fraction, "representation": representation,
            "definition": "Y-center-mu; same frozen preprocessing across all model seeds and contexts",
            "original_units": location_scale_shape(residual, groups, units="GeV/c"),
            "standardized": location_scale_shape(residual / scale, groups, units="dimensionless")})
    weights = np.asarray(list(planned_contrasts().values()))
    paired_seed_rows, summaries = [], []
    effect_values = seed_gains @ weights.T
    for j, name in enumerate(planned_contrasts()):
        summaries.append({"contrast": name, **stats(effect_values[:, j])})
        paired_seed_rows.extend({"contrast": name, "seed": seed, "gain": float(effect_values[i, j])}
                                for i, seed in enumerate(SEEDS))
    return event_gains, paired_seed_rows, summaries, component_rows, predictive_rows, residual_rows


def source_validation(args, plan, timer):
    import torch
    from representation_flow.engine import RepresentationPreprocessor
    from representation_flow.information_checkpoint import (
        evaluate_information, load_information_checkpoint, make_tensors,
    )
    from representation_flow.information_context import fixed_group_ids
    from representation_flow.run_information_study import prepare_cache, validate_pairing, validate_protocol
    torch.set_num_threads(1)
    protocol = validate_protocol(read_json(args.training_protocol))
    study = args.study.resolve()
    frozen = read_json(study / "FROZEN_CHECKPOINT_MANIFEST.json")
    scores = read_json(study / "scores.json")
    identity = read_json(study / "experiment_identity.json")
    if (frozen.get("schema") != "information-checkpoint-freeze/v1"
            or frozen.get("status") != "COMPLETE_INFORMATION_CHECKPOINT_FREEZE"
            or frozen.get("expected_cells") != 80 or len(frozen.get("models", [])) != 80
            or scores.get("status") != "COMPLETE" or len(scores.get("runs", [])) != 80 or scores.get("failures")
            or frozen["protocol_sha256"] != canonical_hash(protocol)
            or frozen["experiment_identity"] != scores["experiment_identity"]
            or identity.get("sha256") != canonical_hash(identity.get("identity"))
            or identity["sha256"] != scores["experiment_identity"]
            or identity["identity"].get("mode") != "execute"
            or identity["identity"].get("max_updates") != 20000):
        raise ValueError("complete declared source study and identity are required")
    if not (args.cache / "cache_manifest.json").is_file():
        raise ValueError("existing training preprocessing cache is required; analysis does not create it")
    cache = prepare_cache(args.source_cache, args.cache, protocol)
    arrays = {name: np.load(args.source_cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("event_uid", "p_in", "thickness", "charge", "y", "split")}
    valid = np.load(args.cache / "indices_validation.npy", allow_pickle=False)
    common = {name: arrays[name][valid] for name in ("event_uid", "p_in", "thickness", "charge", "y")}
    common["group_id"] = fixed_group_ids(common["p_in"], common["thickness"], common["charge"])
    uid_hash = canonical_hash(common["event_uid"].tolist())
    if uid_hash != cache["validation_uid_sha256"]:
        raise ValueError("source validation event ordering changed")
    frozen_by_id = {row["run_id"]: row for row in frozen["models"]}
    if len(frozen_by_id) != 80:
        raise ValueError("duplicate frozen source model")
    validate_pairing(scores["runs"])
    models, model_summary, loaded_predictions, maximum_replay = [], [], {}, 0.0
    preprocessing = read_json(args.cache / "preprocessing.json")
    for row in scores["runs"]:
        timer()
        if (row.get("status") != "COMPLETE_INFORMATION_CELL" or row.get("max_updates") != 20000
                or row["protocol_sha256"] != canonical_hash(protocol)
                or row["experiment_identity"] != scores["experiment_identity"]
                or row["run_id"] not in frozen_by_id):
            raise ValueError("invalid source run record")
        frozen_row = frozen_by_id[row["run_id"]]
        if any(row[key] != value for key, value in frozen_row.items()):
            raise ValueError("source scores and checkpoint freeze disagree")
        paths = {name: checked_record(study, record, timer) for name, record in row["artifacts"].items()}
        run_record = read_json(paths["run_manifest"])
        if any(row[key] != value for key, value in run_record.items()):
            raise ValueError("source run manifest and score row disagree")
        payload, pre, flow, spec = load_information_checkpoint(
            paths["checkpoint"], expected_sha256=row["artifacts"]["checkpoint"]["sha256"],
            expected_experiment_identity=scores["experiment_identity"])
        if (pre.to_dict() != preprocessing[row["fraction_tag"]][row["representation"]]
                or payload["information_contract"]["protocol_sha256"] != canonical_hash(protocol)
                or payload["validation_uid_sha256"] != uid_hash):
            raise ValueError("checkpoint data/preprocessing/protocol contract differs")
        with np.load(paths["predictions"], allow_pickle=False) as saved:
            for name in COMMON_FIELDS:
                if not np.array_equal(saved[name], common[name]):
                    raise ValueError("validation prediction events/targets differ from source: " + name)
            saved_lp = np.asarray(saved["log_p_y"], dtype=np.float64)
            flow_nll_components({name: saved[name] for name in SCORE_FIELDS})
        replay = evaluate_information(flow, make_tensors(arrays, valid, pre, spec), pre,
                                      protocol["optimization"]["eval_batch_size"])["log_p_y"]
        error = float(np.max(np.abs(replay.astype(np.float64) - saved_lp)))
        maximum_replay = max(maximum_replay, error)
        if not np.allclose(replay, saved_lp, rtol=2e-6, atol=1e-5):
            raise ValueError("actual saved checkpoint does not replay all validation scores")
        nll = float(np.mean(-saved_lp))
        if not np.isclose(nll, row["validation_nll"], rtol=0, atol=1e-10):
            raise ValueError("reported source validation NLL differs from saved event scores")
        best = min(row["history"], key=lambda item: item["validation_nll"])
        if best["update"] != row["selected_update"]:
            raise ValueError("checkpoint selection differs from recorded minimum validation score")
        labels = {key: row[key] for key in ("run_id", "context_label", "representation", "fraction_tag", "capacity", "seed")}
        models.append({"model_id": row["run_id"], "labels": labels, "preprocessor": pre.to_dict()})
        loaded_predictions[row["run_id"]] = paths["predictions"]
        model_summary.append({**labels, "eligible_events": len(valid), "nll": nll,
                              "maximum_replay_error": error, "selected_update": row["selected_update"]})
    validate_new_roster(models)
    def load_scores(model):
        timer()
        with np.load(loaded_predictions[model["model_id"]], allow_pickle=False) as saved:
            return {name: saved[name] for name in SCORE_FIELDS}
    gains, seed_rows, summaries, component, predictive, residual = new_model_diagnostics(models, load_scores, common, timer)
    diagnostic = np.flatnonzero(arrays["split"] == "diagnostic")
    if not len(diagnostic):
        raise ValueError("declared source-only diagnostic pool is empty")
    source_location = []
    for fraction, entries in preprocessing.items():
        for representation, serialized in entries.items():
            timer()
            pre = RepresentationPreprocessor.from_dict(serialized)
            p, t, q, y = (np.asarray(arrays[name][diagnostic]) for name in ("p_in", "thickness", "charge", "y"))
            residual_values = y - pre.center(p, t, q) - pre.mu
            source_location.append({"fraction_tag": fraction, "representation": representation,
                "original_unit_burden": source_location_burden(residual_values, p, t, q, units="GeV/c"),
                "standardized_burden": source_location_burden(residual_values/pre.scale_for(p,t,q), p,t,q,units="dimensionless")})
    save_json(args.output / "source_location_diagnostics.json", {
        "diagnostic_events": len(diagnostic), "diagnostic_uid_sha256": canonical_hash(arrays["event_uid"][diagnostic].tolist()),
        "fresh_data_accessed": False, "rows": source_location})
    write_common_outputs(args.output, model_summary, seed_rows, summaries, component, predictive, residual)
    return {"schema": "information-source-analysis/v1", "status": "COMPLETE_INFORMATION_SOURCE_ANALYSIS",
            "expected_models": 80, "model_count": 80, "protocol_sha256": canonical_hash(protocol),
            "study_freeze_sha256": sha256(study / "FROZEN_CHECKPOINT_MANIFEST.json"),
            "scores_sha256": sha256(study / "scores.json"),
            "source_cache_manifest_sha256": sha256(args.source_cache / "cache_manifest.json"),
            "validation_uid_sha256": uid_hash, "validation_events": len(valid),
            "all_validation_predictions_replayed": True, "maximum_all_event_replay_error": maximum_replay,
            "primary_contrasts": summaries, "fresh_data_accessed": False,
            "scope": "Selected source-validation scores are descriptive; this analysis supplies no fresh-event confidence intervals."}


def write_common_outputs(output, model_summary, seed_rows, summaries, component, predictive, residual):
    write_csv(output / "model_summary.csv", model_summary)
    write_csv(output / "primary_seed_contrasts.csv", seed_rows)
    write_csv(output / "primary_contrast_summary.csv", summaries)
    save_json(output / "density_component_diagnostics.json", {"rows": component})
    save_json(output / "predictive_diagnostics.json", {"rows": predictive})
    save_json(output / "residual_diagnostics.json", {"rows": residual})
    group_seed, component_summary = summarize_component_seed_variation(component)
    write_csv(output / "group_seed_variation.csv", group_seed)
    write_csv(output / "density_component_seed_summary.csv", component_summary)


def summarize_component_seed_variation(component):
    grouped = {}
    fields = ("negative_log_base_density", "negative_spline_logabsdet",
              "log_preprocessing_scale", "log_support_cdf", "nll_gain")
    for row in component:
        for grouping in ("physical_groups", "momentum_groups"):
            for group in row[grouping]["groups"]:
                for field in fields:
                    key = (row["context"], row["fraction_tag"], grouping, group["group_id"], field)
                    by_seed = grouped.setdefault(key, {})
                    if row["seed"] in by_seed:
                        raise ValueError("duplicate seed in density component grouping")
                    by_seed[row["seed"]] = group[field]
    summaries, group_gains = [], []
    for (context, fraction, grouping, group, field), by_seed in grouped.items():
        if set(by_seed) != set(SEEDS):
            raise ValueError("all five paired model seeds are required for each group")
        row = {"context": context, "fraction_tag": fraction, "grouping": grouping,
               "group_id": group, "component": field, **stats([by_seed[s] for s in SEEDS]),
               **{f"seed_{seed}": by_seed[seed] for seed in SEEDS}}
        summaries.append(row)
        if grouping == "physical_groups" and field == "nll_gain":
            group_gains.append(row)
    return group_gains, summaries


def legacy_summaries(models, means):
    from representation_flow.score_correction_confirmation import full160_contrast_specs
    mapping = {}
    for model in models:
        label = model["labels"]
        key = (label["fraction_tag"], label["capacity"], label["context_label"], label["representation"], label["seed"])
        if key in mapping:
            raise ValueError("legacy reuse mapping has duplicate statistical model keys")
        mapping[key] = means[model["model_id"]]
    if len(mapping) != 160:
        raise ValueError("legacy roster must contain exactly160 unique models")
    seeds, summaries = [], []
    for spec in full160_contrast_specs():
        values = []
        labels = {key: value for key, value in spec.items() if key != "terms"}
        for seed in SEEDS:
            # Existing specification weights log densities; means store NLL.
            value = sum(-weight * mapping[tuple(key)+(seed,)] for weight, key in spec["terms"])
            values.append(value)
            seeds.append({**labels, "seed": seed, "gain": value})
        summaries.append({**labels, **stats(values)})
    return seeds, summaries


def validate_fresh_provenance(args, manifest):
    """Check metadata-only provenance before opening fresh score arrays."""
    fresh_plan, union, roster = read_json(args.fresh_plan), read_json(args.union), read_json(args.roster)
    generation = read_json(args.generation_root / "GENERATION_COMPLETE.json")
    scoring = read_json(args.score_root.parent / "scoring_summary.json")
    if (fresh_plan.get("schema") != "fresh-catalog-protocol/v1"
            or fresh_plan.get("status") != "FROZEN_FRESH_CATALOG"
            or fresh_plan["analysis_plan"]["sha256"] != sha256(args.plan)
            or fresh_plan["model_union"]["sha256"] != sha256(args.union)
            or manifest["plan_sha256"] != sha256(args.fresh_plan)
            or manifest["union_sha256"] != sha256(args.union)):
        raise ValueError("fresh scores do not bind the frozen analysis/model/generation plan")
    if (generation.get("status") != "COMPLETE_FRESH_GEANT4_CATALOGS_NO_MODEL_SCORING"
            or generation["plan_sha256"] != sha256(args.fresh_plan)
            or generation["analysis_sha256"] != sha256(args.plan)
            or manifest["generation_sha256"] != sha256(args.generation_root / "GENERATION_COMPLETE.json")):
        raise ValueError("generation receipt does not match frozen plans")
    generated_blocks = {r["block_id"]: r for r in generation["catalogs"]}
    planned_blocks = {r["block_id"]: r for r in fresh_plan["catalogs"]}
    if (set(generated_blocks) != {"C", "D"} or len(generation["catalogs"]) != 2
            or set(planned_blocks) != {"C", "D"} or len(fresh_plan["catalogs"]) != 2):
        raise ValueError("generation plan/receipt must contain exactly C and D")
    for block in ("C", "D"):
        if any(generated_blocks[block].get(key) != value for key,value in planned_blocks[block].items()):
            raise ValueError("generated catalog does not match its predeclared identity")
    if (scoring.get("status") != "COMPLETE_FROZEN_240_CD_SCORES" or scoring.get("failures")
            or scoring["plan_sha256"] != manifest["plan_sha256"]
            or scoring["union_sha256"] != manifest["union_sha256"]
            or scoring["generation_sha256"] != manifest["generation_sha256"]
            or scoring["roster_sha256"] != sha256(args.roster)):
        raise ValueError("both predeclared240-model catalog scores must be complete")
    if (roster.get("schema") != "fresh-model-roster/v1" or roster.get("expected_models") != 240
            or len(roster.get("models", [])) != 240
            or not any(r["sha256"] == sha256(args.roster) for r in fresh_plan["source_files"])):
        raise ValueError("explicit240 model roster was not bound before generation")
    roster_by_id = {r["model_id"]:r for r in roster["models"]}
    if (len(roster_by_id) != 240 or len(roster["expected_model_ids"]) != 240
            or set(roster_by_id) != set(roster["expected_model_ids"])):
        raise ValueError("prespecified roster IDs are incomplete or duplicated")
    catalog_records = {r["block_id"]: r for r in scoring["catalogs"]}
    if set(catalog_records) != {"C", "D"} or len(scoring["catalogs"]) != 2:
        raise ValueError("scoring summary must contain exactly C and D")
    for block, record in catalog_records.items():
        block_path = safe_path(args.score_root.parent, record["path"])
        block_manifest = read_json(block_path)
        if (record["models"] != 240 or sha256(block_path) != record["sha256"]
                or block_manifest.get("status") != "COMPLETE_FROZEN_240_CATALOG_SCORES"
                or block_manifest.get("block_id") != block or len(block_manifest.get("models", [])) != 240
                or block_manifest.get("plan_sha256") != manifest["plan_sha256"]
                or block_manifest.get("union_sha256") != manifest["union_sha256"]
                or block_manifest.get("generation_sha256") != manifest["generation_sha256"]):
            raise ValueError("completed catalog score manifest changed")
    if (catalog_records[manifest["block_id"]]["sha256"] != sha256(args.score_root / "score_manifest.json")
            or manifest.get("failures")):
        raise ValueError("current score manifest is not the frozen completed block")
    if (union.get("schema") != "fresh-model-union/v1"
            or union.get("status") != "COMPLETE_FROZEN_FRESH_MODEL_UNION"
            or union.get("expected_models") != 240 or len(union.get("models", [])) != 240):
        raise ValueError("complete frozen240-model union is required")
    by_id = {row["model_id"]: row for row in union["models"]}
    if (len(by_id) != 240 or set(by_id) != {m["model_id"] for m in manifest["models"]}
            or set(by_id) != set(roster_by_id)):
        raise ValueError("scored model IDs differ from frozen union")
    for model in manifest["models"]:
        original = by_id[model["model_id"]]
        for field in ("model_id", "labels", "cohort", "checkpoint_kind", "root_key"):
            if original[field] != roster_by_id[model["model_id"]][field]:
                raise ValueError("model union disagrees with the prespecified roster")
        for field in ("model_id", "labels", "cohort", "checkpoint_kind", "root_key", "experiment_identity"):
            if model[field] != original[field]:
                raise ValueError("scored model labels/identity differ from frozen union")
        if (model["checkpoint_sha256"] != original["artifacts"]["checkpoint"]["sha256"]
                or model.get("status") != "COMPLETE_FROZEN_MODEL_SCORE"):
            raise ValueError("scored checkpoint is not the complete frozen model")
    for relative, digest in manifest["source_files"].items():
        if sha256(safe_path(ROOT, relative)) != digest:
            raise ValueError("frozen scoring source changed")
    counts_path = args.generation_root / manifest["block_id"] / "fresh_catalog_manifest.json"
    if (sha256(counts_path) != manifest["counts_sha256"]
            or sha256(counts_path) != generated_blocks[manifest["block_id"]]["stratum_counts_sha256"]):
        raise ValueError("generation stratum counts changed")
    counts = read_json(counts_path)
    if (counts.get("schema") != "fresh-catalog-stratum-counts/v1"
            or counts.get("status") != "COMPLETE_42_STRATUM_COUNTS"
            or counts.get("technical_aborted") != 0 or len(counts.get("cells", [])) != 42
            or counts["run_id"] != generated_blocks[manifest["block_id"]]["run_id"]
            or counts["events_sha256"] != manifest["events_sha256"]):
        raise ValueError("score and generation event-file identity differ")
    if {row["group_id"] for row in counts["cells"]} != set(range(42)):
        raise ValueError("generation counts omit or duplicate a physical stratum")
    for row in counts["cells"]:
        if (any(type(row[key]) is not int or row[key] < 0 for key in ("launched", "eligible", "ineligible", "aborted"))
                or row["aborted"] != 0 or row["launched"] != row["eligible"] + row["ineligible"]
                or row["launched"] != row["launched_events"] or row["eligible"] != row["eligible_events"]
                or sum(row["statuses"].values()) != row["launched"]):
            raise ValueError("invalid launched/S0 eligibility or technical-abort counts")
    launched = {str(row["group_id"]): row["launched_events"] for row in counts["cells"]}
    eligible = {str(row["group_id"]): row["eligible_events"] for row in counts["cells"]}
    if launched != manifest["launched_counts"] or eligible != manifest["eligible_counts"]:
        raise ValueError("scoring counts differ from independent generation ledger")
    return {"fresh_plan_sha256": sha256(args.fresh_plan), "union_sha256": sha256(args.union),
            "roster_sha256": sha256(args.roster),
            "generation_sha256": sha256(args.generation_root / "GENERATION_COMPLETE.json"),
            "scoring_summary_sha256": sha256(args.score_root.parent / "scoring_summary.json"),
            "counts_sha256": sha256(counts_path)}


def fresh_catalog(args, plan, timer):
    from representation_flow.information_context import fixed_group_ids
    manifest_path = args.score_root / "score_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("schema") != "information-catalog-scores/v1"
            or manifest.get("status") != "COMPLETE_FROZEN_240_CATALOG_SCORES"
            or manifest.get("block_id") not in ("C", "D") or len(manifest.get("models", [])) != 240):
        raise ValueError("one complete predeclared240-model C/D score manifest is required")
    provenance = validate_fresh_provenance(args, manifest)
    common = {name: checked_array(args.score_root, manifest["arrays"][name], timer) for name in COMMON_FIELDS}
    n = len(common["y"])
    uid_hash = canonical_hash(common["event_uid"].tolist())
    if (n != manifest["eligible_events"] or uid_hash != manifest["event_uid_sha256"]
            or len(np.unique(common["event_uid"])) != n
            or not np.array_equal(common["group_id"], fixed_group_ids(common["p_in"],common["thickness"],common["charge"]))):
        raise ValueError("fresh score event identity/group counts disagree")
    counts = {g: int(np.count_nonzero(common["group_id"] == g)) for g in range(42)}
    launched = {int(key): value for key, value in manifest["launched_counts"].items()}
    if (counts != {int(key): value for key,value in manifest["eligible_counts"].items()}
            or set(launched) != set(range(42)) or sum(launched.values()) != 500000):
        raise ValueError("fresh launched/eligible stratum counts differ from fixed scope")
    models = manifest["models"]
    if len({model["model_id"] for model in models}) != 240:
        raise ValueError("fresh model IDs are not unique")
    new = [model for model in models if model["cohort"] == "information80"]
    legacy = [model for model in models if model["cohort"] == "legacy160"]
    if len(new) != 80 or len(legacy) != 160:
        raise ValueError("fresh cohorts must be new80 and legacy160")
    validate_new_roster(new)
    paths, means, model_summary = {}, {}, []
    for model in models:
        timer()
        if model.get("event_uid_sha256") != uid_hash:
            raise ValueError("model score UID order differs")
        if len(model["checkpoint_sha256"]) != 64:
            raise ValueError("model lacks frozen checkpoint identity")
        paths[model["model_id"]] = {}
        for field in SCORE_FIELDS:
            array = checked_array(args.score_root, model["arrays"][field], timer)
            if array.shape != (n,):
                raise ValueError("fresh model array has wrong event count")
            paths[model["model_id"]][field] = safe_path(args.score_root, model["arrays"][field]["path"])
        lp = np.load(paths[model["model_id"]]["log_p_y"], mmap_mode="r", allow_pickle=False)
        if not np.isfinite(lp).all():
            raise ValueError("nonfinite scored likelihood; preserve failed/incomplete result")
        means[model["model_id"]] = float(np.mean(-lp, dtype=np.float64))
        if (model["eligible_events"] != n
                or not np.isclose(means[model["model_id"]], model["nll"], rtol=0, atol=1e-10)):
            raise ValueError("model score summary differs from its saved event likelihoods")
        model_summary.append({"model_id": model["model_id"], "cohort": model["cohort"],
                              **model["labels"], "eligible_events": n, "nll": means[model["model_id"]]})
    def load_scores(model):
        timer()
        return {name: np.load(path, mmap_mode="r", allow_pickle=False) for name,path in paths[model["model_id"]].items()}
    gains, seed_rows, summaries, component, predictive, residual = new_model_diagnostics(new, load_scores, common, timer)
    bootstrap = stratified_paired_bootstrap(gains, common["group_id"], launched,
        labels=primary_labels(), contrasts=planned_contrasts(), n_resamples=2000, seed=20261108,
        batch_size=32, max_seconds=900, deadline_unix=timer.deadline)
    legacy_seed, legacy_summary = legacy_summaries(legacy, means)
    write_common_outputs(args.output, model_summary, seed_rows, summaries, component, predictive, residual)
    write_csv(args.output / "legacy_seed_contrasts.csv", legacy_seed)
    write_csv(args.output / "legacy_contrast_summary.csv", legacy_summary)
    write_csv(args.output / "primary_uncertainty.csv", bootstrap.summary["effects"])
    write_csv(args.output / "group_uncertainty.csv", bootstrap.summary["primary_groups"])
    save_json(args.output / "uncertainty.json", bootstrap.summary)
    np.savez_compressed(args.output / "bootstrap_replicates.npz", global_contrasts=bootstrap.global_replicates,
                        primary_groups=bootstrap.primary_group_replicates,
                        eligible_counts=bootstrap.eligible_count_replicates)
    return {"schema": "information-fresh-analysis/v1", "status": "COMPLETE_INFORMATION_FRESH_ANALYSIS",
            "block_id": manifest["block_id"], "model_count": 240, "new_models": 80, "legacy_models": 160,
            "eligible_events": n, "launched_events": 500000, "event_uid_sha256": uid_hash,
            "input_provenance": provenance,
            "score_manifest_sha256": sha256(manifest_path), "primary_family_size": 16,
            "legacy_contrast_count": len(legacy_summary), "primary_contrasts": bootstrap.summary["effects"],
            "scope": "Fixed-model event uncertainty for this one predeclared catalog; no pooling, selection, refitting, or fresh inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validation", "fresh"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "protocol/information_statistical_analysis_20261108.json")
    parser.add_argument("--budget", type=Path, default=ROOT.parent / "_research_extension_20261108/AUTHORIZATION_AND_BUDGET.json")
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--study", type=Path, default=ROOT / "reports/information_extension_20261108/study")
    parser.add_argument("--cache", type=Path, default=ROOT / "reports/information_extension_20261108/cache")
    parser.add_argument("--source-cache", type=Path, default=ROOT / "reports/correction_v2_20261107/cache")
    parser.add_argument("--training-protocol", type=Path, default=ROOT / "protocol/information_study_20261108.json")
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--fresh-plan", type=Path)
    parser.add_argument("--union", type=Path)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--generation-root", type=Path)
    args = parser.parse_args()
    if args.mode == "fresh" and any(value is None for value in
            (args.score_root, args.fresh_plan, args.union, args.roster, args.generation_root)):
        parser.error("fresh analysis requires --score-root, --fresh-plan, --union, --roster and --generation-root")
    plan = validate_plan(args.plan)
    timer = Deadline(read_json(args.budget), plan, args.max_seconds)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("preserve existing analysis outputs; choose a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "execution_state.json", {"status":"RUNNING", "started_utc":utc(),
              "deadline_unix":timer.deadline, "plan_sha256":sha256(args.plan), "mode":args.mode})
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result = source_validation(args, plan, timer) if args.mode == "validation" else fresh_catalog(args, plan, timer)
        timer()
        result.update(generated_utc=utc(), statistical_plan_sha256=sha256(args.plan),
                      analysis_sources=plan["implementation_sha256"], elapsed_seconds=time.time()-timer.started)
        save_json(args.output / "analysis.json", result)
        save_json(args.output / "execution_state.json", {"status":"COMPLETE", "mode":args.mode,
                  "elapsed_seconds":time.time()-timer.started,"deadline_unix":timer.deadline})
        files = [{"path":p.name,"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(args.output.iterdir()) if p.is_file()]
        save_json(args.output / "ANALYSIS_MANIFEST.json", {"schema":"information-analysis-output/v1",
                  "status":result["status"],"files":files})
        print(json.dumps({"status":result["status"],"output":str(args.output),"elapsed_seconds":result["elapsed_seconds"]}))
    except Exception as error:
        save_json(args.output / "execution_state.json", {"status":"INCOMPLETE", "mode":args.mode,
                  "elapsed_seconds":time.time()-timer.started,"error_type":type(error).__name__,"error":str(error)})
        raise


if __name__ == "__main__":
    main()
