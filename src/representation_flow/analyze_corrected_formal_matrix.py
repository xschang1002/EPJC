"""Verify and describe all 120 corrected Full-conditioner cells without fitting flows or reading confirmation."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from representation_flow.analyze_correction_study import (
    _inside_file, read_json, summarize_values, validate_study, write_csv,
)
from representation_flow.checkpoint_contract import load_checkpoint
from representation_flow.engine import ConditionalSplineFlow, RepresentationPreprocessor, context_features, subset_indices
from representation_flow.experiment_contract import canonical_hash, sha256, source_hashes, validate_completed
from representation_flow.run_a0_prototype import _state_hash
from representation_flow.run_corrected_formal_extension import cell_id, reused_cell

DISPLAY_CONTRASTS = (("Raw", "Simple"), ("Raw", "Learned"), ("Raw", "CorrectedPhysics"))
PRIMARY_CONTRASTS = (("Raw", "CorrectedPhysics"), ("Learned", "CorrectedPhysics"), ("Simple", "Raw"))
PRIMARY_NAMES = [f"{left}_minus_{right}" for left, right in PRIMARY_CONTRASTS] + [
    "Small_minus_Large_of_Raw_minus_CorrectedPhysics", "f010_minus_f100_of_Raw_minus_CorrectedPhysics"]
BOUNDARY = (
    "All five optimizer seeds share training and validation data; sample SD is seed variability, not independent-data uncertainty. "
    "Validation selected checkpoints, so these paired event differences and comparisons are descriptive, not independent confirmation. "
    "Raw-minus-Simple and Raw-minus-Learned are additional descriptive display contrasts; the prespecified primary contrasts remain separately reported. "
    "The current corrected engine and matched catalog do not reproduce unavailable historical engine bytes. "
    "No confirmation events were read and no neural model was fitted during this analysis."
)


def expected_matrix(protocol):
    if (protocol.get("schema") != "corrected-formal-extension-v1" or protocol.get("representations") != ["Raw", "Simple", "Learned", "CorrectedPhysics"]
            or protocol.get("capacities") != {"Small": 32, "Large": 128} or protocol.get("sample_fractions") != [.1, .3, 1.]
            or protocol.get("seeds") != [3101,3102,3103,3104,3105] or protocol.get("subset_catalog_id") != 4101
            or protocol.get("eligibility") != "forward" or protocol["optimization"]["max_updates"] != 20000
            or protocol.get("primary_contrasts") != PRIMARY_NAMES):
        raise ValueError("analysis requires the complete prespecified corrected Full 120-cell matrix")
    return {cell_id(rep, capacity, fraction, seed, 4101): (rep, capacity, fraction, seed)
            for rep, capacity, fraction, seed in itertools.product(protocol["representations"], protocol["capacities"], protocol["sample_fractions"], protocol["seeds"])}


def check_preprocessing(cache, arrays, definition, protocol, core_cache_record):
    record = read_json(cache / "preprocessing_manifest.json")
    if canonical_hash(record) != definition["preprocessing_sha256"]:
        raise ValueError("extension preprocessing cache identity mismatch")
    expected_identity = {"base_cache_sha256": canonical_hash(core_cache_record), "protocol_sha256": canonical_hash(protocol),
                         "source_hashes": source_hashes((Path(__file__).with_name("run_corrected_formal_extension.py").resolve(),))}
    if record["identity"] != expected_identity:
        raise ValueError("extension preprocessing source/data/protocol identity mismatch")
    files = record["files"]
    names = [item["name"] for item in files]
    if len(set(names)) != 4 or set(names) != {"indices_f010.npy", "indices_f030.npy", "indices_f100.npy", "indices_validation.npy"}:
        raise ValueError("extension index inventory is incomplete or duplicated")
    indices = {}
    for item in files:
        path = _inside_file(cache, item["name"])
        if sha256(path) != item["sha256"]:
            raise ValueError("extension index artifact changed")
        indices[path.stem] = np.load(path, allow_pickle=False)
    valid = np.flatnonzero(arrays["split"] == "validation")
    if not np.array_equal(indices["indices_validation"], valid) or record["validation_events"] != len(valid) or record["validation_uid_sha256"] != canonical_hash(arrays["event_uid"][valid].tolist()):
        raise ValueError("extension validation sample order differs")
    for fraction in protocol["sample_fractions"]:
        tag = f"f{round(fraction * 100):03d}"
        train = subset_indices(arrays, fraction, 4101)
        rec = record["fractions"][tag]
        if not np.array_equal(indices[f"indices_{tag}"], train) or rec["train_events"] != len(train) or rec["ordered_uid_sha256"] != canonical_hash(arrays["event_uid"][train].tolist()):
            raise ValueError("extension training sample order differs")
        if set(rec["preprocessors"]) != set(protocol["representations"]):
            raise ValueError("extension preprocessing matrix is incomplete")
        for name, pre_record in rec["preprocessors"].items():
            pre = RepresentationPreprocessor.from_dict(pre_record)
            center = pre.center(arrays["p_in"][train], arrays["thickness"][train], arrays["charge"][train])
            if pre.name != name or not np.isclose(pre.mu, np.mean(arrays["y"][train] - center), rtol=0, atol=1e-14) or not np.isclose(pre.scale, np.std(arrays["y"][train]), rtol=0, atol=1e-14):
                raise ValueError("preprocessing mean/scale is not train-only")
    return record, indices


def check_new_cell(row, study, arrays, indices, preprocessing, protocol, identity):
    rep, capacity, fraction, seed = row["representation"], row["capacity"], row["sample_fraction"], row["seed"]
    tag = f"f{round(fraction * 100):03d}"
    train, valid = indices[f"indices_{tag}"], indices["indices_validation"]
    rec = preprocessing["fractions"][tag]
    manifest = read_json(_inside_file(study, row["artifacts"]["run_manifest"]["path"]))
    for key in ("run_id", "representation", "capacity", "seed", "selected_update", "initialization_hash", "preprocessor", "checkpoint_contract",
                "experiment_identity", "train_events", "validation_events", "max_updates", "batch_size", "validation_every", "eval_batch_size", "learning_rate", "context_feature_mode"):
        if manifest[key] != row[key]:
            raise ValueError(f"run manifest differs on {key}")
    for key in ("max_updates", "batch_size", "validation_every", "eval_batch_size", "learning_rate"):
        if row[key] != protocol["optimization"][key]:
            raise ValueError(f"run changed frozen {key}")
    if (row["context_feature_mode"], row["context_label"], row["train_events"], row["validation_events"]) != ("full", "Full", len(train), len(valid)):
        raise ValueError("run context or sample counts differ")
    if row["train_event_uid_sha256"] != rec["ordered_uid_sha256"] or row["validation_event_uid_sha256"] != preprocessing["validation_uid_sha256"]:
        raise ValueError("run sample UID digests differ")
    payload, pre = load_checkpoint(_inside_file(study, row["artifacts"]["checkpoint"]["path"]), expected_sha256=row["artifacts"]["checkpoint"]["sha256"])
    if pre.to_dict() != rec["preprocessors"][rep] or pre.to_dict() != row["preprocessor"] or payload["checkpoint_contract"] != row["checkpoint_contract"]:
        raise ValueError("checkpoint preprocessing differs from frozen subset preprocessing")
    if (payload["experiment_identity"], pre.name, payload["seed"], payload["selected_update"], payload["initialization_hash"]) != (identity, rep, seed, row["selected_update"], row["initialization_hash"]):
        raise ValueError("checkpoint labels/identity differ")
    if payload["checkpoint_contract"]["context_feature_mode"] != "full" or payload["checkpoint_contract"]["eligibility"] != "forward":
        raise ValueError("checkpoint context/eligibility differs")
    if payload["training_state"]["update"] != row["selected_update"]:
        raise ValueError("saved training state and selected weights differ in time")
    cfg = payload["model_config"]
    if cfg != {"context_dim": 5, "hidden": protocol["capacities"][capacity], "seed": seed, "bound": 8.0}:
        raise ValueError("checkpoint architecture differs")
    uid = arrays["event_uid"][valid]
    with np.load(_inside_file(study, row["artifacts"]["predictions"]["path"]), allow_pickle=False) as data:
        if not np.array_equal(data["event_uid"], uid) or not np.array_equal(data["y"], arrays["y"][valid]) or not np.array_equal(data["p_in"], arrays["p_in"][valid]):
            raise ValueError("prediction UID order or validation targets differ")
        for key, value in (("representation", rep), ("capacity", capacity), ("seed", seed), ("subset_id", 4101), ("split", "validation")):
            if data[key].shape != uid.shape or not np.all(data[key] == value):
                raise ValueError(f"prediction label differs: {key}")
        logp = data["log_p_y"].astype(np.float64)
        if logp.shape != uid.shape or not np.isfinite(logp).all():
            raise ValueError("prediction log likelihood is nonfinite or has wrong shape")
        center = pre.center(arrays["p_in"][valid], arrays["thickness"][valid], arrays["charge"][valid])
        if not np.allclose(data["c"], center, rtol=0, atol=1e-14) or not np.all(data["mu"] == pre.mu) or not np.allclose(data["s"], pre.scale, rtol=1e-7, atol=0):
            raise ValueError("prediction representation coordinates differ")
        decomposition = data["log_q_z"].astype(np.float64) - np.log(data["s"].astype(np.float64)) - data["log_F_b"].astype(np.float64)
        if not np.allclose(logp, decomposition, rtol=2e-6, atol=2e-6):
            raise ValueError("prediction likelihood decomposition fails")
        history = row["history"]
        opt = protocol["optimization"]
        expected_updates = sorted({1, opt["max_updates"], *range(opt["validation_every"], opt["max_updates"] + 1, opt["validation_every"])})
        if [entry["update"] for entry in history] != expected_updates or not all(np.isfinite(entry["validation_nll"]) for entry in history):
            raise ValueError("checkpoint-selection history is incomplete or nonfinite")
        best = min(history, key=lambda h: h["validation_nll"])
        if row["selected_update"] != best["update"] or not np.isclose(float(np.mean(-data["log_p_y"])), best["validation_nll"], rtol=0, atol=1e-6):
            raise ValueError("checkpoint selection does not follow minimum validation NLL")
    nll = float(np.mean(-logp, dtype=np.float64))
    if not np.isclose(nll, row["validation_nll"], rtol=0, atol=1e-12):
        raise ValueError("validation NLL does not reconstruct from predictions")
    sample = np.unique(np.linspace(0, len(valid) - 1, min(4096, len(valid)), dtype=int))
    idx = valid[sample]
    p, t, q, y = (arrays[k][idx] for k in ("p_in", "thickness", "charge", "y"))
    flow = ConditionalSplineFlow(**cfg)
    if _state_hash(flow.state_dict()) != row["initialization_hash"]:
        raise ValueError("reported initialization does not reproduce from the frozen seed/architecture")
    flow.load_state_dict(payload["state_dict"])
    flow.eval()
    with torch.no_grad():
        replay = flow.original_terms(torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(pre.center(p,t,q), dtype=torch.float32), pre.mu,
                    torch.as_tensor(pre.scale_for(p,t,q), dtype=torch.float32), torch.as_tensor(p, dtype=torch.float32),
                    torch.as_tensor(context_features(p,t,q,"full"), dtype=torch.float32))["log_p_y"].numpy().astype(np.float64)
    if not np.allclose(replay, logp[sample], rtol=2e-6, atol=1e-4):
        raise ValueError("frozen checkpoint does not reproduce sampled validation likelihoods")
    return logp, {"checkpoint_replayed_validation_events": len(sample), "checkpoint_replay_max_abs_logpdf_error": float(np.max(np.abs(replay - logp[sample])))}


def validate_matrix(core_root, extension_root, core_protocol_path, protocol_path, *, merged=None, core_cache=None, extension_cache=None):
    torch.set_num_threads(1)
    core_cache = Path(core_cache) if core_cache is not None else core_root / "cache"
    extension_cache = Path(extension_cache) if extension_cache is not None else extension_root / "cache"
    protocol, core_protocol = read_json(protocol_path), read_json(core_protocol_path)
    cells = expected_matrix(protocol)
    if canonical_hash(core_protocol) != protocol["core_protocol_sha256"]:
        raise ValueError("core protocol identity differs")
    for key in ("main_data", "eligibility", "split", "subset_catalog_id", "seeds", "optimization"):
        if protocol[key] != core_protocol[key]:
            raise ValueError(f"extension and core {key} differ")
    study = extension_root / "study"
    combined, scores = read_json(study / "formal_scores_120.json"), read_json(study / "scores.json")
    if combined.get("status") != "COMPLETE_CORRECTED_FORMAL_120" or combined.get("new_cells") != 100 or combined.get("reused_cells") != 20 or scores.get("status") != "COMPLETE" or scores.get("failures"):
        raise ValueError("corrected formal matrix is incomplete or has failures")
    identity = read_json(study / "experiment_identity.json")
    definition = identity["identity"]
    if identity["sha256"] != canonical_hash(definition) or combined["experiment_identity"] != identity["sha256"] or scores["experiment_identity"] != identity["sha256"]:
        raise ValueError("extension experiment digest differs")
    if definition["mode"] != "execute" or definition["max_updates"] != 20000 or definition["protocol_sha256"] != canonical_hash(protocol):
        raise ValueError("benchmark or changed protocol is not a scientific matrix")
    if definition["extension_source_sha256"] != sha256(Path(__file__).with_name("run_corrected_formal_extension.py")):
        raise ValueError("extension runner source changed")
    core_identity = read_json(core_root / "study/experiment_identity.json")
    core_common = {key: value for key, value in core_identity["identity"].items() if key not in ("mode", "max_updates")}
    if definition["core_common"] != core_common:
        raise ValueError("extension used a different core source/data/environment")
    core_checked = validate_study(core_root / "study", core_cache, core_protocol, merged=merged)
    arrays = core_checked["arrays"]
    preprocessing, indices = check_preprocessing(extension_cache, arrays, definition, protocol, core_checked["cache_record"])
    expected_new = {name for name, (rep, capacity, fraction, seed) in cells.items() if not reused_cell(rep, capacity, fraction)}
    if validate_completed(scores["runs"], study, identity["sha256"], expected_new) != expected_new:
        raise ValueError("extension does not contain exactly 100 new completed cells")
    frozen = read_json(study / "FROZEN_CHECKPOINT_MANIFEST.json")
    if frozen.get("schema") != "corrected-formal-checkpoint-freeze-v1" or frozen.get("experiment_identity") != identity["sha256"] or frozen.get("frozen_before_confirmation") is not True:
        raise ValueError("corrected formal checkpoints are not frozen")
    freeze_rows = {r["matrix_cell_id"]: r for r in frozen["models"]}
    rows = {r["matrix_cell_id"]: r for r in combined["runs"]}
    if len(frozen["models"]) != 120 or len(combined["runs"]) != 120 or set(rows) != set(cells) or set(freeze_rows) != set(cells):
        raise ValueError("corrected formal matrix is incomplete or duplicated")
    originals = {r["run_id"]: r for r in scores["runs"]}
    originals.update({r["run_id"]: r for r in read_json(core_root / "study/scores.json")["runs"]})
    core_replay = {r["run_id"]: r for r in core_checked["cells"]}
    predictions, checked, init_hashes = {}, [], {}
    for name, (rep, capacity, fraction, seed) in sorted(cells.items()):
        row = rows[name]
        reused = reused_cell(rep, capacity, fraction)
        if (row["representation"], row["capacity"], row["sample_fraction"], row["seed"], row["reused"], row["context_label"]) != (rep, capacity, fraction, seed, reused, "Full"):
            raise ValueError("matrix labels do not match the unique expected cell")
        # artifact_root is frozen provenance from the producer's machine. Actual
        # reads always use the caller-supplied roots and safe relative paths.
        if not isinstance(row["artifact_root"], str) or not row["artifact_root"]:
            raise ValueError("matrix artifact root provenance is missing")
        original = originals[row["run_id"]]
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError("combined matrix altered an original cell record")
        if any(freeze_rows[name].get(key) != row[key] for key in ("run_id", "representation", "capacity", "sample_fraction", "seed", "experiment_identity", "reused", "artifact_root", "artifacts")):
            raise ValueError("frozen combined manifest differs from its cell")
        tag = f"f{round(fraction * 100):03d}"
        if reused:
            pre = core_checked["preprocessors"][(tag, rep)]
            if pre.to_dict() != preprocessing["fractions"][tag]["preprocessors"][rep]:
                raise ValueError("reused cell preprocessing differs from extension")
            if row["train_event_uid_sha256"] != preprocessing["fractions"][tag]["ordered_uid_sha256"]:
                raise ValueError("reused cell training UID order differs")
            logp = core_checked["predictions"][("Full", tag, seed, rep)]
            replay = {key: core_replay[row["run_id"]][key] for key in ("checkpoint_replayed_validation_events", "checkpoint_replay_max_abs_logpdf_error")}
        else:
            logp, replay = check_new_cell(row, study, arrays, indices, preprocessing, protocol, identity["sha256"])
        key = (capacity, seed)
        if key in init_hashes and init_hashes[key] != row["initialization_hash"]:
            raise ValueError("paired representations/fractions have different initialization")
        init_hashes[key] = row["initialization_hash"]
        predictions[(capacity, tag, seed, rep)] = logp
        checked.append({**row, "validation_nll": float(np.mean(-logp, dtype=np.float64)), **replay})
    return {"runs": checked, "predictions": predictions, "experiment_identity": identity["sha256"], "protocol": protocol,
            "core_experiment_identity": core_checked["experiment_identity"]}


def contrasts(predictions, protocol, definitions):
    rows, summaries = [], []
    for capacity, fraction, (left, right) in itertools.product(protocol["capacities"], protocol["sample_fractions"], definitions):
        tag = f"f{round(fraction * 100):03d}"
        values = []
        for seed in protocol["seeds"]:
            difference = predictions[(capacity, tag, seed, right)] - predictions[(capacity, tag, seed, left)]
            value = float(np.mean(difference, dtype=np.float64))
            values.append(value)
            rows.append({"capacity": capacity, "sample_fraction": fraction, "fraction_tag": tag, "contrast": f"{left}_minus_{right}", "seed": seed,
                         "nll_difference": value, "validation_events": len(difference), "event_difference_sd_descriptive": float(np.std(difference, ddof=1))})
        summaries.append({"capacity": capacity, "sample_fraction": fraction, "fraction_tag": tag, "contrast": f"{left}_minus_{right}", **summarize_values(values)})
    return rows, summaries


def prespecified_contrasts(predictions, protocol):
    """All five frozen contrast families, with matched seed/event subtraction."""
    if protocol["primary_contrasts"] != PRIMARY_NAMES:
        raise ValueError("frozen primary contrast families changed")
    rows, summaries = contrasts(predictions, protocol, PRIMARY_CONTRASTS)
    comparisons = []
    for fraction in protocol["sample_fractions"]:
        tag = f"f{round(fraction * 100):03d}"
        comparisons.append(("Small_minus_Large", fraction, tag, PRIMARY_NAMES[3], ("Small", tag), ("Large", tag)))
    for capacity in protocol["capacities"]:
        comparisons.append((capacity, None, "f010_minus_f100", PRIMARY_NAMES[4], (capacity, "f010"), (capacity, "f100")))
    for capacity, fraction, tag, name, left, right in comparisons:
        values = []
        for seed in protocol["seeds"]:
            left_gain = predictions[(*left, seed, "CorrectedPhysics")] - predictions[(*left, seed, "Raw")]
            right_gain = predictions[(*right, seed, "CorrectedPhysics")] - predictions[(*right, seed, "Raw")]
            difference = left_gain - right_gain
            value = float(np.mean(difference, dtype=np.float64)); values.append(value)
            rows.append({"capacity": capacity, "sample_fraction": fraction, "fraction_tag": tag, "contrast": name, "seed": seed,
                         "nll_difference": value, "validation_events": len(difference), "event_difference_sd_descriptive": float(np.std(difference, ddof=1))})
        summaries.append({"capacity": capacity, "sample_fraction": fraction, "fraction_tag": tag, "contrast": name, **summarize_values(values)})
    return rows, summaries


def analyze(core_root, extension_root, core_protocol_path, protocol_path, output, *, merged=None, core_cache=None, extension_cache=None):
    if output.exists():
        raise FileExistsError("analysis output exists; preserve it and choose a new directory")
    core_cache = Path(core_cache) if core_cache is not None else core_root / "cache"
    extension_cache = Path(extension_cache) if extension_cache is not None else extension_root / "cache"
    checked = validate_matrix(core_root, extension_root, core_protocol_path, protocol_path, merged=merged, core_cache=core_cache, extension_cache=extension_cache)
    seeds, summary = contrasts(checked["predictions"], checked["protocol"], DISPLAY_CONTRASTS)
    primary_seeds, primary_summary = prespecified_contrasts(checked["predictions"], checked["protocol"])
    result = {"status": "COMPLETE_CORRECTED_FORMAL_120", "analysis_status": "VERIFIED_CORRECTED_FORMAL_120_DESCRIPTIVE_ANALYSIS",
              "experiment_identity": checked["experiment_identity"], "core_experiment_identity": checked["core_experiment_identity"],
              "new_cells": 100, "reused_cells": 20, "runs": checked["runs"], "seed_contrasts": seeds, "contrast_summary": summary,
              "prespecified_seed_contrasts": primary_seeds, "prespecified_contrast_summary": primary_summary,
              "prespecified_contrast_families": PRIMARY_NAMES,
              "artifact_resolution": {"core_study": str((core_root / "study").resolve()), "extension_study": str((extension_root / "study").resolve()),
                                      "core_cache": str(core_cache.resolve()), "extension_cache": str(extension_cache.resolve()),
                                      "rule": "CLI roots determine actual files; frozen artifact_root strings are source provenance, verified against the frozen manifest."},
              "contrast_sign": "NLL(left)-NLL(right); positive means right has lower validation NLL",
              "interaction_sign": "Positive Small-minus-Large means larger corrected-over-Raw gain in Small; positive f010-minus-f100 means larger gain at f010.",
              "interpretation_boundary": BOUNDARY, "confirmation_accessed": False,
              "selection_precision": "float32 mean selects checkpoints; all exported validation NLLs and paired contrasts are reconstructed with float64 accumulation",
              "checkpoint_replay": "up to 4096 ordered validation events per checkpoint, rtol=2e-6 and atol=1e-4"}
    output.mkdir(parents=True, exist_ok=False)
    cell_fields = ("matrix_cell_id", "run_id", "representation", "capacity", "sample_fraction", "seed", "reused", "validation_nll", "selected_update", "max_updates", "experiment_identity", "checkpoint_replayed_validation_events", "checkpoint_replay_max_abs_logpdf_error")
    write_csv(output / "cell_scores.csv", [{key: row[key] for key in cell_fields} for row in checked["runs"]])
    for name, rows in (("seed_contrasts.csv", seeds), ("contrast_summary.csv", summary), ("prespecified_seed_contrasts.csv", primary_seeds), ("prespecified_contrast_summary.csv", primary_summary)):
        write_csv(output / name, rows)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output / "analysis.json").write_text(serialized, encoding="utf-8")
    (output / "formal_scores_120.json").write_text(serialized, encoding="utf-8")
    lines = ["纠正后的完整Full条件120-cell矩阵：描述性验证分析", "", "100个新cell与20个复用cell均通过完整性、来源、冻结权重、预处理、配对UID与checkpoint重载核验。",
             "全矩阵为4种表示×2种容量×3种样本分数×5seed，预算统一20000次更新。复用模型保持原checkpoint字节和实验身份。",
             "单一表示差值=NLL(left)-NLL(right)，正值表示right在验证集更好。交互正值分别表示纠正相对Raw的收益在Small大于Large、在f010大于f100。均值/样本SD保留全部5seed，包括负值和零值。", "",
             "展示对照（Raw−Simple/Raw−Learned为额外描述性比较；Raw−CorrectedPhysics属于预声明主对照）："]
    for row in summary:
        lines.append(f"{row['capacity']} {row['fraction_tag']} {row['contrast']}: mean={row['mean']:+.9f}, seedSD={row['sample_sd']:.9f}; range=[{row['minimum']:+.9f},{row['maximum']:+.9f}] nat/event; positive/negative/zero={row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']}")
    lines += ["", "预声明主对照（与展示对照分开完整报告）："]
    for row in primary_summary:
        lines.append(f"{row['capacity']} {row['fraction_tag']} {row['contrast']}: mean={row['mean']:+.9f}, seedSD={row['sample_sd']:.9f} nat/event")
    lines += ["", "科学边界：五个seed共享数据，SD描述优化随机性，不能解释为独立数据重复的置信区间。验证集参与模型选择，差值不构成独立确认或普遍物理优越性的证明。结果是当前纠正引擎与匹配样本顺序下的新证据，不是不可恢复历史源码的精确重放。分析未读取确认数据、未重新训练神经网络。", "", BOUNDARY]
    (output / "SCIENTIFIC_ANALYSIS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    inputs = [core_root / "study/scores.json", core_root / "study/experiment_identity.json", core_root / "study/FROZEN_CHECKPOINT_MANIFEST.json",
              extension_root / "study/scores.json", extension_root / "study/formal_scores_120.json", extension_root / "study/FROZEN_CHECKPOINT_MANIFEST.json",
              extension_root / "study/experiment_identity.json", core_cache / "cache_manifest.json", extension_cache / "preprocessing_manifest.json", core_protocol_path, protocol_path]
    manifest = {"analysis_source_sha256": sha256(Path(__file__)), "core_analysis_source_sha256": sha256(Path(__file__).with_name("analyze_correction_study.py")),
                "input_sha256": {str(path): sha256(path) for path in inputs}, "outputs": [{"name": path.name, "sha256": sha256(path)} for path in sorted(output.iterdir())]}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, default=Path("reports/correction_v2_20261107"))
    parser.add_argument("--extension-root", type=Path, default=Path("reports/corrected_formal_extension_20261107"))
    parser.add_argument("--core-cache", type=Path, help="Private rebuilt main cache; defaults to --core-root/cache and must match the frozen identity")
    parser.add_argument("--extension-cache", type=Path, help="Private rebuilt extension preprocessing/index cache; defaults to --extension-root/cache and must match the frozen identity")
    parser.add_argument("--core-protocol", type=Path, default=Path("protocol/correction_v2_20261107.json"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/corrected_formal_extension_20261107.json"))
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.core_root, args.extension_root, args.core_protocol, args.protocol, args.output, merged=args.merged,
                     core_cache=args.core_cache, extension_cache=args.extension_cache)
    print(json.dumps({"status": result["analysis_status"], "cells": len(result["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
