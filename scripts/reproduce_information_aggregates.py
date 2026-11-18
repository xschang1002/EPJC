"""Rebuild figures/tables from COMPLETE archived aggregates, without active-run gates.

Supports both lightweight Git-clone mode (verifying all 240 run_manifest.json files
and cross-verifying all 240 checkpoint SHA-256 digests across frozen manifests)
and full binary-checkpoint mode (when --checkpoint-root contains checkpoint.pt files).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe(root: Path, relative: str) -> Path:
    name = PurePosixPath(str(relative).replace("\\", "/"))
    path = (root / name).resolve()
    if name.is_absolute() or ".." in name.parts or ":" in str(name) or not path.is_relative_to(root.resolve()):
        raise ValueError("unsafe archived path: " + str(relative))
    return path


def verify(path: Path, record: dict) -> Path:
    if not path.is_file() or digest(path) != record["sha256"] or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError("missing or modified archived artifact: " + str(path))
    return path


def bands(point, draws):
    sd = np.std(draws, axis=0, ddof=1)
    active = sd > np.finfo(np.float64).eps * np.maximum(np.abs(point), 1.0) * 64
    standardized = np.zeros_like(draws)
    np.divide(draws - point, sd, out=standardized, where=active)
    critical = np.quantile(np.max(np.abs(standardized).reshape(len(draws), -1), axis=1), 0.95, method="linear")
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    return {
        "bootstrap_standard_error": sd,
        "pointwise_percentile_lower": lower,
        "pointwise_percentile_upper": upper,
        "simultaneous_lower": point - critical * np.where(active, sd, 0),
        "simultaneous_upper": point + critical * np.where(active, sd, 0),
    }


def check_bootstrap(bundle):
    uncertainty = read(bundle["paths"]["uncertainty.json"])
    effects = uncertainty["effects"]
    with np.load(bundle["paths"]["bootstrap_replicates.npz"], allow_pickle=False) as arrays:
        global_draws = arrays["global_contrasts"]
        groups = arrays["primary_groups"]
        counts = arrays["eligible_counts"]
        if global_draws.shape != (2000, 16) or groups.shape != (2000, 42, 8) or counts.shape != (2000, 42):
            raise ValueError("aggregate bootstrap shapes changed")
        calculated = bands(np.array([r["estimate"] for r in effects]), global_draws)
        error = 0.0
        for key, values in calculated.items():
            error = max(error, float(np.max(np.abs(values - np.array([r[key] for r in effects])))))
        labels = [f"{c}/{f}" for f in ("f010", "f100") for c in ("Hidden5", "Coarse2", "Coarse8", "Full5")]
        points = np.empty((42, 8))
        for row in uncertainty["primary_groups"]:
            points[row["group_id"], labels.index(row["primary"])] = row["estimate"]
        joint = bands(points, groups)
        for row in uncertainty["primary_groups"]:
            i, j = row["group_id"], labels.index(row["primary"])
            for side in ("lower", "upper"):
                error = max(error, abs(float(joint["simultaneous_" + side][i, j]) - row["all_primary_groups_simultaneous_" + side]))
        for j, label in enumerate(labels):
            local = bands(points[:, j], groups[:, :, j])
            for row in uncertainty["primary_groups"]:
                if row["primary"] != label:
                    continue
                for side in ("lower", "upper"):
                    error = max(error, abs(float(local["simultaneous_" + side][row["group_id"]]) - row["within_primary_simultaneous_" + side]))
        if error > 1e-11:
            raise ValueError("saved bootstrap summaries do not reproduce archived draws")
    return {"draws": 2000, "global_family": 16, "joint_group_family": 336, "within_primary_family": 42, "maximum_absolute_error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", choices=("primary", "all", "none"), default="primary")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Optional root containing binary checkpoint.pt files for byte hashing")
    parser.add_argument("--require-checkpoints", action="store_true", help="Require all 240 binary checkpoint.pt files on disk")
    args = parser.parse_args()

    repo = args.repository.resolve(strict=True)
    ckpt_repo = args.checkpoint_root.resolve(strict=True) if args.checkpoint_root is not None else repo
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("archived reproduction never overwrites existing output")

    started = time.time()
    report = repo / "reports/information_extension_20261108"
    provenance = report / "provenance"
    paper = repo / "manuscript/representation_v2/overleaf_submission"
    sys.path.insert(0, str(paper))
    figures = importlib.import_module("generate_information_figures")
    tables = importlib.import_module("build_information_tables")
    if figures.ROOT.resolve() != repo:
        raise ValueError("presentation module came from another repository")

    fresh_path = repo / "protocol/fresh_catalog_20261108.json"
    plan = read(fresh_path)
    statistical_path = repo / "protocol/information_statistical_analysis_20261108.json"
    statistical = read(statistical_path)
    if plan.get("status") != "FROZEN_FRESH_CATALOG" or statistical.get("status") != "FROZEN_FRESH_STATISTICAL_ANALYSIS":
        raise ValueError("final frozen plans required")

    for record in plan["source_files"]:
        root = repo if record["root_key"] == "repo" else report / "source_snapshot/audit"
        verify(safe(root, record["path"]), record)

    budget_path = verify(report / "source_snapshot/audit/AUTHORIZATION_AND_BUDGET.json", plan["budget_record"])
    union_path = verify(provenance / "FROZEN_FRESH_MODEL_UNION.json", plan["model_union"])
    union = read(union_path)
    roster_path = verify(provenance / "EXPECTED_FRESH_MODELS.json", union["predetermined_roster"])
    roster = read(roster_path)
    ids = [row["model_id"] for row in union["models"]]
    if union.get("status") != "COMPLETE_FROZEN_FRESH_MODEL_UNION" or len(ids) != 240 or len(set(ids)) != 240 or set(ids) != set(roster["expected_model_ids"]):
        raise ValueError("registered 240-run identity differs")

    rel_roots = {
        "core": "reports/correction_v2_20261107/study",
        "full": "reports/corrected_formal_extension_20261107/study",
        "information": "reports/information_extension_20261108/study",
    }
    roots = {k: repo / v for k, v in rel_roots.items()}
    ckpt_roots = {k: ckpt_repo / v for k, v in rel_roots.items()}
    verify(safe(roots[union["information_freeze"]["root_key"]], union["information_freeze"]["path"]), union["information_freeze"])

    study_rosters = {}
    for k, rdir in roots.items():
        fcm = read(rdir / "FROZEN_CHECKPOINT_MANIFEST.json")
        study_rosters[k] = {m["run_id"]: m["artifacts"] for m in fcm["models"]}

    ledger_path = repo / "CHECKPOINT_SHA256_LEDGER.json"
    ledger_map = {m["model_id"]: m for m in read(ledger_path)["models"]} if ledger_path.is_file() else {}

    checked_manifests = []
    checked_checkpoints = []
    manifest_only_checkpoints = []

    for model in union["models"]:
        rk = model["root_key"]
        mid = model["model_id"]
        rm_rec = model["artifacts"]["run_manifest"]
        rm_path = verify(safe(roots[rk], rm_rec["path"]), rm_rec)
        if read(rm_path).get("experiment_identity") != model["experiment_identity"]:
            raise ValueError("run manifest experiment identity differs: " + mid)
        checked_manifests.append({"model_id": mid, "path": str(rm_path.relative_to(repo)), "sha256": rm_rec["sha256"]})

        ck_rec = model["artifacts"]["checkpoint"]
        run_id = Path(ck_rec["path"]).parent.name
        if run_id not in study_rosters[rk] or study_rosters[rk][run_id]["checkpoint"] != ck_rec:
            raise ValueError("checkpoint digest mismatch against study FROZEN_CHECKPOINT_MANIFEST: " + mid)
        if ledger_map and (mid not in ledger_map or ledger_map[mid]["checkpoint_sha256"] != ck_rec["sha256"] or ledger_map[mid]["checkpoint_bytes"] != ck_rec["bytes"]):
            raise ValueError("checkpoint digest mismatch against CHECKPOINT_SHA256_LEDGER: " + mid)

        ck_candidate = safe(ckpt_roots[rk], ck_rec["path"])
        if ck_candidate.is_file():
            verify(ck_candidate, ck_rec)
            checked_checkpoints.append({"model_id": mid, "path": ck_rec["path"], "sha256": ck_rec["sha256"]})
        elif args.require_checkpoints or args.checkpoint_root is not None:
            raise ValueError("missing binary checkpoint under --checkpoint-root: " + str(ck_candidate))
        else:
            manifest_only_checkpoints.append({"model_id": mid, "sha256": ck_rec["sha256"], "bytes": ck_rec["bytes"]})

    if checked_checkpoints and manifest_only_checkpoints:
        raise ValueError("partial binary checkpoint tree detected; supply either 0 (lightweight Git mode) or all 240 checkpoints")

    source = figures.load_bundle(report / "source_analysis", "source", statistical_path)
    bundles = {block: figures.load_bundle(report / "fresh_analysis" / block, block, statistical_path) for block in ("C", "D")}
    tables.validate_pair_provenance(bundles, source, fresh_path)
    for block, bundle in bundles.items():
        refs = bundle["analysis"]["input_provenance"]
        for key, path in (
            ("union_sha256", union_path),
            ("roster_sha256", roster_path),
            ("generation_sha256", provenance / "GENERATION_COMPLETE.json"),
            ("scoring_summary_sha256", provenance / "scoring_summary.json"),
        ):
            if refs[key] != digest(path):
                raise ValueError("archived lineage mismatch: " + key)

    bootstrap = {block: check_bootstrap(bundle) for block, bundle in bundles.items()}
    names = [row["name"] for row in statistical["primary_contrasts"]]
    prepared = {block: tables.table_rows(bundle, names) for block, bundle in {"source": source, **bundles}.items()}
    out.mkdir(parents=True)
    for block, rows in prepared.items():
        stem = "source_information_primary16" if block == "source" else f"fresh_{block}_primary16"
        bundle = source if block == "source" else bundles[block]
        events = bundle["analysis"]["validation_events" if block == "source" else "eligible_events"]
        (out / (stem + ".tex")).write_text(tables.latex_table(rows, block, events), encoding="utf-8", newline="\n")
        tables.write_csv(out / (stem + ".csv"), rows)
        published = paper / "information_tables" / ("source" if block == "source" else "fresh") / (stem + ".csv")
        if digest(out / (stem + ".csv")) != digest(published):
            raise ValueError("regenerated table CSV differs from published values")

    records = []
    if args.figures != "none":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style = {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
        with plt.rc_context(style):
            figures.primary_figure(bundles, out, records)
            if args.figures == "all":
                figures.primary_figure(bundles, out, records, interactions=True)
                figures.absolute_figure(bundles, out, records)
                figures.components_figure(bundles, out, records)
                figures.source_figure(source, out, records)
                figures.local_figures(bundles, out, records)

    for path in (Path(__file__), paper / "generate_information_figures.py", paper / "build_information_tables.py"):
        target = out / "source_snapshot" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)

    ckpt_mode = "binary_checkpoint_bytes_verified" if len(checked_checkpoints) == 240 else "lightweight_git_manifest_sha256_verified"
    result = {
        "schema": "archived-information-aggregate-reproduction/v2-epjc",
        "status": "COMPLETE_ARCHIVED_AGGREGATE_REPRODUCTION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "repository": str(repo),
        "fresh_plan_sha256": digest(fresh_path),
        "historical_budget_sha256": digest(budget_path),
        "mode": "archived-results-read-only",
        "checkpoint_verification_mode": ckpt_mode,
        "all_source_plan_analysis_hash_gates_preserved": True,
        "registered_run_ids_verified": 240,
        "run_manifest_files_verified_count": len(checked_manifests),
        "checkpoint_sha256_digests_cross_verified_count": 240,
        "binary_checkpoint_files_hashed_count": len(checked_checkpoints),
        "bootstrap_saved_draw_reconstruction": bootstrap,
        "tables": {"rows": 48, "published_csv_byte_identity": True},
        "figures": records,
        "model_inference_training_simulation_or_new_bootstrap": False,
        "boundary": "Aggregate and cryptographic lineage verification plus presentation replay; no neural inference, training, or Geant4 execution.",
        "files": [{"path": p.relative_to(out).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file()],
    }
    (out / "ARCHIVED_REPRODUCTION_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint_verification_mode": ckpt_mode,
                "table_rows": 48,
                "figure_count": len(records) // 2,
                "maximum_bootstrap_band_error": max(v["maximum_absolute_error"] for v in bootstrap.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
