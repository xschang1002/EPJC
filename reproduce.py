"""One-command isolated aggregate reconstruction for the EPJC GitHub release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def safe(relative: str) -> Path:
    n = PurePosixPath(str(relative).replace("\\", "/"))
    p = (ROOT / n).resolve()
    if n.is_absolute() or ".." in n.parts or ":" in str(n) or not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError("Input must be a real file within the repository: " + str(n))
    return p


def data_lines(p: Path) -> list[str]:
    return [
        x
        for x in p.read_text(encoding="utf-8").splitlines()
        if " & " in x and x.rstrip().endswith(r"\\") and not x.startswith(("Contrast", r"\toprule"))
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "reproduced", help="New output directory for reconstructed artifacts")
    ap.add_argument("--figures", choices=["all", "none"], default="all", help="Rebuild all 21 vector/PNG figures or skip plotting")
    ap.add_argument("--checkpoint-root", type=Path, default=None, help="Optional directory containing binary checkpoint.pt files to hash")
    args = ap.parse_args()

    out = args.output.resolve()
    if out.exists() or out == ROOT or ROOT.is_relative_to(out):
        raise FileExistsError("Choose a new non-existent output directory outside or alongside the repository tree")

    started = time.time()
    manifest = read(safe("PACKAGE_MANIFEST.json"))
    if manifest["status"] != "COMPLETE_EPJC_LIGHTWEIGHT_AGGREGATE_RELEASE":
        raise ValueError("Expected COMPLETE_EPJC_LIGHTWEIGHT_AGGREGATE_RELEASE manifest")

    checked = set()
    for row in manifest["files"]:
        p = safe(row["path"])
        if row["path"] in checked or sha(p) != row["sha256"] or p.stat().st_size != row["bytes"]:
            raise ValueError("Missing or modified repository input: " + row["path"])
        if p.suffix.lower() in [".npy", ".jsonl", ".sqlite", ".db", ".pt", ".pth"] or (
            p.suffix.lower() == ".npz" and row["path"] not in manifest["array_allowlist"]
        ):
            raise ValueError("Disallowed binary/event array in lightweight release manifest: " + row["path"])
        checked.add(row["path"])

    out.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLCONFIGDIR"] = str(out / "matplotlib_config")

    commands = []

    def run(relative: str, *cmd_args):
        script = safe(relative)
        cmd = [sys.executable, "-B", str(script), *map(str, cmd_args)]
        print("Running " + script.name + " ...", flush=True)
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = out / (script.stem + ".log")
        log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8", newline="\n")
        commands.append({"script": relative, "args": list(map(str, cmd_args)), "returncode": result.returncode, "log": log.name})
        if result.returncode:
            raise RuntimeError(f"Step {script.name} failed (exit {result.returncode}); see {log}")

    gate_args = ["--repository", ROOT, "--output", out / "aggregate_gate", "--figures", "none"]
    if args.checkpoint_root is not None:
        gate_args.extend(["--checkpoint-root", args.checkpoint_root.resolve(strict=True)])
    run("scripts/reproduce_information_aggregates.py", *gate_args)

    epjc_dir = "manuscript/representation_v2/epjc_submission"
    overleaf_dir = "manuscript/representation_v2/overleaf_submission"

    plot_result = {"requested": args.figures == "all"}
    if args.figures == "all":
        run(overleaf_dir + "/editorial_figures/generate_editorial_figures.py", "--output", out / "editorial_figures")
        new_agg = read(out / "editorial_figures/PLOTTED_AGGREGATES.json")
        for target_dir in (epjc_dir, overleaf_dir):
            ref_agg = read(safe(target_dir + "/editorial_figures/PLOTTED_AGGREGATES.json"))
            if ref_agg != new_agg:
                raise ValueError("Regenerated figure numerical inputs differ from " + target_dir)
        pngs = sorted((out / "editorial_figures").glob("*.png"))
        pdfs = sorted((out / "editorial_figures").glob("*.pdf"))
        if len(pngs) != 21 or len(pdfs) != 21:
            raise ValueError(f"Expected 21 PNG and 21 PDF figures, got {len(pngs)} PNG and {len(pdfs)} PDF")
        plot_result.update(
            {
                "figures": 21,
                "epjc_and_overleaf_numerical_inputs_identical": True,
                "png_byte_identical_count": sum(sha(p) == sha(safe(epjc_dir + "/editorial_figures/" + p.name)) for p in pngs),
                "pdf_metadata_may_differ": True,
            }
        )

    run("scripts/rebuild_editorial_tables.py", "--output", out / "editorial_tables")
    table_check = []
    for p in sorted((out / "editorial_tables").glob("*.tex")):
        for target_dir in (epjc_dir, overleaf_dir):
            ref_tex = safe(target_dir + "/editorial_tables/" + p.name)
            if data_lines(p) != data_lines(ref_tex):
                raise ValueError(f"Table rows differ against {target_dir}: {p.name}")
        table_check.append({"file": p.name, "numeric_rows": len(data_lines(p)), "epjc_and_overleaf_identical": True})

    seed = "reports/positioning_revision_20261108/seed_sensitivity"
    info = ROOT / "reports/information_extension_20261108"
    run(
        seed + "/analyze_seed_sensitivity.py",
        "--analysis-c",
        info / "fresh_analysis/C",
        "--analysis-d",
        info / "fresh_analysis/D",
        "--plan",
        safe("protocol/information_statistical_analysis_20261108.json"),
        "--numerical-review",
        info / "provenance/independent_information_numerical_review.json",
        "--output",
        out / "seed_sensitivity",
        "--include-exploratory-t",
    )
    seed_files = ["seed_sensitivity.csv", "paired_seed_values.csv", "leave_one_seed_out.csv", "exploratory_t_intervals.csv"]
    for n in seed_files:
        if sha(out / "seed_sensitivity" / n) != sha(safe(seed + "/results/" + n)):
            raise ValueError("Seed sensitivity CSV changed: " + n)

    run("verify_saved_diagnostics.py", "--output", out / "diagnostics")

    support = ROOT / "reports/support_audit_20261108"
    support_records = [
        {"file": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "role": "saved aggregate support-audit receipt"}
        for p in [
            support / "input_margin_review.json",
            support / "standardization_replay.json",
            support / "support_terms/results/support_audit.json",
        ]
    ]

    for row in manifest["files"]:
        if sha(safe(row["path"])) != row["sha256"]:
            raise ValueError("Reproduction mutated a tracked repository input: " + row["path"])

    import importlib.metadata
    import numpy
    import scipy

    if args.figures == "all":
        import matplotlib

        matplotlib_version = matplotlib.__version__
    else:
        try:
            matplotlib_version = importlib.metadata.version("matplotlib")
        except importlib.metadata.PackageNotFoundError:
            matplotlib_version = "not_used"

    gate = read(out / "aggregate_gate/ARCHIVED_REPRODUCTION_MANIFEST.json")
    result = {
        "schema": "epjc-github-release-reproduction/v1",
        "status": "PASS_COMPLETE_EPJC_AGGREGATE_REPRODUCTION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 4),
        "package_manifest_sha256": sha(ROOT / "PACKAGE_MANIFEST.json"),
        "all_input_paths_resolved_within_repository": True,
        "all_bound_inputs_unchanged": True,
        "registered_runs_verified": gate["registered_run_ids_verified"],
        "run_manifest_files_verified": gate["run_manifest_files_verified_count"],
        "checkpoint_verification_mode": gate["checkpoint_verification_mode"],
        "checkpoint_sha256_digests_verified": gate["checkpoint_sha256_digests_cross_verified_count"],
        "binary_checkpoint_files_hashed": gate["binary_checkpoint_files_hashed_count"],
        "bootstrap_saved_draw_reconstruction": gate["bootstrap_saved_draw_reconstruction"],
        "current_editorial_figures": plot_result,
        "editorial_tables": table_check,
        "seed_csv_byte_identity": seed_files,
        "support_aggregate_receipts": support_records,
        "diagnostic_summary": "diagnostics/DIAGNOSTIC_VERIFICATION.json",
        "software": {
            "python": sys.version,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib_version,
        },
        "commands": commands,
        "no_training_inference_simulation_new_bootstrap": True,
        "file_outputs": [
            {"path": p.relative_to(out).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*"))
            if p.is_file() and "matplotlib_config" not in p.parts
        ],
    }
    (out / "REPRODUCTION_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "seconds": result["elapsed_seconds"],
                "checkpoint_verification_mode": result["checkpoint_verification_mode"],
                "figures": plot_result,
                "tables_verified": len(table_check),
                "result": str(out / "REPRODUCTION_RESULT.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
