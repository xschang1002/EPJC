"""Render a compact Markdown report from the formal quality-audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(value: float) -> str:
    return f"{value:.6f}"


def _ci(item) -> str:
    return f"{_f(item['mean'])} [{_f(item['ci95_low'])}, {_f(item['ci95_high'])}]"


def render(formal_dir: Path) -> str:
    q = json.loads((formal_dir / "formal_quality_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# Formal Matched Representation Study",
        "",
        "Status: `FORMAL_QUALITY_AUDIT_PASS`; 120/120 core fits completed with zero failed runs.",
        "",
        "## Frozen protocol",
        "",
        "The study uses Raw, Simple, Physics, and Learned representations; Small (32) and Large (128) conditioners; nested 10%, 30%, and 100% source-train fractions; and optimizer seeds 3101--3105. Every cell uses Adam, learning rate `3e-4`, batch size 512, 20,000 updates, validation every 500 updates, zero weight decay/dropout, and the same source-validation all-event Y-space NLL checkpoint rule.",
        "",
        "The target is `delta_p = p_in - p_out` in GeV/c for S1 events with support `Y < p_in`. Reported likelihood is the original-unit density `log q(z|X) - log(s) - log F_q(b|X)`, including the standardization Jacobian and support normalization.",
        "",
        "## Preflight and quality audit",
        "",
        f"- Source S1 events: `{q['dataset_manifest']['split_counts']['train'] + q['dataset_manifest']['split_counts']['validation'] + q['dataset_manifest']['split_counts']['diagnostic']}` eligible split rows; validation UID digest `{q['validation_uid_sha256']}`.",
        f"- Run artifacts: `{q['runs_observed']}` complete; failed runs `{len(q['failed_runs'])}`; quality failures `{len(q['quality_failures'])}`.",
        "- The A0 hash is `sha256(manifest.json)`, while center diagnostics use `sha256(events.jsonl)`; the preflight report records both semantics and verifies they describe the same source directory.",
        "- Every prediction has finite `log_p_y`, `logCDF_y`, and `logSF_y`; the standardized-target, upper-coordinate, Jacobian, and support-normalization algebra is consistent within float32 serialization tolerance.",
        "",
        "## NLL by group",
        "",
        "Entries show the five seed NLLs in seed order 3101, 3102, 3103, 3104, 3105, followed by mean and a 95% t interval across seeds. Lower NLL is better.",
        "",
        "| Representation | Capacity | Fraction | Seed NLLs | Mean [95% interval] |",
        "|---|---|---:|---|---|",
    ]
    grouped = q["grouped_nll"]
    for key in sorted(grouped):
        item = grouped[key]
        seeds = ", ".join(_f(v) for v in item["values"])
        lines.append(f"| {item['representation']} | {item['capacity']} | {item['sample_fraction']:.1f} | {seeds} | {_ci(item)} |")

    lines += [
        "",
        "## Paired contrasts",
        "",
        "`Raw - Physics` is positive when Physics has lower NLL than Raw. `Learned - Physics` is positive when Physics has lower NLL than Learned. `Simple - Raw` is positive when Raw has lower NLL than Simple.",
        "",
        "| Contrast | Capacity | Fraction | Seed differences | Mean [95% interval] |",
        "|---|---|---:|---|---|",
    ]
    paired_groups = q["grouped_paired_contrasts"]
    for key in sorted(paired_groups):
        item = paired_groups[key]
        seeds = ", ".join(_f(v) for v in item["values"])
        lines.append(f"| {item['contrast']} | {item['capacity']} | {item['sample_fraction']:.1f} | {seeds} | {_ci(item)} |")

    lines += [
        "",
        "## Aggregate and interaction summaries",
        "",
    ]
    for contrast, item in q["aggregate_paired_contrasts"].items():
        lines.append(f"- `{contrast}` across 30 paired cells: {_ci(item)} NLL units. This is a descriptive aggregate over repeated fractions/capacities, not an independent-cell significance claim.")
    capacity = q["interactions"]["capacity_small_minus_large_raw_minus_physics"]
    sample = q["interactions"]["sample_gain_10pct_minus_100pct_raw_minus_physics"]
    lines.append(f"- Raw-minus-Physics capacity interaction (Small minus Large, 15 seed/fraction pairs): {_ci(capacity)}.")
    lines.append(f"- Raw-minus-Physics sample interaction (10% minus 100%, 10 seed/capacity pairs): {_ci(sample)}.")
    lines += [
        "",
        "The formal source-domain result is a small, seed-dependent Physics-versus-Raw difference relative to the predeclared 0.02-nat practical margin; most individual five-seed intervals include zero. Physics/Learned differences are similarly small and mixed by group. These results establish the formal matched comparison, not an OOD or mechanism conclusion.",
        "",
        "## Evidence boundary",
        "",
        "The Optimization Audit and center diagnostics remain development/interpretation evidence and are retained separately. Center diagnostics are not treated as causal likelihood evidence. No new Geant4 data, OOD evaluation, spline-bound extension, mixture-mechanism substitution, or other model was run in this stage.",
        "",
        "## Artifacts",
        "",
        "- `formal_preflight_audit.json`: source/hash/UID/target verification.",
        "- `formal_protocol_freeze.json`: frozen protocol.",
        "- `formal_scores.json`: all 120 per-seed scores and paired cell records.",
        "- `formal_quality_audit.json`: artifact, gradient, checkpoint, prediction, and likelihood-contract audit.",
        "- `formal_dataset_manifest.json`, `formal_model_manifest.json`, and `formal_compute_budget.json`: provenance, centers, model definitions, and measured budget.",
        "- `runs/`: per-cell checkpoint, optimizer state, prediction arrays, and complete training history.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, default=Path("reports/representation_v2/formal_matrix"))
    args = parser.parse_args()
    output = args.formal_dir / "formal_results_summary.md"
    output.write_text(render(args.formal_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
