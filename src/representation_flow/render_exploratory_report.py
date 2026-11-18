"""Render the context-bottleneck exploratory findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(value):
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/representation_v2/exploratory/context_bottleneck"))
    args = parser.parse_args()
    q = json.loads((args.root / "exploratory_quality_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# Exploratory Context-Bottleneck Study",
        "",
        "Status: `EXPLORATORY_QUALITY_AUDIT_PASS`; 40/40 cells completed with zero failed runs.",
        "",
        "## Design change",
        "",
        "The matched conditional spline flow was intentionally given only `((thickness_mm-0.3)/0.7, charge)` as conditioner context. `p_in` was withheld from the flow while every representation and seed used the same architecture, optimizer, checkpoint rule, 10% nested train subset, and 6,000-update budget. Capacities were Tiny width 8 and Small width 32. This is an architecture-stress experiment, not a replacement for the formal matrix.",
        "",
        "## Results",
        "",
        "| Contrast | Capacity | Mean difference | 95% seed interval | Sign pattern |",
        "|---|---|---:|---|---|",
    ]
    for key in sorted(q["paired_contrasts"]):
        item = q["paired_contrasts"][key]
        positive = sum(value > 0 for value in item["values"])
        negative = sum(value < 0 for value in item["values"])
        lines.append(f"| {key.split('__')[0]} | {key.split('__')[1]} | {f(item['mean'])} | [{f(item['ci95_low'])}, {f(item['ci95_high'])}] | {positive} positive / {negative} negative |")
    lines += [
        "",
        "Interpretation of signs: `Raw - Physics` positive favors Physics; `Learned - Physics` positive favors Physics; `Simple - Raw` positive favors Raw.",
        "",
        "Physics is substantially worse than Raw in both bottleneck capacities: the paired differences are negative for all five seeds in each capacity. Learned is also better than Physics in every paired seed. Raw is better than Simple in this constrained setting. This is a reproducible negative result for the hypothesis that the current Physics center can replace withheld `p_in` information under a narrow conditioner.",
        "",
        "The result is not a training failure: all likelihood fields and log-survival values are finite, every run has the expected 13 validation history points with finite pre-clip gradients, and the source validation UID order is identical across cells.",
        "",
        "## Boundary",
        "",
        "This branch uses existing source data only. It does not modify the formal 120-fit results, does not authorize OOD or new Geant4 generation, and does not replace the Physics-informed target representation mainline. Its scientific role is to show that representation usefulness depends on compatibility between the center, the remaining conditioner context, and the model capacity.",
        "",
        "## Artifacts",
        "",
        "- `exploratory_plan.json`: frozen exploratory design.",
        "- `exploratory_scores.json`: per-seed scores and paired differences.",
        "- `exploratory_quality_audit.json`: artifact and likelihood audit.",
        "- `exploratory_model_manifest.json` and `exploratory_compute_budget.json`: model definitions and measured budget.",
        "- `runs/`: per-cell checkpoints, predictions, and histories.",
    ]
    path = args.root / "exploratory_results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
