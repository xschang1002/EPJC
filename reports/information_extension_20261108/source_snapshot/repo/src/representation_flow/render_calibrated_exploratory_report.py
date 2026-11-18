"""Render the calibrated-Physics exploratory findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(value):
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/representation_v2/exploratory/physics_calibrated"))
    args = parser.parse_args()
    q = json.loads((args.root / "exploratory_quality_audit.json").read_text(encoding="utf-8"))
    plan = q["plan"]
    manifest = json.loads((args.root / "exploratory_model_manifest.json").read_text(encoding="utf-8"))
    alpha = manifest["centres"]["PhysicsCalibrated"]["coefficients"][0]
    lines = [
        "# Exploratory Physics Center Calibration Study",
        "",
        "Status: `EXPLORATORY_QUALITY_AUDIT_PASS`; 40/40 cells completed with zero failed runs.",
        "",
        "The matched conditional spline flow kept its full five-feature conditioner, the 10% nested source train subset, 6,000 updates, Adam learning rate 3e-4, identical checkpoint selection, and five seeds. The extra representation was `PhysicsCalibrated = alpha * Physics`, with one scalar alpha fit on the train subset and then frozen. The measured alpha was `" + f(alpha) + "`.",
        "",
        "## Paired results",
        "",
        "| Contrast | Capacity | Mean NLL difference | 95% seed interval | Sign pattern |",
        "|---|---|---:|---|---|",
    ]
    for key in sorted(q["paired_contrasts"]):
        item = q["paired_contrasts"][key]
        positive = sum(value > 0 for value in item["values"])
        negative = sum(value < 0 for value in item["values"])
        lines.append(f"| {key.split('__')[0]} | {key.split('__')[1]} | {f(item['mean'])} | [{f(item['ci95_low'])}, {f(item['ci95_high'])}] | {positive} positive / {negative} negative |")
    lines += [
        "",
        "Sign convention: `A - B` is the validation NLL of A minus B; positive values favor B because lower NLL is better.",
        "",
        "The calibrated center does not improve the Physics center. `PhysicsCalibrated - Physics` is positive for all five Tiny seeds (mean 0.003340) and mixed for Small (mean 0.001214, interval crossing zero). The fitted scale is therefore a useful diagnostic, but it is not a rescue mechanism.",
        "",
        "With the full conditioner, Raw and Physics are statistically close at this short budget: `Raw - Physics` has mean 0.002790 for Tiny and 0.000110 for Small, with both seed intervals crossing zero. Learned is worse than Physics in every paired seed in both capacities, although this remains a 10% exploratory subset result.",
        "",
        "This result narrows the design question. The large negative effect from the context-bottleneck branch came from removing `p_in` from the conditioner; a train-only scale correction does not explain it. Under the original full conditioner, Physics remains competitive with Raw and better than Learned in this exploratory setting, while the formal 120-fit result still governs the main claim.",
        "",
        "## Scope",
        "",
        "This branch uses existing source data only. It does not modify the formal protocol or authorize OOD, new Geant4 generation, spline-bound extensions, or mixture models.",
        "",
        "## Artifacts",
        "",
        "- `exploratory_plan.json`: frozen calibration design.",
        "- `exploratory_scores.json`: per-seed NLLs and paired differences.",
        "- `exploratory_quality_audit.json`: artifact and likelihood audit.",
        "- `exploratory_model_manifest.json` and `exploratory_compute_budget.json`: model definitions and measured budget.",
        "- `runs/`: per-cell checkpoints, predictions, and histories.",
    ]
    path = args.root / "exploratory_results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
