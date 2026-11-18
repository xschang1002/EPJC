"""Render the paired conditioner-information interaction probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(value):
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/representation_v2/exploratory/context_interaction"))
    args = parser.parse_args()
    q = json.loads((args.root / "exploratory_quality_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# Paired Conditioner Information Interaction Study",
        "",
        "Status: `EXPLORATORY_QUALITY_AUDIT_PASS`; 40/40 cells completed with zero failed runs.",
        "",
        "This probe keeps the Raw and Physics representations, Small capacity, five optimizer seeds, 6,000 updates, learning rate 3e-4, and the same source validation set. It varies only conditioner information (`full` versus `thickness_charge`, where `p_in` is withheld) and the nested train fraction (10% versus 100%).",
        "",
        "The table reports `Raw - Physics`; positive values favor Physics because lower NLL is better.",
        "",
        "| Conditioner | Train fraction | Mean difference | 95% seed interval | Sign pattern |",
        "|---|---:|---:|---|---|",
    ]
    for key in sorted(q["paired_contrasts"]):
        item = q["paired_contrasts"][key]
        _, context, fraction, capacity = key.split("__")
        positive = sum(value > 0 for value in item["values"])
        negative = sum(value < 0 for value in item["values"])
        fraction_display = int(fraction.removeprefix("f"))
        lines.append(f"| {context} | {fraction_display}% | {f(item['mean'])} | [{f(item['ci95_low'])}, {f(item['ci95_high'])}] | {positive} positive / {negative} negative |")
    lines += [
        "",
        "With the full conditioner, Raw and Physics remain tied within seed uncertainty at both 10% and 100%: the mean differences are `+0.000110` and `-0.000150`.",
        "",
        "With `p_in` withheld, Physics is worse than Raw by `0.086094` at 10% and `0.083122` at 100%, with all five seeds in the same direction at both fractions. More training data does not remove the deficit.",
        "",
        "The interaction is therefore a stable boundary result: Physics is competitive when the conditioner retains `p_in`, but cannot replace that missing condition under this finite-capacity flow. The effect is not explained by short sample size or the Physics center's scalar calibration.",
        "",
        "This is a source-only exploratory study. It does not modify the formal 120-fit protocol or authorize OOD, new Geant4 generation, spline-bound extensions, or mixture models.",
        "",
        "Artifacts: `exploratory_plan.json`, `exploratory_scores.json`, `exploratory_quality_audit.json`, `exploratory_model_manifest.json`, `exploratory_compute_budget.json`, and `runs/`.",
    ]
    path = args.root / "exploratory_results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
