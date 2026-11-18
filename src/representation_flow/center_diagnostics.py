"""Source-domain center and spline-interval diagnostics; no model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from representation_flow.engine import fit_preprocessors, load_source_arrays, subset_indices


REPRESENTATIONS = ("Raw", "Simple", "Physics", "Learned")


def _weighted_group_location(y, residual, p, thickness, charge):
    p_bins = np.linspace(0.3, 1.0, 9)
    group = np.column_stack([
        np.digitize(p, p_bins[1:-1], right=False),
        thickness,
        charge,
    ])
    means = []
    counts = []
    for key in np.unique(group, axis=0):
        mask = np.all(group == key, axis=1)
        if np.sum(mask) < 10:
            continue
        means.append(float(np.mean(residual[mask])))
        counts.append(int(np.sum(mask)))
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=float)
    average = float(np.average(means, weights=counts))
    spread = float(np.sqrt(np.average((means - average) ** 2, weights=counts)))
    return {"groups": int(len(means)), "weighted_group_mean": average, "conditional_location_std": spread, "group_counts_min": int(np.min(counts)) if len(counts) else 0}


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    return {key: float(np.quantile(values, q)) for key, q in (("q01", 0.01), ("q05", 0.05), ("q50", 0.5), ("q95", 0.95), ("q99", 0.99))}


def diagnose(merged: Path):
    arrays, manifest = load_source_arrays(merged, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_indices = subset_indices(arrays, fraction=0.1, catalog_id=4101)
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    preprocessors = fit_preprocessors(arrays, train_indices)
    output = {
        "status": "COMPLETED_CENTER_DIAGNOSTICS_NO_NEW_MODELS",
        "source_manifest_sha256": manifest.get("merged_file_sha256", {}).get("events"),
        "train_subset_events": int(len(train_indices)),
        "validation_events": int(len(validation_indices)),
        "bound": 8.0,
        "representations": {},
        "interpretation_boundary": "These are source-domain position and coordinate diagnostics. They do not establish OOD behavior, optimization causality, or Physics superiority.",
    }
    raw_spreads = {}
    for name in REPRESENTATIONS:
        pre = preprocessors[name]
        output["representations"][name] = {}
        for split_name, indices in (("train", train_indices), ("validation", validation_indices)):
            p = arrays["p_in"][indices]
            t = arrays["thickness"][indices]
            q = arrays["charge"][indices]
            y = arrays["y"][indices]
            center = pre.center(p, t, q)
            residual = y - center
            z = (residual - pre.mu) / pre.scale
            location = _weighted_group_location(y, residual, p, t, q)
            raw_spreads.setdefault(split_name, {})
            raw_spreads[split_name][name] = location["conditional_location_std"]
            output["representations"][name][split_name] = {
                "center_mean": float(np.mean(center)),
                "center_std": float(np.std(center)),
                "residual_mean": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
                "conditional_location": location,
                "standardized_target": {"mean": float(np.mean(z)), "std": float(np.std(z)), **_quantiles(z), "outside_spline_fraction_abs_z_gt_8": float(np.mean(np.abs(z) > 8.0)), "min": float(np.min(z)), "max": float(np.max(z))},
                "fit_upper_coordinate": {"q01": float(np.quantile((p - center - pre.mu) / pre.scale, 0.01)), "q50": float(np.quantile((p - center - pre.mu) / pre.scale, 0.5)), "q99": float(np.quantile((p - center - pre.mu) / pre.scale, 0.99)), "outside_spline_fraction_abs_b_gt_8": float(np.mean(np.abs((p - center - pre.mu) / pre.scale) > 8.0))},
            }
    for split_name in ("train", "validation"):
        raw = raw_spreads[split_name]["Raw"]
        for name in REPRESENTATIONS:
            output["representations"][name][split_name]["conditional_location_delta_vs_raw"] = raw_spreads[split_name][name] - raw
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/optimization_audit/center_diagnostics.json"))
    args = parser.parse_args()
    result = diagnose(args.merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
