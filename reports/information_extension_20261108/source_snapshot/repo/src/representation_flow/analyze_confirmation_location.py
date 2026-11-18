"""Evaluate frozen residual-location diagnostics on confirmation blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from representation_flow.engine import fit_preprocessors, load_source_arrays


def _edges(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, 9))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _audit_block(path: Path, main_arrays: dict, main_train: np.ndarray) -> dict:
    arrays, _ = load_source_arrays(path, validation_fraction=0.1, diagnostic_fraction=0.3)
    p = arrays["p_in"].astype(float)
    t = arrays["thickness"].astype(float)
    q = arrays["charge"].astype(float)
    y = arrays["y"].astype(float)
    main_p = main_arrays["p_in"][main_train].astype(float)
    main_t = main_arrays["thickness"][main_train].astype(float)
    main_q = main_arrays["charge"][main_train].astype(float)
    pre = fit_preprocessors(main_arrays, main_train)["Physics"]
    residual = y - pre.center(p, t, q).astype(float)
    strata = {}
    for thickness in sorted(np.unique(t)):
        for charge in sorted(np.unique(q)):
            key = f"t{thickness:g}__q{charge:g}"
            mask_w = (t == thickness) & (q == charge)
            edges = _edges(main_p[(main_t == thickness) & (main_q == charge)])
            bins = np.digitize(p[mask_w], edges[1:-1], right=False)
            entry = {"count": int(np.sum(mask_w)), "p_edges": [float(v) for v in edges], "representations": {}, "bins": {}}
            for name, target in (("Raw", y[mask_w]), ("Physics", residual[mask_w])):
                grand = float(np.mean(target))
                total = float(np.var(target))
                within = 0.0
                between = 0.0
                bin_rows = {}
                for bin_id in range(8):
                    values = target[bins == bin_id]
                    weight = float(len(values) / len(target))
                    mean = float(np.mean(values))
                    var = float(np.var(values))
                    within += weight * var
                    between += weight * (mean - grand) ** 2
                    bin_rows[str(bin_id)] = {"count": int(len(values)), "mean": mean, "variance": var, "p_mean": float(np.mean(p[mask_w][bins == bin_id]))}
                entry["representations"][name] = {"grand_mean": grand, "total_variance": total, "expected_within_variance": within, "between_mean_variance": between, "variance_identity_residual": total - within - between, "bins": bin_rows}
            for bin_id in range(8):
                entry["bins"][str(bin_id)] = {"location_complexity_excess": (entry["representations"]["Physics"]["bins"][str(bin_id)]["mean"] - entry["representations"]["Physics"]["grand_mean"]) ** 2 - (entry["representations"]["Raw"]["bins"][str(bin_id)]["mean"] - entry["representations"]["Raw"]["grand_mean"]) ** 2}
            strata[key] = entry
    return {"events": int(len(y)), "strata": strata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("data/main_1000000_full_merged"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/representation_v2/exploratory/independent_confirmation_location.json"))
    args = parser.parse_args()
    main_arrays, _ = load_source_arrays(args.main, validation_fraction=0.1, diagnostic_fraction=0.3)
    main_train = np.flatnonzero(main_arrays["split"] == "train")
    result = {"status": "INDEPENDENT_CONFIRMATION_LOCATION_AUDIT_COMPLETE", "train_reference_events": int(len(main_train)), "blocks": {}}
    for label in ("a", "b"):
        result["blocks"][label] = _audit_block(args.data_root / f"confirmation_500k_{label}_full_merged", main_arrays, main_train)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "blocks": len(result["blocks"])}, sort_keys=True))


if __name__ == "__main__":
    main()
