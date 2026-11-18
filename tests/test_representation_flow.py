"""Focused correctness checks for the production representation-flow engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from representation_flow.engine import ConditionalSplineFlow, fit_preprocessors, load_source_arrays, run_a0_checks, subset_indices, common_features

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_REPORTS = REPOSITORY_ROOT / "reports/representation_v2"
MAIN_DATA = REPOSITORY_ROOT / "data/main_1000000_full_merged"


def test_production_a0_contract_passes():
    result = run_a0_checks()
    assert result["status"] == "A0_PRODUCTION_ENGINE_PASS"
    assert result["inversion_max_abs_error"] < result["tolerances"]["inversion"]
    assert abs(result["truncated_density_integral"] - 1.0) < 2e-5


def test_flow_checkpoint_terms_are_finite_and_support_normalized(tmp_path):
    flow = ConditionalSplineFlow(hidden=32, seed=19)
    y = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32)
    center = torch.zeros_like(y)
    p_in = torch.tensor([0.5, 0.6, 0.8], dtype=torch.float32)
    context = torch.zeros((3, 5), dtype=torch.float32)
    terms = flow.original_terms(y, center, 0.0, 0.1, p_in, context)
    assert all(torch.isfinite(value).all() for value in terms.values())
    assert torch.all(terms["b"] > terms["z"])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(flow.state_dict(), checkpoint)
    reloaded = ConditionalSplineFlow(hidden=32, seed=19)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    reloaded_terms = reloaded.original_terms(y, center, 0.0, 0.1, p_in, context)
    assert torch.allclose(terms["log_p_y"], reloaded_terms["log_p_y"], atol=1e-7)


@pytest.mark.skipif(
    not (HISTORICAL_REPORTS / "prototype_scores.json").is_file(),
    reason="Historical report assets are not included in the source-only package",
)
def test_final_prototype_reports_have_all_cells_and_required_prediction_fields():
    root = HISTORICAL_REPORTS
    scores = json.loads((root / "prototype_scores.json").read_text(encoding="utf-8"))
    assert scores["status"] == "COMPLETED_A0_PLUS_PROTOTYPE"
    assert len(scores["runs"]) == 24
    assert scores["failed_runs"] == []
    required = {"event_uid", "split", "domain", "representation", "subset_id", "n", "capacity", "seed", "y", "c", "mu", "s", "z", "b", "log_q_z", "log_F_b", "log_p_y", "logCDF_y", "logSF_y", "threshold_Q"}
    sample = next((root / "runs").glob("*/predictions.npz"))
    with np.load(sample) as data:
        assert required.issubset(data.files)
        assert len(data["event_uid"]) == 100042


@pytest.mark.skipif(
    not (MAIN_DATA / "events.jsonl").is_file()
    or not (HISTORICAL_REPORTS / "runs/prototype__Raw__Small__seed11/checkpoint.pt").is_file(),
    reason="Historical checkpoint replay requires the separately supplied main data and model assets",
)
def test_saved_prototype_checkpoint_reload_matches_exported_likelihood():
    root = HISTORICAL_REPORTS
    arrays, _ = load_source_arrays(MAIN_DATA, validation_fraction=0.1, diagnostic_fraction=0.3)
    train_indices = subset_indices(arrays, fraction=0.1, catalog_id=4101)
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    preprocessors = fit_preprocessors(arrays, train_indices)
    flow = ConditionalSplineFlow(hidden=32, seed=11)
    checkpoint = torch.load(root / "runs/prototype__Raw__Small__seed11/checkpoint.pt", weights_only=False)
    flow.load_state_dict(checkpoint["state_dict"])
    idx = validation_indices[:4096]
    p = arrays["p_in"][idx]
    t = arrays["thickness"][idx]
    q = arrays["charge"][idx]
    y = torch.tensor(arrays["y"][idx], dtype=torch.float32)
    c = torch.tensor(preprocessors["Raw"].center(p, t, q), dtype=torch.float32)
    context = torch.tensor(common_features(p, t, q), dtype=torch.float32)
    terms = flow.original_terms(y, c, preprocessors["Raw"].mu, preprocessors["Raw"].scale, torch.tensor(p, dtype=torch.float32), context)
    with np.load(root / "runs/prototype__Raw__Small__seed11/predictions.npz") as exported:
        # The export is float32 and the production scorer evaluates in chunks;
        # pointwise differences at spline knots stay below this audit tolerance.
        assert np.allclose(terms["log_p_y"].detach().numpy(), exported["log_p_y"][:4096], rtol=2e-6, atol=1e-4)
