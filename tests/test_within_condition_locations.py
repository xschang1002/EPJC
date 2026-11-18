"""Hand-calculated weighted within/between decomposition and constant invariance."""
import copy
import math

import pytest

from representation_flow.analyze_within_condition_locations import weighted_location_decomposition


def test_unequal_bin_weights_separate_hidden_momentum_from_retained_W_offsets():
    # W1: weights (1,3), means (0,4) -> mean3, variance3.
    # W2: weights (2,2), means (10,14) -> mean12, variance4.
    # Both groups have N=4: pooled within=3.5, between=20.25,
    # centered global=23.75 and uncentered second moment=80.
    rows = [{"thickness_mm":w,"charge":1.,"events":n,"mean_residual_GeV_c":m,"mean_standardized_residual":m/2}
            for w,n,m in ((.1,1,0.),(.1,3,4.),(1.,2,10.),(1.,2,14.))]
    groups, summary = weighted_location_decomposition(rows)
    assert [g["group_mean"] for g in groups] == [3.,12.]
    assert [g["within_W_variance"] for g in groups] == [3.,4.]
    assert summary["within_W_variance"] == 3.5
    assert summary["within_W_rms"] == pytest.approx(math.sqrt(3.5))
    assert summary["between_W_variance"] == 20.25
    assert summary["global_centered_bin_mean_variance"] == 23.75
    assert summary["global_uncentered_bin_mean_rmse"] ** 2 == pytest.approx(80.)
    shifted = copy.deepcopy(rows)
    for row in shifted:
        row["mean_residual_GeV_c"] -= 17.
    shifted_groups, shifted_summary = weighted_location_decomposition(shifted)
    assert [g["within_W_variance"] for g in shifted_groups] == [3.,4.]
    for key in ("within_W_variance","between_W_variance","global_centered_bin_mean_variance"):
        assert shifted_summary[key] == summary[key]
    assert shifted_summary["global_uncentered_bin_mean_rmse"] != summary["global_uncentered_bin_mean_rmse"]
    _, standardized = weighted_location_decomposition(rows,"mean_standardized_residual")
    assert standardized["within_W_variance"] == .875
