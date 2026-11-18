"""Operational diagnostics for the information study, independent of training.

The exact density-coordinate decomposition is not a location/scale/shape
decomposition. The latter labels below name descriptive, nonorthogonal
statistics; none is a causal contribution to a likelihood difference.
"""
from __future__ import annotations

import numpy as np


def _vector(value, n=None, *, finite=True):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0 and n is not None:
        result = np.full(n, float(result), dtype=np.float64)
    if result.ndim != 1 or (n is not None and len(result) != n):
        raise ValueError("expected an aligned one-dimensional array")
    if (finite and not np.isfinite(result).all()) or np.isnan(result).any():
        raise ValueError("invalid numerical diagnostic input")
    return result


def _ids(group_id, n):
    result = np.asarray(group_id)
    if result.shape != (n,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("group IDs must be aligned integers")
    return result


def flow_nll_components(score, *, atol=1e-5, rtol=2e-6) -> dict:
    """Verify log q=log base+log|dz_base/dz| and original-unit density."""
    lp = _vector(score["log_p_y"])
    n = len(lp)
    base = _vector(score["base_logpdf"], n)
    determinant = _vector(score["spline_logabsdet"], n)
    log_q = _vector(score["log_q_z"], n)
    log_s = _vector(score["log_s"], n)
    support = _vector(score["log_support_cdf"], n)
    if np.any(support > atol):
        raise ValueError("log support CDF cannot be positive")
    reconstructed_q = base + determinant
    reconstructed_lp = reconstructed_q - log_s - support
    if not np.allclose(log_q, reconstructed_q, atol=atol, rtol=rtol):
        raise ValueError("base plus spline Jacobian does not reconstruct log_q_z")
    if not np.allclose(lp, reconstructed_lp, atol=atol, rtol=rtol):
        raise ValueError("four density terms do not reconstruct log_p_y")
    return {
        "negative_log_base_density": -base,
        "negative_spline_logabsdet": -determinant,
        "log_preprocessing_scale": log_s,
        "log_support_cdf": support,
        "nll": -lp,
        "reconstruction": {
            "max_abs_log_q_error": float(np.max(np.abs(log_q - reconstructed_q), initial=0)),
            "max_abs_log_p_y_error": float(np.max(np.abs(lp - reconstructed_lp), initial=0)),
            "absolute_tolerance": atol, "relative_tolerance": rtol,
            "equation": "NLL = -base_logpdf - spline_logabsdet + log_s + log_support_cdf",
        },
    }


def paired_component_summary(raw_score, corrected_score, group_id) -> dict:
    """Raw-minus-C NLL components: a positive total favors CorrectedPhysics."""
    raw = flow_nll_components(raw_score)
    corrected = flow_nll_components(corrected_score)
    if raw["nll"].shape != corrected["nll"].shape:
        raise ValueError("paired component arrays differ in length")
    ids = _ids(group_id, len(raw["nll"]))
    fields = ("negative_log_base_density", "negative_spline_logabsdet",
              "log_preprocessing_scale", "log_support_cdf")
    differences = {field: raw[field] - corrected[field] for field in fields}
    differences["nll_gain"] = raw["nll"] - corrected["nll"]
    reconstruction = sum(differences[field] for field in fields) - differences["nll_gain"]
    def summarize(mask):
        means = {field: float(np.mean(value[mask])) for field, value in differences.items()}
        return {"count": int(np.count_nonzero(mask)), **means,
                "component_sum_minus_gain": float(sum(means[f] for f in fields) - means["nll_gain"])}
    return {
        "schema": "information-exact-density-components/v1",
        "sign": "Raw NLL minus CorrectedPhysics NLL; positive total favors CorrectedPhysics",
        "overall": summarize(np.ones(len(ids), dtype=bool)),
        "groups": [{"group_id": int(group), **summarize(ids == group)} for group in np.unique(ids)],
        "max_abs_paired_reconstruction_error": float(np.max(np.abs(reconstruction), initial=0)),
        "raw_reconstruction": raw["reconstruction"],
        "corrected_reconstruction": corrected["reconstruction"],
        "interpretation_limit": (
            "These are base-coordinate density, spline Jacobian, preprocessing-scale "
            "Jacobian and support-normalization terms. They are not unique location, "
            "scale or shape contributions, nor separate causal mechanisms."
        ),
    }


def location_scale_shape(values, group_id, *, units: str) -> dict:
    """Nonorthogonal empirical location, spread and quantile-shape diagnostics."""
    values = _vector(values)
    ids = _ids(group_id, len(values))
    rows = []
    for group in np.unique(ids):
        x = values[ids == group]
        if len(x) < 2:
            raise ValueError("at least two events per diagnostic group are required")
        q01, q10, q25, q50, q75, q90, q99 = np.quantile(x, [.01, .1, .25, .5, .75, .9, .99])
        iqr, central80 = q75 - q25, q90 - q10
        rows.append({
            "group_id": int(group), "count": len(x), "mean": float(np.mean(x)),
            "median": float(q50), "sample_sd": float(np.std(x, ddof=1)), "iqr": float(iqr),
            "quantile_asymmetry_10_50_90": float((q90 + q10 - 2 * q50) / central80) if central80 > 0 else None,
            "central98_width_over_iqr": float((q99 - q01) / iqr) if iqr > 0 else None,
            "q01": float(q01), "q10": float(q10), "q25": float(q25),
            "q50": float(q50), "q75": float(q75), "q90": float(q90), "q99": float(q99),
        })
    return {
        "schema": "information-location-scale-shape-diagnostics/v1", "units": units, "groups": rows,
        "interpretation_limit": (
            "Location, spread and quantile shape are operational, nonorthogonal summaries. "
            "Finite momentum bins pool different exact conditions, so deterministic "
            "centering may change within-bin spread without changing variance at fixed exact conditions."
        ),
    }


def pit_diagnostics(log_cdf, log_sf, group_id, *, complement_atol=2e-6) -> dict:
    """Observed predictive-CDF probabilities, with stable two-tail checks.

    U<=.5 is median coverage; .25<=U<=.75 is predictive-IQR coverage.
    Correct conditional distributions yield Uniform(0,1) PITs. These pooled
    group checks alone do not establish full conditional calibration.
    """
    lcdf = _vector(log_cdf, finite=False)
    lsf = _vector(log_sf, len(lcdf), finite=False)
    ids = _ids(group_id, len(lcdf))
    if np.any(lcdf > 1e-7) or np.any(lsf > 1e-7):
        raise ValueError("log CDF/SF cannot be positive")
    u, sf = np.exp(lcdf), np.exp(lsf)
    if not np.allclose(u + sf, 1.0, atol=complement_atol, rtol=0):
        raise ValueError("predictive CDF and SF are inconsistent")
    rows = []
    for group in np.unique(ids):
        mask = ids == group
        x, low, high = u[mask], lcdf[mask], lsf[mask]
        q25, median, q75 = np.quantile(x, [.25, .5, .75])
        rows.append({
            "group_id": int(group), "count": len(x), "pit_mean": float(np.mean(x)),
            "pit_median": float(median), "pit_iqr": float(q75 - q25),
            "median_coverage": float(np.mean(low <= np.log(.5))),
            "predictive_iqr_coverage": float(np.mean((low >= np.log(.25)) & (high >= np.log(.25)))),
            "predictive_central90_coverage": float(np.mean((low >= np.log(.05)) & (high >= np.log(.05)))),
            "lower_05_frequency": float(np.mean(low < np.log(.05))),
            "upper_05_frequency": float(np.mean(high < np.log(.05))),
            "lower_01_frequency": float(np.mean(low < np.log(.01))),
            "upper_01_frequency": float(np.mean(high < np.log(.01))),
        })
    return {
        "schema": "information-pit-diagnostics/v1", "groups": rows,
        "uniform_reference": {"pit_mean": .5, "pit_median": .5, "pit_iqr": .5,
                              "median_coverage": .5, "predictive_iqr_coverage": .5,
                              "predictive_central90_coverage": .9,
                              "lower_05_frequency": .05, "upper_05_frequency": .05,
                              "lower_01_frequency": .01, "upper_01_frequency": .01},
        "interpretation_limit": (
            "Median/PIT location, interquartile spread and tail frequencies are not "
            "orthogonal contributions to NLL. Group-aggregated calibration may conceal "
            "within-group errors. No score-based model selection or causal attribution follows."
        ),
    }


def source_location_burden(values, p_in, thickness, charge, *, units: str) -> dict:
    """Source-only 56-bin approximation inside retained condition cells.

    56 equal-width momentum bins refine the reporting 7 and input 2/8 bin
    partitions. Full5 observes exact momentum: its theoretical unobserved-V
    mean burden is zero; a finite-bin statistic is not an estimate of it.
    Values should use the same frozen preprocessor as the compared model.
    """
    values = _vector(values)
    p = _vector(p_in, len(values))
    t = _vector(thickness, len(values))
    q = _vector(charge, len(values))
    thickness_values = np.asarray([.1, .3, 1.0])
    thickness_index = np.argmin(np.abs(t[:, None] - thickness_values), axis=1)
    if (np.any((p < .3) | (p > 1.0))
            or not np.allclose(t, thickness_values[thickness_index], rtol=0, atol=2e-8)
            or not np.isin(q, [-1, 1]).all()):
        raise ValueError("source conditions are outside the frozen physics domain")
    edges = np.linspace(.3, 1.0, 57)
    fine = np.minimum(np.searchsorted(edges, p, side="right") - 1, 55)
    stratum = thickness_index * 2 + (q == 1)
    output = []
    for context, bins in (("Hidden5", 1), ("Coarse2", 2), ("Coarse8", 8)):
        visible_bin = fine // (56 // bins)
        parents = stratum * bins + visible_bin
        rows, weighted_sum = [], 0.0
        for parent in range(6 * bins):
            mask = parents == parent
            if not np.any(mask):
                raise ValueError("every retained-condition cell needs source events")
            parent_mean = float(np.mean(values[mask]))
            position_variance = 0.0
            local_rows = []
            for fine_bin in range((parent % bins) * (56 // bins), (parent % bins + 1) * (56 // bins)):
                submask = mask & (fine == fine_bin)
                n = int(np.count_nonzero(submask))
                if n == 0:
                    raise ValueError("all 56 x 6 source diagnostic bins must be populated")
                mean = float(np.mean(values[submask]))
                position_variance += n * (mean - parent_mean) ** 2
                local_rows.append({"momentum_bin56": fine_bin, "count": n, "mean": mean})
            count = int(np.count_nonzero(mask))
            weighted_sum += position_variance
            rows.append({"retained_cell": parent, "physical_stratum": parent // bins,
                         "visible_momentum_bin": parent % bins, "count": count,
                         "mean": parent_mean, "between_fine_bin_mean_variance": position_variance / count,
                         "fine_bins": local_rows})
        output.append({"context": context, "retained_cells": rows,
                       "event_weighted_position_variance": weighted_sum / len(values),
                       "event_weighted_position_rms": float(np.sqrt(weighted_sum / len(values)))})
    return {
        "schema": "information-source-within-visible-location/v1", "units": units,
        "momentum_edges": edges.tolist(), "contexts": output,
        "Full5": {"exact_conditioning_unobserved_momentum_mean_variance": 0.0,
                  "status": "theoretical property given exact momentum, not a finite-bin estimate"},
        "interpretation_limit": (
            "Source-only binned conditional-mean approximation, not a causal NLL decomposition. "
            "The same preprocessor must be used for each representation/fraction. Global mu "
            "cancels from within-cell location differences. Uniform 42-group fresh reporting "
            "and this 336-cell source diagnostic have different definitions and purposes."
        ),
    }
