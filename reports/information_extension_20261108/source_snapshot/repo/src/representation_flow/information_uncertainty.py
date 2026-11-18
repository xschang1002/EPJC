"""Paired event uncertainty for frozen information-study models.

This module does not load a model, select a checkpoint, or read a catalog.
Inputs are aligned per-event scores and the *launched* counts of the fixed
simulation strata. S0 events enter the resampling distribution as a combined
zero-numerator/zero-denominator atom; their scientific loss is not set to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np


CONTEXT_ORDER = ("Hidden5", "Coarse2", "Coarse8", "Full5")
FRACTION_ORDER = ("f010", "f100")
DEFAULT_BOOTSTRAP_SEED = 20261108


@dataclass
class PairedBootstrapResult:
    summary: dict
    global_replicates: np.ndarray
    primary_group_replicates: np.ndarray
    eligible_count_replicates: np.ndarray


def primary_labels() -> list[str]:
    return [f"{context}/{fraction}" for fraction in FRACTION_ORDER for context in CONTEXT_ORDER]


def planned_contrasts() -> dict[str, np.ndarray]:
    """Eight gains, six adjacent information-addition effects, two endpoint effects.

    Every gain is mean_seed(log p_C - log p_Raw). Information-addition effects
    are more-informed gain minus less-informed gain; their direction is open.
    """
    labels = primary_labels()
    eye = np.eye(len(labels))
    result = {f"gain:{label}": eye[i] for i, label in enumerate(labels)}
    for fraction in FRACTION_ORDER:
        for less, more in zip(CONTEXT_ORDER[:-1], CONTEXT_ORDER[1:]):
            result[f"gain_change:{more}_minus_{less}/{fraction}"] = (
                eye[labels.index(f"{more}/{fraction}")]
                - eye[labels.index(f"{less}/{fraction}")]
            )
        result[f"gain_change:Full5_minus_Hidden5/{fraction}"] = (
            eye[labels.index(f"Full5/{fraction}")]
            - eye[labels.index(f"Hidden5/{fraction}")]
        )
    return result


def paired_seed_gains(raw_log_scores, corrected_log_scores) -> tuple[np.ndarray, dict]:
    """Return the eventwise average of paired seed log-score differences.

    N x S arrays must already have identical event and seed ordering. This is
    an average of fixed-model scores, not a log score of a density ensemble.
    """
    raw = np.asarray(raw_log_scores, dtype=np.float64)
    corrected = np.asarray(corrected_log_scores, dtype=np.float64)
    if raw.ndim != 2 or raw.shape != corrected.shape or min(raw.shape) < 2:
        raise ValueError("paired score arrays must have equal N x S shape, N/S >= 2")
    if not np.isfinite(raw).all() or not np.isfinite(corrected).all():
        raise ValueError("paired log scores must be finite")
    per_seed = np.mean(corrected - raw, axis=0)
    return np.mean(corrected - raw, axis=1), {
        "estimand": "arithmetic mean of paired fixed-model log-score differences",
        "is_density_ensemble_log_score": False,
        "seed_means": per_seed.tolist(),
        "seed_mean": float(np.mean(per_seed)),
        "seed_sample_sd": float(np.std(per_seed, ddof=1)),
        "seed_min": float(np.min(per_seed)),
        "seed_max": float(np.max(per_seed)),
        "seed_variation_scope": "optimizer variation on shared training/evaluation data",
    }


def practical_effect(delta: float, interval: Sequence[float] | None = None) -> dict:
    """Log-score change and its geometric-mean density-ratio interpretation."""
    delta = float(delta)
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    result = {
        "delta_nat_per_eligible_event": delta,
        "geometric_mean_paired_density_ratio": float(np.exp(delta)),
        "geometric_mean_paired_density_ratio_percent_change": float(100 * np.expm1(delta)),
        "interpretation": (
            "For the mean-over-seed score, this is a geometric average over paired "
            "event/model density ratios. It is not an eventwise or uniform ratio, "
            "a prediction-accuracy percentage, or a downstream benefit. N times "
            "delta is a cumulative log-score contrast only. No equivalence margin is implied."
        ),
    }
    if interval is not None:
        bounds = np.asarray(interval, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("interval must be ordered finite lower/upper bounds")
        result["geometric_density_ratio_interval"] = np.exp(bounds).tolist()
    return result


def _bands(point: np.ndarray, replicates: np.ndarray, confidence: float) -> dict:
    """Pointwise percentile intervals and one centered max-|t| family band.

    The t denominator is the bootstrap SD of each effect, shared across
    replicates. The band is bootstrap approximate, not an exact finite-sample
    confidence guarantee. Deterministic coordinates receive zero width.
    """
    point = np.asarray(point, dtype=np.float64)
    replicates = np.asarray(replicates, dtype=np.float64)
    sd = np.std(replicates, axis=0, ddof=1)
    scale = np.maximum(np.abs(point), 1.0)
    active = sd > np.finfo(np.float64).eps * scale * 64
    standardized = np.zeros_like(replicates)
    np.divide(replicates - point, sd, out=standardized, where=active)
    maxima = np.max(np.abs(standardized).reshape(len(replicates), -1), axis=1)
    critical = float(np.quantile(maxima, confidence, method="linear"))
    effective_sd = np.where(active, sd, 0.0)
    alpha = (1.0 - confidence) / 2.0
    percentile = np.quantile(replicates, [alpha, 1.0 - alpha], axis=0, method="linear")
    return {
        "bootstrap_standard_error": sd,
        "pointwise_percentile_lower": percentile[0],
        "pointwise_percentile_upper": percentile[1],
        "simultaneous_lower": point - critical * effective_sd,
        "simultaneous_upper": point + critical * effective_sd,
        "max_abs_t_critical": critical,
    }


def stratified_paired_bootstrap(
    event_gains,
    group_id,
    launched_counts: Mapping[int, int],
    *,
    labels: Sequence[str],
    contrasts: Mapping[str, Sequence[float]] | None = None,
    n_resamples: int = 2000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    batch_size: int = 32,
    confidence: float = 0.95,
    max_seconds: float = 900.0,
    deadline_unix: float | None = None,
) -> PairedBootstrapResult:
    """Exact multinomial bootstrap within fixed launched strata.

    The N rows are eligible S1 events. All compared models use each same
    resampled event weight. Resampling includes S0 counts, so the denominator
    and the eligible mixture across strata can vary. No new randomness for
    optimizer seeds, data-generation settings, or between-catalog effects is
    included. Process/RNG independence assumptions belong in the frozen plan.
    """
    start = time.monotonic()
    gains = np.asarray(event_gains, dtype=np.float64)
    ids = np.asarray(group_id)
    if gains.ndim != 2 or gains.shape[1] != len(labels) or gains.shape[0] < 2:
        raise ValueError("gains must have shape eligible events x distinct primary labels")
    if len(set(labels)) != len(labels) or not np.isfinite(gains).all():
        raise ValueError("primary labels must be unique and gains finite")
    if ids.shape != (len(gains),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("integer group_id must align with every eligible event")
    groups = sorted(launched_counts)
    if not groups or set(np.unique(ids).tolist()) != set(groups):
        raise ValueError("every launched stratum must have observed eligible events, with no extra IDs")
    if n_resamples < 100 or batch_size < 1 or not 0.5 < confidence < 1 or max_seconds <= 0:
        raise ValueError("invalid bootstrap precision, batch, confidence, or time budget")
    if contrasts is None:
        contrasts = {str(name): np.eye(len(labels))[i] for i, name in enumerate(labels)}
    contrast_names = list(contrasts)
    matrix = np.asarray(list(contrasts.values()), dtype=np.float64)
    if matrix.shape != (len(contrast_names), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError("each contrast must give one finite coefficient per primary label")
    if not contrast_names:
        raise ValueError("at least one frozen contrast is required")

    def check_time():
        if time.monotonic() - start > max_seconds or (
            deadline_unix is not None and time.time() >= deadline_unix
        ):
            raise TimeoutError("bootstrap exceeded its fixed local/global deadline; no complete result")

    check_time()
    b, g, k = n_resamples, len(groups), len(labels)
    global_numerator = np.zeros((b, k), dtype=np.float64)
    counts = np.zeros((b, g), dtype=np.int64)
    group_replicates = np.empty((b, g, k), dtype=np.float64)
    group_points = np.empty((g, k), dtype=np.float64)
    observed_counts = []
    strata = []
    streams = np.random.SeedSequence(seed).spawn(g)
    for gi, group in enumerate(groups):
        check_time()
        values = np.ascontiguousarray(gains[ids == group])
        n_eligible = len(values)
        n_launched = launched_counts[group]
        if isinstance(n_launched, (bool, np.bool_)) or int(n_launched) != n_launched:
            raise ValueError("launched counts must be integers")
        n_launched = int(n_launched)
        if n_eligible < 20 or n_launched < n_eligible:
            raise ValueError("each stratum needs >=20 eligible rows and launched >= eligible")
        n_invalid = n_launched - n_eligible
        probabilities = np.full(n_eligible + 1, 1.0 / n_launched, dtype=np.float64)
        probabilities[-1] = n_invalid / n_launched
        rng = np.random.default_rng(streams[gi])
        group_points[gi] = np.mean(values, axis=0)
        observed_counts.append(n_eligible)
        strata.append({"group_id": int(group), "launched": n_launched,
                       "eligible": n_eligible, "ineligible": n_invalid})
        for offset in range(0, b, batch_size):
            check_time()
            stop = min(offset + batch_size, b)
            draw = rng.multinomial(n_launched, probabilities, size=stop - offset)
            weights = draw[:, :-1]
            eligible_count = n_launched - draw[:, -1]
            if np.any(eligible_count == 0):
                raise ValueError("a bootstrap stratum has zero eligible events; group CI undefined")
            numerator = weights @ values
            counts[offset:stop, gi] = eligible_count
            global_numerator[offset:stop] += numerator
            group_replicates[offset:stop, gi] = numerator / eligible_count[:, None]
        check_time()
    global_primary = global_numerator / counts.sum(axis=1, keepdims=True)
    primary_point = np.mean(gains, axis=0)
    effect_point = matrix @ primary_point
    effect_replicates = global_primary @ matrix.T
    global_bands = _bands(effect_point, effect_replicates, confidence)
    joint_group_bands = _bands(group_points, group_replicates, confidence)
    per_primary_bands = [_bands(group_points[:, j], group_replicates[:, :, j], confidence)
                         for j in range(k)]
    effects = []
    for j, name in enumerate(contrast_names):
        item = {"name": name, "weights": matrix[j].tolist(), "estimate": float(effect_point[j])}
        for field in ("bootstrap_standard_error", "pointwise_percentile_lower",
                      "pointwise_percentile_upper", "simultaneous_lower", "simultaneous_upper"):
            item[field] = float(global_bands[field][j])
        if np.count_nonzero(matrix[j]) == 1 and np.sum(matrix[j]) == 1:
            item["practical_effect"] = practical_effect(
                item["estimate"], [item["simultaneous_lower"], item["simultaneous_upper"]])
        else:
            item["interpretation"] = (
                "Difference of representation gains; exponentiating gives a ratio of "
                "geometric mean density ratios, not a single-model accuracy improvement."
            )
        effects.append(item)
    grouped = []
    for gi, group in enumerate(groups):
        for j, name in enumerate(labels):
            separate = per_primary_bands[j]
            grouped.append({
                "group_id": int(group), "primary": str(name),
                "eligible_count": observed_counts[gi], "estimate": float(group_points[gi, j]),
                "bootstrap_standard_error": float(separate["bootstrap_standard_error"][gi]),
                "pointwise_percentile_lower": float(separate["pointwise_percentile_lower"][gi]),
                "pointwise_percentile_upper": float(separate["pointwise_percentile_upper"][gi]),
                "within_primary_simultaneous_lower": float(separate["simultaneous_lower"][gi]),
                "within_primary_simultaneous_upper": float(separate["simultaneous_upper"][gi]),
                "all_primary_groups_simultaneous_lower": float(joint_group_bands["simultaneous_lower"][gi, j]),
                "all_primary_groups_simultaneous_upper": float(joint_group_bands["simultaneous_upper"][gi, j]),
            })
    check_time()
    summary = {
        "schema": "information-paired-event-uncertainty/v1", "status": "COMPLETE",
        "estimator": "mean over fixed paired-seed log scores, conditional on S1",
        "resampling": "exact multinomial within fixed launched strata; shared eligible weights and merged S0 atom",
        "denominator": "resampled eligible counts; mixture weights can vary across strata",
        "confidence": confidence, "n_resamples": b, "seed": seed, "batch_size": batch_size,
        "primary_labels": list(labels), "strata": strata,
        "global_family_size": len(effects),
        "global_family_max_abs_t_critical": global_bands["max_abs_t_critical"],
        "group_family_size_per_primary": g,
        "joint_primary_group_family_size": g * k,
        "joint_primary_group_max_abs_t_critical": joint_group_bands["max_abs_t_critical"],
        "per_primary_group_max_abs_t_critical": {
            str(labels[j]): per_primary_bands[j]["max_abs_t_critical"] for j in range(k)},
        "interval_method": "pointwise percentile; simultaneous centered max-|t| with bootstrap-SD denominator",
        "effects": effects, "primary_groups": grouped,
        "event_uncertainty_scope": (
            "Conditional on fitted models, fixed launched design and simulation settings. "
            "Includes eligible-rate and eligible-mixture variation within that design. "
            "Excludes retraining, optimizer-seed sampling, generator-setting uncertainty, "
            "and unmodeled dependence within a random-number stream."
        ),
        "multiplicity_scope": (
            "Global family covers every supplied planned contrast. Group bands cover one "
            "primary's strata and separately all primary-by-stratum effects. These are "
            "separate families, not joint coverage across catalogs or diagnostics."
        ),
        "elapsed_seconds": time.monotonic() - start,
    }
    return PairedBootstrapResult(summary, effect_replicates, group_replicates, counts)
