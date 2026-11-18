"""Support-aware one-dimensional conditional density primitives.

The upper support is the incident momentum ``p_in`` in the original target
coordinate. Every component is normalized before mixture weights are applied.
This module is analysis code only; it never sees process labels or synthetic
physics output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import t as student_t


NEG_INF = float("-inf")


def _arrays(y, mu, sigma, upper):
    y, mu, sigma, upper = np.broadcast_arrays(
        np.asarray(y, dtype=float), np.asarray(mu, dtype=float),
        np.asarray(sigma, dtype=float), np.asarray(upper, dtype=float)
    )
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("mu must be finite and sigma must be positive")
    if np.any(~np.isfinite(upper)):
        raise ValueError("upper support must be finite")
    return y, mu, sigma, upper


def truncated_normal_logpdf(y, mu, sigma, upper):
    y, mu, sigma, upper = _arrays(y, mu, sigma, upper)
    z = (y - mu) / sigma
    upper_z = (upper - mu) / sigma
    values = -0.5 * z * z - np.log(sigma) - 0.5 * math.log(2 * math.pi) - log_ndtr(upper_z)
    return np.where(y < upper, values, NEG_INF)


def truncated_normal_cdf(y, mu, sigma, upper):
    y, mu, sigma, upper = _arrays(y, mu, sigma, upper)
    upper_cdf = ndtr((upper - mu) / sigma)
    values = ndtr((y - mu) / sigma) / upper_cdf
    return np.where(y < upper, np.clip(values, 0.0, 1.0), 1.0)


def truncated_normal_quantile(q, mu, sigma, upper):
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("q must be in [0, 1]")
    mu, sigma, upper = np.broadcast_arrays(np.asarray(mu, float), np.asarray(sigma, float), np.asarray(upper, float))
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    z_upper = (upper - mu) / sigma
    result = mu + sigma * ndtri(q * ndtr(z_upper))
    return np.where(q == 1, upper, result)


def truncated_student_logpdf(y, mu, sigma, upper, df=3.0):
    y, mu, sigma, upper = _arrays(y, mu, sigma, upper)
    if df <= 0:
        raise ValueError("df must be positive")
    z = (y - mu) / sigma
    norm = student_t.cdf((upper - mu) / sigma, df)
    values = student_t.logpdf(z, df) - np.log(sigma) - np.log(norm)
    return np.where(y < upper, values, NEG_INF)


def truncated_student_cdf(y, mu, sigma, upper, df=3.0):
    y, mu, sigma, upper = _arrays(y, mu, sigma, upper)
    if df <= 0:
        raise ValueError("df must be positive")
    norm = student_t.cdf((upper - mu) / sigma, df)
    values = student_t.cdf((y - mu) / sigma, df) / norm
    return np.where(y < upper, np.clip(values, 0.0, 1.0), 1.0)


def log_mixture_density(log_weights, component_logpdf):
    """Log density for component-first arrays, with normalized weights."""
    weights = np.asarray(log_weights, dtype=float)
    values = np.asarray(component_logpdf, dtype=float)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("component axis mismatch")
    normalized = weights - logsumexp(weights, axis=0, keepdims=True)
    return logsumexp(normalized + values, axis=0)


def mixture_cdf(weights, component_cdf):
    weights = np.asarray(weights, dtype=float)
    cdfs = np.asarray(component_cdf, dtype=float)
    if cdfs.shape[0] != weights.shape[0]:
        raise ValueError("component axis mismatch")
    if np.any(weights < 0) or np.any(~np.isfinite(weights)):
        raise ValueError("weights must be finite and nonnegative")
    if weights.ndim == 1:
        weights = weights[:, None]
    normalized = weights / np.sum(weights, axis=0, keepdims=True)
    return np.sum(normalized * cdfs, axis=0)


def mixture_quantile(q, weights: Sequence[float], component_cdf_functions: Sequence, lower: float, upper: float) -> float:
    if not 0 <= q <= 1 or not lower < upper:
        raise ValueError("invalid quantile or bracket")
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    if q == 0:
        return lower
    if q == 1:
        return upper
    return brentq(lambda value: sum(w * float(fn(value)) for w, fn in zip(weights, component_cdf_functions)) - q, lower, upper)


@dataclass(frozen=True)
class TrainStandardizer:
    mean: float
    scale: float

    def __post_init__(self):
        if not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("standardizer needs finite mean and positive scale")

    def forward(self, y):
        return (np.asarray(y, dtype=float) - self.mean) / self.scale

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.scale + self.mean

    def original_logpdf(self, standardized_logpdf):
        return np.asarray(standardized_logpdf, dtype=float) - math.log(self.scale)


def gain_decomposition(log_full, log_anchor_joint, log_anchor_ind, log_reference, log_weight0, log_responsibility0):
    log_full = np.asarray(log_full, dtype=float)
    log_anchor_joint = np.asarray(log_anchor_joint, dtype=float)
    log_anchor_ind = np.asarray(log_anchor_ind, dtype=float)
    log_reference = np.asarray(log_reference, dtype=float)
    total = log_full - log_reference
    anchor_ind = log_anchor_ind - log_reference
    reorganization = log_anchor_joint - log_anchor_ind
    anchor_joint = log_anchor_joint - log_reference
    direct = log_full - log_anchor_joint
    direct_via_responsibility = np.asarray(log_weight0) - np.asarray(log_responsibility0)
    if not np.allclose(direct, direct_via_responsibility, rtol=1e-10, atol=1e-10):
        raise ValueError("direct gain paths do not close")
    if not np.allclose(total, anchor_ind + reorganization + direct, rtol=1e-10, atol=1e-10):
        raise ValueError("three-term gain decomposition does not close")
    return {"total": total, "anchor_ind": anchor_ind, "anchor_reorganization": reorganization, "anchor_joint": anchor_joint, "direct": direct}
