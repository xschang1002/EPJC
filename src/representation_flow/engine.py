"""Production engine for the first Raw/Simple/Physics/Learned flow prototype.

The target is the observed GeV/c momentum loss for S=1 events.  A model is a
conditional, two-layer rational-quadratic spline mapping from standardized
target space to a standard logistic base.  The original-space likelihood is
normalized on the physical support ``y < p_in`` for every representation.

This module deliberately has no dependency on the archived mixture models.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data_contract.event_split import split_label
from models.anchor import (
    CENTER_IDENTITIES,
    corrected_pion_bethe_bloch_delta_p,
    historical_pion_bethe_bloch_delta_p,
)


TORCH_DTYPE = torch.float32
NP_DTYPE = np.float64
SPLINE_BOUNDS = 8.0
NUM_BINS = 16
MIN_BIN_WIDTH = 1e-3
MIN_BIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logdiffexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable log(exp(a)-exp(b)) for a >= b."""
    gap = torch.clamp(b - a, max=0.0)
    return a + torch.log(-torch.expm1(gap))


def logistic_logpdf(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.logsigmoid(value) + torch.nn.functional.logsigmoid(-value)


def logistic_logcdf(value: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.softplus(-value)


def logistic_logsf(value: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.softplus(value)


def _gather_last(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, -1, index.unsqueeze(-1)).squeeze(-1)


def _spline_parameters(raw: torch.Tensor, bound: float = SPLINE_BOUNDS):
    widths = raw[..., :NUM_BINS]
    heights = raw[..., NUM_BINS:2 * NUM_BINS]
    derivatives = raw[..., 2 * NUM_BINS:]
    widths = torch.softmax(widths, dim=-1)
    heights = torch.softmax(heights, dim=-1)
    widths = MIN_BIN_WIDTH + (2 * bound - NUM_BINS * MIN_BIN_WIDTH) * widths
    heights = MIN_BIN_HEIGHT + (2 * bound - NUM_BINS * MIN_BIN_HEIGHT) * heights
    derivatives = MIN_DERIVATIVE + torch.nn.functional.softplus(derivatives)
    cumwidths = torch.cat(
        [torch.full_like(widths[..., :1], -bound), -bound + torch.cumsum(widths, dim=-1)], dim=-1
    )
    cumheights = torch.cat(
        [torch.full_like(heights[..., :1], -bound), -bound + torch.cumsum(heights, dim=-1)], dim=-1
    )
    return widths, heights, derivatives, cumwidths, cumheights


def _rqs_forward(value: torch.Tensor, raw: torch.Tensor, bound: float = SPLINE_BOUNDS):
    widths, heights, derivatives, cumwidths, cumheights = _spline_parameters(raw, bound)
    inside = (value >= -bound) & (value <= bound)
    safe_value = torch.clamp(value, -bound, bound)
    idx = torch.sum(safe_value.unsqueeze(-1) >= cumwidths[..., 1:-1], dim=-1).clamp(max=NUM_BINS - 1)
    input_cumwidth = _gather_last(cumwidths, idx)
    input_bin_width = _gather_last(widths, idx)
    input_cumheight = _gather_last(cumheights, idx)
    input_bin_height = _gather_last(heights, idx)
    input_derivative = _gather_last(derivatives, idx)
    input_derivative_plus_one = _gather_last(derivatives, idx + 1)
    delta = input_bin_height / input_bin_width
    theta = (safe_value - input_cumwidth) / input_bin_width
    theta_one_minus = theta * (1 - theta)
    numerator = input_bin_height * (delta * theta.square() + input_derivative * theta_one_minus)
    denominator = delta + (input_derivative_plus_one + input_derivative - 2 * delta) * theta_one_minus
    output = input_cumheight + numerator / denominator
    derivative = (delta.square() * (
        input_derivative_plus_one * theta.square()
        + 2 * delta * theta_one_minus
        + input_derivative * (1 - theta).square()
    ) / denominator.square())
    output = torch.where(inside, output, value)
    logabsdet = torch.where(inside, torch.log(torch.clamp(derivative, min=1e-12)), torch.zeros_like(value))
    return output, logabsdet


def _rqs_inverse(value: torch.Tensor, raw: torch.Tensor, bound: float = SPLINE_BOUNDS):
    widths, heights, derivatives, cumwidths, cumheights = _spline_parameters(raw, bound)
    inside = (value >= -bound) & (value <= bound)
    safe_value = torch.clamp(value, -bound, bound)
    idx = torch.sum(safe_value.unsqueeze(-1) >= cumheights[..., 1:-1], dim=-1).clamp(max=NUM_BINS - 1)
    input_cumwidth = _gather_last(cumwidths, idx)
    input_bin_width = _gather_last(widths, idx)
    input_cumheight = _gather_last(cumheights, idx)
    input_bin_height = _gather_last(heights, idx)
    input_derivative = _gather_last(derivatives, idx)
    input_derivative_plus_one = _gather_last(derivatives, idx + 1)
    delta = input_bin_height / input_bin_width
    target = safe_value - input_cumheight
    a = target * (input_derivative + input_derivative_plus_one - 2 * delta) + input_bin_height * (delta - input_derivative)
    b = input_bin_height * input_derivative - target * (input_derivative + input_derivative_plus_one - 2 * delta)
    c = -delta * target
    discriminant = torch.clamp(b.square() - 4 * a * c, min=0.0)
    sqrt_disc = torch.sqrt(discriminant)
    root_quad = (2 * c) / (-b - sqrt_disc + 1e-12)
    root_linear = -c / (b + 1e-12)
    theta = torch.where(torch.abs(a) > 1e-8, root_quad, root_linear).clamp(0.0, 1.0)
    theta_one_minus = theta * (1 - theta)
    denominator = delta + (input_derivative_plus_one + input_derivative - 2 * delta) * theta_one_minus
    derivative = delta.square() * (
        input_derivative_plus_one * theta.square()
        + 2 * delta * theta_one_minus
        + input_derivative * (1 - theta).square()
    ) / denominator.square()
    output = input_cumwidth + theta * input_bin_width
    output = torch.where(inside, output, value)
    logabsdet = torch.where(inside, -torch.log(torch.clamp(derivative, min=1e-12)), torch.zeros_like(value))
    return output, logabsdet


class ConditionalSplineLayer(torch.nn.Module):
    def __init__(self, context_dim: int, hidden: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        output_dim = 2 * NUM_BINS + NUM_BINS + 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, output_dim),
        )
        final = self.net[-1]
        torch.nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        torch.nn.init.constant_(final.bias[: 2 * NUM_BINS], 0.0)
        torch.nn.init.constant_(final.bias[2 * NUM_BINS:], float(math.log(math.expm1(1.0 - MIN_DERIVATIVE))))

    def raw(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class ConditionalSplineFlow(torch.nn.Module):
    """Two-layer conditional RQS flow with an exact identity tail."""

    def __init__(self, context_dim: int = 5, hidden: int = 32, seed: int = 11, bound: float = SPLINE_BOUNDS):
        super().__init__()
        self.context_dim = context_dim
        self.hidden = hidden
        self.seed = seed
        self.bound = bound
        self.layers = torch.nn.ModuleList([
            ConditionalSplineLayer(context_dim, hidden, seed),
            ConditionalSplineLayer(context_dim, hidden, seed + 1),
        ])

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def transform_to_base(self, z: torch.Tensor, context: torch.Tensor):
        value = z
        logdet = torch.zeros_like(z)
        for layer in self.layers:
            value, layer_logdet = _rqs_forward(value, layer.raw(context), self.bound)
            logdet = logdet + layer_logdet
        return value, logdet

    def transform_from_base(self, value: torch.Tensor, context: torch.Tensor):
        result = value
        logdet = torch.zeros_like(value)
        for layer in reversed(self.layers):
            result, layer_logdet = _rqs_inverse(result, layer.raw(context), self.bound)
            logdet = logdet + layer_logdet
        return result, logdet

    def log_q(self, z: torch.Tensor, context: torch.Tensor):
        base, logdet = self.transform_to_base(z, context)
        return logistic_logpdf(base) + logdet

    def log_cdf(self, z: torch.Tensor, context: torch.Tensor):
        base, _ = self.transform_to_base(z, context)
        return logistic_logcdf(base)

    def log_sf(self, z: torch.Tensor, context: torch.Tensor):
        base, _ = self.transform_to_base(z, context)
        return logistic_logsf(base)

    def original_terms(self, y, center, mu, scale, p_in, context):
        if not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, dtype=y.dtype, device=y.device)
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("likelihood scale must be finite and positive")
        if torch.isnan(y).any():
            raise ValueError("likelihood target cannot be NaN")
        z = (y - center - mu) / scale
        upper = (p_in - center - mu) / scale
        base_z, logdet = self.transform_to_base(z, context)
        base_upper, _ = self.transform_to_base(upper, context)
        log_q_z = logistic_logpdf(base_z) + logdet
        log_f_b = logistic_logcdf(base_upper)
        log_p_y = log_q_z - torch.log(scale) - log_f_b
        log_cdf_y = torch.minimum(logistic_logcdf(base_z) - log_f_b, torch.zeros_like(z))
        # [F(B)-F(Z)] / F(B) = sigmoid(-Z) * [1-exp(Z-B)]. This
        # avoids subtracting two nearly-one CDFs or clipping a true zero SF.
        log_sf_y = logistic_logsf(base_z) + torch.log(-torch.expm1(torch.clamp(base_z - base_upper, max=0.0)))
        inside_support = y < p_in
        log_p_y = torch.where(inside_support, log_p_y, torch.full_like(z, -torch.inf))
        log_cdf_y = torch.where(inside_support, log_cdf_y, torch.zeros_like(z))
        log_sf_y = torch.where(inside_support, log_sf_y, torch.full_like(z, -torch.inf))
        return {
            "z": z,
            "b": upper,
            "log_q_z": log_q_z,
            "log_f_b": log_f_b,
            "log_p_y": log_p_y,
            "log_cdf_y": log_cdf_y,
            "log_sf_y": log_sf_y,
        }


def _polynomial_basis(p_in: np.ndarray, thickness: np.ndarray, charge: np.ndarray) -> np.ndarray:
    u = (p_in - 0.65) / 0.35
    v = (thickness - 0.3) / 0.7
    monomials = []
    for total in range(4):
        for i in range(total + 1):
            j = total - i
            monomials.append((u ** i) * (v ** j))
    base = np.column_stack(monomials)
    return np.column_stack([base, charge[:, None] * base])


def common_features(p_in: np.ndarray, thickness: np.ndarray, charge: np.ndarray) -> np.ndarray:
    u = (p_in - 0.65) / 0.35
    v = (thickness - 0.3) / 0.7
    return np.column_stack([u, v, charge, np.log(p_in / 0.65), np.zeros_like(u)]).astype(NP_DTYPE)


def bottleneck_features(p_in: np.ndarray, thickness: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Exploratory context bottleneck: retain only thickness and charge."""
    _ = p_in
    v = (np.asarray(thickness, dtype=NP_DTYPE) - 0.3) / 0.7
    q = np.asarray(charge, dtype=NP_DTYPE)
    return np.column_stack([v, q]).astype(NP_DTYPE)


def context_features(p_in: np.ndarray, thickness: np.ndarray, charge: np.ndarray, mode: str = "full") -> np.ndarray:
    if mode == "full":
        return common_features(p_in, thickness, charge)
    if mode == "thickness_charge":
        return bottleneck_features(p_in, thickness, charge)
    raise ValueError(f"unknown context feature mode: {mode}")


@dataclass
class RepresentationPreprocessor:
    name: str
    coefficients: np.ndarray
    mu: float
    scale: float
    alpha: float | None = None
    scale_coefficients: np.ndarray | None = None
    scale_log_min: float | None = None
    scale_log_max: float | None = None

    @property
    def center_identity(self) -> dict:
        if self.name.startswith("CorrectedPhysics"):
            return dict(CENTER_IDENTITIES["CorrectedPhysics"])
        if self.name.startswith("Physics") or self.name == "HistoricalIncorrectPhysics":
            return dict(CENTER_IDENTITIES["HistoricalIncorrectPhysics"])
        return {"center_id": self.name, "formula_version": "train-only-center-v1"}

    def to_dict(self) -> dict:
        return {
            "schema": "representation-preprocessor/v2", "name": self.name,
            "center_identity": self.center_identity,
            "coefficients": np.asarray(self.coefficients, dtype=float).tolist(),
            "mu": float(self.mu), "scale": float(self.scale),
            "alpha": None if self.alpha is None else float(self.alpha),
            "scale_coefficients": None if self.scale_coefficients is None else np.asarray(self.scale_coefficients, dtype=float).tolist(),
            "scale_log_min": None if self.scale_log_min is None else float(self.scale_log_min),
            "scale_log_max": None if self.scale_log_max is None else float(self.scale_log_max),
        }

    @classmethod
    def from_dict(cls, value: dict):
        required = {"schema", "name", "center_identity", "coefficients", "mu", "scale", "alpha", "scale_coefficients", "scale_log_min", "scale_log_max"}
        if set(value) != required or value["schema"] != "representation-preprocessor/v2":
            raise ValueError("unsupported or incomplete preprocessor contract")
        expected_counts = {"Raw": 1, "RawScaled": 1, "Simple": 4, "Learned": 20,
                           "Physics": 1, "PhysicsScaled": 1, "PhysicsCalibrated": 1,
                           "HistoricalIncorrectPhysics": 1, "CorrectedPhysics": 1,
                           "CorrectedPhysicsScaled": 1, "CorrectedPhysicsCalibrated": 1}
        if value["name"] not in expected_counts:
            raise ValueError("unknown representation name")
        coef = np.asarray(value["coefficients"], dtype=NP_DTYPE)
        if coef.shape != (expected_counts[value["name"]],) or not np.isfinite(coef).all():
            raise ValueError("invalid center coefficients")
        optional = [value[key] for key in ("alpha", "scale_log_min", "scale_log_max") if value[key] is not None]
        if not all(math.isfinite(float(v)) for v in [value["mu"], value["scale"], *optional]) or value["scale"] <= 0:
            raise ValueError("preprocessor values must be finite and scale positive")
        scale_coef = value["scale_coefficients"]
        if scale_coef is not None:
            scale_coef = np.asarray(scale_coef, dtype=NP_DTYPE)
            if scale_coef.shape not in ((4,), (5,)) or not np.isfinite(scale_coef).all():
                raise ValueError("invalid scale coefficients")
        if value["scale_log_min"] is not None and value["scale_log_max"] is not None and value["scale_log_min"] > value["scale_log_max"]:
            raise ValueError("inverted scale clipping bounds")
        pre = cls(value["name"], coef, float(value["mu"]), float(value["scale"]), value["alpha"], scale_coef, value["scale_log_min"], value["scale_log_max"])
        if pre.center_identity != value["center_identity"]:
            raise ValueError("center identity does not match representation")
        return pre

    def center(self, p_in, thickness, charge):
        p_in = np.asarray(p_in, dtype=NP_DTYPE)
        thickness = np.asarray(thickness, dtype=NP_DTYPE)
        charge = np.asarray(charge, dtype=NP_DTYPE)
        if self.name in ("Raw", "RawScaled"):
            return np.zeros_like(p_in)
        if self.name in ("Physics", "PhysicsScaled", "HistoricalIncorrectPhysics"):
            return historical_pion_bethe_bloch_delta_p(p_in, thickness, charge)
        if self.name == "PhysicsCalibrated":
            return float(self.coefficients[0]) * historical_pion_bethe_bloch_delta_p(p_in, thickness, charge)
        if self.name in ("CorrectedPhysics", "CorrectedPhysicsScaled"):
            return corrected_pion_bethe_bloch_delta_p(p_in, thickness, charge)
        if self.name == "CorrectedPhysicsCalibrated":
            return float(self.coefficients[0]) * corrected_pion_bethe_bloch_delta_p(p_in, thickness, charge)
        if self.name == "Simple":
            return np.column_stack([np.ones_like(p_in), (p_in - 0.65) / 0.35, (thickness - 0.3) / 0.7, charge]) @ self.coefficients
        if self.name == "Learned":
            return _polynomial_basis(p_in, thickness, charge) @ self.coefficients
        raise KeyError(self.name)

    def transform(self, y, p_in, thickness, charge):
        c = self.center(p_in, thickness, charge)
        return (np.asarray(y, dtype=NP_DTYPE) - c - self.mu) / self.scale_for(p_in, thickness, charge)

    def scale_for(self, p_in, thickness, charge):
        p_in = np.asarray(p_in, dtype=NP_DTYPE)
        if self.scale_coefficients is None:
            return np.full_like(p_in, self.scale, dtype=NP_DTYPE)
        thickness = np.asarray(thickness, dtype=NP_DTYPE)
        charge = np.asarray(charge, dtype=NP_DTYPE)
        if len(self.scale_coefficients) == 4:
            features = np.column_stack([np.ones_like(p_in), (p_in - 0.65) / 0.35, (thickness - 0.3) / 0.7, charge])
        else:
            features = common_features(p_in, thickness, charge)
        log_scale = features @ np.asarray(self.scale_coefficients, dtype=NP_DTYPE)
        if self.scale_log_min is not None or self.scale_log_max is not None:
            log_scale = np.clip(log_scale, self.scale_log_min, self.scale_log_max)
        return np.exp(log_scale).astype(NP_DTYPE)


def _fit_learned_ridge(y, p_in, thickness, charge) -> tuple[np.ndarray, float]:
    design = _polynomial_basis(p_in, thickness, charge)
    folds = np.arange(len(y), dtype=np.int64) % 3
    alphas = np.asarray([1e-6, 1e-4, 1e-2, 1.0, 100.0], dtype=NP_DTYPE)
    scores = []
    for alpha in alphas:
        fold_scores = []
        for fold in range(3):
            train = folds != fold
            valid = ~train
            penalty = np.eye(design.shape[1], dtype=NP_DTYPE) * alpha
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(design[train].T @ design[train] + penalty, design[train].T @ (y[train]))
            fold_scores.append(float(np.mean((y[valid] - design[valid] @ beta) ** 2)))
        scores.append(float(np.mean(fold_scores)))
    alpha = float(alphas[int(np.argmin(scores))])
    penalty = np.eye(design.shape[1], dtype=NP_DTYPE) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta, alpha


def fit_preprocessors(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    include_physics_calibrated: bool = False,
    *,
    physics_name: str = "Physics",
) -> dict[str, RepresentationPreprocessor]:
    p = arrays["p_in"][train_indices]
    t = arrays["thickness"][train_indices]
    q = arrays["charge"][train_indices]
    y = arrays["y"][train_indices]
    feature = np.column_stack([np.ones_like(p), (p - 0.65) / 0.35, (t - 0.3) / 0.7, q])
    simple_beta = np.linalg.lstsq(feature, y, rcond=None)[0]
    learned_beta, alpha = _fit_learned_ridge(y, p, t, q)
    if physics_name not in ("Physics", "HistoricalIncorrectPhysics", "CorrectedPhysics"):
        raise ValueError("physics_name must explicitly identify a historical or corrected center")
    physics_function = corrected_pion_bethe_bloch_delta_p if physics_name == "CorrectedPhysics" else historical_pion_bethe_bloch_delta_p
    physics_center = physics_function(p, t, q)
    candidates = {
        "Raw": (np.zeros_like(p), np.zeros(1), None),
        "Simple": (feature @ simple_beta, simple_beta, None),
        physics_name: (physics_center, np.zeros(1), None),
        "Learned": (_polynomial_basis(p, t, q) @ learned_beta, learned_beta, alpha),
    }
    if include_physics_calibrated:
        denominator = float(np.dot(physics_center, physics_center))
        physics_alpha = float(np.dot(physics_center, y) / denominator) if denominator > 0 else 1.0
        candidates["CorrectedPhysicsCalibrated" if physics_name == "CorrectedPhysics" else "PhysicsCalibrated"] = (
            physics_alpha * physics_center,
            np.asarray([physics_alpha], dtype=NP_DTYPE),
            None,
        )
    result = {}
    shared_scale = float(np.std(y, ddof=0))
    if not math.isfinite(shared_scale) or shared_scale <= 0:
        raise ValueError("shared target scale is not positive")
    for name, (center, coefficients, selected_alpha) in candidates.items():
        result[name] = RepresentationPreprocessor(name, np.asarray(coefficients), float(np.mean(y - center)), shared_scale, selected_alpha)
    return result


def load_source_arrays(merged_dir: Path, validation_fraction: float = 0.2, diagnostic_fraction: float = 0.0, *, eligibility: str = "forward"):
    """Read only the public target schema and return S=1 arrays plus provenance."""
    keys = {"event_uid": [], "p_in": [], "thickness": [], "charge": [], "y": [], "split": []}
    if eligibility not in ("forward", "legacy_s1"):
        raise ValueError("unknown eligibility contract")
    seen = set()
    path = merged_dir / "events.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("S1"):
                continue
            uid = row["event_uid"]
            p, thickness, y, pdg = (float(row[key]) for key in ("p_in_GeV_c", "thickness_mm", "delta_p_GeV_c", "pdg"))
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("empty, invalid or duplicate eligible event UID")
            if eligibility == "forward" and row.get("exit_face") != "forward":
                raise ValueError(f"S1 event {uid} violates the forward-exit contract")
            if not all(math.isfinite(v) for v in (p, thickness, y, pdg)) or p <= 0 or thickness <= 0 or pdg not in (-211, 211) or y >= p:
                raise ValueError(f"S1 event {uid} violates finite-input or Y < p_in contract")
            if row.get("aborted", False):
                raise ValueError(f"S1 event {uid} was technically aborted")
            seen.add(uid)
            keys["event_uid"].append(row["event_uid"])
            keys["p_in"].append(float(row["p_in_GeV_c"]))
            keys["thickness"].append(float(row["thickness_mm"]))
            keys["charge"].append(float(row["pdg"]) / 211.0)
            keys["y"].append(float(row["delta_p_GeV_c"]))
            keys["split"].append(split_label(row["event_uid"], validation_fraction, diagnostic_fraction))
    for key in ("p_in", "thickness", "charge", "y"):
        keys[key] = np.asarray(keys[key], dtype=NP_DTYPE)
    keys["event_uid"] = np.asarray(keys["event_uid"], dtype=object)
    keys["split"] = np.asarray(keys["split"], dtype=object)
    manifest = json.loads((merged_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {**manifest, "loader_contract": {"eligibility": eligibility, "target_support": "Y < p_in", "eligible_events": len(seen)}}
    return keys, manifest


def subset_indices(arrays: dict[str, np.ndarray], fraction: float = 0.1, catalog_id: int = 4101):
    train = np.flatnonzero(arrays["split"] == "train")
    ranked = sorted(train.tolist(), key=lambda i: hashlib.sha256(f"{catalog_id}:{arrays['event_uid'][i]}".encode()).hexdigest())
    count = int(round(fraction * len(train)))
    return np.asarray(ranked[:count], dtype=np.int64)


def _identity_context(n: int, dtype=TORCH_DTYPE):
    return torch.zeros((n, 5), dtype=dtype)


def run_a0_checks() -> dict:
    torch.manual_seed(123)
    context = torch.zeros((64, 5), dtype=TORCH_DTYPE)
    flow = ConditionalSplineFlow(hidden=32, seed=123)
    values = torch.linspace(-7.7, 7.7, 64, dtype=TORCH_DTYPE)
    base, logdet = flow.transform_to_base(values, context)
    restored, inverse_logdet = flow.transform_from_base(base, context)
    inversion_error = float(torch.max(torch.abs(restored - values)).detach())
    # The identity initialization makes the transport integral a closed logistic check.
    xs = np.linspace(-7.5, 7.5, 401)
    with torch.no_grad():
        xt = torch.tensor(xs, dtype=TORCH_DTYPE)
        lp = flow.log_q(xt, _identity_context(len(xs))).numpy()
    integral = float(np.trapezoid(np.exp(lp), xs))
    upper = torch.full((64,), 1.2, dtype=TORCH_DTYPE)
    y = torch.linspace(-1.5, 0.8, 64, dtype=TORCH_DTYPE)
    center = torch.zeros_like(y)
    result = flow.original_terms(y, center, 0.0, 0.1, upper, context)
    cdf = torch.exp(result["log_cdf_y"]).detach().numpy()
    sf = torch.exp(result["log_sf_y"]).detach().numpy()
    monotone_error = float(np.max(np.maximum(0.0, np.diff(cdf) * -1.0)))
    cdf_sf_error = float(np.max(np.abs(cdf + sf - 1.0)))
    # Compare the exact original-unit support-normalized density under GeV -> MeV.
    y_np = y.numpy()
    with torch.no_grad():
        geV = flow.original_terms(y, center, 0.0, 0.1, upper, context)["log_p_y"].numpy()
        MeV = flow.original_terms(y * 1000, center * 1000, 0.0, 100.0, upper * 1000, context)["log_p_y"].numpy()
    unit_error = float(np.max(np.abs(MeV - (geV - math.log(1000.0)))))
    # A numerical integral of one truncated density at a representative point.
    m, s, b = 0.0, 0.1, 1.2 / 0.1
    z_upper = float(flow.log_cdf(torch.tensor([b]), _identity_context(1))[0].detach())
    # Fixed Gauss-Legendre quadrature avoids adaptive subdivision warnings at
    # spline knots. The omitted logistic tail below -40 is below 4e-18.
    nodes, weights = np.polynomial.legendre.leggauss(2048)
    grid = 0.5 * (b + 40.0) * nodes + 0.5 * (b - 40.0)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=TORCH_DTYPE)
        grid_logpdf = flow.log_q(grid_tensor, _identity_context(len(grid))).numpy()
    integral_truncated = float(0.5 * (b + 40.0) * np.sum(weights * np.exp(grid_logpdf - z_upper)))
    status = "A0_PRODUCTION_ENGINE_PASS" if (
        inversion_error < 2e-4 and abs(integral - 1.0) < 2e-3
        and abs(integral_truncated - 1.0) < 2e-5 and unit_error < 2e-5
        and cdf_sf_error < 2e-5 and monotone_error == 0.0
    ) else "A0_PRODUCTION_ENGINE_FAIL"
    return {
        "status": status,
        "engine": "ConditionalSplineFlow",
        "flow": {"layers": 2, "bins": NUM_BINS, "bound": SPLINE_BOUNDS, "tail": "identity", "base": "standard_logistic"},
        "parameter_count_small": flow.parameter_count(),
        "inversion_max_abs_error": inversion_error,
        "untruncated_logistic_integral_on_reference_grid": integral,
        "truncated_density_integral": integral_truncated,
        "cdf_monotonicity_error": monotone_error,
        "cdf_plus_sf_max_error": cdf_sf_error,
        "gev_to_mev_logpdf_max_error": unit_error,
        "finite_and_gradient_ready": bool(torch.isfinite(result["log_p_y"]).all()),
        "tolerances": {"inversion": 2e-4, "integral": 2e-3, "unit": 2e-5, "cdf_sf": 2e-5},
    }

