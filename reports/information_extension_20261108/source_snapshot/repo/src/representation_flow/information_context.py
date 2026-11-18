"""Explicit five-column information interventions; no frozen engine mutation."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

CONTEXTS = ("Hidden5", "Coarse2", "Coarse8", "Full5")
EDGES = {"Coarse2": [.3, .65, 1.],
         "Coarse8": [.3, .3875, .475, .5625, .65, .7375, .825, .9125, 1.]}


def context_spec(name: str) -> dict:
    if name not in CONTEXTS:
        raise ValueError("unknown information context")
    return {"schema": "information-context/v1", "name": name, "dimension": 5,
            "momentum_range_GeV_c": [.3, 1.], "edges_GeV_c": deepcopy(EDGES.get(name)),
            "momentum_encoding": "exact" if name == "Full5" else "zero" if name == "Hidden5" else "fixed_bin_midpoint",
            "columns": (["zero", "(thickness_mm-.3)/.7", "charge", "zero", "zero"] if name == "Hidden5" else
                        ["(p_encoded-.65)/.35", "(thickness_mm-.3)/.7", "charge", "log(p_encoded/.65)", "zero"]),
            "hidden_momentum_columns": [0, 3] if name == "Hidden5" else [],
            "boundary": "left-closed/right-open except final closed; exact p remains in center and support"}


def validate_context_spec(spec: dict) -> dict:
    if spec != context_spec(spec.get("name")):
        raise ValueError("information context does not match the fixed physical design")
    return spec


def information_features(p_in, thickness, charge, spec):
    spec = validate_context_spec(spec)
    p, t, q = np.broadcast_arrays(np.asarray(p_in, dtype=np.float64), np.asarray(thickness, dtype=np.float64), np.asarray(charge, dtype=np.float64))
    if p.ndim != 1 or not all(np.isfinite(v).all() for v in (p, t, q)):
        raise ValueError("information features require finite one-dimensional arrays")
    if np.any((p < .3) | (p > 1.)) or not np.isin(t, [.1, .3, 1.]).all() or not np.isin(q, [-1., 1.]).all():
        raise ValueError("context outside the declared same-range physical design")
    v = (t - .3) / .7
    if spec["name"] == "Hidden5":
        u, logp = np.zeros_like(p), np.zeros_like(p)
    else:
        encoded = p
        if spec["edges_GeV_c"] is not None:
            edges = np.asarray(spec["edges_GeV_c"])
            index = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
            encoded = (edges[index] + edges[index + 1]) / 2.
        u, logp = (encoded - .65) / .35, np.log(encoded / .65)
    return np.column_stack([u, v, q, logp, np.zeros_like(p)])


def fixed_group_ids(p_in, thickness, charge):
    """Existing seven physical diagnostic bins; distinct from input quantization."""
    p, t, q = map(lambda x: np.asarray(x, dtype=np.float64), (p_in, thickness, charge))
    information_features(p, t, q, context_spec("Full5"))
    pb = np.minimum(np.searchsorted(np.linspace(.3, 1., 8), p, side="right") - 1, 6)
    return (pb * 6 + np.searchsorted(np.array([.1, .3, 1.]), t) * 2 + (q == 1.)).astype(np.int64)
