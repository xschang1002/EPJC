"""Deterministic event-level split contract used by development and main data."""

from __future__ import annotations

import hashlib


def split_label(event_uid: str, validation_fraction: float = 0.2, diagnostic_fraction: float = 0.0) -> str:
    """Return train/validation/diagnostic from one immutable UID hash."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0.0 <= diagnostic_fraction < 1.0:
        raise ValueError("diagnostic_fraction must be in [0, 1)")
    if validation_fraction + diagnostic_fraction >= 1.0:
        raise ValueError("validation and diagnostic fractions must leave a train split")
    value = int(hashlib.sha256(event_uid.encode("utf-8")).hexdigest()[:8], 16) / 2**32
    if value < validation_fraction:
        return "validation"
    if value < validation_fraction + diagnostic_fraction:
        return "diagnostic"
    return "train"
