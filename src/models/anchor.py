"""Versioned silicon pion centers; the public default has consistent MeV units.

The historical formula is preserved byte-for-byte in anchor_legacy.py and is
only a historical comparator, never the corrected physics representation.
"""
from __future__ import annotations

import numpy as np

from models.anchor_legacy import pion_bethe_bloch_delta_p as historical_pion_bethe_bloch_delta_p

CENTER_IDENTITIES = {
    "HistoricalIncorrectPhysics": {
        "center_id": "HistoricalIncorrectPhysics",
        "formula_version": "historical-anchor-v1-unit-mismatch",
        "unit_contract": "input=p_GeV_c,thickness_mm; output=GeV_c; historical mixed-unit logarithm",
    },
    "CorrectedPhysics": {
        "center_id": "CorrectedPhysics", "formula_version": "p0c-anchor-v1-all-mev",
        "unit_contract": "input=p_GeV_c,thickness_mm; internal=MeV; output=GeV_c",
    },
}


def corrected_pion_bethe_bloch_delta_p(p_GeV_c, thickness_mm, charge=1):
    """Nominal Bethe center with I=173 eV, without density-effect correction."""
    p, thickness, q = np.broadcast_arrays(np.asarray(p_GeV_c, dtype=np.float64),
        np.asarray(thickness_mm, dtype=np.float64), np.asarray(charge, dtype=np.float64))
    if any(np.any(~np.isfinite(value)) for value in (p, thickness, q)):
        raise ValueError("momentum, thickness and charge must be finite")
    if np.any(p <= 0) or np.any(thickness <= 0):
        raise ValueError("momentum and thickness must be positive")
    mass, electron_mass, excitation = 139.57039, 0.51099895069, 0.000173
    p_mev = 1000.0 * p
    energy = np.sqrt(p_mev**2 + mass**2)
    beta, gamma = p_mev / energy, energy / mass
    tmax = 2 * electron_mass * beta**2 * gamma**2 / (1 + 2 * gamma * electron_mass / mass + (electron_mass / mass)**2)
    argument = 2 * electron_mass * beta**2 * gamma**2 * tmax / excitation**2
    if np.any(argument <= 0) or np.any(~np.isfinite(argument)):
        raise FloatingPointError("invalid Bethe logarithm argument")
    stopping = .307075 * (14.0 / 28.0855) * q**2 / beta**2 * (.5 * np.log(argument) - beta**2)
    energy_loss = np.maximum(stopping * 2.329 * thickness / 10.0, 0.0)
    outgoing_energy = np.maximum(energy - energy_loss, mass)
    return np.maximum(p_mev - np.sqrt(np.maximum(outgoing_energy**2 - mass**2, 0.0)), 0.0) / 1000.0


def pion_bethe_bloch_delta_p(p_GeV_c, thickness_mm, charge=1):
    """Corrected default; historical analyses must select the historical function."""
    return corrected_pion_bethe_bloch_delta_p(p_GeV_c, thickness_mm, charge)
