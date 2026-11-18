"""Documented analytic Silicon stopping-power center for the pilot anchor."""

from __future__ import annotations

import numpy as np


# Bethe-Bloch constants in MeV cm^2 / g and Silicon nominal properties.
K = 0.307075
Z_OVER_A = 14.0 / 28.0855
RHO_SI_G_CM3 = 2.329
I_SI_MEV = 0.000173
M_PI_GEV = 0.13957039


def pion_bethe_bloch_delta_p(p_GeV_c, thickness_mm, charge=1):
    """Approximate mean pion momentum loss in GeV/c for nominal Silicon.

    This is a documented starting center, not a claim that it replaces the
    Geant4 transport. The density-effect correction is omitted at this pilot
    momentum range and the result is clipped to a physical positive loss.
    """
    p = np.asarray(p_GeV_c, dtype=float)
    thickness = np.asarray(thickness_mm, dtype=float)
    if np.any(p <= 0) or np.any(thickness <= 0):
        raise ValueError("momentum and thickness must be positive")
    mass = M_PI_GEV
    energy = np.sqrt(p * p + mass * mass)
    beta = p / energy
    gamma = energy / mass
    me = 0.00051099895
    tmax = (2 * me * beta * beta * gamma * gamma) / (1 + 2 * gamma * me / mass + (me / mass) ** 2)
    argument = np.maximum(2 * me * beta * beta * gamma * gamma * tmax / (I_SI_MEV * I_SI_MEV), 1.0000001)
    stopping_mev_cm2_g = K * Z_OVER_A * (charge * charge / (beta * beta)) * (0.5 * np.log(argument) - beta * beta)
    dE_GeV = np.maximum(stopping_mev_cm2_g * RHO_SI_G_CM3 * (thickness / 10.0) / 1000.0, 0.0)
    p_out = np.sqrt(np.maximum((energy - dE_GeV) ** 2 - mass * mass, 0.0))
    return np.maximum(p - p_out, 0.0)
