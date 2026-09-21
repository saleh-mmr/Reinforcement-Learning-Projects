from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

@dataclass
class MagnetoresistanceParams:
    """
    Parameters from experiments (paper):
      - GP = a*log10(x) + b
      - if GP <= Gthreshold: GAP = GP
        else GAP = GP * (1 + c * (GP - Gthreshold)^(3/4))
      - Noise: add Gaussian noise_realization to G
    """
    a: float
    b: float
    c: float
    g_threshold: float
    sigma_pulse_noise: float
    min_pulse_index_for_log: int = 1


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def gp_from_pulses(x, a, b, min_x=1):
    pulse_index_for_log = max(int(x), int(min_x))
    return float(a * np.log10(pulse_index_for_log) + b)


def gap_from_gp(gp, c, g_threshold):
    if gp <= g_threshold:
        return float(gp)
    return float(gp * (1.0 + c * (gp - g_threshold) ** (3.0 / 4.0)))


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

@dataclass
class CrosspointState:
    """
    Pulse index + current noise realization.
    """
    x: int = 0
    noise_realization: float = 0.0


# ---------------------------------------------------------------------
# Base Crosspoint (NEW)
# ---------------------------------------------------------------------

class BaseCrosspoint:
    """
    Base class for all crosspoints.
    Implements:
      - pulse updates
      - noise redraw
      - GP programming law
    """

    def __init__(self, params: MagnetoresistanceParams, rng=None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self.a = float(params.a)
        self.b = float(params.b)

    def redraw_noise(self, state: CrosspointState):
        sigma = float(self.params.sigma_pulse_noise)
        state.noise_realization = float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0

    def increment_pulses(self, state):
        n = 1
        state.x += int(n)
        self.redraw_noise(state)

    def gp(self, state: CrosspointState) -> float:
        gp = gp_from_pulses(
            x=state.x,
            a=self.a,
            b=self.b,
            min_x=self.params.min_pulse_index_for_log,
        )
        return float(gp + state.noise_realization)


# ---------------------------------------------------------------------
# Magnetic '+' Crosspoint
# ---------------------------------------------------------------------

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """

    def conductance_p(self, state: CrosspointState) -> float:
        return self.gp(state)

    def conductance_ap(self, state: CrosspointState) -> float:
        gp_noisy = self.gp(state)
        return gap_from_gp(
            gp=gp_noisy,
            c=self.params.c,
            g_threshold=self.params.g_threshold,
        )


# ---------------------------------------------------------------------
# Non-magnetic '-' Crosspoint
# ---------------------------------------------------------------------

class NonMagneticCrosspoint(BaseCrosspoint):
    """
    Bias crosspoint (no magnetoresistance).
    """

    def conductance(self, state: CrosspointState) -> float:
        return self.gp(state)
