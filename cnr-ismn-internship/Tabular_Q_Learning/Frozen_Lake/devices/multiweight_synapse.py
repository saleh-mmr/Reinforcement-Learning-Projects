from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from .magnetoresistance import (
    MagnetoresistiveCrosspoint,
    NonMagneticCrosspoint,
    CrosspointState,
)


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------

@dataclass
class MultiWeightSynapseSpec:
    n_plus: int
    scaling_factor: float


# ---------------------------------------------------------------------
# Composite Synapse
# ---------------------------------------------------------------------

class MultiWeightSynapse:
    """
    One composite synapse:
      - N '+' magnetic crosspoints
      - 1 '-' bias crosspoint
    """

    def __init__(self, spec, params, rng=None):
        self.spec = spec
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()

        self.plus_devices: List[MagnetoresistiveCrosspoint] = [
            MagnetoresistiveCrosspoint(params, self.rng)
            for _ in range(spec.n_problem)
        ]
        self.bias_device = NonMagneticCrosspoint(params, self.rng)

        self.plus_states = [CrosspointState() for _ in range(spec.n_problem)]
        self.bias_state = CrosspointState()

        # Initial noise draw
        for dev, st in zip(self.plus_devices, self.plus_states):
            dev.redraw_noise(st)
        self.bias_device.redraw_noise(self.bias_state)

    # ------------------------------------------------------------------
    # Weight evaluation
    # ------------------------------------------------------------------

    def weight(self, ap_index):
        assert 0 <= ap_index < self.spec.n_problem

        g_sum = 0.0
        for i, (dev, st) in enumerate(zip(self.plus_devices, self.plus_states)):
            if i == ap_index:
                g_sum += dev.conductance_ap(st)
            else:
                g_sum += dev.conductance_p(st)

        g_bias = self.bias_device.conductance(self.bias_state)
        return float(self.spec.scaling_factor * (g_sum - g_bias)) , float(g_sum)

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------

    def increase_plus(self, index):
        assert 0 <= index < self.spec.n_problem
        self.plus_devices[index].increment_pulses(self.plus_states[index])

    def increase_bias(self):
        self.bias_device.increment_pulses(self.bias_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pulse_indices(self):
        return {
            "plus": [st.x for st in self.plus_states],
            "bias": self.bias_state.x,
        }
