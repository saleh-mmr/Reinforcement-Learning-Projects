import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class NonMagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Negative crosspoint.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        g_bias_coefficient = self.params.g_bias_coefficient
        index, noise = state.get_state()
        return float((g_bias_coefficient * np.log10(index)) * (1+noise))