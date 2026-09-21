import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        g_p_coefficient = self.params.g_p_coefficient
        shift_parameter = self.params.shift_parameter
        index, noise = state.get_state()
        return float((g_p_coefficient * np.log10(index+shift_parameter)) * (1+noise))

    def conductance_ap(self, state: CrosspointState) -> float:
        g_ap_coefficient = self.params.g_ap_coefficient
        index, noise = state.get_state()
        return float((g_ap_coefficient * np.log10(index)) * (1+noise))
