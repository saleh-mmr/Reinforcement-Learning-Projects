import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from devices.baseCrosspoint import BaseCrosspoint
from devices.crosspointState import CrosspointState

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        return self.gp(state)

    def conductance_ap(self, state: CrosspointState) -> float:
        gp_noisy = self.gp(state)
        g_threshold = self.params.g_threshold
        if gp_noisy <= g_threshold:
            return float(gp_noisy)
        c = self.params.c
        return float(gp_noisy * (1.0 + c * (gp_noisy - g_threshold) ** (3.0 / 4.0)))
