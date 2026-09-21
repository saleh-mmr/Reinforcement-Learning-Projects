from devices.baseCrosspoint import BaseCrosspoint
from devices.crosspointState import CrosspointState

class NonMagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Negative crosspoint.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        return self.gp(state)