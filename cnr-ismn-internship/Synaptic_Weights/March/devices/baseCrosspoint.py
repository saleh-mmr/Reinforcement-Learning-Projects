from .crosspointParams import CrossPointParams
from .crosspointState import CrosspointState
import numpy as np


class BaseCrosspoint:
    def __init__(self, params: CrossPointParams, state: CrosspointState):
        self.params = params
        self.state = state

    #  Following Method Redraws Noise
    def redraw_noise(self, sigma):
        # Generates a single random number from a Gaussian (normal) distribution centered at 0.
        noise = float(np.random.normal(0.0, sigma)) if sigma > 0 else 0.0
        return noise

    # Following Method Changes State (increment X and redraw noise)
    def update_state(self):
        self.state.increment_index()
        sigma = self.params.get_noise_stddev()
        noise = self.redraw_noise(sigma)
        self.state.update_noise(noise)