import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CrossPointParams:
    """
    Parameters from experiments:
      - GP = a*log10(x) + b
      - if GP <= Gthreshold: GAP = GP
        else GAP = GP * (1 + c * (GP - Gthreshold)^(3/4))
        - Noise: add Gaussian noise_realization to G
    """

    def __init__(self, a, b, c, g_threshold, sigma_pulse_noise):
        self.a = a
        self.b = b
        self.c = c
        self.g_threshold = g_threshold
        self.sigma_pulse_noise = sigma_pulse_noise
        self.min_pulse_index_for_log = 1

    def get_sigma(self):
        return self.sigma_pulse_noise