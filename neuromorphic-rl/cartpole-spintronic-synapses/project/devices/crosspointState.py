import numpy as np

class CrosspointState:
    def __init__(self, params):
        # Determine the initial index based on the ratio of g_ap_coefficient to g_p_coefficient and the shift parameter
        k = params.g_ap_coefficient / params.g_p_coefficient
        i = 1
        while i**k<=i+params.shift_parameter:
            i += 1
        self.x = i
        print("Initial index for crosspoint state: ", self.x)
        # Generates a single random number from a Gaussian (normal) distribution centered at 0.
        self.noise_realization = float(np.random.normal(0.0, params.noise_stddev))

    def update_noise(self, noise):
        self.noise_realization = noise

    def increment_index(self):
        self.x += 1
        return self.x

    def get_state(self):
        return self.x, self.noise_realization