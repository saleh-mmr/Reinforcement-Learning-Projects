import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CrosspointState:
    def __init__(self):
        self.x = 1
        self.noise_realization = 0.0

    def update_noise(self, noise):
        self.noise_realization = noise

    def increment_index(self):
        self.x += 1
        return self.x

    def get_state(self):
        return self.x, self.noise_realization