import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.config as config
import numpy as np
from devices.crosspointState import CrosspointState
from devices.magnetoresistiveCrosspoint import MagnetoresistiveCrosspoint
from devices.nonMagnetoresistiveCrosspoint import NonMagnetoresistiveCrosspoint
from devices.crosspointParams import CrossPointParams

class MultiWeightSynapseSpec:
    def __init__(self, n_problem, scaling_factor):
        self.n_problem = n_problem
        self.scaling_factor = scaling_factor

class MultiWeightSynapse:
    def __init__(self, multiweight_spec, crosspoint_params):
        self.spec = multiweight_spec
        self.params = crosspoint_params
        self.rng = np.random.default_rng(seed=config.seed)
        self.positive_crosspoints_states = [CrosspointState() for _ in range(self.spec.n_problem)]
        self.bias_state = CrosspointState()
        self.positive_crosspoint = [
            MagnetoresistiveCrosspoint(self.params, state, i)
            for i, state in enumerate(self.positive_crosspoints_states, start=1)
        ]
        self.bias_crosspoint = NonMagnetoresistiveCrosspoint(self.params,self.bias_state)


    def weight(self, ap_index):
        assert 0 <= ap_index < self.spec.n_problem
        g_p = 0
        for i in range(self.spec.n_problem):
            if i == ap_index:
                g_p += self.positive_crosspoint[i].conductance_ap(self.positive_crosspoints_states[i])
            else:
                g_p += self.positive_crosspoint[i].conductance_p(self.positive_crosspoints_states[i])
        g_bias = self.bias_crosspoint.conductance_p(self.bias_state)
        weight = self.spec.scaling_factor * (g_p - g_bias)
        return weight

    def increase_positive_crosspoint_index(self, index_positive_crosspoint):
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        self.positive_crosspoint[index_positive_crosspoint].update_state()

    def increase_bias_crosspoint_index(self):
        self.bias_crosspoint.update_state()

    def get_positive_crosspoint_state(self, index_positive_crosspoint):
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        return self.positive_crosspoint[index_positive_crosspoint].state.get_state()

    def get_bias_crosspoint_state(self):
        return self.bias_crosspoint.state.get_state()

    def get_positive_crosspoint_conductance_p(self, index_positive_crosspoint):
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        return self.positive_crosspoint[index_positive_crosspoint].conductance_p(self.positive_crosspoints_states[index_positive_crosspoint])

    def get_positive_crosspoint_conductance_ap(self, index_positive_crosspoint):
        assert 0 <= index_positive_crosspoint < self.spec.n_problem
        return self.positive_crosspoint[index_positive_crosspoint].conductance_ap(self.positive_crosspoints_states[index_positive_crosspoint])

    def get_bias_crosspoint_conductance(self):
        return self.bias_crosspoint.conductance_p(self.bias_state)

    def __str__(self):
        return f"MultiWeightSynapse(positive_crosspoints={self.positive_crosspoint}, bias_crosspoint={self.bias_crosspoint})"