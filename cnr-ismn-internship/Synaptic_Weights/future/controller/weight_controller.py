import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from devices.multiWeightSynapse import MultiWeightSynapse, MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams


class ManhattanWeightController:
    def __init__(self, model):
        # Parameters for crosspoints and synapse spec
        a = 1.566e-8
        b = 0.350e-8
        c = 5e4
        g_threshold = 0.350e-8
        sigma_pulse_noise = 0.0
        scaling_factor = 7e7
        n_problem = 2

        # model is the neural network whose weights we want to control
        self.model = model


        # initialize synapses for each trainable parameter in the model
        self.synapses = []
        self.params = CrossPointParams(a, b, c, g_threshold, sigma_pulse_noise)
        self.spec = MultiWeightSynapseSpec(n_problem, scaling_factor)

        # iterates through all parameters in the model, giving a tuple of (name, parameter tensor)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            syn_array = np.empty(param.shape, dtype=object)
            # self.track_values[name] = {}
            # For each trainable parameter tensor, create a synapse object per element.
            # np.ndindex(param.shape) iterates over all index tuples of the tensor.
            # (e.g., a tensor of shape (4, 8) has 32 elements).
            for index in np.ndindex(param.shape):
                current_synapse = MultiWeightSynapse(self.spec, self.params)
                syn_array[index] = current_synapse


            # this is not recommend for lage networks,
            self.synapses.append((param, syn_array))

    @torch.no_grad()
    def step(self):
        for param, syn_array in self.synapses:
            if param.grad is None:
                continue
            grad = param.grad
            for index in np.ndindex(param.shape):
                g_value = grad[index].item()
                syn = syn_array[index]
                sign = (g_value > 0) - (g_value < 0)
                if sign == 0:
                    continue
                if sign>0:
                    syn.increase_bias_crosspoint_index()
                elif sign<0:
                    syn.increase_positive_crosspoint_index(0)

                weight = syn.weight(0)
                param[index].copy_(
                    torch.tensor(weight, dtype=param.dtype, device=param.device)
                )