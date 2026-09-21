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
        a=1.566e-8
        b=0.350e-8
        c=1e6
        g_threshold=0.350e-8
        sigma_pulse_noise = 0.0
        scaling_factor=5e7
        n_problem = 2

        # model is the neural network whose weights we want to control
        self.model = model

        self.track_values = {}

        # initialize synapses for each trainable parameter in the model
        self.synapses = []
        self.params = CrossPointParams(a, b, c, g_threshold, sigma_pulse_noise)
        self.spec = MultiWeightSynapseSpec(n_problem, scaling_factor)

        # iterates through all parameters in the model, giving a tuple of (name, parameter tensor)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Ensure we have a dict to hold per-index tracking for this parameter
            self.track_values[name] = {}

            syn_array = np.empty(param.shape, dtype=object)
            # self.track_values[name] = {}
            # For each trainable parameter tensor, create a synapse object per element.
            # np.ndindex(param.shape) iterates over all index tuples of the tensor.
            # (e.g., a tensor of shape (4, 8) has 32 elements).
            for index in np.ndindex(param.shape):
                # initialize tracking for synapses corresponding to the current parameter index
                self.track_values[name][index] = {
                    "x_index": {0:[], 1:[]},
                    "bias_index": [],
                    "g_ap": {0:[], 1:[]},
                    "g_bias": [],
                    "weight": {0:[], 1:[]},
                    "loss": {},
                }
                current_synapse = MultiWeightSynapse(self.spec, self.params)
                syn_array[index] = current_synapse
                idx0 = current_synapse.get_positive_crosspoint_state(0)[0]
                idx1 = current_synapse.get_positive_crosspoint_state(1)[0]
                idb = current_synapse.get_bias_crosspoint_state()[0]
                ap0 = current_synapse.get_positive_crosspoint_conductance_ap(0)
                ap1 = current_synapse.get_positive_crosspoint_conductance_ap(1)
                bias = current_synapse.get_bias_crosspoint_conductance()
                weight0 = current_synapse.weight(0)
                weight1 = current_synapse.weight(1)
                self.track_values[name][index]["x_index"][0].append(idx0)
                self.track_values[name][index]["x_index"][1].append(idx1)
                self.track_values[name][index]["bias_index"].append(idb)
                self.track_values[name][index]["g_ap"][0].append(ap0)
                self.track_values[name][index]["g_ap"][1].append(ap1)
                self.track_values[name][index]["g_bias"].append(bias)
                self.track_values[name][index]["weight"][0].append(weight0)
                self.track_values[name][index]["weight"][1].append(weight1)

            # this is not recommend for lage networks,
            self.synapses.append(((name, param), syn_array))

    @torch.no_grad()
    def step(self, ap_index, step_counter):
        # normalize ap_index: accept int or single-element list/tuple/ndarray
        if isinstance(ap_index, (list, tuple, np.ndarray)):
            if len(ap_index) == 1:
                ap_index = int(ap_index[0])
            else:
                raise ValueError("ap_index must be an int or a single-element list/tuple/ndarray")
        else:
            ap_index = int(ap_index)

        for (name, param), syn_array in self.synapses:
            if param.grad is None:
                continue

            grad = param.grad
            for index in np.ndindex(param.shape):
                g_value = grad[index].item()
                syn = syn_array[index]
                sign = (g_value > 0) - (g_value < 0)
                if sign == 0:
                    continue
                if sign > 0:
                    syn.increase_bias_crosspoint_index()
                elif sign < 0:
                    syn.increase_positive_crosspoint_index(ap_index)

                weight = syn.weight(ap_index)
                param[index].copy_(
                    torch.tensor(weight, dtype=param.dtype, device=param.device)
                )
                if step_counter%100 == 0:  # Track every 100 steps
                    self.track_values[name][index]["weight"][ap_index].append(weight)
                    self.track_values[name][index]["g_ap"][ap_index].append(syn.get_positive_crosspoint_conductance_ap(ap_index))
                    if ap_index == 0:
                        self.track_values[name][index]["g_bias"].append(syn.get_bias_crosspoint_conductance())
                    self.track_values[name][index]["x_index"][ap_index].append(syn.get_positive_crosspoint_state(ap_index)[0])
                    self.track_values[name][index]["bias_index"].append(syn.get_bias_crosspoint_state()[0])