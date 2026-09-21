import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from devices.multiWeightSynapse import MultiWeightSynapse, MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams


class SynapticWeightController:
    def __init__(self, model, g_ap, g_p, shift_parameter, g_bias, noise_stddev):
        self.model = model

        params = CrossPointParams(g_ap_coefficient=g_ap, g_p_coefficient=g_p, shift_parameter=shift_parameter, g_bias_coefficient=g_bias, noise_stddev=noise_stddev)
        spec = MultiWeightSynapseSpec(n_problem=3, scaling_factor=1)

        self.synapses = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            shape = param.shape
            if len(shape) == 2:  # weight matrix
                self.synapses[name] = [
                    [MultiWeightSynapse(spec, params) for _ in range(shape[1])]
                    for _ in range(shape[0])
                ]

            elif len(shape) == 1:  # bias vector
                self.synapses[name] = [
                    MultiWeightSynapse(spec, params) for _ in range(shape[0])
                ]


    @torch.no_grad()
    def step(self, ap_index):
        # print("------------------Updating synaptic weights based on gradients for AP index: ", ap_index, "------------------")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                 continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid
            neg = (grad < 0) & valid


            # if name == "FC.0.weight":
            #     print("BEFORE update for FC.0.weight neuron:")
            #     print(f"Gradient: {grad[0, 0].item():.4f}")
            #     print(f"AP Positive Crosspoint {ap_index} index: {self.synapses[name][0][0].get_positive_crosspoint_state(ap_index)}")
            #     if ap_index == 0:
            #         print(f"P Positive Crosspoint 1 index: {self.synapses[name][0][0].get_positive_crosspoint_state(1)}")
            #     else:
            #         print(f"P Positive Crosspoint 0 index: {self.synapses[name][0][0].get_positive_crosspoint_state(0)}")
            #     print(f"bias crosspoint index: {self.synapses[name][0][0].get_bias_crosspoint_state()}")
            #     print("\n")


            if grad.ndim == 2:
                if pos.any():
                    for i in range(pos.shape[0]):
                        for j in range(pos.shape[1]):
                            if pos[i, j]:
                                self.synapses[name][i][j].increase_bias_crosspoint_index()
                if neg.any():
                    for i in range(neg.shape[0]):
                        for j in range(neg.shape[1]):
                            if neg[i, j]:
                                self.synapses[name][i][j].increase_positive_crosspoint_index(ap_index)
            elif grad.ndim == 1:
                if pos.any():
                    for i in range(pos.shape[0]):
                        if pos[i]:
                            self.synapses[name][i].increase_bias_crosspoint_index()
                if neg.any():
                    for i in range(neg.shape[0]):
                        if neg[i]:
                            self.synapses[name][i].increase_positive_crosspoint_index(ap_index)

            # if name == "FC.0.weight":
            #     print("After update for FC.0.weight neuron:")
            #     print(f"Ap Positive Crosspoint {ap_index} index: {self.synapses[name][0][0].get_positive_crosspoint_state(ap_index)}")
            #     if ap_index == 0:
            #         print(f"P Positive Crosspoint 1 index: {self.synapses[name][0][0].get_positive_crosspoint_state(1)}")
            #     else:
            #         print(f"P Positive Crosspoint 0 index: {self.synapses[name][0][0].get_positive_crosspoint_state(0)}")
            #     print(f"bias crosspoint index: {self.synapses[name][0][0].get_bias_crosspoint_state()}")
            #     print("-------------------")
            #     print("\n")

    @torch.no_grad()
    def load_weights(self, ap_index):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            grad = param.grad
            if grad is None:
                continue

            st = self.synapses[name]
            if param.ndim == 2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        param[i, j].copy_(torch.tensor(st[i][j].weight(ap_index), dtype=param.dtype))
            elif param.ndim == 1:
                for i in range(param.shape[0]):
                    param[i].copy_(torch.tensor(st[i].weight(ap_index), dtype=param.dtype))

            # if name == "FC.0.weight":
            #     print(f"In Controller Loaded Weights for ap_index: {ap_index} is {param[0, 0].item():.4f} ")
            #     print(f"Ap Positive Crosspoint {ap_index} index: {self.synapses[name][0][0].get_positive_crosspoint_state(ap_index)}")
            #     if ap_index == 0:
            #         print(f"P Positive Crosspoint 1 index: {self.synapses[name][0][0].get_positive_crosspoint_state(1)}")
            #     else:
            #         print(f"P Positive Crosspoint 0 index: {self.synapses[name][0][0].get_positive_crosspoint_state(0)}")
            #     print(f"bias crosspoint index: {self.synapses[name][0][0].get_bias_crosspoint_state()}")
            #     print(f"G_ap : {self.synapses[name][0][0].get_positive_crosspoint_conductance_ap(ap_index):.9e}")
            #     if ap_index == 0:
            #         print(f"G_p : {self.synapses[name][0][0].get_positive_crosspoint_conductance_p(1):.9e}")
            #     else:
            #         print(f"G_p : {self.synapses[name][0][0].get_positive_crosspoint_conductance_p(0):.9e}")
            #     print(f"G_bias : {self.synapses[name][0][0].get_bias_crosspoint_conductance():.9e}")
            #     print("-------------------")
            #     print("\n")