import numpy as np
import torch
import matplotlib.pyplot as plt


class ManhattanWeightController:
    def __init__(self, model, sigma=1.7e-9):
        self.model = model

        # constants for idx → conductance mapping
        self.a = 1.566e-8
        self.b = 0.350e-8
        self.base_scale = 9e7
        self.sigma = sigma

        self.state = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            shape = param.data.shape

            # index tensors
            g_plus_idx = torch.ones(shape, dtype=torch.long, device=device)
            g_minus_idx = torch.zeros(shape, dtype=torch.long, device=device)

            # initial conductances
            g_plus = self._conductance(g_plus_idx, param.dtype)
            g_minus = self._conductance(g_minus_idx, param.dtype)

            self.state[name] = {
                "param": param,
                "g_plus_idx": g_plus_idx,
                "g_minus_idx": g_minus_idx,
                "g_plus": g_plus,
                "g_minus": g_minus,
            }

    def _conductance(self, idx, dtype):
        idx_f = idx.to(dtype=torch.float32)
        one = torch.tensor(1.0, device=idx_f.device, dtype=idx_f.dtype)
        x = idx_f + (idx_f == 0) * one
        value = (self.a * torch.log10(x) + self.b)
        noise = torch.randn_like(value) * self.sigma
        # noise = noise.clamp(-0.001 * value.abs(), 0.001 * value.abs())
        value += noise
        value *= self.base_scale
        return value.to(dtype=dtype)

    @torch.no_grad()
    def step(self):
        for st in self.state.values():
            param = st["param"]
            grad = param.grad
            if grad is None:
                continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid  # want W down
            neg = (grad < 0) & valid  # want W up

            gp = st["g_plus_idx"]
            gm = st["g_minus_idx"]

            # grad > 0 → increase G-
            if pos.any():
                gm[pos] += 1

            # grad < 0 → increase G+
            if neg.any():
                gp[neg] += 1

            # update conductances
            st["g_plus"].copy_(self._conductance(gp, param.dtype))
            st["g_minus"].copy_(self._conductance(gm, param.dtype))

            # write back parameter
            param.data.copy_(st["g_plus"] - st["g_minus"])