import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


class SynapticWeightController:
    def __init__(self, model, g_ap, g_p, shift_parameter, g_bias, noise_stddev):
        self.model = model

        # Conductance parameters
        self.g_ap = float(g_ap)
        self.g_p = float(g_p)
        self.shift_parameter = float(shift_parameter)
        self.g_bias = float(g_bias)
        self.noise_stddev = float(noise_stddev)

        # Same as your old MultiWeightSynapseSpec
        self.n_problem = 2
        self.scaling_factor = 1.0
        self.bias_x = {}
        self.bias_noise = {}
        self.positive_x = {}
        self.positive_noise = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            dtype = param.dtype
            shape = param.shape

            initial_x = self._initial_index(device=device, dtype=dtype)

            self.bias_x[name] = torch.full(
                shape,
                initial_x,
                dtype=dtype,
                device=device,
            )

            self.bias_noise[name] = self._draw_noise(
                shape,
                device=device,
                dtype=dtype,
            )

            self.positive_x[name] = torch.full(
                (self.n_problem, *shape),
                initial_x,
                dtype=dtype,
                device=device,
            )

            self.positive_noise[name] = self._draw_noise(
                (self.n_problem, *shape),
                device=device,
                dtype=dtype,
            )

        # Helps avoid unnecessary reloading later.
        # We keep it simple for now.
        self.current_loaded_ap_index = None
        self.weights_dirty = True

    def _initial_index(self, device, dtype):
        """
        Same logic as CrosspointState.__init__:

            k = g_ap / g_p
            i = 1
            while i**k <= i + shift_parameter:
                i += 1

        Returns the initial index as a float tensor value.
        """
        k = self.g_ap / self.g_p
        i = 1

        while i ** k <= i + self.shift_parameter:
            i += 1

        return float(i)

    def _draw_noise(self, shape, device, dtype):
        """
        Same behavior as np.random.normal(0.0, noise_stddev), but vectorized.

        If noise_stddev <= 0, return zero noise.
        """
        if self.noise_stddev > 0:
            return torch.normal(
                mean=0.0,
                std=self.noise_stddev,
                size=shape,
                device=device,
                dtype=dtype,
            )

        return torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, ap_index):
        """
        Vectorized replacement for the old per-weight update.

        Old behavior:

        if grad > 0:
            increase bias crosspoint index

        if grad < 0:
            increase positive crosspoint index for this ap_index
        """
        assert 0 <= ap_index < self.n_problem

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid
            neg = (grad < 0) & valid

            # --------------------------------------------------
            # Debug print for FC.2.weight[0][0]
            # --------------------------------------------------
            # if name == "FC.2.weight":
            #     row = 0
            #     col = 0
            #
            #     grad_00 = grad[row, col].item()
            #
            #     old_bias_x_00 = self.bias_x[name][row, col].item()
            #     old_bias_noise_00 = self.bias_noise[name][row, col].item()
            #
            #     old_pos_x_values = []
            #     old_pos_noise_values = []
            #
            #     for problem_index in range(self.n_problem):
            #         old_pos_x_values.append(
            #             self.positive_x[name][problem_index, row, col].item()
            #         )
            #         old_pos_noise_values.append(
            #             self.positive_noise[name][problem_index, row, col].item()
            #         )
            #
            #     if grad_00 > 0:
            #         update_type = "positive gradient -> update bias crosspoint"
            #     elif grad_00 < 0:
            #         update_type = f"negative gradient -> update positive crosspoint for ap_index={ap_index}"
            #     elif not torch.isfinite(grad[row, col]):
            #         update_type = "invalid gradient -> no update"
            #     else:
            #         update_type = "zero gradient -> no update"
            #
            #     print(
            #         "----------------------------------------\n"
            #         f"step(ap_index={ap_index}) | FC.2.weight[0][0]\n"
            #         f"gradient = {grad_00:.6f}\n"
            #         f"update   = {update_type}\n"
            #         f"before bias: x = {old_bias_x_00:.0f}, noise = {old_bias_noise_00:.6f}\n"
            #         f"before positive problem 0: x = {old_pos_x_values[0]:.0f}, noise = {old_pos_noise_values[0]:.6f}\n"
            #         f"before positive problem 1: x = {old_pos_x_values[1]:.0f}, noise = {old_pos_noise_values[1]:.6f}\n"
            #         f"before positive problem 2: x = {old_pos_x_values[2]:.0f}, noise = {old_pos_noise_values[2]:.6f}\n"
            #         "----------------------------------------"
            #     )

            # Positive gradient:
            # old code called increase_bias_crosspoint_index()
            if pos.any():
                self.bias_x[name][pos] += 1.0
                self.bias_noise[name][pos] = self._draw_noise(
                    self.bias_noise[name][pos].shape,
                    device=param.device,
                    dtype=param.dtype,
                )

            # Negative gradient:
            # old code called increase_positive_crosspoint_index(ap_index)
            if neg.any():
                self.positive_x[name][ap_index][neg] += 1.0
                self.positive_noise[name][ap_index][neg] = self._draw_noise(
                    self.positive_noise[name][ap_index][neg].shape,
                    device=param.device,
                    dtype=param.dtype,
                )

        self.weights_dirty = True

    @torch.no_grad()
    def load_weights(self, ap_index):
        """
        Load physical synaptic weights into the PyTorch model.

        Multiplicative noise model:

            G_ap   = (g_ap   * log10(x_ap))                 * (1 + noise_ap)
            G_p    = (g_p    * log10(x_p + shift_parameter)) * (1 + noise_p)
            G_bias = (g_bias * log10(x_bias))               * (1 + noise_bias)

        Weight:

            weight = scaling_factor * (G_total - G_bias)
        """
        assert 0 <= ap_index < self.n_problem

        if self.current_loaded_ap_index == ap_index and not self.weights_dirty:
            return

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # --------------------------------------------------
            # Bias conductance
            # --------------------------------------------------
            x_bias = self.bias_x[name]
            noise_bias = self.bias_noise[name]

            g_bias = (self.g_bias * torch.log10(x_bias)) * (1.0 + noise_bias)

            # --------------------------------------------------
            # Positive crosspoint conductance total
            # --------------------------------------------------
            g_total = torch.zeros_like(param)

            for problem_index in range(self.n_problem):
                x_pos = self.positive_x[name][problem_index]
                noise_pos = self.positive_noise[name][problem_index]

                if problem_index == ap_index:
                    # Anti-parallel conductance for selected problem
                    g = (self.g_ap * torch.log10(x_pos)) * (1.0 + noise_pos)
                else:
                    # Parallel conductance for non-selected problems
                    g = (self.g_p * torch.log10(x_pos + self.shift_parameter)) * (1.0 + noise_pos)

                g_total += g

            # --------------------------------------------------
            # Final physical weight
            # --------------------------------------------------
            weight = self.scaling_factor * (g_total - g_bias)

            param.copy_(weight)

            # --------------------------------------------------
            # Debug print for FC.2.weight[0][0]
            # --------------------------------------------------
            # if name == "FC.2.weight":
            #     row = 0
            #     col = 0
            #
            #     # Bias debug values
            #     x_bias_00 = self.bias_x[name][row, col]
            #     noise_bias_00 = self.bias_noise[name][row, col]
            #
            #     g_bias_00 = (
            #         self.g_bias * torch.log10(x_bias_00)
            #     ) * (1.0 + noise_bias_00)
            #
            #     # Positive crosspoint debug values
            #     g_values = []
            #     x_values = []
            #     noise_values = []
            #     conductance_types = []
            #
            #     for problem_index in range(self.n_problem):
            #         x_pos_00 = self.positive_x[name][problem_index, row, col]
            #         noise_pos_00 = self.positive_noise[name][problem_index, row, col]
            #
            #         if problem_index == ap_index:
            #             g_00 = (
            #                 self.g_ap * torch.log10(x_pos_00)
            #             ) * (1.0 + noise_pos_00)
            #             conductance_type = "G_ap"
            #         else:
            #             g_00 = (
            #                 self.g_p * torch.log10(x_pos_00 + self.shift_parameter)
            #             ) * (1.0 + noise_pos_00)
            #             conductance_type = "G_p"
            #
            #         g_values.append(g_00.item())
            #         x_values.append(x_pos_00.item())
            #         noise_values.append(noise_pos_00.item())
            #         conductance_types.append(conductance_type)
            #
            #     g_total_00 = sum(g_values)
            #     weight_00 = param[row, col].item()
            #
            #     print(
            #         "----------------------------------------\n"
            #         f"load_weights(ap_index={ap_index}) | FC.2.weight[0][0]\n"
            #         f"Problem 0 | {conductance_types[0]} = {g_values[0]:.6f}, "
            #         f"x0 = {x_values[0]:.0f}, noise0 = {noise_values[0]:.6f}\n"
            #         f"Problem 1 | {conductance_types[1]} = {g_values[1]:.6f}, "
            #         f"x1 = {x_values[1]:.0f}, noise1 = {noise_values[1]:.6f}\n"
            #         f"Problem 2 | {conductance_types[2]} = {g_values[2]:.6f}, "
            #         f"x2 = {x_values[2]:.0f}, noise2 = {noise_values[2]:.6f}\n"
            #         f"G_bias   = {g_bias_00.item():.6f}, "
            #         f"x_bias = {x_bias_00.item():.0f}, "
            #         f"noise_bias = {noise_bias_00.item():.6f}\n"
            #         f"G_total  = {g_total_00:.6f}\n"
            #         f"weight   = scaling_factor * (G_total - G_bias)\n"
            #         f"weight   = {self.scaling_factor} * "
            #         f"({g_total_00:.6f} - {g_bias_00.item():.6f})\n"
            #         f"weight   = {weight_00:.6f}\n"
            #         "----------------------------------------"
            #     )

        self.current_loaded_ap_index = ap_index
        self.weights_dirty = False