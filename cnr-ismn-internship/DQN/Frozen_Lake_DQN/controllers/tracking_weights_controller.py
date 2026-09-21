import csv

import numpy as np
import torch
import matplotlib.pyplot as plt


class ManhattanWeightController:
    def __init__(self, model):
        self.model = model

        # constants for idx → conductance mapping
        self.a = 1.566e-8
        self.b = 0.350e-8
        self.base_scale = 9e7
        self.sigma = 1.7e-9

        self.state = {}

        # ---- tracking configuration (internal) ----
        self.track_param_name = "FC.0.weight"
        self.track_index = (1, 1)

        self.g_pos_history = []
        self.g_min_history = []
        self.weight_history = []

        self._tracked_state = None

        # initialize state
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

            # bind tracked neuron AFTER state creation
            if name == self.track_param_name:
                self._tracked_state = self.state[name]

        # ---- safety checks ----
        if self._tracked_state is None:
            raise ValueError(
                f"Tracked parameter '{self.track_param_name}' not found in model."
            )

        shape = self._tracked_state["g_plus"].shape
        if any(i < 0 or i >= s for i, s in zip(self.track_index, shape)):
            raise IndexError(
                f"track_index {self.track_index} out of bounds for shape {shape}"
            )

    def _conductance(self, idx, dtype):
        idx_f = idx.to(dtype=torch.float32)
        one = torch.tensor(1.0, device=idx_f.device, dtype=idx_f.dtype)
        x = idx_f + (idx_f == 0) * one
        value = (self.a * torch.log10(x) + self.b)
        noise = torch.randn_like(value) * self.sigma
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
            pos = (grad > 0) & valid
            neg = (grad < 0) & valid

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

        # ---- record tracked neuron ONCE per step ----
        idx = self.track_index
        gp_val = self._tracked_state["g_plus"][idx].item()
        gm_val = self._tracked_state["g_minus"][idx].item()
        w_val = (self._tracked_state["param"].data[self.track_index].item())
        self.weight_history.append(w_val)

        self.g_pos_history.append(gp_val)
        self.g_min_history.append(gm_val)

    def plot_tracked_conductance(self, save_path='g_history_plot.png'):
        if len(self.g_pos_history) == 0:
            raise RuntimeError("No conductance history recorded yet.")

        fig, ax1 = plt.subplots(figsize=(9, 5))

        # ---- conductances (left axis) ----
        ax1.plot(self.g_pos_history, label="G+", color="tab:blue", linestyle="-")
        ax1.plot(self.g_min_history, label="G−", color="tab:orange", linestyle="-")
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("Conductance")
        ax1.grid(True)

        # ---- weight (right axis) ----
        ax2 = ax1.twinx()
        ax2.plot(self.weight_history, label="Weight", color="tab:red", marker="o", linestyle="None", markersize=4)
        ax2.set_ylabel("Weight value")

        # ---- title ----
        ax1.set_title(
            f"{self.track_param_name}[{self.track_index[0]}, {self.track_index[1]}]"
        )

        # ---- combined legend ----
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")

        plt.show()

    def save_tracked_history_csv(self, filepath):
        if len(self.g_pos_history) == 0:
            raise RuntimeError("No history recorded yet.")

        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)

            # header
            writer.writerow(["g_pos", "g_neg", "weight"])

            # rows
            for gp, gm, w in zip(
                    self.g_pos_history,
                    self.g_min_history,
                    self.weight_history,
            ):
                writer.writerow([gp, gm, w])

        print(f"CSV saved to: {filepath}")

