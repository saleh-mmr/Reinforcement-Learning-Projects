import torch


class ManhattanController:
    def __init__(self, model):
        self.model = model

        self.state = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            shape = param.data.shape

            g_plus_idx = torch.ones(shape, dtype=torch.long, device=device)
            g_minus_idx = torch.zeros(shape, dtype=torch.long, device=device)

            g_plus = self._conductance(g_plus_idx).to(dtype=param.dtype)
            g_minus = self._conductance(g_minus_idx).to(dtype=param.dtype)

            self.state[name] = {
                "g_plus_idx": g_plus_idx,
                "g_minus_idx": g_minus_idx,
                "g_plus": g_plus,
                "g_minus": g_minus,
            }

    def _conductance(self, idx):
        idx_f = idx.to(dtype=torch.float32)
        value = idx_f * 0.001
        return value

    @torch.no_grad()
    def step(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            st = self.state[name]

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid
            neg = (grad < 0) & valid

            # Run this block only if there is at least one True value in pos
            if pos.any():
                st["g_minus_idx"][pos] += 1

            # Run this block only if there is at least one True value in neg
            if neg.any():
                st["g_plus_idx"][neg] += 1

            # a.copy_(b) means copy all values from tensor b into tensor a without creating new tensor
            st["g_plus"].copy_(self._conductance(st["g_plus_idx"]).to(dtype=param.dtype))
            st["g_minus"].copy_(self._conductance(st["g_minus_idx"]).to(dtype=param.dtype))

            weight = st["g_plus"] - st["g_minus"]
            param.copy_(weight)


            # if name == "FC.0.weight":
            #     grad_value = param.grad[0, 0].item()
            #     print(f"grad: {grad_value:.3f}")
            #     print(
            #         f"weight: {param[0, 0].item():.5f} | "
            #         f"g_plus_idx: {st['g_plus_idx'][0, 0].item()} => "
            #         f"g+: {st['g_plus'][0, 0].item():.5f} ||||||| "
            #         f"g_minus_idx: {st['g_minus_idx'][0, 0].item()} => "
            #         f"g-: {st['g_minus'][0, 0].item():.5f}"
            #     )
            #     print("\n")