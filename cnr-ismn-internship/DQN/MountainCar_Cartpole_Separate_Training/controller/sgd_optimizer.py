import torch


class GDOptimizer:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def step(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            weight = param.data
            weight += -0.0001 * grad

            param.copy_(weight)

            # if name == "FC.0.weight":
            #     if abs(param[0, 0].item()) > 10:
            #         print(f"grad: {grad[0, 0].item():.3f}")
            #         print(f"weight: {param[0, 0].item():.5f}")
            #         print("state:", self.model.last_input)
            #         print("\n")