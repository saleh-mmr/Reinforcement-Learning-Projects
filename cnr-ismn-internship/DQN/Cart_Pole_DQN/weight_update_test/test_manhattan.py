import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import Linear, ReLU

from manhattan_weight_controller import ManhattanWeightController

# reproducibility
torch.manual_seed(0)

# dataset
N = 256
x = torch.randn(N, 1)


true_w = torch.tensor([[3.0]])
true_b = torch.tensor([1.0])

# y = 3x + 1
y = x @ true_w + true_b


model = torch.nn.Sequential(
    torch.nn.Linear(1, 48),
    torch.nn.LeakyReLU(negative_slope=0.01),
    torch.nn.Linear(48, 48),
    torch.nn.LeakyReLU(negative_slope=0.01),
    torch.nn.Linear(48, 1),
)

controller = ManhattanWeightController(model, "output.csv")
criterion = torch.nn.MSELoss()


loss_history = []

for step in range(500):
    # forward
    pred = model(x)
    loss = criterion(pred, y)

    # backward
    model.zero_grad()
    loss.backward()

    # custom update
    controller.step()

    loss_history.append(loss.item())

    print(f"Step {step:4d} | Loss {loss.item():.6f}")





    # print(f"weight: {model[0].weight.data[0:1]}")
    # print(f"Loss: {model[0].bias.data}")


# conductance = pd.read_csv('datafile.csv', header=0)
# values = conductance.values.astype("float32").reshape(-1)
# values *= 2e7
# plt.plot(values)
# plt.show()