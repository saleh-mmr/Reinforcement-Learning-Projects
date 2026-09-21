import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Increase font size for all text elements
plt.rcParams.update({'font.size': 16})

# Paths to your models
base_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_dir, "../weights/we")
step = 305556
path1 = os.path.join(folder, f"CP_best_model_{step}.pth")
path2 = os.path.join(folder, f"MC_best_model_{step}.pth")




# Load weights
state_dict1 = torch.load(path1, map_location="cpu")
state_dict2 = torch.load(path2, map_location="cpu")

# ---------------------------------------------------------
# 1. FC.2 Weight Heatmaps
# ---------------------------------------------------------
# Extract FC.2.weight
w1 = state_dict1["FC.2.weight"].detach().cpu().numpy()
w2 = state_dict2["FC.2.weight"].detach().cpu().numpy()

max_abs = max(
    abs(w1.min()), abs(w1.max()),
    abs(w2.min()), abs(w2.max())
)

plt.figure(figsize=(14, 6))

# Plot 1
plt.subplot(1, 2, 1)
sns.heatmap(w1, center=0, vmin=-max_abs, vmax=max_abs,
            xticklabels=5, yticklabels=5)
plt.title("FC.2.weight - CartPole (L=0.5)")
plt.xticks(rotation=0)  # text stands upright
plt.yticks(rotation=0)  # text stands upright

# Plot 2
plt.subplot(1, 2, 2)
sns.heatmap(w2, center=0, vmin=-max_abs, vmax=max_abs,
            xticklabels=5, yticklabels=5)
plt.title("FC.2.weight - CartPole (L=0.7)")
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. FC.2 Weight Difference
# ---------------------------------------------------------
diff = w1 - w2
max_abs_diff = np.max(np.abs(diff))

plt.figure(figsize=(8, 6))
sns.heatmap(diff, cmap="seismic", center=0,
            vmin=-max_abs_diff, vmax=max_abs_diff,
            xticklabels=5, yticklabels=5)
plt.title("Difference in FC.2.weight (L=0.5 - L=0.7)")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()

# ---------------------------------------------------------
# 3. FC.2 Bias Heatmaps
# ---------------------------------------------------------
# Extract bias
b1 = state_dict1["FC.2.bias"].detach().cpu().numpy().reshape(1, -1)
b2 = state_dict2["FC.2.bias"].detach().cpu().numpy().reshape(1, -1)

# Shared color scale
max_abs_b = max(
    abs(b1.min()), abs(b1.max()),
    abs(b2.min()), abs(b2.max())
)

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
sns.heatmap(b1, cmap="seismic", center=0,
            vmin=-max_abs_b, vmax=max_abs_b,
            xticklabels=5, cbar=True)
plt.title("FC.2.bias - CartPole (L=0.5)")
plt.xticks(rotation=0)
plt.yticks([]) # Biases only have 1 row, so we hide y-ticks

plt.subplot(1, 2, 2)
sns.heatmap(b2, cmap="seismic", center=0,
            vmin=-max_abs_b, vmax=max_abs_b,
            xticklabels=5, cbar=True)
plt.title("FC.2.bias - CartPole (L=0.7)")
plt.xticks(rotation=0)
plt.yticks([])

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. FC.2 Bias Difference
# ---------------------------------------------------------
# Difference
diff_b = b1 - b2
max_abs_diff_b = np.max(np.abs(diff_b))

plt.figure(figsize=(10, 3))
sns.heatmap(diff_b, cmap="seismic", center=0,
            vmin=-max_abs_diff_b, vmax=max_abs_diff_b,
            xticklabels=5, cbar=True)
plt.title("Bias Difference (L=0.5 - L=0.7)")
plt.xticks(rotation=0)
plt.yticks([])
plt.show()