import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import random
import numpy as np
import torch
from learning.trainer import Trainer
from datetime import datetime
from pathlib import Path


network_size = [80, 256]
g_ap = [20, 25, 30]
g_p = [8, 10, 15]
g_bias = [30, 40, 50]
regularization_C = [10000, 100000, 1000000]


hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 1200,
    "warmup_size": 1200,
    "network_size": 100,
    "max_steps_per_episode": 140,
    "max_episodes": 80000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000009,
    "memory_capacity": 10000,
    "g_ap": 25.0,
    "g_p": 22.0,
    "shift_parameter": 20,
    "g_bias": 62.0,
    "regularization_C": 100000.0,
    "noise_stddev": 0.0001,
    "CP_pole_length_1": 0.5,
    "CP_pole_mass_1": 0.1,
    "CP_pole_length_2": 8.0,
    "CP_pole_mass_2": 1.3,
    "CP_pole_length_3": 20.0,
    "CP_pole_mass_3": 4.0,
}


if __name__ == "__main__":
    seed = 873
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    root_folder = Path("../weights/grid_search") / f"grid_{timestamp}"
    root_folder.mkdir(parents=True, exist_ok=True)
    for network_size, g_ap, g_p, g_bias, regularization_C in itertools.product(network_size, g_ap, g_p, g_bias, regularization_C):
        hyperparams = base_hyperparams.copy()
        hyperparams["g_ap"] = float(g_ap)
        hyperparams["g_p"] = float(g_p)
        hyperparams["g_bias"] = float(g_bias)
        hyperparams["regularization_C"] = float(regularization_C)
        hyperparams["network_size"] = int(network_size)
        run_name = f"network_size_{network_size}_g_ap_{g_ap}_g_p_{g_p}_g_bias_{g_bias}_regularization_C_{regularization_C}"
        folder = root_folder / run_name
        folder.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Running: {run_name}")

        trainer = Trainer(hyperparams, seed, folder)
        rewards = trainer.train()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(rewards, label="Reward", linewidth=4)

        ax.set_xlabel("Episode", fontsize=15)
        ax.set_ylabel("Reward", fontsize=15)
        ax.set_title(
            f"g_ap={g_ap}, g_p={g_p}, g_bias={g_bias}",
            fontsize=16
        )
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plot_path = folder / "training_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)