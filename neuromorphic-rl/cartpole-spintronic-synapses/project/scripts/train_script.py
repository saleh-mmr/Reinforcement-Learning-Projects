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

hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 3000,
    "warmup_size": 3000,
    "network_size": 100,
    "max_steps_per_episode": 150,
    "max_episodes": 6000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00001,
    "memory_capacity": 10000,
    "g_ap": 25.0,
    "g_p": 22.0,
    "shift_parameter": 20,
    "g_bias": 45.0,
    "noise_stddev": 0.02,
    "CP_pole_length_1": 12.0,
    "CP_pole_mass_1": 3.0,
    "CP_pole_length_2": 5.0,
    "CP_pole_mass_2": 0.5
}



if __name__ == "__main__":
    seed = 873
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(f"run_{timestamp}")
    # Create the folder
    folder.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(hyperparams, seed, folder)
    rewards_cp, rewards_mc = trainer.train()

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- CP Plot -----
    axes[0].plot(rewards_cp, label="Reward CP", linewidth=2)
    axes[0].set_xlabel("Episode", fontsize=14)
    axes[0].set_ylabel("Reward", fontsize=14)
    axes[0].set_title(f"Training on CP with Pole {hyperparams["CP_pole_length_1"]}", fontsize=16)
    axes[0].grid(True)

    # ----- MC Plot -----
    axes[1].plot(rewards_mc, label="Reward MC", linewidth=2)
    axes[1].set_xlabel("Episode", fontsize=14)
    axes[1].set_ylabel("Reward", fontsize=14)
    axes[1].set_title(f"Training on CP with Pole Length {hyperparams["CP_pole_length_2"]}", fontsize=16)
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = folder / "training_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()