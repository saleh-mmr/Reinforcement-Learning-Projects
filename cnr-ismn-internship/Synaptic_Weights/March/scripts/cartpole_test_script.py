import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from learning.trainer import Trainer


def load_hyperparams(folder):
    row = pd.read_csv(folder / "details_log.csv").iloc[0]

    return {
        "discount_factor": float(row["discount_factor"]),
        "batch_size": int(row["batch_size"]),
        "warmup_size": int(row["warmup_size"]),
        "network_size": int(row["network_size"]),
        "max_steps_per_episode": int(row["max_steps_per_episode"]),
        "max_episodes": int(row["max_episodes"]),
        "epsilon_max": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": float(row["epsilon_decay"]),
        "memory_capacity": int(row["memory_size"]),
        "g_ap": float(row["G_ap_coefficient"]),
        "g_p": float(row["G_p_coefficient"]),
        "shift_parameter": float(row["shift parameter"]),
        "g_bias": float(row["G_bias_coefficient"]),
        "regularization_C": 1e-8,
        "noise_stddev": float(row["noise_stddev"]),
        "CP_pole_length_1": float(row["CP_pole_length_1"]),
        "CP_pole_mass_1": float(row["CP_pole_mass_1"]),
        "CP_pole_length_2": float(row["CP_pole_length_2"]),
        "CP_pole_mass_2": float(row["CP_pole_mass_2"]),
        "CP_pole_length_3": float(row["CP_pole_length_3"]),
        "CP_pole_mass_3": float(row["CP_pole_mass_3"]),
    }


if __name__ == "__main__":

    folder_name = "run_2026-06-26_00-54-50"
    weigh_step = 3624
    num_tests = 2000

    folder = SCRIPT_DIR / "three_problems" / folder_name

    hyperparams = load_hyperparams(folder)

    trainer = Trainer(hyperparams, seed=None, folder=folder)

    pairs = [
        (0, "MC1"),
        (1, "MC2"),
        (2, "MC3"),
    ]

    plt.figure(figsize=(10, 6))

    for cartpole, keyword in pairs:
        model_path = folder / f"{keyword}_{weigh_step}.pth"

        test_log = trainer.test(
            model_path=str(model_path),
            num_tests=num_tests,
            cartpole=cartpole,
        )

        rewards = test_log["reward"].values
        episodes = range(len(rewards))

        plt.scatter(
            episodes,
            rewards,
            s=30,          # Larger scatter points
            alpha=0.6,
            label=f"Task {cartpole + 1}",
        )

        print(
            f"CartPole {cartpole}: "
            f"mean={rewards.mean():.3f}, "
            f"std={rewards.std():.3f}"
        )

    # Axis labels
    plt.xlabel("Test Episode", fontsize=20)
    plt.ylabel("Reward", fontsize=20)

    # Tick label size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Grid
    plt.grid(True)

    # Legend with larger text and marker
    plt.legend(fontsize=18, markerscale=2.5)

    plt.tight_layout()
    plt.show()