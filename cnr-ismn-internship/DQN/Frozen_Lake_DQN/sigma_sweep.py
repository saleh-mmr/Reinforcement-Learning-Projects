import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from model_train_test import ModelTrainTest
from config import seed
import os


def sigma_to_str(sigma):
    return f"{sigma:.1e}".replace("+", "")


RESULTS_DIR = "frozenlake_results"
REWARDS_DIR = os.path.join(RESULTS_DIR, "rewards")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")

os.makedirs(REWARDS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


SIGMA_LIST = [
    1.7e-13,
    1.7e-12,
    1.7e-11,
    1.7e-10,
    1.7e-9,
]
map_size = 4
BASE_HYPERPARAMS = {
    "train_mode": True,
    "clip_grad_norm": 5,
    "learning_rate": 1e-4,
    "discount_factor": 0.9,
    "batch_size": 32,
    "update_frequency": 10,
    "max_episodes": 3000,
    "max_steps": 200,

    "max_epsilon": 1.0,
    "min_epsilon": 0.02,
    "epsilon_decay": 0.999,

    "memory_capacity": 150000,
    "render": False,
    "render_fps": 60,
    "number_render": 0,

    # required by your existing code
    "num_states": map_size**2,        # keep same as run.py if defined elsewhere
    "map_size": map_size,          # keep same as run.py if defined elsewhere
}
all_reward_histories = {}

for sigma in SIGMA_LIST:
    print(f"\n==============================")
    print(f" Training with sigma = {sigma}")
    print(f"==============================")

    # make a fresh copy of hyperparameters
    hyperparams = copy.deepcopy(BASE_HYPERPARAMS)

    # inject sigma
    hyperparams["sigma"] = sigma

    # ensure reproducibility per run
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create trainer
    trainer = ModelTrainTest(hyperparams)

    # train and get rewards
    rewards = trainer.train()
    all_reward_histories[sigma] = rewards

    # build filenames
    sigma_str = sigma_to_str(sigma)

    reward_file = os.path.join(REWARDS_DIR, f"sigma_{sigma_str}.npy")
    model_file = os.path.join(MODELS_DIR, f"sigma_{sigma_str}.pth")

    # save reward history (overwrite-safe)
    np.save(reward_file, np.array(rewards))

    # save model weights (overwrite-safe)
    torch.save(
        trainer.agent.main_network.state_dict(),
        model_file
    )

    print(f"Saved rewards → {reward_file}")
    print(f"Saved model   → {model_file}")


def moving_average(data, win):
    data = np.array(data)
    return np.convolve(data, np.ones(win) / win, mode="valid")

plt.figure(figsize=(10, 6))

window = 100

for sigma, rewards in all_reward_histories.items():
    smoothed = moving_average(rewards, win=window)

    plt.plot(
        range(window - 1, window - 1 + len(smoothed)),
        smoothed,
        label=f"sigma = {sigma:.1e}"
    )

plt.xlabel("Episode")
plt.ylabel(f"Mean Reward (last {window} episodes)")
plt.title("FrozenLake Training (50-episode Moving Average)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
