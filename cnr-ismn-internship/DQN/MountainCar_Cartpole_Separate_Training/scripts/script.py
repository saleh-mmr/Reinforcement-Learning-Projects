import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import random
import numpy as np
import torch
from learning.trainer import Trainer


controller = {
    1: "manhattan_linear",
    2: "manhattan_logarithmic",
    3: "sgd",
    4: "rmsprop"
}

problem = {
    1: "CartPole",
    2: "MountainCar",
}

hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 100,
    "max_episodes": 1000,
    "max_steps": 200,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00005,
    "memory_capacity": 100000,
    "problem": 1,
    "controller": 2,
}

train_mode = True


if __name__ == "__main__":

    if train_mode:
        seeds = [49]
        rewards_list = {}
        loss_list = {}
        for seed in seeds:
            print(f"Training with seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            trainer = Trainer(hyperparams, seed)
            rewards, loss = trainer.train()
            rewards_list[seed] = rewards
            loss_list[seed] = loss
        reward_runs = list(rewards_list.values())
        mean_rewards = np.mean(reward_runs, axis=0)
        std_rewards = np.std(reward_runs, axis=0)

        # Plot results
        plt.figure(figsize=(10,6))
        for seed, rewards in rewards_list.items():
            plt.plot(rewards, label=f"Seed {seed}", alpha=0.7)
        plt.plot(mean_rewards, label="Mean Reward", linewidth=3)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"DQN Training on {problem[hyperparams["problem"]]} with {controller[hyperparams["controller"]]} Controller")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        trainer = Trainer(hyperparams, seed=None)
        trainer.test("best_model_seed_49.pth")