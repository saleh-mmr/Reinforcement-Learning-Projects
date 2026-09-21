import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch


class TrainingLogger:
    # sets up default folder names and filenames for storing
    def __init__(
        self,
        results_dir="results",
        reward_plot_name="rewards.png"
    ):
        # Create directories if not exist
        os.makedirs(results_dir, exist_ok=True)

        # File paths
        self.rewards_plot_path = os.path.join(results_dir, reward_plot_name)

        self.weight_history = []
        self.episode_rewards = []
        self.epsilon_values = []
        self.loss_values = []


    def log_episode(self, episode, reward, epsilon, loss):
        self.episode_rewards.append(reward)
        self.epsilon_values.append(epsilon)
        self.loss_values.append(loss)

    def finalize_results(self, model):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Performance")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.rewards_plot_path)
        plt.close()
        print(f"Saved rewards plot to: {self.rewards_plot_path}")