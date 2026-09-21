import csv
import os
import matplotlib.pyplot as plt
import torch


class TrainingLogger:
    # sets up default folder names and filenames for storing
    def __init__(
        self,
        csv_dir="outputs",
        csv_name="training_log.csv",
        weights_dir="weights",
        weights_name="weights.pth",
        results_dir="results",
        reward_plot_name="rewards.png"
    ):
        # Create directories if not exist
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # File paths
        self.csv_path = os.path.join(csv_dir, csv_name)
        self.weights_path = os.path.join(weights_dir, weights_name)
        self.rewards_plot_path = os.path.join(results_dir, reward_plot_name)
        self.loss_plot_path = os.path.join(results_dir, "loss.png")

        self.weight_history = []
        self.episode_rewards = []
        self.epsilon_values = []

        # Create CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "epsilon", "loss"])

    def log_episode(self, episode, reward, epsilon):
        self.episode_rewards.append(reward)
        self.epsilon_values.append(epsilon)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, epsilon])

    def finalize_results(self, model, loss_history):
        """Save trained weights + reward plot"""
        torch.save(model.state_dict(), self.weights_path)
        print(f"Saved model weights to: {self.weights_path}")

        # Save Reward Plot
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

        # Save Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_plot_path)
        plt.close()
        print(f"Saved loss plot to: {self.loss_plot_path}")

        print(f"Training log saved to: {self.csv_path}")