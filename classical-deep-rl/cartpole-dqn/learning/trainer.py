import os
import random
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent
from envs.cartpole import CartPoleEnv


class Trainer:
    """Coordinates DQN training and evaluation for CartPole."""

    def __init__(self, hyperparams, seed):
        """Configure environment and agent from a hyperparameter dictionary."""
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.max_episodes = hyperparams["max_episodes"]
        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.memory_capacity = hyperparams["memory_capacity"]
        self.goal = hyperparams["goal"]
        self.network_size = hyperparams["network_size"]
        self.seed = seed
        self.env = CartPoleEnv(render_mode=None, seed=seed, max_steps=self.goal)

        self.agent = DQNAgent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            network_size=self.network_size
        )

    def train(self):
        """Run training episodes and save the best moving-average checkpoint."""
        total_steps = 0
        total_reward = []
        loss_track = []
        best_so_far = -float("inf")
        window_size = 5
        recent_avg = -float("inf")

        for episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            step_counter = 0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                step_counter += 1

                self.agent.replay_memory.store(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward

                if len(self.agent.replay_memory) >= self.batch_size:
                    loss = self.agent.learn(self.batch_size)
                    loss_track.append(loss)

            total_steps += step_counter
            total_reward.append(episode_reward)
            self.agent.update_epsilon(total_steps)

            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )

            # Keep the best checkpoint by recent average reward.
            if len(total_reward) >= window_size:
                recent_avg = np.mean(total_reward[-window_size:])
                if recent_avg >= best_so_far:
                    best_so_far = recent_avg
                    model_path = f"best_model_seed_{self.seed}.pth"
                    torch.save(
                        self.agent.q_network.state_dict(),
                        model_path
                    )
                    print(f"New best model saved (seed {self.seed}) with recent average reward {recent_avg:.2f} -> {model_path}")

        return total_reward, loss_track

    def test(self, model_path, num_tests=100):
        """Evaluate a saved model with greedy policy across random test seeds."""

        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        for test_num in range(num_tests):
            seed = random.randint(0, 3000)
            env = CartPoleEnv(render_mode=None, seed=seed)
            state = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done:
                # Pure exploitation at test time.
                action = self.agent.select_action(state, epsilon=0)

                next_state, reward, done = env.step(action)

                state = next_state
                total_reward += reward
                step_counter += 1

            rewards.append(total_reward)

            print(f"Test {test_num + 1} | Seed {seed} | Reward {total_reward}")

        print("\nMean Test Reward:", np.mean(rewards))
        print("Std Reward:", np.std(rewards))

        return rewards
