import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent
from envs.cartpole import CartPoleEnv
from envs.mountaincar import MountainCarEnv


class Trainer:
    def __init__(self, hyperparams, seed):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.max_steps = hyperparams["max_steps"]                       # Episode timeout
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size
        self.seed = seed
        if hyperparams["problem"] == 1:
            self.env = CartPoleEnv(render_mode=None, seed=seed)
            self.problem = 1
        else:
            self.env = MountainCarEnv(render_mode=None, seed=seed)
            self.problem = 2

        self.agent = DQNAgent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            optimizer_selector=hyperparams["controller"]
        )


    def train(self):
        self.warmup_replay_memory(20000)
        total_steps = 0
        total_reward = []
        loss_track = []
        best_so_far = -float("inf")
        window_size = 5
        recent_avg = -float("inf")

        for episode in range(1, self.max_episodes + 1):
            # Initial observation from environment
            state = self.env.reset()
            # Flags to track episode completion for each environment
            done = False
            # Total reward accumulated in this episode each environment (for logging)
            episode_reward = 0
            step_counter = 0 # Step counter inside episode
            while not done and step_counter < self.max_steps:
                # For each environment, if it's not done, select action, step, store experience, and accumulate reward
                action = self.agent.select_action(state)
                # Step in the environment and get next state, reward, and done flag
                next_state, reward, done = self.env.step(action)
                step_counter += 1

                # Store experience in the corresponding replay memory
                self.agent.replay_memory.store(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward

                if len(self.agent.replay_memory) >= self.batch_size:
                    loss = self.agent.learn(self.batch_size)
                    loss_track.append(loss)

            total_steps += step_counter
            total_reward.append(episode_reward)
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )
            # SAVE BEST MODEL
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

        # load trained weights
        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        for test_num in range(num_tests):
            seed = random.randint(0, 3000)
            if self.problem == 1:
                env = CartPoleEnv(render_mode=None, seed=seed)
            else:
                env = MountainCarEnv(render_mode=None, seed=seed)
            state = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done and step_counter < self.max_steps:
                # greedy action (no exploration)
                action = self.agent.select_action(state, epsilon=0)

                next_state, reward, done = env.step(action)
                # env.render()

                state = next_state
                total_reward += reward
                step_counter += 1

            rewards.append(total_reward)

            print(f"Test {test_num + 1} | Seed {seed} | Reward {total_reward}")

        print("\nMean Test Reward:", np.mean(rewards))
        print("Std Reward:", np.std(rewards))

        return rewards

    def warmup_replay_memory(self, num_steps):
        state = self.env.reset()
        for _ in range(num_steps):
            # random action for exploration
            action = self.env.action_space.sample()
            next_state, reward, done = self.env.step(action)
            self.agent.replay_memory.store(state, action, next_state, reward, done)
            state = self.env.reset() if done else next_state

