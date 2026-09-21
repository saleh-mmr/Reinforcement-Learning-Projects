import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import config
from envs.frozen_lake_agent import DQNAgent
from envs.frozenlake import FrozenLakeEnv


class Trainer:
    def __init__(self, hyperparams):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.max_steps = hyperparams["max_steps"]                       # Episode timeout
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size

        self.frozen_lake_env = FrozenLakeEnv()
        self.agent = DQNAgent(
            n_action_space=self.frozen_lake_env.action_space.n,
            n_observation_space=self.frozen_lake_env.observation_space.n,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            )

    def train(self):
        total_steps = 0
        total_reward_fl = []

        for episode in range(1, self.max_episodes + 1):
            # Initial observation from environment
            state_fl = self.frozen_lake_env.reset()
            # Flags to track episode completion for each environment
            done_fl = False
            # Total reward accumulated in this episode each environment (for logging)
            episode_reward_fl = 0
            step_counter = 0 # Step counter inside episode
            while not done_fl:
                # For each environment, if it's not done, select action, step, store experience, and accumulate reward
                action_fl = self.agent.select_action(state_fl)
                # Step in the environment and get next state, reward, and done flag
                next_state_fl, reward_fl, done_fl = self.frozen_lake_env.step(action_fl)
                # Store experience in the corresponding replay memory
                self.agent.frozen_lake_memory.store(state_fl, action_fl, next_state_fl, reward_fl, done_fl)
                state_fl = next_state_fl
                episode_reward_fl += reward_fl
                if len(self.agent.frozen_lake_memory) > self.batch_size:
                    current_total_step = total_steps + step_counter
                    self.agent.learn(self.batch_size,  ap_index=0, step_counter=current_total_step)

                step_counter += 1

            total_steps += step_counter
            total_reward_fl.append(episode_reward_fl)
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward: {episode_reward_fl:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )
        return total_reward_fl