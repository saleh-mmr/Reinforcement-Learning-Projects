import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import config
from agents.agent import DQNAgent
from envs.cartepole import CartPoleEnv
from envs.mountaincar import MountainCarEnv


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

        self.cart_pole_env = CartPoleEnv(render_mode=None)
        self.mountain_car_env = MountainCarEnv(render_mode=None)
        self.envs = [self.cart_pole_env, self.mountain_car_env]
        if (
            self.cart_pole_env.action_space.n == self.mountain_car_env.action_space.n
            and self.cart_pole_env.observation_space.shape == self.mountain_car_env.observation_space.shape
        ):
            self.agent = DQNAgent(
                n_action_space=self.cart_pole_env.action_space.n,
                n_observation_space=self.cart_pole_env.observation_space.shape[0],
                epsilon_max=self.epsilon_max,
                epsilon_min=self.epsilon_min,
                epsilon_decay=self.epsilon_decay,
                discount=self.discount_factor,
                memory_capacity=self.memory_capacity,
            )
        else:
            raise ValueError("CartPole and MountainCar environments have incompatible action or observation spaces.")


    def train(self):
        total_steps = 0
        total_reward_cp = []
        total_reward_mc = []

        for episode in range(1, self.max_episodes + 1):
            # Initial observation from environment
            state_cp = self.cart_pole_env.reset()
            state_mc = self.mountain_car_env.reset()
            # Flags to track episode completion for each environment
            done_cp = False
            done_mc = False
            # Total reward accumulated in this episode each environment (for logging)
            episode_reward_cp = 0
            episode_reward_mc = 0
            step_counter = 0 # Step counter inside episode
            while not (done_cp and done_mc):
                # For each environment, if it's not done, select action, step, store experience, and accumulate reward
                if not done_cp:
                    action_cp = self.agent.select_action(state_cp)
                    # Step in the environment and get next state, reward, and done flag
                    next_state_cp, reward_cp, done_cp = self.cart_pole_env.step(action_cp)
                    # Store experience in the corresponding replay memory
                    self.agent.cart_pole_memory.store(state_cp, action_cp, next_state_cp, reward_cp, done_cp)
                    state_cp = next_state_cp
                    episode_reward_cp += reward_cp

                if not done_mc:
                    action_mc = self.agent.select_action(state_mc)
                    next_state_mc, reward_mc, done_mc = self.mountain_car_env.step(action_mc)
                    self.agent.mountain_car_memory.store(state_mc, action_mc, next_state_mc, reward_mc, done_mc)
                    state_mc = next_state_mc
                    episode_reward_mc += reward_mc

                for ap_index in range(len(self.envs)):
                    current_total_step = total_steps + step_counter
                    if len(self.agent.replay_memory[ap_index]) > self.batch_size:
                        self.agent.learn(self.batch_size, ap_index, current_total_step)

                step_counter += 1

            total_steps += step_counter
            total_reward_cp.append(episode_reward_cp)
            total_reward_mc.append(episode_reward_mc)
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward CP: {episode_reward_cp:.2f}, "
                f"Reward MC: {episode_reward_mc:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )
            # loss_cp = self.agent.loss_history[0]
            # loss_mc = self.agent.loss_history[1]
        return total_reward_cp , total_reward_mc