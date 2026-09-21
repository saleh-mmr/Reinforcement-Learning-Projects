import random
import numpy as np
import torch
from agents.agent import DQNAgent
from envs.cartpole import CartPoleEnv
from envs.mountaincar import MountainCarEnv
from envs.mycartpole import MyCartPoleEnv
import pandas as pd


class Trainer:
    def __init__(self, hyperparams, seed, folder):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size
        self.warmup_size = hyperparams["warmup_size"]                   # Number of random steps to fill replay memory before learning starts
        self.network_size = hyperparams["network_size"]                 # Number of neurons in hidden layers
        self.max_steps_per_episode = hyperparams["max_steps_per_episode"] # Max steps per episode to prevent infinite loops
        self.g_ap = hyperparams["g_ap"]                                  # Coefficient for Conductance ap
        self.g_p = hyperparams["g_p"]                                    # Coefficient for Conductance p
        self.shift_parameter = hyperparams["shift_parameter"]            # used in log(index + c) in conductance calculation
        self.g_bias = hyperparams["g_bias"]                              # Coefficient for Conductance bias
        self.noise_stddev = hyperparams["noise_stddev"]                  # Standard deviation of noise added to weight updates (for realism)
        self.CP_pole_length_1 = hyperparams["CP_pole_length_1"]          # Pole length for My Cart Pole environment
        self.CP_pole_mass_1 = hyperparams["CP_pole_mass_1"]              # Pole mass for My Cart Pole environment
        self.CP_pole_length_2 = hyperparams["CP_pole_length_2"]          # Pole length for My Cart Pole environment
        self.CP_pole_mass_2 = hyperparams["CP_pole_mass_2"]              # Pole mass for My Cart Pole environment
        self.seed = seed
        self.folder = folder
        self.first_cartpole_env = MyCartPoleEnv(render_mode=None, seed=seed, max_steps=self.max_steps_per_episode, pole_length=self.CP_pole_length_1, pole_mass=self.CP_pole_mass_1)
        self.second_cartpole_env = MyCartPoleEnv(render_mode=None, seed=seed, max_steps=self.max_steps_per_episode, pole_length=self.CP_pole_length_2, pole_mass=self.CP_pole_mass_2)



        self.agent = DQNAgent(
            first_cartpole_env= self.first_cartpole_env,
            second_cartpole_env= self.second_cartpole_env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            network_size=self.network_size,
            g_ap=self.g_ap,
            g_p=self.g_p,
            shift_parameter = self.shift_parameter,
            g_bias=self.g_bias,
            noise_stddev=self.noise_stddev
        )

    def train(self):
        self.warmup_replay_memory(self.warmup_size)
        total_steps = 0
        total_rewards_in_episodes = []

        # ---------Logging setup---------
        training_logs = pd.DataFrame(columns=["Episode", f"Steps", "Epsilon"])
        details_logs = pd.DataFrame(columns=["batch_size", "epsilon_decay", "memory_size", "network_size", "warmup_size", "seed", "max_episodes", "max_steps_per_episode", "discount_factor", "G_ap_coefficient", "G_p_coefficient", "shift parameter", "G_bias_coefficient", "noise_stddev", "CP_pole_length_1", "CP_pole_mass_1", "CP_pole_length_2", "CP_pole_mass_2"])
        details_logs.loc[len(details_logs)] = [self.batch_size, self.epsilon_decay, self.memory_capacity, self.network_size, self.warmup_size, self.seed, self.max_episodes, self.max_steps_per_episode, self.discount_factor, self.g_ap, self.g_p,self.shift_parameter, self.g_bias, self.noise_stddev, self.CP_pole_length_1, self.CP_pole_mass_1, self.CP_pole_length_2, self.CP_pole_mass_2]
        details_logs.to_csv(self.folder / "details_log.csv", index=False)

        for episode in range(1, self.max_episodes + 1):
            # Initial observation from environment
            state_mc1 = self.first_cartpole_env.reset()
            state_mc2 = self.second_cartpole_env.reset()
            # Flags to track episode completion for each environment
            done_mc1 = False
            done_mc2 = False
            # Total reward accumulated in this episode each environment (for logging)
            step_counter = 0                # Step counter inside episode
            while not done_mc1 and not done_mc2:
                step_counter += 1
                total_steps += 1

                # ---- First Cart Pole ----
                if not done_mc1:
                    action_mc1 = self.agent.select_action(state_mc1, ap_index=0)
                    next_state_mc1, reward_mc1, done_mc1 = self.first_cartpole_env.step(action_mc1)
                    self.agent.first_cartpole_memory.store(state_mc1, action_mc1, next_state_mc1, reward_mc1, done_mc1)
                    state_mc1 = next_state_mc1
                    self.agent.learn(self.batch_size, ap_index=0)

                # ---- Second Cart Pole ----
                if not done_mc2:
                    action_mc2 = self.agent.select_action(state_mc2, ap_index=1)
                    next_state_mc2, reward_mc2, done_mc2 = self.second_cartpole_env.step(action_mc2)
                    self.agent.second_cartpole_memory.store(state_mc2, action_mc2, next_state_mc2, reward_mc2, done_mc2)
                    state_mc2 = next_state_mc2
                    self.agent.learn(self.batch_size, ap_index=1)

            total_rewards_in_episodes.append(step_counter)
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )
            training_logs.loc[len(training_logs)] = [episode, step_counter, self.agent.epsilon]

            # SAVE BEST MODEL FOR EACH CARTPOLE BASED ON STEPS SURVIVED IN EPISODE
            if step_counter >= self.max_steps_per_episode:
                model_path_1 = f"MC1_{total_steps}.pth"
                model_path_2 = f"MC2_{total_steps}.pth"
                self.agent.weight_controller.load_weights(0)
                torch.save(self.agent.q_network.state_dict(), self.folder /model_path_1)
                print(f"First Cartpole New best model saved (seed {self.seed}) with reward {step_counter:.2f} -> {model_path_1}")
                self.agent.weight_controller.load_weights(1)
                torch.save(self.agent.q_network.state_dict(), self.folder /model_path_2)
                print(f"Second Cartpole New best model saved (seed {self.seed}) with reward {step_counter:.2f} -> {model_path_2}")

        training_logs.to_csv(self.folder /"training_log.csv", index=False)
        return total_rewards_in_episodes




    def test(self, model_path, num_tests, cartpole, render=False):
        # load trained weights
        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        tests_logs = pd.DataFrame(columns=["test", "reward"])
        for test_num in range(num_tests):
            seed = random.randint(0, 50000)
            # Use human render mode when requested so env.render() will display the environment.
            render_mode = "human" if render else None
            if cartpole == 0:
                env = MyCartPoleEnv(render_mode=render_mode, seed=seed, max_steps=self.max_steps_per_episode, pole_length=self.CP_pole_length_1, pole_mass=self.CP_pole_mass_1)
            else:
                env = MyCartPoleEnv(render_mode=render_mode, seed=seed, max_steps=self.max_steps_per_episode, pole_length=self.CP_pole_length_2, pole_mass=self.CP_pole_mass_2)
            state = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done:
                # Render only when requested (env must be created with a render_mode that supports rendering)
                if render:
                    env.render()
                # greedy action (no exploration)
                action = self.agent.select_action_test(state)
                next_state, reward, done = env.step(action)
                state = next_state
                total_reward += reward
                step_counter += 1

            rewards.append(total_reward)
            # print(f"Test {test_num + 1} | Seed {seed} | Reward {total_reward}")
            tests_logs.loc[len(tests_logs)] = [test_num + 1 , total_reward]
        return tests_logs


    def warmup_replay_memory(self, num_steps):
        state_mc1 = self.first_cartpole_env.reset()
        state_mc2 = self.second_cartpole_env.reset()
        for _ in range(num_steps):
            # random action for exploration
            action_mc1 = self.first_cartpole_env.action_space.sample()
            action_mc2 = self.second_cartpole_env.action_space.sample()
            next_state_mc1, reward_mc1, done_mc1 = self.first_cartpole_env.step(action_mc1)
            next_state_mc2, reward_mc2, done_mc2 = self.second_cartpole_env.step(action_mc2)
            self.agent.first_cartpole_memory.store(state_mc1, action_mc1, next_state_mc1, reward_mc1, done_mc1)
            self.agent.second_cartpole_memory.store(state_mc2, action_mc2, next_state_mc2, reward_mc2, done_mc2)
            state_mc1 = self.first_cartpole_env.reset() if done_mc1 else next_state_mc1
            state_mc2 = self.second_cartpole_env.reset() if done_mc2 else next_state_mc2