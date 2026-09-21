import gymnasium as gym
import numpy as np
import pygame
import torch
from matplotlib import pyplot as plt

from config import device, seed
from agent import DQNAgent


class ModelTrainTest:
    def __init__(self, hyperparams):
        self.train_mode = hyperparams['train_mode']
        self.clip_grad_norm = hyperparams['clip_grad_norm']
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.batch_size = hyperparams['batch_size']
        self.update_frequency = hyperparams['update_frequency']
        self.max_episodes = hyperparams['max_episodes']
        self.max_steps = hyperparams['max_steps']
        self.render = hyperparams['render']
        self.max_epsilon = hyperparams['max_epsilon']
        self.min_epsilon = hyperparams['min_epsilon']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.memory_capacity = hyperparams['memory_capacity']
        self.num_states = hyperparams['num_states']
        self.map_size = hyperparams['map_size']
        self.render_fps = hyperparams['render_fps']

        self.env = gym.make('FrozenLake-v1', map_name=f"{self.map_size}x{self.map_size}",
                            is_slippery=False, max_episode_steps=self.max_steps,
                            render_mode='human' if self.render else None)
        self.env.metadata['render_fps']=self.render_fps
        self.agent = DQNAgent(env=self.env,
                              epsilon_max=self.max_epsilon,
                              epsilon_min=self.min_epsilon,
                              epsilon_decay=self.epsilon_decay,
                              clip_grad_norm=self.clip_grad_norm,
                              learning_rate=self.learning_rate,
                              memory_capacity=self.memory_capacity,
                              discount=self.discount_factor,
                              sigma=hyperparams["sigma"])

    def state_preprocess(self, state, num_states):
        """
        Convert a state to a tensor, and basically it encodes the state into
        an onehot vector. For example, the return can be something like tensor([0,0,1,0,0])
        which could mean agent is at state 2 from total of 5 states.
        """
        onehot_vector = torch.zeros(num_states, dtype=torch.float32, device=device)
        onehot_vector[state] = 1
        return onehot_vector

    def train(self):
        total_steps = 0
        self.reward_history = []
        for episode in range(1, self.max_episodes + 1):

            # --- Enable rendering only for first N and last N episodes ---
            # N = 2 # Number of episodes to render at start and end
            # if episode <= N or episode > self.max_episodes - N:
            #     render_mode = "human"
            # else:
            #     render_mode = None
            render_mode = None

            # Recreate env with correct render mode
            self.env = gym.make(
                'FrozenLake-v1',
                map_name=f"{self.map_size}x{self.map_size}",
                is_slippery=False,
                max_episode_steps=self.max_steps,
                render_mode=render_mode
            )
            self.env.metadata['render_fps'] = self.render_fps

            # ---------------------------------------------------------------

            state, _ = self.env.reset(seed=seed)
            state = self.state_preprocess(state, num_states=self.num_states)

            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_state = self.state_preprocess(next_state, num_states=self.num_states)

                self.agent.replay_memory.store(state, action, next_state, reward, done)

                if len(self.agent.replay_memory) > self.batch_size and sum(self.reward_history) > 0:
                    self.agent.learn(self.batch_size, (done or truncation))

                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            # log history
            self.reward_history.append(episode_reward)
            total_steps += step_size

            # decay epsilon
            self.agent.update_epsilon()

            print(
                f"Episode: {episode}, Total Steps: {total_steps}, Ep Step: {step_size}, Reward: {episode_reward:.2f}, Epsilon: {self.agent.epsilon_max:.2f}")


        # Tracking weights updates using tracking_weight_controller.py
        # self.agent.weight_controller.save_tracked_history_csv("tracked_neuron_history.csv")
        # self.agent.weight_controller.plot_tracked_conductance()

        # Plot using the configured maximum episodes to avoid referencing a local variable
        self.plot_training(self.max_episodes)

        return self.reward_history


    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                state = self.state_preprocess(state, num_states=self.num_states)
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit()  # close the rendering window

    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./4x4_plots/reward_plot.png', format='png', dpi=600, bbox_inches='tight')