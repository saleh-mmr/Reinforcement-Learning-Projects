import warnings
import os
import gymnasium as gym
import numpy as np
import pygame
import torch
from matplotlib import pyplot as plt

from config import device, seed
from dqn_agent import DQNAgent
from step_wrapper import StepWrapper


class ModelTrainTest():
    def __init__(self, hyperparams):

        # Define RL Hyperparameters
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]

        self.memory_capacity = hyperparams["memory_capacity"]

        self.render_fps = hyperparams["render_fps"]

        # Define Env
        self.env = gym.make('MountainCar-v0', max_episode_steps=self.max_steps,
                            render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps  # For max frame rate make it 0

        """
        The 'MountainCar-v0' environment in the 'gymnasium' library generates 
        UserWarnings about deprecated methods.
        These warnings are related to the 'size' and 'shape' methods which are 
        being phased out in a future version of the library.
        Even though we are not directly using these methods in our code, the 
        warnings are still displayed.
        To keep our output clean and focused on our own program's execution, 
        the following line of code is for ignoring these warnings.
        """
        warnings.filterwarnings("ignore", category=UserWarning)

        # Apply RewardWrapper
        self.env = StepWrapper(self.env)

        # Define the agent class
        self.agent = DQNAgent(env=self.env,
                               epsilon_max=self.epsilon_max,
                               epsilon_min=self.epsilon_min,
                               epsilon_decay=self.epsilon_decay,
                               clip_grad_norm=self.clip_grad_norm,
                               learning_rate=self.learning_rate,
                               discount=self.discount_factor,
                               memory_capacity=self.memory_capacity)

        os.makedirs('plots', exist_ok=True)

    def train(self):
        """
        Reinforcement learning training loop.
        """

        total_steps = 0
        self.reward_history = []

        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):

            # Decide if we should render this episode
            render_this = (episode <= 1) or (episode > self.max_episodes - 10)

            # Re-create environment based on rendering choice
            self.env = gym.make('MountainCar-v0',
                                max_episode_steps=self.max_steps,
                                render_mode="human" if render_this else None)
            self.env.metadata['render_fps'] = self.render_fps
            self.env = StepWrapper(self.env)

            # Reset environment
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                self.agent.replay_memory.store(state, action, next_state, reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            # Appends for tracking history
            self.reward_history.append(episode_reward)  # episode reward
            total_steps += step_size

            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

            # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode == self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.2f}")
            print(result)
        self.plot_training(episode)

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

        # Clip max (high) values for better plot analysis
        reward_history = np.clip(self.reward_history, a_min=None, a_max=100)
        sma = np.clip(sma, a_min=None, a_max=100)

        plt.figure()
        plt.title("Obtained Rewards")
        plt.plot(reward_history, label='Raw Reward', color='#4BA754', alpha=1)
        plt.plot(sma, label='SMA 50', color='#F08100')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('plots/reward_plot.png', format='png', dpi=600, bbox_inches='tight')

        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        # Loss plot
        plt.figure()
        plt.title("Network Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#8921BB', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        if episode == self.max_episodes:
            plt.savefig('plots/Loss_plot.png', format='png', dpi=600, bbox_inches='tight')

        plt.tight_layout()
        plt.grid(True)
        plt.show()
