import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# Add parent directory to path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MyCartPoleEnv:
    def __init__(self, render_mode=None, seed=None, max_steps=100, pole_length=0.7, pole_mass=0.2):
        # Initialize the standard environment
        self.env = TimeLimit(
            gym.make("CartPole-v1", render_mode=render_mode),
            max_episode_steps=max_steps
        )

        # Access the internal physics parameters
        unwrapped = self.env.unwrapped
        unwrapped.length = pole_length      # Increase pole length - defaults is 0.5
        unwrapped.masspole = pole_mass   # Increase pole mass -  defaults is 0.1

        # recompute dependent values
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

        # Set seeds for reproducibility
        if seed is not None:
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        # Gymnasium reset returns (state, info)
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        # Combine flags for a simpler 'done' signal
        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()