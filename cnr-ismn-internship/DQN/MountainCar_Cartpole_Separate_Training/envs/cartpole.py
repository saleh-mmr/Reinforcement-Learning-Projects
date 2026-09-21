import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym


class CartPoleEnv:
    def __init__(self, render_mode=None, seed=None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.seed = seed

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
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()