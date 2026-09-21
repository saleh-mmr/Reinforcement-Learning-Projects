import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium import spaces


class MountainCarEnv:
    """
    Modified MountainCar:
    - 4D observation: [position, velocity, sin(position), cos(position)]
    - 2 discrete actions: 0=left, 1=right
    """

    def __init__(self, render_mode=None, seed=None):
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)

        # New 2-action space
        self._action_space = spaces.Discrete(2)

        # New 4D observation space
        low = np.array([-1.2, -0.07, -1.0, -1.0], dtype=np.float32)
        high = np.array([0.6, 0.07, 1.0, 1.0], dtype=np.float32)
        self._observation_space = spaces.Box(low, high, dtype=np.float32)

        self.env.reset(seed=seed)
        self._action_space.seed(seed)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @staticmethod
    def transform_state(state):
        position, velocity = state
        return np.array([
            position,
            velocity,
            np.sin(position),
            np.cos(position)
        ], dtype=np.float32)

    def reset(self):
        state, _ = self.env.reset()
        return self.transform_state(state)

    def step(self, action):
        # Map 2 actions → original 3 actions
        mapped_action = 0 if action == 0 else 2

        next_state, reward, terminated, truncated, _ = self.env.step(mapped_action)
        if terminated:
            reward = 100
        return self.transform_state(next_state), reward, terminated, truncated

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
