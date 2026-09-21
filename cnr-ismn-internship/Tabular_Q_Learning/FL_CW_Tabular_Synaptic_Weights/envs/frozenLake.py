import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.config as config
from .mapGenerator import MapGenerator


import numpy as np
import gymnasium as gym


class FrozenLakeEnv:
    def __init__(self):
        map_generator = MapGenerator(
            height=4,
            width=12,
            p=0.8,
        )
        custom_map = map_generator.generate_random_map_rect()
        self.env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)
        self.rng = np.random.default_rng(config.seed)


    def reset(self):
        obs, info = self.env.reset(seed=config.seed)
        s = int(obs)
        return s, self.one_hot(s), info

    def one_hot(self, s):
        x = np.zeros(self.n_states, dtype=np.float32)
        x[s] = 1.0
        return x

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        next_state = int(obs)
        return (
            next_state,
            self.one_hot(next_state),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def sample_action(self):
        # epsilon-greedy will live elsewhere; this is just a helper.
        return int(self.env.action_space.sample())

    def close(self):
        self.env.close()