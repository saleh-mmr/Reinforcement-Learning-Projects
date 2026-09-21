from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from utils import config


class FrozenLakeEnv:
    """
    Thin wrapper around Gym/Gymnasium FrozenLake that also provides a one-hot
    feature encoder φ(s) for linear Q approximators.

    We start here because all later memristor-weight code will plug into:
        Q(s,a) = φ(s)^T θ[:,a]
    """
    def __init__(self):
        self.env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=False,
        )
        # FrozenLake observation is discrete: 0..(S-1)
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)

        # Reproducibility
        self._np_rng = np.random.default_rng(config.seed)


    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        obs, _ = self.env.reset(seed=config.seed)
        s = int(obs)
        return self.one_hot(s)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        s2 = int(obs)
        done = terminated or truncated
        return self.one_hot(s2), float(reward), done

    def one_hot(self, s):
        x = np.zeros(self.n_states, dtype=np.float32)
        x[s] = 1.0
        return x

    def sample_action(self):
        # epsilon-greedy will live elsewhere; this is just a helper.
        return int(self.env.action_space.sample())

    def close(self):
        self.env.close()
