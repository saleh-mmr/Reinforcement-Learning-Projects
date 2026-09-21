from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.config as config
import numpy as np
import gymnasium as gym



class CliffWalkingEnv:
    """
    Thin wrapper around Gymnasium CliffWalking.
    Provides one-hot feature encoder φ(s) for tabular / linear Q.

        Q(s,a) = φ(s)^T θ[:,a]
    """

    def __init__(self, seed):
        self.env = gym.make("CliffWalking-v1")
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)
        if seed is None:
            config_seed = config.get_default_seed()
            self._np_rng = np.random.default_rng(config_seed)
        else:
            self._np_rng = np.random.default_rng(seed)

    def reset(self):
        config_seed = config.get_default_seed()
        obs, info = self.env.reset(seed=config_seed)
        s = int(obs)
        return s, self.one_hot(s), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        s2 = int(obs)
        return (
            s2,
            self.one_hot(s2),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def one_hot(self, s: int) -> np.ndarray:
        x = np.zeros(self.n_states, dtype=np.float32)
        x[s] = 1.0
        return x

    def sample_action(self) -> int:
        return int(self.env.action_space.sample())

    def close(self):
        self.env.close()
