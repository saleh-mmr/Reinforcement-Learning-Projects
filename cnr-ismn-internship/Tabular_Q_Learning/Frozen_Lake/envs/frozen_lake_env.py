from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import gymnasium as gym


@dataclass
class FrozenLakeSpec:
    map_name: str = "4x4"          # "4x4" or "8x8"
    is_slippery: bool = False
    seed: int = 0


class FrozenLakeEnv:
    """
    Thin wrapper around Gym/Gymnasium FrozenLake that also provides a one-hot
    feature encoder φ(s) for linear Q approximators.

    We start here because all later memristor-weight code will plug into:
        Q(s,a) = φ(s)^T θ[:,a]
    """
    def __init__(self, spec):
        self.spec = spec
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=spec.map_name,
            is_slippery=spec.is_slippery,
        )
        # FrozenLake observation is discrete: 0..(S-1)
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)

        # Reproducibility
        self._np_rng = np.random.default_rng(spec.seed)

    def reset(self):
        obs, info = self.env.reset(seed=self.spec.seed)
        s = int(obs)
        return s, self.one_hot(s), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        s2 = int(obs)
        return s2, self.one_hot(s2), float(reward), bool(terminated), bool(truncated), info

    def one_hot(self, s):
        x = np.zeros(self.n_states, dtype=np.float32)
        x[s] = 1.0
        return x

    def sample_action(self):
        # epsilon-greedy will live elsewhere; this is just a helper.
        return int(self.env.action_space.sample())

    def close(self):
        self.env.close()
