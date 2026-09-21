import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium.wrappers import TimeLimit


class CartPoleEnv:
    """Thin wrapper around Gymnasium CartPole with optional seeding and time limit."""

    def __init__(self, render_mode=None, seed=None, max_steps=200):
        """Create a CartPole-v1 environment capped at ``max_steps`` per episode."""
        self.env = TimeLimit(
            gym.make("CartPole-v1", render_mode=render_mode),
            max_episode_steps=max_steps
        )
        # Seed both env reset and action space for reproducible rollouts.
        if seed is not None:
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)

    @property
    def action_space(self):
        """Expose action-space metadata."""
        return self.env.action_space

    @property
    def observation_space(self):
        """Expose observation-space metadata."""
        return self.env.observation_space

    def reset(self):
        """Reset the environment and return only the state array."""
        state, _ = self.env.reset()
        return state

    def step(self, action):
        """Apply an action and merge terminated/truncated into a single ``done`` flag."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        """Render one frame using Gymnasium's configured mode."""
        self.env.render()

    def close(self):
        """Release environment resources."""
        self.env.close()