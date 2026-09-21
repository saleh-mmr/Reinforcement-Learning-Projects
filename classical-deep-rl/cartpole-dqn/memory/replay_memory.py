import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import deque
import numpy as np
import torch
from utils import config


class ReplayMemory:
    """Fixed-size replay buffer used to sample decorrelated training batches."""

    def __init__(self, capacity):
        """Create replay memory with FIFO eviction when capacity is reached."""
        self.capacity = capacity

        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        """Store one transition tuple ``(s, a, s', r, done)``."""
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """Sample a random mini-batch and return tensors on ``config.device``."""
        # Sample unique indices so the same transition is not repeated in one batch.
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.as_tensor(
            np.array([self.states[i] for i in indices]),
            dtype=torch.float32,
            device=config.device
        )

        next_states = torch.as_tensor(
            np.array([self.next_states[i] for i in indices]),
            dtype=torch.float32,
            device=config.device
        )

        actions = torch.as_tensor(
            [self.actions[i] for i in indices],
            dtype=torch.long,
            device=config.device
        )

        rewards = torch.as_tensor(
            [self.rewards[i] for i in indices],
            dtype=torch.float32,
            device=config.device
        )

        dones = torch.as_tensor(
            [self.dones[i] for i in indices],
            dtype=torch.bool,
            device=config.device
        )

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """Return the number of currently stored transitions."""
        return len(self.dones)
