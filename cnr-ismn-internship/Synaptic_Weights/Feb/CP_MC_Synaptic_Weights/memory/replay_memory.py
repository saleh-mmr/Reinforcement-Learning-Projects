import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import deque
import numpy as np
import torch
from utils import config

class ReplayMemory:
    """
    Experience Replay buffer storing past transitions for off-policy learning.
    - Improves sample efficiency by reusing experiences
    - Stabilizes training compared to learning only from latest transition
    - deques is an efficient structure for sliding memory buffer
    """

    def __init__(self, capacity):
        """
        capacity : int
            Maximum buffer size. Old future are removed automatically once capacity is exceeded.
        """
        self.capacity = capacity

        self.states = deque(maxlen=capacity)             # Current observation [cart_pos, cart_vel, angle, ang_vel]
        self.actions = deque(maxlen=capacity)            # 0 = left, 1 = right
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)              # TRUE = episode ended due to failure

    def store(self, state, action, next_state, reward, done):
        """
        Store a new transition into the replay buffer.
        FIFO behavior: if full, oldest transitions are automatically discarded.
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions for training.
        Returns tensors directly on the correct device (CPU/GPU).
        """
        # Generate random unique indices,  replace=False → prevents picking same memory twice
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        # Convert selected transitions into batched torch tensors

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

        # Done flags are boolean tensors
        dones = torch.as_tensor(
            [self.dones[i] for i in indices],
            dtype=torch.bool,
            device=config.device
        )


        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        Current size of memory (number of stored transitions).
        """
        return len(self.dones)
