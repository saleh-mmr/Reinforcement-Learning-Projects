from collections import deque
import numpy as np
import torch
from marl_meeting_task.src.config import device


class ReplayMemory:
    """
    Experience Replay memory storing past transitions for off-policy learning.
    - Improves sample efficiency by reusing experiences
    - Stabilizes training compared to learning only from latest transition
    - deques is an efficient structure for sliding memory buffer
    
    Adapted for MARL: handles dict-based observations and actions keyed by agent id.
    """

    def __init__(self, capacity):
        """
        capacity : int
            Maximum memory size. Old future are removed automatically once capacity is exceeded.
        """
        self.capacity = capacity

        self.states = deque(maxlen=capacity)              # Current observation dict {agent_id: np.array}
        self.actions = deque(maxlen=capacity)             # Action dict {agent_id: action}
        self.next_states = deque(maxlen=capacity)         # Next observation dict {agent_id: np.array}
        self.rewards = deque(maxlen=capacity)             # Scalar reward
        self.dones = deque(maxlen=capacity)               # TRUE = episode ended (terminated or truncated)

    def store(self, state, action, next_state, reward, done):
        """
        Store a new transition into the replay memory.
        FIFO behavior: if full, oldest transitions are automatically discarded.
        
        Args:
            state: dict {agent_id: np.array} - current observation
            action: dict {agent_id: int} - actions taken
            next_state: dict {agent_id: np.array} - next observation
            reward: float - scalar reward
            done: bool - whether episode ended
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
        
        For MARL, returns dicts of tensors keyed by agent id.
        
        Returns:
            states: dict {agent_id: torch.Tensor} - batched states
            actions: dict {agent_id: torch.Tensor} - batched actions
            next_states: dict {agent_id: torch.Tensor} - batched next states
            rewards: torch.Tensor - batched rewards
            dones: torch.Tensor - batched done flags
        """
        assert len(self) > 0, "Cannot sample from empty replay memory"
        
        # Generate random indices, use replace=True if memory is smaller than batch_size
        replace = len(self) < batch_size
        indices = np.random.choice(len(self), size=batch_size, replace=replace)

        # Get agent ids from first state (assuming all states have same agent ids)
        agent_ids = list(self.states[0].keys())
        
        # Convert selected transitions into batched torch tensors per agent
        states = {}
        actions = {}
        next_states = {}
        
        for agent_id in agent_ids:
            # States for this agent
            agent_states = [self.states[i][agent_id] for i in indices]
            states[agent_id] = torch.as_tensor(
                np.array(agent_states),
                dtype=torch.float32,
                device=device
            )
            
            # Next states for this agent
            agent_next_states = [self.next_states[i][agent_id] for i in indices]
            next_states[agent_id] = torch.as_tensor(
                np.array(agent_next_states),
                dtype=torch.float32,
                device=device
            )
            
            # Actions for this agent
            agent_actions = [self.actions[i][agent_id] for i in indices]
            actions[agent_id] = torch.as_tensor(
                agent_actions,
                dtype=torch.long,
                device=device
            )

        # Rewards and dones are scalar/shared
        rewards = torch.as_tensor(
            [self.rewards[i] for i in indices],
            dtype=torch.float32,
            device=device
        )

        # Done flags are boolean tensors
        dones = torch.as_tensor(
            [self.dones[i] for i in indices],
            dtype=torch.bool,
            device=device
        )

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        Current size of memory (number of stored transitions).
        """
        return len(self.dones)

