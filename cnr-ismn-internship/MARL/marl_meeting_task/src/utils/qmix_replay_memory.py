from collections import deque
import numpy as np
import torch
from marl_meeting_task.src.config import device


class QMIXReplayMemory:
    """
    Centralized Experience Replay Memory for QMIX.
    
    Stores transitions with:
    - Global state s
    - Individual agent observations o_i
    - Individual agent actions a_i
    - Joint reward noise_realization
    - Next global state s_next
    - Next individual agent observations o_i_next
    - Done flag
    
    This is different from standard replay memory which stores
    dict-based observations. QMIX needs explicit global state.
    """
    
    def __init__(self, capacity):
        """
        Initialize QMIX Replay Memory.
        
        Parameters:
        -----------
        capacity : int
            Maximum number of transitions to store
        """
        self.capacity = capacity
        
        # Centralized storage
        self.states = deque(maxlen=capacity)              # Global state s: np.array
        self.observations = deque(maxlen=capacity)        # Dict {agent_id: np.array} - local obs
        self.actions = deque(maxlen=capacity)             # Dict {agent_id: int} - actions
        self.rewards = deque(maxlen=capacity)              # Scalar reward
        self.next_states = deque(maxlen=capacity)          # Next global state s_next
        self.next_observations = deque(maxlen=capacity)     # Dict {agent_id: np.array} - next local obs
        self.dones = deque(maxlen=capacity)                # Done flag
    
    def store(
        self,
        state,
        observations,
        actions,
        reward,
        next_state,
        next_observations,
        done
    ):
        """
        Store a transition.
        
        Deep-copies all inputs to prevent silent mutation that destroys training signal.
        
        Parameters:
        -----------
        state : np.ndarray
            Global state s (shape: state_dim)
        observations : dict
            Local observations {agent_id: np.array}
        actions : dict
            Actions taken {agent_id: int}
        reward : float
            Joint reward
        next_state : np.ndarray
            Next global state s_next
        next_observations : dict
            Next local observations {agent_id: np.array}
        done : bool
            Whether episode ended
        """
        # Deep-copy to prevent reference mutation
        self.states.append(state.copy())
        self.observations.append({agent_id: obs.copy() for agent_id, obs in observations.items()})
        self.actions.append(actions.copy())  # Dict copy (shallow is fine for ints)
        self.rewards.append(reward)
        self.next_states.append(next_state.copy())
        self.next_observations.append({agent_id: obs.copy() for agent_id, obs in next_observations.items()})
        self.dones.append(done)
    
    def sample(self, batch_size, n_agents):
        """
        Sample a batch of transitions.
        
        Parameters:
        -----------
        batch_size : int
            Number of transitions to sample
        n_agents : int
            Number of agents
            
        Returns:
        --------
        tuple
            (states, observations, actions, rewards, next_states, next_observations, dones)
            All as tensors on the correct device
        """
        assert len(self) > 0, "Cannot sample from empty replay memory"
        
        # Generate random indices
        replace = len(self) < batch_size
        indices = np.random.choice(len(self), size=batch_size, replace=replace)
        
        # Convert to tensors
        states = torch.as_tensor(
            np.array([self.states[i] for i in indices]),
            dtype=torch.float32,
            device=device
        )  # [batch_size, state_dim]
        
        # Observations: dict of tensors
        observations = {}
        for agent_id in range(n_agents):
            observations[agent_id] = torch.as_tensor(
                np.array([self.observations[i][agent_id] for i in indices]),
                dtype=torch.float32,
                device=device
            )  # [batch_size, obs_dim]
        
        # Actions: dict of tensors
        actions = {}
        for agent_id in range(n_agents):
            actions[agent_id] = torch.as_tensor(
                [self.actions[i][agent_id] for i in indices],
                dtype=torch.long,
                device=device
            )  # [batch_size]
        
        # Rewards
        rewards = torch.as_tensor(
            [self.rewards[i] for i in indices],
            dtype=torch.float32,
            device=device
        )  # [batch_size]
        
        # Next states
        next_states = torch.as_tensor(
            np.array([self.next_states[i] for i in indices]),
            dtype=torch.float32,
            device=device
        )  # [batch_size, state_dim]
        
        # Next observations: dict of tensors
        next_observations = {}
        for agent_id in range(n_agents):
            next_observations[agent_id] = torch.as_tensor(
                np.array([self.next_observations[i][agent_id] for i in indices]),
                dtype=torch.float32,
                device=device
            )  # [batch_size, obs_dim]
        
        # Dones
        dones = torch.as_tensor(
            [self.dones[i] for i in indices],
            dtype=torch.bool,
            device=device
        )  # [batch_size]
        
        return states, observations, actions, rewards, next_states, next_observations, dones
    
    def __len__(self):
        """Return current size of replay buffer."""
        return len(self.states)
