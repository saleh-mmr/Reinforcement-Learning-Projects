import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from marl_meeting_task.src.models.qvalue_network import QValueNetwork
from marl_meeting_task.src.utils.replay_memory import ReplayMemory
from marl_meeting_task.src.config import device


class PS_DQNAgent:
    """
    Parameter-Shared Deep Q-Network Agent Component.
    
    This class contains the shared learning components for PS-DQN:
    - Shared Q-value network (main and target)
    - Shared optimizer
    - Shared experience replay buffer
    - Action selection logic
    - Training step logic
    
    All agents share the same parameters, enabling parameter sharing
    which can improve sample efficiency and generalization.
    """
    
    def __init__(
        self,
        n_agents: int,
        input_dim: int,
        num_actions: int,
        hidden_dim: int,
        learning_rate: float,
        memory_capacity: int,
        gamma: float,
        batch_size: int,
    ):
        """
        Initialize Parameter-Shared DQN Agent.
        
        Parameters:
        -----------
        n_agents : int
            Number of agents (default: 2)
        input_dim : int
            Dimension of observation vector (default: 4)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden layer dimension for Q-network (default: 64)
        learning_rate : float
            Learning rate for optimizer (default: 1e-3)
        memory_capacity : int
            Capacity of shared replay buffer (default: 10000)
        gamma : float
            Discount factor (default: 0.99)
        batch_size : int
            Batch size for training (default: 32)
        """
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Shared Q-network (main)
        self.q_network = QValueNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Shared Q-network (target)
        self.target_network = QValueNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        self.update_target_network()  # Initialize with same weights
        
        # Shared optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Shared replay buffer (stores joint transitions, extracts individual agent data during sampling)
        self.replay_memory = ReplayMemory(capacity=memory_capacity)
    
    # ========================================================================
    # Network Management
    # ========================================================================
    
    def update_target_network(self) -> None:
        """Copy weights from main Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # ========================================================================
    # Action Selection
    # ========================================================================
    
    def select_actions(self, obs: Dict[int, np.ndarray], epsilon: float) -> Dict[int, int]:
        """
        Select actions for all agents using epsilon-greedy policy with shared network.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Observations keyed by agent_id
        epsilon : float
            Exploration probability in [0, 1]
            
        Returns:
        --------
        Dict[int, int]
            Actions keyed by agent_id
        """
        actions = {}
        
        # Ensure epsilon is exactly 0 for greedy policy
        if epsilon <= 0.0:
            # Exploitation: greedy actions (no exploration)
            # Ensure network is in eval mode (should already be set, but double-check)
            if self.q_network.training:
                self.q_network.eval()
            for agent_id in range(self.n_agents):
                with torch.no_grad():
                    # Convert observation to float32 tensor (observations are int64 from env)
                    obs_tensor = torch.as_tensor(
                        obs[agent_id],
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)  # Add batch dimension: (1, input_dim)
                    q_values = self.q_network(obs_tensor)
                    actions[agent_id] = q_values.argmax().item()
        else:
            # Epsilon-greedy: explore with probability epsilon
            for agent_id in range(self.n_agents):
                if np.random.random() < epsilon:
                    # Exploration: random action
                    actions[agent_id] = np.random.randint(0, self.num_actions)
                else:
                    # Exploitation: greedy action using shared network
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(
                            obs[agent_id],
                            dtype=torch.float32,
                            device=device
                        ).unsqueeze(0)  # Add batch dimension: (1, input_dim)
                        q_values = self.q_network(obs_tensor)
                        actions[agent_id] = q_values.argmax().item()
        
        return actions
    
    # ========================================================================
    # Experience Storage
    # ========================================================================
    
    def store_transitions(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        next_obs: Dict[int, np.ndarray],
        reward: float,
        done: bool
    ) -> None:
        """
        Store transitions in shared replay buffer (one joint transition per timestep).
        
        Stores dict-based transitions. During sampling, individual agent transitions
        are extracted and combined to enable parameter sharing across all agents.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Current observations keyed by agent_id
        actions : Dict[int, int]
            Actions taken keyed by agent_id
        next_obs : Dict[int, np.ndarray]
            Next observations keyed by agent_id
        reward : float
            Scalar reward (shared by all agents)
        done : bool
            Whether episode terminated or truncated
        """
        # Store one joint transition per timestep (dict-based)
        self.replay_memory.store(
            state=obs,
            action=actions,
            next_state=next_obs,
            reward=reward,
            done=done
        )
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using shared network.
        
        Extracts individual agent transitions from sampled dict-based transitions
        and combines them to enable parameter sharing across all agents.
        
        Returns:
        --------
        Optional[float]
            Training loss if buffer has enough future, None otherwise
        """
        # Check if buffer has enough future
        if len(self.replay_memory) < self.batch_size:
            return None
        
        # Sample batch from shared replay buffer (returns dicts)
        states_dict, actions_dict, next_states_dict, rewards, dones = \
            self.replay_memory.sample(self.batch_size)
        
        # Extract and flatten individual agent transitions from dicts
        # This enables the shared network to learn from all agent experiences
        observations_list = []
        actions_list = []
        next_observations_list = []
        
        for agent_id in range(self.n_agents):
            observations_list.append(states_dict[agent_id])
            actions_list.append(actions_dict[agent_id])
            next_observations_list.append(next_states_dict[agent_id])
        
        # Concatenate all agent transitions: (batch_size * n_agents, ...)
        observations = torch.cat(observations_list, dim=0)  # (batch_size * n_agents, input_dim)
        actions = torch.cat(actions_list, dim=0)  # (batch_size * n_agents,)
        next_observations = torch.cat(next_observations_list, dim=0)  # (batch_size * n_agents, input_dim)
        
        # Repeat rewards and dones for each agent (same reward/done for all agents in a timestep)
        rewards = rewards.repeat_interleave(self.n_agents, dim=0)  # (batch_size * n_agents,)
        dones = dones.repeat_interleave(self.n_agents, dim=0)  # (batch_size * n_agents,)
        
        # Current Q-values: Q(s, a) for selected actions
        q_values = self.q_network(observations)  # (batch_size * n_agents, num_actions)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size * n_agents,)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_observations)  # (batch_size * n_agents, num_actions)
            next_q_value = next_q_values.max(1)[0]  # (batch_size * n_agents,)
            # Bellman target: noise_realization + γ * max_a' Q_target(s', a') * (1 - done)
            target = rewards + self.gamma * next_q_value * (~dones)
        
        # Compute MSE loss
        loss = nn.MSELoss()(q_value, target)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

