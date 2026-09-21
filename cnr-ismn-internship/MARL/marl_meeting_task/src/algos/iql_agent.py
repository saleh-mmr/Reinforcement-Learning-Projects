import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from marl_meeting_task.src.models.qvalue_network import QValueNetwork
from marl_meeting_task.src.utils.replay_memory import ReplayMemory
from marl_meeting_task.src.config import device


class IQLAgent:
    """
    Independent Q-Learning Agent.
    
    Each agent maintains its own:
    - Q-value network (main and target)
    - Optimizer
    - Experience replay buffer
    
    This enables independent learning where each agent treats other agents
    as part of the environment, learning its own Q-function.
    """
    
    def __init__(
        self,
        agent_id: int,
        input_dim: int,
        num_actions: int,
        hidden_dim: int,
        learning_rate: float,
        memory_capacity: int,
    ):
        """
        Initialize an independent Q-learning agent.
        
        Parameters:
        -----------
        agent_id : int
            Unique identifier for this agent
        input_dim : int
            Dimension of observation vector (default: 4)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden layer dimension for Q-network (default: 64)
        learning_rate : float
            Learning rate for optimizer (default: 1e-3)
        memory_capacity : int
            Capacity of replay buffer (default: 10000)
        """
        self.agent_id = agent_id
        self.num_actions = num_actions
        
        # Main Q-value network
        self.q_network = QValueNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Target Q-value network (for stable learning)
        self.target_network = QValueNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        self.update_target_network()  # Initialize with same weights
        
        # Optimizer for main Q-network
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.replay_memory = ReplayMemory(capacity=memory_capacity)
    
    # ========================================================================
    # Network Management
    # ========================================================================
    
    def update_target_network(self) -> None:
        """
        Copy weights from main Q-network to target network.
        
        This is called periodically during training to stabilize learning
        by using a fixed target network for computing Q-targets.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # ========================================================================
    # Action Selection
    # ========================================================================
    
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon, selects a random action (exploration).
        Otherwise, selects the action with highest Q-value (exploitation).
        
        Parameters:
        -----------
        obs : np.ndarray
            Observation vector of shape (input_dim,)
        epsilon : float
            Exploration probability in [0, 1]
            
        Returns:
        --------
        int
            Selected action index
        """
        # Ensure epsilon is exactly 0 for greedy policy
        if epsilon <= 0.0:
            # Exploitation: greedy action (no exploration)
            with torch.no_grad():
                # Convert observation to float32 tensor (observations are int64 from env)
                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)  # Add batch dimension: (1, input_dim)
                # Ensure network is in eval mode (should already be set, but double-check)
                if self.q_network.training:
                    self.q_network.eval()
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()
        else:
            # Epsilon-greedy: explore with probability epsilon
            if np.random.random() < epsilon:
                # Exploration: random action
                return np.random.randint(0, self.num_actions)
            else:
                # Exploitation: greedy action
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)  # Add batch dimension: (1, input_dim)
                    q_values = self.q_network(obs_tensor)
                    return q_values.argmax().item()
    
    # ========================================================================
    # Learning
    # ========================================================================
    
    def learn(self, batch_size: int, gamma: float = 0.99) -> Optional[float]:
        """
        Perform one gradient update step using a batch from replay buffer.
        
        Implements the DQN loss with target network:
        L = (Q(s, a) - (noise_realization + γ * max_a' Q_target(s', a') * (1 - done)))^2
        
        Parameters:
        -----------
        batch_size : int
            Number of transitions to sample from replay buffer
        gamma : float
            Discount factor for future rewards (default: 0.99)
            
        Returns:
        --------
        Optional[float]
            Training loss if buffer has enough future, None otherwise
        """
        # Check if buffer has enough future
        if len(self.replay_memory) < batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        
        # Extract this agent's data
        agent_states = states[self.agent_id]  # (batch_size, input_dim)
        agent_actions = actions[self.agent_id]  # (batch_size,)
        agent_next_states = next_states[self.agent_id]  # (batch_size, input_dim)
        
        # Current Q-values: Q(s, a) for selected actions
        q_values = self.q_network(agent_states)  # (batch_size, num_actions)
        q_value = q_values.gather(1, agent_actions.unsqueeze(1)).squeeze(1)  # (batch_size,)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(agent_next_states)  # (batch_size, num_actions)
            next_q_value = next_q_values.max(1)[0]  # (batch_size,)
            # Bellman target: noise_realization + γ * max_a' Q_target(s', a') * (1 - done)
            target = rewards + gamma * next_q_value * (~dones)
        
        # Compute MSE loss
        loss = nn.MSELoss()(q_value, target)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

