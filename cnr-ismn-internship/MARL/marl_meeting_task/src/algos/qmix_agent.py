import numpy as np
import torch
import torch.nn as nn
from marl_meeting_task.src.models.qvalue_network import QValueNetwork
from marl_meeting_task.src.config import device


class QMIXAgent:
    """
    QMIX Agent Component.
    
    Each agent has its own Q-network that takes local observations
    and outputs Q-values for actions. These are used during execution
    (decentralized) and combined via the mixing network during training
    (centralized).
    """
    
    def __init__(
        self,
        agent_id,
        n_agents,
        input_dim,   # Base observation dim (will have agent_id appended)
        num_actions,
        hidden_dim,
    ):
        """
        Initialize QMIX Agent.
        
        Parameters:
        -----------
        agent_id : int
            Unique identifier for this agent
        n_agents : int
            Total number of agents (for one-hot encoding)
        input_dim : int
            Dimension of cartpole observation vector (default: 4)
            Note: agent_id will be appended as one-hot, so final input_dim = input_dim + n_agents
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden layer dimension for Q-network (default: 64)
        """
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.base_input_dim = input_dim  # Base observation dimension
        self.input_dim = input_dim + n_agents  # Final input dim (cartpole + one-hot agent_id)
        self.num_actions = num_actions
        
        # Agent Q-network (main)
        # Note: input_dim includes cartpole observation + one-hot agent_id
        self.q_network = QValueNetwork(
            input_dim=self.input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Agent Q-network (target)
        # Note: input_dim includes cartpole observation + one-hot agent_id
        self.target_network = QValueNetwork(
            input_dim=self.input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        self.update_target_network()  # Initialize with same weights
    
    def update_target_network(self):
        """Copy weights from main Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _append_agent_id(self, obs):
        """
        Append one-hot agent ID to observation.
        
        Parameters:
        -----------
        obs : np.ndarray
            Base observation vector of shape (base_input_dim,)
            
        Returns:
        --------
        np.ndarray
            Observation with one-hot agent ID appended, shape (input_dim,)
        """
        # Create one-hot encoding for agent_id
        agent_id_onehot = np.zeros(self.n_agents, dtype=np.float32)
        agent_id_onehot[self.agent_id] = 1.0
        
        # Concatenate cartpole observation with one-hot agent_id
        obs_with_id = np.concatenate([obs, agent_id_onehot])
        return obs_with_id
    
    def select_action(self, obs, epsilon):
        """
        Select action using epsilon-greedy policy.
        
        Parameters:
        -----------
        obs : np.ndarray
            Local observation vector
        epsilon : float
            Exploration probability
            
        Returns:
        --------
        int
            Selected action index
        """
        # Append one-hot agent ID to observation
        obs_with_id = self._append_agent_id(obs)
        
        # Ensure epsilon is exactly 0 for greedy policy
        if epsilon <= 0.0:
            # Exploitation: greedy action (no exploration)
            with torch.no_grad():
                # Convert observation to float32 tensor (observations are int64 from env)
                obs_tensor = torch.as_tensor(
                    obs_with_id,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)  # Add batch dimension
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
                        obs_with_id,
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)  # Add batch dimension
                    q_values = self.q_network(obs_tensor)
                    return q_values.argmax().item()
