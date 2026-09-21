import numpy as np
import torch
from typing import Optional
from marl_meeting_task.src.models.qvalue_network import QValueNetwork
from marl_meeting_task.src.config import device


class VDNAgent:
    """
    VDN Agent Component.

    Each agent has its own Q-network that takes local observations
    and outputs Q-values for actions. Execution is decentralized.
    """

    def __init__(
        self,
        agent_id: int,
        n_agents: int,
        input_dim: int,   # Base observation dim (will have agent_id appended)
        num_actions: int,
        hidden_dim: int,
    ):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.base_input_dim = input_dim
        # Final input dim is cartpole + one-hot agent id
        self.input_dim = input_dim + n_agents
        self.num_actions = num_actions

        # Agent Q-network (main)
        self.q_network = QValueNetwork(
            input_dim=self.input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)

        # Agent Q-network (target)
        self.target_network = QValueNetwork(
            input_dim=self.input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        self.update_target_network()

    def update_target_network(self) -> None:
        """Copy weights from main Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _append_agent_id(self, obs: np.ndarray) -> np.ndarray:
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
        agent_id_onehot = np.zeros(self.n_agents, dtype=np.float32)
        agent_id_onehot[self.agent_id] = 1.0
        obs_with_id = np.concatenate([obs, agent_id_onehot])
        return obs_with_id

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        obs_with_id = self._append_agent_id(obs)

        # Exploitation (greedy)
        if epsilon <= 0.0:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(
                    obs_with_id,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                if self.q_network.training:
                    self.q_network.eval()
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()
        else:
            if np.random.random() < epsilon:
                return np.random.randint(0, self.num_actions)
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(
                        obs_with_id,
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)
                    q_values = self.q_network(obs_tensor)
                    return q_values.argmax().item()

