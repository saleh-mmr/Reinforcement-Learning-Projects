import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    """
    QMIX Mixing Network.
    
    Combines individual agent Q-values into a joint Q-value Q_tot.
    Enforces monotonicity constraint: ∂Q_tot/∂Q_i >= 0 for all i.
    
    Architecture:
    - Takes agent Q-values [Q_1, Q_2, ...] and global state s
    - Uses hypernetworks to generate positive weights from state
    - Mixes Q-values with state-dependent positive weights
    - Output: Q_tot(s, a_1, a_2, ...)
    
    Reference:
    Rashid, T., et al. (2018). QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning. ICML.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        state_dim: int = 6,  # Global state: [a1_x, a1_y, a2_x, a2_y, g_x, g_y]
        mixing_hidden_dim: int = 64,
    ):
        """
        Initialize QMIX Mixing Network.
        
        Parameters:
        -----------
        n_agents : int
            Number of agents (default: 2)
        state_dim : int
            Dimension of global state (default: 6)
        mixing_hidden_dim : int
            Hidden dimension for mixing network (default: 64)
        """
        super(MixingNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        
        # Hypernetworks: generate mixing weights from state
        # These ensure monotonicity by producing positive weights
        
        # First layer hypernetwork: state -> [n_agents, mixing_hidden_dim] weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, n_agents * mixing_hidden_dim)
        )
        
        # First layer hypernetwork: state -> [mixing_hidden_dim] bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim)
        )
        
        # Second layer hypernetwork: state -> [mixing_hidden_dim, 1] weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim)
        )
        
        # Second layer hypernetwork: state -> [1] bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1)
        )
    
    def forward(self, agent_qs, states):
        """
        Forward pass through mixing network.
        
        Parameters:
        -----------
        agent_qs : torch.Tensor
            Individual agent Q-values of shape [batch_size, n_agents]
        states : torch.Tensor
            Global states of shape [batch_size, state_dim]
            
        Returns:
        --------
        torch.Tensor
            Joint Q-value Q_tot of shape [batch_size, 1]
        """
        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.unsqueeze(1)  # [batch_size, 1, n_agents]
        
        # Generate positive weights using hypernetworks and softplus
        w1 = F.softplus(self.hyper_w1(states))  # [batch_size, n_agents * mixing_hidden_dim]
        w1 = w1.view(batch_size, self.n_agents, self.mixing_hidden_dim)
        
        b1 = self.hyper_b1(states)  # [batch_size, mixing_hidden_dim]
        b1 = b1.unsqueeze(1)  # [batch_size, 1, mixing_hidden_dim]
        
        # First mixing layer
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [batch_size, 1, mixing_hidden_dim]
        
        # Second layer
        w2 = F.softplus(self.hyper_w2(states))  # [batch_size, mixing_hidden_dim]
        w2 = w2.view(batch_size, self.mixing_hidden_dim, 1)
        
        b2 = self.hyper_b2(states)  # [batch_size, 1]
        
        # Final mixing
        q_tot = torch.bmm(hidden, w2) + b2.unsqueeze(1)  # [batch_size, 1, 1]
        q_tot = q_tot.squeeze(1)  # [batch_size, 1]
        
        return q_tot
