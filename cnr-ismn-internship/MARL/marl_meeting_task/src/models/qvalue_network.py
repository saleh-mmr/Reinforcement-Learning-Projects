from torch import nn


class QValueNetwork(nn.Module):
    """
    Q-Value Network for Multi-Agent Reinforcement Learning.
    
    A feedforward neural network that estimates Q-values for state-action pairs.
    Used in Independent Q-Learning (IQL) and other value-based MARL algorithms.
    
    Architecture:
    - Input: observation vector (default: 6 dims)
      [own_x, own_y, other_agent_x, other_agent_y, goal_x, goal_y]
    - Hidden layers: Linear(64) -> ReLU -> Linear(64) -> ReLU
    - Output: Q-values for each action (default: 5 actions)
      [up, down, left, right, stay]
    """
    
    def __init__(self, input_dim=6, num_actions=5, hidden_dim=64):
        """
        Initialize Q-Value Network.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input observation vector (default: 6)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Dimension of hidden layers (default: 64)
        """
        super(QValueNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        """
        Forward pass through the Q-value network.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input observation(s) of shape [batch_size, input_dim]
            
        Returns:
        -------
        torch.Tensor
            Q-values for each action of shape [batch_size, num_actions]
        """
        return self.network(x)

