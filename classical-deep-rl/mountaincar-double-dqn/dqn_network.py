import torch.nn as nn

class DQNNetwork(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    This network consist of Fully Connected (FC) layers with ReLU activation Functions.
    """


    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.
        Parameters:
            num_actions (int): The number of possible actions in the environment.
            input_dim (int): The dimensionality of the input state space.
        """
        super(DQNNetwork, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions)
        )

        # Initialize the FC layer weights using He Initialization
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of the network to find the Q_values of the actions.
        Parameters:
            x (torch.Tensor): Input tensor representing the state.
        Returns:
            Q (torch.Tensor): Tensor containing Q-values for each action.
        """
        Q = self.FC(x)
        return Q