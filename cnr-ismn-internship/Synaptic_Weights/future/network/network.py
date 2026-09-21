from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 0.76)  # set weights
            nn.init.constant_(layer.bias, 0.56)  # set bias


class DQNNetwork(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQNNetwork, self).__init__()

        # Fully Connected (FC) model
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.LeakyReLU(negative_slope=0.01),          # LeakyReLU activation function helps learn non-linear patterns.

            nn.Linear(10, 10),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(10, num_actions)        # [Q_left, Q_right]  → choose max action
        )

        # Apply custom initialization
        # self.FC.apply(init_weights)

    def forward(self, x):
        """
        Forward pass through the Q-network.

        Parameters:
        ----------
        x : Tensor
            Input state(s) as a tensor [batch_size, input_dim]

        Returns:
        -------
        Q-values for each possible action [batch_size, num_actions]
        """
        Q = self.FC(x)
        return Q                                    # Q = [Q(action=left), Q(action=right)]
