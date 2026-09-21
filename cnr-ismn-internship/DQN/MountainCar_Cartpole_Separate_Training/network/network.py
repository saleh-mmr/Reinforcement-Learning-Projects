from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DQNNetwork(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQNNetwork, self).__init__()

        # Fully Connected (FC) model
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.LeakyReLU(negative_slope=0.01),          # LeakyReLU activation function helps learn non-linear patterns.

            nn.Linear(80, 80),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(80, num_actions)        # [Q_left, Q_right]  → choose max action
        )

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
