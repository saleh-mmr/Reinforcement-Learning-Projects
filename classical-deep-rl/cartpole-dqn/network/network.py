from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DQNNetwork(nn.Module):
    """Small feed-forward network that maps state vectors to action Q-values."""

    def __init__(self, num_actions, input_dim, network_size):
        super(DQNNetwork, self).__init__()

        # Two hidden layers are enough for baseline CartPole performance.
        self.FC = nn.Sequential(
            nn.Linear(input_dim, network_size),
            nn.ReLU(),

            nn.Linear(network_size, network_size),
            nn.ReLU(),

            nn.Linear(network_size, num_actions)
        )

    def forward(self, x):
        """Return per-action Q-values for a batch of states."""
        q_values = self.FC(x)
        return q_values
