from torch import nn


class DQNNetwork(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQNNetwork, self).__init__()

        # Fully Connected (FC) model
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LeakyReLU(negative_slope=0.01),    # Relu activation function helps learn non-linear patterns.

            nn.Linear(48, 48),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(48, num_actions)        # [Q_left, Q_right]  → choose max action
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
