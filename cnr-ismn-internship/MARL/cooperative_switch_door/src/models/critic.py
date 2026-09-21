from torch import nn


class CriticNetwork(nn.Module):


    def __init__(self, input_dim=8, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

