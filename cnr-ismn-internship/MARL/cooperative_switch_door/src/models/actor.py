# src/models/actor.py
import torch
from torch import nn
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """
    MAPPO Actor (policy network)
    - Input: observation (obs_dim=7)
    - Output: action logits (n_actions=5)
    """

    def __init__(self, obs_dim=7, n_actions=5, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        """
        obs: [batch, obs_dim]
        returns logits: [batch, n_actions]
        """
        return self.net(obs)

    def get_dist(self, obs):
        # Network outputs scores (logits), e.g. [-0.8, 0.4, 1.6, -0.2, 0.1]
        logits = self.forward(obs)
        return Categorical(logits=logits)

    @torch.no_grad()
    def sample_action(self, obs):
        """
        For rollout collection.
        Returns:
          action: [batch]
          log_prob: [batch]
          entropy: [batch]
        """
        # Get action distribution. e.g.	[action_0 = 5% , action_1 = 17%, action_2 = 57%, action_3 = 9%, action_4 = 12%]
        dist = self.get_dist(obs)

        # Pick an action. Actions with higher prob are more likely to be picked. This is exploration.
        action = dist.sample()

        # How confident is the policy about this action? PPO needs this to learn.
        log_prob = dist.log_prob(action)

        # How uncertain is the policy? Higher entropy = more exploration.
        entropy = dist.entropy()

        # action   = chosen action (e.g. 2)
        # log_prob = confidence of that choice
        # entropy  = randomness level
        return action, log_prob, entropy
