import numpy as np
import torch


class RolloutBuffer:
    """
    The rollout buffer is just a memory box where we store what happened at each step so PPO can learn later.
    """
    def __init__(self, obs_dim, state_dim, n_agents, buffer_size):
        """
        Initialize the rollout buffer.
        """
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.buffer_size = buffer_size

        self.obs = None         # observations per agent per step
        self.states = None      # global states per step
        self.actions = None     # actions taken by each agent
        self.log_probs = None   # log probabilities of actions taken
        self.values = None      # centralized critic values
        self.rewards = None     # rewards per step
        self.dones = None       # done flags per step

        self.clear()

    def clear(self):
        """
        Clear the buffer.
        """
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, obs, state, actions, log_probs, value, reward, done):
        """
        Add a new experience to the buffer.
        """
        self.obs.append([obs[i] for i in range(self.n_agents)])
        self.actions.append([actions[i] for i in range(self.n_agents)])
        self.log_probs.append([log_probs[i] for i in range(self.n_agents)])
        self.states.append(state)
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def get(self):
        return {
            "obs": torch.tensor(self.obs, dtype=torch.float32),
            "states": torch.tensor(self.states, dtype=torch.float32),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }

    def prepare_for_ppo(self, advantages, returns, normalize_adv=True):
        """
        Prepare flattened tensors for PPO training.

        advantages: torch.Tensor [T]
        returns:    torch.Tensor [T]
        """
        T = len(self.rewards)
        n = self.n_agents

        obs_flat = []
        actions_flat = []
        log_probs_flat = []
        advantages_flat = []
        returns_flat = []

        for t in range(T):
            for i in range(n):
                obs_flat.append(self.obs[t][i])
                actions_flat.append(self.actions[t][i])
                log_probs_flat.append(self.log_probs[t][i])
                advantages_flat.append(advantages[t].item())
                returns_flat.append(returns[t].item())

        self.obs_flat = torch.tensor(obs_flat, dtype=torch.float32)
        self.actions_flat = torch.tensor(actions_flat, dtype=torch.long)
        self.log_probs_flat = torch.tensor(log_probs_flat, dtype=torch.float32)
        self.advantages_flat = torch.tensor(advantages_flat, dtype=torch.float32)
        self.returns_flat = torch.tensor(returns_flat, dtype=torch.float32)

        if normalize_adv:
            self.advantages_flat = (
                (self.advantages_flat - self.advantages_flat.mean()) /
                (self.advantages_flat.std() + 1e-8)
            )

    def get_minibatches(self, batch_size, device=None, shuffle=True):
        """
        Yield PPO minibatches.
        """
        N = self.obs_flat.size(0)
        indices = np.arange(N)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, N, batch_size):
            idx = indices[start:start + batch_size]

            obs = self.obs_flat[idx]
            actions = self.actions_flat[idx]
            old_log_probs = self.log_probs_flat[idx]
            advantages = self.advantages_flat[idx]
            returns = self.returns_flat[idx]

            if device is not None:
                obs = obs.to(device)
                actions = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                advantages = advantages.to(device)
                returns = returns.to(device)

            yield obs, actions, old_log_probs, advantages, returns

    @property
    def size(self):
        return self.obs_flat.size(0)
