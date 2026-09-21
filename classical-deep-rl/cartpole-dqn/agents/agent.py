import os
import numpy as np
import sys
from torch.optim import RMSprop
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch import nn
import torch
from utils import config
from memory.replay_memory import ReplayMemory
from network.network import DQNNetwork


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and replay-memory learning."""

    def __init__(
        self,
        env,
        epsilon_max,
        epsilon_min,
        epsilon_decay,
        discount,
        memory_capacity,
        network_size,
    ):
        """Initialize networks, replay memory, optimizer, and exploration schedule."""
        self.loss_history = []

        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.env = env

        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.q_network = DQNNetwork(output_dim, input_dim, network_size).to(config.device)

        self.criterion = nn.MSELoss()

        self.weight_controller = RMSprop(self.q_network.parameters(), lr=0.001)

        # Optional target-network implementation scaffold.
        # self.learn_steps = 0
        # self.target_update_freq = 1000  # example
        # self.target_network = DQNNetwork(output_dim, input_dim, network_size).to(config.device)
        # self.target_network.load_state_dict(self.q_network.state_dict())
        # self.target_network.eval()
        # for p in self.target_network.parameters():
        #     p.requires_grad = False

    def select_action(self, state, epsilon=None):
        """Select an action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)

        state = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)

        # Exploit: choose action with the highest estimated Q-value.
        return torch.argmax(q_values, dim=1).item()

    def learn(self, batch_size):
        """Run one DQN optimization step from a sampled mini-batch."""
        if len(self.replay_memory) < batch_size:
            return None

        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        # Align shapes for gather/target math: [B] -> [B, 1].
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        q_all = self.q_network(states)
        predicted_q = q_all.gather(1, actions)

        # Bellman target: r + gamma * max_a' Q(s', a').
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values
            next_q[dones] = 0.0
        targets = rewards + self.discount * next_q

        # Optional target-network variant.
        # with torch.no_grad():
        #     next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
        #     next_q[dones] = 0.0
        # targets = rewards + self.discount * next_q

        loss = self.criterion(predicted_q, targets)
        self.loss_history.append(loss.item())

        for param in self.q_network.parameters():
            if param.grad is not None:
                param.grad.zero_()
        loss.backward()
        self.weight_controller.step()

        # Optional target-network update hook.
        # self.learn_steps += 1
        # if self.learn_steps % self.target_update_freq == 0:
        #     self.update_target_network()

        return loss.item()

    def update_epsilon(self, steps_done):
        """Decay epsilon exponentially as training progresses."""
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    def save(self, path):
        """Persist Q-network parameters to disk."""
        torch.save(self.q_network.state_dict(), path)

    # Optional target-network method.
    # def update_target_network(self):
    #     self.target_network.load_state_dict(self.q_network.state_dict())