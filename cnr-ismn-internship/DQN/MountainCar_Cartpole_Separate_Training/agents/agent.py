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
from controller.sgd_optimizer import GDOptimizer
from controller.linear_function_conductance import ManhattanController
from controller.logarithmic_function_conductance import LogarithmicManhattanController

class DQNAgent:
    def __init__(
        self,
        env,                                                      # Gym environment
        epsilon_max,                                              # Start with more exploration
        epsilon_min,                                              # Minimum exploration threshold
        epsilon_decay,                                            # How fast exploration decreases
        discount,                                                 # future reward discount factor
        memory_capacity,                                          # Replay buffer size
        optimizer_selector,
    ):

        # Logging fields
        self.loss_history = []

        # Hyperparameters
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        # Environment
        self.env = env

        # Replay buffer
        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        # Q-Network
        input_dim = self.env.observation_space.shape[0]                       # network input = state size (4)
        output_dim = self.env.action_space.n                                  # network output = number of actions (2)
        self.q_network = DQNNetwork(output_dim, input_dim).to(config.device)

        # use a squared-error loss just to get gradients,
        self.criterion = nn.MSELoss()

        if optimizer_selector == 1:
            self.weight_controller = ManhattanController(self.q_network)
        elif optimizer_selector == 2:
            self.weight_controller = LogarithmicManhattanController(self.q_network)
        elif optimizer_selector == 3:
            self.weight_controller = GDOptimizer(self.q_network)
        elif optimizer_selector == 4:
            self.weight_controller = RMSprop(self.q_network.parameters(), lr=0.001)


        # Target Network
        # self.learn_steps = 0
        # self.target_update_freq = 1000  # example
        # # Target Q-Network
        # self.target_network = DQNNetwork(output_dim, input_dim).to(config.device)
        # self.target_network.load_state_dict(self.q_network.state_dict())
        # self.target_network.eval()
        # for p in self.target_network.parameters():
        #     p.requires_grad = False

    # Action Selection (epsilon-greedy)
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        # exploration
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)

        # exploration
        state = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)

        return torch.argmax(q_values, dim=1).item()        # exploration

    # Learning step
    def learn(self, batch_size):
        if len(self.replay_memory) < batch_size:                # Not enough future in replay => Skip learning
            return None

        # Pulls a random batch from replay memory for training
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        # Shape Fixing: Convert from shape (B,) [0, 1, 1, 0] → (B,1) [[0], [1], [1], [0]]
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # self.q_network(states) → outputs all Q-values
        # .gather(1, actions) → picks only Q-values of the taken actions
        q_all = self.q_network(states)
        predicted_q = q_all.gather(1, actions)

        # DEBUG: print Q-values
        # if len(self.loss_history) % 200 == 0:
        #     print("Sample Q-values:", q_all[0].detach().cpu().numpy())

        # Max future reward if the episode is not terminal
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
            next_q[dones] = 0.0
        targets = rewards + self.discount * next_q

        # Target Network
        # with torch.no_grad():
        #     next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
        #     next_q[dones] = 0.0
        # targets = rewards + self.discount * next_q

        # compare current guess vs target (criterion is MSELoss)
        loss = self.criterion(predicted_q, targets)

        # store loss for future logging and visualization
        self.loss_history.append(loss.item())

        # Clear old gradients
        for param in self.q_network.parameters():
            if param.grad is not None:
                param.grad.zero_()
        loss.backward()
        self.weight_controller.step()

        # Target Network
        # self.learn_steps += 1
        # if self.learn_steps % self.target_update_freq == 0:
        #     self.update_target_network()

        return loss.item()

    # Epsilon update using ε(t) = ε_min + (ε_max − ε_min) * exp(−λ * t)
    def update_epsilon(self, steps_done):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    # Model saving
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)             # Stores parameters (weights) to a file

    # Target Network
    # def update_target_network(self):
    #     self.target_network.load_state_dict(self.q_network.state_dict())