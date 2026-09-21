import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch import nn
import torch
from utils import config
from memory.replay_memory import ReplayMemory
from network.network import DQNNetwork
from controller.weight_controller_tracking import ManhattanWeightController

class DQNAgent:
    def __init__(
        self,
        # env,                                                      # Gym environment
        n_action_space,
        n_observation_space,
        epsilon_max,                                              # Start with more exploration
        epsilon_min,                                              # Minimum exploration threshold
        epsilon_decay,                                            # How fast exploration decreases
        discount,                                                 # future reward discount factor
        memory_capacity,                                          # Replay buffer size
    ):

        # Logging fields
        self.loss_history = {0:[],1:[]}

        # Hyperparameters
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        # Environment
        self.action_space = n_action_space                              # Saves how many actions the agent can take
        self.observation_space = n_observation_space                    # Saves the full observation space object

        # Replay buffer
        self.cart_pole_memory = ReplayMemory(capacity=memory_capacity)
        self.mountain_car_memory = ReplayMemory(capacity=memory_capacity)
        self.replay_memory = [self.cart_pole_memory, self.mountain_car_memory]  # List of replay buffers for each environment

        # Q-Network
        input_dim = self.observation_space                       # network input = state size (4)
        output_dim = self.action_space                                  # network output = number of actions (2)
        self.q_network = DQNNetwork(output_dim, input_dim).to(config.device)

        # use a squared-error loss just to get gradients,
        self.criterion = nn.MSELoss()

        # Manhattan-style discrete weight controller
        self.weight_controller = ManhattanWeightController(self.q_network)


    # Action Selection (epsilon-greedy)
    def select_action(self, state):
        # exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_space)
        # exploitation
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=config.device)  # Convert state to tensor
        state = state.unsqueeze(0)
        with torch.no_grad():                                   # Disable gradient tracking (faster + no memory waste)
            q_values = self.q_network(state)                    # Compute Q-values: [Q_left, Q_right]
            return torch.argmax(q_values).item()                # Pick action with the highest expected reward

    # Learning step
    def learn(self, batch_size, ap_index, total_steps):
        if len(self.replay_memory[ap_index]) < batch_size:                # Not enough future in replay => Skip learning
            return None

        # Pulls a random batch from replay memory for training
        states, actions, next_states, rewards, dones = self.replay_memory[ap_index].sample(batch_size)

        # Shape Fixing: Convert from shape (B,) [0, 1, 1, 0] → (B,1) [[0], [1], [1], [0]]
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # self.q_network(states) → outputs all Q-values
        # .gather(1, actions) → picks only Q-values of the taken actions
        predicted_q = self.q_network(states).gather(1, actions)       # This is the Q(s,a) value from Bellman equation

        # Max future reward if the episode is not terminal
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
            next_q[dones] = 0.0
        targets = rewards + self.discount * next_q

        # compare current guess vs target (criterion is MSELoss)
        loss = self.criterion(predicted_q, targets)

        # store loss for future logging and visualization
        self.loss_history[ap_index].append(loss.item())

        # Backprop
        self.q_network.zero_grad()
        loss.backward()                     # Compute gradients
        self.weight_controller.step(ap_index=ap_index, step_counter=total_steps)
        return None

    # Epsilon update using ε(t) = ε_min + (ε_max − ε_min) * exp(−λ * t)
    def update_epsilon(self, steps_done):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    # Model saving
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)             # Stores parameters (weights) to a file
