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
from controller.synaptic_weight_controller import SynapticWeightController


class DQNAgent:
    def __init__(
        self,
        first_cartpole_env,                                             # Gym cartpole environment
        second_cartpole_env,                                          # Gym mountaincar environment
        epsilon_max,                                              # Start with more exploration
        epsilon_min,                                              # Minimum exploration threshold
        epsilon_decay,                                            # How fast exploration decreases
        discount,                                                 # future reward discount factor
        memory_capacity,                                          # Replay buffer size
        network_size,                                             # Number of neurons in hidden layers
        g_ap,                                                     # Coefficient for Conductance ap
        g_p,                                                      # Coefficient for Conductance p
        shift_parameter,                                          # used in log(index+shift_parameter) for conductance calculation
        g_bias,                                                   # Coefficient for Conductance bias
        noise_stddev,                                             # Standard deviation of noise added to weight updates
    ):
        # Hyperparameters
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        # Environment
        self.first_cartpole_env = first_cartpole_env
        self.second_cartpole_env = second_cartpole_env
        assert first_cartpole_env.action_space.n == second_cartpole_env.action_space.n, "Action space dimensions must match"
        assert first_cartpole_env.observation_space.shape[0] == second_cartpole_env.observation_space.shape[0], "Observation space dimensions must match"
        self.action_space_dim = first_cartpole_env.action_space.n
        self.observation_space_dim = first_cartpole_env.observation_space.shape[0]


        # Replay buffer
        self.first_cartpole_memory = ReplayMemory(capacity=memory_capacity)
        self.second_cartpole_memory = ReplayMemory(capacity=memory_capacity)
        self.replay_memory = [self.first_cartpole_memory, self.second_cartpole_memory]

        # Q-Network
        input_dim = self.observation_space_dim                                # network input = state size (4)
        output_dim = self.action_space_dim                                    # network output = number of actions (2)
        self.q_network = DQNNetwork(output_dim, input_dim, network_size).to(config.device)

        # use a squared-error loss just to get gradients,
        self.criterion = nn.MSELoss()

        self.weight_controller = SynapticWeightController(self.q_network, g_ap, g_p, shift_parameter, g_bias, noise_stddev)

    # Action Selection (epsilon-greedy)
    def select_action(self, state, ap_index, epsilon=None):

        if epsilon is None:
            epsilon = self.epsilon

        # exploration
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_space_dim)

        # exploration
        state = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            # weight_matrix = self.q_network.state_dict()["FC.0.weight"]
            # print(weight_matrix[0, 0].item())
            self.weight_controller.load_weights(ap_index)
            q_values = self.q_network(state)
            # print(f"Q-values for AP index {ap_index}: {q_values.cpu().numpy()}")
        return torch.argmax(q_values, dim=1).item()        # exploration



    def select_action_test(self, state):
        # exploration
        state = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values, dim=1).item()        # exploration


    # Learning step
    def learn(self, batch_size, ap_index):
        if len(self.replay_memory[ap_index]) < batch_size:                # Not enough future in replay => Skip learning
            return None

        # Pulls a random batch from replay memory for training
        states, actions, next_states, rewards, dones = self.replay_memory[ap_index].sample(batch_size)

        # Shape Fixing: Convert from shape (B,) [0, 1, 1, 0] → (B,1) [[0], [1], [1], [0]]
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        self.weight_controller.load_weights(ap_index)  # Load current weights from the controller before forward pass
        # self.q_network(states) → outputs all Q-values
        # .gather(1, actions) → picks only Q-values of the taken actions
        q_all = self.q_network(states)
        predicted_q = q_all.gather(1, actions)

        # Max future reward if the episode is not terminal
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
            next_q[dones] = 0.0
        targets = rewards + self.discount * next_q

        # compare current guess vs target (criterion is MSELoss)
        loss = self.criterion(predicted_q, targets)

        # Clear old gradients
        self.q_network.zero_grad()
        loss.backward()
        self.weight_controller.step(ap_index)

        return loss.item()

    # Epsilon update using ε(t) = ε_min + (ε_max − ε_min) * exp(−λ * t)
    def update_epsilon(self, steps_done):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    # Model saving
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)             # Stores parameters (weights) to a file