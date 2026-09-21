import numpy as np
import torch
from torch import nn, optim

from config import seed, device
from dqn_network import DQNNetwork
from replay_memory import ReplayMemory



class DQNAgent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update and action selection based on the
    Q-values of actions or epsilon-greedy policy.
    """
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay,
                 clip_grad_norm, learning_rate, discount, memory_capacity):

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        # Initiate the network models
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.n
        self.main_network = DQNNetwork(num_actions=output_dim, input_dim=input_dim).to(device)
        self.target_network = DQNNetwork(num_actions=output_dim, input_dim=input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)


    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy OR based on Q-values.
        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """
        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        # Exploitation: the action is selected based on Q-values
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()
            return action


    def learn(self, batch_size, done):
        """
        Train the main network using a batch of  experiences sampled from the replay memory.

        Parameters:
            batch_size (int): The number of experiences sample from the replay memory
            done (int): Indicates whether the episode is done or not. If done, calculate
            the loss of the episode and append it in a list of plot.
        """
        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        # Ensure shapes match expected (batch, 1)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # Q(s, a) from main network
        predicted_q = self.main_network(states) # Forward pass through the main network to find Q-values
        predicted_q = predicted_q.gather(dim=1, index=actions) # Selecting the Q-values of the actions that

        # Computing the maximum Q-value for the next states using the target network
        # ---------- DOUBLE DQN TARGET ----------
        with torch.no_grad():
            # Choose actions using main network
            next_actions = self.main_network(next_states).argmax(dim=1, keepdim=True)

            # Evaluate chosen actions using the target network
            next_target_q_value = self.target_network(next_states).gather(1, next_actions)

            # Zero next state value if episode terminated
            next_target_q_value[dones] = 0.0

        # Bellman target
        y_js = rewards + self.discount * next_target_q_value
        # ---------------------------------------

        loss = self.criterion(predicted_q, y_js)

        # For logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        # Episode-level loss logging when an episode ends
        if done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0

        # Standard backprop
        self.optimizer.zero_grad() # Zero gradients
        loss.backward() # Perform backward pass and update gradients

        # Use the in-place version for gradient clipping
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()


    def hard_update(self):
        """
        Navie update: update target network's parameters by directly
        copying the parameters from the main network.
        """
        self.target_network.load_state_dict(self.main_network.state_dict())


    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.
        This method decrease epsilon over time according to a decay factor, assuring that
        the agent becomes less exploratory and more exploitive as training progress.
        """
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max*self.epsilon_decay)


    def save(self, path):
        """
        save the parameters of main network to a file with .pth extension.
        """
        torch.save(self.main_network.state_dict(), path)


