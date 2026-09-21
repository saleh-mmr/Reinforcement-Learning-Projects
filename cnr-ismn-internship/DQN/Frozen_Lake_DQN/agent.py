import numpy as np
import torch
from torch import nn

from config import seed, device
from network import DQNNetwork
from RL.frozen_lake.controllers.manhattan_weight_controller import ManhattanWeightController
from replay_memory import ReplayMemory



class DQNAgent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update and action selection based on the
    Q-values of actions or epsilon-greedy policy.
    """
    def __init__(self,
                 env,
                 epsilon_max,
                 epsilon_min,
                 epsilon_decay,
                 clip_grad_norm,
                 learning_rate,
                 discount,
                 memory_capacity,
                 sigma):

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # DQN hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.action_space = env.action_space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        # Initiate the network models
        self.main_network = DQNNetwork(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device)
        self.target_network = DQNNetwork(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        # self.clip_grad_norm = clip_grad_norm

        self.weight_controller = ManhattanWeightController(self.main_network, sigma=sigma)

    def select_action(self, state, eval_mode=False):
        """
        Selects an action using epsilon-greedy (training)
        or greedy-only (evaluation).
        """

        # Evaluation: pure greedy
        if eval_mode:
            with torch.no_grad():
                Q_values = self.main_network(state)
                return torch.argmax(Q_values).item()

        # Training: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        with torch.no_grad():
            Q_values = self.main_network(state)
            return torch.argmax(Q_values).item()

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
        predicted_q = self.main_network(states)  # forward pass through the main network to find the Q-values of the states
        predicted_q = predicted_q.gather(dim=1,index=actions)  # selecting the Q-values of the actions that were actually taken

        # Computing the maximum Q-value for the next states using the target network
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            # Mask terminal states
            next_target_q_value[dones] = 0

        # DQN target: y = noise_realization + γ * max_a' Q_target(s', a')
        y_js = rewards + (self.discount * next_target_q_value)
        loss = self.criterion(predicted_q, y_js)

        # For logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1


        # Backprop
        self.main_network.zero_grad()
        loss.backward()                     # Compute gradients
        self.weight_controller.step()

        # If episode finished → return average loss
        if done:
            avg_loss = (
                self.running_loss / self.learned_counts
                if self.learned_counts > 0 else 0.0
            )
            # Reset counters here
            self.running_loss = 0
            self.learned_counts = 0
            return avg_loss

        return None



        # # Episode-level loss logging when an episode ends
        # if done:
        #     episode_loss = self.running_loss / self.learned_counts
        #     self.loss_history.append(episode_loss)
        #     self.running_loss = 0
        #     self.learned_counts = 0
        #
        # # Standard backprop
        # self.optimizer.zero_grad() # Zero gradients
        # loss.backward() # Perform backward pass and update gradients
        #
        # # Use the in-place version for gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        # self.optimizer.step()


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
