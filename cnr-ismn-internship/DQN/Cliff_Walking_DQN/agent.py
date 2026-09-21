import numpy as np
import torch
from torch import nn, optim

from RL.Cliff_Walking_DQN.config import seed, device
from RL.Cliff_Walking_DQN.network import DQNNetwork
from RL.Cliff_Walking_DQN.replay_memory import ReplayMemory
from RL.Cliff_Walking_DQN.manhattan_weight_controller import ManhattanWeightController


class Agent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """

    def __init__(self, env,
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
        self.action_space.seed(seed)  # Set the seed to get reproducible results when sampling the action space
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)

        # Initiate the network models
        self.main_network = DQNNetwork(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device)
        self.target_network = DQNNetwork(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(
            device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.critertion = nn.MSELoss()

        self.weight_controller = ManhattanWeightController(self.main_network, sigma=sigma)

        # self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        # self.clip_grad_norm = clip_grad_norm  # For clipping exploding gradients caused by high reward value

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy OR based on the Q-values.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """

        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        # Exploitation: the action is selected based on the Q-values.
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()

            return action

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.

        Parameters:
            batch_size (int): The number of experiences to sample from the replay memory.
            done (bool): Indicates whether the episode is done or not. If done,
            calculate the loss of the episode and append it in a list for plot.
        """

        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        # # The following prints are for debugging. Use them to indicate the correct shape of the tensors.
        # print('Before--------Before')
        # print("states:", states.shape)
        # print("actions:", actions.shape)
        # print("next_states:", next_states.shape)
        # print("rewards:", rewards.shape)
        # print("dones:", dones.shape)

        # # Preprocess the data for training
        # states        = states.unsqueeze(1)
        # next_states   = next_states.unsqueeze(1)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # # The following prints are for debugging. Use them to indicate the correct shape of the tensors.
        # print()
        # print('After--------After')
        # print("states:", states.shape)
        # print("actions:", actions.shape)
        # print("next_states:", next_states.shape)
        # print("rewards:", rewards.shape)
        # print("dones:", dones.shape)

        predicted_q = self.main_network(
            states)  # forward pass through the main network to find the Q-values of the states
        predicted_q = predicted_q.gather(dim=1,
                                         index=actions)  # selecting the Q-values of the actions that were actually taken

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[
                0]  # not argmax (cause we want the maxmimum q-value, not the action that maximize it)

        next_target_q_value[dones] = 0  # Set the Q-value for terminal states to zero
        y_js = rewards + (self.discount * next_target_q_value)  # Compute the target Q-values
        loss = self.critertion(predicted_q, y_js)  # Compute the loss

        # Update the running loss and learned counts for logging and plotting
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





        # if done:
        #     episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
        #     self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
        #     # Reset the running loss and learned counts
        #     self.running_loss = 0
        #     self.learned_counts = 0
        # self.optimizer.zero_grad()  # Zero the gradients
        # loss.backward()  # Perform backward pass and update the gradients
        # torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        # self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def hard_update(self):
        """
        Navie update: Update the target network parameters by directly copying
        the parameters from the main network.
        """

        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.

        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """

        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.

        """
        torch.save(self.main_network.state_dict(), path)
