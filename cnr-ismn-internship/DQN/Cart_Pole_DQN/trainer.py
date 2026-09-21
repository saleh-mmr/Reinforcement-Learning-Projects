import warnings
import gymnasium as gym
import pygame
import torch
from config import seed
from dqn_agent import DQNAgent
from logger import TrainingLogger


class Trainer:
    def __init__(self, hyperparams):
        # Load parameters
        self.load_pth = hyperparams["RL_load_pth"]
        self.learning_rate = hyperparams["learning_rate"]               # Optimizer learning rate
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.max_steps = hyperparams["max_steps"]                       # Episode timeout
        self.render = hyperparams["render"]                             # Set True to visually inspect
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size
        self.render_fps = hyperparams["render_fps"]                     # Visualization frame rate

        # Create environment
        self.env = gym.make(
            'CartPole-v1',
            max_episode_steps=self.max_steps,
            render_mode="human" if self.render else None                # Opens a window for visualization (if enabled)
        )
        self.env.metadata['render_fps'] = self.render_fps
        warnings.filterwarnings("ignore", category=UserWarning)  # Cleaner console output

        # Initialize DQN Agent
        self.agent = DQNAgent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            learning_rate=self.learning_rate,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            weight_datafile_path=hyperparams["weight_datafile_path"],
        )


        # Create logger
        self.logger = TrainingLogger()

    # TRAINING LOOP
    def train(self):
        total_steps = 0                                     # Count steps across all episodes (used for epsilon decay

        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)            # Initial observation from environment
            done = False                                    # Episode ended because of failure
            episode_reward = 0                              # Episode ended because of failure
            step_counter = 0                                # Step counter inside episode
            while not done:                                 # The agent keeps taking steps until episode ends
                action = self.agent.select_action(state)    # Using epsilon-greedy strategy—exploration or exploitation
                next_state, reward, terminated, truncated, _ = self.env.step(action) # Environment responds
                done = terminated or truncated
                self.agent.replay_memory.store(state, action, next_state, reward, done) # This is essential for off-policy learning
                if len(self.agent.replay_memory) > self.batch_size:         # Only learn when enough future collected
                    self.agent.learn(self.batch_size, done)
                # Tracking step and reward progress
                state = next_state
                episode_reward += reward
                step_counter += 1

            total_steps += step_counter
            # Log episode results (reward, epsilon) for later analysis and visualization
            self.logger.log_episode(
                episode=episode,
                reward=episode_reward,
                epsilon=self.agent.epsilon,
             )
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )

            # Saving best model
            if episode_reward >= 500:
                torch.save(self.agent.q_network.state_dict(), self.load_pth)
                print("Best solved model saved!")

        loss_history = self.agent.loss_history

        self.logger.finalize_results(self.agent.q_network, loss_history)

    # TEST LOOP
    def test(self, max_episodes):
        # Load best model weights
        self.agent.q_network.load_state_dict(torch.load(self.load_pth))
        # Set network to evaluation mode:
        # - Disables dropout (if existed)
        # - Disables unnecessary gradient tracking
        self.agent.q_network.eval()
        # Disable exploration during testing
        self.agent.epsilon = 0.0
        self.env = gym.make(
            'CartPole-v1',
            max_episode_steps=self.max_steps,
            render_mode="human" if self.render else None
        )
        # Resets before each episode and prepares reward counters.
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.select_action(state)                    # Fully greedy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if self.render:
                    self.env.render()
                state = next_state
                episode_reward += reward
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}")
        pygame.quit()
