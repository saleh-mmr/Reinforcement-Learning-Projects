import os
import numpy as np
import torch
from typing import Dict, Optional, Any
from marl_meeting_task.src.algos.ps_dqn_agent import PS_DQNAgent
from marl_meeting_task.src.config import device
from marl_meeting_task.src.utils.logger import Logger


class PS_DQN:
    """
    Parameter-Shared Deep Q-Network (PS-DQN) for Multi-Agent Reinforcement Learning.
    """
    
    def __init__(
        self,
        n_agents,
        input_dim,        # observation vector: [own_x, own_y, goal_x, goal_y]
        num_actions,      # actions: up, down, left, right, stay
        hidden_dim,
        learning_rate,
        memory_capacity,
        gamma,          # Discount factor
        epsilon_start,  # Initial epsilon
        epsilon_end,    # Final epsilon
        epsilon_decay_steps,  # Steps over which epsilon decays
        batch_size,       # Batch size for training
        target_update_freq,  # Update target network every N steps
    ):
        """
        Initialize Parameter-Shared DQN.
        """
        # Store hyperparameters
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Training state
        self.total_steps = 0
        
        # Initialize shared agent component
        self.agent = PS_DQNAgent(
            n_agents=n_agents,
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            memory_capacity=memory_capacity,
            gamma=gamma,
            batch_size=batch_size,
        )
        
        # Logger will be initialized in train() method
        self._logger: Optional[Logger] = None
    
    def _print_initialization_summary(self, logger):
        """Print initialization summary."""
        logger.info(f"PS-DQN initialized with {self.n_agents} agents sharing parameters")
        logger.info(f"  - Input dimension: {self.input_dim}")
        logger.info(f"  - Number of actions: {self.num_actions}")
        logger.info(f"  - Shared Q-value network: {self.input_dim} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.num_actions}")
        logger.info(f"  - Target network: (updates every {self.target_update_freq} steps)")
        logger.info(f"  - Shared optimizer: Adam (lr={self.learning_rate})")
        logger.info(f"  - Shared replay buffer: capacity={self.memory_capacity}")
        logger.info(f"  - Hyperparameters:")
        logger.info(f"    * Gamma (discount): {self.gamma}")
        logger.info(f"    * Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_steps} steps")
        logger.info(f"    * Batch size: {self.batch_size}")
    
    # ========================================================================
    # Exploration Schedule
    # ========================================================================
    
    def get_epsilon(self):
        """
        Compute current epsilon based on linear decay schedule.
        """
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.total_steps / self.epsilon_decay_steps
        )
        return max(epsilon, self.epsilon_end)
    
    # ========================================================================
    # Action Selection
    # ========================================================================
    
    def select_actions(self, obs):
        """
        Select actions for all agents using epsilon-greedy policy with shared network.
        """
        epsilon = self.get_epsilon()
        return self.agent.select_actions(obs, epsilon)
    
    # ========================================================================
    # Experience Storage
    # ========================================================================
    
    def store_transitions(
        self,
        obs,
        actions,
        next_obs,
        reward,
        done
    ):
        """
        Store transitions in shared replay buffer.
        """
        self.agent.store_transitions(obs, actions, next_obs, reward, done)
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train_step(self):
        """
        Perform one training step using shared network.
        """
        return self.agent.train_step()
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    def evaluate(
        self,
        env,
        n_episodes = 20,
        max_steps = 50,
    ):
        """
        Evaluate the current policy with greedy actions (epsilon=0).
        """
        # Set network to eval mode for inference
        self.agent.q_network.eval()
        
        eval_successes = []
        eval_lengths = []
        eval_returns = []
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=None)
            episode_reward = 0.0
            episode_terminated = False
            
            for t in range(max_steps):
                if self.agent.q_network.training:
                    self.agent.q_network.eval()
                
                actions = {}
                for agent_id in range(self.n_agents):
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(
                            obs[agent_id],
                            dtype=torch.float32,
                            device=device
                        ).unsqueeze(0)
                        q_values = self.agent.q_network(obs_tensor)
                        actions[agent_id] = q_values.argmax().item()
                
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                if terminated:
                    episode_terminated = True
                
                obs = next_obs
                episode_reward += reward
                
                if done:
                    break
            
            eval_successes.append(1 if episode_terminated else 0)
            eval_lengths.append(t + 1)
            eval_returns.append(episode_reward)
        
        self.agent.q_network.train()
        
        success_rate = np.mean(eval_successes)
        avg_episode_length = np.mean(eval_lengths)
        avg_return = np.mean(eval_returns)
        
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_episode_length,
            'avg_return': avg_return,
        }
    
    def train(
        self,
        env,
        max_episodes,
        max_steps = 50,
        train_freq = 1,
        min_buffer_size = 1000,
        verbose = True,
        log_dir = "runs/ps_dqn",
        eval_episodes = 20,
        env_seed = None,
    ):
        """
        Main training loop for Parameter-Shared DQN.
        """
        logger = Logger(verbose=verbose, log_dir=log_dir)
        self._logger = logger
        
        # Print initialization summary
        self._print_initialization_summary(logger)
        
        # Episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []  # 0 or 1 for each episode
        episode_losses = []
        
        window_size = 100
        success_window = []
        length_window = []
        return_window = []
        
        for episode in range(max_episodes):
            episode_seed = None if env_seed is None else env_seed + episode
            obs, info = env.reset(seed=episode_seed)
            episode_reward = 0.0
            episode_terminated = False  # Track if episode ended with success
            episode_loss_sum = 0.0
            episode_loss_count = 0
            
            for t in range(max_steps):
                actions = self.select_actions(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                if terminated:
                    episode_terminated = True
                
                # Store transitions in shared buffer
                self.store_transitions(obs, actions, next_obs, reward, done)
                
                # Train if buffer is large enough
                if self.total_steps % train_freq == 0:
                    if len(self.agent.replay_memory) >= min_buffer_size:
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss_sum += loss
                            episode_loss_count += 1
                
                # Update target network periodically
                if self.total_steps > 0 and self.total_steps % self.target_update_freq == 0:
                    self.agent.update_target_network()
                
                obs = next_obs
                episode_reward += reward
                self.total_steps += 1
                
                if done:
                    break
            
            # Record episode statistics
            episode_length = t + 1
            episode_success = 1 if episode_terminated else 0
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(episode_success)
            
            if episode_loss_count > 0:
                avg_loss = episode_loss_sum / episode_loss_count
                episode_losses.append(avg_loss)
            else:
                episode_losses.append(None)
            
            # Update moving average windows
            success_window.append(episode_success)
            length_window.append(episode_length)
            return_window.append(episode_reward)
            
            # Keep window size fixed
            if len(success_window) > window_size:
                success_window.pop(0)
                length_window.pop(0)
                return_window.pop(0)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                success_rate = np.mean(episode_successes[-100:])
                current_epsilon = self.get_epsilon()
                logger.progress(
                    episode=episode + 1,
                    max_episodes=max_episodes,
                    avg_reward=avg_reward,
                    avg_length=avg_length,
                    success_rate=success_rate,
                    epsilon=current_epsilon,
                    total_steps=self.total_steps,
                )
        
        # Run final evaluation after all training episodes
        final_eval_metrics = self.evaluate(env, n_episodes=eval_episodes, max_steps=max_steps)
        
        # Print final evaluation results
        logger.evaluation(
            episode=None,
            success_rate=final_eval_metrics['success_rate'],
            avg_episode_length=final_eval_metrics['avg_episode_length'],
            avg_return=final_eval_metrics['avg_return'],
            is_final=True,
        )
        
        # Close logger (closes TensorBoard writer)
        logger.close()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'episode_losses': episode_losses,
            'total_steps': self.total_steps,
            'final_eval_metrics': final_eval_metrics,
        }
