import os
import numpy as np
from typing import Dict, Optional, Any
from marl_meeting_task.src.algos.iql_agent import IQLAgent
from marl_meeting_task.src.utils.logger import Logger


class IQL:
    """
    Independent Q-Learning (IQL) for Multi-Agent Reinforcement Learning.
    
    IQL is a baseline MARL algorithm where each agent learns independently,
    treating other agents as part of the environment. Each agent maintains:
    - Its own Q-value network (main and target)
    - Its own optimizer
    - Its own experience replay buffer
    
    This approach is simple but effective, though it may struggle with
    non-stationarity since each agent's policy changes during training.
    
    Reference:
    Tan, M. (1993). Multi-agent reinforcement learning: Independent vs.
    cooperative agents. ICML.
    """
    
    def __init__(
        self,
        n_agents: int,
        input_dim: int,              # observation vector: [own_x, own_y, goal_x, goal_y]
        num_actions: int,            # actions: up, down, left, right, stay
        hidden_dim: int,
        learning_rate: float,
        memory_capacity: int,
        gamma: float,                # Discount factor
        epsilon_start: float,        # Initial epsilon
        epsilon_end: float,          # Final epsilon
        epsilon_decay_steps: int,    # Steps over which epsilon decays
        batch_size: int,             # Batch size for training
        target_update_freq: int,     # Update target network every N steps
    ):
        """
        Initialize Independent Q-Learning with multiple agents.
        
        Parameters:
        -----------
        n_agents : int
            Number of independent agents (default: 2)
        input_dim : int
            Dimension of observation vector (default: 4)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden layer dimension for Q-networks (default: 64)
        learning_rate : float
            Learning rate for optimizers (default: 1e-3)
        memory_capacity : int
            Capacity of each agent's replay buffer (default: 10000)
        gamma : float
            Discount factor (default: 0.99)
        epsilon_start : float
            Initial exploration rate (default: 1.0)
        epsilon_end : float
            Final exploration rate (default: 0.05)
        epsilon_decay_steps : int
            Steps over which epsilon decays (default: 50000)
        batch_size : int
            Batch size for training (default: 32)
        target_update_freq : int
            Update target network every N steps (default: 500)
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
        
        # Initialize independent agents
        self.agents: Dict[int, IQLAgent] = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = IQLAgent(
                agent_id=agent_id,
                input_dim=input_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                memory_capacity=memory_capacity,
            )
        
        # Logger will be initialized in train() method
        self._logger: Optional[Logger] = None
    
    def _print_initialization_summary(self, logger: Logger) -> None:
        """Print initialization summary."""
        logger.info(f"IQL initialized with {self.n_agents} independent agents")
        logger.info(f"  - Input dimension: {self.input_dim}")
        logger.info(f"  - Number of actions: {self.num_actions}")
        logger.info(f"  - Each agent has:")
        logger.info(f"    * Q-value network: {self.input_dim} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.num_actions}")
        logger.info(f"    * Target network: (updates every {self.target_update_freq} steps)")
        logger.info(f"    * Optimizer: Adam (lr={self.learning_rate})")
        logger.info(f"    * Replay buffer: capacity={self.memory_capacity}")
        logger.info(f"  - Hyperparameters:")
        logger.info(f"    * Gamma (discount): {self.gamma}")
        logger.info(f"    * Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_steps} steps")
        logger.info(f"    * Batch size: {self.batch_size}")
    
    # ========================================================================
    # Agent Access
    # ========================================================================
    
    def get_agent(self, agent_id: int) -> IQLAgent:
        """
        Get agent by ID.
        
        Parameters:
        -----------
        agent_id : int
            Agent identifier
            
        Returns:
        --------
        IQLAgent
            The requested agent
        """
        return self.agents[agent_id]
    
    def get_all_agents(self) -> Dict[int, IQLAgent]:
        """
        Get all agents.
        
        Returns:
        --------
        Dict[int, IQLAgent]
            Dictionary of all agents keyed by agent_id
        """
        return self.agents
    
    # ========================================================================
    # Exploration Schedule
    # ========================================================================
    
    def get_epsilon(self) -> float:
        """
        Compute current epsilon based on linear decay schedule.
        
        Returns:
        --------
        epsilon : float
            Current exploration rate
        """
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        # Linear decay: ε = ε_start - (ε_start - ε_end) * (steps / decay_steps)
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.total_steps / self.epsilon_decay_steps
        )
        return max(epsilon, self.epsilon_end)
    
    # ========================================================================
    # Action Selection
    # ========================================================================
    
    def select_actions(self, obs: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Select actions for all agents using epsilon-greedy policy.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Observations keyed by agent_id
            
        Returns:
        --------
        Dict[int, int]
            Actions keyed by agent_id
        """
        epsilon = self.get_epsilon()
        actions = {}
        for agent_id in range(self.n_agents):
            actions[agent_id] = self.agents[agent_id].select_action(obs[agent_id], epsilon)
        return actions
    
    # ========================================================================
    # Experience Storage
    # ========================================================================
    
    def store_transitions(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        next_obs: Dict[int, np.ndarray],
        reward: float,
        done: bool
    ) -> None:
        """
        Store transitions in each agent's replay buffer.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Current observations keyed by agent_id
        actions : Dict[int, int]
            Actions taken keyed by agent_id
        next_obs : Dict[int, np.ndarray]
            Next observations keyed by agent_id
        reward : float
            Scalar reward (shared by all agents)
        done : bool
            Whether episode terminated or truncated
        """
        for agent_id in range(self.n_agents):
            self.agents[agent_id].replay_memory.store(
                state=obs,
                action=actions,
                next_state=next_obs,
                reward=reward,
                done=done
            )
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train_step(self) -> Dict[int, Optional[float]]:
        """
        Perform one training step for all agents.
        
        Returns:
        --------
        Dict[int, Optional[float]]
            Training losses keyed by agent_id (None if buffer too small)
        """
        losses = {}
        for agent_id in range(self.n_agents):
            losses[agent_id] = self.agents[agent_id].learn(self.batch_size, self.gamma)
        return losses
    
    def update_target_networks(self) -> None:
        """Update target networks for all agents."""
        for agent_id in range(self.n_agents):
            self.agents[agent_id].update_target_network()
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    def evaluate(
        self,
        env,
        n_episodes: int = 20,
        max_steps: int = 50,
    ) -> Dict[str, float]:
        """
        Evaluate the current policy with greedy actions (epsilon=0).
        
        Parameters:
        -----------
        env : MeetingGridworldEnv
            The environment to evaluate on
        n_episodes : int
            Number of evaluation episodes (default: 20)
        max_steps : int
            Maximum steps per episode (default: 50)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing evaluation metrics:
            - success_rate: fraction of successful episodes
            - avg_episode_length: average episode length
            - avg_return: average cumulative reward
        """
        # Set all networks to eval mode for inference
        for agent in self.agents.values():
            agent.q_network.eval()
        
        eval_successes = []
        eval_lengths = []
        eval_returns = []
        
        for episode in range(n_episodes):
            # Use None seed for evaluation to get diverse episodes
            obs, info = env.reset(seed=None)
            episode_reward = 0.0
            episode_terminated = False
            
            for t in range(max_steps):
                # Greedy action selection (epsilon=0) - use main networks
                # Ensure networks remain in eval mode
                for agent in self.agents.values():
                    if agent.q_network.training:
                        agent.q_network.eval()
                
                actions = {}
                for agent_id in range(self.n_agents):
                    # Explicitly pass epsilon=0.0 for greedy policy
                    actions[agent_id] = self.agents[agent_id].select_action(obs[agent_id], epsilon=0.0)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                if terminated:
                    episode_terminated = True
                
                obs = next_obs
                episode_reward += reward
                
                if done:
                    break
            
            # Record evaluation episode statistics
            eval_successes.append(1 if episode_terminated else 0)
            eval_lengths.append(t + 1)
            eval_returns.append(episode_reward)
        
        # Set networks back to train mode
        for agent in self.agents.values():
            agent.q_network.train()
        
        # Compute evaluation metrics
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
        max_episodes: int,
        max_steps: int = 50,
        train_freq: int = 1,
        min_buffer_size: int = 1000,
        verbose: bool = True,
        log_dir: Optional[str] = "runs/iql",
        eval_episodes: int = 20,
        env_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop for Independent Q-Learning.
        
        Parameters:
        -----------
        env : MeetingGridworldEnv
            The environment to train on
        max_episodes : int
            Maximum number of episodes to train
        max_steps : int
            Maximum steps per episode (default: 50)
        train_freq : int
            Train every N steps (default: 1)
        min_buffer_size : int
            Minimum buffer size before training starts (default: 1000)
        verbose : bool
            Whether to print training progress (default: True)
        log_dir : Optional[str]
            Directory for TensorBoard logs (default: "runs/iql")
            If None, TensorBoard logging is disabled
        eval_episodes : int
            Number of episodes to run during evaluation (default: 20)
        env_seed : Optional[int]
            Seed for environment resets (default: None, uses config seed)
            
        Returns:
        --------
        training_stats : dict
            Dictionary containing training statistics
        """
        # Initialize logger
        logger = Logger(verbose=verbose, log_dir=log_dir)
        self._logger = logger
        
        # Print initialization summary
        self._print_initialization_summary(logger)
        
        # Episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []  # 0 or 1 for each episode
        episode_losses = {agent_id: [] for agent_id in range(self.n_agents)}
        
        # Moving averages for TensorBoard (window size: 100 episodes)
        window_size = 100
        success_window = []
        length_window = []
        return_window = []
        
        for episode in range(max_episodes):
            # Use seed derived from cartpole seed and episode for reproducibility with diversity
            episode_seed = None if env_seed is None else env_seed + episode
            obs, info = env.reset(seed=episode_seed)
            episode_reward = 0.0
            episode_terminated = False  # Track if episode ended with success
            episode_loss_sum = {agent_id: 0.0 for agent_id in range(self.n_agents)}
            episode_loss_count = {agent_id: 0 for agent_id in range(self.n_agents)}
            
            for t in range(max_steps):
                # Select actions for all agents (epsilon-greedy)
                actions = self.select_actions(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                # Track if episode ended with success (all agents reached goal)
                if terminated:
                    episode_terminated = True
                
                # Store transitions in each agent's buffer
                self.store_transitions(obs, actions, next_obs, reward, done)
                
                # Train agents if buffer is large enough
                if self.total_steps % train_freq == 0:
                    # Check if any agent has enough future
                    if all(len(self.agents[i].replay_memory) >= min_buffer_size 
                           for i in range(self.n_agents)):
                        losses = self.train_step()
                        for agent_id, loss in losses.items():
                            if loss is not None:
                                episode_loss_sum[agent_id] += loss
                                episode_loss_count[agent_id] += 1
                
                # Update target networks periodically
                if self.total_steps > 0 and self.total_steps % self.target_update_freq == 0:
                    self.update_target_networks()
                
                # Update state
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
            
            for agent_id in range(self.n_agents):
                if episode_loss_count[agent_id] > 0:
                    avg_loss = episode_loss_sum[agent_id] / episode_loss_count[agent_id]
                    episode_losses[agent_id].append(avg_loss)
                else:
                    episode_losses[agent_id].append(None)
            
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

