import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Any

from marl_meeting_task.src.algos.qmix_agent import QMIXAgent
from marl_meeting_task.src.models.mixing_network import MixingNetwork
from marl_meeting_task.src.utils.qmix_replay_memory import QMIXReplayMemory
from marl_meeting_task.src.config import device
from marl_meeting_task.src.utils.logger import Logger


class QMIX:
    """
    QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent DQN.
    
    QMIX uses:
    - Individual agent Q-networks (decentralized execution)
    - Mixing network (centralized training)
    - Centralized replay buffer with global state
    
    Key property: Q_tot is monotonic in each Q_i, enabling decentralized
    execution while maintaining centralized training benefits.
    
    Reference:
    Rashid, T., et al. (2018). QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning. ICML.
    """
    
    def __init__(
        self,
        n_agents: int,
        input_dim: int,  # Local observation dimension: [own_x, own_y, goal_x, goal_y]
        state_dim: int,  # Global state dimension: [a1_x, a1_y, a2_x, a2_y, g_x, g_y]
        num_actions: int,
        hidden_dim: int,
        mixing_hidden_dim: int,  # Increased from 64 for better mixing capacity
        learning_rate: float,
        memory_capacity: int,
        gamma: float,
        epsilon_start,
        epsilon_end,
        epsilon_decay_steps,  # Slowed down from 50000 for more gradual exploration decay
        batch_size,
        target_update_freq,
    ):
        """
        Initialize QMIX.
        
        Parameters:
        -----------
        n_agents : int
            Number of agents (default: 2)
        input_dim : int
            Dimension of local observation (default: 4)
        state_dim : int
            Dimension of global state (default: 6)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden dimension for agent Q-networks (default: 64)
        mixing_hidden_dim : int
            Hidden dimension for mixing network (default: 64)
        learning_rate : float
            Learning rate (default: 1e-3)
        memory_capacity : int
            Capacity of centralized replay buffer (default: 10000)
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
            Update target networks every N steps (default: 500)
        """
        # Store hyperparameters
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.mixing_hidden_dim = mixing_hidden_dim
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
        self.grid_size = 5  # Default grid size (will be updated from env if needed)
        
        # Initialize agent Q-networks (one per agent)
        # Note: input_dim is cartpole observation dim, agent_id will be appended as one-hot
        self.agents: Dict[int, QMIXAgent] = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = QMIXAgent(
                agent_id=agent_id,
                n_agents=n_agents,
                input_dim=input_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
            )
        
        # Mixing network (main)
        self.mixing_network = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim
        ).to(device)
        
        # Mixing network (target)
        self.target_mixing_network = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim
        ).to(device)
        self.update_target_networks()  # Initialize with same weights
        
        # Separate optimizers for better control
        # Agent networks with standard learning rate
        agent_params = []
        for agent in self.agents.values():
            agent_params.extend(list(agent.q_network.parameters()))
        
        # Mixing network with lower learning rate (more stable)
        mixing_lr = learning_rate * 0.5  # Half the learning rate for mixing network
        mixing_params = list(self.mixing_network.parameters())
        
        # Create separate parameter groups
        self.optimizer = optim.Adam([
            {'params': agent_params, 'lr': learning_rate},
            {'params': mixing_params, 'lr': mixing_lr}
        ])
        
        # Centralized replay buffer
        self.replay_memory = QMIXReplayMemory(capacity=memory_capacity)
        
        # Logger will be initialized in train() method
        self._logger: Optional[Logger] = None
    
    def _print_initialization_summary(self, logger: Logger) -> None:
        """Print initialization summary."""
        logger.info(f"QMIX initialized with {self.n_agents} agents")
        logger.info(f"  - Base observation dimension: {self.input_dim}")
        logger.info(f"  - Agent input dimension: {self.input_dim + self.n_agents} (base + one-hot agent_id)")
        logger.info(f"  - Global state dimension: {self.state_dim}")
        logger.info(f"  - Number of actions: {self.num_actions}")
        logger.info(f"  - Agent Q-networks: {self.input_dim + self.n_agents} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.num_actions}")
        logger.info(f"  - Mixing network: combines {self.n_agents} Q-values using state (hidden_dim={self.mixing_hidden_dim})")
        logger.info(f"  - Target networks: (update every {self.target_update_freq} episodes)")
        logger.info(f"  - Optimizer: Adam (lr={self.learning_rate})")
        logger.info(f"  - Centralized replay buffer: capacity={self.memory_capacity}")
        logger.info(f"  - Hyperparameters:")
        logger.info(f"    * Gamma (discount): {self.gamma}")
        logger.info(f"    * Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_steps} steps")
        logger.info(f"    * Batch size: {self.batch_size}")
    
    # ========================================================================
    # Network Management
    # ========================================================================
    
    def update_target_networks(self) -> None:
        """Copy weights from main networks to target networks."""
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        for agent in self.agents.values():
            agent.update_target_network()
    
    # ========================================================================
    # Helper: Append Agent IDs to Observations
    # ========================================================================
    
    def _append_agent_ids_batch(
        self, 
        observations: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Append one-hot agent IDs to batch observations.
        
        Parameters:
        -----------
        observations : Dict[int, torch.Tensor]
            Batch observations keyed by agent_id, each of shape [batch_size, base_input_dim]
            
        Returns:
        --------
        Dict[int, torch.Tensor]
            Observations with one-hot agent IDs appended, each of shape [batch_size, input_dim]
        """
        observations_with_id = {}
        for agent_id in range(self.n_agents):
            # Get cartpole observation: [batch_size, base_input_dim]
            base_obs = observations[agent_id]
            
            # Create one-hot agent_id: [batch_size, n_agents]
            batch_size = base_obs.shape[0]
            agent_id_onehot = torch.zeros(
                batch_size, 
                self.n_agents, 
                dtype=torch.float32, 
                device=device
            )
            agent_id_onehot[:, agent_id] = 1.0
            
            # Concatenate: [batch_size, base_input_dim + n_agents]
            obs_with_id = torch.cat([base_obs, agent_id_onehot], dim=1)
            observations_with_id[agent_id] = obs_with_id
        
        return observations_with_id
    
    # ========================================================================
    # Helper: Extract Global State
    # ========================================================================
    
    def _get_global_state(self, env) -> np.ndarray:
        """
        Extract global state from environment and normalize to [0, 1] range.
        
        Global state: [a1_x, a1_y, a2_x, a2_y, g_x, g_y]
        Normalized by grid_size to improve mixing network learning.
        
        Parameters:
        -----------
        env : MeetingGridworldEnv
            The environment
            
        Returns:
        --------
        np.ndarray
            Normalized global state vector in [0, 1] range
        """
        # Update grid_size from environment if available
        if hasattr(env, 'grid_size'):
            self.grid_size = env.grid_size
        
        state = []
        for agent_id in range(self.n_agents):
            x, y = env.agent_pos[agent_id]
            # Normalize coordinates to [0, 1] range
            state.extend([x / self.grid_size, y / self.grid_size])
        gx, gy = env.goal_pos
        # Normalize goal coordinates to [0, 1] range
        state.extend([gx / self.grid_size, gy / self.grid_size])
        return np.array(state, dtype=np.float32)
    
    # ========================================================================
    # Exploration Schedule
    # ========================================================================
    
    def get_epsilon(self) -> float:
        """
        Compute current epsilon based on linear decay schedule.
        
        Returns:
        --------
        float
            Current exploration rate
        """
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.total_steps / self.epsilon_decay_steps
        )
        return max(epsilon, self.epsilon_end)
    
    # ========================================================================
    # Action Selection (Decentralized)
    # ========================================================================
    
    def select_actions(self, obs: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Select actions for all agents (decentralized execution).
        
        Each agent selects action based on its local observation.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Local observations keyed by agent_id
            
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
    
    def store_transition(
        self,
        state: np.ndarray,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        reward: float,
        next_state: np.ndarray,
        next_obs: Dict[int, np.ndarray],
        done: bool
    ) -> None:
        """
        Store transition in centralized replay buffer.
        
        Parameters:
        -----------
        state : np.ndarray
            Global state s
        obs : Dict[int, np.ndarray]
            Local observations o_i
        actions : Dict[int, int]
            Actions a_i
        reward : float
            Joint reward noise_realization
        next_state : np.ndarray
            Next global state s_next
        next_obs : Dict[int, np.ndarray]
            Next local observations o_i_next
        done : bool
            Whether episode ended
        """
        self.replay_memory.store(
            state=state,
            observations=obs,
            actions=actions,
            reward=reward,
            next_state=next_state,
            next_observations=next_obs,
            done=done
        )
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Computes Q_tot loss and updates all networks.
        
        Returns:
        --------
        Optional[float]
            Training loss if buffer has enough future, None otherwise
        """
        if len(self.replay_memory) < self.batch_size:
            return None
        
        # Sample batch
        states, observations, actions, rewards, next_states, next_observations, dones = \
            self.replay_memory.sample(self.batch_size, self.n_agents)
        
        # Append one-hot agent IDs to observations
        observations_with_id = self._append_agent_ids_batch(observations)
        next_observations_with_id = self._append_agent_ids_batch(next_observations)
        
        # Compute current Q_tot(s, a_1, a_2, ...)
        # Get Q_i(o_i, a_i) for each agent
        agent_qs = []
        for agent_id in range(self.n_agents):
            q_values = self.agents[agent_id].q_network(observations_with_id[agent_id])  # [batch_size, num_actions]
            # Select Q-value for taken action
            agent_q = q_values.gather(1, actions[agent_id].unsqueeze(1)).squeeze(1)  # [batch_size]
            agent_qs.append(agent_q)
        
        # Stack: [batch_size, n_agents]
        agent_qs = torch.stack(agent_qs, dim=1)
        
        # Mix to get Q_tot
        q_tot = self.mixing_network(agent_qs, states)  # [batch_size, 1]
        q_tot = q_tot.squeeze(1)  # [batch_size]
        
        # Compute target Q_tot(s_next, a_1', a_2', ...)
        # Double-Q: Select actions using online networks, evaluate with target networks
        with torch.no_grad():
            # Step 1: Select greedy actions using online (main) networks
            next_actions = []
            for agent_id in range(self.n_agents):
                next_q_values_online = self.agents[agent_id].q_network(next_observations_with_id[agent_id])  # [batch_size, num_actions]
                next_action = next_q_values_online.argmax(1)  # [batch_size] - greedy action from online network
                next_actions.append(next_action)
            
            # Step 2: Evaluate selected actions using target networks
            next_agent_qs = []
            for agent_id in range(self.n_agents):
                next_q_values_target = self.agents[agent_id].target_network(next_observations_with_id[agent_id])  # [batch_size, num_actions]
                # Evaluate the action selected by online network using target network
                next_agent_q = next_q_values_target.gather(1, next_actions[agent_id].unsqueeze(1)).squeeze(1)  # [batch_size]
                next_agent_qs.append(next_agent_q)
            
            # Stack: [batch_size, n_agents]
            next_agent_qs = torch.stack(next_agent_qs, dim=1)
            
            # Mix to get Q_tot_target
            q_tot_target = self.target_mixing_network(next_agent_qs, next_states)  # [batch_size, 1]
            q_tot_target = q_tot_target.squeeze(1)  # [batch_size]
            
            # Bellman target
            target = rewards + self.gamma * q_tot_target * (~dones)
        
        # Compute loss
        loss = nn.MSELoss()(q_tot, target)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability - clip all parameters (agent networks + mixing network)
        all_params = []
        for param_group in self.optimizer.param_groups:
            all_params.extend(param_group['params'])
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    def evaluate(
        self,
        env,
        n_episodes: int = 500,
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
            Dictionary containing evaluation metrics
        """
        # Set all networks to eval mode for inference
        for agent in self.agents.values():
            agent.q_network.eval()
        self.mixing_network.eval()
        
        eval_successes = []
        eval_lengths = []
        eval_returns = []
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=None)
            episode_reward = 0.0
            episode_terminated = False
            
            for t in range(max_steps):
                # Ensure networks remain in eval mode
                for agent in self.agents.values():
                    if agent.q_network.training:
                        agent.q_network.eval()
                if self.mixing_network.training:
                    self.mixing_network.eval()
                
                # Greedy action selection (epsilon=0) - use main networks
                # Explicitly pass epsilon=0.0 for greedy policy
                actions = {}
                for agent_id in range(self.n_agents):
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
            
            eval_successes.append(1 if episode_terminated else 0)
            eval_lengths.append(t + 1)
            eval_returns.append(episode_reward)
        
        # Set networks back to train mode
        for agent in self.agents.values():
            agent.q_network.train()
        self.mixing_network.train()
        
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
        log_dir: Optional[str] = "runs/qmix",
        eval_episodes: int = 500,
        env_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop for QMIX.
        
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
            Directory for TensorBoard logs (default: "runs/qmix")
        eval_episodes : int
            Number of episodes to run during evaluation (default: 20)
        env_seed : Optional[int]
            Seed for environment resets (default: None)
            
        Returns:
        --------
        Dict[str, Any]
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
        episode_successes = []
        episode_losses = []
        
        # Moving averages
        window_size = 100
        success_window = []
        length_window = []
        return_window = []
        
        for episode in range(max_episodes):
            episode_seed = None if env_seed is None else env_seed + episode
            obs, info = env.reset(seed=episode_seed)
            state = self._get_global_state(env)
            
            episode_reward = 0.0
            episode_terminated = False
            episode_loss_sum = 0.0
            episode_loss_count = 0
            
            for t in range(max_steps):
                # Select actions (decentralized)
                actions = self.select_actions(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                next_state = self._get_global_state(env)
                
                if terminated:
                    episode_terminated = True
                
                # Store transition
                self.store_transition(
                    state=state,
                    obs=obs,
                    actions=actions,
                    reward=reward,
                    next_state=next_state,
                    next_obs=next_obs,
                    done=done
                )
                
                # Train if buffer is large enough (warm-up period enforced by min_buffer_size)
                # Increased min_buffer_size reduces learning noise by ensuring diverse future before training
                if self.total_steps % train_freq == 0:
                    if len(self.replay_memory) >= min_buffer_size:
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss_sum += loss
                            episode_loss_count += 1
                
                # Update state
                obs = next_obs
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                
                if done:
                    break
            
            # Update target networks every target_update_freq episodes
            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_networks()
            
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
            
            # Update moving averages
            success_window.append(episode_success)
            length_window.append(episode_length)
            return_window.append(episode_reward)
            
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

