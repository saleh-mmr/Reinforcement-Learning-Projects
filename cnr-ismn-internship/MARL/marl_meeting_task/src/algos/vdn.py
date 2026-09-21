import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Any

from marl_meeting_task.src.algos.vdn_agent import VDNAgent
from marl_meeting_task.src.models.qvalue_network import QValueNetwork
from marl_meeting_task.src.utils.qmix_replay_memory import QMIXReplayMemory
from marl_meeting_task.src.config import device
from marl_meeting_task.src.utils.logger import Logger


class VDN:
    """
    Value Decomposition Networks (VDN) implementation.

    Mirrors the structure of QMIX but uses a simple summation of individual
    agent Q-values to obtain the joint Q_tot for centralized training.
    """

    def __init__(
        self,
        n_agents: int,
        input_dim: int,  # cartpole local observation dim
        state_dim: int,
        num_actions: int,
        hidden_dim: int,
        learning_rate: float,
        memory_capacity: int,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        batch_size: int,
        target_update_freq: int,
    ):
        # Hyperparameters
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.state_dim = state_dim
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

        # Initialize agents
        self.agents: Dict[int, VDNAgent] = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = VDNAgent(
                agent_id=agent_id,
                n_agents=n_agents,
                input_dim=input_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
            )

        # Single optimizer over all agent network parameters
        agent_params = []
        for agent in self.agents.values():
            agent_params.extend(list(agent.q_network.parameters()))
        self.optimizer = optim.Adam(agent_params, lr=learning_rate)

        # Centralized replay buffer (reuse QMIXReplayMemory)
        self.replay_memory = QMIXReplayMemory(capacity=memory_capacity)

        # Logger placeholder
        self._logger: Optional[Logger] = None

    def _print_initialization_summary(self, logger: Logger) -> None:
        logger.info(f"VDN initialized with {self.n_agents} agents")
        logger.info(f"  - Base observation dim: {self.input_dim}")
        logger.info(f"  - Agent input dim (base + one-hot): {self.input_dim + self.n_agents}")
        logger.info(f"  - State dim: {self.state_dim}")
        logger.info(f"  - Num actions: {self.num_actions}")
        logger.info(f"  - Hidden dim: {self.hidden_dim}")
        logger.info(f"  - Memory capacity: {self.memory_capacity}")
        logger.info(f"  - Learning rate: {self.learning_rate}")

    # ========================================================================
    # Network Management
    # ========================================================================

    def update_target_networks(self) -> None:
        for agent in self.agents.values():
            agent.update_target_network()

    def _append_agent_ids_batch(self, observations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        observations_with_id = {}
        for agent_id in range(self.n_agents):
            base_obs = observations[agent_id]  # [batch_size, base_input_dim]
            batch_size = base_obs.shape[0]
            agent_id_onehot = torch.zeros(batch_size, self.n_agents, dtype=torch.float32, device=device)
            agent_id_onehot[:, agent_id] = 1.0
            observations_with_id[agent_id] = torch.cat([base_obs, agent_id_onehot], dim=1)
        return observations_with_id

    # ========================================================================
    # Exploration Schedule
    # ========================================================================

    def get_epsilon(self) -> float:
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        epsilon = self.epsilon_start - (self.total_steps / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_end)
        return max(epsilon, self.epsilon_end)

    # ========================================================================
    # Action Selection
    # ========================================================================

    def select_actions(self, obs: Dict[int, np.ndarray]) -> Dict[int, int]:
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
        done: bool,
    ) -> None:
        self.replay_memory.store(
            state=state,
            observations=obs,
            actions=actions,
            reward=reward,
            next_state=next_state,
            next_observations=next_obs,
            done=done,
        )

    # ========================================================================
    # Training
    # ========================================================================

    def train_step(self) -> Optional[float]:
        if len(self.replay_memory) < self.batch_size:
            return None

        states, observations, actions, rewards, next_states, next_observations, dones = \
            self.replay_memory.sample(self.batch_size, self.n_agents)

        # Append agent ids
        observations_with_id = self._append_agent_ids_batch(observations)
        next_observations_with_id = self._append_agent_ids_batch(next_observations)

        # Compute individual Q_i for taken actions
        agent_qs = []
        for agent_id in range(self.n_agents):
            q_values = self.agents[agent_id].q_network(observations_with_id[agent_id])  # [batch, num_actions]
            agent_q = q_values.gather(1, actions[agent_id].unsqueeze(1)).squeeze(1)  # [batch]
            agent_qs.append(agent_q)

        # Stack to [batch, n_agents] and sum to get Q_tot
        agent_qs = torch.stack(agent_qs, dim=1)
        q_tot = agent_qs.sum(dim=1)  # [batch]

        # Compute target Q_tot by summing target networks' max Q for next observations
        with torch.no_grad():
            next_agent_qs = []
            for agent_id in range(self.n_agents):
                next_q_values = self.agents[agent_id].target_network(next_observations_with_id[agent_id])  # [batch, num_actions]
                next_agent_q = next_q_values.max(1)[0]  # [batch]
                next_agent_qs.append(next_agent_q)
            next_agent_qs = torch.stack(next_agent_qs, dim=1)
            q_tot_target = next_agent_qs.sum(dim=1)  # [batch]

            target = rewards + self.gamma * q_tot_target * (~dones)

        # Loss and optimization
        loss = nn.MSELoss()(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate(self, env, n_episodes: int = 20, max_steps: int = 50) -> Dict[str, float]:
        for agent in self.agents.values():
            agent.q_network.eval()

        eval_successes = []
        eval_lengths = []
        eval_returns = []

        for episode in range(n_episodes):
            obs, info = env.reset(seed=None)
            episode_reward = 0.0
            episode_terminated = False

            for t in range(max_steps):
                # Greedy actions
                actions = {}
                for agent_id in range(self.n_agents):
                    actions[agent_id] = self.agents[agent_id].select_action(obs[agent_id], epsilon=0.0)

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

        for agent in self.agents.values():
            agent.q_network.train()

        return {
            'success_rate': np.mean(eval_successes),
            'avg_episode_length': np.mean(eval_lengths),
            'avg_return': np.mean(eval_returns),
        }

    def train(
        self,
        env,
        max_episodes: int,
        max_steps: int = 50,
        train_freq: int = 1,
        min_buffer_size: int = 1000,
        verbose: bool = True,
        log_dir: Optional[str] = "runs/vdn",
        eval_episodes: int = 20,
        env_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        logger = Logger(verbose=verbose, log_dir=log_dir)
        self._logger = logger

        self._print_initialization_summary(logger)

        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_losses = []

        window_size = 100
        success_window = []
        length_window = []
        return_window = []

        for episode in range(max_episodes):
            episode_seed = None if env_seed is None else env_seed + episode
            obs, info = env.reset(seed=episode_seed)
            # Ensure we always have a valid global state (fallback when _get_global_state is not available)
            if hasattr(self, '_get_global_state'):
                state = self._get_global_state(env)
            else:
                s = []
                for agent_id in range(self.n_agents):
                    x, y = env.agent_pos[agent_id]
                    s.extend([x / env.grid_size, y / env.grid_size])
                gx, gy = env.goal_pos
                s.extend([gx / env.grid_size, gy / env.grid_size])
                state = np.array(s, dtype=np.float32)

            episode_reward = 0.0
            episode_terminated = False
            episode_loss_sum = 0.0
            episode_loss_count = 0

            for t in range(max_steps):
                actions = self.select_actions(obs)
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                # Ensure next_state is valid as well
                if hasattr(self, '_get_global_state'):
                    next_state = self._get_global_state(env)
                else:
                    ns = []
                    for agent_id in range(self.n_agents):
                        x, y = env.agent_pos[agent_id]
                        ns.extend([x / env.grid_size, y / env.grid_size])
                    ngx, ngy = env.goal_pos
                    ns.extend([ngx / env.grid_size, ngy / env.grid_size])
                    next_state = np.array(ns, dtype=np.float32)

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
                    done=done,
                )

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

            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_networks()

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

            success_window.append(episode_success)
            length_window.append(episode_length)
            return_window.append(episode_reward)

            if len(success_window) > window_size:
                success_window.pop(0)
                length_window.pop(0)
                return_window.pop(0)

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

        final_eval_metrics = self.evaluate(env, n_episodes=eval_episodes, max_steps=max_steps)

        logger.evaluation(
            episode=None,
            success_rate=final_eval_metrics['success_rate'],
            avg_episode_length=final_eval_metrics['avg_episode_length'],
            avg_return=final_eval_metrics['avg_return'],
            is_final=True,
        )

        logger.close()

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'episode_losses': episode_losses,
            'total_steps': self.total_steps,
            'final_eval_metrics': final_eval_metrics,
        }

