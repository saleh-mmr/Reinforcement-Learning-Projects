"""
Smoke test for PS-DQN implementation.
Verifies basic functionality without full training.
"""

import numpy as np
import torch
from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.algos.ps_dqn import PS_DQN


def test_ps_dqn():
    """Test PS-DQN basic functionality."""
    print("="*60)
    print("PS-DQN Smoke Test")
    print("="*60)
    
    # Create environment
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    print("\n✓ Environment created")
    
    # Initialize PS-DQN
    ps_dqn = PS_DQN(
        n_agents=2,
        input_dim=4,
        num_actions=5,
        hidden_dim=64,
        learning_rate=1e-3,
        memory_capacity=1000,  # Small for testing
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=1000,
        batch_size=32,
        target_update_freq=100,
    )
    print("✓ PS-DQN initialized")
    
    # Test action selection
    obs, info = env.reset(seed=2025)
    actions = ps_dqn.select_actions(obs)
    print(f"✓ Action selection works: {actions}")
    assert isinstance(actions, dict)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions.values())
    
    # Test storing transitions
    next_obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    ps_dqn.store_transitions(obs, actions, next_obs, reward, done)
    print(f"✓ Stored transitions (buffer size: {len(ps_dqn.agent.replay_memory)})")
    assert len(ps_dqn.agent.replay_memory) == 1  # 1 joint transition per timestep
    
    # Fill buffer a bit
    for _ in range(50):
        obs = next_obs
        actions = ps_dqn.select_actions(obs)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        ps_dqn.store_transitions(obs, actions, next_obs, reward, done)
        ps_dqn.total_steps += 1
        if done:
            obs, info = env.reset(seed=None)
    
    print(f"✓ Filled buffer (size: {len(ps_dqn.agent.replay_memory)})")
    
    # Test training step
    loss = ps_dqn.train_step()
    print(f"✓ Training step works (loss: {loss:.4f})")
    assert loss is not None
    assert isinstance(loss, float)
    
    # Test epsilon decay
    epsilon = ps_dqn.get_epsilon()
    print(f"✓ Epsilon schedule works (current epsilon: {epsilon:.3f})")
    assert 0 <= epsilon <= 1.0
    
    # Test target network update
    ps_dqn.agent.update_target_network()
    print("✓ Target network update works")
    
    # Test evaluation
    eval_metrics = ps_dqn.evaluate(env, n_episodes=5, max_steps=50)
    print(f"✓ Evaluation works:")
    print(f"    Success rate: {eval_metrics['success_rate']:.2%}")
    print(f"    Avg length: {eval_metrics['avg_episode_length']:.1f}")
    print(f"    Avg return: {eval_metrics['avg_return']:.2f}")
    assert 'success_rate' in eval_metrics
    assert 'avg_episode_length' in eval_metrics
    assert 'avg_return' in eval_metrics
    
    print("\n" + "="*60)
    print("All scripts passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_ps_dqn()

