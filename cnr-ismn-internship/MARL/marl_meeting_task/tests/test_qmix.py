"""
Smoke test for QMIX implementation.
Verifies basic functionality without full training.
"""

import numpy as np
import torch
from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.algos.qmix import QMIX


def test_qmix():
    """Test QMIX basic functionality."""
    print("="*60)
    print("QMIX Smoke Test")
    print("="*60)
    
    # Create environment
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    print("\n✓ Environment created")
    
    # Initialize QMIX
    qmix = QMIX(
        n_agents=2,
        input_dim=4,  # Base observation: [own_x, own_y, goal_x, goal_y]
        state_dim=6,
        num_actions=5,
        hidden_dim=64,
        mixing_hidden_dim=64,
        learning_rate=3e-4,
        memory_capacity=1000,  # Small for testing
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=1000,
        batch_size=32,
        target_update_freq=100,
    )
    print("✓ QMIX initialized")
    
    # Test action selection (decentralized)
    obs, info = env.reset(seed=2025)
    actions = qmix.select_actions(obs)
    print(f"✓ Action selection works: {actions}")
    assert isinstance(actions, dict)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions.values())
    
    # Test storing transitions
    state = qmix._get_global_state(env)
    next_obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    next_state = qmix._get_global_state(env)
    
    qmix.store_transition(
        state=state,
        obs=obs,
        actions=actions,
        reward=reward,
        next_state=next_state,
        next_obs=next_obs,
        done=done
    )
    print(f"✓ Stored transition (buffer size: {len(qmix.replay_memory)})")
    assert len(qmix.replay_memory) == 1
    
    # Fill buffer a bit
    for _ in range(50):
        obs = next_obs
        state = next_state
        actions = qmix.select_actions(obs)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        next_state = qmix._get_global_state(env)
        
        qmix.store_transition(
            state=state,
            obs=obs,
            actions=actions,
            reward=reward,
            next_state=next_state,
            next_obs=next_obs,
            done=done
        )
        qmix.total_steps += 1
        if done:
            obs, info = env.reset(seed=None)
            state = qmix._get_global_state(env)
    
    print(f"✓ Filled buffer (size: {len(qmix.replay_memory)})")
    
    # Test training step
    loss = qmix.train_step()
    print(f"✓ Training step works (loss: {loss:.4f})")
    assert loss is not None
    assert isinstance(loss, float)
    
    # Test epsilon decay
    epsilon = qmix.get_epsilon()
    print(f"✓ Epsilon schedule works (current epsilon: {epsilon:.3f})")
    assert 0 <= epsilon <= 1.0
    
    # Test target network update
    qmix.update_target_networks()
    print("✓ Target network update works")
    
    # Test evaluation
    eval_metrics = qmix.evaluate(env, n_episodes=5, max_steps=50)
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
    test_qmix()

