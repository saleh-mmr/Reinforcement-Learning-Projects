import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.config import seed


def test_reset_uniqueness():
    """Test 1: After reset, agent positions are unique and different from goal"""
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    
    # Reset multiple times to check uniqueness
    for _ in range(10):
        obs, info = env.reset(seed=seed)
        
        # Check agent 0 != agent 1
        assert env.agent_pos[0] != env.agent_pos[1], "Agents should be at different positions"
        
        # Check agents != goal
        assert env.agent_pos[0] != env.goal_pos, "Agent 0 should not be on goal"
        assert env.agent_pos[1] != env.goal_pos, "Agent 1 should not be on goal"
    
    print("Test 1 passed: Reset uniqueness")


def test_boundary_movement():
    """Test 2: Agents cannot move outside boundaries"""
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    obs, info = env.reset(seed=seed)
    
    # Manually place agent 0 at (0, 0)
    env.agent_pos[0] = (0, 0)
    
    # Try to move up (action 0) - should stay at (0, 0)
    actions = {0: 0, 1: 4}  # agent 0: up, agent 1: stay
    obs, reward, terminated, truncated, info = env.step(actions)
    assert env.agent_pos[0] == (0, 0), "Agent at (0,0) should not move up"
    
    # Try to move left (action 2) - should stay at (0, 0)
    actions = {0: 2, 1: 4}  # agent 0: left, agent 1: stay
    obs, reward, terminated, truncated, info = env.step(actions)
    assert env.agent_pos[0] == (0, 0), "Agent at (0,0) should not move left"
    
    print("Test 2 passed: Boundary movement")


def test_success_reward():
    """Test 3: When all agents are on goal and take stay, reward=10.0 and terminated=True"""
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    obs, info = env.reset(seed=seed)
    
    # Manually place both agents on the goal
    goal_pos = env.goal_pos
    env.agent_pos[0] = goal_pos
    env.agent_pos[1] = goal_pos
    
    # Take stay action for both agents
    actions = {0: 4, 1: 4}  # both agents: stay
    obs, reward, terminated, truncated, info = env.step(actions)
    
    assert reward == 10.0, f"Expected reward 10.0, got {reward}"
    assert terminated == True, "Should be terminated when all agents on goal"
    
    print("Test 3 passed: Success reward")


def test_timeout():
    """Test 4: After max_steps without success, truncated=True"""
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=5)  # Small max_steps for testing
    
    obs, info = env.reset(seed=seed)
    
    # Run max_steps steps without reaching goal (just take random/stay actions)
    for step in range(env.max_steps):
        actions = {0: 4, 1: 4}  # both agents: stay (won't reach goal)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Should not terminate before max_steps
        if step < env.max_steps - 1:
            assert terminated == False, "Should not terminate before max_steps"
        else:
            # On the last step, should be truncated
            assert truncated == True, "Should be truncated after max_steps"
    
    print("Test 4 passed: Timeout")


if __name__ == "__main__":
    print("Running environment scripts...\n")
    
    test_reset_uniqueness()
    test_boundary_movement()
    test_success_reward()
    test_timeout()
    
    print("\nAll scripts passed!")

