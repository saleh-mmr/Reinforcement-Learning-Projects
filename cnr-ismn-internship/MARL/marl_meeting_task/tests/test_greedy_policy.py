import numpy as np

from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.config import seed


def greedy_action(agent_pos, goal_pos, grid_size):
    """Choose action that minimizes Manhattan distance to goal.
    If already on goal, return stay.
    """
    ax, ay = agent_pos
    gx, gy = goal_pos
    
    # If already on goal, stay
    if ax == gx and ay == gy:
        return 4  # stay
    
    # Calculate distance for each action
    # Action 0: up (x-1), Action 1: down (x+1), Action 2: left (y-1), Action 3: right (y+1)
    distances = []

    # Up
    new_x, new_y = max(0, ax - 1), ay
    distances.append(abs(new_x - gx) + abs(new_y - gy))

    # Down
    new_x, new_y = min(grid_size - 1, ax + 1), ay
    distances.append(abs(new_x - gx) + abs(new_y - gy))

    # Left
    new_x, new_y = ax, max(0, ay - 1)
    distances.append(abs(new_x - gx) + abs(new_y - gy))

    # Right
    new_x, new_y = ax, min(grid_size - 1, ay + 1)
    distances.append(abs(new_x - gx) + abs(new_y - gy))

    # Find minimum distance
    min_dist = min(distances)

    # Randomly choose among equally good actions
    best_actions = [i for i, d in enumerate(distances) if d == min_dist]
    best_action = np.random.choice(best_actions)

    return best_action



def main():
    print("Testing greedy sanity policy")
    print("=" * 50)
    
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    
    # Test multiple episodes
    n_episodes = 10
    step_counts = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        
        if episode == 0:
            print(f"\nEpisode {episode+1} - Initial state:")
            env.render()
        
        step = 0
        while step < env.max_steps:
            # Get greedy actions for each agent
            actions = {}
            for i in range(env.n_agents):
                agent_pos = env.agent_pos[i]
                actions[i] = greedy_action(agent_pos, env.goal_pos, env.grid_size)
            
            obs, reward, terminated, truncated, info = env.step(actions)
            step += 1
            
            if episode == 0 and step <= 3:
                print(f"\nStep {step}:")
                action_names = {i: env.ACTIONS[actions[i]] for i in range(env.n_agents)}
                print(f"  Actions: {action_names}")
                env.render()
            
            if terminated:
                step_counts.append(step)
                if episode == 0:
                    print(f"\nSuccess! Reached goal in {step} steps")
                    print(f"Final state:")
                    env.render()
                break
            
            if truncated:
                step_counts.append(step)
                print(f"\nEpisode {episode+1} failed: Truncated at {step} steps")
                break
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Successes: {len([s for s in step_counts if s < env.max_steps])}")
    print(f"  Step counts: {step_counts}")
    if step_counts:
        print(f"  Average steps: {np.mean(step_counts):.1f}")
        print(f"  Min steps: {min(step_counts)}")
        print(f"  Max steps: {max(step_counts)}")
    
    # Sanity check
    print("\n" + "=" * 50)
    if all(s <= 15 for s in step_counts):
        print("SANITY CHECK PASSED: All episodes succeeded in ≤15 steps")
    else:
        print("SANITY CHECK FAILED: Some episodes took >15 steps")
        print("   This may indicate an environment bug!")


if __name__ == "__main__":
    main()

