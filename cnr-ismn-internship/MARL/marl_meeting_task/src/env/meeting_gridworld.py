import numpy as np
from src import config


class MeetingGridworldEnv:
    ACTIONS = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
        4: "stay",
    }

    def __init__(self, grid_size=5, n_agents=2, max_steps=50):
        assert n_agents >= 2, "Environment requires at least 2 agents."

        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps

        # Environment-owned RNG
        self.rng = np.random.default_rng(config.seed)

        # Episode state
        self.agent_pos = {i: (None,None) for i in range(self.n_agents)}
        self.goal_pos = None
        self.step_count = None


    def __str__(self):
        return f'{self.grid_size}x{self.grid_size} environment with {self.n_agents} agents'


    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(config.seed)

        # Sample distinct cells for all agents + goal
        n_entities = self.n_agents + 1
        all_cells = self.grid_size * self.grid_size
        chosen = self.rng.choice(all_cells, size=n_entities, replace=False)

        def idx_to_xy(idx):
            pos = (idx // self.grid_size, idx % self.grid_size)
            return pos

        agent_cells = chosen[:self.n_agents]
        goal_cell = chosen[-1]

        for i, cell in enumerate(agent_cells):
            self.agent_pos[i] = idx_to_xy(cell)

        self.goal_pos = idx_to_xy(goal_cell)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, actions):
        assert self.step_count is not None, "Call reset() before step()."
        assert isinstance(actions, dict), "Actions must be a dict keyed by agent id."
        assert len(actions) == self.n_agents, "Must provide one action per agent."
        assert all(i in actions for i in range(self.n_agents)), "Actions must contain keys for all agent ids."

        def move(x, y, action):
            if action == 0:      # up
                x -= 1
            elif action == 1:    # down
                x += 1
            elif action == 2:    # left
                y -= 1
            elif action == 3:    # right
                y += 1
            elif action == 4:    # stay
                pass

            x = max(0, min(self.grid_size - 1, x))
            y = max(0, min(self.grid_size - 1, y))
            return (x, y)

        # Simultaneous movement
        new_positions = {}
        for i in range(self.n_agents):
            x, y = self.agent_pos[i]
            new_positions[i] = move(x, y, actions[i])

        self.agent_pos = new_positions
        self.step_count += 1

        # Success: ALL agents on the goal
        success = all(
            self.agent_pos[i] == self.goal_pos
            for i in range(self.n_agents)
        )

        reward = 10.0 if success else -0.01
        terminated = success
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------
    # Observation helper
    # --------------------------------------------------
    def _get_obs(self):
        obs = {}

        for i in range(self.n_agents):
            own_x, own_y = self.agent_pos[i]

            # Observation: [own_x, own_y, goal_x, goal_y]
            # Does NOT include other agent positions
            obs[i] = np.array(
                [own_x, own_y, *self.goal_pos],
                dtype=np.int64
            )

        return obs

    # --------------------------------------------------
    # Render (debug tool)
    # --------------------------------------------------
    def render(self):
        """Render the gridworld state as ASCII art.
        A = agent, G = goal, . = empty cell
        """
        assert self.step_count is not None, "Call reset() before render()."
        
        # Create empty grid
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark goal position
        if self.goal_pos is not None:
            gx, gy = self.goal_pos
            grid[gx][gy] = 'G'
        
        # Mark agent positions (overwrites goal if same cell)
        for i in range(self.n_agents):
            if self.agent_pos[i] is not None:
                ax, ay = self.agent_pos[i]
                grid[ax][ay] = 'A'
        
        # Print grid
        for row in grid:
            print(' '.join(row))
        print()  # Empty line after grid
