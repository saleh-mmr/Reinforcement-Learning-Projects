# src/env/switch_door_env.py
import numpy as np


class SwitchDoorEnv:
    """
    Cooperative Switch–Door task (MAPPO-friendly, CTDE).

    Key rule (role specialization):
    - Door is open IFF at least one agent is standing on the switch cell.
    - If nobody is on the switch, the door closes immediately.
    """

    ACTIONS = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
        4: "stay",
        # NOTE: No toggle/press action. Holding the switch is position-based.
    }

    def __init__(
        self,
        grid_size: int = 5,
        n_agents: int = 2,
        max_steps: int = 30,
        switch_pos=(1, 1),
        door_pos=(3, 3),
    ):
        assert n_agents == 2, "This env is defined for exactly 2 agents."
        assert grid_size == 5, "This env is specified as 5x5 for the project."
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps

        # Fixed positions (do not change across episodes)
        self.switch_pos = tuple(switch_pos)
        self.door_pos = tuple(door_pos)

        # RNG (seeded in reset)
        self.rng = np.random.default_rng()

        # Episode state
        self.agent_pos = {i: (None, None) for i in range(self.n_agents)}
        self.switch_on = False   # True if any agent on switch
        self.door_state = False  # door_state == switch_on
        self.step_count = None

    def __str__(self):
        return f"{self.grid_size}x{self.grid_size} SwitchDoorEnv with {self.n_agents} agents"

    # -----------------------------
    # Core helpers
    # -----------------------------
    def _clamp(self, x, y):
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        return x, y

    def _move(self, x, y, action):
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
        else:
            raise ValueError(f"Invalid action {action}. Must be in [0..4].")

        return self._clamp(x, y)

    def _update_switch_and_door(self):
        # Switch is ON if any agent stands on switch cell
        self.switch_on = any(self.agent_pos[i] == self.switch_pos for i in range(self.n_agents))
        # Door is open iff switch is on
        self.door_state = self.switch_on




    # -----------------------------
    # API
    # -----------------------------
    def reset(self, seed=None):
        """
        Returns:
          obs_dict: {0: obs0, 1: obs1}
          info: {}
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample distinct cells for all agents + door + switch
        n_entities = self.n_agents + 2
        all_cells = self.grid_size * self.grid_size
        chosen = self.rng.choice(all_cells, size=n_entities, replace=False)

        agent_cells = chosen[:self.n_agents]
        door_cell = chosen[-1]
        switch_cell = chosen[-2]

        def idx_to_xy(idx):
            pos = (idx // self.grid_size, idx % self.grid_size)
            return pos

        for i, cell in enumerate(agent_cells):
            self.agent_pos[i] = idx_to_xy(cell)

        self.door_pos = idx_to_xy(door_cell)
        self.switch_pos = idx_to_xy(switch_cell)
        self.step_count = 0
        self._update_switch_and_door()

        obs = self._get_obs()
        info = {}
        return obs, info



    def step(self, actions):
        """
        actions: {0: a0, 1: a1}

        Returns:
          obs_dict, reward, terminated, truncated, info
        """
        assert self.step_count is not None, "Call reset() before step()."
        assert isinstance(actions, dict), "Actions must be a dict keyed by agent id."
        assert set(actions.keys()) == set(range(self.n_agents)), "Actions must contain keys {0,1}."

        # Simultaneous movement (compute from old positions)
        new_positions = {}
        for i in range(self.n_agents):
            x, y = self.agent_pos[i]
            new_positions[i] = self._move(x, y, int(actions[i]))

        self.agent_pos = new_positions
        self.step_count += 1

        # Update switch/door (position-based)
        self._update_switch_and_door()

        # Success: door open AND any agent on door cell
        agent_on_door = any(self.agent_pos[i] == self.door_pos for i in range(self.n_agents))
        success = self.door_state and agent_on_door

        reward = 10.0 if success else -0.05
        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    # -----------------------------
    # Observations (actors)
    # -----------------------------
    def _get_obs(self):
        """
        Actor observation (NO other-agent position):

        o_i = [
            own_x, own_y,
            x_door,   y_door,
            x_switch, y_switch,
            door_state,   # 0/1 (same as switch_on)
        ]
        """
        obs = {}
        for i in range(self.n_agents):
            own_x, own_y = self.agent_pos[i]
            obs[i] = np.array([own_x, own_y, *self.door_pos, *self.switch_pos, int(self.door_state)], dtype=np.int64)

        return obs

    # -----------------------------
    # State (critic, centralized)
    # -----------------------------
    def get_state(self):
        """
        Centralized state for critic:

        s = [x0, y0, x1, y1, x_door, y_door, x_switch, y_switch, door_state]
        """
        assert self.step_count is not None, "Call reset() before get_state()."
        x0, y0 = self.agent_pos[0]
        x1, y1 = self.agent_pos[1]
        return np.array([x0, y0, x1, y1, *self.door_pos, *self.switch_pos, int(self.door_state)], dtype=np.int64)

    # -----------------------------
    # Render (debug)
    # -----------------------------
    def render(self):
        """
        ASCII render:
          . = empty
          S = switch
          D = door
          0/1 = agents
        """
        assert self.step_count is not None, "Call reset() before render()."
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        sx, sy = self.switch_pos
        dx, dy = self.door_pos
        grid[sx][sy] = "S"
        grid[dx][dy] = "D"

        for i in range(self.n_agents):
            ax, ay = self.agent_pos[i]
            grid[ax][ay] = str(i)

        print(f"step={self.step_count} door_state={int(self.door_state)}")
        for row in grid:
            print(" ".join(row))
        print()
