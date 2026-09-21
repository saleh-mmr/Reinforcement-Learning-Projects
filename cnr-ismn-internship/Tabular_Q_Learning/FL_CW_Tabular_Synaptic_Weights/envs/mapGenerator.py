import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
import numpy as np
from collections import deque
import utils.config as config


def is_solvable(grid):
    h, w = grid.shape
    start = (0, 0)
    goal = (h - 1, w - 1)

    queue = deque([start])
    visited = {start}

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                    0 <= nr < h and
                    0 <= nc < w and
                    (nr, nc) not in visited and
                    grid[nr, nc] != "H"
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False


class MapGenerator:
    def __init__(self, height, width, p=0.8):
        """
        Constructing (calling) MapGenerator will return the generated map (a list of strings).
        We create a temporary instance to reuse the existing instance method generate_random_map_rect.
        """
        self.height = height
        self.width = width
        self.p = p

    def generate_random_map_rect(self):
        rng = np.random.default_rng(config.seed)

        while True:
            grid = rng.choice(
                ["F", "H"],
                size=(self.height, self.width),
                p=[self.p, 1 - self.p]
            )

            grid[0, 0] = "S"
            grid[-1, -1] = "G"

            if is_solvable(grid):
                return ["".join(row) for row in grid]