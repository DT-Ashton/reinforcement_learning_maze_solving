import numpy as np


class MazeGenerator:
    def __init__(self, size: int, seed: int = 42):
        self.size = size
        self.seed = seed
        self.grid = np.zeros((2 * size + 1, 2 * size + 1), dtype=np.int32)

    def generate(self) -> np.ndarray:
        self.grid = np.zeros((2 * self.size + 1, 2 * self.size + 1), dtype=np.int32)
        for r in range(self.size):
            for c in range(self.size):
                self.grid[2 * r + 1, 2 * c + 1] = 1

        rng = np.random.default_rng(self.seed)
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                rng.shuffle(neighbors)
                nr, nc, dr, dc = neighbors[0]
                self.grid[2 * r + 1 + dr, 2 * c + 1 + dc] = 1
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        return self.grid

    def get_start(self) -> tuple:
        return (0, 0)

    def get_goal(self) -> tuple:
        return (self.size - 1, self.size - 1)
