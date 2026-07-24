import numpy as np
import gymnasium

from maze.generator import MazeGenerator
from visualization.pygame_renderer import PygameRenderer


class MazeEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, size: int = 10, seed: int = 42, render_mode: str = None,
                 grid: np.ndarray = None):
        super().__init__()
        self.size = size
        self.seed_val = seed
        self.render_mode = render_mode
        self.max_steps = size * size * 4

        self.action_space = gymnasium.spaces.Discrete(4)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=3,
            shape=((2 * size + 1) * (2 * size + 1),),
            dtype=np.float32,
        )

        self._generator = MazeGenerator(size, seed)
        self._agent_cell = (0, 0)
        self._goal_cell = (size - 1, size - 1)
        self._step_count = 0

        if grid is not None:
            expected_shape = (2 * size + 1, 2 * size + 1)
            assert grid.shape == expected_shape, (
                f"grid shape {grid.shape} does not match expected {expected_shape} for size={size}"
            )
            self._base_grid = grid.copy().astype(np.int32)
            self._preset_grid = True
        else:
            self._base_grid = None
            self._preset_grid = False

        self.window = None
        self.clock = None

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        if seed is not None and not self._preset_grid:
            self.seed_val = seed
            self._generator = MazeGenerator(self.size, seed)
            self._generator.generate()
            self._base_grid = self._generator.grid.copy()
        elif self._base_grid is None:
            self._generator.generate()
            self._base_grid = self._generator.grid.copy()

        self._agent_cell = (0, 0)
        self._step_count = 0

        obs = self._get_obs()
        self.render()
        return obs, {}

    def step(self, action: int):
        action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = action_map[action]
        r, c = self._agent_cell
        nr, nc = r + dr, c + dc

        if (0 <= nr < self.size and 0 <= nc < self.size
                and self._base_grid[2 * r + 1 + dr, 2 * c + 1 + dc] == 1):
            self._agent_cell = (nr, nc)

        self._step_count += 1
        terminated = self._agent_cell == self._goal_cell
        truncated = self._step_count >= self.max_steps
        reward = 100.0 if terminated else -1.0

        obs = self._get_obs()
        self.render()
        return obs, reward, terminated, truncated, {"step": self._step_count}

    def _get_obs(self) -> np.ndarray:
        grid = self._base_grid.copy()
        gr, gc = self._goal_cell
        grid[2 * gr + 1, 2 * gc + 1] = 3
        ar, ac = self._agent_cell
        grid[2 * ar + 1, 2 * ac + 1] = 2
        return grid.flatten().astype(np.float32)

    def render(self) -> None:
        if self.render_mode != "human":
            return
        import pygame
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                PygameRenderer.get_window_size(self.size)
            )
            pygame.display.set_caption("Maze RL")
            self.clock = pygame.time.Clock()
        PygameRenderer.draw(self.window, self._get_obs(), self.size)
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self.window is not None:
            import pygame
            pygame.quit()
        self.window = None
        self.clock = None
