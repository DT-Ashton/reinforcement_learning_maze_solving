# Phase 1: Maze Generation + Environment
**Covers spec:** FR-01, FR-02, FR-03, FR-04, FR-05, FR-06
**User stories:** [P1] Reproducible maze generator; [P1] Gymnasium-compatible MazeEnv; [P2] Pygame visualization demo

---

## Files to Create

### Step 1 — Directory Scaffold
Create the following empty directories (each needs a `__init__.py` where listed):
```
maze/
env/
algorithms/
visualization/
metrics/
configs/
models/          # no __init__.py needed — stores .zip/.npy files
logs/            # no __init__.py needed — stores TensorBoard event files
```

### `requirements.txt`
**Purpose:** Pin all runtime dependencies.

```
gymnasium>=0.29.0
numpy>=1.24.0
pygame>=2.5.0
stable-baselines3>=2.0.0
torch>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
tensorboard>=2.13.0
jupyter>=1.0.0
pandas>=2.0.0
```

---

### `maze/generator.py`
**Purpose:** Generate a perfect maze (exactly one path between any two cells) using Recursive Backtracker DFS, stored as a (2N+1)x(2N+1) numpy array.

**Classes/Functions:**

- `MazeGenerator(size: int, seed: int = 42)`
  - `self.size`: logical grid dimension N (cells are `N x N`).
  - `self.seed`: stored for reproducibility.
  - `self.grid`: numpy array shape `(2*size+1, 2*size+1)`, dtype `np.int32`, initialized all zeros (all walls).
  - Implementation note: do NOT call `generate()` in `__init__`; keep construction and generation separate so the object can be inspected before carving.

- `generate() -> np.ndarray`
  - Carves a perfect maze into `self.grid` using iterative DFS (stack-based Recursive Backtracker) and returns the completed grid.
  - Implementation steps:
    1. Reset `self.grid` to all zeros.
    2. Mark every cell center `(2r+1, 2c+1)` as `1` (passage) for `r, c` in `range(size)`.
    3. Seed numpy: `rng = np.random.default_rng(self.seed)`.
    4. Start at cell `(0, 0)`. Maintain a `visited` set and a `stack`.
    5. At each step: find unvisited neighbors (UP=(-1,0), DOWN=(1,0), LEFT=(0,-1), RIGHT=(0,1)), shuffle them with `rng.shuffle(neighbors)`, pick first unvisited, carve wall between current and neighbor by setting the wall grid slot to `1`, push neighbor, mark visited.
    6. If no unvisited neighbors, pop the stack.
    7. Continue until stack is empty.
  - Wall carving formula: wall between cell `(r, c)` and neighbor `(r+dr, c+dc)` is at grid position `(2r+1+dr, 2c+1+dc)`. Set that slot to `1`.
  - Returns `self.grid` (the same array, mutated in-place).
  - Implementation note: use iterative stack (not Python recursion) to avoid `RecursionError` on large mazes (N=30 → 900 cells deep possible).

- `get_start() -> tuple[int, int]`
  - Returns logical cell `(0, 0)` — always the start.

- `get_goal() -> tuple[int, int]`
  - Returns logical cell `(size-1, size-1)` — always bottom-right.

---

### `maze/__init__.py`
**Purpose:** Public API for the maze package.

```python
from maze.generator import MazeGenerator

__all__ = ["MazeGenerator"]
```

---

### `env/maze_env.py`
**Purpose:** Full Gymnasium environment wrapping MazeGenerator, implementing the RL loop with correct 5-tuple step returns and optional Pygame rendering.

**Classes/Functions:**

- `MazeEnv(size: int = 10, seed: int = 42, render_mode: str = None)`
  - Inherits `gymnasium.Env`.
  - Class-level: `metadata = {"render_modes": ["human"], "render_fps": 10}`.
  - `self.size`: N.
  - `self.seed_val`: stored seed (do NOT shadow `self.seed` which belongs to gymnasium).
  - `self.render_mode`: stores the render mode string.
  - `self.action_space = gymnasium.spaces.Discrete(4)` — actions: `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`.
  - `self.observation_space = gymnasium.spaces.Box(low=0, high=3, shape=((2*size+1)*(2*size+1),), dtype=np.float32)`.
  - **Fix (reviewer):** Use `np.float32` — SB3's MlpPolicy requires float observations; np.int32 causes dtype warnings or assertion errors in stricter SB3 builds.
  - `self.max_steps = size * size * 4`.
  - `self._generator = MazeGenerator(size, seed)` — but do NOT call `.generate()` here; call it in `reset()`.
  - `self._base_grid`: initialized to `None` in `__init__`; set to the carved maze (walls only) on first `reset()`. After that, reused across episodes unless `seed` changes.
  - `self._agent_cell = (0, 0)`: tuple `(row, col)` — initialized in `__init__`, reset in `reset()`. (**Fix:** prevents `AttributeError` if `_get_obs()` or `step()` is called before `reset()`.)
  - `self._goal_cell = (size-1, size-1)`: always bottom-right cell — fixed for the lifetime of the env.
  - `self._step_count = 0`: int — initialized in `__init__`, reset to 0 on each `reset()`. (**Fix:** same reason as `_agent_cell`.)
  - Pygame attributes: `self.window = None`, `self.clock = None` — initialized lazily in `render()`.
  - Implementation note: do NOT `import pygame` at the top of this module. Use a lazy import inside `render()` to avoid crashes on headless systems.

- `reset(seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]`
  - FIRST LINE must be: `super().reset(seed=seed)` — this seeds gymnasium's internal RNG and is required for SB3 compatibility.
  - **Fix (reviewer — CRITICAL):** Only regenerate the maze when necessary:
    - If `seed` is not None: update `self.seed_val = seed`, recreate `self._generator = MazeGenerator(self.size, seed)`, call `self._generator.generate()`, store as `self._base_grid = self._generator.grid.copy()`.
    - If `self._base_grid is None` (first call ever): call `self._generator.generate()`, store as `self._base_grid = self._generator.grid.copy()`.
    - Otherwise (subsequent resets with same seed): skip generate() — just reset agent position.
    - **Why:** If the maze changes every episode, the Q-table entries for state `s` refer to different physical cells across episodes. Tabular agents learn across randomly changing MDPs and will never converge.
  - Reset agent to `self._agent_cell = (0, 0)`, `self._step_count = 0`.
  - Build obs via `_get_obs()`.
  - Call `self.render()` — shows initial maze state; no-op when `render_mode != "human"`. (**Fix:** without this, demo starts blank — first rendered frame is after the first step, not after reset.)
  - Return `(obs, {})`.

- `step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]`
  - Action map: `0=UP → dr=-1,dc=0`, `1=DOWN → dr=1,dc=0`, `2=LEFT → dr=0,dc=-1`, `3=RIGHT → dr=0,dc=1`.
  - Compute candidate new cell: `nr = self._agent_cell[0] + dr`, `nc = self._agent_cell[1] + dc`.
  - Boundary check: `0 <= nr < size` and `0 <= nc < size`.
  - Wall check: the wall slot between current cell `(r,c)` and `(nr,nc)` is `self._base_grid[2*r+1+dr, 2*c+1+dc]`. If that slot is `1` (passage), the move is valid.
  - If valid: update `self._agent_cell = (nr, nc)`.
  - If invalid (wall or out-of-bounds): agent stays, no extra penalty.
  - Increment `self._step_count`.
  - Compute reward: `+100` if agent is now at `self._goal_cell`, else `-1`.
  - `terminated = (self._agent_cell == self._goal_cell)`.
  - `truncated = (self._step_count >= self.max_steps)`.
  - Call `self.render()` — but render() is a no-op when `render_mode != "human"`.
  - Return `(_get_obs(), reward, terminated, truncated, {"step": self._step_count})`.
  - Implementation note: call `render()` at end of step (not inside the reward logic), so the final goal frame is also displayed.

- `_get_obs() -> np.ndarray`
  - Start from a copy of `self._base_grid`.
  - Set `grid[2*self._goal_cell[0]+1, 2*self._goal_cell[1]+1] = 3`.
  - Set `grid[2*self._agent_cell[0]+1, 2*self._agent_cell[1]+1] = 2`.
  - Return `grid.flatten().astype(np.float32)`. (**Fix:** match observation_space dtype=np.float32)
  - Implementation note: copy the base grid each call — do not modify `self._base_grid` in-place.

- `render() -> None`
  - If `self.render_mode != "human"`: return immediately.
  - Lazy import: `import pygame` here (not at module top).
  - If `self.window is None`:
    - `pygame.init()`.
    - `self.window = pygame.display.set_mode(PygameRenderer.get_window_size(self.size))`. (**Fix:** result of `set_mode()` must be assigned to `self.window`; passing `None` to `PygameRenderer.draw()` crashes with `TypeError`.)
    - `self.clock = pygame.time.Clock()`. (**Fix:** `pygame.time.Clock()` not `pygame.Clock()`.)
  - Delegate all drawing to `PygameRenderer.draw(self.window, self._get_obs(), self.size)` (see `visualization/pygame_renderer.py`).
  - Call `pygame.event.pump()` and `pygame.display.flip()`.
  - Call `self.clock.tick(self.metadata["render_fps"])` — FPS throttle happens HERE, not in step().

- `close() -> None`
  - `if self.window is not None: import pygame; pygame.quit()`. (**Fix:** `import pygame` inside the guard — `pygame` is only in scope inside `render()` (lazy import); calling `pygame.quit()` without re-importing raises `NameError` even though the module is already in `sys.modules`. `self.window is not None` is the correct predicate, since `self.window` is only assigned inside `render()` after `pygame.init()`.)
  - `self.window = None; self.clock = None`.

---

### `env/__init__.py`
**Purpose:** Export MazeEnv and register it with Gymnasium so `gym.make("MazeEnv-v0")` works.

```python
from env.maze_env import MazeEnv
import gymnasium

gymnasium.register(
    id="MazeEnv-v0",
    entry_point="env.maze_env:MazeEnv",
    kwargs={"size": 10, "seed": 42, "render_mode": None},
)

__all__ = ["MazeEnv"]
```

Implementation note: `gymnasium.register()` is called at import time. In `main.py`, do `import env` (or `from env import MazeEnv`) before any `gym.make()` call. The `entry_point` string `"env.maze_env:MazeEnv"` must exactly match the module path relative to the project root.

---

### `visualization/pygame_renderer.py`
**Purpose:** Stateless drawing utility called by MazeEnv.render(); keeps all pygame surface/draw logic out of the environment class.

**Classes/Functions:**

- `PygameRenderer` (class with only static/class methods — no instance state needed)

- `PygameRenderer.draw(surface, obs: np.ndarray, size: int, cell_px: int = 30) -> None`
  - Receives the pygame surface, the flattened observation array, and the logical maze size N.
  - Reshapes `obs` to `(2*size+1, 2*size+1)`.
  - Iterates over every `(row, col)` in the grid and draws a rectangle of size `cell_px x cell_px`.
  - Color map:
    - `0` (wall): `(40, 40, 40)` — dark grey
    - `1` (passage): `(255, 255, 255)` — white
    - `2` (agent): `(0, 120, 255)` — bright blue
    - `3` (goal): `(220, 50, 50)` — red
  - Rectangle for grid cell `(row, col)`: `pygame.Rect(col * cell_px, row * cell_px, cell_px, cell_px)`.
  - Fills the surface with `(0,0,0)` black before drawing all rects.
  - Implementation note: this method does NOT call `pygame.display.flip()` or `clock.tick()` — those are the environment's responsibility.

- `PygameRenderer.get_window_size(size: int, cell_px: int = 30) -> tuple[int, int]`
  - Returns `((2*size+1) * cell_px, (2*size+1) * cell_px)` — the pixel dimensions for the pygame window.
  - Used in `MazeEnv.render()` when creating the window: `pygame.display.set_mode(PygameRenderer.get_window_size(self.size))`.

---

## Implementation Order

1. Create all package directories and empty `__init__.py` files.
2. Write `requirements.txt`.
3. Implement `maze/generator.py` — `MazeGenerator` with iterative DFS.
4. Write `maze/__init__.py`.
5. Implement `visualization/pygame_renderer.py` — needed before env so render() has something to call.
6. Implement `env/maze_env.py` — core environment.
7. Write `env/__init__.py` with `gymnasium.register()`.
8. Smoke test: run the acceptance checks below.

---

## Acceptance Check

- [ ] `python -c "from maze.generator import MazeGenerator; g = MazeGenerator(10, 42); grid = g.generate(); print(grid.shape)"` prints `(21, 21)`.
- [ ] `python -c "from maze.generator import MazeGenerator; import numpy as np; g = MazeGenerator(10, 42); g1 = g.generate().copy(); g2 = MazeGenerator(10, 42).generate(); print(np.array_equal(g1, g2))"` prints `True` (seed reproducibility).
- [ ] `python -c "import env; import gymnasium; env_inst = gymnasium.make('MazeEnv-v0'); obs, info = env_inst.reset(); print(obs.shape)"` prints `(441,)` (for N=10: (21*21)=441).
- [ ] `python -c "import env; import gymnasium; from gymnasium.utils.env_checker import check_env; e = gymnasium.make('MazeEnv-v0').unwrapped; check_env(e)"` completes with no errors.
- [ ] `python -c "import env; import gymnasium; e = gymnasium.make('MazeEnv-v0'); obs, _ = e.reset(); obs2, r, term, trunc, info = e.step(1); print(r, term, trunc)"` prints without exception (reward is -1 or 100, booleans are False/True).
