# Phase 6: Unseen Maze Live Demo (Q-Learning, SARSA, Dyna-Q)
**Covers spec:** FR-16, FR-17, FR-18, FR-19, FR-20
**User story:** [P2] As a student, I want to see Q-Learning, SARSA, and Dyna-Q learn live on a maze never used during development, so I can visually compare convergence speed.

DQN is **out of scope** for this demo (retraining from scratch on an unseen maze takes ~17 min — too slow for a live SPACE-to-start demo). The existing DQN demo on the seed=42 maze is unchanged.

---

## Files to Create / Modify

### `maze/pool_generator.py` (new)
**Purpose:** Generate a pool of random mazes of varying sizes and persist them to `mazes/` for later use by the live demo.

**Functions:**

- `generate_pool(sizes: list[int] = [5, 10, 15], n_per_size: int = 3, base_seed: int = 100, out_dir: str = "mazes") -> list[str]`
  - For each `size` in `sizes`, for `i` in `range(n_per_size)`:
    - `seed = base_seed + size * 100 + i` (deterministic, reproducible, never collides across sizes).
    - `gen = MazeGenerator(size, seed)`, `grid = gen.generate()`.
    - Write JSON to `{out_dir}/maze_{size}x{size}_seed{seed}.json`:
      ```json
      {
        "size": 10,
        "seed": 1100,
        "start": [0, 0],
        "goal": [9, 9],
        "grid": [[...], [...], ...]
      }
      ```
      (`grid.tolist()` — `(2*size+1) x (2*size+1)` int array, same encoding as `MazeGenerator.grid`.)
  - After generating all files, write `{out_dir}/index.json`: a list of `{"file": ..., "size": ..., "seed": ...}` entries (no grid data — quick lookup for the demo picker).
  - Create `out_dir` with `os.makedirs(out_dir, exist_ok=True)` if missing.
  - Return list of written file paths.

- `if __name__ == "__main__":` — argparse with `--sizes` (nargs, default `[5, 10, 15]`), `--n-per-size` (default 3), `--base-seed` (default 100), `--out-dir` (default `"mazes"`). Calls `generate_pool(...)` and prints how many files were written.

**Run command:** `python -m maze.pool_generator` (or `python maze/pool_generator.py` thanks to the `sys.path` pattern used elsewhere — follow `test_function/` convention if running as a script directly).

---

### `env/maze_env.py` (modify)

- `__init__(self, size: int = 10, seed: int = 42, render_mode: str = None, grid: np.ndarray = None)`
  - New optional `grid` parameter.
  - If `grid is not None`:
    - Validate `grid.shape == (2*size+1, 2*size+1)` (`assert`, with a clear message).
    - `self._base_grid = grid.copy().astype(np.int32)`.
    - `self._preset_grid = True`.
    - Skip `MazeGenerator` construction (or still construct it lazily — not used for preset).
  - Else:
    - Existing behavior unchanged: `self._generator = MazeGenerator(size, seed)`, `self._base_grid = None`, `self._preset_grid = False`.

- `reset(self, seed: int = None, options: dict = None)` — update the maze-(re)generation branch only:
  ```python
  if seed is not None and not self._preset_grid:
      self.seed_val = seed
      self._generator = MazeGenerator(self.size, seed)
      self._generator.generate()
      self._base_grid = self._generator.grid.copy()
  elif self._base_grid is None:
      self._generator.generate()
      self._base_grid = self._generator.grid.copy()
  # else: keep existing self._base_grid (preset grid, or already generated) — Phase 1 reviewer fix preserved
  ```
  - Rest of `reset()` unchanged (`_agent_cell = (0, 0)`, `_step_count = 0`, etc).

### `env/maze_loader.py` (new)
**Purpose:** Load a `MazeEnv` from a maze pool JSON file.

```python
import json
import numpy as np
from env.maze_env import MazeEnv


def load_maze_env(path: str, render_mode: str = None) -> MazeEnv:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grid = np.array(data["grid"], dtype=np.int32)
    return MazeEnv(size=data["size"], render_mode=render_mode, grid=grid)
```

---

### `algorithms/dyna_q.py` (new)
**Purpose:** Dyna-Q agent — Q-Learning + a learned deterministic model + extra "planning" updates per real step. Goal: converge in far fewer real episodes than plain Q-Learning on a never-seen maze.

```python
import numpy as np

from algorithms.q_learning import QLearningAgent
from metrics.training_metrics import TrainingMetrics


class DynaQAgent(QLearningAgent):
    ALGO_NAME = "dyna_q"

    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.997,
                 epsilon_min: float = 0.01, planning_steps: int = 10):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.planning_steps = planning_steps
        self.model = {}  # (state, action) -> (reward, next_state, done)

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        super().update(s, a, r, s_next, done)
        self.model[(s, a)] = (r, s_next, done)

        if len(self.model) < self.planning_steps:
            return
        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            ps, pa = keys[np.random.randint(len(keys))]
            pr, ps_next, pdone = self.model[(ps, pa)]
            super().update(ps, pa, pr, ps_next, pdone)
```

- **Inherits `train()` from `QLearningAgent` unchanged** — same episode loop, `metrics.algo_name = self.ALGO_NAME`.
- Planning updates call `super().update(...)` (the plain Q-Learning TD rule) directly — they do NOT recurse into `DynaQAgent.update()` (which would re-store into the model and re-run planning, causing exponential blowup). This is naturally avoided because `super().update()` resolves to `QLearningAgent.update`, not `DynaQAgent.update`.
- `epsilon_decay=0.997` (slower than Q-Learning's `0.995` default) per researcher recommendation — mitigates premature exploitation from an overconfident model on a deterministic maze.
- `self.model` is fresh per `DynaQAgent` instance — since each unseen-maze run constructs a new agent, no stale-model risk across mazes.

### `algorithms/q_learning.py` (modify — minimal)
- Add class attribute `ALGO_NAME = "qlearning"`.
- In `train()`, change `algo_name="qlearning"` → `algo_name=self.ALGO_NAME` so `DynaQAgent` (subclass) reports `"dyna_q"` without overriding `train()`.

### `algorithms/sarsa.py` (modify — minimal, for consistency)
- Add class attribute `ALGO_NAME = "sarsa"` (not strictly required since SARSA isn't subclassed, but keeps the convention consistent — optional, low priority).

### `algorithms/__init__.py` (modify)
```python
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dqn_trainer import DQNTrainer
from algorithms.dyna_q import DynaQAgent

__all__ = ["QLearningAgent", "SARSAAgent", "DQNTrainer", "DynaQAgent"]
```

---

### `demo_unseen_maze.py` (new, root)
**Purpose:** Pygame demo — pick a maze from `mazes/`, show it statically, wait for SPACE, then train Q-Learning / SARSA / Dyna-Q live (from scratch) on that exact maze with rendering, and print a convergence comparison.

**Functions:**

- `pick_maze_file(mazes_dir: str = "mazes", path: str = None) -> str`
  - If `path` given, return it directly.
  - Else load `{mazes_dir}/index.json` and pick a random entry (`np.random.choice` or `random.choice`). Raise a clear error if `mazes/` is empty (instruct user to run `python -m maze.pool_generator` first).

- `wait_for_space(window) -> None`
  - Pygame event loop: render the static maze (agent at start, goal visible — reuse `PygameRenderer.draw` with the env's initial `_get_obs()`), overlay text "Press SPACE to start" (via `pygame.font`), block until `pygame.KEYDOWN` with `key == pygame.K_SPACE` or `pygame.QUIT`.

- `run_live(agent_cls, env, label: str, episode_cap: int = 1000, target_success: float = 0.9, render_every: int = 200) -> tuple[TrainingMetrics, int, int]`
  - Constructs `agent = agent_cls(env)` (fresh — no pretrained weights).
  - Custom episode loop (NOT `agent.train()`, so render throttling and early-stop are controllable). **Mirror the canonical loops in `q_learning.py`/`sarsa.py` exactly** — in particular, pass `done=terminated` (NOT `terminated or truncated`) to `update()`, per the Phase 2 reviewer fix. A code comment must call this out so future edits to the canonical loops are mirrored here too.
    - For `ep in range(episode_cap)`:
      - `env.render_mode = "human" if (ep < 10 or ep % render_every == 0) else None` — render throttling. Default `render_every=200`.
      - Even when `render_mode is None`, call `pygame.event.pump()` once at the end of the episode (cheap) so the OS doesn't mark the window "Not Responding" during long non-rendered stretches.
      - Run one episode (algorithm-specific loop). Append to a local `TrainingMetrics`.
      - If `metrics.rolling_success_rate[-1] >= target_success` (after at least 100 episodes): record `episodes_to_converge = ep + 1` and `break`.
    - If never reached `target_success`, `episodes_to_converge = None`.
    - Final greedy rollout: `env.render_mode = "human"`, run one episode with `get_action(state, explore=False)`, record `final_path_length` (= step count).
    - Return `(metrics, episodes_to_converge, final_path_length)`.

- `main()`
  - `path = pick_maze_file()` (CLI arg via `argparse` optional `--maze <path>`; default picks a **10x10** maze from the pool, consistent with the existing demos).
  - For each `(label, agent_cls)` in `[("Q-Learning", QLearningAgent), ("SARSA", SARSAAgent), ("Dyna-Q", DynaQAgent)]`:
    - `env = load_maze_env(path, render_mode="human")` — fresh env per algorithm (identical unseen maze, no state leakage).
    - On the first iteration only: `env.render()` (creates `env.window`), then `wait_for_space(env.window)`.
    - Print `f"\n=== {label} ==="`.
    - `metrics, eps_to_converge, path_len = run_live(agent_cls, env, label)`.
    - Print `f"  episodes_to_converge={eps_to_converge}, final_path_length={path_len}"`.
    - **`env.close()` before the next iteration's `load_maze_env(...)`** — avoids leaking the pygame window/display surface (plan-reviewer ACCEPTED finding).
  - Print final comparison table (label, episodes_to_converge, final_path_length) — plain `print()` formatting, no extra deps.

**Feasibility note (plan-reviewer, addressed):** On a 10x10 maze, `max_steps = size*size*4 = 400` steps/episode worst case → at 10 FPS a fully-rendered episode is ≤40s. With `render_every=200`, `episode_cap=1000`, and the first 10 episodes always rendered: at most ~15 rendered episodes per algorithm (~10 min worst case per algorithm). Convergence (90% rolling success) typically happens well before the 1000-episode cap, especially for Dyna-Q, so actual wall-clock is usually a few minutes. Larger mazes (15x15) remain usable via `--maze <path>` but will take proportionally longer.

- `if __name__ == "__main__": main()`

---

### `test_function/phase6_unseen_maze.py` (new)
Following the style of `test_function/phase3_dqn.py` (UTF-8 stdout reconfigure, `sys.path` insert, `PASS:` prints).

**Tests:**

1. `test_pool_generation()` — call `generate_pool(sizes=[5], n_per_size=1, out_dir="mazes/_test_pool")`, assert file exists, JSON has `size`, `seed`, `start`, `goal`, `grid` keys, `grid` shape matches `(2*5+1, 2*5+1)`. Clean up `_test_pool` dir after.
2. `test_preset_grid_env()` — `MazeEnv(size=5, grid=some_grid)`, `reset()` twice without seed, assert `_base_grid` identical both times (preset grid persists, Phase 1 fix not broken).
3. `test_load_maze_env()` — generate one pool file, `load_maze_env(path)`, assert `env.size` matches, `env.reset()` works, `check_env`-style sanity (obs shape correct).
4. `test_dyna_q_model_population()` — `DynaQAgent(env, planning_steps=5)`, run a handful of episodes (e.g. 50 on a 5x5 maze), assert `len(agent.model) > 0` and `agent.q_table.shape == (25, 4)`.
5. `test_dyna_q_sample_efficiency()` — on a small maze (5x5, capped at e.g. 200 episodes each, no rendering — `render_mode=None`), compare `QLearningAgent` vs `DynaQAgent(planning_steps=10)`: assert Dyna-Q reaches `rolling_success_rate >= 0.9` in fewer episodes than Q-Learning (or Dyna-Q converges within the cap while asserting it's <= Q-Learning's convergence episode). Keep episode cap small enough to run in a few seconds.

`if __name__ == "__main__":` runs all 5 tests in order, prints `"Phase 6: ALL TESTS PASS"`.

---

## Implementation Order

1. `maze/pool_generator.py` — standalone, no dependencies on other new code.
2. `env/maze_env.py` — add `grid` param + `_preset_grid` flag, fix `reset()` branch.
3. `env/maze_loader.py` — `load_maze_env()`.
4. `algorithms/q_learning.py` — add `ALGO_NAME` class attr (1-line change), update `train()` to use `self.ALGO_NAME`.
5. `algorithms/sarsa.py` — add `ALGO_NAME = "sarsa"` (optional consistency).
6. `algorithms/dyna_q.py` — `DynaQAgent`.
7. `algorithms/__init__.py` — register `DynaQAgent`.
8. `test_function/phase6_unseen_maze.py` — write and run; fix any issues in steps 1-7 before moving on.
9. `demo_unseen_maze.py` — live demo script (depends on everything above).
10. Run `python -m maze.pool_generator` to populate `mazes/`, then `python demo_unseen_maze.py` for a manual smoke test (visual — cannot be automated).

---

## Acceptance Check

- [ ] `python -m maze.pool_generator` creates `mazes/index.json` and `(len(sizes) * n_per_size)` JSON files.
- [ ] `python test_function/phase6_unseen_maze.py` → `Phase 6: ALL TESTS PASS`.
- [ ] `MazeEnv(size=5, grid=preset_grid)` — two consecutive `reset()` calls (no seed) produce the same `_base_grid` (preset persists).
- [ ] `DynaQAgent(env, planning_steps=10).train(...)` returns `TrainingMetrics` with `algo_name == "dyna_q"`.
- [ ] On a 5x5 unseen maze (no rendering, capped episodes), Dyna-Q(n=10) reaches 90% rolling success rate in fewer episodes than plain Q-Learning.
- [ ] `python demo_unseen_maze.py` (manual/visual): opens pygame window with static maze + "Press SPACE to start" text; SPACE triggers live training/animation for Q-Learning, SARSA, Dyna-Q in sequence; prints a final comparison table.
