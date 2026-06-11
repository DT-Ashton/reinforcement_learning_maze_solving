# Phase 4: Visualization + CLI
**Covers spec:** FR-11, FR-15 (plus supporting FR-10 for metrics export)
**User stories:** [P2] Matplotlib plots for report; [P2] Pygame demo via main.py; [P1] main.py CLI entry point tying all phases together

---

## Files to Create

### `visualization/matplotlib_plots.py`
**Purpose:** Stateless functions that take `TrainingMetrics` objects and produce/save publication-ready Matplotlib figures.

**Module-level imports:**
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
```
(**Fix:** `matplotlib.ticker` must be imported explicitly — `import matplotlib.pyplot as plt` does NOT expose `matplotlib.ticker`. Without it, `success_rate_curve()` raises `AttributeError` when calling `matplotlib.ticker.PercentFormatter`.)

**Classes/Functions:**

- `reward_curve(metrics: TrainingMetrics, ax=None, label: str = None, color: str = None) -> matplotlib.axes.Axes`
  - Plots `metrics.episode_rewards` (y-axis) against episode index (x-axis).
  - Adds a 100-episode rolling average line on top: `pd.Series(metrics.episode_rewards).rolling(100).mean()` — or compute manually with `np.convolve` to avoid pandas dependency.
  - Manual rolling mean: `np.convolve(rewards, np.ones(100)/100, mode='valid')` — note the output is shorter by 99 elements; offset x-axis by 99.
  - If `ax` is None, creates a new `fig, ax = plt.subplots()`.
  - Labels: x = "Episode", y = "Total Reward", title = f"Reward Curve — {metrics.algo_name}". Note: when called from `comparison_plot()` with an existing `ax`, the caller sets the subplot title after the loop — this function's `ax.set_title()` call will be overridden. This is intentional.
  - Returns `ax` so the caller can overlay multiple algorithms.

- `success_rate_curve(metrics: TrainingMetrics, ax=None, label: str = None, color: str = None) -> matplotlib.axes.Axes`
  - Plots `metrics.rolling_success_rate` (y-axis: 0.0 to 1.0) against episode index.
  - Adds a horizontal dashed line at the target threshold (`0.80` for Q-Learning, `0.75` for SARSA) — or just at `0.80` universally.
  - y-axis formatted as percentage: `ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))`.
  - Returns `ax`.

- `q_value_heatmap(q_table: np.ndarray, size: int, action: int, ax=None, title: str = None) -> matplotlib.axes.Axes`
  - Visualizes the Q-values for a single action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) as a 2D heatmap.
  - Extract: `values = q_table[:, action].reshape(size, size)`.
  - Plot: `im = ax.imshow(values, cmap="RdYlGn", origin="upper")`.
  - Add colorbar: `plt.colorbar(im, ax=ax)`.
  - Title: `title or f"Q-values: {['UP','DOWN','LEFT','RIGHT'][action]}"`.
  - Returns `ax`.

- `comparison_plot(metrics_list: list[TrainingMetrics], save_path: str = None) -> None`
  - **Fix:** Filter None at function start — Phase 5 notebook passes `dqn_metrics=None` when model pre-loaded but metrics pickle missing:
    ```python
    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        print("No metrics to plot."); return
    ```
  - Creates `fig, axs = plt.subplots(2, 2, figsize=(14, 10))`.
  - Top-left (`axs[0,0]`): loop over metrics, call `reward_curve(m, ax=axs[0,0], label=m.algo_name)` for each. After loop: `axs[0,0].set_title("Reward Curves"); axs[0,0].legend()`. (**Fix:** `label=m.algo_name` required — without it legend shows nothing. `set_title()` must be called here, NOT inside `reward_curve()` — `reward_curve` sets `f"Reward Curve — {m.algo_name}"` which overwrites on every call leaving only the last algo name.)
  - Top-right (`axs[0,1]`): same pattern with `success_rate_curve(m, ax=axs[0,1], label=m.algo_name)`. After loop: `axs[0,1].set_title("Success Rates"); axs[0,1].legend()`.
  - Bottom-left (`axs[1,0]`): Q-value heatmap for action=0 (UP). Extract first tabular agent:
    ```python
    tabular_metrics = next((m for m in metrics_list if m.q_table is not None), None)
    ```
    Guard: `if tabular_metrics is not None: q_value_heatmap(tabular_metrics.q_table, tabular_metrics.maze_size, action=0, ax=axs[1,0])` else `axs[1,0].text(0.5, 0.5, "Q-table not available", ha="center", va="center", transform=axs[1,0].transAxes)`.
  - Bottom-right (`axs[1,1]`): same guard, `action=1` (DOWN).
  - `plt.tight_layout()`.
  - If `save_path` is not None: `plt.savefig(save_path, dpi=150, bbox_inches="tight")`.
  - Always calls `plt.show()` at the end.

- `save_figure(fig, path: str) -> None`
  - Utility: `fig.savefig(path, dpi=150, bbox_inches="tight")`. Called by individual plot functions when saving standalone files.

---

### `visualization/__init__.py`
**Purpose:** Public API for the visualization package.

```python
from visualization.pygame_renderer import PygameRenderer
from visualization.matplotlib_plots import (
    reward_curve, success_rate_curve, q_value_heatmap, comparison_plot
)

__all__ = [
    "PygameRenderer",
    "reward_curve", "success_rate_curve", "q_value_heatmap", "comparison_plot",
]
```

---

### `metrics/__init__.py`
**Purpose:** Public API for the metrics package.

```python
from metrics.training_metrics import TrainingMetrics

__all__ = ["TrainingMetrics"]
```

---

### `configs/default.yaml`
**Purpose:** Single source of truth for all hyperparameters; loaded by `main.py` so changing a value here propagates everywhere without code edits.

```yaml
maze:
  size: 10
  seed: 42

qlearning:
  alpha: 0.1
  gamma: 0.99
  epsilon: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
  n_episodes: 3000

sarsa:
  alpha: 0.1
  gamma: 0.99
  epsilon: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
  n_episodes: 3000

dqn:
  total_timesteps: 100000
  learning_rate: 0.001
  buffer_size: 10000
  learning_starts: 1000
  batch_size: 64
  gamma: 0.99
  exploration_fraction: 0.3
  exploration_final_eps: 0.05
  tensorboard_log: "./logs/dqn/"
  model_save_path: "./models/dqn_maze"

visualization:
  cell_px: 30
  fps: 10
  plot_save_dir: "./reports/"
```

---

### `main.py`
**Purpose:** Single argparse CLI entry point that orchestrates training, demo playback, and plot generation for all three algorithms.

**Full argument specification (FR-15):**

```
python main.py [MODE] [OPTIONS]

Modes (at least one required):
  --train           Train the selected algorithm
  --demo            Run Pygame demo with a trained model (requires prior --train)
  --plot            Generate and save comparison plots

Shared options:
  --algo {qlearning,sarsa,dqn}   Algorithm to use (default: qlearning)
  --size N                        Maze size N×N (default: 10)
  --seed N                        Random seed (default: 42)
  --config PATH                   Path to YAML config file (default: configs/default.yaml)
```

**Module-level imports:**
```python
import os
import argparse
import pickle
import numpy as np
import yaml
from env.maze_env import MazeEnv
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dqn_trainer import DQNTrainer
from visualization.matplotlib_plots import comparison_plot
```
(**Fix:** explicit import list prevents `NameError` for `np`, `pickle`, `os` used throughout `run_train`, `run_demo`, `run_plot` but not previously specified.)

**Function breakdown:**

- `load_config(path: str) -> dict`
  - `import yaml; with open(path) as f: return yaml.safe_load(f)`. (**Fix:** close file handle — `open()` without `with` leaks the fd.)

- `build_env(size: int, seed: int, render_mode: str = None) -> MazeEnv`
  - `import env` (triggers gymnasium.register), then `return MazeEnv(size=size, seed=seed, render_mode=render_mode)`.
  - Implementation note: always call `import env` before any `gymnasium.make()` or direct `MazeEnv()` instantiation to ensure the package's `__init__.py` has run.

- `run_train(args, cfg: dict) -> None`
  - `os.makedirs("./models/", exist_ok=True)`. (**Fix:** prevents `FileNotFoundError` when `np.save()` or `pickle.dump()` target `./models/` before the directory exists on a fresh clone.)
  - `env = build_env(cfg["maze"]["size"], cfg["maze"]["seed"])`. (**Fix:** use `build_env()` — triggers `import env` for gymnasium registration and uses cfg values already overridden by CLI; never create `MazeEnv(...)` directly here.)
  - Dispatches on `args.algo`:
    - `"qlearning"`:
      ```python
      # Fix: filter out n_episodes before unpacking — constructor doesn't accept it
      agent_cfg = {k: v for k, v in cfg["qlearning"].items() if k != "n_episodes"}
      agent = QLearningAgent(env, **agent_cfg)
      metrics = agent.train(cfg["qlearning"]["n_episodes"])
      np.save("./models/qlearning_qtable.npy", agent.get_q_table())
      ```
    - `"sarsa"`: same pattern, `agent_cfg` filter from `cfg["sarsa"]`, save as `./models/sarsa_qtable.npy`.
    - `"dqn"`:
      ```python
      trainer = DQNTrainer(env, total_timesteps=cfg["dqn"]["total_timesteps"],
                           tensorboard_log=cfg["dqn"]["tensorboard_log"])
      trainer.build_model(**{k: cfg["dqn"][k] for k in [
          "learning_rate","buffer_size","learning_starts","batch_size",
          "gamma","exploration_fraction","exploration_final_eps"]})
      dqn_metrics = trainer.train()   # train() now returns TrainingMetrics via MetricsCallback
      with open("./models/dqn_metrics.pkl", "wb") as f:
          pickle.dump(dqn_metrics, f)
      ```
  - After training (tabular only): print `f"[{args.algo}] Final success rate: {metrics.final_success_rate():.2%}"`.
  - For tabular agents: store q_table before pickling — `metrics.q_table = agent.get_q_table()`. (**Fix — reviewer:** comparison_plot needs q_table; without this, `comparison_plot` raises AttributeError when rendering heatmap panels.)
  - Save `TrainingMetrics` as pickle:
    ```python
    with open(f"./models/{args.algo}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    ```

- `run_demo(args, cfg: dict) -> None`
  - `env = build_env(cfg["maze"]["size"], cfg["maze"]["seed"], render_mode="human")`. (**Fix:** same reason as `run_train()` — use `build_env()` for consistent cfg routing and registration.)
  - Override render FPS from config: `env.metadata["render_fps"] = cfg["visualization"]["fps"]`. (**Fix:** wires the `visualization.fps` config value that was previously unused. `cell_px` stays at `PygameRenderer.draw()`'s default of 30 — matches the config value, no extra wiring needed.)
  - Dispatches on `args.algo`:
    - `"qlearning"` / `"sarsa"`: 
      ```python
      q_table = np.load(f"./models/{args.algo}_qtable.npy")
      # Fix: use constructor defaults — hyperparameters don't matter since q_table is overwritten immediately
      agent = QLearningAgent(env)  # or SARSAAgent(env)
      agent.q_table = q_table           # Fix (reviewer — CRITICAL): must explicitly assign loaded Q-table
      agent.epsilon = 0.0               # pure greedy
      ```
      Full demo loop (**Fix:** previously incomplete — implementer had to guess `_obs_to_state` usage):
      ```python
      import pygame
      obs, _ = env.reset()
      state = agent._obs_to_state(obs)
      terminated, truncated = False, False
      while not (terminated or truncated):
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  env.close(); return
          action = agent.get_action(state, explore=False)
          obs, reward, terminated, truncated, _ = env.step(action)
          state = agent._obs_to_state(obs)
      print(f"Steps: {env._step_count}, {'Success' if terminated else 'Timeout'}")
      env.close()
      ```
      Without the explicit `agent.q_table = q_table` assignment, the agent has an all-zero Q-table and always takes action 0 (UP), bouncing off the top wall forever.
    - `"dqn"`:
      ```python
      # Fix: DQNTrainer.__init__ requires env — must create instance first, THEN load
      trainer = DQNTrainer(env)
      trainer.load(cfg["dqn"]["model_save_path"])
      ```
      Full demo loop (**Fix:** previously a comment placeholder):
      ```python
      import pygame
      obs, _ = env.reset()
      terminated, truncated = False, False
      while not (terminated or truncated):
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  env.close(); return
          action = trainer.predict(obs)
          obs, reward, terminated, truncated, _ = env.step(action)
      print(f"Steps: {env._step_count}, {'Success' if terminated else 'Timeout'}")
      env.close()
      ```

- `run_plot(args, cfg: dict) -> None`
  - `os.makedirs("./reports/", exist_ok=True)` — create output directory if missing.
  - Load all available metrics pickles (**Fix:** stale comment removed — DQN metrics ARE produced by `trainer.train()` via `_MetricsCallback`. **Fix:** guard added — `comparison_plot([])` crashes without it):
    ```python
    metrics_list = []
    for algo in ["qlearning", "sarsa", "dqn"]:
        path = f"./models/{algo}_metrics.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                metrics_list.append(pickle.load(f))
    if not metrics_list:
        print("No metrics files found. Run --train first.")
        return
    ```
  - Calls `comparison_plot(metrics_list, save_path="./reports/comparison.png")`.

- `main() -> None` (the `if __name__ == "__main__"` entry)
  - Build argparser:
    ```python
    parser = argparse.ArgumentParser(description="Reinforcement Learning Maze Solver")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--algo", choices=["qlearning","sarsa","dqn"], default="qlearning")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    ```
  - Guard: `if not (args.train or args.demo or args.plot): parser.error("specify at least one of --train, --demo, --plot")`.
  - Load config: `cfg = load_config(args.config)`.
  - Override config with CLI args: `cfg["maze"]["size"] = args.size; cfg["maze"]["seed"] = args.seed`.
  - Dispatch: `if args.train: run_train(args, cfg)`; `if args.demo: run_demo(args, cfg)`; `if args.plot: run_plot(args, cfg)`.
  - Implementation note: all three flags can be combined: `python main.py --train --demo --algo qlearning` will train then immediately demo. The dispatch order (train → demo → plot) is intentional.

---

## Implementation Order

1. Create `metrics/__init__.py`.
2. Implement `visualization/matplotlib_plots.py` (no env dependency — can be tested standalone).
3. Create `visualization/__init__.py` exporting `PygameRenderer` and the plot functions.
4. Write `configs/default.yaml`.
5. Implement `main.py` — start with `--train` flow (no render), then add `--demo`, then `--plot`.
6. Create `reports/` directory (used by `--plot` to save figures).
7. Run end-to-end acceptance checks below.

---

## Acceptance Check

- [ ] Train + plot in one command:
  ```bash
  python main.py --train --algo qlearning --size 10 --seed 42
  python main.py --train --algo sarsa --size 10 --seed 42
  python main.py --plot
  # ./reports/comparison.png should exist with 4 subplots
  ```

- [ ] Demo runs without crash:
  ```bash
  python main.py --demo --algo qlearning --size 10 --seed 42
  # Pygame window opens, agent moves, window closes after episode ends
  ```

- [ ] `--plot` without prior training gives a clean error, not a stack trace:
  ```bash
  python main.py --plot
  # Should print: "No metrics files found. Run --train first."
  ```

- [ ] `comparison_plot` produces a file at `./reports/comparison.png` that is a valid PNG (non-zero file size).

- [ ] Config override works:
  ```bash
  python main.py --train --algo qlearning --size 5 --seed 7
  # Trains on 5×5 maze, not 10×10
  ```

- [ ] `python main.py` with no flags prints help (the `parser.error()` guard triggers).

- [ ] All three `--algo` values work for `--train` without modification:
  ```bash
  python main.py --train --algo qlearning
  python main.py --train --algo sarsa
  python main.py --train --algo dqn
  ```
