# Phase 5: Jupyter Notebook — Analysis + Report
**Covers spec:** All phases (analysis output)
**User stories:** [P2] Research documentation via Jupyter for course report

---

## Files to Create

### `notebooks/maze_rl_analysis.ipynb`
**Purpose:** Interactive analysis notebook for the course report — runs training, generates all plots, and documents algorithm comparisons.

**Notebook Structure (cells in order):**

#### Section 0 — Setup
```python
# Cell 0: imports + path setup
import sys, os
sys.path.insert(0, os.path.abspath(".."))  # project root

import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import env  # triggers gymnasium.register

from maze.generator import MazeGenerator
from env.maze_env import MazeEnv
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dqn_trainer import DQNTrainer
from metrics.training_metrics import TrainingMetrics
from visualization.matplotlib_plots import (
    reward_curve, success_rate_curve, q_value_heatmap, comparison_plot
)
```

#### Section 1 — Maze Generation Demo
- Markdown cell: explain Recursive Backtracker algorithm
- Code: generate maze, visualize with matplotlib `imshow`
  ```python
  gen = MazeGenerator(size=10, seed=42)
  grid = gen.generate()
  plt.figure(figsize=(6, 6))
  plt.imshow(grid, cmap="gray_r", interpolation="nearest")
  plt.title("Generated Maze (10×10, seed=42)")
  plt.axis("off")
  plt.show()
  ```
- Code: show reproducibility (same seed → same maze)
- Code: show different sizes and seeds side by side (1×3 subplot)

#### Section 2 — Environment Sanity Check
- Markdown: explain observation space, action space, reward structure
- Code: `check_env(MazeEnv(...))` — confirm pass
- Code: run 10 random-action episodes, print mean reward and success rate
  ```python
  env_inst = MazeEnv(size=10, seed=42)
  for _ in range(10):
      obs, _ = env_inst.reset()
      for _ in range(400):
          action = env_inst.action_space.sample()
          obs, r, terminated, truncated, _ = env_inst.step(action)
          if terminated or truncated:
              break
  ```

#### Section 3 — Q-Learning Training
- Markdown: Q-Learning algorithm explanation, update rule formula (LaTeX)
- Code: train Q-Learning agent for 3000 episodes
- Code: plot reward curve and success rate curve
- Code: print final success rate, mean steps to goal (last 100 episodes)

#### Section 4 — SARSA Training
- Markdown: SARSA vs Q-Learning — on-policy vs off-policy explanation
- Code: train SARSA agent for 3000 episodes
- Code: overlay reward curve with Q-Learning (same axes) — save the `ax` object from Section 3 and reuse it:
  ```python
  # In Section 3 (save the returned ax):
  fig3, ax_reward = plt.subplots()
  reward_curve(ql_metrics, ax=ax_reward, label="Q-Learning")

  # In Section 4 (overlay on same ax):
  reward_curve(sarsa_metrics, ax=ax_reward, label="SARSA")
  ax_reward.set_title("Reward Curves — Q-Learning vs SARSA")
  ax_reward.legend()
  plt.show()
  ```
  (**Fix:** without passing `ax=ax_reward`, Section 4 creates a new figure and Q-Learning curve disappears from the comparison.)
- Code: print comparison table

#### Section 5 — DQN Training
- Markdown: Deep Q-Network, neural network architecture, experience replay explanation
- Code: check if `./models/dqn_maze.zip` exists — load if yes, train if no:
  ```python
  import os, pickle
  trainer = DQNTrainer(MazeEnv(size=10, seed=42))
  if os.path.exists("../models/dqn_maze.zip"):
      trainer.load("../models/dqn_maze")
      print("Loaded pre-trained DQN model.")
      # Fix: load saved metrics so Section 6 comparison_plot works
      if os.path.exists("../models/dqn_metrics.pkl"):
          with open("../models/dqn_metrics.pkl", "rb") as f:
              dqn_metrics = pickle.load(f)
      else:
          dqn_metrics = None  # model exists but metrics lost — skip DQN in comparison
  else:
      trainer.build_model()
      dqn_metrics = trainer.train()  # ~5-10 min on CPU — run once, reuse after
  ```
  (**Fix:** `dqn_metrics` was only set in the `else` branch — Section 6 `comparison_plot([ql_metrics, sarsa_metrics, dqn_metrics])` raised `NameError` when loading a pre-trained model.)
- Note: DQN training ~5-10 min CPU. Run once; notebook re-uses saved model on subsequent runs.

#### Section 6 — Comparison Analysis
- Markdown: comparison methodology
- Code: `comparison_plot([ql_metrics, sarsa_metrics, dqn_metrics])` — 4-panel figure
- Code: comparison table (pandas DataFrame):
  ```python
  import pandas as pd
  data = {
      "Algorithm": ["Q-Learning", "SARSA", "DQN"],
      "Final Success Rate": [...],
      "Mean Steps (last 100)": [...],
      "Training Time (s)": [...],
  }
  pd.DataFrame(data).set_index("Algorithm")
  ```

#### Section 7 — Q-Value Policy Visualization
- Markdown: what Q-values tell us about the learned policy
- Code: 2×2 grid of Q-value heatmaps (UP/DOWN/LEFT/RIGHT) for Q-Learning agent
- Code: extract and visualize the greedy policy as arrows on the maze grid

#### Section 8 — Ablation: Maze Sizes
- Markdown: how performance scales with maze size
- Code: train Q-Learning on sizes 5×5, 10×10, 15×15, plot success rate convergence speed

#### Section 9 — Conclusions
- Markdown cell only: answer these questions:
  1. Which algorithm converges fastest?
  2. Which achieves the highest final success rate?
  3. What are the limitations of tabular RL for larger mazes?
  4. When would DQN be preferred over Q-Learning?

---

## Implementation Order

1. Create `notebooks/` directory.
2. Write the notebook (`maze_rl_analysis.ipynb`) using Jupyter or VS Code notebook interface.
3. Run all cells top-to-bottom — verify no import errors or runtime failures.
4. Save trained model artifacts (`./models/`) and generated figures (`./reports/`) from inside the notebook.
5. Run acceptance checks below.

**Implementation note:** Run the notebook AFTER phases 1–4 are complete and `check_env()` passes. The notebook calls the same code as `main.py` but in an interactive, documented way.

---

## Acceptance Check

- [ ] All cells execute top-to-bottom without errors (`Kernel → Restart & Run All`).
- [ ] Section 1 displays a maze image (non-white, non-black grid visible).
- [ ] Section 2 `check_env()` prints nothing (pass).
- [ ] Section 6 comparison table shows ≥ 3 rows (Q-Learning, SARSA, DQN).
- [ ] Section 7 Q-value heatmaps are non-uniform (agent has learned something — not all-zero values).
- [ ] `./reports/comparison.png` exists after running Section 6.
