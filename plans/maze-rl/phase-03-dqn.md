# Phase 3: Deep RL (DQN via Stable-Baselines3)
**Covers spec:** FR-12, FR-13, FR-14
**User stories:** [P1] DQN trains on MazeEnv without errors and logs reward to TensorBoard

---

## Files to Create

### `algorithms/dqn_trainer.py`
**Purpose:** Thin wrapper around SB3's DQN that wires up MazeEnv, configures hyperparameters, handles save/load, and sets TensorBoard log directory.

**Classes/Functions:**

- `DQNTrainer(env, total_timesteps: int = 100_000, tensorboard_log: str = "./logs/dqn/")`
  - `self.env`: the MazeEnv instance (unwrapped — not wrapped in a `gym.make` Monitor, since SB3 will add its own wrappers).
  - `self.total_timesteps`: training budget.
  - `self.tensorboard_log`: path string for TensorBoard event files.
  - `self.model_save_path`: `"./models/dqn_maze"` (SB3 appends `.zip` automatically).
  - `self.model`: initialized to `None`; set to the SB3 DQN object in `build_model()`.
  - Implementation note: do NOT create the SB3 DQN in `__init__`. Call `build_model()` explicitly before `train()` so the caller can inspect/modify hyperparameters first.

- `build_model(learning_rate: float = 1e-3, buffer_size: int = 10_000, learning_starts: int = 1_000, batch_size: int = 64, gamma: float = 0.99, exploration_fraction: float = 0.3, exploration_final_eps: float = 0.05) -> None`
  - Imports SB3: `from stable_baselines3 import DQN`.
  - Creates `self.model = DQN(policy="MlpPolicy", env=self.env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, gamma=gamma, exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, tensorboard_log=self.tensorboard_log, verbose=1)`.
  - The `env` passed to DQN must already be the raw `MazeEnv` instance with `render_mode=None`. SB3 wraps it internally with a `VecEnv`.
  - Implementation note: `policy="MlpPolicy"` is correct here — the observation is a flat integer vector, not an image. SB3's `MlpPolicy` with default architecture (two 64-unit hidden layers) is sufficient for N=10 (441-dim input).

- `_MetricsCallback` (inner class, inherits `BaseCallback` from `stable_baselines3.common.callbacks`)
  - **Fix:** Add `from stable_baselines3.common.callbacks import BaseCallback` at the TOP of `dqn_trainer.py` (file-level import, not inside a method). Without it, `class _MetricsCallback(BaseCallback):` raises `NameError` at module import time.
  - Captures per-episode stats during `learn()` so `DQNTrainer` can produce a `TrainingMetrics` compatible with tabular agents.
  - `__init__(self, max_steps: int)`: `super().__init__()`, `self.max_steps = max_steps`, `self.episode_rewards = []`, `self.episode_lengths = []`, `self.success_flags = []`. (**Fix:** `episode_lengths` must be collected — `TrainingMetrics.episode_lengths` will be empty otherwise, breaking Phase 5 comparison table's "Mean Steps" column.)
  - `_on_step(self) -> bool`:
    ```python
    for info in self.locals.get("infos", []):
        if "episode" in info:                         # SB3 sets this at episode end
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_lengths.append(info["episode"]["l"])
            # Fix: use episode length < max_steps, not r > 0.
            # r > 0 fails when path length > 99 (e.g. DQN takes 150 steps → r=-50 but still won).
            # Episode ended before timeout ↔ agent reached goal.
            self.success_flags.append(info["episode"]["l"] < self.max_steps)
    return True
    ```
  - Why: `ep_info_buffer` (deque maxlen=100) only retains the last 100 episodes — not enough for a comparison plot vs 3000-episode tabular curves. Callback captures every episode.

- `train() -> TrainingMetrics`
  - Guard: `if self.model is None: raise RuntimeError("Call build_model() before train()")`. (**Fix:** without this, forgetting `build_model()` raises confusing `AttributeError: 'NoneType' object has no attribute 'learn'`.)
  - Create callback: `cb = self._MetricsCallback(max_steps=self.env.max_steps)`.
  - Calls `self.model.learn(total_timesteps=self.total_timesteps, tb_log_name="DQN_MazeEnv", reset_num_timesteps=True, progress_bar=True, callback=cb)`.
  - After `learn()`: calls `self.save()`.
  - Build and return metrics:
    ```python
    from metrics.training_metrics import TrainingMetrics
    metrics = TrainingMetrics(algo_name="dqn", maze_size=self.env.size,
                              episode_rewards=cb.episode_rewards,
                              episode_lengths=cb.episode_lengths,
                              success_flags=cb.success_flags)
    # compute rolling_success_rate
    for i in range(len(metrics.success_flags)):
        window = metrics.success_flags[max(0, i-99): i+1]
        metrics.rolling_success_rate.append(sum(window) / len(window))
    metrics.n_episodes = len(metrics.success_flags)
    return metrics
    ```
  - Implementation note: `tb_log_name="DQN_MazeEnv"` auto-increments to `DQN_MazeEnv_1`, `DQN_MazeEnv_2` per run.

- `save(path: str = None) -> None`
  - If `path` is None, uses `self.model_save_path`.
  - Calls `self.model.save(path)`.
  - Prints confirmation: `f"Model saved to {path}.zip"`.
  - Implementation note: `models/` directory must exist before calling `save()`. Create it with `os.makedirs("models", exist_ok=True)` at the top of this method.

- `load(path: str = None) -> None`
  - If `path` is None, uses `self.model_save_path`.
  - `from stable_baselines3 import DQN`.
  - `self.model = DQN.load(path, env=self.env)`.
  - Implementation note: `DQN.load()` requires passing `env=` so the loaded model can step the environment. Without it, calling `predict()` works but `learn()` (fine-tuning) would fail.

- `predict(obs: np.ndarray) -> int`
  - Returns `int(self.model.predict(obs, deterministic=True)[0])`.
  - Used by the demo runner in `main.py` to get greedy actions from a trained DQN.
  - Implementation note: `model.predict()` returns a tuple `(action, state)` — index `[0]` for the action. With `deterministic=True`, it uses the greedy policy (no exploration).

---

## SB3 Compatibility — Why MazeEnv Already Works

SB3 2.x requires the environment to implement the Gymnasium (not Gym) API. MazeEnv from Phase 1 is already compatible because:

1. **5-tuple step return**: `(obs, reward, terminated, truncated, info)` — SB3 reads `terminated` and `truncated` separately.
2. **`reset()` returns `(obs, info)`** — SB3 calls `env.reset()` and unpacks both values.
3. **`observation_space` is a `Box` with `dtype=np.float32`** — Phase 1 was updated to use float32 (per reviewer fix). SB3's MlpPolicy expects float observations; int32 causes dtype warnings in stricter SB3 builds.
4. **`action_space` is `Discrete(4)`** — DQN requires discrete action space.

**Fix (reviewer):** SB3 2.x does NOT automatically call `check_env()` — it only wraps the env in a `VecEnv`. You must call `check_env()` manually before training to catch API issues. Add this as part of the acceptance check, not as an assumption about SB3 behavior.

Verify SB3 compatibility manually before training:
```python
from stable_baselines3.common.env_checker import check_env
from env.maze_env import MazeEnv
check_env(MazeEnv(size=10, seed=42, render_mode=None))
# Should print nothing (no warnings)
```

---

## TensorBoard Logging

- Log directory: `./logs/dqn/` (relative to project root).
- SB3 automatically logs: `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `train/loss`, `train/exploration_rate`.
- Start TensorBoard: `tensorboard --logdir ./logs/dqn/`.
- Each call to `train()` creates a new subdirectory `DQN_MazeEnv_1`, `DQN_MazeEnv_2`, etc. — run side by side to compare hyperparameter sweeps.
- Implementation note: create the log directory with `os.makedirs(self.tensorboard_log, exist_ok=True)` in `DQNTrainer.__init__` so the directory exists before SB3 tries to write to it. (**Fix:** use `self.tensorboard_log`, not hardcoded `"./logs/dqn/"` — if the user passes a custom `tensorboard_log` path, the hardcoded version creates the wrong directory and SB3 crashes on first write.)

---

## Implementation Order

1. Verify Phase 1 passes `check_env()` with zero warnings.
2. Create `models/` and `logs/dqn/` directories (or handle via `os.makedirs` in trainer).
3. Implement `algorithms/dqn_trainer.py`.
4. Update `algorithms/__init__.py` to uncomment the `DQNTrainer` import.
5. Run acceptance checks below.

---

## DQN Debugging History (Session 2026-06-05)

### Attempt 1 — Raw 441-dim obs, 100k steps → 0% success
- DQN trained on raw flat grid obs `(2N+1)^2 = 441` dimensions with default hyperparams.
- Result: `ep_rew_mean = -400`, `ep_len_mean = 400`. Never found goal.
- Root cause: 441-dim sparse vector (mostly 0/1 walls, two non-zero cells) is too high-dimensional for MLP-DQN to extract position signal from in 100k steps.

### Attempt 2 — Position obs [ar/N, ac/N, 1.0, 1.0], 300k steps → 0% success
- Added `_PosObsWrapper`: converts 441-dim obs to 4-dim `[ar/N, ac/N, 1.0, 1.0]`.
- Also added potential-based reward shaping: `bonus = 1 - manhattan_dist/max_dist ∈ [0, 1]`.
- Extended `max_steps = N*N*10 = 1000` (from 400).
- Result: `ep_rew_mean = -685`, still 0% success.
- Root cause: Expected random walk hitting time in 10×10 perfect maze (tree graph) is O(N^4) ≈ 10000 steps. With budget 1000 steps/episode, agent almost never finds goal during exploration → no learning signal.

### Attempt 3 — Size=5 maze (interim working solution)
- Switched to `--size 5` to validate architecture.
- 5×5 maze: expected hitting time ≈ 200-500 steps. With `max_steps = 250`, DQN finds goal in exploration.
- Result: **converged**. `ep_len_mean = 14.7`, `ep_rew_mean = 93.8`. Demo: `Steps: 14, Success`.
- Confirmed architecture (4-dim obs + reward shaping) works on small maze.

### Attempt 4 — Extended budget for 10×10, 1M steps → 0% EVAL success
- Changed `dqn_max_steps = N*N*100 = 10000`, `total_timesteps = 1_000_000`, `exploration_fraction = 0.5`.
- Training metrics looked good: `ep_len_mean = 78.8`, `ep_rew_mean = 49.5` (positive!).
- But eval showed **0/20 success** (all timed out at 10000 steps).
- Root cause: `predict()` uses `deterministic=True`. Policy with position-only obs `[ar/9, ac/9, 1.0, 1.0]` has **no wall information**. Deterministic greedy policy gets stuck in fixed cycles (same state → same action → same adjacent wall → same state). Training metrics were optimistic because `exploration_rate=0.05` (5% random noise) masked the cycle during training.

### Fix (Current) — 6-dim obs with local wall flags
- Changed `_PosObsWrapper._pos()` to return `[ar/N, ac/N, can_N, can_S, can_W, can_E]`.
- `can_X = 1.0` if the passage in direction X is open (not a wall), `0.0` if blocked/boundary.
- Changed `observation_space` to `Box(0.0, 1.0, (6,))`.
- With local wall flags, DQN knows which directions are navigable and can learn to avoid dead-ends without needing full maze map.
- Retraining with same budget (1M steps, `dqn_max_steps=10000`) — in progress.

### Key Lesson
Position-only observation causes deterministic policy cycles in maze environments. The minimum viable obs for DQN maze solving includes local navigability information (wall flags per direction). Reward shaping and extended episodes alone are insufficient if the observation doesn't encode wall structure.

---

## Acceptance Check

- [ ] SB3 check_env passes:
  ```python
  from stable_baselines3.common.env_checker import check_env
  from env.maze_env import MazeEnv
  check_env(MazeEnv(size=10, seed=42, render_mode=None))
  ```
  No output means pass.

- [ ] Short training run (1000 steps) completes without error:
  ```python
  import env
  from env.maze_env import MazeEnv
  from algorithms.dqn_trainer import DQNTrainer
  e = MazeEnv(size=10, seed=42, render_mode=None)
  trainer = DQNTrainer(e, total_timesteps=1000)
  trainer.build_model()
  metrics = trainer.train()
  # Should print progress bar, then "Model saved to ./models/dqn_maze.zip"
  assert isinstance(metrics.episode_rewards, list)
  print(f"Episodes captured: {metrics.n_episodes}")
  ```

- [ ] Model save/load round trip:
  ```python
  trainer.save("./models/dqn_test")
  trainer.load("./models/dqn_test")
  import numpy as np
  obs = np.zeros((2*10+1)**2, dtype=np.float32)  # Fix: float32 matches observation_space dtype
  action = trainer.predict(obs)
  assert 0 <= action <= 3
  ```

- [ ] `./logs/dqn/DQN_MazeEnv_1/` directory exists and contains at least one `events.out.tfevents.*` file after training.

- [ ] Full 100k step training run completes in under 10 minutes on CPU. (SB3 DQN on a 441-input MLP should manage ~10k steps/sec on modern hardware.)
