# Phase 2: Tabular RL (Q-Learning + SARSA)
**Covers spec:** FR-07, FR-08, FR-09, FR-10, FR-11
**User stories:** [P1] Q-Learning agent >= 80% success rate after 3000 episodes on 10x10 maze; [P1] SARSA agent >= 75% success rate after 3000 episodes on 10x10 maze

---

## Files to Create

### `metrics/training_metrics.py`
**Purpose:** Typed container for per-episode training statistics returned by every agent's `train()` method.

**Classes/Functions:**

- `TrainingMetrics` (Python `dataclass`)
  ```python
  from dataclasses import dataclass, field

  @dataclass
  class TrainingMetrics:
      # Fix: list fields MUST use field(default_factory=list) — mutable defaults not allowed in dataclasses
      episode_rewards: list = field(default_factory=list)
      episode_lengths: list = field(default_factory=list)
      success_flags: list = field(default_factory=list)
      rolling_success_rate: list = field(default_factory=list)
      algo_name: str = ""
      maze_size: int = 10
      n_episodes: int = 0
      q_table: object = None  # np.ndarray for tabular agents; None for DQN
  ```
  - Without `field(default_factory=list)`, `TrainingMetrics()` raises `ValueError: mutable default is not allowed`. The acceptance check `TrainingMetrics()` would fail immediately.
  - `rolling_success_rate[i]` = mean of `success_flags[max(0, i-99) : i+1]`.
  - Computed and appended inside the training loop, NOT computed in batch at the end.
  - Implementation note: `episode_rewards`, `episode_lengths`, `success_flags`, `rolling_success_rate` must be plain Python lists (not numpy arrays) so they are easily serializable.

- `TrainingMetrics.final_success_rate() -> float`
  - Returns `rolling_success_rate[-1]` if non-empty, else `0.0`.

- `TrainingMetrics.mean_reward_last_100() -> float`
  - Returns `float(np.mean(self.episode_rewards[-100:]))` if `self.episode_rewards` is non-empty, else `0.0`. (**Fix:** `np.mean([])` returns `nan` — guards against calling before training completes, e.g. in Phase 5 comparison table.)

---

### `algorithms/base_agent.py`
**Purpose:** Abstract base class defining the interface all tabular agents must implement, ensuring `QLearningAgent` and `SARSAAgent` are interchangeable.

**Classes/Functions:**

- `BaseAgent(env, alpha: float, gamma: float, epsilon: float, epsilon_decay: float, epsilon_min: float)` (abstract, `ABC`)
  - `self.env`: the MazeEnv instance (stored, not copied).
  - `self.alpha`, `self.gamma`, `self.epsilon`, `self.epsilon_decay`, `self.epsilon_min`: hyperparameters.
  - `self.size`: `env.size`.
  - `self.n_states`: `self.size * self.size`.
  - `self.n_actions`: `4`.
  - `self.q_table`: `np.zeros((self.n_states, 4), dtype=np.float64)`.
  - Implementation note: Q-table is a numpy array indexed by integer state, NOT a defaultdict. This gives O(1) lookup and enables vectorized heatmap export.
  - Implementation note: `base_agent.py` must include `from abc import ABC, abstractmethod` at the top. Without `@abstractmethod` on `update()` and `train()`, Python allows `BaseAgent()` instantiation without error — the ABC enforcement only works when the class inherits from `ABC` AND the abstract methods are decorated with `@abstractmethod`.

- `_obs_to_state(obs: np.ndarray) -> int` (concrete helper on BaseAgent, shared by both subclasses)
  - Reshapes `obs` to `(2*self.size+1, 2*self.size+1)`.
  - **Fix:** Cast to int32 trước khi so sánh — `obs` là float32, nếu có floating point imprecision thì `obs == 2` có thể miss:
    ```python
    obs_int = obs.astype(np.int32)
    assert 2 in obs_int, "agent marker missing from observation"
    flat_idx = np.argmax(obs_int == 2)
    gr = flat_idx // (2 * self.size + 1)
    gc = flat_idx % (2 * self.size + 1)
    return (gr // 2) * self.size + (gc // 2)
    ```
  - Assert unconditional (không phải debug-only) — fail rõ ràng hơn là silently return state 0.

- `get_action(state: int, explore: bool = True) -> int` (concrete on BaseAgent)
  - If `explore=True` and `np.random.random() < self.epsilon`: return `np.random.randint(4)` (random action).
  - Else: return `int(np.argmax(self.q_table[state]))` (greedy action).
  - Implementation note: use `np.random.random()` (global numpy RNG) consistently; do not mix with `random.random()`.

- `_decay_epsilon(self) -> None` (concrete on BaseAgent)
  - `self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)`.
  - Called once per episode at the end of the episode loop.

- `update(s: int, a: int, r: float, s_next: int, done: bool) -> None` (abstract — `@abstractmethod`)
  - Subclasses implement their specific Bellman update here.

- `train(n_episodes: int) -> TrainingMetrics` (abstract — `@abstractmethod`)
  - Subclasses implement the episode loop here.

- `get_q_table() -> np.ndarray`
  - Returns `self.q_table` (shape `(N*N, 4)`).
  - Used by visualization to render the Q-value heatmap.

---

### `algorithms/q_learning.py`
**Purpose:** Off-policy tabular Q-Learning agent using the max-next-Q Bellman update.

**Classes/Functions:**

- `QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)` (default 3000 episodes)
  - Inherits `BaseAgent`. No additional attributes needed.

- `update(s: int, a: int, r: float, s_next: int, done: bool) -> None`
  - Q-Learning (off-policy) update rule:
    ```
    td_target = r + gamma * max(Q[s_next]) * (1 - done)
    td_error  = td_target - Q[s, a]
    Q[s, a]  += alpha * td_error
    ```
  - In code: `self.q_table[s, a] += self.alpha * (r + self.gamma * np.max(self.q_table[s_next]) * (1 - int(done)) - self.q_table[s, a])`.
  - When `done=True`, the `(1 - done)` term zeros out the future reward component — no bootstrapping beyond the terminal state.

- `train(n_episodes: int = 3000) -> TrainingMetrics`
  - Creates `metrics = TrainingMetrics(algo_name="qlearning", maze_size=self.size, n_episodes=n_episodes)`.
  - Episode loop (repeat `n_episodes` times):
    1. `obs, _ = self.env.reset()` — always reset without seed to reuse the same maze structure.
    2. `state = self._obs_to_state(obs)`.
    3. `total_reward = 0`, `steps = 0`, `terminated = False`, `truncated = False`.
    4. Inner loop until `terminated or truncated`:
       a. `action = self.get_action(state, explore=True)`.
       b. `obs_next, reward, terminated, truncated, _ = self.env.step(action)`.
       c. `s_next = self._obs_to_state(obs_next)`.
       d. `self.update(state, action, reward, s_next, terminated)`.
         - **Fix (reviewer — CRITICAL):** Pass `done = terminated` ONLY (not `terminated or truncated`).
         - When `truncated=True` (timeout), the next state is a VALID state — zeroing the bootstrap biases value estimates near the timeout boundary. Only zero when the episode truly TERMINATES at the goal.
         - Episode exit is handled by the loop condition (`if terminated or truncated: break`), separate from the update.
       e. `state = s_next`, `total_reward += reward`, `steps += 1`.
       f. `if terminated or truncated: break`.
    5. After inner loop: `self._decay_epsilon()`.
    6. Append to metrics: `episode_rewards`, `episode_lengths`, `success_flags` (= `terminated`).
    7. Compute and append `rolling_success_rate`: `mean(success_flags[max(0, ep-99):ep+1])`.
  - Return `metrics`.

---

### `algorithms/sarsa.py`
**Purpose:** On-policy tabular SARSA agent; differs from Q-Learning only in the update step (uses actual next action, not greedy).

**Classes/Functions:**

- `SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)`
  - Inherits `BaseAgent`. No additional attributes needed.

- `update(s: int, a: int, r: float, s_next: int, done: bool, a_next: int = 0) -> None`
  - SARSA (on-policy) update rule:
    ```
    td_target = r + gamma * Q[s_next, a_next] * (1 - done)
    td_error  = td_target - Q[s, a]
    Q[s, a]  += alpha * td_error
    ```
  - In code: `self.q_table[s, a] += self.alpha * (r + self.gamma * self.q_table[s_next, a_next] * (1 - int(done)) - self.q_table[s, a])`.
  - The `a_next` parameter is the key difference from Q-Learning — it uses the action actually chosen (epsilon-greedy) rather than the greedy max.
  - Implementation note: override the `update` signature to add `a_next`; the abstract base has `a_next` as optional to keep the interface compatible.

- `train(n_episodes: int = 3000) -> TrainingMetrics`
  - Creates `metrics = TrainingMetrics(algo_name="sarsa", maze_size=self.size, n_episodes=n_episodes)`.
  - Episode loop — SARSA requires choosing `a_next` before the update, so the inner loop structure differs:
    1. `obs, _ = self.env.reset()`.
    2. `state = self._obs_to_state(obs)`.
    3. `action = self.get_action(state, explore=True)` — choose first action BEFORE the loop.
    4. `total_reward = 0`, `steps = 0`, `terminated = False`, `truncated = False`.
    5. Inner loop until `terminated or truncated`:
       a. `obs_next, reward, terminated, truncated, _ = self.env.step(action)`.
       b. `s_next = self._obs_to_state(obs_next)`.
       c. `a_next = self.get_action(s_next, explore=True)` — choose next action while still exploring.
       d. `self.update(state, action, reward, s_next, terminated, a_next)`.
         - **Fix (reviewer — CRITICAL):** Same as Q-Learning — pass `done = terminated` only.
       e. `state = s_next`, `action = a_next`, `total_reward += reward`, `steps += 1`.
    6. After inner loop: `self._decay_epsilon()`.
    7. Append to metrics lists, compute rolling success rate.
  - Return `metrics`.
  - Implementation note: in SARSA, `a_next` is sampled with the current epsilon — if `terminated=True`, the inner loop exits before `a_next` is ever used in a step, so passing any value for `a_next` when `done=True` is safe because `(1 - done)` zeroes the future term anyway.

---

### `algorithms/__init__.py`
**Purpose:** Public API for the algorithms package.

```python
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
# DQNTrainer added in Phase 3 — do NOT import here yet (file doesn't exist)

__all__ = ["QLearningAgent", "SARSAAgent"]
```
**Phase 3 completion:** update `__init__.py` thêm:
```python
from algorithms.dqn_trainer import DQNTrainer
__all__ = ["QLearningAgent", "SARSAAgent", "DQNTrainer"]
```
Nếu import DQNTrainer trước khi `dqn_trainer.py` tồn tại → `ModuleNotFoundError` ngay khi `import algorithms`.

---

## Implementation Order

1. Create `metrics/training_metrics.py` — needed by agents before they're written.
2. Create `metrics/__init__.py` exporting `TrainingMetrics`.
3. Implement `algorithms/base_agent.py` — abstract base with `_obs_to_state`, `get_action`, `_decay_epsilon`.
4. Implement `algorithms/q_learning.py` — `update()` then `train()`.
5. Implement `algorithms/sarsa.py` — `update()` (add `a_next`) then `train()` (pre-loop action selection).
6. Write `algorithms/__init__.py` (DQNTrainer import can be commented out until Phase 3).
7. Run acceptance checks below.

---

## Acceptance Check

- [ ] `python -c "from metrics.training_metrics import TrainingMetrics; m = TrainingMetrics(); print(m.final_success_rate())"` prints `0.0`.
- [ ] Q-Learning smoke test:
  ```python
  import env
  import gymnasium
  from algorithms.q_learning import QLearningAgent
  e = gymnasium.make("MazeEnv-v0")
  agent = QLearningAgent(e.unwrapped)
  metrics = agent.train(3000)
  print(f"Final success rate: {metrics.final_success_rate():.2%}")
  # Expected: >= 0.80
  ```
- [ ] SARSA smoke test:
  ```python
  import env
  import gymnasium
  from algorithms.sarsa import SARSAAgent
  e = gymnasium.make("MazeEnv-v0")
  agent = SARSAAgent(e.unwrapped)
  metrics = agent.train(3000)
  print(f"Final success rate: {metrics.final_success_rate():.2%}")
  # Expected: >= 0.75
  ```
- [ ] Q-table shape check: `assert agent.get_q_table().shape == (100, 4)` (for N=10).
- [ ] Both agents complete 3000 episodes on N=10 in under 30 seconds total (no rendering).
- [ ] `_obs_to_state` round-trip: reset env, call `_obs_to_state(obs)` → should return `0` (agent starts at cell (0,0), state = 0*10+0 = 0).
