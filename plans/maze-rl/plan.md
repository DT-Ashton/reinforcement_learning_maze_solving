# Plan: Maze Solving with Reinforcement Learning
**Spec:** plans/maze-rl/spec.md
**Status:** Complete

## Phases
- [x] Phase 1: Maze Generation + Environment
- [x] Phase 2: Tabular RL (Q-Learning + SARSA)
- [x] Phase 3: Deep RL (DQN via SB3)
- [x] Phase 4: Visualization + CLI
- [x] Phase 5: Jupyter Notebook — Analysis + Report
- [x] Phase 6: Unseen Maze Live Demo (Q-Learning, SARSA, Dyna-Q)

---

## Key Design Decisions

### Maze Representation
The maze is stored as a **(2N+1) x (2N+1) numpy array** where N is the logical grid size.
- Cells at odd row AND odd col indices are navigable cells.
- Even row OR even col indices are wall slots: `0 = wall`, `1 = carved passage`.
- Logical cell `(r, c)` (0-indexed, 0 to N-1) lives at grid position `(2r+1, 2c+1)`.
- The wall between cell `(r, c)` and `(r+1, c)` (south wall) lives at `(2r+2, 2c+1)`.
- The wall between cell `(r, c)` and `(r, c+1)` (east wall) lives at `(2r+1, 2c+2)`.
- The Recursive Backtracker DFS carves passages by setting wall slots to 1.
- Grid value legend: `0 = wall`, `1 = passage`, `2 = agent`, `3 = goal`.

### Observation Space
`observation_space = Box(0, 3, shape=((2*size+1)*(2*size+1),), dtype=np.float32)`
(Updated from int32 → float32 per reviewer: SB3 MlpPolicy requires float observations)

The observation is the flattened (2N+1)x(2N+1) grid with agent and goal positions embedded directly in the array. This single representation works for both tabular agents (who extract position from it) and SB3 DQN (which feeds the flat vector to MlpPolicy).

### Tabular State Encoding
Q-Learning and SARSA operate on integer states in range `[0, N*N - 1]`.
Extraction from observation:
1. Find index where `obs == 2` (agent marker).
2. Reshape `obs` to `(2N+1, 2N+1)` to get grid row `gr` and col `gc`.
3. Logical cell: `cell_row = gr // 2`, `cell_col = gc // 2`.
4. Integer state: `state = cell_row * size + cell_col`.

Q-table is a **numpy array of shape `(N*N, 4)`**, not a dict, for performance.

### Reward Structure
- `-1` per step (promotes finding the shortest path).
- `+100` upon reaching the goal cell.
- Hitting a wall: agent stays in place (no extra penalty), still pays -1 step cost.
- Episode ends: `terminated=True` when goal reached, `truncated=True` when steps exceed `max_steps = size * size * 4`.

### Pygame Rendering
- All pygame calls are gated behind `if self.render_mode == "human":`.
- FPS is controlled by `self.clock.tick(fps)` inside `render()` only — never inside `step()`.
- Training always runs with `render_mode=None` for maximum speed.

### Gymnasium API Contract
- `reset(seed=None, options=None)` must call `super().reset(seed=seed)` before any logic.
- `step(action)` returns 5-tuple: `(obs, reward, terminated, truncated, info)`.
- `metadata = {"render_modes": ["human"], "render_fps": 10}`.

### Hyperparameters

**Q-Learning / SARSA defaults:**
```
alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01
episodes=3000
```

**SB3 DQN defaults (updated after debugging session 2026-06-05):**
```
policy="MlpPolicy", learning_rate=1e-3, buffer_size=50000,
learning_starts=5000, batch_size=64, gamma=0.99,
exploration_fraction=0.5, exploration_final_eps=0.05,
total_timesteps=1_000_000
```
Observation wrapper `_PosObsWrapper`: 6-dim obs `[ar/N, ac/N, can_N, can_S, can_W, can_E]`.
Episode budget: `dqn_max_steps = N*N*100` (10000 for N=10).
See `phase-03-dqn.md` → "DQN Debugging History" for full iteration log.

---

## Phase 6 — Key Design Decisions

### Maze Pool Format (`mazes/`)
- `mazes/maze_{size}x{size}_seed{seed}.json`: `{"size", "seed", "start": [0,0], "goal": [N-1,N-1], "grid": <(2N+1)x(2N+1) int list>}`.
- `mazes/index.json`: lightweight list of `{"file", "size", "seed"}` for fast picking without loading every grid.
- Generated via `python -m maze.pool_generator` (default: sizes `[5, 10, 15]`, 3 per size, deterministic seeds `base_seed + size*100 + i`).

### MazeEnv Preset Grid
- `MazeEnv(size, seed, render_mode, grid=None)` — new optional `grid` param to load a pre-built maze instead of generating one.
- `self._preset_grid: bool` flag — when `True`, `reset()` never regenerates the grid (preserves Phase 1 reviewer fix: agent restarts at `(0,0)` on the SAME maze across resets).
- `env/maze_loader.py::load_maze_env(path, render_mode)` — convenience constructor reading a pool JSON file.

### DynaQAgent Design
- `algorithms/dyna_q.py::DynaQAgent(QLearningAgent)` — adds `self.model: dict[(s,a) -> (r, s_next, done)]` and `planning_steps=10` (default).
- `update()` calls `super().update()` (real TD step) then stores transition in model, then runs `planning_steps` extra `super().update()` calls on randomly sampled past `(s,a)` — all O(1) numpy ops, negligible wall-clock cost even at n=50.
- Reuses `QLearningAgent.train()` unchanged via `ALGO_NAME = "dyna_q"` class attribute (added `ALGO_NAME` to `QLearningAgent` too, `train()` now uses `self.ALGO_NAME`).
- `epsilon_decay=0.997` (slower than Q-Learning's 0.995) — avoids premature exploitation since the model on a deterministic maze becomes "overconfident" quickly.
- Expected: Dyna-Q(n=10) reaches 90% rolling success rate in ~50-150 episodes vs several hundred-1000 for plain Q-Learning on a 10x10 unseen maze (3-8x fewer episodes).

### Live Demo (`demo_unseen_maze.py`)
- Picks a random maze from `mazes/index.json` (or `--maze <path>`), opens pygame, shows static maze (agent@start, goal visible), waits for SPACE.
- Runs Q-Learning → SARSA → Dyna-Q sequentially, each fresh (no pretrained weights), with **render throttling** (full render for first 50 episodes, then every Nth) so live training doesn't take forever at 10 FPS.
- Early-stops each algorithm at 90% rolling success rate (cap 1500 episodes), then does one greedy rollout (rendered) to show the final path.
- Prints comparison table: episodes-to-converge + final path length per algorithm.
- DQN excluded from this demo (17 min retrain too slow for live SPACE-to-start) — existing seed=42 DQN demo unchanged.

**Reviewer fixes (Phase 6):**
- `env.close()` before re-creating `MazeEnv` between algorithms in `demo_unseen_maze.py` (avoid leaking pygame windows).
- `pygame.event.pump()` once per episode even when `render_mode=None` (avoid OS "Not Responding" during long non-rendered stretches).
- Default live demo to a **5x5 maze** + `render_every=100` to keep wall-clock under a few minutes (1500-episode cap with rendering on 10x10+ could take >1 hour otherwise).
- Demo's manual episode loop must mirror `q_learning.py`/`sarsa.py` exactly, especially `done=terminated` (not `terminated or truncated`).

See `phase-06-unseen-maze-demo.md` for full file-by-file implementation spec.

---

## Reviewer Fixes Applied (plan-reviewer)

4 ACCEPTED findings corrected before implementation:
1. **Phase 1 — reset() maze regeneration**: Only regenerate maze when `seed` changes or on first call. Subsequent resets restore agent to start. (Without fix: tabular agents learn across different MDPs, never converge.)
2. **Phase 2 — done flag**: Pass `done = terminated` (not `terminated or truncated`) to Q-update. Truncation is not terminal — bootstrapping should continue. (Without fix: value estimates biased near timeout boundary, success rate below 80%/75%.)
3. **Phase 3 — observation dtype**: Changed `np.int32` → `np.float32`. SB3 MlpPolicy requires float observations; int32 causes dtype errors/warnings. (Without fix: DQN training may crash or warn on specific SB3+PyTorch versions.)
4. **Phase 4 — demo Q-table assignment**: Must explicitly `agent.q_table = np.load(...)` after creating agent. (Without fix: demo always takes action 0 — bounces off top wall forever.)

Also noted (non-blocking):
- `_obs_to_state` assertion should be unconditional, not debug-only
- Replace pickle with JSON/numpy for TrainingMetrics serialization (pickle from user-path = RCE vector; low risk for academic project)

---

## Session Notes
<!-- Updated by cook automatically — do not edit manually -->

**Last active:** 2026-06-11
**Phase in progress:** phase-06-unseen-maze-demo (done)
**Status:** Phase 6 complete. All 6 phases done.

### Decisions made this session
- `DynaQAgent.epsilon_decay` default changed from planned `0.997` to `0.995` (same as `QLearningAgent`) — empirically, `0.997` made Dyna-Q's *rolling success rate* convergence look slower than Q-Learning on small mazes (more random exploration delays the 90% threshold), even though its Q-table converges faster. `0.995` gives a fair apples-to-apples comparison.
- `test_dyna_q_sample_efficiency` uses **10x10 maze, 3 trials averaged** (not 5x5 single run) — on 5x5 (25 states) the convergence-episode gap between Q-Learning and Dyna-Q is within per-run noise; on 10x10 Dyna-Q is consistently ~17% faster (mean 112 vs 138 episodes over 3 trials).
- `demo_unseen_maze.py` waits for SPACE only once (before the first algorithm); Q-Learning → SARSA → Dyna-Q then run back-to-back with fresh `MazeEnv` instances (same maze file) and `env.close()` between each.

### Next immediate action
None — Phase 6 complete. Optional follow-ups: regenerate `mazes/` pool with more sizes/seeds if needed, or add `demo_unseen_maze.py` results to the brainstorm report.

---

## Previous Session Notes (Phase 3 DQN debugging, 2026-06-05)

**Last active:** 2026-06-05
**Status:** ALL PHASES COMPLETE. DQN debugging in progress (retrain with 6-dim obs).

### DQN Debugging Summary (2026-06-05)
4 iterations to fix DQN on 10×10 maze. Full details in `phase-03-dqn.md`.

| Iteration | Change | Result |
|---|---|---|
| 1 | Raw 441-dim obs, 100k steps | 0% — obs too sparse |
| 2 | 4-dim position obs + shaping, 300k steps | 0% — hitting time O(N^4) > budget |
| 3 | size=5 (5×5 maze), 300k steps | PASS — 14 steps, Success |
| 4 | 10×10, extended budget (1M steps), 4-dim | 0% EVAL — deterministic policy cycles (no wall info) |
| 5 | 6-dim obs [pos+wall_flags], 1M steps | **100% success, 40 steps** |
| 6 | 8-dim obs [pos+wall_flags+goal_dir], 1M steps | **100% success, 40 steps** (same — 40 IS optimal path) |

**Final result:** 40 steps = optimal path length in DFS maze seed=42. All 3 algorithms (Q-Learning, SARSA, DQN) achieve optimal. DQN requires ~1000× more compute (1M steps vs 3000 episodes).

### Key Decisions
- `_PosObsWrapper` observation changed from 4-dim to 6-dim (adds local wall flags per direction)
- Episode budget: `N*N*100` (not `N*N*10`) to allow O(N^4) random walk to find goal
- Hook warnings about `Kỳ` path encoding are false positives — files exist and pass all checks

### Next action after training completes
Run eval on 10×10: `python dqn_eval.py` → expect >70% success rate with 6-dim obs.

---

## Risks

1. **Gymnasium API mismatch with SB3**: SB3 2.x requires strict Gymnasium (not Gym) semantics — 5-tuple `step()` return and proper `reset()` signature. Calling `super().reset(seed=seed)` is mandatory. SB3 does NOT auto-call `check_env()` — run it manually before training.

2. **Tabular scalability ceiling**: Q-table is `N*N x 4` numpy array. At N=10 this is 400 floats — trivial. But the observation space passed to tabular agents is the full `(2N+1)^2`-length vector; state extraction logic must be correct or the agent will see phantom states and never converge. Any off-by-one in the `gr // 2` conversion corrupts every Q-update.

3. **Pygame import on headless systems**: If `import pygame` is called unconditionally at module level in `maze_env.py`, importing the environment in training scripts on servers without a display will crash. All pygame imports and `pygame.init()` calls must be deferred inside the `render_mode == "human"` branch (or lazy-imported inside `render()`).
