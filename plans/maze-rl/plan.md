# Plan: Maze Solving with Reinforcement Learning
**Spec:** plans/maze-rl/spec.md
**Status:** Complete

## Phases
- [x] Phase 1: Maze Generation + Environment
- [x] Phase 2: Tabular RL (Q-Learning + SARSA)
- [x] ~~Phase 3: Deep RL (DQN via SB3)~~ — removed (2026-06-12), see "DQN Removal" note
- [x] Phase 4: Visualization + CLI
- [x] Phase 5: Jupyter Notebook — Analysis + Report
- [x] Phase 6: Unseen Maze Live Demo (Q-Learning, SARSA, Dyna-Q)

**DQN Removal (2026-06-12):** Phase 3 (DQN via Stable-Baselines3) was implemented, debugged, and working (100% success, 40 steps on 10x10 seed=42), but later removed from the project entirely per user decision — project now compares **Q-Learning, SARSA, and Dyna-Q** only. Removed: `algorithms/dqn_trainer.py`, `--algo dqn` CLI option, `dqn:` config section, `stable-baselines3`/`torch`/`tensorboard` deps, `tests/phase3_dqn.py`, DQN notebook section, and saved `models/dqn_*` artifacts.

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

The observation is the flattened (2N+1)x(2N+1) grid with agent and goal positions embedded directly in the array. Tabular agents extract the agent's position from it.

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

---

## Phase 6 — Key Design Decisions

### Maze Pool Format (`maze_pool/`)
- `maze_pool/maze_{size}x{size}_seed{seed}.json`: `{"size", "seed", "start": [0,0], "goal": [N-1,N-1], "grid": <(2N+1)x(2N+1) int list>}`.
- `maze_pool/index.json`: lightweight list of `{"file", "size", "seed"}` for fast picking without loading every grid.
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
- Picks a random maze from `maze_pool/index.json` (or `--maze <path>`), opens pygame, shows static maze (agent@start, goal visible), waits for SPACE.
- Before training (first algorithm only), `maze_is_solvable()` runs a BFS reachability check from start to goal on `_base_grid`. If unreachable, the demo prints "KHONG THE GIAI DUOC" and exits immediately — avoids wasting `episode_cap` episodes on an unsolvable maze.
- Runs Q-Learning → SARSA → Dyna-Q sequentially, each fresh (no pretrained weights). Training runs **headless** (`render_mode=None`, console progress every 50 episodes) — at 10 FPS, rendering 100s of episodes took minutes and the agent appeared "frozen" between throttled frames.
- During headless training, `_show_processing_message()` draws the static maze + "Dang xu ly maze, vui long cho..." once on the existing pygame window so it doesn't look frozen/blank. The text disappears automatically once the final greedy episode starts re-rendering frames.
- Early-stops each algorithm once rolling success rate ≥ 90% **and** a verification greedy rollout actually reaches the goal (`_greedy_reaches_goal`) — cap 1000 episodes. If `episode_cap` is reached without converging, prints "Khong hoi tu sau N episodes..." and still animates the current (best-effort) policy.
- Then animates one greedy rollout (rendered, 10 FPS) to show the final path.
- Prints comparison table: episodes-to-converge + final path length per algorithm.

**Reviewer fixes (Phase 6):**
- `env.close()` before re-creating `MazeEnv` between algorithms in `demo_unseen_maze.py` (avoid leaking pygame windows).
- `pygame.event.pump()` once per episode during headless training (avoid OS "Not Responding" during long non-rendered stretches).
- Default live demo to a **10x10 maze**, `episode_cap=1000`.
- Demo's manual episode loop must mirror `q_learning.py`/`sarsa.py` exactly, especially `done=terminated` (not `terminated or truncated`).

**Post-release fix (2026-06-12):** 90% rolling success rate during epsilon-greedy training (epsilon still ~0.5 around episode 100-150) does NOT guarantee the **greedy** policy solves the maze — it could oscillate in a 2-cycle near start and time out (`final_path_length == max_steps`). Fixed by adding `_greedy_reaches_goal()`: a no-render greedy rollout checked before declaring convergence; if it fails, training continues. Also switched training from render-throttled (10 FPS, render_every=200) to fully headless with console progress prints — the throttled approach made the demo take minutes and look stuck on a frozen frame.

**Post-release fix 2 (2026-06-12):** Two follow-up UX issues from user testing:
1. Goal completely walled off (unreachable) → headless training would burn through `episode_cap` for nothing while the window sat frozen. Fixed with `maze_is_solvable()` (BFS reachability pre-check) — aborts the demo with a clear message before training starts.
2. If the maze is reachable but doesn't converge within `episode_cap`, the window also looked frozen with no feedback. Fixed by printing "Khong hoi tu sau N episodes..." and still animating the best-effort greedy policy.
3. Even on the normal/solvable path, the window appeared blank/frozen during the entire headless training phase (could be many seconds). Fixed with `_show_processing_message()` — draws the static maze plus a "Dang xu ly maze, vui long cho..." overlay once before headless training starts; the overlay disappears automatically when the final greedy animation begins re-rendering frames.

See `phase-06-unseen-maze-demo.md` for full file-by-file implementation spec.

---

## Reviewer Fixes Applied (plan-reviewer)

4 ACCEPTED findings corrected before implementation:
1. **Phase 1 — reset() maze regeneration**: Only regenerate maze when `seed` changes or on first call. Subsequent resets restore agent to start. (Without fix: tabular agents learn across different MDPs, never converge.)
2. **Phase 2 — done flag**: Pass `done = terminated` (not `terminated or truncated`) to Q-update. Truncation is not terminal — bootstrapping should continue. (Without fix: value estimates biased near timeout boundary, success rate below 80%/75%.)
3. ~~**Phase 3 — observation dtype**: Changed `np.int32` → `np.float32`. SB3 MlpPolicy requires float observations.~~ (Phase 3/DQN later removed entirely — see "DQN Removal" note above; float32 retained as it's harmless for tabular agents.)
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
None — Phase 6 complete. Optional follow-ups: regenerate `maze_pool/` pool with more sizes/seeds if needed, or add `demo_unseen_maze.py` results to the brainstorm report.

---

## Previous Session Notes (Phase 3 DQN debugging, 2026-06-05 — historical, DQN later removed)

DQN was implemented (SB3, 8-dim `_PosObsWrapper` obs with wall flags + goal direction) and after 6 debugging iterations achieved 100% success at the optimal 40-step path on the 10x10 seed=42 maze, at ~1000x the compute cost of tabular methods (1M timesteps vs 3000 episodes). DQN was subsequently removed from the project on 2026-06-12 — see "DQN Removal" note at the top of this file.

---

## Risks

1. **Tabular scalability ceiling**: Q-table is `N*N x 4` numpy array. At N=10 this is 400 floats — trivial. But the observation space passed to tabular agents is the full `(2N+1)^2`-length vector; state extraction logic must be correct or the agent will see phantom states and never converge. Any off-by-one in the `gr // 2` conversion corrupts every Q-update.

2. **Pygame import on headless systems**: If `import pygame` is called unconditionally at module level in `maze_env.py`, importing the environment in training scripts on servers without a display will crash. All pygame imports and `pygame.init()` calls must be deferred inside the `render_mode == "human"` branch (or lazy-imported inside `render()`).
