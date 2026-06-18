# Spec: Maze Solving with Reinforcement Learning

**Date:** 2026-06-04
**Status:** Ready

---

## Problem Statement

Đồ án môn AI Kỳ 8: build hệ thống giải mê cung bằng RL, so sánh hiệu quả Q-Learning, SARSA và DQN trên cùng một môi trường chuẩn.

---

## User Stories

- **[P1]** As a student, I want a random maze generator with configurable size and seed so that I can reproduce experiments.
  Accepted when: `MazeGenerator(size=10, seed=42)` tạo ra cùng maze khi gọi lại với seed giống nhau.

- **[P1]** As a student, I want a Gymnasium-compatible MazeEnv so that all three RL algorithms share the same interface.
  Accepted when: `env = gym.make('MazeEnv-v0')`, `obs, info = env.reset()`, `obs, reward, terminated, truncated, info = env.step(action)` chạy không lỗi.

- **[P1]** As a student, I want a Q-Learning agent that trains on MazeEnv so that I can see tabular RL in action.
  Accepted when: Agent đạt success rate ≥ 80% trên maze 10×10 sau 3000 episodes.

- **[P1]** As a student, I want a SARSA agent with the same interface as Q-Learning so that comparison is fair.
  Accepted when: Agent đạt success rate ≥ 75% trên maze 10×10 sau 3000 episodes.

- **[P1]** As a student, I want a DQN agent via Stable-Baselines3 so that I have a deep RL baseline for comparison.
  Accepted when: SB3 DQN train được trên MazeEnv không lỗi, log reward lên TensorBoard.

- **[P2]** As a student, I want Pygame visualization showing the agent navigating the maze so that I can demo the project.
  Accepted when: `python main.py --demo --algo qlearning` mở cửa sổ pygame hiển thị agent di chuyển.

- **[P2]** As a student, I want Matplotlib plots of reward curves and heatmaps so that I can include them in the report.
  Accepted when: `python main.py --plot` tạo ra reward curve, success rate, và Q-value heatmap.

- **[P2]** As a student, I want to see Q-Learning, SARSA, and Dyna-Q learn live on a maze never used during development, so I can visually compare convergence speed.
  Accepted when: `python demo_unseen_maze.py` mở pygame hiển thị 1 maze từ `maze_pool/`, chờ SPACE, sau đó train trực tiếp (không dùng model pretrained) cả 3 thuật toán trên maze đó và in ra bảng so sánh episodes-to-converge + final path length.

- **[P3]** PPO agent — out of scope, có thể thêm sau khi 3 phase chính xong.

---

## Functional Requirements

### Phase 1 — Maze + Environment

1. **FR-01**: `MazeGenerator(size: int, seed: int)` tạo perfect maze (1 đường đi duy nhất) bằng Recursive Backtracker DFS.
2. **FR-02**: Maze có thể render ra numpy array `(N, N)` với `0=wall, 1=path, 2=agent, 3=goal`.
3. **FR-03**: `MazeEnv(size, seed, render_mode)` kế thừa `gymnasium.Env` với:
   - `observation_space`: `Box(0, 3, shape=(N*N,), dtype=np.int32)` — flattened grid
   - `action_space`: `Discrete(4)` — UP/DOWN/LEFT/RIGHT
   - `reset(seed)` → trả về `(obs, info)`
   - `step(action)` → trả về `(obs, reward, terminated, truncated, info)`
4. **FR-04**: Reward structure: `-1` mỗi bước, `+100` khi đến goal, tường = blocked (không penalty thêm).
5. **FR-05**: Episode kết thúc khi agent đến goal hoặc vượt `max_steps = size * size * 4`.
6. **FR-06**: Pygame rendering khi `render_mode='human'`: vẽ maze, agent (màu xanh), goal (màu đỏ), tốc độ điều chỉnh được.

### Phase 2 — Q-Learning & SARSA

7. **FR-07**: `QLearningAgent(env, alpha, gamma, epsilon, epsilon_decay, epsilon_min)` với Q-table là `dict` hoặc numpy array.
8. **FR-08**: `SARSAAgent` cùng interface với `QLearningAgent`, chỉ khác update rule (on-policy).
9. **FR-09**: Cả hai agent có method `train(n_episodes)` → trả về `TrainingMetrics`.
10. **FR-10**: `TrainingMetrics` lưu: rewards theo episode, success rate (rolling 100 ep), số bước đến goal.
11. **FR-11**: Sau training, Q-table có thể export heatmap (matplotlib) cho mỗi action.

### Phase 3 — DQN

12. **FR-12**: `DQNTrainer(env, total_timesteps, tensorboard_log)` wrap SB3 DQN.
13. **FR-13**: Log training metrics lên TensorBoard tại `./logs/dqn/`.
14. **FR-14**: Model được save tại `./models/dqn_maze.zip` sau training.

### Phase 6 — Unseen Maze Live Demo

16. **FR-16**: `maze/pool_generator.py::generate_pool(sizes, n_per_size, base_seed, out_dir)` tạo nhiều maze ngẫu nhiên kích thước khác nhau, lưu mỗi maze thành 1 file JSON (`size`, `seed`, `start`, `goal`, `grid`) trong `maze_pool/`, kèm `maze_pool/index.json` liệt kê tất cả file.
17. **FR-17**: `MazeEnv(size, seed, render_mode, grid=None)` nhận thêm `grid` để load maze có sẵn thay vì tự generate; `env/maze_loader.py::load_maze_env(path, render_mode)` đọc 1 file pool và trả về `MazeEnv` tương ứng. `reset()` không regenerate khi `_preset_grid=True`.
18. **FR-18**: `algorithms/dyna_q.py::DynaQAgent(QLearningAgent)` — thêm model `dict[(s,a) -> (r, s_next, done)]` và `planning_steps` (default 10); mỗi `update()` thật chạy thêm `planning_steps` update mô phỏng từ model. `train()` kế thừa từ `QLearningAgent`, báo cáo `algo_name="dyna_q"`.
19. **FR-19**: `demo_unseen_maze.py` — pick maze từ `maze_pool/`, hiện pygame tĩnh (agent@start, goal hiển thị) + text "Press SPACE to start", chờ SPACE rồi train live (không pretrained) Q-Learning → SARSA → Dyna-Q tuần tự trên cùng maze, có render throttling. DQN không tham gia demo này.
20. **FR-20**: Mỗi thuật toán dừng sớm khi rolling success rate ≥ 90% (cap 1500 episodes), sau đó chạy 1 episode greedy (rendered) để show đường đi cuối. In bảng so sánh: episodes-to-converge + final path length cho 3 thuật toán.

### Entry Point

15. **FR-15**: `main.py` hỗ trợ CLI flags:
    - `--train --algo [qlearning|sarsa|dqn]` — train algorithm
    - `--demo --algo [qlearning|sarsa|dqn]` — chạy pygame demo với model đã train
    - `--plot` — vẽ comparison plots giữa các algorithms
    - `--size N` — kích thước maze (default: 10)
    - `--seed N` — random seed (default: 42)

---

## Non-Functional Requirements

- **Performance**: Q-Learning/SARSA train 3000 episodes trên maze 10×10 trong < 30 giây (không render).
- **Reproducibility**: Cùng seed → cùng maze → cùng training trajectory (với `numpy.random.seed`).
- **Compatibility**: Python 3.10+, Gymnasium ≥ 0.29, Stable-Baselines3 ≥ 2.0, PyTorch ≥ 2.0.

---

## Success Criteria

- [ ] `MazeGenerator(10, 42).generate()` trả về numpy array không có isolated cells
- [ ] `gym.make('MazeEnv-v0')` pass `gymnasium.utils.env_checker.check_env()`
- [ ] Q-Learning đạt success rate ≥ 80% sau 3000 episodes trên maze 10×10
- [ ] SARSA đạt success rate ≥ 75% sau 3000 episodes trên maze 10×10
- [ ] DQN train được 100k steps không crash, reward curve hiển thị trong TensorBoard
- [ ] Pygame demo chạy được ở tốc độ ≥ 10 FPS
- [ ] `--plot` tạo ra ít nhất 3 plots: reward curve, success rate, Q-value heatmap
- [ ] `python -m maze.pool_generator` tạo `maze_pool/index.json` + các file maze JSON
- [ ] Trên 1 maze unseen (5x5/10x10), Dyna-Q(n=10) đạt rolling success rate ≥ 90% với số episodes ít hơn Q-Learning thuần
- [ ] `python demo_unseen_maze.py` chạy được: hiện maze tĩnh → SPACE → train live 3 thuật toán → in bảng so sánh

---

## Project Structure

```
reinforcement_learning_maze_solving/
├── maze/
│   ├── __init__.py
│   └── generator.py          # MazeGenerator — Recursive Backtracker
├── env/
│   ├── __init__.py
│   └── maze_env.py           # MazeEnv — Gymnasium wrapper
├── algorithms/
│   ├── __init__.py
│   ├── q_learning.py         # QLearningAgent
│   ├── sarsa.py              # SARSAAgent
│   └── dqn_trainer.py        # DQNTrainer (SB3 wrapper)
├── visualization/
│   ├── __init__.py
│   ├── pygame_renderer.py    # Real-time pygame rendering
│   └── matplotlib_plots.py   # Metrics plots và heatmaps
├── metrics/
│   ├── __init__.py
│   └── training_metrics.py   # TrainingMetrics dataclass
├── configs/
│   └── default.yaml          # Hyperparameters mặc định
├── models/                   # Saved model weights
├── logs/                     # TensorBoard logs
├── plans/                    # Brainstorm + spec
├── requirements.txt
└── main.py                   # CLI entry point
```

---

## Out of Scope

- PPO agent (noted as P3, future work)
- Multi-agent scenarios
- Curriculum learning (tăng độ khó tự động)
- Web UI

---

## Assumptions

- Maze size mặc định 10×10, tối đa 30×30 (tabular methods vẫn feasible ở 900 states)
- Observation là flattened grid — phù hợp cả tabular (extract position) lẫn DQN (Box input)
- SB3 DQN dùng MlpPolicy (không phải CNN) vì observation là flat vector
- Không có wall penalty — agent bị block tại chỗ, chỉ mất -1 step penalty
