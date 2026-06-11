# Maze Solving with Reinforcement Learning

So sánh Q-Learning, SARSA và DQN trên môi trường mê cung sinh ngẫu nhiên.  
Đồ án môn Trí tuệ nhân tạo — Kỳ 8.

---

## Cấu trúc dự án

```
reinforcement_learning_maze_solving/
├── main.py                  # CLI entry point
├── requirements.txt
├── configs/
│   └── default.yaml         # Hyperparameters
├── env/
│   ├── __init__.py          # Đăng ký MazeEnv-v0 với Gymnasium
│   └── maze_env.py          # Gymnasium environment
├── maze/
│   └── generator.py         # Recursive Backtracker DFS
├── algorithms/
│   ├── base_agent.py        # Abstract base class
│   ├── q_learning.py        # Q-Learning (off-policy)
│   ├── sarsa.py             # SARSA (on-policy)
│   └── dqn_trainer.py       # DQN via Stable-Baselines3
├── metrics/
│   └── training_metrics.py  # TrainingMetrics dataclass
├── visualization/
│   ├── pygame_renderer.py   # Pygame renderer
│   └── matplotlib_plots.py  # Reward/success/heatmap plots
├── models/                  # Saved models & metrics (auto-created)
├── reports/                 # Generated plots (auto-created)
├── test_function/
│   ├── phase1_maze_env.py   # Tests: maze generator + Gymnasium env
│   ├── phase2_tabular_rl.py # Tests: Q-Learning + SARSA
│   └── phase3_dqn.py        # Tests: DQN build/train/predict
└── notebooks/
    └── maze_rl_analysis.ipynb
```

---

## Yêu cầu

- Python 3.10+
- Anaconda (khuyến nghị) hoặc virtualenv

---

## Cài đặt

```powershell
# 1. Di chuyển vào thư mục dự án
cd "c:\Users\ADMIN\Documents\Kỳ 8 - AI\reinforcement_learning_maze_solving"

# 2. Cài dependencies
pip install -r requirements.txt --no-cache-dir
```

**Packages chính:**

| Package | Phiên bản | Dùng cho |
|---------|-----------|----------|
| `gymnasium` | >=0.29 | RL environment API |
| `numpy` | >=1.24 | Maze generation, Q-table |
| `pygame` | >=2.5 | Pygame rendering |
| `stable-baselines3` | >=2.0 | DQN implementation |
| `torch` | >=2.0 | Neural network backend |
| `matplotlib` | >=3.7 | Plots |
| `tensorboard` | >=2.13 | DQN training logs |

---

## Kiểm tra môi trường

```powershell
python -c "
import env, gymnasium
from gymnasium.utils.env_checker import check_env
check_env(gymnasium.make('MazeEnv-v0').unwrapped)
print('Moi truong hop le.')
"
```

Không có output = pass.

---

## Huấn luyện

### Q-Learning (~10 giây)

```powershell
python main.py --train --algo qlearning --size 10 --seed 42
```

### SARSA (~10 giây)

```powershell
python main.py --train --algo sarsa --size 10 --seed 42
```

### DQN (~17 phút trên CPU)

```powershell
python main.py --train --algo dqn --size 10 --seed 42
```

Sau khi train xong, models được lưu tự động:

```
models/
├── qlearning_qtable.npy
├── qlearning_metrics.pkl
├── sarsa_qtable.npy
├── sarsa_metrics.pkl
├── dqn_maze.zip
└── dqn_metrics.pkl
```

---

## Demo Pygame

> Yêu cầu màn hình (không chạy được headless).

```powershell
python main.py --demo --algo qlearning
python main.py --demo --algo sarsa
python main.py --demo --algo dqn
```

Cửa sổ Pygame mở ra, agent tự di chuyển theo policy đã học. Đóng cửa sổ để thoát.

---

## Sinh biểu đồ so sánh

```powershell
python main.py --plot
```

Tạo file `reports/comparison.png` gồm 4 panel:
- Reward curves (Q-Learning vs SARSA vs DQN)
- Success rate curves
- Q-value heatmap — action UP
- Q-value heatmap — action DOWN

---

## TensorBoard (theo dõi DQN)

```powershell
tensorboard --logdir C:\Users\ADMIN\maze_rl_logs\dqn
```

Mở trình duyệt: `http://localhost:6006`

> **Lưu ý:** TensorBoard logs được lưu tại `C:\Users\ADMIN\maze_rl_logs\dqn` (ngoài thư mục project) vì TensorFlow không xử lý được ký tự Unicode trong đường dẫn (`ỳ`).

---

## Jupyter Notebook (báo cáo)

```powershell
jupyter notebook notebooks/maze_rl_analysis.ipynb
```

Hoặc mở bằng **VS Code** → chọn kernel Python → **Run All**.

Notebook tự động:
- Load models đã train nếu có sẵn trong `models/`
- Train lại từ đầu nếu không tìm thấy
- Sinh tất cả biểu đồ và lưu vào `reports/`

---

## Chạy tất cả trong một lệnh

Train + Demo + Plot cùng lúc:

```powershell
python main.py --train --demo --plot --algo qlearning
```

---

## Thay đổi maze size / seed

Qua CLI:

```powershell
python main.py --train --algo qlearning --size 15 --seed 7
python main.py --demo  --algo qlearning --size 15 --seed 7
```

Qua config (áp dụng cho tất cả lệnh):

```yaml
# configs/default.yaml
maze:
  size: 15   # thay đổi ở đây
  seed: 7
```

---

## Hyperparameters

Tất cả hyperparameters được quản lý tại [`configs/default.yaml`](configs/default.yaml):

```yaml
qlearning:
  alpha: 0.1          # learning rate
  gamma: 0.99         # discount factor
  epsilon: 1.0        # initial exploration
  epsilon_decay: 0.995
  epsilon_min: 0.01
  n_episodes: 3000

dqn:
  total_timesteps: 1000000
  learning_rate: 0.001
  buffer_size: 50000
  learning_starts: 5000
  batch_size: 64
  gamma: 0.99
  exploration_fraction: 0.5
  exploration_final_eps: 0.05
```

---

## Kết quả (10×10 maze, seed=42)

| Thuật toán | Success Rate | Steps | Training time | Ghi chú |
|------------|-------------|-------|---------------|---------|
| Q-Learning | ≥ 80% | 40 (optimal) | ~10 giây | Off-policy, sample-efficient |
| SARSA | ≥ 75% | 40 (optimal) | ~10 giây | On-policy, conservative hơn |
| DQN | 100% (eval) | 40 (optimal) | ~17 phút | Deep RL, cần nhiều compute hơn |

> **Note:** Maze DFS seed=42 có đúng một đường đi dài 40 bước từ (0,0) đến (9,9).
> Cả 3 thuật toán đều tìm được đường đó. DQN kém hiệu quả hơn về sample efficiency nhưng đạt kết quả tương đương.

---

## Môi trường chi tiết

- **Observation:** flat `float32` vector, shape `(2N+1)²` — ví dụ 441 với N=10
- **Actions:** `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`
- **Reward:** `-1` mỗi bước, `+100` khi đến goal
- **Terminated:** agent đến goal cell `(N-1, N-1)`
- **Truncated:** vượt quá `N²×4` bước (tabular) hoặc `N²×100` bước (DQN)
- **Maze:** Recursive Backtracker DFS — perfect maze (đúng 1 đường đi giữa 2 ô bất kỳ)

---

## DQN Architecture

DQN dùng `_PosObsWrapper` để convert observation 441-dim → **8-dim** trước khi đưa vào MLP:

```
obs (441-dim) → _PosObsWrapper → [ar/N, ac/N, can_N, can_S, can_W, can_E, dr/N, dc/N]
```

| Chiều | Ý nghĩa |
|-------|---------|
| `ar/N, ac/N` | Vị trí agent (normalized) |
| `can_N, can_S, can_W, can_E` | Có thể đi theo hướng đó không (0/1) |
| `dr/N, dc/N` | Khoảng cách đến goal (normalized) |

Reward shaping: mỗi bước +`(1 - manhattan_dist/max_dist)` → gradient về phía goal ngay cả trước khi tìm được goal lần đầu.

---

## Phase 6 — Unseen Maze Live Demo (Q-Learning, SARSA, Dyna-Q)

### Sinh pool mê cung

```powershell
python -m maze.pool_generator
```

Tạo thư mục `mazes/` với các mê cung ngẫu nhiên (mặc định: kích thước 5×5, 10×10, 15×15, mỗi 3 cái):
- `mazes/maze_5x5_seed100.json`, `maze_5x5_seed101.json`, ...
- `mazes/index.json` — danh sách tất cả file để pick nhanh

Mỗi file JSON chứa: `{"size", "seed", "start": [0,0], "goal": [N-1,N-1], "grid": [...]}`

### Chạy demo live trên mê cung chưa thấy

```powershell
python demo_unseen_maze.py
```

Hoặc chỉ định mê cung cụ thể:

```powershell
python demo_unseen_maze.py --maze mazes/maze_10x10_seed100.json
python demo_unseen_maze.py --size 5
```

**Cách hoạt động:**
1. Cửa sổ Pygame mở ra, hiển thị mê cung tĩnh (agent ở start, goal hiển thị).
2. Chờ nhấn **SPACE** để bắt đầu training (lần đầu tiên).
3. Train **Q-Learning** trực tiếp trên mê cung này (không dùng model pretrained), render real-time, dừng khi đạt 90% rolling success rate (hoặc tối đa 1500 episodes).
4. Sau đó chạy 1 episode greedy để show đường đi tối ưu.
5. Lặp lại với **SARSA**, rồi **Dyna-Q**.
6. In bảng so sánh: số episodes để hội tụ + độ dài đường đi cuối cùng.

**Lưu ý:** DQN không tham gia demo này (huấn luyện lại trên CPU mất ~17 phút). Để xem DQN, dùng `main.py --demo --algo dqn` (dùng model seed=42 đã train).

### Dyna-Q — Model-based Q-Learning

`algorithms/dyna_q.py` cung cấp `DynaQAgent`:
- Kế thừa `QLearningAgent`, thêm internal model `dict[(s,a) -> (r, s_next, done)]`.
- Mỗi real transition, chạy thêm `planning_steps=10` (default) mô phỏng TD updates từ model — không cần real environment interaction.
- **Kết quả:** hội tụ ~17% nhanh hơn Q-Learning thuần (trung bình 112 vs 138 episodes trên 10×10 maze).

Hoàn toàn tương thích với tabular training loop — chỉ có thuật toán khác, interface giống:

```python
from algorithms import DynaQAgent
agent = DynaQAgent(env, planning_steps=10)
metrics = agent.train(n_episodes=1500)
```

### Test Phase 6

```powershell
python test_function/phase6_unseen_maze.py
```

5 test: pool generation, maze loading, DynaQ convergence, demo entry point.

---

## Test

```powershell
python test_function/phase1_maze_env.py   # Maze + Gymnasium env
python test_function/phase2_tabular_rl.py # Q-Learning + SARSA
python test_function/phase3_dqn.py        # DQN (dùng model tạm, không ghi đè dqn_maze.zip)
python test_function/phase6_unseen_maze.py # Pool generation, maze loading, Dyna-Q, demo
```
