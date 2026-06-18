# Maze Solving with Reinforcement Learning

So sánh Q-Learning, SARSA và Dyna-Q trên môi trường mê cung sinh ngẫu nhiên.  
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
│   └── dyna_q.py            # Dyna-Q (model-based Q-Learning)
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
│   └── phase6_unseen_maze.py # Tests: pool generation, Dyna-Q, demo
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
| `matplotlib` | >=3.7 | Plots |

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

Sau khi train xong, models được lưu tự động:

```
models/
├── qlearning_qtable.npy
├── qlearning_metrics.pkl
├── sarsa_qtable.npy
└── sarsa_metrics.pkl
```

---

## Demo Pygame

> Yêu cầu màn hình (không chạy được headless).

```powershell
python main.py --demo --algo qlearning
python main.py --demo --algo sarsa
```

Cửa sổ Pygame mở ra, agent tự di chuyển theo policy đã học. Đóng cửa sổ để thoát.

---

## Sinh biểu đồ so sánh

```powershell
python main.py --plot
```

Tạo file `reports/comparison.png` gồm 4 panel:
- Reward curves (Q-Learning vs SARSA)
- Success rate curves
- Q-value heatmap — action UP
- Q-value heatmap — action DOWN

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

## Chạy tất cả trong một lệnh (Q-Learning, SARSA)

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
```

---

## Kết quả (10×10 maze, seed=42)

| Thuật toán | Success Rate | Steps | Training time | Ghi chú |
|------------|-------------|-------|---------------|---------|
| Q-Learning | ≥ 80% | 40 (optimal) | ~10 giây | Off-policy, sample-efficient |
| SARSA | ≥ 75% | 40 (optimal) | ~10 giây | On-policy, conservative hơn |

> **Note:** Maze DFS seed=42 có đúng một đường đi dài 40 bước từ (0,0) đến (9,9).
> Cả 2 thuật toán đều tìm được đường đó.

---

## Môi trường chi tiết

- **Observation:** flat `float32` vector, shape `(2N+1)²` — ví dụ 441 với N=10
- **Actions:** `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`
- **Reward:** `-1` mỗi bước, `+100` khi đến goal
- **Terminated:** agent đến goal cell `(N-1, N-1)`
- **Truncated:** vượt quá `N²×4` bước
- **Maze:** Recursive Backtracker DFS — perfect maze (đúng 1 đường đi giữa 2 ô bất kỳ)

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
3. Train **Q-Learning** trực tiếp trên mê cung này (không dùng model pretrained) ở chế độ headless (nhanh, không render), in tiến trình ra console mỗi 50 episode. Cửa sổ Pygame hiện dòng chữ "Dang xu ly maze, vui long cho..." trong lúc training (agent chưa di chuyển). Dừng khi đạt 90% rolling success rate **và** policy greedy thực sự đi tới đích (hoặc tối đa 1000 episodes).
4. Sau đó animate 1 episode greedy (real-time, 10 FPS) để show đường đi tối ưu tới đích — dòng chữ "Dang xu ly..." tự biến mất khi agent bắt đầu di chuyển.
5. Lặp lại với **SARSA**, rồi **Dyna-Q**.
6. In bảng so sánh: số episodes để hội tụ + độ dài đường đi cuối cùng.

**Maze không thể giải:** trước khi train, demo kiểm tra (BFS) xem có đường đi từ start đến goal hay không. Nếu maze bị chặn hoàn toàn, demo dừng ngay và báo "KHONG THE GIAI DUOC". Nếu maze giải được nhưng không hội tụ trong 1000 episodes, demo vẫn animate policy hiện tại và báo "Khong hoi tu sau N episodes...".

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
python test_function/phase6_unseen_maze.py # Pool generation, maze loading, Dyna-Q, demo
```
