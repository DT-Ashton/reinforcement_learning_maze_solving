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

### DQN (~5–10 phút trên CPU)

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
  total_timesteps: 100000
  learning_rate: 0.001
  buffer_size: 10000
  batch_size: 64
  gamma: 0.99
  exploration_fraction: 0.3
```

---

## Kết quả (10×10 maze, seed=42)

| Thuật toán | Success Rate (3000 ep) | Ghi chú |
|------------|------------------------|---------|
| Q-Learning | ≥ 80% | Off-policy, hội tụ nhanh |
| SARSA | ≥ 75% | On-policy, conservative hơn |
| DQN | — | 100k steps, ~5–10 phút CPU |

---

## Môi trường chi tiết

- **Observation:** flat `float32` vector, shape `(2N+1)²` — ví dụ 441 với N=10
- **Actions:** `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`
- **Reward:** `-1` mỗi bước, `+100` khi đến goal
- **Terminated:** agent đến goal cell `(N-1, N-1)`
- **Truncated:** vượt quá `N²×4` bước
- **Maze:** Recursive Backtracker DFS — perfect maze (đúng 1 đường đi giữa 2 ô bất kỳ)
