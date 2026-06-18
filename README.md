# Maze Solving with Reinforcement Learning

So sánh Q-Learning, SARSA và Dyna-Q trên môi trường mê cung sinh ngẫu nhiên.  
Đồ án môn Trí tuệ nhân tạo — Kỳ 8.

---

## Cấu trúc dự án

```
reinforcement_learning_maze_solving/
├── demo_unseen_maze.py      # Entry point chính — live demo
├── requirements.txt
├── env/
│   ├── __init__.py          # Đăng ký MazeEnv-v0 với Gymnasium
│   ├── maze_env.py          # Gymnasium environment
│   └── maze_loader.py       # Load maze từ file JSON
├── maze/
│   ├── generator.py         # Recursive Backtracker DFS
│   └── pool_generator.py    # Sinh pool mê cung ngẫu nhiên
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
├── mazes/                   # Pool mê cung JSON
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
cd "c:\Users\ADMIN\Documents\Kỳ 8 - AI\reinforcement_learning_maze_solving"
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

---

## Demo live (entry point chính)

### Sinh pool mê cung (chạy 1 lần)

```powershell
python -m maze.pool_generator
python -m maze.pool_generator --sizes 10 15 20 --n-per-size 5   # tuỳ chỉnh
```

### Chạy demo

```powershell
python demo_unseen_maze.py                                          # random 10x10
python demo_unseen_maze.py --size 15                                # random 15x15
python demo_unseen_maze.py --maze mazes/maze_10x10_seed1100.json    # maze cụ thể
python demo_unseen_maze.py --size 20 --episodes 1600                # maze lớn hơn
```

**Luồng:** mở Pygame → xem maze tĩnh → **SPACE** bắt đầu → train headless (console progress) → animate greedy → in bảng so sánh. Nhấn **ESC** để thoát.

**3 thuật toán so sánh:** Q-Learning → SARSA → Dyna-Q (mỗi cái train từ đầu, không dùng pretrained).

**Xử lý maze đặc biệt:**
- Maze không có đường đi → báo "KHONG THE GIAI DUOC", dừng ngay
- Không hội tụ trong `episode_cap` → báo "Khong hoi tu sau N episodes", animate policy hiện tại

---

## Dyna-Q — Model-based Q-Learning

`algorithms/dyna_q.py` cung cấp `DynaQAgent`:
- Kế thừa `QLearningAgent`, thêm internal model `dict[(s,a) -> (r, s_next, done)]`.
- Mỗi real transition, chạy thêm `planning_steps=10` mô phỏng TD updates — không cần thêm real interaction.
- **Kết quả:** hội tụ ~17% nhanh hơn Q-Learning thuần (112 vs 138 episodes trung bình trên 10×10).

---

## Jupyter Notebook (báo cáo)

```powershell
jupyter notebook notebooks/maze_rl_analysis.ipynb
```

Hoặc mở bằng **VS Code** → chọn kernel Python → **Run All**.

8 sections: Maze demo → Env check → Q-Learning → SARSA → Comparison → Q-heatmap → Ablation sizes → Conclusions.  
Notebook tự train lại từ đầu nếu không tìm thấy file metrics đã lưu.

---

## Tests

```powershell
python test_function/phase1_maze_env.py    # Maze + Gymnasium env
python test_function/phase2_tabular_rl.py  # Q-Learning + SARSA
python test_function/phase6_unseen_maze.py # Pool generation, maze loading, Dyna-Q, demo
```

---

## Kết quả (10×10 unseen maze)

| Thuật toán | Episodes hội tụ (TB) | Path length | Ghi chú |
|------------|---------------------|-------------|---------|
| Q-Learning | ~138 | optimal | Off-policy |
| SARSA | ~160 | optimal | On-policy, conservative hơn |
| Dyna-Q | ~112 | optimal | Model-based, ~17% nhanh hơn Q-Learning |

---

## Môi trường chi tiết

- **Observation:** flat `float32` vector, shape `(2N+1)²` — ví dụ 441 với N=10
- **Actions:** `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`
- **Reward:** `-1` mỗi bước, `+100` khi đến goal
- **Terminated:** agent đến goal cell `(N-1, N-1)`
- **Truncated:** vượt quá `N²×4` bước
- **Maze:** Recursive Backtracker DFS — perfect maze (đúng 1 đường đi giữa 2 ô bất kỳ)
