# Brainstorm: Maze Solving with Reinforcement Learning

**Date:** 2026-06-04

---

## Ideas Explored

| Hướng | Mô tả | Quyết định |
|-------|--------|------------|
| Dùng Stable-Baselines3 cho tất cả | Nhanh, ít code. Nhưng không học được internals. | Bác bỏ — mục tiêu môn học là hiểu thuật toán |
| Tự build tất cả (kể cả DQN) | Học sâu nhất. DQN from scratch phức tạp, dễ bug. | Bác bỏ DQN scratch — dùng SB3 cho phase 3 |
| **3 giai đoạn rõ ràng (được chọn)** | Phase 1: Env, Phase 2: Tabular RL, Phase 3: DQN | **Chọn** |
| OpenAI Gym vs Gymnasium | Gymnasium là bản cập nhật mới, active hơn. | Dùng Gymnasium |
| Prim's vs Recursive Backtracker | Recursive Backtracker đơn giản hơn, maze đẹp hơn. | Recursive Backtracker |
| Chỉ matplotlib vs cả pygame | Pygame cho demo, matplotlib cho metrics — linh hoạt. | Cả hai với mode riêng |

---

## User's Direction

3 giai đoạn tuần tự:
1. **Maze Generator + Gymnasium Environment** — nền tảng cho cả project
2. **Q-Learning và SARSA tự implement** — hiểu tabular RL từ đầu
3. **DQN dùng Stable-Baselines3** — deep RL với thư viện hỗ trợ

Visualization: pygame cho demo agent chạy live, matplotlib cho reward curves và heatmaps (2 mode riêng biệt).

---

## Open Questions

- Kích thước maze mặc định và tối đa chưa được xác nhận → giả định 10×10 default, tối đa 30×30
- Reward structure chưa được chỉ định → dùng standard: -1/step, +100 goal, wall = blocked (không penalty riêng)
- Số episodes training mặc định chưa rõ → giả định Q/SARSA: 5000 ep, DQN: 100k steps

## Risks

1. **State space cho DQN**: Nếu dùng full grid làm observation → input lớn (NxN), DQN cần nhiều episode hơn. Có thể dùng (row, col) flatten để đơn giản hơn.
2. **Pygame performance**: Real-time rendering khi training Q-Learning có thể làm chậm đáng kể. Cần có flag `render_mode='human'` vs `None`.
3. **SB3 compatibility**: Gymnasium API mới nhất có thể cần điều chỉnh nhỏ với SB3 (truncated vs terminated).

---

## Actual Results (2026-06-05)

### Risks materialised

**Risk 1 (State space DQN) — confirmed and resolved:**
- Raw 441-dim obs → DQN không học được (0% success sau 300k steps)
- 4-dim position obs → DQN học được trên 5×5 nhưng 10×10 vẫn 0% eval (policy cycles)
- **Fix:** 8-dim obs `[ar/N, ac/N, can_N, can_S, can_W, can_E, dr/N, dc/N]` → 100% success
- Root cause của cycles: position-only obs không có wall info → deterministic greedy policy bị kẹt

**Risk 3 (SB3 compatibility) — minor:** Cần `progress_bar=True` → phải install `tqdm` + `rich`.

### Final Numbers (10×10, seed=42)

| Thuật toán | Success | Steps | Training |
|---|---|---|---|
| Q-Learning | ~100% | 40 | 3000 ep, ~10 giây |
| SARSA | ~100% | 40 | 3000 ep, ~10 giây |
| DQN | 100% | 40 | 1M steps, ~17 phút |

**40 steps là optimal** — DFS maze seed=42 chỉ có 1 đường đi dài đúng 40 ô từ (0,0) đến (9,9).

### Key Lesson về DQN vs Tabular

DQN cần ~1000× compute hơn (1M steps vs 3000 episodes) nhưng đạt kết quả tương đương trên bài toán discrete state space nhỏ. Đây là kết quả học thuật phù hợp: tabular methods dominate khi state space nhỏ và có thể enumerate; DQN có lợi thế khi state space liên tục hoặc rất lớn.
