"""
So sanh BFS (classical) vs Q-Learning / SARSA / Dyna-Q (RL) tren cung 1 maze.

Metrics:
  - path_length   : so buoc tu start den goal (it hon = tot hon)
  - vs_bfs        : chenh lech % so voi BFS optimal (0% = optimal)
  - episodes      : so episode can de hoi tu (RL only)
  - train_time_s  : tong thoi gian train (RL) hoac thoi gian giai (BFS, ms)

Usage:
  cd reinforcement_learning_maze_solving
  python comparison/run_comparison.py
  python comparison/run_comparison.py --maze maze_pool/maze_10x10_seed1100.json
  python comparison/run_comparison.py --size 15 --runs 10 --episodes 2000
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.maze_loader import load_maze_env
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dyna_q import DynaQAgent
from comparison.bfs_solver import bfs_shortest_path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _greedy_path_length(agent, env) -> int | None:
    """Chay 1 greedy episode (no render), tra ve so buoc hoac None neu khong den goal."""
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = agent.get_action(state, explore=False)
        obs, _, terminated, truncated, _ = env.step(action)
        state = agent._obs_to_state(obs)
        steps += 1
    return steps if terminated else None


def _run_episode_qlearning(agent, env):
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = agent.get_action(state, explore=True)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        s_next = agent._obs_to_state(obs_next)
        agent.update(state, action, reward, s_next, terminated)
        state = s_next
        steps += 1
    return bool(terminated), steps


def _run_episode_sarsa(agent, env):
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    action = agent.get_action(state, explore=True)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        obs_next, reward, terminated, truncated, _ = env.step(action)
        s_next = agent._obs_to_state(obs_next)
        a_next = agent.get_action(s_next, explore=True)
        agent.update(state, action, reward, s_next, terminated, a_next)
        state = s_next
        action = a_next
        steps += 1
    return bool(terminated), steps


def train_rl(agent_cls, env, episode_cap: int, target_success: float = 0.9):
    """Train 1 agent headless. Returns (episodes_to_converge, final_path_len, train_time_s, total_steps)."""
    agent = agent_cls(env)
    success_flags = []
    episodes_to_converge = None
    total_steps = 0

    t0 = time.perf_counter()
    for ep in range(episode_cap):
        if isinstance(agent, SARSAAgent):
            success, steps = _run_episode_sarsa(agent, env)
        else:
            success, steps = _run_episode_qlearning(agent, env)

        agent._decay_epsilon()
        success_flags.append(success)
        total_steps += steps

        if ep >= 99 and episodes_to_converge is None:
            rolling = sum(success_flags[max(0, ep - 99): ep + 1]) / min(ep + 1, 100)
            if rolling >= target_success:
                if _greedy_path_length(agent, env) is not None:
                    episodes_to_converge = ep + 1
                    break

    train_time = time.perf_counter() - t0
    final_path = _greedy_path_length(agent, env)
    return episodes_to_converge, final_path, train_time, total_steps


def pick_maze(maze_arg, size):
    if maze_arg:
        return maze_arg
    index_path = os.path.join(os.path.dirname(__file__), "..", "maze_pool", "index.json")
    with open(index_path) as f:
        index = json.load(f)
    candidates = [e for e in index if e["size"] == size] or index
    random.seed(42)
    entry = random.choice(candidates)
    return os.path.join(os.path.dirname(__file__), "..", "maze_pool", os.path.basename(entry["file"]))


def main():
    parser = argparse.ArgumentParser(description="So sanh BFS vs RL tren maze")
    parser.add_argument("--maze", default=None, help="Path den file maze JSON")
    parser.add_argument("--size", type=int, default=10, help="Kich thuoc maze (neu khong dung --maze)")
    parser.add_argument("--runs", type=int, default=5, help="So lan chay moi thuat toan de lay trung binh")
    parser.add_argument("--episodes", type=int, default=1500, help="Episode toi da moi lan chay RL")
    args = parser.parse_args()

    maze_path = pick_maze(args.maze, args.size)
    print(f"Maze: {maze_path}\n")

    # ---- BFS ----
    env_bfs = load_maze_env(maze_path, render_mode=None)
    env_bfs.reset()
    t0 = time.perf_counter()
    bfs_path = bfs_shortest_path(env_bfs)
    bfs_time_s = time.perf_counter() - t0
    bfs_len = len(bfs_path) - 1 if bfs_path else None
    print(f"[BFS] path_length={bfs_len} steps | time={bfs_time_s * 1000:.3f} ms")

    # ---- RL ----
    algos = [
        ("Q-Learning", QLearningAgent),
        ("SARSA",      SARSAAgent),
        ("Dyna-Q",     DynaQAgent),
    ]

    rl_results = {}
    for label, agent_cls in algos:
        print(f"\n[{label}] {args.runs} runs x max {args.episodes} episodes …")
        ep_list, path_list, time_list, steps_list = [], [], [], []

        for run in range(args.runs):
            env = load_maze_env(maze_path, render_mode=None)
            env.reset()
            eps, path_len, t, total_steps = train_rl(agent_cls, env, args.episodes)
            ep_list.append(eps if eps is not None else args.episodes)
            path_list.append(path_len if path_len is not None else 9999)
            time_list.append(t)
            steps_list.append(total_steps)
            conv_str = str(eps) if eps else f">{args.episodes}"
            print(f"  run {run + 1}: episodes={conv_str}, path={path_len}, time={t:.2f}s")

        gap_pct = (np.mean(path_list) - bfs_len) / bfs_len * 100 if bfs_len else None
        rl_results[label] = {
            "eps_mean":   np.mean(ep_list),
            "eps_std":    np.std(ep_list),
            "path_mean":  np.mean(path_list),
            "gap_pct":    gap_pct,
            "time_mean":  np.mean(time_list),
            "steps_mean": np.mean(steps_list),
        }

    # ---- Print table ----
    W = 72
    print("\n" + "=" * W)
    print(f"{'Method':<12} {'Episodes':>14} {'Path len':>10} {'vs BFS':>10} {'Train time':>14}")
    print("-" * W)
    print(f"{'BFS':<12} {'— (1-shot)':>14} {bfs_len!s:>10} {'0% (optimal)':>10} {bfs_time_s * 1000:>11.3f} ms")
    for label, r in rl_results.items():
        gap_str = f"+{r['gap_pct']:.1f}%" if r["gap_pct"] is not None else "—"
        print(
            f"{label:<12} "
            f"{r['eps_mean']:>8.0f} ±{r['eps_std']:.0f} "
            f"{r['path_mean']:>10.1f} "
            f"{gap_str:>10} "
            f"{r['time_mean']:>12.2f} s"
        )
    print("=" * W)
    print(
        "\nKey insight: BFS giai trong <1ms vi biet toan bo map."
        "\nRL khong biet map truoc, phai kham pha — chi co y nghia khi map an/stochastic/thay doi."
    )

    # ---- Save CSV ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "episodes_mean", "episodes_std", "path_mean", "vs_bfs_pct", "train_time_s"])
        w.writerow(["BFS", "1-shot", "—", bfs_len, "0.0", f"{bfs_time_s * 1000:.4f}ms"])
        for label, r in rl_results.items():
            w.writerow([
                label,
                f"{r['eps_mean']:.1f}",
                f"{r['eps_std']:.1f}",
                f"{r['path_mean']:.1f}",
                f"{r['gap_pct']:.1f}" if r["gap_pct"] is not None else "—",
                f"{r['time_mean']:.3f}",
            ])
    print(f"\nCSV saved -> {csv_path}")

    # ---- Save plots ----
    labels_rl = list(rl_results.keys())
    colors_rl = ["#4C72B0", "#DD8452", "#55A868"]
    color_bfs = "#2ca02c"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"RL vs Classical (BFS) — Maze {args.size}×{args.size}, {args.runs} runs",
        fontweight="bold", fontsize=13,
    )

    # --- Plot 1: episodes to converge ---
    ep_means = [rl_results[l]["eps_mean"] for l in labels_rl]
    ep_stds  = [rl_results[l]["eps_std"]  for l in labels_rl]
    axes[0].bar(labels_rl, ep_means, yerr=ep_stds, capsize=6,
                color=colors_rl, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Episodes to Converge (RL only)")
    axes[0].set_ylabel("Episodes")
    axes[0].set_xlabel("Algorithm")

    # --- Plot 2: path length vs BFS optimal ---
    all_labels = ["BFS\n(optimal)"] + labels_rl
    all_paths  = [bfs_len] + [rl_results[l]["path_mean"] for l in labels_rl]
    all_colors = [color_bfs] + colors_rl
    bars = axes[1].bar(all_labels, all_paths, color=all_colors, edgecolor="white", linewidth=1.2)
    axes[1].axhline(y=bfs_len, color=color_bfs, linestyle="--", alpha=0.6, label=f"BFS optimal ({bfs_len} steps)")
    axes[1].set_title("Final Path Length vs BFS Optimal")
    axes[1].set_ylabel("Steps to goal")
    axes[1].legend(fontsize=9)
    for bar, val in zip(bars, all_paths):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # --- Plot 3: wall-clock time (log scale) ---
    all_times  = [bfs_time_s] + [rl_results[l]["time_mean"] for l in labels_rl]
    axes[2].bar(all_labels, all_times, color=all_colors, edgecolor="white", linewidth=1.2)
    axes[2].set_yscale("log")
    axes[2].set_title("Wall-Clock Time (log scale)")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xlabel("Algorithm")
    axes[2].annotate("BFS: < 1ms", xy=(0, bfs_time_s), xytext=(0.5, bfs_time_s * 20),
                     fontsize=8, color=color_bfs,
                     arrowprops=dict(arrowstyle="->", color=color_bfs, lw=1))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "rl_vs_classical.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
