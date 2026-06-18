"""
Live demo: agent gap maze chua tung thay, hoc truc tiep (online).
So sanh toc do hoi tu giua Q-Learning, SARSA, va Dyna-Q.

Run:
    python -m maze.pool_generator   # tao maze_pool/ neu chua co
    python demo_unseen_maze.py
    python demo_unseen_maze.py --maze maze_pool/maze_10x10_seed1100.json
"""

import argparse
import json
import os
import random
from collections import deque

import pygame

from env.maze_loader import load_maze_env
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dyna_q import DynaQAgent
from metrics.training_metrics import TrainingMetrics
from visualization.pygame_renderer import PygameRenderer


def pick_maze_file(maze_pool_dir: str = "maze_pool", path: str = None, size: int = 10) -> str:
    if path is not None:
        return path

    index_path = os.path.join(maze_pool_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Khong tim thay {index_path}. "
            f"Chay 'python -m maze.pool_generator' de tao maze pool truoc."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    candidates = [e for e in index if e["size"] == size] or index
    entry = random.choice(candidates)
    return os.path.join(maze_pool_dir, os.path.basename(entry["file"]))


def maze_is_solvable(env) -> bool:
    """BFS tu start (0,0) den goal (size-1,size-1) tren _base_grid.
    Kiem tra truoc khi train de tranh chay het episode_cap (vo ich) tren
    mot maze khong co duong di -- demo se dung lai va bao loi ngay."""
    size = env.size
    grid = env._base_grid
    start = (0, 0)
    goal = (size - 1, size - 1)

    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and grid[2 * r + 1 + dr, 2 * c + 1 + dc] == 1
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return goal in visited


def wait_for_space(env) -> None:
    font = pygame.font.SysFont(None, 28)
    text_surface = font.render("Press SPACE to start, ESC to quit", True, (255, 255, 0))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

        PygameRenderer.draw(env.window, env._get_obs(), env.size)
        env.window.blit(text_surface, (10, 10))
        pygame.display.flip()
        env.clock.tick(env.metadata["render_fps"])


def _show_processing_message(env) -> None:
    """Ve lai maze + text 'Dang xu ly...' tren window hien co (window khong
    tu cap nhat trong luc training headless, ket qua se trong nhu 'dung im')."""
    font = pygame.font.SysFont(None, 28)
    text_surface = font.render("Dang xu ly maze, vui long cho...", True, (255, 255, 0))
    PygameRenderer.draw(env.window, env._get_obs(), env.size)
    env.window.blit(text_surface, (10, 10))
    pygame.display.flip()
    pygame.event.pump()


def _run_qlearning_episode(agent, env):
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(state, explore=True)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        s_next = agent._obs_to_state(obs_next)
        # Fix (Phase 2 reviewer): pass done=terminated only, NOT (terminated or truncated)
        agent.update(state, action, reward, s_next, terminated)
        state = s_next
        total_reward += reward
        steps += 1

    return total_reward, steps, terminated


def _run_sarsa_episode(agent, env):
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    action = agent.get_action(state, explore=True)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs_next, reward, terminated, truncated, _ = env.step(action)
        s_next = agent._obs_to_state(obs_next)
        a_next = agent.get_action(s_next, explore=True)
        # Fix (Phase 2 reviewer): pass done=terminated only, NOT (terminated or truncated)
        agent.update(state, action, reward, s_next, terminated, a_next)
        state = s_next
        action = a_next
        total_reward += reward
        steps += 1

    return total_reward, steps, terminated


def _greedy_reaches_goal(agent, env) -> bool:
    """Chay 1 episode greedy (khong render) de kiem tra policy hien tai
    co thuc su giai duoc maze hay chua (90% rolling success rate khi train
    van con epsilon cao, khong dam bao greedy policy da hoi tu)."""
    saved_mode = env.render_mode
    env.render_mode = None
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(state, explore=False)
        obs, _, terminated, truncated, _ = env.step(action)
        state = agent._obs_to_state(obs)

    env.render_mode = saved_mode
    return terminated


def _run_greedy_episode(agent, env):
    obs, _ = env.reset()
    state = agent._obs_to_state(obs)
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit
        action = agent.get_action(state, explore=False)
        obs, _, terminated, truncated, _ = env.step(action)
        state = agent._obs_to_state(obs)
        steps += 1

    return steps


def run_live(agent_cls, env, episode_cap: int = 1000,
             target_success: float = 0.9, progress_every: int = 50):
    agent = agent_cls(env)
    metrics = TrainingMetrics(algo_name=agent.ALGO_NAME, maze_size=env.size, n_episodes=0)
    episodes_to_converge = None

    # Train headless (fast) -- rendering at 10 FPS during 100s-1000s of
    # episodes would take minutes and made the agent look "stuck" on the
    # frozen last-rendered frame. Print progress to console instead.
    env.render_mode = None
    _show_processing_message(env)
    for ep in range(episode_cap):
        if isinstance(agent, SARSAAgent):
            total_reward, steps, terminated = _run_sarsa_episode(agent, env)
        else:
            total_reward, steps, terminated = _run_qlearning_episode(agent, env)

        agent._decay_epsilon()
        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.success_flags.append(bool(terminated))
        window = metrics.success_flags[max(0, ep - 99): ep + 1]
        metrics.rolling_success_rate.append(sum(window) / len(window))
        metrics.n_episodes = ep + 1

        pygame.event.pump()

        if (ep + 1) % progress_every == 0:
            print(f"  episode {ep + 1:4d}  rolling_success={metrics.rolling_success_rate[-1]:.0%}"
                  f"  epsilon={agent.epsilon:.3f}")

        if ep >= 99 and metrics.rolling_success_rate[-1] >= target_success:
            if _greedy_reaches_goal(agent, env):
                episodes_to_converge = ep + 1
                break

    if episodes_to_converge is None:
        print(f"  -> Khong hoi tu sau {metrics.n_episodes} episodes "
              f"(rolling success cuoi: {metrics.rolling_success_rate[-1]:.0%}). "
              f"Hien animation voi policy hien tai...")

    # Animate the (converged or best-effort) greedy policy.
    env.render_mode = "human"
    final_path_length = _run_greedy_episode(agent, env)

    return metrics, episodes_to_converge, final_path_length


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live demo: Q-Learning vs SARSA vs Dyna-Q tren maze chua tung thay"
    )
    parser.add_argument("--maze", type=str, default=None,
                        help="Path toi 1 file maze JSON (mac dinh: random 10x10 tu maze_pool/)")
    parser.add_argument("--size", type=int, default=10,
                        help="Kich thuoc maze de pick neu khong dung --maze")
    parser.add_argument("--episodes", type=int, default=None,
                        help="So episode toi da khi train (mac dinh: 1000 voi size<=15, tu dong tang voi size lon hon)")
    args = parser.parse_args()

    path = pick_maze_file(path=args.maze, size=args.size)
    print(f"Maze: {path}")

    if args.episodes is not None:
        episode_cap = args.episodes
    else:
        # auto-scale: 1000 cho size<=15, tang theo N^2 voi size lon hon
        episode_cap = max(1000, args.size * args.size * 4)
    print(f"episode_cap={episode_cap}")

    algos = [
        ("Q-Learning", QLearningAgent),
        ("SARSA", SARSAAgent),
        ("Dyna-Q", DynaQAgent),
    ]
    results = []

    for i, (label, agent_cls) in enumerate(algos):
        env = load_maze_env(path, render_mode="human")
        env.reset()

        if i == 0:
            if not maze_is_solvable(env):
                print(f"\nMaze {path} KHONG THE GIAI DUOC "
                      f"(khong co duong tu start den goal). Dung demo.")
                env.close()
                return
            wait_for_space(env)

        print(f"\n=== {label} ===")
        metrics, eps_to_converge, path_len = run_live(agent_cls, env, episode_cap=episode_cap)
        results.append((label, eps_to_converge, path_len, metrics.n_episodes))

        conv_str = str(eps_to_converge) if eps_to_converge is not None \
            else f">{metrics.n_episodes} (chua hoi tu)"
        print(f"  episodes_to_converge={conv_str}, final_path_length={path_len}")

        env.close()

    print("\n=== So sanh hoi tu ===")
    print(f"{'Thuat toan':<12} {'Episodes':<18} {'Path length':<12}")
    for label, eps, plen, n_ep in results:
        eps_str = str(eps) if eps is not None else f">{n_ep} (chua hoi tu)"
        print(f"{label:<12} {eps_str:<18} {plen:<12}")


if __name__ == "__main__":
    main()
