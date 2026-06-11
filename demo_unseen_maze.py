"""
Live demo: agent gap maze chua tung thay, hoc truc tiep (online).
So sanh toc do hoi tu giua Q-Learning, SARSA, va Dyna-Q.

Run:
    python -m maze.pool_generator   # tao mazes/ neu chua co
    python demo_unseen_maze.py
    python demo_unseen_maze.py --maze mazes/maze_10x10_seed1100.json
"""

import argparse
import json
import os
import random

import pygame

from env.maze_loader import load_maze_env
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dyna_q import DynaQAgent
from metrics.training_metrics import TrainingMetrics
from visualization.pygame_renderer import PygameRenderer


def pick_maze_file(mazes_dir: str = "mazes", path: str = None, size: int = 10) -> str:
    if path is not None:
        return path

    index_path = os.path.join(mazes_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Khong tim thay {index_path}. "
            f"Chay 'python -m maze.pool_generator' de tao maze pool truoc."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    candidates = [e for e in index if e["size"] == size] or index
    entry = random.choice(candidates)
    return os.path.join(mazes_dir, os.path.basename(entry["file"]))


def wait_for_space(env) -> None:
    font = pygame.font.SysFont(None, 28)
    text_surface = font.render("Press SPACE to start", True, (255, 255, 0))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

        PygameRenderer.draw(env.window, env._get_obs(), env.size)
        env.window.blit(text_surface, (10, 10))
        pygame.display.flip()
        env.clock.tick(env.metadata["render_fps"])


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
             target_success: float = 0.9, render_every: int = 200):
    agent = agent_cls(env)
    metrics = TrainingMetrics(algo_name=agent.ALGO_NAME, maze_size=env.size, n_episodes=0)
    episodes_to_converge = None

    for ep in range(episode_cap):
        env.render_mode = "human" if (ep < 10 or ep % render_every == 0) else None

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

        if env.render_mode is None:
            pygame.event.pump()

        if ep >= 99 and metrics.rolling_success_rate[-1] >= target_success:
            episodes_to_converge = ep + 1
            break

    env.render_mode = "human"
    final_path_length = _run_greedy_episode(agent, env)

    return metrics, episodes_to_converge, final_path_length


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live demo: Q-Learning vs SARSA vs Dyna-Q tren maze chua tung thay"
    )
    parser.add_argument("--maze", type=str, default=None,
                        help="Path toi 1 file maze JSON (mac dinh: random 10x10 tu mazes/)")
    parser.add_argument("--size", type=int, default=10,
                        help="Kich thuoc maze de pick neu khong dung --maze")
    args = parser.parse_args()

    path = pick_maze_file(path=args.maze, size=args.size)
    print(f"Maze: {path}")

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
            wait_for_space(env)

        print(f"\n=== {label} ===")
        metrics, eps_to_converge, path_len = run_live(agent_cls, env)
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
