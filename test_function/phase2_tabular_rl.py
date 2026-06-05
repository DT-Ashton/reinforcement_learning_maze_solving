# -*- coding: utf-8 -*-
"""
Phase 2 Test -- Q-Learning & SARSA (Tabular RL)
===============================================
Tests:
  1. Q-table shape and initialization
  2. Q-Learning update rule (off-policy Bellman)
  3. SARSA update rule (on-policy Bellman)
  4. Epsilon-greedy action selection
  5. Short training run -> success rate > 0%
  6. Q-Learning vs SARSA: both converge on small maze

Run:
    python test_function/phase2_tabular_rl.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


# --------------------------------------------------
# Test 1: Q-table shape and initialization
# --------------------------------------------------
def test_qtable_init():
    from env.maze_env import MazeEnv
    from algorithms.q_learning import QLearningAgent

    print("\n=== Test 1: Q-table init ===")
    env = MazeEnv(size=5, seed=42)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0)

    assert agent.q_table.shape == (25, 4), f"Wrong Q-table shape: {agent.q_table.shape}"
    assert np.all(agent.q_table == 0.0), "Q-table should start at zeros"
    print(f"  Q-table shape={agent.q_table.shape}, all zeros  OK")

    env.close()
    print("PASS: Q-table init\n")


# --------------------------------------------------
# Test 2: Q-Learning update rule
# --------------------------------------------------
def test_qlearning_update():
    from env.maze_env import MazeEnv
    from algorithms.q_learning import QLearningAgent

    print("=== Test 2: Q-Learning update rule ===")
    env = MazeEnv(size=5, seed=42)
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, epsilon=0.0)

    # Manual update: s=0, a=1, r=10, s_next=1, done=False
    # Expected: Q[0,1] += 0.5 * (10 + 0.9 * max(Q[1]) - Q[0,1])
    #         = 0 + 0.5 * (10 + 0 - 0) = 5.0
    agent.update(s=0, a=1, r=10.0, s_next=1, done=False)
    assert abs(agent.q_table[0, 1] - 5.0) < 1e-6, f"Got {agent.q_table[0, 1]}"
    print(f"  Q[0,1] after update = {agent.q_table[0,1]:.4f} (expected 5.0)  OK")

    # done=True: bootstrap term = 0
    # Q[0,2] += 0.5 * (100 + 0 - 0) = 50.0
    agent2 = QLearningAgent(env, alpha=0.5, gamma=0.9, epsilon=0.0)
    agent2.update(s=0, a=2, r=100.0, s_next=1, done=True)
    assert abs(agent2.q_table[0, 2] - 50.0) < 1e-6, f"Got {agent2.q_table[0, 2]}"
    print(f"  Q[0,2] terminal update = {agent2.q_table[0,2]:.4f} (expected 50.0)  OK")

    env.close()
    print("PASS: Q-Learning update\n")


# --------------------------------------------------
# Test 3: SARSA update rule
# --------------------------------------------------
def test_sarsa_update():
    from env.maze_env import MazeEnv
    from algorithms.sarsa import SARSAAgent

    print("=== Test 3: SARSA update rule ===")
    env = MazeEnv(size=5, seed=42)
    agent = SARSAAgent(env, alpha=0.5, gamma=0.9, epsilon=0.0)

    # SARSA dung Q[s_next, a_next] thay vi max Q[s_next]
    # Q[0,1] += 0.5 * (10 + 0.9 * Q[1,2] - Q[0,1]) = 5.0 (Q[1,2]=0)
    agent.update(s=0, a=1, r=10.0, s_next=1, done=False, a_next=2)
    assert abs(agent.q_table[0, 1] - 5.0) < 1e-6, f"Got {agent.q_table[0, 1]}"
    print(f"  Q[0,1] = {agent.q_table[0,1]:.4f} (expected 5.0)  OK")

    # SARSA vs Q-Learning differ when next Q values differ
    agent2 = SARSAAgent(env, alpha=0.5, gamma=0.9, epsilon=0.0)
    agent2.q_table[1, 3] = 20.0
    # SARSA: Q[0,0] += 0.5*(5 + 0.9*20 - 0) = 0.5*23 = 11.5
    agent2.update(s=0, a=0, r=5.0, s_next=1, done=False, a_next=3)
    assert abs(agent2.q_table[0, 0] - 11.5) < 1e-6, f"Got {agent2.q_table[0, 0]}"
    print(f"  Q[0,0] with a_next Q=20 -> {agent2.q_table[0,0]:.4f} (expected 11.5)  OK")

    env.close()
    print("PASS: SARSA update\n")


# --------------------------------------------------
# Test 4: Epsilon-greedy
# --------------------------------------------------
def test_epsilon_greedy():
    from env.maze_env import MazeEnv
    from algorithms.q_learning import QLearningAgent

    print("=== Test 4: Epsilon-greedy ===")
    env = MazeEnv(size=5, seed=42)

    # epsilon=1.0 -> all random
    agent = QLearningAgent(env, epsilon=1.0)
    actions = [agent.get_action(0, explore=True) for _ in range(200)]
    unique = set(actions)
    assert len(unique) > 1, "epsilon=1.0 should explore all actions"
    print(f"  epsilon=1.0 -> {len(unique)} unique actions in 200 samples  OK")

    # epsilon=0.0 -> always greedy
    agent2 = QLearningAgent(env, epsilon=0.0)
    agent2.q_table[0, 2] = 99.0  # best action = 2
    greedy_actions = [agent2.get_action(0, explore=False) for _ in range(20)]
    assert all(a == 2 for a in greedy_actions), "epsilon=0 should always pick best action"
    print("  epsilon=0.0 -> always action 2 (best Q)  OK")

    env.close()
    print("PASS: Epsilon-greedy\n")


# --------------------------------------------------
# Test 5: Short training run
# --------------------------------------------------
def test_short_training():
    from env.maze_env import MazeEnv
    from algorithms.q_learning import QLearningAgent
    from algorithms.sarsa import SARSAAgent

    print("=== Test 5: Short training (500 episodes, 5x5 maze) ===")

    for AgentClass, name in [(QLearningAgent, "Q-Learning"), (SARSAAgent, "SARSA")]:
        env = MazeEnv(size=5, seed=42)
        agent = AgentClass(env, alpha=0.1, gamma=0.99,
                           epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01)
        metrics = agent.train(n_episodes=500)

        sr = metrics.final_success_rate()
        assert sr > 0.0, f"{name}: success rate should be > 0 after 500 episodes"
        assert len(metrics.episode_rewards) == 500
        assert len(metrics.rolling_success_rate) == 500
        print(f"  {name}: success_rate={sr:.1%}, episodes={len(metrics.episode_rewards)}  OK")
        env.close()

    print("PASS: Short training\n")


# --------------------------------------------------
# Test 6: Full convergence (3000 ep, 10x10)
# --------------------------------------------------
def test_convergence():
    from env.maze_env import MazeEnv
    from algorithms.q_learning import QLearningAgent
    from algorithms.sarsa import SARSAAgent

    print("=== Test 6: Convergence 10x10, 3000 episodes ===")
    print("  (mat ~10-20 giay...)")

    targets = {"Q-Learning": 0.80, "SARSA": 0.75}
    results = {}

    for AgentClass, name, target in [
        (QLearningAgent, "Q-Learning", 0.80),
        (SARSAAgent,     "SARSA",      0.75),
    ]:
        env = MazeEnv(size=10, seed=42)
        agent = AgentClass(env, alpha=0.1, gamma=0.99,
                           epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
        metrics = agent.train(n_episodes=3000)
        sr = metrics.final_success_rate()
        results[name] = sr
        status = "OK" if sr >= target else "FAIL"
        print(f"  {name}: success_rate={sr:.1%} (target >={target:.0%})  {status}")
        env.close()

    for name, target in targets.items():
        assert results[name] >= target, \
            f"{name} success rate {results[name]:.1%} < target {target:.0%}"

    print("PASS: Convergence\n")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    test_qtable_init()
    test_qlearning_update()
    test_sarsa_update()
    test_epsilon_greedy()
    test_short_training()
    test_convergence()

    print("=" * 50)
    print("Phase 2: ALL TESTS PASS")
