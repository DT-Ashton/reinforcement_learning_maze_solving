# -*- coding: utf-8 -*-
"""
Phase 6 Test -- Unseen Maze Live Demo (Maze Pool, Preset Grid, Dyna-Q)
=======================================================================
Tests:
  1. Maze pool generation -- generate_pool() writes valid JSON files
  2. MazeEnv preset grid -- grid persists across resets (no regeneration)
  3. load_maze_env -- loads a pool file into a working MazeEnv
  4. DynaQAgent model population -- model dict gets populated during training
  5. Dyna-Q sample efficiency -- Dyna-Q converges in <= episodes vs Q-Learning

Run:
    python tests/test_unseen_maze_demo.py
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

_TEST_POOL_DIR = os.path.abspath("./maze_pool/_test_pool")


def _cleanup():
    if os.path.exists(_TEST_POOL_DIR):
        shutil.rmtree(_TEST_POOL_DIR)


# --------------------------------------------------
# Test 1: Maze pool generation
# --------------------------------------------------
def test_pool_generation():
    from maze.pool_generator import generate_pool
    import json

    print("\n=== Test 1: Maze pool generation ===")
    _cleanup()

    files = generate_pool(sizes=[5], n_per_size=1, base_seed=900, out_dir=_TEST_POOL_DIR)
    assert len(files) == 1, f"Expected 1 file, got {len(files)}"
    assert os.path.exists(files[0]), "Maze JSON file not written"

    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["size"] == 5
    assert data["start"] == [0, 0]
    assert data["goal"] == [4, 4]
    grid = np.array(data["grid"], dtype=np.int32)
    assert grid.shape == (11, 11), f"Expected (11, 11), got {grid.shape}"

    index_path = os.path.join(_TEST_POOL_DIR, "index.json")
    assert os.path.exists(index_path), "index.json not written"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    assert len(index) == 1
    assert index[0]["size"] == 5

    print("  pool generation valid                OK")
    print("PASS: Maze pool generation\n")


# --------------------------------------------------
# Test 2: MazeEnv preset grid persists
# --------------------------------------------------
def test_preset_grid_env():
    from env.maze_env import MazeEnv
    from maze.generator import MazeGenerator

    print("=== Test 2: MazeEnv preset grid ===")
    grid = MazeGenerator(5, seed=999).generate()

    env = MazeEnv(size=5, grid=grid)
    obs1, _ = env.reset()
    base_grid_1 = env._base_grid.copy()

    obs2, _ = env.reset()
    base_grid_2 = env._base_grid.copy()

    assert np.array_equal(base_grid_1, base_grid_2), "Preset grid changed across resets"
    assert np.array_equal(base_grid_1, grid), "Preset grid does not match input grid"
    print("  preset grid persists across resets   OK")
    env.close()
    print("PASS: MazeEnv preset grid\n")


# --------------------------------------------------
# Test 3: load_maze_env
# --------------------------------------------------
def test_load_maze_env():
    from maze.pool_generator import generate_pool
    from env.maze_loader import load_maze_env

    print("=== Test 3: load_maze_env ===")
    _cleanup()
    files = generate_pool(sizes=[5], n_per_size=1, base_seed=910, out_dir=_TEST_POOL_DIR)

    env = load_maze_env(files[0])
    assert env.size == 5
    obs, _ = env.reset()
    assert obs.shape == ((2 * 5 + 1) * (2 * 5 + 1),)
    print("  load_maze_env produces working env   OK")
    env.close()
    _cleanup()
    print("PASS: load_maze_env\n")


# --------------------------------------------------
# Test 4: DynaQAgent model population
# --------------------------------------------------
def test_dyna_q_model_population():
    from env.maze_env import MazeEnv
    from algorithms.dyna_q import DynaQAgent

    print("=== Test 4: DynaQAgent model population ===")
    env = MazeEnv(size=5, seed=42)
    agent = DynaQAgent(env, planning_steps=5)
    agent.train(50)

    assert len(agent.model) > 0, "Dyna-Q model should be populated after training"
    assert agent.q_table.shape == (25, 4), f"Expected (25, 4), got {agent.q_table.shape}"
    print(f"  model entries={len(agent.model)}, q_table shape={agent.q_table.shape}  OK")
    env.close()
    print("PASS: DynaQAgent model population\n")


# --------------------------------------------------
# Test 5: Dyna-Q sample efficiency vs Q-Learning
# --------------------------------------------------
def test_dyna_q_sample_efficiency():
    from env.maze_env import MazeEnv
    from algorithms.q_learning import QLearningAgent
    from algorithms.dyna_q import DynaQAgent

    print("=== Test 5: Dyna-Q sample efficiency ===")
    print("  (mat vai giay...)")

    # 5x5 has only 25 states -- the convergence-speed gap is within
    # per-run noise. 10x10 (100 states) shows Dyna-Q's planning
    # advantage consistently, so use multiple trials averaged.
    n_episodes = 300
    n_trials = 3
    target = 0.9

    def first_convergence(metrics, target):
        for i, rate in enumerate(metrics.rolling_success_rate):
            if i >= 99 and rate >= target:
                return i + 1
        return n_episodes  # did not converge within cap

    q_convs, dq_convs = [], []
    for _ in range(n_trials):
        env_q = MazeEnv(size=10, seed=42)
        q_agent = QLearningAgent(env_q)
        q_convs.append(first_convergence(q_agent.train(n_episodes), target))
        env_q.close()

        env_dq = MazeEnv(size=10, seed=42)
        dq_agent = DynaQAgent(env_dq, planning_steps=10)
        dq_convs.append(first_convergence(dq_agent.train(n_episodes), target))
        env_dq.close()

    q_mean = sum(q_convs) / n_trials
    dq_mean = sum(dq_convs) / n_trials

    print(f"  Q-Learning converge episodes (mean of {n_trials}): {q_mean:.1f} {q_convs}")
    print(f"  Dyna-Q     converge episodes (mean of {n_trials}): {dq_mean:.1f} {dq_convs}")

    assert dq_mean < q_mean, "Dyna-Q should converge faster (fewer episodes) than Q-Learning on average"
    print("PASS: Dyna-Q sample efficiency\n")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    test_pool_generation()
    test_preset_grid_env()
    test_load_maze_env()
    test_dyna_q_model_population()
    test_dyna_q_sample_efficiency()
    _cleanup()

    print("=" * 50)
    print("Phase 6: ALL TESTS PASS")
    print()
    print("NOTE: Live demo -> python demo_unseen_maze.py")
