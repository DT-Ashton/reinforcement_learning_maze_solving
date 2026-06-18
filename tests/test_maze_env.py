# -*- coding: utf-8 -*-
"""
Phase 1 Test -- Maze Generator + Gymnasium Environment
=====================================================
Tests:
  1. MazeGenerator: random maze, adjustable size, configurable seed
  2. MazeEnv: reset/step API (Gymnasium 5-tuple), observation shape
  3. Pygame rendering (render_mode="human") -- mo cua so 3 giay roi tu dong
  4. Gymnasium check_env validation

Run:
    python tests/test_maze_env.py
    python tests/test_maze_env.py --no-render   # headless / no pygame
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


# --------------------------------------------------
# Test 1: MazeGenerator
# --------------------------------------------------
def test_maze_generator():
    from maze.generator import MazeGenerator

    print("\n=== Test 1: MazeGenerator ===")

    # Adjustable size
    for size in [5, 10, 15]:
        gen = MazeGenerator(size=size, seed=42)
        grid = gen.generate()
        expected_shape = (2 * size + 1, 2 * size + 1)
        assert grid.shape == expected_shape, f"Wrong shape: {grid.shape}"
        print(f"  size={size} -> grid shape {grid.shape}  OK")

    # Configurable seed -- same seed = same maze
    gen_a = MazeGenerator(size=10, seed=7)
    gen_b = MazeGenerator(size=10, seed=7)
    assert np.array_equal(gen_a.generate(), gen_b.generate()), "Same seed -> different maze!"
    print("  seed=7 reproducible                OK")

    # Different seed = different maze
    gen_c = MazeGenerator(size=10, seed=99)
    assert not np.array_equal(gen_a.grid, gen_c.generate()), "Different seeds -> same maze?"
    print("  seed=7 vs seed=99 differ           OK")

    # Perfect maze: all cells reachable (no isolated cell)
    gen = MazeGenerator(size=10, seed=42)
    grid = gen.generate()
    cell_count = sum(
        grid[2 * r + 1, 2 * c + 1] == 1
        for r in range(10) for c in range(10)
    )
    assert cell_count == 100, f"Not all cells carved: {cell_count}/100"
    print(f"  All {cell_count}/100 cells reachable       OK")

    print("PASS: MazeGenerator\n")


# --------------------------------------------------
# Test 2: MazeEnv headless (no render)
# --------------------------------------------------
def test_maze_env_headless():
    from env.maze_env import MazeEnv

    print("=== Test 2: MazeEnv headless ===")

    env = MazeEnv(size=10, seed=42, render_mode=None)

    # reset returns (obs, info)
    obs, info = env.reset()
    obs_size = (2 * 10 + 1) ** 2
    assert obs.shape == (obs_size,), f"Wrong obs shape: {obs.shape}"
    assert obs.dtype == np.float32, f"Wrong dtype: {obs.dtype}"
    print(f"  reset() obs shape={obs.shape} dtype={obs.dtype}  OK")

    # step returns 5-tuple
    result = env.step(1)
    assert len(result) == 5, f"step() should return 5 values, got {len(result)}"
    obs2, reward, terminated, truncated, info2 = result
    assert obs2.shape == (obs_size,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    print(f"  step() -> reward={reward}, terminated={terminated}, truncated={truncated}  OK")

    # Adjustable size
    for size in [5, 8, 12]:
        e = MazeEnv(size=size, seed=0)
        o, _ = e.reset()
        assert o.shape == ((2 * size + 1) ** 2,)
        print(f"  size={size} obs shape={o.shape}  OK")

    # Goal reached gives +100
    # Tim o ke hop le voi goal thay vi hardcode -- maze walls phu thuoc seed
    env2 = MazeEnv(size=5, seed=42, render_mode=None)
    env2.reset()
    goal_r, goal_c = env2._goal_cell  # (4, 4)
    found = False
    for action, (dr, dc) in [(0, (-1, 0)), (1, (1, 0)), (2, (0, -1)), (3, (0, 1))]:
        pr, pc = goal_r - dr, goal_c - dc
        if (0 <= pr < 5 and 0 <= pc < 5 and
                env2._base_grid[2 * pr + 1 + dr, 2 * pc + 1 + dc] == 1):
            env2._agent_cell = (pr, pc)
            _, reward_goal, terminated_goal, _, _ = env2.step(action)
            found = True
            break
    assert found, "Khong tim duoc duong vao goal"
    assert terminated_goal, "Should be terminated at goal"
    assert reward_goal == 100.0, f"Goal reward should be 100, got {reward_goal}"
    print("  goal reward=100, terminated=True            OK")

    env.close()
    print("PASS: MazeEnv headless\n")


# --------------------------------------------------
# Test 3: Pygame rendering (optional)
# --------------------------------------------------
def test_maze_env_render():
    import time
    from env.maze_env import MazeEnv

    print("=== Test 3: Pygame rendering (3 giay) ===")
    env = MazeEnv(size=10, seed=42, render_mode="human")
    obs, _ = env.reset()

    start = time.time()
    steps = 0
    import pygame
    actions = [1, 1, 1, 3, 3, 3, 0, 0, 2, 2]
    terminated = truncated = False
    while time.time() - start < 3.0 and not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        action = actions[steps % len(actions)]
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1

    env.close()
    print(f"  Rendered {steps} steps in 3s               OK")
    print("PASS: Pygame rendering\n")


# --------------------------------------------------
# Test 4: Gymnasium check_env
# --------------------------------------------------
def test_gymnasium_check():
    from env.maze_env import MazeEnv
    from gymnasium.utils.env_checker import check_env

    print("=== Test 4: Gymnasium check_env ===")
    env = MazeEnv(size=5, seed=42, render_mode=None)
    check_env(env, warn=True, skip_render_check=True)
    env.close()
    print("  check_env passed (no exceptions)    OK")
    print("PASS: Gymnasium check_env\n")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true",
                        help="Skip pygame test (headless)")
    args = parser.parse_args()

    test_maze_generator()
    test_maze_env_headless()

    if not args.no_render:
        test_maze_env_render()
    else:
        print("=== Test 3: Pygame rendering -- SKIP (--no-render) ===\n")

    test_gymnasium_check()

    print("=" * 50)
    print("Phase 1: ALL TESTS PASS")
