# -*- coding: utf-8 -*-
"""
Phase 3 Test -- DQN via Stable-Baselines3
=========================================
Tests:
  1. DQNTrainer build_model() -- model initialized correctly
  2. Short training (10k steps) -- no crash, returns TrainingMetrics
  3. Save & load model -- predict after load
  4. Metrics shape -- episode_rewards, success_flags populated

Run:
    python test_function/phase3_dqn.py

Note: Full training -> python main.py --train --algo dqn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# Test models go here -- never overwrites production ./models/dqn_maze.zip
_TEST_MODEL_PATH = os.path.abspath("./models/_test_dqn_tmp")


def _make_trainer(env, total_timesteps, **build_kwargs):
    """Create a DQNTrainer with a temp save path so tests never clobber dqn_maze.zip."""
    from algorithms.dqn_trainer import DQNTrainer
    trainer = DQNTrainer(env, total_timesteps=total_timesteps)
    trainer.model_save_path = _TEST_MODEL_PATH
    trainer.build_model(**build_kwargs)
    return trainer


def _cleanup():
    for ext in [".zip"]:
        p = _TEST_MODEL_PATH + ext
        try:
            os.remove(p)
        except OSError:
            pass


# --------------------------------------------------
# Test 1: build_model
# --------------------------------------------------
def test_build_model():
    from env.maze_env import MazeEnv

    print("\n=== Test 1: DQNTrainer build_model ===")
    env = MazeEnv(size=5, seed=42)
    trainer = _make_trainer(env, total_timesteps=1000,
                            learning_rate=1e-3, buffer_size=1000,
                            learning_starts=100, batch_size=32)

    assert trainer.model is not None, "model should not be None after build_model()"
    print("  model built successfully            OK")
    env.close()
    print("PASS: build_model\n")


# --------------------------------------------------
# Test 2: Short training (10k steps)
# --------------------------------------------------
def test_short_training():
    from env.maze_env import MazeEnv
    from metrics.training_metrics import TrainingMetrics

    print("=== Test 2: Short training (10k steps, 5x5 maze) ===")
    print("  (mat ~30-60 giay...)")

    env = MazeEnv(size=5, seed=42)
    trainer = _make_trainer(env, total_timesteps=10_000,
                            learning_rate=1e-3, buffer_size=2000,
                            learning_starts=500, batch_size=32,
                            exploration_fraction=0.5)

    metrics = trainer.train()

    assert isinstance(metrics, TrainingMetrics), "train() should return TrainingMetrics"
    assert metrics.algo_name == "dqn"
    assert metrics.n_episodes > 0, "Should have completed at least 1 episode"
    assert len(metrics.episode_rewards) == metrics.n_episodes
    assert len(metrics.rolling_success_rate) == metrics.n_episodes

    print(f"  episodes={metrics.n_episodes}, "
          f"final_success_rate={metrics.final_success_rate():.1%}  OK")
    env.close()
    print("PASS: Short training\n")
    return trainer


# --------------------------------------------------
# Test 3: Save & load + predict
# --------------------------------------------------
def test_save_load_predict(trainer=None):
    from env.maze_env import MazeEnv
    from algorithms.dqn_trainer import DQNTrainer

    print("=== Test 3: Save, load, predict ===")

    env = MazeEnv(size=5, seed=42)

    if trainer is None:
        trainer = _make_trainer(env, total_timesteps=5_000,
                                learning_starts=200, buffer_size=1000)
        trainer.train()

    trainer.save(_TEST_MODEL_PATH)
    assert os.path.exists(_TEST_MODEL_PATH + ".zip"), "Model zip not found after save()"
    print(f"  saved to {_TEST_MODEL_PATH}.zip    OK")

    env2 = MazeEnv(size=5, seed=42)
    trainer2 = DQNTrainer(env2)
    trainer2.load(_TEST_MODEL_PATH)
    assert trainer2.model is not None

    obs, _ = env2.reset()
    action = trainer2.predict(obs)
    assert action in [0, 1, 2, 3], f"Invalid action: {action}"
    print(f"  loaded model predict action={action}  OK")

    env.close()
    env2.close()
    _cleanup()
    print("PASS: Save/load/predict\n")


# --------------------------------------------------
# Test 4: TrainingMetrics populated correctly
# --------------------------------------------------
def test_metrics_structure():
    from env.maze_env import MazeEnv

    print("=== Test 4: Metrics structure ===")
    env = MazeEnv(size=5, seed=42)
    trainer = _make_trainer(env, total_timesteps=8_000,
                            learning_starts=200, buffer_size=1000,
                            exploration_fraction=0.5)
    metrics = trainer.train()

    assert all(isinstance(r, float) for r in metrics.episode_rewards), \
        "episode_rewards should be floats"
    assert all(isinstance(s, bool) for s in metrics.success_flags), \
        "success_flags should be bools"
    assert len(metrics.rolling_success_rate) == len(metrics.success_flags), \
        "rolling_success_rate length mismatch"
    assert metrics.maze_size == 5

    print(f"  n_episodes={metrics.n_episodes}")
    print(f"  mean_reward_last_100={metrics.mean_reward_last_100():.1f}")
    print(f"  final_success_rate={metrics.final_success_rate():.1%}")
    print("  metrics structure valid              OK")

    env.close()
    _cleanup()
    print("PASS: Metrics structure\n")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    test_build_model()
    trainer = test_short_training()
    test_save_load_predict(trainer)
    test_metrics_structure()

    print("=" * 50)
    print("Phase 3: ALL TESTS PASS")
    print()
    print("NOTE: Full training -> python main.py --train --algo dqn")
