import os
import argparse
import pickle

import numpy as np
import yaml

from env.maze_env import MazeEnv
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.dqn_trainer import DQNTrainer
from visualization.matplotlib_plots import comparison_plot


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(size: int, seed: int, render_mode: str = None) -> MazeEnv:
    import env  # noqa: F401 — triggers gymnasium.register
    return MazeEnv(size=size, seed=seed, render_mode=render_mode)


def run_train(args, cfg: dict) -> None:
    os.makedirs("./models/", exist_ok=True)
    environment = build_env(cfg["maze"]["size"], cfg["maze"]["seed"])

    if args.algo == "qlearning":
        agent_cfg = {k: v for k, v in cfg["qlearning"].items() if k != "n_episodes"}
        agent = QLearningAgent(environment, **agent_cfg)
        metrics = agent.train(cfg["qlearning"]["n_episodes"])
        metrics.q_table = agent.get_q_table()
        np.save("./models/qlearning_qtable.npy", agent.get_q_table())
        print(f"[qlearning] Final success rate: {metrics.final_success_rate():.2%}")
        with open("./models/qlearning_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

    elif args.algo == "sarsa":
        agent_cfg = {k: v for k, v in cfg["sarsa"].items() if k != "n_episodes"}
        agent = SARSAAgent(environment, **agent_cfg)
        metrics = agent.train(cfg["sarsa"]["n_episodes"])
        metrics.q_table = agent.get_q_table()
        np.save("./models/sarsa_qtable.npy", agent.get_q_table())
        print(f"[sarsa] Final success rate: {metrics.final_success_rate():.2%}")
        with open("./models/sarsa_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

    elif args.algo == "dqn":
        tb_log = cfg["dqn"].get("tensorboard_log") or None
        trainer = DQNTrainer(
            environment,
            total_timesteps=cfg["dqn"]["total_timesteps"],
            tensorboard_log=tb_log,
        )
        trainer.build_model(**{k: cfg["dqn"][k] for k in [
            "learning_rate", "buffer_size", "learning_starts",
            "batch_size", "gamma", "exploration_fraction", "exploration_final_eps",
        ]})
        dqn_metrics = trainer.train()
        with open("./models/dqn_metrics.pkl", "wb") as f:
            pickle.dump(dqn_metrics, f)


def run_demo(args, cfg: dict) -> None:
    import pygame
    environment = build_env(
        cfg["maze"]["size"], cfg["maze"]["seed"], render_mode="human"
    )
    environment.metadata["render_fps"] = cfg["visualization"]["fps"]

    if args.algo in ("qlearning", "sarsa"):
        qtable_path = f"./models/{args.algo}_qtable.npy"
        if not os.path.exists(qtable_path):
            print(f"No saved Q-table at {qtable_path}. Run --train first.")
            return
        q_table = np.load(qtable_path)
        AgentClass = QLearningAgent if args.algo == "qlearning" else SARSAAgent
        agent = AgentClass(environment)
        agent.q_table = q_table
        agent.epsilon = 0.0

        obs, _ = environment.reset()
        state = agent._obs_to_state(obs)
        terminated, truncated = False, False
        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    environment.close()
                    return
            action = agent.get_action(state, explore=False)
            obs, _, terminated, truncated, _ = environment.step(action)
            state = agent._obs_to_state(obs)
        print(f"Steps: {environment._step_count}, "
              f"{'Success' if terminated else 'Timeout'}")
        environment.close()

    elif args.algo == "dqn":
        model_path = cfg["dqn"]["model_save_path"]
        if not os.path.exists(model_path + ".zip"):
            print(f"No saved DQN model at {model_path}.zip. Run --train first.")
            return
        trainer = DQNTrainer(environment)
        trainer.load(model_path)

        obs, _ = environment.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    environment.close()
                    return
            action = trainer.predict(obs)
            obs, _, terminated, truncated, _ = environment.step(action)
        print(f"Steps: {environment._step_count}, "
              f"{'Success' if terminated else 'Timeout'}")
        environment.close()


def run_plot(args, cfg: dict) -> None:
    os.makedirs("./reports/", exist_ok=True)
    metrics_list = []
    for algo in ["qlearning", "sarsa", "dqn"]:
        path = f"./models/{algo}_metrics.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                metrics_list.append(pickle.load(f))
    if not metrics_list:
        print("No metrics files found. Run --train first.")
        return
    comparison_plot(metrics_list, save_path="./reports/comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reinforcement Learning Maze Solver")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--algo", choices=["qlearning", "sarsa", "dqn"],
                        default="qlearning")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    if not (args.train or args.demo or args.plot):
        parser.error("specify at least one of --train, --demo, --plot")

    cfg = load_config(args.config)
    cfg["maze"]["size"] = args.size
    cfg["maze"]["seed"] = args.seed

    if args.train:
        run_train(args, cfg)
    if args.demo:
        run_demo(args, cfg)
    if args.plot:
        run_plot(args, cfg)


if __name__ == "__main__":
    main()
