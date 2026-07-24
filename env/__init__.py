from env.maze_env import MazeEnv
import gymnasium

gymnasium.register(
    id="MazeEnv-v0",
    entry_point="env.maze_env:MazeEnv",
    kwargs={"size": 10, "seed": 42, "render_mode": None},
)

__all__ = ["MazeEnv"]
