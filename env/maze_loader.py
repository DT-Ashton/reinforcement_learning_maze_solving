import json

import numpy as np

from env.maze_env import MazeEnv


def load_maze_env(path: str, render_mode: str = None) -> MazeEnv:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grid = np.array(data["grid"], dtype=np.int32)
    return MazeEnv(size=data["size"], render_mode=render_mode, grid=grid)
