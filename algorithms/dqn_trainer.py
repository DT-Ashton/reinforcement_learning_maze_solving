import os
import numpy as np
import gymnasium
from stable_baselines3.common.callbacks import BaseCallback

from metrics.training_metrics import TrainingMetrics


def _default_tb_log() -> str:
    # TF's gfile C++ backend cannot handle non-ASCII paths (e.g. 'Kỳ 8 - AI').
    # Use USERPROFILE-relative path which is always ASCII on standard Windows installs.
    home = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    return os.path.join(home, "maze_rl_logs", "dqn")


class _PosObsWrapper(gymnasium.Wrapper):
    """
    Adapts MazeEnv for DQN training in two ways:

    1. Observation: converts the (2N+1)^2 flat grid to 4 normalized floats
       [agent_row/N, agent_col/N, 1.0, 1.0] (goal is always at corner (N-1,N-1)).
       A 441-dim sparse vector is too hard for MLP-DQN; 4-dim position converges.

    2. Reward shaping: adds proximity bonus in [0, 1] so DQN gets a gradient
       toward the goal even before it finds it for the first time.
       bonus = 1 - manhattan_distance(agent, goal) / max_possible_distance
       (potential-based shaping, does not change the optimal policy)

    3. Extended max_steps: 10× the tabular budget so random exploration
       can find the goal at least once (random walk on 10×10 maze needs
       ~1000+ steps on average to reach the corner).
    """
    def __init__(self, env):
        super().__init__(env)
        n = env.size
        # 8-dim obs: [agent_row/N, agent_col/N, can_N, can_S, can_W, can_E,
        #             rows_to_goal/N, cols_to_goal/N]
        # Wall flags prevent cycles; goal direction helps DQN prefer goal-approaching moves.
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self._n = n
        self._norm = float(max(n - 1, 1))
        self._max_dist = 2.0 * (n - 1)
        # O(N^4) random walk hitting time on a perfect maze: need N*N*100 budget
        self.dqn_max_steps = n * n * 100  # 10000 for 10×10, 2500 for 5×5
        env.max_steps = self.dqn_max_steps  # extend MazeEnv's own limit to match
        self._ep_step = 0

    def _pos(self, obs: np.ndarray) -> np.ndarray:
        size = self._n
        g = obs.astype(np.int32).reshape(2 * size + 1, 2 * size + 1)
        positions = np.argwhere(g == 2)
        gr, gc = int(positions[0, 0]), int(positions[0, 1])
        ar, ac = gr // 2, gc // 2
        # Local passability flags (1=open, 0=wall/boundary)
        can_n = float(ar > 0       and g[2*ar,     2*ac+1] == 1)
        can_s = float(ar < size-1  and g[2*ar+2,   2*ac+1] == 1)
        can_w = float(ac > 0       and g[2*ar+1,   2*ac  ] == 1)
        can_e = float(ac < size-1  and g[2*ar+1,   2*ac+2] == 1)
        # Normalized distance to goal (goal is always at (N-1, N-1))
        dr = (size - 1 - ar) / self._norm
        dc = (size - 1 - ac) / self._norm
        return np.array(
            [ar / self._norm, ac / self._norm, can_n, can_s, can_w, can_e, dr, dc],
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self._ep_step = 0
        obs, info = self.env.reset(**kwargs)
        return self._pos(obs), info

    def step(self, action):
        obs, reward, terminated, _env_trunc, info = self.env.step(action)
        self._ep_step += 1
        # Use our budget — ignore env's truncated (hits at 400 steps, too short)
        truncated = (self._ep_step >= self.dqn_max_steps) and not terminated

        pos_obs = self._pos(obs)
        if not terminated:
            ar = pos_obs[0] * (self._n - 1)
            ac = pos_obs[1] * (self._n - 1)
            dist = abs(ar - (self._n - 1)) + abs(ac - (self._n - 1))
            reward += 1.0 - dist / self._max_dist   # shaping bonus in [0, 1]

        return pos_obs, reward, terminated, truncated, info


class DQNTrainer:
    def __init__(self, env, total_timesteps: int = 100_000,
                 tensorboard_log: str = None):
        self.env = env
        self.total_timesteps = total_timesteps
        self.tensorboard_log = (
            tensorboard_log if tensorboard_log is not None else _default_tb_log()
        )
        self.model_save_path = os.path.abspath("./models/dqn_maze")
        self.model = None
        self._wrapped = None
        os.makedirs(self.tensorboard_log, exist_ok=True)

    def build_model(self, learning_rate: float = 1e-3, buffer_size: int = 10_000,
                    learning_starts: int = 1_000, batch_size: int = 64,
                    gamma: float = 0.99, exploration_fraction: float = 0.3,
                    exploration_final_eps: float = 0.05) -> None:
        from stable_baselines3 import DQN
        self._wrapped = _PosObsWrapper(self.env)
        self.model = DQN(
            policy="MlpPolicy",
            env=self._wrapped,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            tensorboard_log=self.tensorboard_log,
            verbose=1,
        )

    class _MetricsCallback(BaseCallback):
        def __init__(self, max_steps: int):
            super().__init__()
            self.max_steps = max_steps
            self.episode_rewards = []
            self.episode_lengths = []
            self.success_flags = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    self.success_flags.append(
                        info["episode"]["l"] < self.max_steps
                    )
            return True

    def train(self) -> TrainingMetrics:
        if self.model is None:
            raise RuntimeError("Call build_model() before train()")

        dqn_max = self._wrapped.dqn_max_steps if self._wrapped else self.env.max_steps
        cb = self._MetricsCallback(max_steps=dqn_max)
        self.model.learn(
            total_timesteps=self.total_timesteps,
            tb_log_name="DQN_MazeEnv",
            reset_num_timesteps=True,
            progress_bar=True,
            callback=cb,
        )
        self.save()

        metrics = TrainingMetrics(
            algo_name="dqn",
            maze_size=self.env.size,
            episode_rewards=cb.episode_rewards,
            episode_lengths=cb.episode_lengths,
            success_flags=cb.success_flags,
        )
        for i in range(len(metrics.success_flags)):
            window = metrics.success_flags[max(0, i - 99): i + 1]
            metrics.rolling_success_rate.append(sum(window) / len(window))
        metrics.n_episodes = len(metrics.success_flags)
        return metrics

    def save(self, path: str = None) -> None:
        if path is None:
            path = self.model_save_path
        os.makedirs("models", exist_ok=True)
        self.model.save(path)
        try:
            print(f"Model saved to {path}.zip")
        except UnicodeEncodeError:
            print("Model saved.")

    def load(self, path: str = None) -> None:
        from stable_baselines3 import DQN
        if path is None:
            path = self.model_save_path
        self._wrapped = _PosObsWrapper(self.env)
        self.model = DQN.load(path, env=self._wrapped)

    def predict(self, obs: np.ndarray) -> int:
        # Convert raw grid obs to 4-dim position obs before predicting
        wrapper = _PosObsWrapper(self.env)
        pos_obs = wrapper._pos(obs)
        return int(self.model.predict(pos_obs, deterministic=True)[0])
