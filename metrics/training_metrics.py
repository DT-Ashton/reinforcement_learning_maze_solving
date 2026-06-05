import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    episode_rewards: list = field(default_factory=list)
    episode_lengths: list = field(default_factory=list)
    success_flags: list = field(default_factory=list)
    rolling_success_rate: list = field(default_factory=list)
    algo_name: str = ""
    maze_size: int = 10
    n_episodes: int = 0
    q_table: object = None

    def final_success_rate(self) -> float:
        return self.rolling_success_rate[-1] if self.rolling_success_rate else 0.0

    def mean_reward_last_100(self) -> float:
        return float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
