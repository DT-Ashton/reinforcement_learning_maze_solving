import numpy as np
from abc import ABC, abstractmethod

from metrics.training_metrics import TrainingMetrics


class BaseAgent(ABC):
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.size = env.size
        self.n_states = self.size * self.size
        self.n_actions = 4
        self.q_table = np.zeros((self.n_states, 4), dtype=np.float64)

    def _obs_to_state(self, obs: np.ndarray) -> int:
        obs_int = obs.astype(np.int32)
        assert 2 in obs_int, "agent marker missing from observation"
        flat_idx = int(np.argmax(obs_int == 2))
        gr = flat_idx // (2 * self.size + 1)
        gc = flat_idx % (2 * self.size + 1)
        return (gr // 2) * self.size + (gc // 2)

    def get_action(self, state: int, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(self.q_table[state]))

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table(self) -> np.ndarray:
        return self.q_table

    @abstractmethod
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        pass

    @abstractmethod
    def train(self, n_episodes: int) -> TrainingMetrics:
        pass
