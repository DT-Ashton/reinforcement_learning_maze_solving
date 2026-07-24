import numpy as np

from algorithms.q_learning import QLearningAgent


class DynaQAgent(QLearningAgent):
    ALGO_NAME = "dyna_q"

    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, planning_steps: int = 10):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.planning_steps = planning_steps
        self.model = {}  # (state, action) -> (reward, next_state, done)

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        super().update(s, a, r, s_next, done)
        self.model[(s, a)] = (r, s_next, done)

        if len(self.model) < self.planning_steps:
            return

        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            ps, pa = keys[np.random.randint(len(keys))]
            pr, ps_next, pdone = self.model[(ps, pa)]
            super().update(ps, pa, pr, ps_next, pdone)
