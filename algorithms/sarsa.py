import numpy as np

from algorithms.base_agent import BaseAgent
from metrics.training_metrics import TrainingMetrics


class SARSAAgent(BaseAgent):
    ALGO_NAME = "sarsa"

    def update(self, s: int, a: int, r: float, s_next: int, done: bool,
               a_next: int = 0) -> None:
        self.q_table[s, a] += self.alpha * (
            r + self.gamma * self.q_table[s_next, a_next] * (1 - int(done))
            - self.q_table[s, a]
        )

    def train(self, n_episodes: int = 3000) -> TrainingMetrics:
        metrics = TrainingMetrics(
            algo_name=self.ALGO_NAME,
            maze_size=self.size,
            n_episodes=n_episodes,
        )

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            state = self._obs_to_state(obs)
            action = self.get_action(state, explore=True)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                obs_next, reward, terminated, truncated, _ = self.env.step(action)
                s_next = self._obs_to_state(obs_next)
                a_next = self.get_action(s_next, explore=True)
                self.update(state, action, reward, s_next, terminated, a_next)
                state = s_next
                action = a_next
                total_reward += reward
                steps += 1

            self._decay_epsilon()
            metrics.episode_rewards.append(total_reward)
            metrics.episode_lengths.append(steps)
            metrics.success_flags.append(bool(terminated))
            window = metrics.success_flags[max(0, ep - 99): ep + 1]
            metrics.rolling_success_rate.append(sum(window) / len(window))

        metrics.q_table = self.q_table
        return metrics
