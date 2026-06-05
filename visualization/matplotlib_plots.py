import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

from metrics.training_metrics import TrainingMetrics


def reward_curve(metrics: TrainingMetrics, ax=None, label: str = None,
                 color: str = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    rewards = np.array(metrics.episode_rewards)
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color=color, label=None)
    if len(rewards) >= 100:
        kernel = np.ones(100) / 100
        rolling = np.convolve(rewards, kernel, mode="valid")
        ax.plot(np.arange(99, len(rewards)), rolling, color=color,
                label=label or metrics.algo_name)
    else:
        ax.plot(episodes, rewards, color=color, label=label or metrics.algo_name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"Reward Curve — {metrics.algo_name}")
    return ax


def success_rate_curve(metrics: TrainingMetrics, ax=None, label: str = None,
                       color: str = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    episodes = np.arange(len(metrics.rolling_success_rate))
    ax.plot(episodes, metrics.rolling_success_rate, color=color,
            label=label or metrics.algo_name)
    ax.axhline(0.80, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Success Rate — {metrics.algo_name}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    return ax


def q_value_heatmap(q_table: np.ndarray, size: int, action: int,
                    ax=None, title: str = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    values = q_table[:, action].reshape(size, size)
    im = ax.imshow(values, cmap="RdYlGn", origin="upper")
    plt.colorbar(im, ax=ax)
    ax.set_title(title or f"Q-values: {action_names[action]}")
    return ax


def comparison_plot(metrics_list: list, save_path: str = None) -> None:
    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        print("No metrics to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for m in metrics_list:
        reward_curve(m, ax=axs[0, 0], label=m.algo_name)
    axs[0, 0].set_title("Reward Curves")
    axs[0, 0].legend()

    for m in metrics_list:
        success_rate_curve(m, ax=axs[0, 1], label=m.algo_name)
    axs[0, 1].set_title("Success Rates")
    axs[0, 1].legend()

    tabular_metrics = next((m for m in metrics_list if m.q_table is not None), None)
    for idx, action in enumerate([0, 1]):
        ax = axs[1, idx]
        if tabular_metrics is not None:
            q_value_heatmap(tabular_metrics.q_table, tabular_metrics.maze_size,
                            action=action, ax=ax)
        else:
            ax.text(0.5, 0.5, "Q-table not available",
                    ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_figure(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")


