# Maze Solving using Reinforcement Learning

A reinforcement learning project focused on solving randomly generated mazes using multiple RL algorithms, including Q-Learning, SARSA, and Deep Q-Network (DQN).

The project aims to compare the performance, learning efficiency, and navigation behavior of different reinforcement learning approaches in dynamic maze environments.

---

## Project Objectives

This project aims to:

- Generate random mazes with adjustable complexity
- Design a custom reinforcement learning environment
- Implement and compare multiple RL algorithms:
  - Q-Learning
  - SARSA
  - Deep Q-Network (DQN)
  - PPO (optional extension)
- Visualize agent learning behavior and evaluation metrics
- Analyze algorithm performance under different maze configurations

---

## Features

### Maze System
- Random maze generation
- Adjustable maze size
- Configurable random seed
- Multiple difficulty levels

### Reinforcement Learning Algorithms
- Q-Learning
- SARSA
- Deep Q-Network (DQN)
- PPO (optional)

### Visualization
- Real-time maze rendering
- Agent movement animation
- Training visualization
- Replay mode
- Heatmaps and policy visualization

### Evaluation & Metrics
- Episode rewards
- Success rate
- Average steps to goal
- Training convergence
- Path optimality
- Algorithm comparison dashboard

---

## Tech Stack

### Programming Language
- Python 3.x

### Reinforcement Learning & Environment
- Gymnasium
- NumPy
- PyTorch
- Stable-Baselines3 (for DQN/PPO)

### Visualization
- Pygame
- Matplotlib
- TensorBoard

### Utilities
- Pandas
- Seaborn (optional)
- Jupyter Notebook

---

## Project Structure

```text
reinforcement_learning_maze_solving
│
├── env/                  # Maze environment
├── algorithms/           # RL algorithms
│   ├── q_learning/
│   ├── sarsa/
│   └── dqn/
│
├── maze/                 # Maze generation logic
├── visualization/        # Rendering and visualization
├── metrics/              # Evaluation metrics
├── experiments/          # Experiment scripts
├── configs/              # Configuration files
├── notebooks/            # Research notebooks
├── assets/               # Images/videos
└── main.py