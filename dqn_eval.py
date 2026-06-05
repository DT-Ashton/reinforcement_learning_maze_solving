import sys, os
sys.path.insert(0, ".")
from env.maze_env import MazeEnv
from algorithms.dqn_trainer import DQNTrainer

env = MazeEnv(size=10, seed=42)
trainer = DQNTrainer(env)
trainer.load("./models/dqn_maze")

successes = 0
total = 20
for ep in range(total):
    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = 0
    while not (terminated or truncated):
        action = trainer.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
    if terminated:
        successes += 1
    print(f"ep {ep+1:2d}: {'OK' if terminated else 'FAIL'} {steps} steps")

print(f"Success: {successes}/{total} = {successes*5}%")
env.close()
