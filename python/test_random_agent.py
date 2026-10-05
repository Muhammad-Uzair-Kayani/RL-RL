import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random

from teamsports_env import TeamSportsEnv

def random_action():
    return np.array([
        random.uniform(-1.0, 1.0),
        random.uniform(-1.0, 1.0),
        1.0 if random.random() > 0.9 else 0.0
    ], dtype=np.float32)

def main():
    env = TeamSportsEnv()
    obs, _info = env.reset()
    print("Initial observation shape:", obs.shape)
    print(env.debug_text())

    total_reward = 0.0
    steps = 0
    max_steps = 1000
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < max_steps:
        action = random_action()
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 20 == 0 or terminated or truncated:
            print(f"--- Step {steps} ---")
            print(env.debug_text(), end="")
            print(f"Action: move_x={action[0]:.2f}, move_y={action[1]:.2f}, kick={action[2]:.0f}")
            print(f"Reward: {reward:.3f}, Total: {total_reward:.3f}, Done: {terminated or truncated}\n")

    print(f"Episode finished after {steps} steps. Total reward: {total_reward:.3f}")

if __name__ == "__main__":
    main()
