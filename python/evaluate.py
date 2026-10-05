import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from teamsports_env import TeamSportsEnv
from scripted_baseline import scripted_action

def random_action():
    return np.array([
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(-1.0, 1.0),
        1.0 if np.random.random() > 0.9 else 0.0
    ], dtype=np.float32)

def run_agent(agent_fn, num_episodes=10, max_steps=1000):
    env = TeamSportsEnv()
    rewards = []
    lengths = []
    wins = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total = 0.0
        steps = 0
        last_info = {}
        while not (terminated or truncated) and steps < max_steps:
            action = agent_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            last_info = info
        rewards.append(total)
        lengths.append(steps)
        a_score = last_info.get("team_a_score", 0)
        b_score = last_info.get("team_b_score", 0)
        if a_score > b_score:
            wins += 1
    return rewards, lengths, wins

def main():
    print("Evaluating random agent...")
    rr, rl, rw = run_agent(random_action, num_episodes=10)
    print(f"Random  mean reward={np.mean(rr):.3f} std={np.std(rr):.3f} mean length={np.mean(rl):.1f} wins={rw}")

    print("\nEvaluating scripted baseline...")
    sr, sl, sw = run_agent(scripted_action, num_episodes=10)
    print(f"Scripted mean reward={np.mean(sr):.3f} std={np.std(sr):.3f} mean length={np.mean(sl):.1f} wins={sw}")

if __name__ == "__main__":
    main()
