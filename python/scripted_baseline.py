import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teamsports_env import TeamSportsEnv

def scripted_action(obs):
    # obs indices:
    # 0,1: self pos
    # 2,3: self vel
    # 4,5: ball relative pos
    # 6,7: ball velocity
    # 8,9: teammate relative pos
    # 10,11: teammate velocity
    # 12,13: opponent relative pos
    # 14,15: opponent velocity
    # 16,17: own goal relative pos
    # 18,19: enemy goal relative pos
    # 18,19 removed; current obs has 18 dims
    # 16,17: own goal relative

    ball_rel_x = obs[4]
    ball_rel_y = obs[5]
    team_rel_x = obs[8]
    team_rel_y = obs[9]
    own_goal_rel_x = obs[16]
    own_goal_rel_y = obs[17]
    # If close to ball, move toward ball
    ball_dist = np.sqrt(ball_rel_x**2 + ball_rel_y**2)
    action = np.zeros(3, dtype=np.float32)

    # Teammate has ball -> support toward enemy goal with lateral offset
    if np.linalg.norm(team_rel_x + own_goal_rel_x * 0.0) > -1.0:
        target_x = ball_rel_x
        target_y = ball_rel_y - 0.15
        if team_rel_x > -own_goal_rel_x * 0.3:
            target_x += 0.1
        norm = np.sqrt(target_x**2 + target_y**2) + 1e-8
        action[0] = np.clip(target_x / norm, -1, 1)
        action[1] = np.clip(target_y / norm, -1, 1)

    # If ball is close, try to approach/kick
    if ball_dist < 0.1:
        action[0] = np.clip(ball_rel_x / (ball_dist + 1e-8), -1, 1)
        action[1] = np.clip(ball_rel_y / (ball_dist + 1e-8), -1, 1)
        action[2] = 1.0  # kick

    return action

def main():
    env = TeamSportsEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 1000
    done = False
    while not done and steps < max_steps:
        action = scripted_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if steps % 30 == 0 or done:
            print(f"--- Step {steps} ---")
            print(env.debug_text(), end="")
            print(f"Action: move_x={action[0]:.2f}, move_y={action[1]:.2f}, kick={action[2]:.0f}")
            print(f"Reward: {reward:.3f}, Total: {total_reward:.3f}, Done: {done}\n")
    print(f"Episode finished after {steps} steps. Total reward: {total_reward:.3f}")

if __name__ == "__main__":
    main()
