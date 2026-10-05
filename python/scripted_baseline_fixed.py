import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from teamsports_env import TeamSportsEnv

OBS_NAMES = [
    "self_pos_x", "self_pos_y",
    "self_vel_x", "self_vel_y",
    "ball_rel_x", "ball_rel_y",
    "ball_vel_x", "ball_vel_y",
    "team_rel_x", "team_rel_y",
    "team_vel_x", "team_vel_y",
    "opp_rel_x", "opp_rel_y",
    "opp_vel_x", "opp_vel_y",
    "own_goal_rel_x", "own_goal_rel_y",
    "enemy_goal_rel_x", "enemy_goal_rel_y",
]

def scripted_action(obs):
    ball_rx, ball_ry = obs[4], obs[5]
    team_rx, team_ry = obs[8], obs[9]
    enemy_rx, enemy_ry = obs[18], obs[19]

    team_has_ball = np.hypot(team_rx - ball_rx, team_ry - ball_ry) < 0.05
    self_has_ball = np.hypot(ball_rx, ball_ry) < 0.05

    action = np.zeros(3, dtype=np.float32)

    if team_has_ball:
        target_x = team_rx + enemy_rx * 0.3
        target_y = team_ry + 0.15
        if target_x < 0.05:
            target_x = 0.05
    elif self_has_ball:
        target_x = enemy_rx
        target_y = enemy_ry
    else:
        target_x = ball_rx
        target_y = ball_ry

    dist = np.hypot(target_x, target_y) + 1e-8
    action[0] = np.clip(target_x / dist, -1.0, 1.0)
    action[1] = np.clip(target_y / dist, -1.0, 1.0)

    if self_has_ball and target_x > 0.0:
        action[2] = 1.0
    else:
        action[2] = 0.0

    return action

def run_baseline(num_episodes=5, max_steps=1000, render=False):
    env = TeamSportsEnv()
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            action = scripted_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if render and steps % 30 == 0:
                print(env.debug_text())

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep + 1}: reward={total_reward:.3f}, length={steps}")

    print(f"Mean reward: {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")
    return episode_rewards, episode_lengths

if __name__ == "__main__":
    run_baseline(num_episodes=5, render=True)
