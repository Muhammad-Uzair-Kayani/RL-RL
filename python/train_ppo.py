import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import time

from teamsports_env import TeamSportsEnv

OBS_DIM = 18
ACT_CONTINUOUS_DIM = 2

def make_env():
    return TeamSportsEnv()

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.move_mean = nn.Linear(hidden, ACT_CONTINUOUS_DIM)
        self.move_logstd = nn.Parameter(torch.zeros(ACT_CONTINUOUS_DIM))
        self.kick_logit = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = self.shared(obs)
        move_mean = torch.tanh(self.move_mean(x))
        kick_prob = torch.sigmoid(self.kick_logit(x))
        value = self.value(x)
        return move_mean, self.move_logstd, kick_prob, value

def select_action(model, obs):
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        move_mean, move_logstd, kick_prob, value = model(obs_t)
    move_std = torch.exp(move_logstd)
    move_dist = Normal(move_mean, move_std)
    move_action = move_dist.sample()
    move_log_prob = move_dist.log_prob(move_action).sum(dim=-1)
    kick_dist = Categorical(torch.stack([1 - kick_prob, kick_prob], dim=-1).squeeze(0))
    kick_action = kick_dist.sample()
    kick_log_prob = kick_dist.log_prob(kick_action)
    log_prob = move_log_prob + kick_log_prob
    action_np = np.zeros(3, dtype=np.float32)
    action_np[0:2] = move_action.squeeze(0).numpy()
    action_np[2] = float(kick_action.item())
    return action_np, log_prob.squeeze(0).item(), value.squeeze(0).item()

def collect_rollout(env, model, steps_per_ep=256):
    obs_list, action_list, reward_list, logprob_list, value_list, done_list = [], [], [], [], [], []
    obs, _, done = env.reset()
    ep_reward = 0.0
    for _ in range(steps_per_ep):
        action, log_prob, value = select_action(model, obs)
        next_obs, reward, done = env.step(action)
        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        logprob_list.append(log_prob)
        value_list.append(value)
        done_list.append(float(done))
        obs = next_obs
        ep_reward += reward
        if done:
            obs, _, done = env.reset()
    return obs_list, action_list, reward_list, logprob_list, value_list, done_list

def compute_returns(rewards, values, dones, gamma=0.99):
    returns = []
    R = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        if d > 0.5:
            R = 0.0
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def train_ppo(epochs=1000, steps_per_ep=256, lr=3e-4, gamma=0.99, clip_eps=0.2, batches=4, updates_per_epoch=4):
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    env = make_env()

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        obs_list, action_list, rewards, logprob_list, values, dones = collect_rollout(env, model, steps_per_ep)

        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)
        old_log_probs = torch.tensor(logprob_list, dtype=torch.float32)
        returns = compute_returns(rewards, values, dones, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        total_loss = 0.0
        for _ in range(updates_per_epoch):
            indices = np.arange(len(obs_list))
            np.random.shuffle(indices)
            batch_size = len(indices) // batches
            for i in range(batches):
                batch_idx = indices[i * batch_size : (i + 1) * batch_size]
                if len(batch_idx) == 0:
                    continue
                b_obs = obs_t[batch_idx]
                b_returns = returns[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_actions = np.array([action_list[j] for j in batch_idx])
                b_moves = torch.tensor(b_actions[:, 0:2], dtype=torch.float32)
                b_kicks = torch.tensor(b_actions[:, 2], dtype=torch.long)

                move_mean, move_logstd, kick_prob, value = model(b_obs)
                move_std = torch.exp(move_logstd)
                move_dist = Normal(move_mean, move_std)
                move_log_prob = move_dist.log_prob(b_moves).sum(dim=-1)
                kick_dist = Categorical(torch.stack([1 - kick_prob.squeeze(-1), kick_prob.squeeze(-1)], dim=-1))
                kick_log_prob = kick_dist.log_prob(b_kicks)
                new_log_prob = move_log_prob + kick_log_prob

                ratio = torch.exp(new_log_prob - b_old_log_probs[batch_idx])
                advantage = b_returns - value.squeeze(-1)

                surr1 = ratio * advantage.detach()
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = advantage.pow(2).mean()
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()

        avg_return = returns.mean().item()
        print(f"Epoch {epoch + 1}/{epochs} | Avg return: {avg_return:.3f} | Last reward: {rewards[-1]:.3f}")

        if (epoch + 1) % 100 == 0:
            path = f"checkpoints/ppo_model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved checkpoint: {path}")

    torch.save(model.state_dict(), "checkpoints/ppo_model_final.pt")

if __name__ == "__main__":
    train_ppo()
