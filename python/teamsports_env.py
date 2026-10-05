import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import teamsports_rl

OBS_DIM = 22

class TeamSportsEnv(gym.Env):
    """Gymnasium environment wrapper around the C++ mock simulation."""

    def __init__(self):
        super().__init__()
        self._env = teamsports_rl.TeamSportsEnv()

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Flat action: [move_x, move_y, kick]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.set_seed(seed)
        obs, info = self._env.reset()
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,):
            raise ValueError(f"Expected action shape (3,), got {action.shape}")
        action = action.copy()
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], -1.0, 1.0)
        action[2] = 1.0 if action[2] > 0.5 else 0.0

        obs, reward, terminated, truncated, info = self._env.step(action)
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def debug_text(self):
        return self._env.debug_text()
