import numpy as np
import teamsports_rl

class TeamSportsEnv:
    """Gymnasium-like wrapper around the C++ mock simulation."""

    def __init__(self):
        self._env = teamsports_rl.TeamSportsEnv()
        self.observation_space = None  # placeholder
        self.action_space = None       # placeholder

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        obs, reward, done = self._env.reset()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, reward, done = self._env.step(action)
        return obs, reward, done, False, {}

    def debug_text(self):
        return self._env.debug_text()
