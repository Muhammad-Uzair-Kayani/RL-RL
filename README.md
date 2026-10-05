# 2D Team Sports RL Prototype — Stage 1

This workspace contains the first milestone of the C++ ? Python reinforcement-learning prototype for a 2D team sports game.

## What's implemented

- `cpp/rl_env.cpp` — minimal headless C++14 simulation with:
  - 1 AI field player (Team A)
  - 1 scripted teammate (Team A)
  - 1 scripted opponent (Team B)
  - Ball, goals, field boundaries, possession flags, basic scoring
- `python/teamsports_rl.pyd` — pybind11 compiled extension (generated)
- `python/teamsports_env.py` — Gymnasium wrapper
- `python/test_random_agent.py` — random action loop
- `python/scripted_baseline.py` — rule-based baseline
- `python/evaluate.py` — random vs scripted comparison
- `python/train_ppo.py` — PyTorch PPO implementation with GAE

## Project layout

```
PythonApplication1/
??? cpp/
?   ??? rl_env.cpp
?   ??? TeamSportsEnv.sln
?   ??? TeamSportsEnv.vcxproj
??? python/
?   ??? teamsports_env.py
?   ??? test_random_agent.py
?   ??? scripted_baseline.py
?   ??? evaluate.py
?   ??? train_ppo.py
??? venv/
??? requirements.txt
??? setup.py
??? build_cpp.bat
??? build_env.py
??? run_random_agent.py
??? README.md
```

## Quick start

```cmd
python build_env.py --install
build_cpp.bat
python run_random_agent.py
```

## Low-level pybind11 API

```python
import teamsports_rl
import numpy as np

env = teamsports_rl.TeamSportsEnv()
obs, info = env.reset()

action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
```

## Gymnasium wrapper API

```python
from teamsports_env import TeamSportsEnv

env = TeamSportsEnv()
obs, info = env.reset()

obs, reward, terminated, truncated, info = env.step(
    np.array([0.5, 0.0, 0.0], dtype=np.float32)
)
```

## Observation schema

`OBS_DIM = 22`

| Indices | Content |
|---|---|
| 0-1 | self position (normalized) |
| 2-3 | self velocity |
| 4-5 | ball relative position |
| 6-7 | ball velocity |
| 8-9 | teammate relative position |
| 10-11 | teammate velocity |
| 12-13 | opponent relative position |
| 14-15 | opponent velocity |
| 16-17 | own goal relative position |
| 18-19 | enemy goal relative position |
| 20 | self has ball (0/1) |
| 21 | teammate has ball (0/1) |

## Action schema

`[move_x, move_y, kick]`

- `move_x ? [-1, 1]`
- `move_y ? [-1, 1]`
- `kick ? {0, 1}`

## Next steps

- Run random and scripted baselines.
- Tune reward function.
- Train PPO with small rollouts first.
- Add goalkeeper and second human field player.
