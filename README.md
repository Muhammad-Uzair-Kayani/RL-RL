# RL-RL
# RocketAI

A small research prototype for training a reinforcement-learning agent to play a Rocket League-like 2D game.

The project focuses on the **AI architecture and C++ ↔ Python interaction**, rather than building the full game first.

## Current Goal

Build a minimal, headless C++ simulation that provides a game-relevant environment for an RL agent.

The initial pipeline is:

```text
C++ Simulation
      ↓
 Observation
      ↓
 Python / PyTorch
      ↓
    Action
      ↓
 C++ Simulation
      ↓
    Reward
      ↓
 Python / PPO
```

Once this works, the same AI interface can eventually be connected to the actual 2D game.

## Prototype Scope

The first simulation is intentionally small:

* 2 teams
* 1 AI field player
* 1 teammate
* 1 opposing player
* 1 ball
* 2 goals
* Basic movement
* Basic ball movement
* Basic kicking/ball interaction
* Ball possession
* Scoring
* Episode reset and termination

The goalkeeper, human-player switching, rendering, advanced physics, and complex tactics will be added later.

## AI Objective

The long-term objective is to train field-player agents that can **support their team and react to the game state**, rather than simply chase the ball.

Potential learned behaviors include:

* Positioning
* Supporting the ball carrier
* Attacking
* Defending
* Ball interaction
* Passing
* Creating opportunities
* Reacting to opponents

## Observation

The agent receives structured numerical state rather than raw pixels.

A possible observation contains:

```text
Self position
Self velocity

Ball relative position
Ball velocity

Teammate relative position
Opponent relative position

Own goal relative position
Enemy goal relative position

Ball possession
Teammate possession
```

The observation representation will be kept compact and easy to replace when the real game is implemented.

## Action Space

The initial action space is:

```text
move_x ∈ [-1, 1]
move_y ∈ [-1, 1]
kick ∈ {0, 1}
```

The exact action representation may evolve with the game.

## Reinforcement Learning

The planned RL algorithm is **Proximal Policy Optimization (PPO)** implemented with PyTorch.

The first development stages use random and scripted agents to verify the environment before training begins.

```text
Environment
    ↓
Random Agent
    ↓
Scripted Baseline
    ↓
PPO
    ↓
Evaluation
```

## Technology

### C++

* C++17/20
* Visual Studio 2022
* pybind11

### Python

* Python 3.12
* PyTorch
* NumPy
* Gymnasium
* Matplotlib
* TensorBoard

Python dependencies are isolated using a project-local virtual environment.

## Project Philosophy

The simulation is a **temporary stand-in for the future game**, not the final game itself.

The project intentionally follows an incremental approach:

```text
Minimal Simulation
        ↓
C++ ↔ Python Interface
        ↓
Observation / Action API
        ↓
Reward System
        ↓
Baseline Agents
        ↓
PPO
        ↓
Expanded Simulation
        ↓
Actual 2D Game
        ↓
AI Integration
```

The goal is to avoid building a complete game before knowing whether the AI architecture works.

## Status

🚧 **Early development**

Current priority:

* [ ] Minimal C++ simulation
* [ ] pybind11 interface
* [ ] `reset()`
* [ ] `get_observation()`
* [ ] `step(action)`
* [ ] Reward system
* [ ] Episode termination
* [ ] Python random agent
* [ ] Scripted baseline
* [ ] PPO implementation
* [ ] Training/evaluation tools
* [ ] Expand simulation toward Prototype 1
* [ ] Build actual 2D game
* [ ] Connect trained agent to the game

## Future Prototype 1

The eventual game prototype will contain:

```text
Team A                         Team B

Goalkeeper                     Goalkeeper
AI Field Player                AI Field Player
Human Field Player             Human Field Player
```

Total: **6 active players**

The human player will be able to switch between their field players. The uncontrolled field player will be controlled by the AI.

The goalkeeper will initially use rule-based logic.

## License

License to be determined.
