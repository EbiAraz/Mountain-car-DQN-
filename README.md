# Mountain Car DQN

A clean Deep Q-Network implementation for solving `MountainCar-v0` with PyTorch and Gymnasium.

This project trains an agent to drive the underpowered car up the hill by learning a value function over the discrete action space. It includes checkpoint loading, reward-curve plotting, and policy rendering for quick experimentation.

## Highlights

- PyTorch DQN with a two-layer MLP
- Experience replay buffer
- Target network synchronization
- Checkpoint save and reload support
- Reward shaping to speed up Mountain Car learning
- Training curve visualization and rendered evaluation

## Preview

The agent learns from `(position, velocity)` observations and chooses one of three actions:

| Action | Meaning |
| --- | --- |
| `0` | Push left |
| `1` | No push |
| `2` | Push right |

## Project Structure

```text
.
├── Mountain Car.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python "Mountain Car.py"
```

## Configuration

You can adjust the main hyperparameters at the top of the script:

- `MAX_EPISODES`: total training episodes
- `MAX_STEPS`: max steps per episode
- `EPS_START`, `EPS_END`, `EPS_DECAY`: epsilon-greedy schedule
- `TARGET_UPDATE_FREQ`: target network refresh interval
- `FORCE_TRAIN`: retrain even if a saved checkpoint exists
- `RENDER_EPISODES`: number of evaluation episodes after training

## How It Works

1. The agent collects transitions in a replay buffer.
2. Mini-batches are sampled to stabilize updates.
3. A target network provides bootstrapped Q targets.
4. Reward shaping helps the agent discover uphill progress earlier.
5. The trained weights are stored in `mountaincar_dqn.pth`.

## Tech Stack

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

## Notes

- `MountainCar-v0` is slow to solve without reward shaping.
- Rendering opens a local window and may not work in headless environments.
- If you only want to evaluate a trained checkpoint, keep `FORCE_TRAIN = False`.

## GitHub Description

Deep Q-Network for `MountainCar-v0` using PyTorch, experience replay, target networks, and reward shaping.