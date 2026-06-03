import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class TrainConfig:
    episodes: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    learning_rate: float = 1e-3
    target_update_interval: int = 10
    replay_capacity: int = 10000
    max_steps: int = 200


def select_action(model: DQN, state: np.ndarray, epsilon: float, action_size: int, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(torch.argmax(model(state_tensor), dim=1).item())


def optimize_model(
    model: DQN,
    target_model: DQN,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> None:
    if len(buffer) < batch_size:
        return

    transitions = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = model(states_tensor).gather(1, actions_tensor).squeeze(1)
    with torch.no_grad():
        next_q_values = target_model(next_states_tensor).max(1).values
        target_q_values = rewards_tensor + gamma * next_q_values * (1.0 - dones_tensor)

    loss = nn.functional.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(config: TrainConfig) -> None:
    import gymnasium as gym

    env = gym.make("MountainCar-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(state_size, action_size).to(device)
    target_model = DQN(state_size, action_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.replay_capacity)

    epsilon = config.epsilon_start
    for episode in range(config.episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(config.max_steps):
            action = select_action(model, state, epsilon, action_size, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, done)
            optimize_model(model, target_model, optimizer, buffer, config.batch_size, config.gamma, device)

            state = next_state
            episode_reward += reward
            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if (episode + 1) % config.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode + 1}/{config.episodes} Reward: {episode_reward:.2f} Epsilon: {epsilon:.3f}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on MountainCar-v0 using gymnasium and torch.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(episodes=args.episodes, batch_size=args.batch_size, max_steps=args.max_steps)
    train(config)


if __name__ == "__main__":
    main()
