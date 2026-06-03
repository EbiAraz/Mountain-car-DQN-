import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
import matplotlib.pyplot as plt

## Hyperparameters

ENV_NAME = 'MountainCar-v0'
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
BATCH_SIZE = 64
LR = 0.001
MEMORY_SIZE = 100_000
TARGET_UPDATE_FREQ = 500
MAX_EPISODES = 800
MAX_STEPS =  500  # MuntainCar default is 200, but we give more room
MODEL_PATH = "mountaincar_dqn.pth"
FORCE_TRAIN = False
RENDER_EPISODES = 1

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


def build_networks():
    env = gym.make(ENV_NAME)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError(f"Expected Discrete action space, got {type(env.action_space).__name__}")

    state_dim = int(np.prod(env.observation_space.shape)) if env.observation_space.shape else 1
    action_dim = env.action_space.n
    env.close()

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net


def load_checkpoint(policy_net, target_net, optimizer):
    epsilon = EPS_START
    loaded_checkpoint = False

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            policy_net.load_state_dict(checkpoint["policy_state_dict"])
            if "target_state_dict" in checkpoint:
                target_net.load_state_dict(checkpoint["target_state_dict"])
            else:
                target_net.load_state_dict(policy_net.state_dict())
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epsilon = float(checkpoint.get("epsilon", EPS_END))
        else:
            policy_net.load_state_dict(checkpoint)
            target_net.load_state_dict(policy_net.state_dict())
            epsilon = EPS_END
        loaded_checkpoint = True
        print(f"Loaded model from {MODEL_PATH}")

    return epsilon, loaded_checkpoint


def select_action(policy_net, state, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()

    with torch.no_grad():
        q_values = policy_net(torch.FloatTensor(state))
        return q_values.argmax().item()


def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_agent(policy_net, target_net, optimizer, epsilon):
    env = gym.make(ENV_NAME)
    memory = ReplayBuffer(MEMORY_SIZE)
    episode_rewards = []
    steps_done = 0

    policy_net.train()
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            steps_done += 1
            action = select_action(policy_net, state, epsilon, env.action_space)

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)

            shaped_reward = reward + (next_state[0] + 0.5) * 5.0
            memory.push(state, action, shaped_reward, next_state, done or truncated)

            state = next_state
            optimize_model(policy_net, target_net, optimizer, memory)

            if done or truncated:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        print(f"Episode {episode:04d} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"--> last 50 episodes avg reward = {avg_reward:.1f}")

    env.close()
    return epsilon, episode_rewards


def save_checkpoint(policy_net, target_net, optimizer, epsilon):
    torch.save(
        {
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": epsilon,
        },
        MODEL_PATH,
    )
    print(f"Saved learned model to {MODEL_PATH}")


def plot_rewards(episode_rewards):
    if not episode_rewards:
        return

    plt.plot(episode_rewards)
    plt.title("MountainCar-v0 - DQN learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


def render_policy(policy_net):
    policy_net.eval()
    render_env = gym.make(ENV_NAME, render_mode="human")

    for render_episode in range(1, RENDER_EPISODES + 1):
        state, _ = render_env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = q_values.argmax().item()
            state, reward, done, truncated, _ = render_env.step(action)
            total_reward += float(reward)
            if done or truncated:
                break

        print(f"Render episode {render_episode:02d} | Reward: {total_reward:.2f}")

    render_env.close()


def main():
    policy_net, target_net = build_networks()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    epsilon, loaded_checkpoint = load_checkpoint(policy_net, target_net, optimizer)

    episode_rewards = []
    if FORCE_TRAIN or not loaded_checkpoint:
        epsilon, episode_rewards = train_agent(policy_net, target_net, optimizer, epsilon)
        save_checkpoint(policy_net, target_net, optimizer, epsilon)
    else:
        print("Using saved model. Set FORCE_TRAIN = True to retrain.")

    plot_rewards(episode_rewards)
    render_policy(policy_net)


if __name__ == "__main__":
    main()



