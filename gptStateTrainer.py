import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def epsilon_greedy_action(agent, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size-1)
    else:
        with torch.no_grad():
            state = torch.tensor(np.asarray(state), dtype=torch.float32).unsqueeze(0)
            q_values = agent(state)
            return q_values.argmax(dim=1).item()

def optimize_agent(agent, target_agent, memory, batch_size, optimizer, gamma):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    state, action, reward, next_state, done = zip(*batch)
    state = torch.tensor(np.asarray(state), dtype=torch.float32)
    action = torch.tensor(np.asarray(action), dtype=torch.int64).unsqueeze(-1)
    reward = torch.tensor(np.asarray(reward), dtype=torch.float32).unsqueeze(-1)
    next_state = torch.tensor(np.asarray(next_state), dtype=torch.float32)
    done = torch.tensor(np.asarray(done), dtype=torch.float32).unsqueeze(-1)

    q_values = agent(state).gather(1, action)
    with torch.no_grad():
        target_q_values = target_agent(next_state).max(dim=1, keepdim=True)[0]
        target_q_values = reward + gamma * (1 - done) * target_q_values

    loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Hyperparameters
episodes = 5000
batch_size = 64
memory_size = 10000
gamma = 0.99
learning_rate = 0.001
target_update_frequency = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Setup environment
env = gym.make('Taxi-v3')
state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize agent, target agent, and memory
agent = DQNAgent(state_size, action_size)
target_agent = DQNAgent(state_size, action_size)
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# Training loop
step_count = 0
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        action = epsilon_greedy_action(agent, state, epsilon)
        next_state, reward, win, trunc, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))

        optimize_agent(agent, target_agent, memory, batch_size, optimizer, gamma)
        step_count += 1
        state = next_state

        if step_count % target_update_frequency == 0:
            target_agent.load_state_dict(agent.state_dict())

    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Epsilon: {epsilon:.2f}")

# Testing the trained agent
test_episodes = 10
agent.eval()
env = gym.make('Taxi-v3', render_mode='human')
for episode in range(test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = epsilon_greedy_action(agent, state, 0.0)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    print(f"Test Episode: {episode + 1}, Reward: {episode_reward}")

env.close()