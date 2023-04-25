import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the Agent
class Agent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.model = QNetwork(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999999
        self.epsilon_min = 0.01

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.tensor(state, dtype=torch.float).view(1, 1).to(device)
        actions = self.model(state)
        return torch.argmax(actions).item()

    def train(self, state, action, next_state, reward, done, info):
        if((action == 4 or action == 5) and (info['action_mask'][action] == 1)):
            reward = 25
            #print("Successful pick-up.")
        if reward == 20:
            reward = 1000
            print("Winning move.")
        target = reward
        if not done:
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).view(1, 1).to(device)
            target = (reward + self.gamma * torch.max(self.model(next_state)).item())

        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).view(1, 1).to(device)
        predicted_target = self.model(state)[0][action]

        loss = self.loss_fn(predicted_target, torch.tensor(target).float().to(device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training
def train_agent(agent, env, episodes=100000, save_path="taxi_agent.pt"):
    scores = deque(maxlen=episodes)
    for episode in range(1, episodes + 1):
        state, oldInfo = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, win, loss, info = env.step(action)
            done = win or loss
            agent.train(state, action, next_state, reward, done, oldInfo)
            state = next_state
            oldInfo = info
            total_reward += reward

        scores.append(total_reward)
        mean_score = np.mean(scores)

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {mean_score}")


    print("Training complete.")
    torch.save(agent.model.state_dict(), save_path)


def test_agent(agent, env, episodes=100):
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)

    env.close()
    mean_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {mean_reward}")


if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    env = gym.make("Taxi-v3")
    action_size = env.action_space.n

    # Training
    agent = Agent(action_size)
    train_agent(agent, env)

    # Testing
    saved_agent = Agent(action_size)
    saved_agent.model.load_state_dict(torch.load("taxi_agent.pt"))
    saved_agent.epsilon = 0.0  # Turn off exploration during testing
    test_agent(saved_agent, env)
