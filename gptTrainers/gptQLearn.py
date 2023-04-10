import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('Taxi-v3', render_mode="rgb_array")
state_space = env.observation_space.shape
action_space = env.action_space.n

class DQNCNN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxPool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.maxPool(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

model = DQNCNN(70,110,6).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

episodes = 10000
totalWins = 0
batchSize = 32

memory = deque(maxlen=10000)
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = (episodes-2)/episodes
precision = len(str(episodes))-1
explore = 0
exploit = 0


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    global explore
    global exploit
    if np.random.rand() <= epsilon:
        explore += 1
        return env.action_space.sample()
    state_tensor = Variable(torch.from_numpy(state.copy()).float().unsqueeze(0)).to(device)
    q_values = model(state_tensor)
    _, action = torch.max(q_values, 1)
    exploit += 1
    return int(action.item())

def replay(batch_size):
    global epsilon
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            next_state_tensor = Variable(torch.from_numpy(next_state.copy()).float().unsqueeze(0)).to(device)
            target = (reward + gamma * torch.max(model(next_state_tensor)).item())
        state_tensor = Variable(torch.from_numpy(state.copy()).float().unsqueeze(0)).to(device)
        target_f = model(state_tensor)
        target_f[0][action] = target
        loss = criterion(target_f, model(state_tensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


for e in range(episodes):
    env.reset()
    state = env.render()
    state = Image.fromarray(state)
    state = state.resize((110,70))
    state = np.asarray(state)
    state = np.transpose(state, (2, 0, 1))
    for time in range(201):
        action = act(state)
        numNextState, reward, term, trunc, info = env.step(action)
        done = term or trunc
        next_state = env.render()
        next_state = Image.fromarray(next_state)
        next_state = next_state.resize((110,70))
        next_state = np.transpose(next_state, (2, 0, 1))
        next_state = np.asarray(next_state)
        remember(state, action, reward, next_state, done)
        state = next_state
        if term:
            totalWins += 1
        if done:
            print("episode: {}/{} - explore/exploit: {}/{}, score: {}, Success: {}, e: {:.{precision}f}, Total Wins: {}".format(e, episodes, explore, exploit, time+1, term, epsilon, totalWins, precision=precision))
            explore = 0
            exploit = 0
            break
    if len(memory) > batchSize:
        replay(batchSize)

torch.save(model, "gptModel0.pt")