import time
import random
import numpy as np
from collections import deque
import gymnasium as gym
from numpy import argmax
from torch import optim

from qNet import qNet
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class DQNCNN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        #self.maxPool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.maxPool(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))





def fillMemory(env, memory):
    #, render_mode="human"
    #`pip install gym[toy_text]`
    observation, info = env.reset()
    x = 10000
    for _ in range(x):
        a = env.action_space.sample()
        pictureA = env.render()
        pictureA = Image.fromarray(pictureA)
        pictureA = pictureA.resize((55,35))
        pictureA.show()
        pictureA = np.asarray(pictureA)
        observation, reward, terminated, truncated, info = env.step(a)
        pictureB = env.render()
        pictureB = Image.fromarray(pictureB)
        pictureB = pictureB.resize((55, 35))
        pictureB = np.asarray(pictureB)
        memory.append((pictureA, a, reward, pictureB))
        if terminated or truncated:
            observation, info = env.reset()

        if _ % 100 == 0:
            print("Fill Memory Progress: ", _, " / ", x)

    return memory

def trainMain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = qNet().to(device)
    memory = []
    memory = fillMemory(env, memory)
    epsilon = 1
    LR = 1e-4
    loss = nn.MSELoss()
    x = 1000000000
    batch_size = 128
    for eps in range(x):
        sample = random.sample(memory, batch_size)
        prev_img = [tup[0] for tup in sample]
    
        npImg = np.array(prev_img)
        tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        tensorIMG = tensorIMG.permute(0,3,2,1)
        #tensorIMG = tensorIMG.unsqueeze(0)
        qsa = model(tensorIMG)
        actions = [tup[1] for tup in sample]
        qsa = model(tensorIMG).gather(1, torch.tensor(actions).unsqueeze(1).to(device))
        next_img = [tup[3] for tup in sample]
        reward = [tup[2] for tup in sample]

        next_img = np.array(next_img)
        next_img = torch.from_numpy(next_img).to(torch.float).to(device)
        nextImg = next_img.permute(0,3,2,1)
        #nextImg = nextImg.unsqueeze(0)
        with torch.no_grad():
            qsPlus1A = model(nextImg)
        qsPlus1A = [torch.max(q) for q in qsPlus1A]

        qsPlus1A = [qsPlus1A[i]*.9 + reward[i] for i in range(len(qsPlus1A))]

        error = loss(qsa, torch.tensor(qsPlus1A).to(device))

        optimizer = optim.AdamW(model.parameters(), lr=LR, amsgrad=True)
        error.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

        if eps % 100 == 0:
            print("Training Progress: ", eps, " / ", x)
        if eps % 100000 == 0:
            torch.save(model, "DQN.pt")
            testMain()

    torch.save(model, "DQN.pt")
    testMain()


def testMain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = torch.load("gptModel0.pt").to(device)


    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    observation, info = env.reset()
    observation = env.render()

    total = 0
    wins = 0

    for _ in range(10000):

        s = observation
        npImg = np.array(s)
        npImg = Image.fromarray(npImg)
        npImg.show()
        npImg = npImg.resize((55, 35))
        npImg = np.asarray(npImg)
        tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        tensorIMG = tensorIMG.permute(2, 1, 0)
        tensorIMG = tensorIMG.unsqueeze(0)

        with torch.no_grad():
            output = model(tensorIMG)

        #print(output)
        large = torch.finfo(output.dtype).max
        ac = (output - large * (1 - torch.from_numpy(info["action_mask"]).to(device)) - large * (1 - torch.from_numpy(info["action_mask"]).to(device))).argmax()
        output = torch.argmax(output, dim = 1)


        observation, reward, terminated, truncated, info = env.step(ac.item())
        picture = env.render()

        if terminated or truncated:
            observation, info = env.reset()
            total = total + 1
            if terminated:
                wins = wins + 1
                #print("Game Won!")
            if truncated:
                wins = wins
                #print("Game Lost")
            print("Total Wins: ", wins, " / ", total)
        observation = env.render()

    env.close()






#def testMain():
if __name__ == "__main__":
    trainMain()

