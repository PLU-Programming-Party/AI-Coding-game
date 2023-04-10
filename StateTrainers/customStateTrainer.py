import copy
import random
import numpy as np
import gymnasium as gym
from torch import optim

import torch
import torch.nn as nn


class DQNAgent(nn.Module):
    def __init__(self, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def fillMemory(env, memory):
    observation, info = env.reset()
    x = 10000
    for _ in range(x):
        a = env.action_space.sample()
        #pictureA = env.render()
        #pictureA = Image.fromarray(pictureA)
        #pictureA = pictureA.resize((55, 35))
        # pictureA.show()
        #pictureA = np.asarray(pictureA)
        newObservation, reward, terminated, truncated, info = env.step(a)
        #pictureB = env.render()
        #pictureB = Image.fromarray(pictureB)
        #pictureB = pictureB.resize((55, 35))
        #pictureB = np.asarray(pictureB)
        memory.append((observation, a, reward, newObservation))
        observation = newObservation
        if terminated or truncated:
            observation, info = env.reset()

        if _ % 100 == 0:
            print("Fill Memory Progress: ", _, " / ", x)

    return memory


def trainMain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = DQNAgent(6).to(device)
    theOGModel = copy.deepcopy(model)
    memory = []
    memory = fillMemory(env, memory)
    epsilon = 1
    LR = 1e-4
    loss = nn.MSELoss()
    x = 1000000
    batch_size = 128
    for eps in range(x):
        sample = random.sample(memory, batch_size)
        prev_img = [tup[0] for tup in sample]
        tensorIMG = torch.tensor(np.asarray(prev_img), dtype=torch.float32).view(batch_size, 1)

        #npImg = np.array(prev_img)
        #tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        #tensorIMG = tensorIMG.permute(0, 3, 2, 1)
        ##### tensorIMG = tensorIMG.unsqueeze(0)
        qsa = model(tensorIMG)
        actions = [tup[1] for tup in sample]
        qsa = model(tensorIMG).gather(1, torch.tensor(actions).unsqueeze(1).to(device))
        next_img = [tup[3] for tup in sample]
        reward = [tup[2] for tup in sample]

        nextImg = torch.tensor(np.asarray(next_img), dtype=torch.float32).view(batch_size, 1)
        #next_img = np.array(next_img)
        #next_img = torch.from_numpy(next_img).to(torch.float).to(device)
        #nextImg = next_img.permute(0, 3, 2, 1)
        # nextImg = nextImg.unsqueeze(0)
        with torch.no_grad():
            qsPlus1A = model(nextImg)
        qsPlus1A = [torch.max(q) for q in qsPlus1A]

        qsPlus1A = [qsPlus1A[i] * .9 + reward[i] for i in range(len(qsPlus1A))]

        error = loss(qsa, torch.tensor(qsPlus1A).to(device))

        optimizer = optim.AdamW(model.parameters(), lr=LR, amsgrad=True)
        error.backward()

        #for param, param_copy in zip(model.parameters(), theOGModel.parameters()):
            #param.data.copy_(0.99 * param_copy.data + 0.01 * param.data)

        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

        if eps % 100 == 0:
            print("Training Progress: ", eps, " / ", x)
        if eps % 10000 == 0:
            torch.save(model, "stateDQN.pt")
            testMain()
            #memory = []
            #memory = fillMemory(env, memory)

    torch.save(model, "stateDQN.pt")
    testMain()


def testMain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = torch.load("stateDQN.pt", map_location=torch.device(device)).to(device)

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    observation, info = env.reset()
    #observation = env.render()

    total = 0
    wins = 0

    for _ in range(10000):

        s = observation
       # npImg = np.array(s)
        #npImg = Image.fromarray(npImg)
        #npImg.show()
        #npImg = npImg.resize((55, 35))
        #npImg = np.asarray(npImg)
        #tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        #tensorIMG = tensorIMG.permute(2, 1, 0)
        #tensorIMG = tensorIMG.unsqueeze(0)

        with torch.no_grad():
            output = model(torch.tensor(np.asarray(observation), dtype=torch.float32).unsqueeze(0))

        # print(output)
        large = torch.finfo(output.dtype).max
        ac = (output - large * (1 - torch.from_numpy(info["action_mask"]).to(device)) - large * (
                    1 - torch.from_numpy(info["action_mask"]).to(device))).argmax()
        output = torch.argmax(output)

        observation, reward, terminated, truncated, info = env.step(ac.item())
        #picture = env.render()

        if terminated or truncated:
            observation, info = env.reset()
            total = total + 1
            if terminated:
                wins = wins + 1
                # print("Game Won!")
            if truncated:
                wins = wins
                # print("Game Lost")
            print("Total Wins: ", wins, " / ", total)
        #observation = env.render()

    env.close()


# def testMain():
if __name__ == "__main__":
    trainMain()

