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







def fillMemory(env, memory):
    #, render_mode="human"
    #`pip install gym[toy_text]`
    observation, info = env.reset()
    x = 5000
    for _ in range(x):
        a = env.action_space.sample()
        pictureA = env.render()
        pictureA = Image.fromarray(pictureA)
        pictureA = pictureA.resize((55,35))
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
    x = 10000000
    for eps in range(x):
        sample = memory[random.randint(0,len(memory)-1)]
        prev_img = sample[0]
    
        npImg = np.array(prev_img)
        tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        tensorIMG = tensorIMG.permute(2,1,0)
        tensorIMG = tensorIMG.unsqueeze(0)
        qsa = model(tensorIMG)
        qsa = qsa[0, sample[1]]
        next_img = sample[3]
        reward = sample[2]

        next_img = np.array(next_img)
        next_img = torch.from_numpy(next_img).to(torch.float).to(device)
        nextImg = next_img.permute(2,1,0)
        nextImg = nextImg.unsqueeze(0)
        with torch.no_grad():
            qsPlus1A = model(nextImg)
        qsPlus1A = torch.max(qsPlus1A)

        qsPlus1A = (reward + .9 * qsPlus1A)

        error = loss(qsa, qsPlus1A)

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
    model = torch.load("DQN.pt").to(device)


    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    observation, info = env.reset()
    observation = env.render()

    total = 0
    wins = 0

    for _ in range(10000):

        s = observation
        npImg = np.array(s)
        npImg = Image.fromarray(npImg)
        #npImg.show()
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

