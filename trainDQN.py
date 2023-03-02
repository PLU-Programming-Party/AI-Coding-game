import time
import random
import argparts
import numpy as np
from collections import deque
import gymnasium as gym
from qNet import qNet
import torch
from PIL import Image
import torch.nn as nn







def fillMemory(env, memory):
    #, render_mode="human"
    #`pip install gym[toy_text]`
    observation, info = env.reset()
    for _ in range(10000):
        s = observation
        a = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(a)
        memory.append((s, a, reward, observation))
        if terminated or truncated:
            observation, info = env.reset()
    return memory

def trainMain(r_mode):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = qNet()
    memory = []
    memory = fillMemory(env, memory)
    epsilon = 1
    loss = nn.MSELoss()
    for eps in range(10000):
        sample = memory[random.randint(len(memory))]
        prev_img = sample[0]
    
        npImg = np.array(prev_img)
        tensorIMG = torch.from_numpy(npImg).to(torch.float)
        tensorIMG = tensorIMG.permute(2,1,0)
        tensorIMG = tensorIMG.unsqueeze(0)
        qsa = model(tensorIMG)
        qsa = qsa.numpy()
        qsa = qsa[sample[1]]
        next_img = sample[3]
        reward = sample[2]



#def testMain():
if __name__ == "__main__":
    trainMain("test")

