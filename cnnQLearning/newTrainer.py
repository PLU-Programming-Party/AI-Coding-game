import copy
import random
import numpy as np
import gymnasium as gym
from torch import optim

from cnnQLearning.qNet import qNet
import torch
from PIL import Image
import torch.nn as nn



def fillMemory(env, memory):
    seed_num = 1
    observation, info = env.reset(seed=seed_num)
    mem_Size = 1000
    winners = 10
    winCount = 0
    while len(memory) < mem_Size or winCount < winners:
        a = env.action_space.sample()
        pictureA = env.render()
        pictureA = Image.fromarray(pictureA)
        #pictureA = pictureA.resize((55, 35))
        # pictureA.show()
        pictureA = np.asarray(pictureA)
        pictureA = pictureA/255
        observation, reward, terminated, truncated, info = env.step(a)
        pictureB = env.render()
        pictureB = Image.fromarray(pictureB)
        #pictureB = pictureB.resize((55, 35))
        pictureB = np.asarray(pictureB)
        pictureB = pictureB/255
        memory.append((pictureA, a, reward, pictureB))

        if reward == 20:
            winCount = winCount + 1
        if(len(memory) % 100 == 0):
            print("Fill Memory Progress: ", len(memory), " / ", mem_Size, "Total Wins: ", winCount)

        if terminated or truncated:
            observation, info = env.reset(seed=seed_num)

        # if (winCount < winners):
        #     memory = []
        if(len(memory) > mem_Size*2):
            losers = [i for i, t in enumerate(memory) if t[2] != 20]
            removeMe = set(random.sample(losers, len(memory) - mem_Size))
            newMem = [x for i, x in enumerate(memory) if i not in removeMe]
            memory = newMem
            losers = []
            removeMe = []

            newMem = []



    # randomly samples the memory for losers
    # removes all the sampled loser indexes from memory
    losers = [i for i, t in enumerate(memory) if t[2] != 20]
    removeMe = set(random.sample(losers, len(memory) - mem_Size))
    newMem = [x for i, x in enumerate(memory) if i not in removeMe]
    memory = newMem
    #
    # testState = newMem[0]
    # newMem = newMem[:mem_Size]
    return memory


def trainMain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = qNet().to(device)
    theOGModel = copy.deepcopy(model)
    memory = []
    memory = fillMemory(env, memory)
    epsilon = 1
    LR = 1e-4
    loss = nn.MSELoss()
    iteration = 500000
    batch_size = 16
    for eps in range(iteration):
        sample = random.sample(memory, batch_size)
        prev_img = [tup[0] for tup in sample]

        npImg = np.array(prev_img)
        tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        tensorIMG = tensorIMG.permute(0, 3, 2, 1)
        # tensorIMG = tensorIMG.unsqueeze(0)
        actions = [tup[1] for tup in sample]
        #qsa = model(tensorIMG)
        qsa = model(tensorIMG).gather(1, torch.tensor(actions).unsqueeze(1).to(device))
        next_img = [tup[3] for tup in sample]
        reward = [tup[2] for tup in sample]

        next_img = np.array(next_img)
        next_img = torch.from_numpy(next_img).to(torch.float).to(device)
        nextImg = next_img.permute(0, 3, 2, 1)
        # nextImg = nextImg.unsqueeze(0)
        with torch.no_grad():
            qsPlus1A = theOGModel(nextImg)
        qsPlus1A = [torch.max(q) for q in qsPlus1A]

        #Source of much woe.
        qsPlus1A = [2000 if reward[i]==20 else qsPlus1A[i] * .9 + reward[i] for i in range(len(qsPlus1A))]
        #qsPlus1A = [qsPlus1A[i] * .9 + reward[i] for i in range(len(qsPlus1A))]
        # qsPlus1A = [20.0 for i in range(len(qsPlus1A))]

        error = loss(qsa, torch.tensor(qsPlus1A).to(torch.float).to(device))

        optimizer = optim.AdamW(model.parameters(), lr=LR, amsgrad=True)
        error.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

        for param, param_copy in zip(model.parameters(), theOGModel.parameters()):
            param.data.copy_(0.99 * param_copy.data + 0.01 * param.data)
        theOGModel = copy.deepcopy(model)

        if eps % 10 == 0:
            print("Training Progress: ", eps, " / ", iteration)
        if eps % 10000 == 0:
            torch.save(model, "newDQN.pt")
            testMain()
            #memory = []
            #memory = fillMemory(env, memory)

    torch.save(model, "newDQN.pt")
    testMain()


def testMain():
    print("Testing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    model = torch.load("newDQN.pt", map_location=torch.device(device)).to(device)

    #memory = []
    #memory, testState = fillMemory(env, memory)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    observation, info = env.reset()
    observation = env.render()

    total = 0
    wins = 0

    for memInstance in range(1000):

        s = observation
        npImg = np.array(s)
        npImg = Image.fromarray(npImg)
        #npImg.show()
        #npImg = npImg.resize((55, 35))
        npImg = np.asarray(npImg)/255
        tensorIMG = torch.from_numpy(npImg).to(torch.float).to(device)
        tensorIMG = tensorIMG.permute(2, 1, 0)
        tensorIMG = tensorIMG.unsqueeze(0)

        with torch.no_grad():
            output = model(tensorIMG)

        # print(output)
        large = torch.finfo(output.dtype).max
        #print(output)
        ac = (output - large * (1 - torch.from_numpy(info["action_mask"]).to(device)) - large * (
                   1 - torch.from_numpy(info["action_mask"]).to(device))).argmax()
        output = torch.argmax(output, dim=1)
        #print("Selected Action: " + str(output))


        observation, reward, terminated, truncated, info = env.step(ac.item())
        picture = env.render()

        if terminated or truncated:
            observation, info = env.reset()
            total = total + 1
            if terminated:
                wins = wins + 1
                print("Game Won!")
            if truncated:
                wins = wins
                print("Game Lost")
            print("Total Wins: ", wins, " / ", total)
        observation = env.render()

    env.close()


if __name__ == "__main__":

    trainMain()

