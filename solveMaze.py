import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import numpy
#env = gym.make("CartPole-v1")
#observation, info = env.reset(seed=42)

env = gym.make("Taxi-v3", render_mode="human")
#, render_mode="human"
observation, info = env.reset()
env.render()

with open('my_array.pkl', 'rb') as f:
    # load the 2D array from the file
    qTable = pickle.load(f)


#Q(s,a,) = r(s’|s,a) + Y(max(Q(s’,a’))
#observation = Q(s,a,)
#reward = r(s’|s,a)
#best action based on previous interations? = Y(max(Q(s’,a’))
#s’ subsequent action
#a’ subsequent state
for _ in range(100000):
    
    s = observation
    a = numpy.argmax(qTable[s])
    observation, reward, terminated, truncated, info = env.step(a)
    if terminated or truncated:
        observation, info = env.reset()

env.close()