from PIL import Image
import gym
import matplotlib.pyplot as plt
import pickle
#env = gym.make("CartPole-v1")
#observation, info = env.reset(seed=42)

env = gym.make("Taxi-v3", render_mode="rgb_array")
#, render_mode="human"
#`pip install gym[toy_text]`
observation, info = env.reset()
picture = env.render()
img = Image.fromarray(picture)

# show the image
img.show()
#2d array contains values at specific spaces and actions
#discrete spaces
rows = 500
#actions
cols = 6

# create a 2D array filled with zeroes
qTable = [[0 for j in range(cols)] for i in range(rows)]


#Q(s,a,) = r(s’|s,a) + Y(max(Q(s’,a’))
#observation = Q(s,a,)
#reward = r(s’|s,a)
#best action based on previous interations? = Y(max(Q(s’,a’))
#s’ subsequent action
#a’ subsequent state
for _ in range(100000):
    
    s = observation
    a = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(a)
    qTable[s][a] = reward + .9 * max(qTable[observation])
    if terminated or truncated:
        observation, info = env.reset()

    #env.render()

qTable[observation]

with open('my_array.pkl', 'wb') as f:
    pickle.dump(qTable, f)

env.close()