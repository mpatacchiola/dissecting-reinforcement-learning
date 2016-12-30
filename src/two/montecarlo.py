import numpy as np
from gridworld import GridWorld

env = GridWorld(3, 4)
state_matrix = np.zeros((3,4))
state_matrix[0, 3] = 1
state_matrix[1, 3] = 1
state_matrix[1, 1] = -1

reward_matrix = np.zeros((3,4))
reward_matrix.fill(-0.04)
reward_matrix[0, 3] = 1
reward_matrix[1, 3] = -1

env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward_matrix)

env.reset()

while(True):
    observation, reward, done = env.step(0)
    print("===========")
    env.render()
    print(observation)
    print(reward)
    print(done)
    if done: break


