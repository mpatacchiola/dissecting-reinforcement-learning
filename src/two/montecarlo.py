#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I will use the class gridworld to generate a 3x4 world
#in which the cleaning robot will move. In this example the robot follows
#a policy which is optimal, reaching the terminal state (+1) with high probability.

import numpy as np
from gridworld import GridWorld

env = GridWorld(3, 4)

#Define the state matrix
state_matrix = np.zeros((3,4))
state_matrix[0, 3] = 1
state_matrix[1, 3] = 1
state_matrix[1, 1] = -1

#Define the reward matrix
reward_matrix = np.full((3,4), -0.04)
reward_matrix[0, 3] = 1
reward_matrix[1, 3] = -1

#Define the transition matrix
transition_matrix = np.array([[0.90, 0.05, 0.0, 0.05],
                              [0.05, 0.90, 0.05, 0.0],
                              [0.0, 0.05, 0.90, 0.05],
                              [0.05, 0.0, 0.05, 0.90]])

#Define the policy matrix
policy_matrix = np.array([[1, 1, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 3],
                          [0, 3, 3, 3]])


env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward_matrix)
env.setTransitionMatrix(transition_matrix)

#Reset the environment
observation = env.reset()
env.render()

for _ in range(1000):
    action = policy_matrix[observation[0], observation[1]]
    observation, reward, done = env.step(action)
    print("")
    print("ACTION: " + str(action))
    print("REWARD: " + str(reward))
    print("DONE: " + str(done))
    env.render()
    if done: break


