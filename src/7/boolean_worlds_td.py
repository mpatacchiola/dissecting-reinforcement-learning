#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#In this script the TD(0) tabular algorithm is used to estimate the utilities
#of the boolean worlds.

import numpy as np
from gridworld import GridWorld


def init_and():
    '''Init the AND boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[-1.0, 0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0, -1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1]]),
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    return env, np.zeros((5,5))

def init_nand():
    '''Init the NAND boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[1.0, 0.0, 0.0, 0.0, -1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    return env, np.zeros((5,5))

def init_or():
    '''Init the OR boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    return env, np.zeros((5,5))

def init_xor():
    '''Init the XOR boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[-1.0, 0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0, -1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    return env, np.zeros((5,5))



def update_utility(utility_matrix, observation, new_observation, 
                   reward, alpha, gamma, done):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix
    '''
    if done:
        u = utility_matrix[observation[0], observation[1]]
        utility_matrix[observation[0], observation[1]] += alpha * (reward - u)
    else:
        u = utility_matrix[observation[0], observation[1]]
        u_t1 = utility_matrix[new_observation[0], new_observation[1]]
        utility_matrix[observation[0], observation[1]] += \
            alpha * (reward + gamma * u_t1 - u)
    return utility_matrix


def main():

    env, utility_matrix_and = init_and()
    env_nand, utility_matrix_nand = init_nand()
    env_or, utility_matrix_or = init_or()
    env_xor, utility_matrix_xor = init_xor()

    gamma = 0.999
    alpha = 0.1 #constant step size
    tot_epoch = 300000
    print_epoch = 1000

    for epoch in range(tot_epoch):
        #Reset and return the first observation
        observation = env.reset(exploring_starts=False)
        for step in range(1000):
            #Take the action from the action matrix
            action = np.random.randint(0,4)
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            utility_matrix_and = update_utility(utility_matrix_and, observation, 
                                            new_observation, reward, alpha, gamma, done)
            observation = new_observation
            #print(utility_matrix)
            if done: break

        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(np.flipud(utility_matrix_and))
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(np.flipud(utility_matrix_and))



if __name__ == "__main__":
    main()
