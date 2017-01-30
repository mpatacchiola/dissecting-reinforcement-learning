#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I will use the class gridworld to generate a 3x4 world
#in which the cleaning robot will move. Using the Monte Carlo method I
#will estimate the utility values of each state.

import numpy as np
from gridworld import GridWorld

def get_return(state_list, gamma):
    counter = 0
    return_value = 0
    for visit in state_list:
        reward = visit[1]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value


def main():

    env = GridWorld(3, 4)

    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    #Define the policy matrix
    #This is the optimal policy for world with reward=-0.04
    policy_matrix = np.array([[1,      1,  1,  -1],
                              [0, np.NaN,  0,  -1],
                              [0,      3,  3,   3]])
    print("Policy Matrix:")
    print(policy_matrix)

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3,4))
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((3,4), 1.0e-10) 
    gamma = 0.999
    tot_epoch = 50000
    print_epoch = 1000

    for epoch in range(tot_epoch):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=False)
        for _ in range(1000):
            #Take the action from the action matrix
            action = policy_matrix[observation[0], observation[1]]
            #Move one step in the environment and get obs and reward
            observation, reward, done = env.step(action)
            #Append the visit in the episode list
            episode_list.append((observation, reward))
            if done: break
        #The episode is finished, now estimating the utilities
        counter = 0
        #Checkup to identify if it is the first visit to a state
        checkup_matrix = np.zeros((3,4))
        #This cycle is the implementation of First-Visit MC.
        #For each state stored in the episode list check if it
        #is the rist visit and then estimate the return.
        for visit in episode_list:
            observation = visit[0]
            row = observation[0]
            col = observation[1]
            reward = visit[1]
            if(checkup_matrix[row, col] == 0):
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                utility_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(utility_matrix / running_mean_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility_matrix / running_mean_matrix)



if __name__ == "__main__":
    main()
