#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I will use the class gridworld to generate a 3x4 world
#in which the cleaning robot will move. Using the TD(0) algorithm I
#will estimate the utility values of each state.

import numpy as np
from gridworld import GridWorld


def update_utility(utility_matrix, trace_matrix, alpha, delta):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param alpha the step size (learning rate)
    @param delta the error (Taget-OldEstimte) 
    @return the updated utility matrix
    '''
    utility_matrix += alpha * delta * trace_matrix
    return utility_matrix

def update_eligibility(trace_matrix, gamma, lambda_):
    '''Return the updated trace_matrix

    @param trace_matrix the eligibility traces matrix
    @param gamma discount factor
    @param lambda_ the decaying value
    @return the updated trace_matrix
    '''
    trace_matrix = trace_matrix * gamma * lambda_
    return trace_matrix

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

    #Define and print the eligibility trace matrix
    trace_matrix = np.zeros((3,4))
    print("Trace Matrix:")
    print(trace_matrix)

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3,4))
    gamma = 0.999 #discount rate
    alpha = 0.1 #constant step size
    lambda_ = 0.5 #decaying factor
    tot_epoch = 300000
    print_epoch = 100

    for epoch in range(tot_epoch):
        #Reset and return the first observation
        observation = env.reset(exploring_starts=True)        
        for step in range(1000):
            #Take the action from the action matrix
            action = policy_matrix[observation[0], observation[1]]
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            #Estimate the error delta (Target - OldEstimate)
            delta = reward + gamma * utility_matrix[new_observation[0], new_observation[1]] - \
                                     utility_matrix[observation[0], observation[1]]
            #Adding +1 in the trace matrix for the state visited
            trace_matrix[observation[0], observation[1]] += 1
            #Update the utility matrix
            utility_matrix = update_utility(utility_matrix, trace_matrix, alpha, delta)
            #Update the trace matrix (decaying)
            trace_matrix = update_eligibility(trace_matrix, gamma, lambda_)
            observation = new_observation
            if done: break #return

        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(utility_matrix)
            print(trace_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility_matrix)



if __name__ == "__main__":
    main()
