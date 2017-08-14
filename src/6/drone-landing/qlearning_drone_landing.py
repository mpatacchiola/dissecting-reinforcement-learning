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

#In this example I will use the class DroneLanding to generate a 3D world
#in which the drone will move. Using the Q-learning algorithm I
#will estimate the state-action matrix.

import numpy as np
from drone_landing import DroneLanding
import matplotlib.pyplot as plt

def update_state_action(state_action_matrix, visit_counter_matrix, observation, new_observation, 
                        action, reward, alpha, gamma):
    """Return the updated state-action matrix

    @param state_action_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param action the action at t
    @param new_action the action at t+1
    @param reward the reward observed after the action
    @param alpha the ste size (learning rate)
    @param gamma the discount factor
    @return the updated state action matrix
    """
    #Getting the values of Q at t and estimating q
    x = observation[0]
    y = observation[1]
    z = observation[2]
    q = state_action_matrix[x,y,z,action] # Estimating q 
    # Estimating the q_t1 using observation at t+1
    x_t1 = new_observation[0]
    y_t1 = new_observation[1]
    z_t1 = new_observation[2]
    q_t1 = np.amax(state_action_matrix[x_t1,y_t1,z_t1,:])
    #Calculate alpha based on how many time it has been visited
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[x,y,z,action])
    #Applying the update rule
    #Here you can change "alpha" with "alpha_counted" if you want
    #to take into account how many times that particular state-action
    #pair has been visited until now.
    state_action_matrix[x,y,z,action] = state_action_matrix[x,y,z,action] + alpha * (reward + gamma * q_t1 - q)
    return state_action_matrix

def update_visit_counter(visit_counter_matrix, observation, action):
    """Update the visit counter
   
    Counting how many times a state-action pair has been 
    visited. This information can be used during the update.
    @param visit_counter_matrix a matrix initialised with zeros
    @param observation the state observed
    @param action the action taken
    """
    x = observation[0]
    y = observation[1]
    z = observation[2]
    visit_counter_matrix[x,y,z,action] += 1.0
    return visit_counter_matrix

def update_policy(policy_matrix, state_action_matrix, observation):
    """Return the updated policy matrix (q-learning)

    @param policy_matrix the matrix before the update
    @param state_action_matrix the state-action matrix
    @param observation the state obsrved at t
    @return the updated state action matrix
    """
    x = observation[0]
    y = observation[1]
    z = observation[2]
    #Getting the index of the action with the highest utility
    best_action = np.argmax(state_action_matrix[x,y,z,:])
    #Updating the policy
    policy_matrix[x,y,z] = best_action
    return policy_matrix

def return_epsilon_greedy_action(policy_matrix, observation, epsilon=0.1):
    x = observation[0]
    y = observation[1]
    z = observation[2]
    # Get the total number of actions
    tot_actions = int(np.nanmax(policy_matrix) + 1)
    # Return a random action or the one with highest utility
    if np.random.uniform(0, 1) <= epsilon:
        action = np.random.randint(low=0, high=tot_actions)
    else:
        action = int(policy_matrix[x,y,z])
    return action


def return_decayed_value(starting_value, global_step, decay_step):
        """Returns the decayed value.

        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
        @param starting_value the value before decaying
        @param global_step the global step to use for decay (positive integer)
        @param decay_step the step at which the value is decayed
        """
        decayed_value = starting_value * np.power(0.1, (global_step/decay_step))
        return decayed_value

def plot_curve(data_list, filepath="./my_plot.png", 
               x_label="X", y_label="Y", 
               x_range=(0, 1), y_range=(0,1), color="-r", kernel_size=50, alpha=0.4, grid=True):
        """Plot a graph using matplotlib

        """
        if(len(data_list) <=1):
            print("[WARNING] the data list is empty, no plot will be saved.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
        ax.grid(grid)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(data_list, color, alpha=alpha)  # The original data is showed in background
        kernel = np.ones(int(kernel_size))/float(kernel_size)  # Smooth the graph using a convolution
        tot_data = len(data_list)
        lower_boundary = int(kernel_size/2.0)
        upper_boundary = int(tot_data-(kernel_size/2.0))
        data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
        #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
        #print("Convolved: " + str(np.arange(tot_data).shape))
        ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary], data_convolved_array, color, alpha=1.0)  # Convolved plot
        fig.savefig(filepath)
        fig.clear()
        plt.close(fig)
        # print(plt.get_fignums())  # print the number of figures opened in background

def main():

    world_size = 21
    env = DroneLanding(world_size)
    tot_actions = 6

    #Define the state matrix
    state_matrix = np.zeros((world_size,world_size,world_size,tot_actions))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Random policy
    policy_matrix = np.random.randint(low=0, high=6, size=(world_size,world_size,world_size)).astype(np.float32)

    # Q-table and visit counter
    state_action_matrix = np.zeros((world_size,world_size,world_size,tot_actions))
    visit_counter_matrix = np.zeros((world_size,world_size,world_size,tot_actions))

    # Hyperparameters
    gamma = 0.999
    alpha = 0.001 #constant step size
    tot_episode = 2500000
    print_episode = 1000
    render_episode = 100000
    save_log_episode = 10

    reward_list = list()

    for episode in range(tot_episode+1):
        #Reset and return the first observation
        observation = env.reset(exploring_starts=True)
        #epsilon = return_decayed_value(0.1, episode, decay_step=50000)
        epsilon = 0.1
        cumulated_reward = 0
        for step in range(50):
            #Take the action from the action matrix
            #action = policy_matrix[observation[0], observation[1]]
            #Take the action using epsilon-greedy
            action = return_epsilon_greedy_action(policy_matrix, observation, epsilon=epsilon)
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            #Updating the state-action matrix
            state_action_matrix = update_state_action(state_action_matrix, visit_counter_matrix, observation, new_observation, 
                                                      action, reward, alpha, gamma)
            #Updating the policy
            policy_matrix = update_policy(policy_matrix, state_action_matrix, observation)
            #Increment the visit counter
            visit_counter_matrix = update_visit_counter(visit_counter_matrix, observation, action)
            observation = new_observation
            cumulated_reward += reward
            if done: break

        if(episode % save_log_episode == 0):
            reward_list.append(cumulated_reward)
        if(episode % print_episode == 0):
            print("")
            print("Episode: " + str(episode) + " of " + str(tot_episode))
            print("Epsilon: " + str(epsilon))
            print("Cumulated reward: " + str(cumulated_reward))
            print("Q-max: " + str(np.amax(state_action_matrix)))
            print("Q-mean: " + str(np.mean(state_action_matrix)))
            print("Q-min: " + str(np.amin(state_action_matrix)))
        if episode % render_episode == 0:
            print("Saving the gif in ./drone_landing.gif")
            env.render(file_path='./drone_landing.gif', mode='gif')
            print("Done!")
            print("Saving the reward graph in ./reward.png")
            plot_curve(reward_list, filepath="./reward.png", 
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-1.55,1.05),
                       color="red", kernel_size=500, 
                       alpha=0.4, grid=True)
            print("Done!")
    #Training complete
    print("Finished!!!")



if __name__ == "__main__":
    main()
